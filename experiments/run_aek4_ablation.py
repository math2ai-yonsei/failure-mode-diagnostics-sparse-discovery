"""
AEK-4 Gate4 D-optimal Ablation Runner

Purpose: D-optimal selection의 인과적 기여 검증 (Confound-free 설계)
System: AEK Self-balancing Motorcycle (Reaction Wheel Inverted Pendulum)

설계 원칙:
- 동일 Pool, 다른 Selection → 순수 Selection 효과 측정
- Pool: seed=1, pool_seed=42, pool_size=2000 (고정)
- D-optimal: 1 run
- Random: 3 runs (selection_seed=0,1,2)

AEK vs Cart-Pole 핵심 차이:
- GMM: 5D (4 IC + 1 I_w_C) vs Cart-Pole 6D (4 IC + 2 params)
- Dynamics: AEKSimulator (solve_ivp) vs inline cartpole_dynamics
- dx: Analytic (dynamics 직접 호출) vs Savitzky-Golay
- Library: AEK 14 terms vs Cart-Pole 21 terms (gate0_min)
- Normalization: Raw inputs → ColumnScaler vs normalized inputs
- Metric: Spurious-primary (sign flip) vs dynamics-primary
- Fragile: Pure precision collapse (20-21 spurious, 0 dynamics)

산출물 (per run):
- manifest.json, metrics.json
- z_after.npy, fragile_z_after.npy, fragile_aug_pure.npy
- fragile_z_before.npy, effective_pairs.json, selected_indices.npy
- ablation_summary.json (overall)

Author: Claude (AEK-4 Gate4 Ablation)
Date: 2026-02-09
Version: v1.3 (+ excitation tail injection for Step 2 collinearity test)
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import hashlib
import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings
import traceback

import numpy as np
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture

# 프로젝트 모듈 import
from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble
from src.sindy.aek_library import (
    build_aek_library,
    get_aek_oracle_support,
    AEK_FEATURE_NAMES,
    AEK_TARGET_NAMES,
    N_AEK_FEATURES,
)
from src.simulators.aek_simulator import AEKSimulator

# Gate3에서 시스템-무관 유틸리티만 import
from experiments.run_gate3_v2 import (
    create_rng_streams,
    generate_run_id,
    compute_fragile_feature_sets,
    compute_gram_contributions_by_target,
    greedy_dopt_selection,
)


# ============================================================
# Constants
# ============================================================

AEK_DYNAMICS_TARGET_INDICES = [1, 3]  # phi_ddot, theta_w_ddot

# AEK QC bounds (physics-based, relaxed for augmentation)
AEK_QC_BOUNDS = {
    'max_phi': 0.15,          # rad (~8.6°) - training max is 0.043, allow 3x
    'max_phi_dot': 5.0,       # rad/s - training max is 0.46
    'max_theta_w_dot': 150.0, # rad/s - training max is 96.8
}

# AEK GMM physical constraints
# CRITICAL: AEK is highly unstable (time_to_fall=0.1s).
# Training I_w_C: [6.95e-5, 8.69e-5]. GMM bounds must stay near this range.
# OOD test is 1.04e-4 (1.2x nominal), so we allow modest extrapolation.
AEK_GMM_BOUNDS = {
    'I_w_C_min': 5.0e-05,     # 0.58x nominal (covers stress train 5e-5)
    'I_w_C_max': 1.2e-04,     # 1.38x nominal (slightly above OOD test 1.04e-4)
    'phi_range': (-0.05, 0.05),    # rad, training IC: ±0.03
    'phi_dot_range': (-0.8, 0.8),  # rad/s, training IC: ±0.46
    'theta_w_range': (-1.2, 1.2),  # rad, training IC: ±0.90
    'theta_w_dot_range': (-12.0, 12.0),  # rad/s, training IC: ±9.85
}


# ============================================================
# Configuration
# ============================================================

@dataclass
class AEK4AblationConfig:
    """AEK-4 D-optimal Ablation 실험 설정"""
    # Fixed Pool Settings (Confound-free)
    seed: int = 1
    pool_seed: int = 42
    pool_size: int = 2000

    # Selection Settings
    n_select: int = 200
    reject_ratio: float = 0.10  # Track A reject ratio

    # D-optimal Settings
    dopt_lambda: float = 1e-6
    dopt_gram_energy_mode: str = 'unit_trace'
    dopt_selection_variant: str = 'greedy'
    dopt_use_teacher_intersection: bool = True
    dopt_top_m_ratio: float = 2.0
    dopt_trace_power: float = 1.0

    # D-optimal mode control
    skip_dopt: bool = False               # Skip D-opt entirely (random-only)
    dopt_target: str = 'fragile'          # 'fragile' | 'oracle'

    # Random Selection Seeds
    random_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Paths
    dataset_version: str = 'aek_ood_v1'
    system: str = 'aek'
    baseline_source: str = ''   # AEK-3 baseline directory
    dataset_path: str = ''      # Override dataset path (optional)

    # Output
    results_base: str = 'results/aek_ood_v1/gate4/ablation/d_optimal_vs_random'
    note: str = 'aek4_ablation'

    # E-SINDy Settings
    bootstrap_B: int = 100
    threshold: float = 0.05
    n_train: int = 10

    # GMM Settings
    gmm_n_components: int = 3
    gmm_covariance_type: str = 'full'

    # Pool Generation
    max_pool_attempts: int = 15000
    pool_u_variant: str = 'IC'  # Reuse training u

    # Excitation Tail Injection (Step 2: break cos(phi)≈1 collinearity)
    excitation_tail: bool = False       # Enable phi0 tail injection
    excitation_phi_range: float = 0.12  # ±range for tail phi0
    excitation_fraction: float = 0.30   # Fraction of GMM samples to replace
    excitation_max_phi_qc: float = 0.20 # Relaxed QC max_phi for excitation

    # Simulation (from aek.yaml)
    sim_dt: float = 0.01
    sim_T_steps: int = 201

    # SSOT Metrics Settings
    ci_bootstrap_B: int = 2000
    ci_alpha: float = 0.05
    gate2_ceiling: float = 0.058  # Gate2 ceiling

    # Z-metric
    z_eps: float = 1e-6
    z0: float = 2.0  # z-threshold for "confident non-zero"
    tau_support: float = 0.5  # inclusion prob threshold


# ============================================================
# AEK GMM Proposal Sampler
# ============================================================

class AEKGMMSampler:
    """
    GMM-based proposal sampler for AEK system.

    Fits 5D GMM: [phi0, phi_dot0, theta_w0, theta_w_dot0, I_w_C]
    (4 IC + 1 OOD parameter)
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None
        self._is_fitted = False

    def fit(self, train_x: np.ndarray, train_params: np.ndarray) -> 'AEKGMMSampler':
        """
        Fit GMM on training IC + params.

        Args:
            train_x: (N, T, 4) training trajectories
            train_params: (N, 1) I_w_C values per trajectory
        """
        N = train_x.shape[0]

        # Extract IC: first time step
        ic = train_x[:, 0, :]  # (N, 4) [phi0, phi_dot0, theta_w0, theta_w_dot0]
        params = train_params.reshape(N, -1)  # (N, 1) [I_w_C]

        # Stack to 5D
        data = np.hstack([ic, params])  # (N, 5)
        print(f"  GMM data: {data.shape}, components={self.n_components}")

        # Auto-adjust components if too few samples
        max_components = max(1, N // 3)
        actual_components = min(self.n_components, max_components)
        if actual_components < self.n_components:
            print(f"  ⚠️ Reduced components: {self.n_components} → {actual_components} (N={N})")
            self.n_components = actual_components

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
            n_init=3,
        )
        self.gmm.fit(data)
        self._is_fitted = True

        print(f"  GMM converged: {self.gmm.converged_}, "
              f"n_iter: {self.gmm.n_iter_}")
        return self

    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample IC + params from fitted GMM with physical constraints.

        Returns:
            ic: (n_samples, 4) initial conditions
            params: (n_samples, 1) I_w_C values
        """
        if not self._is_fitted:
            raise ValueError("GMM not fitted. Call fit() first.")

        bounds = AEK_GMM_BOUNDS
        n_valid = 0
        ic_list = []
        params_list = []
        max_attempts = n_samples * 10

        for attempt in range(max_attempts):
            if n_valid >= n_samples:
                break

            # Sample batch
            batch_size = min(n_samples - n_valid, 500)
            samples, _ = self.gmm.sample(batch_size)

            for s in samples:
                phi0, phi_dot0, theta_w0, theta_w_dot0, I_w_C = s

                # Physical constraints
                if not (bounds['I_w_C_min'] <= I_w_C <= bounds['I_w_C_max']):
                    continue
                if not (bounds['phi_range'][0] <= phi0 <= bounds['phi_range'][1]):
                    continue
                if not (bounds['phi_dot_range'][0] <= phi_dot0 <= bounds['phi_dot_range'][1]):
                    continue
                if not (bounds['theta_w_range'][0] <= theta_w0 <= bounds['theta_w_range'][1]):
                    continue
                if not (bounds['theta_w_dot_range'][0] <= theta_w_dot0 <= bounds['theta_w_dot_range'][1]):
                    continue

                ic_list.append([phi0, phi_dot0, theta_w0, theta_w_dot0])
                params_list.append([I_w_C])
                n_valid += 1

                if n_valid >= n_samples:
                    break

        if n_valid < n_samples:
            print(f"  ⚠️ Only generated {n_valid}/{n_samples} valid samples "
                  f"in {max_attempts} attempts")

        ic = np.array(ic_list[:n_samples])
        params = np.array(params_list[:n_samples])
        return ic, params

    def get_params_dict(self) -> Dict[str, Any]:
        """Return GMM parameters for hashing/verification."""
        if not self._is_fitted:
            return {}
        return {
            'weights': self.gmm.weights_.tolist(),
            'means': self.gmm.means_.tolist(),
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
        }


# ============================================================
# AEK Pool Generator
# ============================================================

class AEKPoolGenerator:
    """
    Generate augmentation pool for AEK system.

    Pipeline:
    1. Sample IC + I_w_C from GMM
    2. Simulate with PD stabilizing controller (NOT open-loop u replay)
    3. Compute SavGol dx from clean teacher trajectory (§3.3)
    4. QC filter (bounds check)

    CRITICAL DESIGN NOTE:
    Unlike Cart-Pole (semi-stable, tolerates open-loop u replay),
    AEK is an unstable inverted pendulum (time_to_fall=0.1s).
    Open-loop replay of training u on different ICs causes instant divergence.
    Therefore we simulate with a PD controller that:
    - Stabilizes the system (keeps motorcycle balanced)
    - Produces realistic u within tau_max constraints
    - Generates physically plausible (IC, u, trajectory, dx) tuples
    """

    # PD controller gains (derived from linearized dynamics)
    # Linearized: phi_ddot = (M*g*h/I_p)*phi - tau/I_p
    # M*g*h ≈ 0.277 N*m/rad (nominal) → stability requires K_phi > 0.277
    # tau_max = 0.02 N*m → max stabilizable phi ≈ tau_max/K_phi
    # With K_phi=0.5: stable phi ≈ ±0.04 rad (matches training range)
    PD_K_PHI = 0.50     # N*m/rad - 80% margin above M*g*h
    PD_K_D = 0.04       # N*m*s/rad - damping (higher for faster settling)
    TAU_MAX = 0.02       # N*m - motor saturation

    def __init__(
        self,
        gmm_sampler: AEKGMMSampler,
        train_u: np.ndarray,
        dt: float,
        T_steps: int,
        rng: np.random.Generator,
        excitation_tail: bool = False,
        excitation_phi_range: float = 0.12,
        excitation_fraction: float = 0.30,
        excitation_max_phi_qc: float = 0.20,
    ):
        self.gmm = gmm_sampler
        self.train_u = train_u  # (N_train, T, 1) kept for stats only
        self.dt = dt
        self.T_steps = T_steps
        self.rng = rng
        self.excitation_tail = excitation_tail
        self.excitation_phi_range = excitation_phi_range
        self.excitation_fraction = excitation_fraction
        self.excitation_max_phi_qc = excitation_max_phi_qc

    def _pd_controller(self, t: float, state: np.ndarray) -> float:
        """
        PD stabilizing controller: tau = +K_phi*phi + K_d*phi_dot, clipped.

        Sign convention (from EOM):
          phi_ddot = (M*g*h*sin(phi) - tau) / I_p
        tau enters with MINUS sign, so to counteract gravity (M*g*h*sin(phi) > 0
        when phi > 0), we need tau > 0 when phi > 0.
        Therefore: tau = +K*phi (NOT -K*phi).
        """
        phi = state[0]
        phi_dot = state[1]
        tau = self.PD_K_PHI * phi + self.PD_K_D * phi_dot
        return np.clip(tau, -self.TAU_MAX, self.TAU_MAX)

    def generate_pool(
        self,
        target_n_accept: int,
        max_attempts: int = 15000,
    ) -> Dict[str, Any]:
        """Generate pool of accepted trajectories with retry loop."""
        trajectories = []
        dx_list = []
        params_list = []
        ic_list = []
        u_list = []

        n_attempted = 0
        n_sim_fail = 0
        n_qc_fail = 0
        batch_round = 0
        max_rounds = 10  # Safety limit on retry rounds

        while len(trajectories) < target_n_accept and batch_round < max_rounds:
            batch_round += 1
            remaining = target_n_accept - len(trajectories)

            # Sample batch (5x remaining to account for rejection)
            n_to_sample = min(max_attempts, remaining * 5)
            sampled_ic, sampled_params = self.gmm.sample(n_to_sample, self.rng)
            n_available = len(sampled_ic)

            # --- Excitation Tail Injection (Step 2) ---
            if self.excitation_tail and n_available > 0:
                n_tail = int(self.excitation_fraction * n_available)
                tail_idx = self.rng.choice(n_available, size=n_tail, replace=False)
                phi_range = self.excitation_phi_range
                sampled_ic[tail_idx, 0] = self.rng.uniform(
                    -phi_range, phi_range, size=n_tail
                )
                if batch_round == 1:
                    print(f"  [EXCITATION] Injected {n_tail}/{n_available} tail samples "
                          f"(phi0 ±{phi_range:.2f})")
                    print(f"    Tail phi0 range: [{sampled_ic[tail_idx, 0].min():.4f}, "
                          f"{sampled_ic[tail_idx, 0].max():.4f}]")

            if batch_round == 1:
                print(f"  Sampled {n_available} IC+params from GMM")
                if n_available > 0:
                    print(f"    Sampled I_w_C range: [{sampled_params[:, 0].min():.4e}, "
                          f"{sampled_params[:, 0].max():.4e}]")
                    print(f"    Sampled phi0 range: [{sampled_ic[:, 0].min():.4f}, "
                          f"{sampled_ic[:, 0].max():.4f}]")
            elif batch_round > 1:
                print(f"  Retry round {batch_round}: sampled {n_available}, "
                      f"need {remaining} more")

            for i in range(n_available):
                if len(trajectories) >= target_n_accept:
                    break

                n_attempted += 1
                ic = sampled_ic[i]
                I_w_C = sampled_params[i, 0]

                # Create simulator with this I_w_C
                sim = AEKSimulator(params={'I_w_C': I_w_C})

                # Simulate with PD controller
                try:
                    t_out, traj, u_out = sim.simulate(
                        x0=ic,
                        t_span=(0, (self.T_steps - 1) * self.dt),
                        dt=self.dt,
                        controller=self._pd_controller,
                        method='RK45',
                        rtol=1e-8,
                        atol=1e-10,
                    )

                    # Ensure correct length
                    if traj.shape[0] < self.T_steps:
                        n_sim_fail += 1
                        continue

                    traj = traj[:self.T_steps]
                    u_out = u_out[:self.T_steps]

                except (RuntimeError, Exception):
                    n_sim_fail += 1
                    continue

                # SavGol dx from clean teacher trajectory (§3.3 consistency)
                dx = np.zeros_like(traj)
                for s in range(4):  # state_dim=4
                    dx[:, s] = savgol_filter(
                        traj[:, s],
                        window_length=7,
                        polyorder=3,
                        deriv=1,
                        delta=self.dt,
                    )

                # QC check
                if not self._qc_check(traj):
                    n_qc_fail += 1
                    if n_qc_fail <= 5:
                        max_phi = np.max(np.abs(traj[:, 0]))
                        max_phi_dot = np.max(np.abs(traj[:, 1]))
                        max_tw_dot = np.max(np.abs(traj[:, 3]))
                        phi_exceed = np.where(np.abs(traj[:, 0]) > AEK_QC_BOUNDS['max_phi'])[0]
                        t_exceed = phi_exceed[0] * self.dt if len(phi_exceed) > 0 else -1
                        print(f"    QC FAIL #{n_qc_fail}: max_phi={max_phi:.3f}, "
                              f"max_phi_dot={max_phi_dot:.1f}, max_tw_dot={max_tw_dot:.1f}, "
                              f"phi0={ic[0]:.4f}, I_w_C={I_w_C:.2e}, "
                              f"t_exceed={t_exceed:.3f}s")
                    continue

                # Check for NaN/Inf
                if not (np.isfinite(traj).all() and np.isfinite(dx).all()):
                    n_qc_fail += 1
                    continue

                trajectories.append(traj)
                dx_list.append(dx)
                params_list.append(sampled_params[i])
                ic_list.append(ic)
                u_list.append(u_out)

                if len(trajectories) % 200 == 0:
                    print(f"    Accepted: {len(trajectories)}/{target_n_accept} "
                          f"(attempted: {n_attempted}, sim_fail: {n_sim_fail}, "
                          f"qc_fail: {n_qc_fail})")

        n_accepted = len(trajectories)
        print(f"  Pool generation complete:")
        print(f"    Rounds: {batch_round}")
        print(f"    Attempted: {n_attempted}")
        print(f"    Sim failures: {n_sim_fail}")
        print(f"    QC failures: {n_qc_fail}")
        print(f"    Accepted: {n_accepted}")
        print(f"    Acceptance rate: {n_accepted / max(n_attempted, 1):.1%}")

        if n_accepted == 0:
            raise RuntimeError("Pool generation produced 0 accepted trajectories!")

        if n_accepted < target_n_accept:
            raise RuntimeError(
                f"Pool generation shortfall: {n_accepted}/{target_n_accept} "
                f"after {max_rounds} rounds. Adjust GMM bounds or QC thresholds."
            )

        return {
            'trajectories': np.array(trajectories),   # (N, T, 4)
            'dx': np.array(dx_list),                   # (N, T, 4)
            'params': np.array(params_list),           # (N, 1)
            'ic': np.array(ic_list),                   # (N, 4)
            'u': np.array(u_list),                     # (N, T, 1)
            'stats': {
                'n_attempted': n_attempted,
                'n_sim_fail': n_sim_fail,
                'n_qc_fail': n_qc_fail,
                'n_accepted': n_accepted,
                'acceptance_rate': n_accepted / max(n_attempted, 1),
                'controller': 'PD',
                'K_phi': self.PD_K_PHI,
                'K_d': self.PD_K_D,
                'tau_max': self.TAU_MAX,
            },
        }

    def _qc_check(self, traj: np.ndarray) -> bool:
        """QC check: reject diverged trajectories."""
        phi = traj[:, 0]
        phi_dot = traj[:, 1]
        theta_w_dot = traj[:, 3]

        bounds = AEK_QC_BOUNDS
        max_phi = self.excitation_max_phi_qc if self.excitation_tail else bounds['max_phi']
        if np.any(np.abs(phi) > max_phi):
            return False
        if np.any(np.abs(phi_dot) > bounds['max_phi_dot']):
            return False
        if np.any(np.abs(theta_w_dot) > bounds['max_theta_w_dot']):
            return False
        return True


# ============================================================
# AEK Track A: Teacher Alignment Filtering
# ============================================================

def aek_track_a_selection(
    pool: Dict[str, np.ndarray],
    teacher_coefficients: np.ndarray,
    reject_ratio: float = 0.10,
    n_select: int = 200,
    min_after_a: int = 50,
) -> Dict[str, Any]:
    """
    Track A: Filter pool by teacher alignment error.

    AEK-specific: Uses raw (unnormalized) features.
    Teacher coefficients are in unscaled (physical) units.
    Error computed on dynamics targets only (indices [1, 3]).

    Args:
        pool: Generated pool dict
        teacher_coefficients: (14, 4) teacher coefficient matrix
        reject_ratio: Fraction of worst trajectories to reject
        n_select: Target number for downstream selection
        min_after_a: Minimum pool size after filtering

    Returns:
        Dict with selected_indices, errors, threshold, stats
    """
    traj = pool['trajectories']  # (N, T, 4)
    dx = pool['dx']              # (N, T, 4)
    u = pool['u']                # (N, T, 1)
    N, T, _ = traj.shape

    # Compute per-trajectory alignment error
    errors = np.zeros(N)

    for i in range(N):
        # Build library features for this trajectory
        x_i = traj[i]  # (T, 4)
        u_i = u[i]     # (T, 1)
        Theta_i, _ = build_aek_library(x_i, u_i)  # (T, 14)

        # Predict dx using teacher coefficients
        dx_pred = Theta_i @ teacher_coefficients  # (T, 4)

        # Error on dynamics targets only [phi_ddot, theta_w_ddot]
        dx_actual = dx[i]  # (T, 4)
        residual = dx_pred[:, AEK_DYNAMICS_TARGET_INDICES] - dx_actual[:, AEK_DYNAMICS_TARGET_INDICES]

        # Per-trajectory RMSE
        errors[i] = np.sqrt(np.mean(residual ** 2))

    # Reject top reject_ratio% by error
    n_reject = max(1, int(N * reject_ratio))
    error_threshold = np.sort(errors)[-n_reject] if n_reject < N else np.inf

    # Selected: below threshold (lower error = better aligned)
    selected_mask = errors <= error_threshold
    selected_indices = np.where(selected_mask)[0]

    # Relaxation: if too few pass, accept all
    relaxed = False
    if len(selected_indices) < min_after_a:
        print(f"  ⚠️ Track A relaxation: {len(selected_indices)} < {min_after_a}, accepting all")
        selected_indices = np.arange(N)
        relaxed = True

    stats = {
        'n_pool': N,
        'n_rejected': int((~selected_mask).sum()) if not relaxed else 0,
        'n_passed': len(selected_indices),
        'reject_ratio': reject_ratio,
        'error_threshold': float(error_threshold),
        'error_mean': float(errors.mean()),
        'error_std': float(errors.std()),
        'error_min': float(errors.min()),
        'error_max': float(errors.max()),
        'relaxed': relaxed,
    }

    print(f"  Track A: {len(selected_indices)}/{N} passed "
          f"(error threshold: {error_threshold:.6f})")
    print(f"  Error range: [{errors.min():.6f}, {errors.max():.6f}], "
          f"mean: {errors.mean():.6f}")

    return {
        'selected_indices': selected_indices,
        'errors': errors,
        'threshold': error_threshold,
        'stats': stats,
    }


# ============================================================
# AEK D-optimal Selection (Track B)
# ============================================================

def aek_track_b_dopt_selection(
    pool: Dict[str, np.ndarray],
    track_a_result: Dict[str, Any],
    fragile_pairs: List[Dict],
    teacher_support: np.ndarray,
    train_x: np.ndarray,
    train_u: np.ndarray,
    n_select: int = 200,
    top_m_ratio: float = 2.0,
    lambda_reg: float = 1e-6,
    use_teacher_intersection: bool = True,
    gram_energy_mode: str = 'unit_trace',
    trace_power: float = 1.0,
    dopt_target: str = 'fragile',  # 'fragile' or 'oracle'
    rng: Optional[np.random.Generator] = None,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    D-optimal selection on AEK features.

    dopt_target modes:
    - 'fragile': Maximize information for fragile feature set (original behavior).
                 WARNING: On spurious-primary systems like AEK, this can reinforce
                 spurious terms (observed in fail-fast experiment).
    - 'oracle':  Maximize information for oracle-active features only.
                 Hypothesis: Strengthening true features may indirectly suppress
                 spurious terms by improving relative signal quality.
    """
    if dopt_target == 'oracle':
        # Oracle feature set: only features present in true dynamics
        oracle_support = get_aek_oracle_support()  # (14, 4) bool
        F_by_target = {}
        for t in AEK_DYNAMICS_TARGET_INDICES:
            oracle_features = list(np.where(oracle_support[:, t])[0])
            if oracle_features:
                F_by_target[t] = oracle_features
        print(f"  D-opt target: ORACLE features")
    else:
        # Original: fragile feature set
        fragile_tuples = [(p['feature_idx'], p['target_idx']) for p in fragile_pairs]
        F_by_target = compute_fragile_feature_sets(
            fragile_pairs=fragile_tuples,
            teacher_support=teacher_support,
            dynamics_target_indices=AEK_DYNAMICS_TARGET_INDICES,
            use_teacher_intersection=use_teacher_intersection,
        )
        print(f"  D-opt target: FRAGILE features")

    print(f"  Fragile feature sets per target:")
    for t, F_t in F_by_target.items():
        target_name = AEK_TARGET_NAMES[t] if t < len(AEK_TARGET_NAMES) else f"target_{t}"
        feature_names = [AEK_FEATURE_NAMES[f] for f in F_t] if len(F_t) > 0 else ['(none)']
        print(f"    Target {t} ({target_name}): {len(F_t)} features → {feature_names}")

    # Check: any fragile features for D-optimal?
    total_fragile_features = sum(len(F_t) for F_t in F_by_target.values())
    if total_fragile_features == 0:
        print("  ⚠️ No fragile features in dynamics targets for D-optimal!")
        print("  Falling back to random selection from Track A candidates")
        # Return all Track A candidates (effectively random)
        candidate_indices = track_a_result['selected_indices']
        if len(candidate_indices) > n_select:
            rng_fb = rng if rng is not None else np.random.default_rng(0)
            chosen = rng_fb.choice(len(candidate_indices), size=n_select, replace=False)
            candidate_indices = np.sort(candidate_indices[chosen])
        return _build_selection_result(pool, candidate_indices, 'd_optimal_fallback_random')

    # Pre-gate: take top_M candidates by alignment error (lowest error = best)
    candidate_indices = track_a_result['selected_indices']
    errors = track_a_result['errors']
    top_M = min(int(n_select * top_m_ratio), len(candidate_indices))

    # Sort by error (ascending = best first)
    candidate_errors = errors[candidate_indices]
    sorted_order = np.argsort(candidate_errors)
    pregate_indices = candidate_indices[sorted_order[:top_M]]

    print(f"  Pre-gate: {len(pregate_indices)} candidates (top {top_M} by alignment error)")

    # Compute library features for pre-gate candidates (ColumnScaler-scaled)
    # First: fit ColumnScaler on training data
    train_N = train_x.shape[0]
    train_T = train_x.shape[1]
    x_flat_train = train_x.reshape(-1, 4)
    u_flat_train = train_u.reshape(-1, 1)
    Theta_train, _ = build_aek_library(x_flat_train, u_flat_train)

    scaler = ColumnScaler()
    scaler.fit(Theta_train)

    # Compute Theta for each pre-gate candidate trajectory
    pool_traj = pool['trajectories']
    pool_u = pool['u']
    T_pool = pool_traj.shape[1]

    N_cand = len(pregate_indices)
    Theta_cand = np.zeros((N_cand, T_pool, N_AEK_FEATURES))

    for local_i, pool_i in enumerate(pregate_indices):
        x_i = pool_traj[pool_i]  # (T, 4)
        u_i = pool_u[pool_i]     # (T, 1)
        Theta_i_raw, _ = build_aek_library(x_i, u_i)  # (T, 14)
        Theta_i_scaled = scaler.transform(Theta_i_raw)  # (T, 14)
        Theta_cand[local_i] = Theta_i_scaled

    # Compute Gram contributions per target
    G_by_target, trace_by_target = compute_gram_contributions_by_target(
        Theta=Theta_cand,
        F_by_target=F_by_target,
        gram_energy_mode=gram_energy_mode,
        trace_power=trace_power,
    )

    # Greedy D-optimal selection
    selected_pool_indices, selection_trace = greedy_dopt_selection(
        G_by_target=G_by_target,
        candidate_pool_indices=pregate_indices,
        n_select=n_select,
        lambda_reg=lambda_reg,
    )

    # Save trace if results_dir provided
    if results_dir is not None:
        trace_path = results_dir / 'dopt_selection_trace.json'
        with open(trace_path, 'w') as f:
            json.dump(selection_trace, f, indent=2, default=str)

    return _build_selection_result(pool, selected_pool_indices, 'd_optimal')


def _build_selection_result(
    pool: Dict[str, np.ndarray],
    selected_indices: np.ndarray,
    method: str,
) -> Dict[str, Any]:
    """Build selection result dict from pool indices."""
    selected_indices = np.sort(selected_indices)
    return {
        'trajectories': pool['trajectories'][selected_indices],
        'dx': pool['dx'][selected_indices],
        'params': pool['params'][selected_indices],
        'ic': pool['ic'][selected_indices],
        'u': pool['u'][selected_indices],
        'original_indices': selected_indices.copy(),
        'stats': {
            'n_pool': len(pool['trajectories']),
            'n_selected': len(selected_indices),
            'selection_mode': method,
        },
    }


# ============================================================
# Random Selection
# ============================================================

def aek_random_selection(
    pool: Dict[str, np.ndarray],
    track_a_result: Dict[str, Any],
    n_select: int,
    selection_seed: int,
) -> Dict[str, Any]:
    """Random selection from Track A passed candidates."""
    candidate_indices = track_a_result['selected_indices']
    n_available = len(candidate_indices)

    rng = np.random.default_rng(selection_seed)

    if n_available <= n_select:
        print(f"  ⚠️ Only {n_available} available, selecting all")
        final_indices = candidate_indices.copy()
    else:
        chosen = rng.choice(n_available, size=n_select, replace=False)
        final_indices = candidate_indices[chosen]

    final_indices = np.sort(final_indices)

    result = _build_selection_result(pool, final_indices, 'random')
    result['stats']['selection_seed'] = selection_seed
    result['stats']['n_available'] = n_available

    print(f"  Random selection (seed={selection_seed}): "
          f"{len(final_indices)}/{n_available} selected")
    return result


# ============================================================
# AEK E-SINDy Evaluation
# ============================================================

def aek_evaluate_with_esindy(
    train_x: np.ndarray,
    train_dx: np.ndarray,
    train_u: np.ndarray,
    aug_x: np.ndarray,
    aug_dx: np.ndarray,
    aug_u: np.ndarray,
    bootstrap_B: int = 100,
    threshold: float = 0.05,
    seed: int = 1,
    z_eps: float = 1e-6,
    tau_support: float = 0.5,
    z0: float = 2.0,
) -> Dict[str, Any]:
    """
    Evaluate augmented data with E-SINDy (AEK-specific).

    Uses build_aek_library (14 terms) + ColumnScaler + ESINDyEnsemble.
    Returns z-scores, support masks, coefficients.
    """
    # Combine original + augmented
    n_orig = train_x.shape[0]
    n_aug = aug_x.shape[0]
    T = train_x.shape[1]
    T_aug = aug_x.shape[1]

    # Flatten
    x_orig_flat = train_x.reshape(-1, 4)
    dx_orig_flat = train_dx.reshape(-1, 4)
    u_orig_flat = train_u.reshape(-1, 1)

    x_aug_flat = aug_x.reshape(-1, 4)
    dx_aug_flat = aug_dx.reshape(-1, 4)
    u_aug_flat = aug_u.reshape(-1, 1)

    # Build library features separately (may have different T)
    Theta_orig, _ = build_aek_library(x_orig_flat, u_orig_flat)
    Theta_aug, _ = build_aek_library(x_aug_flat, u_aug_flat)

    dx_all = np.vstack([dx_orig_flat, dx_aug_flat])

    # v1.2 HOTFIX P0-2: fit scaler on ORIG only (avoid selection-dependent confound)
    scaler = ColumnScaler()
    scaler.fit(Theta_orig)
    Theta_scaled = np.vstack([
        scaler.transform(Theta_orig),
        scaler.transform(Theta_aug),
    ])

    # Total trajectories and effective T
    # ESINDyEnsemble needs (n_traj * T) samples with trajectory-level bootstrap
    # Since orig and aug may have different T, we need to handle this carefully
    n_total_traj = n_orig + n_aug

    # For trajectory-level bootstrap, all trajectories must have same T
    # If T != T_aug, we need to truncate or pad
    if T != T_aug:
        T_common = min(T, T_aug)
        print(f"  ⚠️ T mismatch: orig={T}, aug={T_aug}, using T_common={T_common}")

        # Re-flatten with common T
        x_orig_trunc = train_x[:, :T_common, :]
        dx_orig_trunc = train_dx[:, :T_common, :]
        u_orig_trunc = train_u[:, :T_common, :]
        x_aug_trunc = aug_x[:, :T_common, :]
        dx_aug_trunc = aug_dx[:, :T_common, :]
        u_aug_trunc = aug_u[:, :T_common, :]

        x_flat = np.vstack([
            x_orig_trunc.reshape(-1, 4),
            x_aug_trunc.reshape(-1, 4),
        ])
        dx_flat = np.vstack([
            dx_orig_trunc.reshape(-1, 4),
            dx_aug_trunc.reshape(-1, 4),
        ])
        u_flat = np.vstack([
            u_orig_trunc.reshape(-1, 1),
            u_aug_trunc.reshape(-1, 1),
        ])
        Theta_rebuild, _ = build_aek_library(x_flat, u_flat)
        # v1.2 HOTFIX P0-2: fit scaler on ORIG-only portion (T mismatch case)
        Theta_orig_trunc_flat, _ = build_aek_library(
            x_orig_trunc.reshape(-1, 4), u_orig_trunc.reshape(-1, 1))
        scaler2 = ColumnScaler()
        scaler2.fit(Theta_orig_trunc_flat)
        Theta_scaled = scaler2.transform(Theta_rebuild)
        dx_all = dx_flat
        T_eff = T_common
        scaler = scaler2
    else:
        T_eff = T

    # Fit E-SINDy ensemble
    ensemble = ESINDyEnsemble(
        n_bootstrap=bootstrap_B,
        threshold=threshold,
        random_state=seed,
    )
    ensemble.fit(
        Theta=Theta_scaled,
        dx=dx_all,
        n_trajectories=n_total_traj,
        T=T_eff,
        scaler=scaler,
        target_scale=None,
    )

    # Get results
    result = ensemble.get_result()
    coef_mean = result.coefficients_mean    # (14, 4)
    coef_std = result.coefficients_std      # (14, 4)
    inclusion_prob = result.inclusion_probability  # (14, 4)

    # Z-metric
    z_scores = np.abs(coef_mean) / (coef_std + z_eps)

    # Support masks
    support_mask = inclusion_prob >= tau_support
    oracle_support = get_aek_oracle_support()

    stable_core_mask = support_mask == oracle_support
    fragile_pool_mask = ~stable_core_mask

    return {
        'coefficients_mean': coef_mean,
        'coefficients_std': coef_std,
        'inclusion_probability': inclusion_prob,
        'z_scores': z_scores,
        'support_mask': support_mask,
        'stable_core_mask': stable_core_mask,
        'fragile_pool_mask': fragile_pool_mask,
        'n_total': n_total_traj,
        'n_original': n_orig,
        'n_augmented': n_aug,
        'T_effective': T_eff,
    }


# ============================================================
# Ablation Run Result
# ============================================================

@dataclass
class AblationRunResult:
    """Single ablation run result."""
    run_id: str
    selection_method: str
    selection_seed: Optional[int]
    results_dir: Path
    status: str
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# ============================================================
# AEK-4 Ablation Runner
# ============================================================

class AEK4AblationRunner:
    """AEK-4 D-optimal vs Random Ablation Runner."""

    def __init__(self, config: AEK4AblationConfig):
        self.cfg = config
        self.results_base = Path(config.results_base)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.project_root = _PROJECT_ROOT

        # Shared artifacts
        self.dataset = None
        self.pool = None
        self.track_a_result = None
        self.teacher_support = None
        self.teacher_coefficients = None
        self.fragile_pairs = None     # List[Dict] with 'type' field
        self.fragile_z_before = None
        self.z_before = None
        self.gmm_sampler = None
        self.rng_streams = None

        # Hashes for SSOT audit
        self._pool_sha = ''
        self._gmm_fit_sha = ''
        self._z_before_sha = ''

    def run_all(self) -> Dict[str, Any]:
        """Run all ablation experiments."""
        print("=" * 70)
        print("AEK-4 Gate4 D-optimal Ablation Study (Confound-free)")
        print("=" * 70)
        print(f"System: AEK Self-balancing Motorcycle")
        dopt_mode = "SKIPPED" if self.cfg.skip_dopt else f"target={self.cfg.dopt_target}"
        print(f"Pool: seed={self.cfg.seed}, pool_seed={self.cfg.pool_seed}, "
              f"size={self.cfg.pool_size}")
        print(f"Selection: D-optimal ({dopt_mode}) vs Random "
              f"({len(self.cfg.random_seeds)} seeds)")
        print(f"Results: {self.results_base}")
        if self.cfg.excitation_tail:
            print(f"⚡ EXCITATION TAIL: phi0 ±{self.cfg.excitation_phi_range}, "
                  f"fraction={self.cfg.excitation_fraction}")
        print("=" * 70)

        all_results = {}

        try:
            print("\n[Phase 0] Setup...")
            self._setup()

            print("\n[Phase 1] Loading AEK-3 baseline artifacts...")
            self._load_artifacts()

            print("\n[Phase 2] Generating shared pool...")
            self._generate_shared_pool()

            print("\n[Phase 3] Track A selection...")
            self._run_track_a()

            if not self.cfg.skip_dopt:
                print(f"\n[Phase 4] D-optimal selection "
                      f"(target={self.cfg.dopt_target}) + evaluation...")
                dopt_result = self._run_dopt_selection()
                all_results['d_optimal'] = dopt_result
            else:
                print("\n[Phase 4] D-optimal SKIPPED (--skip_dopt)")

            for seed_val in self.cfg.random_seeds:
                print(f"\n[Phase 5.{seed_val}] Random selection "
                      f"(seed={seed_val}) + evaluation...")
                random_result = self._run_random_selection(seed_val)
                all_results[f'random_s{seed_val}'] = random_result

            print("\n[Phase 6] Generating summary...")
            summary = self._generate_summary(all_results)
            return summary

        except Exception as e:
            print(f"\n❌ Ablation failed: {e}")
            traceback.print_exc()
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': self.timestamp,
            }

    # ----------------------------------------------------------
    # Phase 0: Setup
    # ----------------------------------------------------------
    def _setup(self):
        """Setup directories and RNG."""
        self.results_base.mkdir(parents=True, exist_ok=True)
        self.rng_streams = create_rng_streams(
            self.cfg.seed, pool_seed=self.cfg.pool_seed
        )
        print(f"  Results: {self.results_base}")
        print(f"  RNG: seed={self.cfg.seed}, pool_seed={self.cfg.pool_seed}")

    # ----------------------------------------------------------
    # Phase 1: Load Artifacts
    # ----------------------------------------------------------
    def _load_artifacts(self):
        """Load dataset and AEK-3 baseline artifacts."""
        cfg = self.cfg

        # 1. Dataset
        if cfg.dataset_path:
            dataset_path = Path(cfg.dataset_path)
        else:
            dataset_path = paths.get_dataset_path(cfg.dataset_version, system=cfg.system)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        validate_dataset_lite(dataset_path)
        self.dataset = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  ✅ Dataset: {dataset_path}")
        print(f"     train_x: {self.dataset['train_x'].shape}")

        # 2. AEK-3 baseline directory
        baseline_path = Path(cfg.baseline_source)
        if not baseline_path.exists():
            baseline_path = self.project_root / cfg.baseline_source
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"AEK-3 baseline not found: {cfg.baseline_source}")

        # 3. Teacher support
        teacher_support_path = baseline_path / 'teacher_support.npy'
        if not teacher_support_path.exists():
            raise FileNotFoundError(f"teacher_support.npy not found in {baseline_path}")
        self.teacher_support = np.load(teacher_support_path)
        print(f"  ✅ Teacher support: {self.teacher_support.shape}")

        # 4. Teacher coefficients (sindy_coefficients.csv)
        coef_path = baseline_path / 'sindy_coefficients.csv'
        if coef_path.exists():
            self.teacher_coefficients = self._load_coefficients_csv(coef_path)
            print(f"  ✅ Teacher coefficients: {self.teacher_coefficients.shape}")
        else:
            print(f"  ⚠️ sindy_coefficients.csv not found, "
                  f"using teacher_support as mask")
            self.teacher_coefficients = self.teacher_support.astype(float)

        # v1.2 HOTFIX P0-1: Teacher coefficients top-10 diagnostic
        if self.teacher_coefficients is not None:
            tc = self.teacher_coefficients
            abs_flat = np.abs(tc).ravel()
            top_k = min(10, len(abs_flat))
            top_indices = np.argsort(abs_flat)[::-1][:top_k]
            print(f"  [DIAG] Teacher coefficients top-{top_k} by |value|:")
            for rank, flat_idx in enumerate(top_indices):
                f_idx = flat_idx // tc.shape[1]
                t_idx = flat_idx % tc.shape[1]
                f_name = AEK_FEATURE_NAMES[f_idx] if f_idx < len(AEK_FEATURE_NAMES) else f"f{f_idx}"
                t_name = AEK_TARGET_NAMES[t_idx] if t_idx < len(AEK_TARGET_NAMES) else f"t{t_idx}"
                val = tc[f_idx, t_idx]
                oracle = get_aek_oracle_support()
                is_oracle = "ORACLE" if oracle[f_idx, t_idx] else "spurious"
                print(f"    #{rank+1}: ({f_idx},{t_idx}) {f_name}->{t_name} = "
                      f"{val:.6e} [{is_oracle}]")

        # 5. z_before (control reference)
        z_before_path = baseline_path / 'z_before.npy'
        if not z_before_path.exists():
            raise FileNotFoundError(f"z_before.npy not found in {baseline_path}")
        self.z_before = np.load(z_before_path)
        with open(z_before_path, 'rb') as f:
            self._z_before_sha = hashlib.sha256(f.read()).hexdigest()[:16]
        print(f"  ✅ z_before: {self.z_before.shape}, SHA={self._z_before_sha}")

        # 6. Fragile pairs (with type information)
        fp_path = baseline_path / 'fragile_pairs.json'
        if not fp_path.exists():
            raise FileNotFoundError(f"fragile_pairs.json not found in {baseline_path}")
        with open(fp_path, 'r', encoding='utf-8') as f:
            fp_data = json.load(f)

        self.fragile_pairs = fp_data.get('pairs', [])
        n_dynamics = sum(1 for p in self.fragile_pairs if p.get('type') == 'dynamics')
        n_spurious = sum(1 for p in self.fragile_pairs if p.get('type') == 'spurious')
        print(f"  ✅ Fragile pairs: {len(self.fragile_pairs)} "
              f"(dynamics={n_dynamics}, spurious={n_spurious})")

        # 7. z_before is already a flat 1D array aligned with fragile_pairs order
        #    (AEK-3 saves z_before = z[fragile_mask], same iteration order as fragile_pairs)
        n_fragile = len(self.fragile_pairs)
        if self.z_before.ndim == 1 and len(self.z_before) == n_fragile:
            # Already flat and aligned — use directly
            self.fragile_z_before = self.z_before.copy()
            self._fragile_types = [p.get('type', 'unknown') for p in self.fragile_pairs]
            print(f"  ✅ fragile_z_before (flat): n={len(self.fragile_z_before)}")
        elif self.z_before.ndim == 2:
            # 2D matrix format (Cart-Pole style) — extract per pair
            fragile_z_before_list = []
            fragile_types = []
            for pair in self.fragile_pairs:
                f_idx = pair['feature_idx']
                t_idx = pair['target_idx']
                if f_idx < self.z_before.shape[0] and t_idx < self.z_before.shape[1]:
                    fragile_z_before_list.append(self.z_before[f_idx, t_idx])
                    fragile_types.append(pair.get('type', 'unknown'))
            self.fragile_z_before = np.array(fragile_z_before_list)
            self._fragile_types = fragile_types
            print(f"  ✅ fragile_z_before (2D extract): n={len(self.fragile_z_before)}")
        else:
            raise ValueError(
                f"z_before shape {self.z_before.shape} incompatible with "
                f"{n_fragile} fragile pairs"
            )

    def _load_coefficients_csv(self, path: Path) -> np.ndarray:
        """Load sindy_coefficients.csv → (n_features, n_targets) array."""
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        n_features = len(rows)
        n_targets = len(header) - 1  # First column is feature name
        coef = np.zeros((n_features, n_targets))
        for i, row in enumerate(rows):
            for j in range(n_targets):
                coef[i, j] = float(row[j + 1])
        return coef

    # ----------------------------------------------------------
    # Phase 2: Generate Shared Pool
    # ----------------------------------------------------------
    def _generate_shared_pool(self):
        """Generate pool shared across all selection methods."""
        cfg = self.cfg
        train_x = self.dataset['train_x'][:cfg.n_train]  # §3.3
        train_params = self.dataset['train_params'][:cfg.n_train]
        train_u = self.dataset['train_u'][:cfg.n_train]

        print(f"  Training data: n={cfg.n_train}, T={train_x.shape[1]}")

        # Diagnostic: training data statistics
        ic_train = train_x[:, 0, :]  # (N, 4)
        print(f"  Training IC ranges:")
        for dim, name in enumerate(['phi', 'phi_dot', 'theta_w', 'theta_w_dot']):
            vals = ic_train[:, dim]
            print(f"    {name}: [{vals.min():.6f}, {vals.max():.6f}], "
                  f"mean={vals.mean():.6f}, std={vals.std():.6f}")
        print(f"  Training I_w_C range: [{train_params.min():.6e}, {train_params.max():.6e}]")
        print(f"  Training u range: [{train_u.min():.6f}, {train_u.max():.6f}]")

        # v1.2 HOTFIX: clip I_w_C bounds to training range + margin
        IWC_MARGIN_FRAC = 0.10
        iwc_min_train = float(train_params.min())
        iwc_max_train = float(train_params.max())
        margin = IWC_MARGIN_FRAC * (iwc_max_train - iwc_min_train)
        iwc_old_min = AEK_GMM_BOUNDS['I_w_C_min']
        iwc_old_max = AEK_GMM_BOUNDS['I_w_C_max']
        AEK_GMM_BOUNDS['I_w_C_min'] = iwc_min_train - margin
        AEK_GMM_BOUNDS['I_w_C_max'] = iwc_max_train + margin
        print(f"  [HOTFIX] I_w_C bounds clipped: "
              f"[{iwc_old_min:.6e}, {iwc_old_max:.6e}] -> "
              f"[{AEK_GMM_BOUNDS['I_w_C_min']:.6e}, {AEK_GMM_BOUNDS['I_w_C_max']:.6e}]")
        print(f"           Training range: [{iwc_min_train:.6e}, {iwc_max_train:.6e}], "
              f"margin_frac={IWC_MARGIN_FRAC}")
        
        # Check: do training trajectories themselves stay within QC?
        train_max_phi = np.max(np.abs(train_x[:, :, 0]))
        train_max_phi_dot = np.max(np.abs(train_x[:, :, 1]))
        train_max_tw_dot = np.max(np.abs(train_x[:, :, 3]))
        print(f"  Training trajectory max: phi={train_max_phi:.4f}, "
              f"phi_dot={train_max_phi_dot:.4f}, theta_w_dot={train_max_tw_dot:.4f}")

        # Fit AEK GMM
        self.gmm_sampler = AEKGMMSampler(
            n_components=cfg.gmm_n_components,
            covariance_type=cfg.gmm_covariance_type,
            random_state=cfg.pool_seed,
        )
        self.gmm_sampler.fit(train_x, train_params)

        # GMM fit hash
        gmm_params = self.gmm_sampler.get_params_dict()
        gmm_json = json.dumps(gmm_params, sort_keys=True,
                              default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        self._gmm_fit_sha = hashlib.sha256(gmm_json.encode()).hexdigest()[:16]
        print(f"  GMM fit hash: {self._gmm_fit_sha}")

        # Generate pool
        pool_gen = AEKPoolGenerator(
            gmm_sampler=self.gmm_sampler,
            train_u=train_u,
            dt=cfg.sim_dt,
            T_steps=cfg.sim_T_steps,
            rng=self.rng_streams['pool'],
            excitation_tail=cfg.excitation_tail,
            excitation_phi_range=cfg.excitation_phi_range,
            excitation_fraction=cfg.excitation_fraction,
            excitation_max_phi_qc=cfg.excitation_max_phi_qc,
        )

        self.pool = pool_gen.generate_pool(
            target_n_accept=cfg.pool_size,
            max_attempts=cfg.max_pool_attempts,
        )

        n_gen = len(self.pool['trajectories'])
        print(f"  ✅ Pool: {n_gen} trajectories")

        # Pool hash
        pool_hash_data = {
            'n_trajectories': n_gen,
            'ic_mean': self.pool['ic'].mean(axis=0).tolist(),
            'params_mean': self.pool['params'].mean(axis=0).tolist(),
        }
        pool_json = json.dumps(pool_hash_data, sort_keys=True)
        self._pool_sha = hashlib.sha256(pool_json.encode()).hexdigest()[:16]
        print(f"  Pool hash: {self._pool_sha}")

    # ----------------------------------------------------------
    # Phase 3: Track A
    # ----------------------------------------------------------
    def _run_track_a(self):
        """Shared Track A filtering."""
        self.track_a_result = aek_track_a_selection(
            pool=self.pool,
            teacher_coefficients=self.teacher_coefficients,
            reject_ratio=self.cfg.reject_ratio,
            n_select=self.cfg.n_select,
        )
        n_passed = len(self.track_a_result['selected_indices'])
        n_pool = len(self.pool['trajectories'])
        print(f"  ✅ Track A: {n_passed}/{n_pool} passed "
              f"({n_passed / n_pool:.1%})")

    # ----------------------------------------------------------
    # Phase 4: D-optimal
    # ----------------------------------------------------------
    def _run_dopt_selection(self) -> AblationRunResult:
        """Run D-optimal selection."""
        cfg = self.cfg
        dopt_suffix = f"dopt_{cfg.dopt_target}"
        run_id = f"{self.timestamp}_nogit_aek4_ablation_{dopt_suffix}"
        results_dir = self.results_base / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"  run_id: {run_id}")

        try:
            train_x = self.dataset['train_x'][:cfg.n_train]  # §3.3
            train_u = self.dataset['train_u'][:cfg.n_train]

            selected = aek_track_b_dopt_selection(
                pool=self.pool,
                track_a_result=self.track_a_result,
                fragile_pairs=self.fragile_pairs,
                teacher_support=self.teacher_support,
                train_x=train_x,
                train_u=train_u,
                n_select=cfg.n_select,
                top_m_ratio=cfg.dopt_top_m_ratio,
                lambda_reg=cfg.dopt_lambda,
                use_teacher_intersection=cfg.dopt_use_teacher_intersection,
                gram_energy_mode=cfg.dopt_gram_energy_mode,
                trace_power=cfg.dopt_trace_power,
                dopt_target=cfg.dopt_target,
                rng=self.rng_streams['select'],
                results_dir=results_dir,
            )

            print(f"  Selected: {selected['stats']['n_selected']} trajectories")
            method_label = f'd_optimal_{cfg.dopt_target}'
            metrics = self._evaluate_selection(selected, results_dir, method_label)
            self._save_run_artifacts(
                run_id, results_dir, method_label, None, selected, metrics)

            return AblationRunResult(
                run_id=run_id,
                selection_method='d_optimal',
                selection_seed=None,
                results_dir=results_dir,
                status='completed',
                metrics=metrics,
            )
        except Exception as e:
            traceback.print_exc()
            return AblationRunResult(
                run_id=run_id,
                selection_method='d_optimal',
                selection_seed=None,
                results_dir=results_dir,
                status='failed',
                error_message=str(e),
            )

    # ----------------------------------------------------------
    # Phase 5: Random
    # ----------------------------------------------------------
    def _run_random_selection(self, selection_seed: int) -> AblationRunResult:
        """Run random selection with specific seed."""
        cfg = self.cfg
        run_id = f"{self.timestamp}_nogit_aek4_ablation_random_s{selection_seed}"
        results_dir = self.results_base / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"  run_id: {run_id}")

        try:
            selected = aek_random_selection(
                pool=self.pool,
                track_a_result=self.track_a_result,
                n_select=cfg.n_select,
                selection_seed=selection_seed,
            )

            metrics = self._evaluate_selection(selected, results_dir, 'random')
            self._save_run_artifacts(
                run_id, results_dir, 'random', selection_seed, selected, metrics)

            return AblationRunResult(
                run_id=run_id,
                selection_method='random',
                selection_seed=selection_seed,
                results_dir=results_dir,
                status='completed',
                metrics=metrics,
            )
        except Exception as e:
            traceback.print_exc()
            return AblationRunResult(
                run_id=run_id,
                selection_method='random',
                selection_seed=selection_seed,
                results_dir=results_dir,
                status='failed',
                error_message=str(e),
            )

    # ----------------------------------------------------------
    # Evaluation (Spurious-Primary Metric)
    # ----------------------------------------------------------
    def _evaluate_selection(
        self,
        selected: Dict[str, Any],
        results_dir: Path,
        method_name: str,
    ) -> Dict[str, Any]:
        """Evaluate selection with E-SINDy and compute AEK metrics."""
        cfg = self.cfg
        train_x = self.dataset['train_x'][:cfg.n_train]    # §3.3
        train_dx = self.dataset['train_dx_savgol'][:cfg.n_train]  # §3.3
        train_u = self.dataset['train_u'][:cfg.n_train]

        aug_x = selected['trajectories']
        aug_dx = selected['dx']
        aug_u = selected['u']

        print(f"  E-SINDy: {train_x.shape[0]} orig + {aug_x.shape[0]} aug")

        eval_result = aek_evaluate_with_esindy(
            train_x=train_x,
            train_dx=train_dx,
            train_u=train_u,
            aug_x=aug_x,
            aug_dx=aug_dx,
            aug_u=aug_u,
            bootstrap_B=cfg.bootstrap_B,
            threshold=cfg.threshold,
            seed=cfg.seed,
            z_eps=cfg.z_eps,
            tau_support=cfg.tau_support,
            z0=cfg.z0,
        )

        z_after = eval_result['z_scores']  # (14, 4)
        coef_mean = eval_result['coefficients_mean']

        # ============================================================
        # AEK Metric: Spurious-primary aug_pure (SIGN FLIP)
        # ============================================================
        fragile_z_after = []
        fragile_aug_pure = []
        fragile_abs_mean_after = []
        fragile_abs_mean_before = []

        n_effective = 0
        for idx, pair in enumerate(self.fragile_pairs):
            f_idx = pair['feature_idx']
            t_idx = pair['target_idx']
            pair_type = pair.get('type', 'unknown')

            if f_idx >= z_after.shape[0] or t_idx >= z_after.shape[1]:
                continue
            if idx >= len(self.fragile_z_before):
                continue

            z_a = z_after[f_idx, t_idx]
            z_b = self.fragile_z_before[idx]  # Aligned 1:1 with fragile_pairs
            abs_mean_after = abs(coef_mean[f_idx, t_idx])

            # |mean| before from teacher coefficients (baseline)
            abs_mean_before = None
            if self.teacher_coefficients is not None:
                abs_mean_before = float(abs(self.teacher_coefficients[f_idx, t_idx]))

            fragile_z_after.append(z_a)
            fragile_abs_mean_after.append(abs_mean_after)
            fragile_abs_mean_before.append(abs_mean_before)

            # Sign-aware aug_pure (AEK4 Metric SSOT)
            if pair_type == 'spurious':
                # z decrease = improvement → aug_pure = z_ctrl - z_gen
                aug_pure_val = z_b - z_a
            elif pair_type == 'dynamics':
                # z increase = improvement → aug_pure = z_gen - z_ctrl
                aug_pure_val = z_a - z_b
            else:
                # Default: dynamics convention
                aug_pure_val = z_a - z_b

            fragile_aug_pure.append(aug_pure_val)
            n_effective += 1

        fragile_z_after = np.array(fragile_z_after)
        fragile_aug_pure = np.array(fragile_aug_pure)
        fragile_abs_mean_after = np.array(fragile_abs_mean_after)
        fragile_abs_mean_before_arr = np.array([
            v if v is not None else np.nan for v in fragile_abs_mean_before
        ])

        # Primary metric: median_aug_pure (positive = improvement)
        median_aug_pure = float(np.median(fragile_aug_pure)) if n_effective > 0 else None

        # Bootstrap CI
        ci_lower, ci_upper = self._compute_bootstrap_ci(
            fragile_aug_pure, cfg.ci_bootstrap_B, cfg.ci_alpha, cfg.seed)

        # Pass level
        pass_level = self._classify_pass_level(
            median_aug_pure, ci_lower, cfg.gate2_ceiling)

        # ============================================================
        # Spurious-specific secondary metrics
        # ============================================================
        spurious_mask = np.array([
            p.get('type') == 'spurious' for p in self.fragile_pairs[:n_effective]
        ])
        dynamics_mask = np.array([
            p.get('type') == 'dynamics' for p in self.fragile_pairs[:n_effective]
        ])

        spurious_aug_pure = fragile_aug_pure[spurious_mask] if spurious_mask.any() else np.array([])
        spurious_abs_mean = fragile_abs_mean_after[spurious_mask] if spurious_mask.any() else np.array([])

        # Spurious |mean| before/after/delta (SSOT guard against fake improvement)
        spurious_abs_mean_before = fragile_abs_mean_before_arr[spurious_mask] if spurious_mask.any() else np.array([])
        spurious_abs_mean_before_valid = spurious_abs_mean_before[~np.isnan(spurious_abs_mean_before)]
        spurious_abs_mean_delta = None
        if len(spurious_abs_mean_before_valid) > 0 and len(spurious_abs_mean) > 0:
            # delta = before - after (positive = spurious |coef| decreased = improvement)
            valid_mask_s = ~np.isnan(spurious_abs_mean_before)
            if valid_mask_s.any():
                delta_arr = spurious_abs_mean_before[valid_mask_s] - spurious_abs_mean[valid_mask_s]
                spurious_abs_mean_delta = float(np.median(delta_arr))

        # Build metrics dict
        metrics = {
            'system': 'aek',
            'method': method_name,

            # Primary SSOT metrics (sign-corrected)
            'median_aug_pure': median_aug_pure,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pass_level': pass_level,

            # Control reference
            'z_before_sha': self._z_before_sha,

            # Secondary metrics
            'z_after_median': float(np.median(fragile_z_after)) if n_effective > 0 else None,
            'z_after_mean': float(np.mean(fragile_z_after)) if n_effective > 0 else None,

            # Spurious-specific (GPT recommended |mean| guard)
            'n_spurious_pairs': int(spurious_mask.sum()),
            'n_dynamics_pairs': int(dynamics_mask.sum()),
            'spurious_aug_pure_median': float(np.median(spurious_aug_pure)) if len(spurious_aug_pure) > 0 else None,
            'spurious_abs_mean_after': float(np.mean(spurious_abs_mean)) if len(spurious_abs_mean) > 0 else None,
            'spurious_abs_mean_before': float(np.mean(spurious_abs_mean_before_valid)) if len(spurious_abs_mean_before_valid) > 0 else None,
            'spurious_abs_mean_delta_median': spurious_abs_mean_delta,

            # Pair tracking
            'n_pairs_loaded': len(self.fragile_pairs),
            'n_pairs_effective': n_effective,

            # Sample counts
            'n_total_samples': eval_result['n_total'],
            'n_original': eval_result['n_original'],
            'n_augmented': eval_result['n_augmented'],

            # Config
            'bootstrap_B': cfg.bootstrap_B,
            'ci_bootstrap_B': cfg.ci_bootstrap_B,
            'ci_alpha': cfg.ci_alpha,
            'gate2_ceiling': cfg.gate2_ceiling,

            # Ceiling margin
            'ceiling_margin': float(ci_lower - cfg.gate2_ceiling) if ci_lower is not None else None,

            # Audit trail
            'pool_sha': self._pool_sha,
            'gmm_fit_sha': self._gmm_fit_sha,
            'seed': cfg.seed,
            'pool_seed': cfg.pool_seed,
            'pool_size': cfg.pool_size,
            'n_select': cfg.n_select,
            'runner_version': 'aek4_v1.3',

            # E-SINDy support summary
            'support_terms_total': int(eval_result['support_mask'].sum()),
            'stable_terms_total': int(eval_result['stable_core_mask'].sum()),
            'fragile_terms_total': int(eval_result['fragile_pool_mask'].sum()),
        }

        # Save arrays
        np.save(results_dir / 'z_after.npy', z_after)
        np.save(results_dir / 'fragile_z_after.npy', fragile_z_after)
        np.save(results_dir / 'fragile_aug_pure.npy', fragile_aug_pure)
        np.save(results_dir / 'fragile_z_before.npy',
                self.fragile_z_before[:n_effective])
        np.save(results_dir / 'fragile_abs_mean_after.npy', fragile_abs_mean_after)
        np.save(results_dir / 'fragile_abs_mean_before.npy', fragile_abs_mean_before_arr)
        np.save(results_dir / 'sindy_coefficients_mean.npy', coef_mean)
        np.save(results_dir / 'sindy_coefficients_std.npy',
                eval_result['coefficients_std'])

        # Save sindy_coefficients.csv (SSOT)
        self._save_coefficients_csv(
            results_dir / 'sindy_coefficients.csv',
            coef_mean,
            AEK_FEATURE_NAMES,
            AEK_TARGET_NAMES,
        )

        # effective_pairs.json
        effective_pairs_data = {
            'system': 'aek',
            'pairs': [],
            'n_loaded': len(self.fragile_pairs),
            'n_effective': n_effective,
            'feature_names': list(AEK_FEATURE_NAMES),
            'target_names': list(AEK_TARGET_NAMES),
        }
        for pair in self.fragile_pairs[:n_effective]:
            effective_pairs_data['pairs'].append({
                'feature_idx': pair['feature_idx'],
                'target_idx': pair['target_idx'],
                'type': pair.get('type', 'unknown'),
                'feature_name': pair.get('feature_name', ''),
                'target_name': pair.get('target_name', ''),
            })
        with open(results_dir / 'effective_pairs.json', 'w') as f:
            json.dump(effective_pairs_data, f, indent=2)

        # Print summary
        print(f"  Support: {metrics['support_terms_total']}, "
              f"Stable: {metrics['stable_terms_total']}, "
              f"Fragile: {metrics['fragile_terms_total']}")
        print(f"  median_aug_pure: {median_aug_pure:.4f}, "
              f"CI: [{ci_lower:.4f}, {ci_upper:.4f}], "
              f"pass_level: {pass_level}")
        if metrics['ceiling_margin'] is not None:
            print(f"  ceiling_margin: {metrics['ceiling_margin']:.4f}")
        print(f"  Spurious pairs: {metrics['n_spurious_pairs']}, "
              f"spurious_aug_pure_median: {metrics.get('spurious_aug_pure_median', 'N/A')}")
        if metrics.get('spurious_abs_mean_delta_median') is not None:
            print(f"  |mean| guard: before={metrics.get('spurious_abs_mean_before', 'N/A'):.6f}, "
                  f"after={metrics.get('spurious_abs_mean_after', 'N/A'):.6f}, "
                  f"delta_median={metrics['spurious_abs_mean_delta_median']:.6f}")

        return metrics

    def _save_coefficients_csv(
        self,
        path: Path,
        coef: np.ndarray,
        feature_names: List[str],
        target_names: List[str],
    ):
        """Save coefficients as CSV."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature'] + list(target_names))
            for i, fname in enumerate(feature_names):
                row = [fname] + [f"{coef[i, j]:.10e}" for j in range(coef.shape[1])]
                writer.writerow(row)

    def _compute_bootstrap_ci(
        self,
        data: np.ndarray,
        n_bootstrap: int,
        alpha: float,
        seed: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Bootstrap percentile CI for median."""
        if len(data) == 0:
            return (None, None)
        rng = np.random.default_rng(seed)
        boot_medians = []
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            boot_medians.append(np.median(sample))
        boot_medians = np.array(boot_medians)
        ci_lower = float(np.percentile(boot_medians, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
        return (ci_lower, ci_upper)

    def _classify_pass_level(
        self,
        median_aug_pure: Optional[float],
        ci_lower: Optional[float],
        gate2_ceiling: float,
    ) -> str:
        """Classify pass level (AEK Metric SSOT)."""
        if median_aug_pure is None or ci_lower is None:
            return "NULL"
        if ci_lower > gate2_ceiling:
            return "CEILING_BREAK"
        elif ci_lower > 0:
            return "STRONG_PASS"
        elif median_aug_pure > 0:
            return "SOFT_PASS"
        else:
            return "NULL"

    # ----------------------------------------------------------
    # Save Artifacts
    # ----------------------------------------------------------
    def _save_run_artifacts(
        self,
        run_id: str,
        results_dir: Path,
        method: str,
        selection_seed: Optional[int],
        selected: Dict[str, Any],
        metrics: Dict[str, Any],
    ):
        """Save manifest and metrics for a single run."""
        cfg = self.cfg

        manifest = {
            'run_id': run_id,
            'system': 'aek',
            'experiment': 'aek4_dopt_ablation',
            'design': 'confound_free',
            'timestamp': self.timestamp,

            'fixed_conditions': {
                'seed': cfg.seed,
                'pool_seed': cfg.pool_seed,
                'pool_size': cfg.pool_size,
                'pool_sha': self._pool_sha,
                'gmm_fit_sha': self._gmm_fit_sha,
                'z_before_sha': self._z_before_sha,
            },

            'selection': {
                'method': method,
                'selection_seed': selection_seed,
                'n_select': cfg.n_select,
                'reject_ratio': cfg.reject_ratio,
            },

            'evaluation': {
                'bootstrap_B': cfg.bootstrap_B,
                'threshold': cfg.threshold,
                'n_pairs_loaded': len(self.fragile_pairs),
                'ci_bootstrap_B': cfg.ci_bootstrap_B,
                'ci_alpha': cfg.ci_alpha,
                'gate2_ceiling': cfg.gate2_ceiling,
                'metric_convention': 'spurious_primary_sign_flip',
                'primary_metric': 'median_aug_pure',
                'sign_rule': {
                    'spurious': 'aug_pure = z_ctrl - z_gen (decrease = improvement)',
                    'dynamics': 'aug_pure = z_gen - z_ctrl (increase = improvement)',
                },
                'pass_level_rule': {
                    'CEILING_BREAK': 'ci_lower > gate2_ceiling',
                    'STRONG_PASS': 'ci_lower > 0',
                    'SOFT_PASS': 'median > 0',
                    'NULL': 'otherwise',
                },
            },

            'runner_version': 'aek4_v1.3',
            'hotfix_scaler_orig_only': True,
            'hotfix_iwc_clip': True,
            'excitation_tail': cfg.excitation_tail,
            'excitation_phi_range': cfg.excitation_phi_range if cfg.excitation_tail else None,
            'excitation_fraction': cfg.excitation_fraction if cfg.excitation_tail else None,

            'artifacts': [
                'manifest.json', 'metrics.json',
                'z_after.npy', 'fragile_z_after.npy', 'fragile_aug_pure.npy',
                'fragile_z_before.npy', 'fragile_abs_mean_after.npy',
                'fragile_abs_mean_before.npy',
                'effective_pairs.json', 'selected_indices.npy',
                'sindy_coefficients.csv',
                'sindy_coefficients_mean.npy', 'sindy_coefficients_std.npy',
            ],
        }

        with open(results_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        np.save(results_dir / 'selected_indices.npy', selected['original_indices'])

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    def _generate_summary(self, all_results: Dict[str, AblationRunResult]) -> Dict[str, Any]:
        """Generate ablation summary."""
        cfg = self.cfg

        dopt_result = all_results.get('d_optimal')
        random_results = [all_results[f'random_s{s}'] for s in cfg.random_seeds
                          if f'random_s{s}' in all_results]

        dopt_metrics = (dopt_result.metrics
                        if dopt_result and dopt_result.status == 'completed' else None)
        random_metrics_list = [
            r.metrics for r in random_results
            if r.status == 'completed' and r.metrics
        ]

        # Build runs list
        runs_list = []
        if dopt_result and dopt_result.status == 'completed' and dopt_metrics:
            runs_list.append({
                'method': 'd_optimal',
                'selection_seed': None,
                'run_id': dopt_result.run_id,
                'status': dopt_result.status,
                'median_aug_pure': dopt_metrics.get('median_aug_pure'),
                'ci_lower': dopt_metrics.get('ci_lower'),
                'ci_upper': dopt_metrics.get('ci_upper'),
                'pass_level': dopt_metrics.get('pass_level'),
                'ceiling_margin': dopt_metrics.get('ceiling_margin'),
                'spurious_aug_pure_median': dopt_metrics.get('spurious_aug_pure_median'),
                'spurious_abs_mean_delta_median': dopt_metrics.get('spurious_abs_mean_delta_median'),
            })

        for r in random_results:
            if r.status == 'completed' and r.metrics:
                runs_list.append({
                    'method': 'random',
                    'selection_seed': r.selection_seed,
                    'run_id': r.run_id,
                    'status': r.status,
                    'median_aug_pure': r.metrics.get('median_aug_pure'),
                    'ci_lower': r.metrics.get('ci_lower'),
                    'ci_upper': r.metrics.get('ci_upper'),
                    'pass_level': r.metrics.get('pass_level'),
                    'ceiling_margin': r.metrics.get('ceiling_margin'),
                    'spurious_aug_pure_median': r.metrics.get('spurious_aug_pure_median'),
                    'spurious_abs_mean_delta_median': r.metrics.get('spurious_abs_mean_delta_median'),
                })

        # Aggregate random
        if random_metrics_list:
            r_aug = [m['median_aug_pure'] for m in random_metrics_list
                     if m.get('median_aug_pure') is not None]
            pass_counts = {'NULL': 0, 'SOFT_PASS': 0, 'STRONG_PASS': 0, 'CEILING_BREAK': 0}
            for m in random_metrics_list:
                pl = m.get('pass_level', 'NULL')
                if pl in pass_counts:
                    pass_counts[pl] += 1
        else:
            r_aug = []
            pass_counts = {'NULL': 0, 'SOFT_PASS': 0, 'STRONG_PASS': 0, 'CEILING_BREAK': 0}

        summary = {
            'system': 'aek',
            'experiment_type': 'aek4_dopt_ablation',
            'design': 'confound_free',
            'metric_convention': 'spurious_primary_sign_flip',
            'timestamp': self.timestamp,

            'fixed_conditions': {
                'seed': cfg.seed,
                'pool_seed': cfg.pool_seed,
                'pool_size': cfg.pool_size,
                'n_select': cfg.n_select,
                'pool_sha': self._pool_sha,
                'gmm_fit_sha': self._gmm_fit_sha,
                'z_before_sha': self._z_before_sha,
                'bootstrap_B': cfg.bootstrap_B,
                'ci_bootstrap_B': cfg.ci_bootstrap_B,
                'gate2_ceiling': cfg.gate2_ceiling,
                'runner_version': 'aek4_v1.3',
                'skip_dopt': cfg.skip_dopt,
                'dopt_target': cfg.dopt_target,
                'excitation_tail': cfg.excitation_tail,
                'excitation_phi_range': cfg.excitation_phi_range if cfg.excitation_tail else None,
                'excitation_fraction': cfg.excitation_fraction if cfg.excitation_tail else None,
            },

            'runs': runs_list,

            'comparison': {
                'd_optimal': {
                    'median_aug_pure': dopt_metrics.get('median_aug_pure') if dopt_metrics else None,
                    'ci_lower': dopt_metrics.get('ci_lower') if dopt_metrics else None,
                    'ci_upper': dopt_metrics.get('ci_upper') if dopt_metrics else None,
                    'pass_level': dopt_metrics.get('pass_level') if dopt_metrics else None,
                    'ceiling_margin': dopt_metrics.get('ceiling_margin') if dopt_metrics else None,
                },
                'random': {
                    'median_aug_pure_mean': float(np.mean(r_aug)) if r_aug else None,
                    'median_aug_pure_std': float(np.std(r_aug)) if r_aug else None,
                    'pass_level_distribution': pass_counts,
                    'n_completed': len(random_metrics_list),
                },
            },

            'verdict': self._compute_verdict(dopt_metrics, random_metrics_list),
        }

        summary_path = self.results_base / 'ablation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print
        print(f"\n{'=' * 70}")
        mode_label = "Random-Only" if cfg.skip_dopt else "Spurious-Primary, Sign-Flipped"
        print(f"AEK-4 ABLATION SUMMARY ({mode_label})")
        print(f"{'=' * 70}")
        d = summary['comparison']['d_optimal']
        r = summary['comparison']['random']
        if d['median_aug_pure'] is not None:
            print(f"  D-optimal: median_aug_pure={d['median_aug_pure']:.4f}, "
                  f"pass={d['pass_level']}")
        elif cfg.skip_dopt:
            print(f"  D-optimal: SKIPPED")
        if r['median_aug_pure_mean'] is not None:
            print(f"  Random:    median_aug_pure={r['median_aug_pure_mean']:.4f} "
                  f"± {r['median_aug_pure_std']:.4f}")
        print(f"  Random pass distribution: {pass_counts}")
        print(f"\n  Verdict: {summary['verdict']['conclusion']}")
        print(f"{'=' * 70}")
        print(f"\n✅ Summary saved: {summary_path}")

        return summary

    def _compute_verdict(
        self,
        dopt_metrics: Optional[Dict],
        random_metrics_list: List[Dict],
    ) -> Dict[str, Any]:
        """Compute verdict comparing D-optimal vs Random."""
        if not dopt_metrics or not random_metrics_list:
            return {'conclusion': 'INCOMPLETE', 'reason': 'Missing metrics'}

        dopt_aug = dopt_metrics.get('median_aug_pure')
        dopt_pl = dopt_metrics.get('pass_level', 'NULL')

        random_augs = [m.get('median_aug_pure') for m in random_metrics_list
                       if m.get('median_aug_pure') is not None]

        if not random_augs or dopt_aug is None:
            return {'conclusion': 'INCOMPLETE', 'reason': 'Missing median_aug_pure'}

        r_mean = np.mean(random_augs)
        r_std = np.std(random_augs)
        advantage = dopt_aug - r_mean
        outside_1std = abs(advantage) > r_std if r_std > 0 else advantage != 0

        pass_order = {'NULL': 0, 'SOFT_PASS': 1, 'STRONG_PASS': 2, 'CEILING_BREAK': 3}
        dopt_score = pass_order.get(dopt_pl, 0)
        random_pls = [m.get('pass_level', 'NULL') for m in random_metrics_list]
        random_max_score = max(pass_order.get(pl, 0) for pl in random_pls)

        if dopt_aug > r_mean and outside_1std:
            conclusion = 'DOPT_ADVANTAGE_OBSERVED'
        elif dopt_aug < r_mean and outside_1std:
            conclusion = 'RANDOM_ADVANTAGE_OBSERVED'
        else:
            conclusion = 'NO_CLEAR_DIFFERENCE'

        return {
            'conclusion': conclusion,
            'dopt_median_aug_pure': float(dopt_aug),
            'random_median_aug_pure_mean': float(r_mean),
            'random_median_aug_pure_std': float(r_std),
            'dopt_advantage': float(advantage),
            'advantage_outside_1std': bool(outside_1std),
            'dopt_pass_level': dopt_pl,
            'dopt_pass_level_higher_than_all_random': dopt_score > random_max_score,
            'note': (
                f'Heuristic comparison. n={len(random_metrics_list)} random samples. '
                f'AEK metric uses spurious-primary sign flip.'
            ),
        }


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='AEK-4 Gate4 D-optimal Ablation Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (seed 0, 3 random seeds):
  python experiments/run_aek4_ablation.py ^
    --baseline_source results/aek_ood_v1/gate1/standardized/esindy/n10/seed0/20260207_143246_nogit_aek3_baseline ^
    --seed 1 --pool_seed 42 --pool_size 2000

Example (seed 1 baseline):
  python experiments/run_aek4_ablation.py ^
    --baseline_source results/aek_ood_v1/gate1/standardized/esindy/n10/seed1/20260207_143412_nogit_aek3_baseline ^
    --seed 1 --pool_seed 42 --pool_size 2000

Confound-free design:
  - Same pool (pool_seed=42, size=2000)
  - Same control reference (z_before from AEK-3)
  - Different selection only (D-optimal vs Random)
  - AEK metric: spurious-primary with sign flip
        """
    )

    # Required
    parser.add_argument('--baseline_source', type=str, required=True,
                        help='Path to AEK-3 baseline run directory')

    # Optional dataset override
    parser.add_argument('--dataset_path', type=str, default='',
                        help='Override dataset path (default: auto from paths.py)')

    # Pool settings
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pool_seed', type=int, default=42)
    parser.add_argument('--pool_size', type=int, default=2000)
    parser.add_argument('--n_select', type=int, default=200)
    parser.add_argument('--n_train', type=int, default=10)

    # Random seeds
    parser.add_argument('--random_seeds', type=int, nargs='+', default=[0, 1, 2])

    # D-optimal mode
    parser.add_argument('--skip_dopt', action='store_true',
                        help='Skip D-optimal, run random-only')
    parser.add_argument('--dopt_target', choices=['fragile', 'oracle'],
                        default='fragile',
                        help='D-opt feature target: fragile (default) or oracle')

    # Output
    parser.add_argument('--results_base', type=str,
                        default='results/aek_ood_v1/gate4/ablation/d_optimal_vs_random')

    # E-SINDy
    parser.add_argument('--bootstrap_B', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.05)

    # CI
    parser.add_argument('--ci_bootstrap_B', type=int, default=2000)
    parser.add_argument('--gate2_ceiling', type=float, default=0.058)

    # Step 2: Excitation tail injection
    parser.add_argument('--excitation_tail', action='store_true',
                        help='Enable phi0 tail injection to break cos(phi)≈1 collinearity')
    parser.add_argument('--excitation_phi_range', type=float, default=0.12,
                        help='±range for tail phi0 (default: 0.12)')
    parser.add_argument('--excitation_fraction', type=float, default=0.30,
                        help='Fraction of GMM samples to replace (default: 0.30)')

    return parser.parse_args()


def main():
    args = parse_args()

    config = AEK4AblationConfig(
        seed=args.seed,
        pool_seed=args.pool_seed,
        pool_size=args.pool_size,
        n_select=args.n_select,
        n_train=args.n_train,
        random_seeds=args.random_seeds,
        skip_dopt=args.skip_dopt,
        dopt_target=args.dopt_target,
        baseline_source=args.baseline_source,
        dataset_path=args.dataset_path,
        results_base=args.results_base,
        bootstrap_B=args.bootstrap_B,
        threshold=args.threshold,
        ci_bootstrap_B=args.ci_bootstrap_B,
        gate2_ceiling=args.gate2_ceiling,
        excitation_tail=args.excitation_tail,
        excitation_phi_range=args.excitation_phi_range,
        excitation_fraction=args.excitation_fraction,
    )

    runner = AEK4AblationRunner(config)
    summary = runner.run_all()

    print("\n" + "=" * 70)
    print("AEK-4 Ablation Study Complete")
    print("=" * 70)

    if summary.get('status') == 'failed':
        print(f"❌ Failed: {summary.get('error')}")
        sys.exit(1)

    print(f"Summary: {config.results_base}/ablation_summary.json")
    print("\nNext steps:")
    print("  1. Review ablation_summary.json")
    print("  2. GPT cross-review (especially sign flip logic)")
    print("  3. Expand to 10 random seeds if directional confirmation")
    print("  4. Run on seed1 baseline for cross-seed validation")


if __name__ == '__main__':
    main()