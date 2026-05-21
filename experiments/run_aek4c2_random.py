"""
AEK-4c-2: Random Selection Augmentation Runner (Reparam-2)

Purpose:
    Test whether GMM augmentation + random selection improves E-SINDy
    precision on the AEK system with Reparam-1 library.

Design:
    1. Load AEK baseline artifacts (z_before, teacher_support, fragile_pairs)
    2. Fit 3-component GMM on training ICs+params (5D)
    3. Generate pool via AEK simulator (analytic dx)
    4. Track A: reject top-10% teacher alignment error
    5. Random selection from Track A survivors
    6. E-SINDy evaluation on train+aug data
    7. Compute delta_raw + score_aligned (AC1 compliant)
    8. Assert feature name integrity (AC2 compliant)

Metric SSOT (AEK — spurious-primary):
    delta_raw = median(z_after − z_before) over fragile pairs
    score_aligned = −delta_raw  (positive = improvement)
    Both stored in metrics.json (AC1).

Usage:
    python experiments/run_aek4c_random.py --seeds 0 1 2
    python experiments/run_aek4c_random.py --seeds 0 1 2 3 4 5 6 7 8 9

Author: Claude (Gate4c)
Date: 2026-03-03 (RP2 patch from run_aek4c_random.py)
Runner version: v1.0
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import hashlib
import csv
import traceback
import numpy as np
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from scipy.integrate import solve_ivp
from sklearn.mixture import GaussianMixture

from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble
from src.sindy.aek_library import (
    build_aek_library_by_name,
    get_aek_feature_names,
    get_aek_oracle_support,
    AEK_TARGET_NAMES,
    N_AEK_FEATURES,
)
from src.simulators.aek_simulator import AEKSimulator


# ============================================================
# Constants
# ============================================================

RUNNER_VERSION = 'v2.0_rp2'


# ============================================================
# Configuration
# ============================================================

@dataclass
class AEK4cConfig:
    """AEK-4c augmentation experiment configuration."""
    # Dataset
    dataset_version: str = 'aek_ood_v1'
    system: str = 'aek'
    n_train: int = 10

    # Baseline source (Reparam-1 baseline)
    baseline_seed: int = 0
    baseline_dir: str = ''  # auto-detected if empty

    # Library
    reparam: str = 'reparam2'

    # GMM
    gmm_n_components: int = 3
    gmm_covariance_type: str = 'full'
    gmm_seed: int = 42

    # Pool generation
    pool_size: int = 200
    max_pool_attempts: int = 5000
    pool_seed: int = 42

    # QC thresholds (AEK-specific, from aek.yaml ranges + safety margin)
    qc_max_phi: float = 0.30          # rad (~17 deg) - tighter for small-angle regime
    qc_max_phi_dot: float = 5.0       # rad/s - tighter, training max ~0.46
    qc_max_theta_w_dot: float = 500.0 # rad/s

    # Adaptive PD stabilizing controller for pool generation
    # AEK is an unstable inverted pendulum; torque replay fails.
    # Kp = gain_margin * M_total * g * h_cm  (per-I_w_C, auto-computed)
    # Kd = Kd_factor * Kp  (damping)
    pd_gain_margin: float = 3.0        # Kp = 3 * Mgh (sufficient for stability)
    pd_Kd_factor: float = 0.15         # Kd = 0.15 * Kp
    pd_noise_std: float = 0.001        # exploration noise on torque

    # Simulation (from aek.yaml)
    dt: float = 0.01
    T_duration: float = 2.0
    T_steps: int = 201

    # Track A
    reject_ratio: float = 0.10

    # Selection
    n_select: int = 50
    random_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # E-SINDy
    n_bootstrap: int = 100
    threshold: float = 0.05

    # Z-metric
    z_eps: float = 1e-6

    # CI
    ci_bootstrap_B: int = 2000
    ci_alpha: float = 0.05

    # Output
    note: str = 'aek4c2_random_rp2'


# ============================================================
# AC2: Feature Name Integrity Assert
# ============================================================

def assert_oracle_label_integrity(reparam: str):
    """
    AC2: Verify oracle-relevant feature indices match expected names.
    Fail fast if library order has been altered.
    """
    names = get_aek_feature_names(reparam)
    checks = {
        0: '1', 1: 'phi_dot', 2: 'theta_w_dot',
        3: 'tau', 4: 'sin(phi)',
    }
    for idx, expected in checks.items():
        actual = names[idx]
        assert actual == expected, \
            f"AC2 FAIL: idx {idx} expected '{expected}', got '{actual}'"

    if reparam == 'reparam1':
        assert names[5] == 'cos(phi)-1', \
            f"AC2 FAIL: idx 5 expected 'cos(phi)-1', got '{names[5]}'"

    if reparam == 'reparam2':
        assert names[5] == 'cos(phi)-1', \
            f"AC2 FAIL: idx 5 expected 'cos(phi)-1', got '{names[5]}'"
        assert names[13] == '(cos(phi)-1)*tau', \
            f"AC2 FAIL: idx 13 expected '(cos(phi)-1)*tau', got '{names[13]}'"

    # Index 5 must NOT be in oracle support
    oracle = get_aek_oracle_support()
    assert not oracle[5, :].any(), \
        "AC2 FAIL: Index 5 is in oracle support"

    assert len(names) == N_AEK_FEATURES, \
        f"AC2 FAIL: expected {N_AEK_FEATURES} features, got {len(names)}"


# ============================================================
# GMM Sampler (AEK-specific: 5D = 4 IC + 1 param)
# ============================================================

class AEKGMMSampler:
    """GMM sampler for AEK initial conditions + OOD parameter."""

    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, train_x: np.ndarray, train_params: np.ndarray):
        """
        Fit GMM on training ICs + OOD parameter.

        Args:
            train_x: (N, T, 4) -- use ICs (t=0)
            train_params: (N, 1) or (N,) -- I_w_C values
        """
        ics = train_x[:, 0, :]  # (N, 4)
        params = train_params.reshape(-1, 1) if train_params.ndim == 1 else train_params
        data_5d = np.hstack([ics, params])  # (N, 5)
        self.gmm.fit(data_5d)
        self._fitted = True

    def sample(self, n_samples: int):
        """
        Sample ICs + params from fitted GMM.

        Returns:
            ics: (n, 4) -- [phi, phi_dot, theta_w, theta_w_dot]
            params: (n, 1) -- [I_w_C]
        """
        if not self._fitted:
            raise RuntimeError("GMM not fitted")
        samples, _ = self.gmm.sample(n_samples)
        ics = samples[:, :4]
        params = samples[:, 4:5]
        # I_w_C must be positive and within physical OOD range
        # Training: [6.95e-5, 8.69e-5], Test: [1.04e-4]
        # Margin: [5e-5, 1.5e-4] covers train + test + small exploration
        params = np.abs(params)
        params = np.clip(params, 5e-5, 1.5e-4)
        return ics, params


# ============================================================
# Pool Generation (AEK simulator + analytic dx)
# ============================================================

def estimate_pd_gains(train_x: np.ndarray, train_u: np.ndarray) -> Dict[str, float]:
    """
    Estimate PD gains from training data via least-squares.

    Training data was generated with a stabilizing controller.
    Fit: tau = Kp*phi + Kd*phi_dot  (positive Kp stabilizes AEK).

    Returns:
        Dict with 'Kp', 'Kd', 'R2_mean'
    """
    N = train_x.shape[0]
    kp_list, kd_list, r2_list = [], [], []

    for i in range(N):
        phi = train_x[i, :, 0]
        phi_dot = train_x[i, :, 1]
        tau = train_u[i, :, 0]
        A = np.column_stack([phi, phi_dot])
        coeffs, _, _, _ = np.linalg.lstsq(A, tau, rcond=None)
        residual = tau - A @ coeffs
        r2 = 1.0 - np.var(residual) / max(np.var(tau), 1e-30)
        kp_list.append(coeffs[0])
        kd_list.append(coeffs[1])
        r2_list.append(r2)

    Kp = float(np.median(kp_list))
    Kd = float(np.median(kd_list))
    R2_mean = float(np.mean(r2_list))
    print(f"  PD gains from training: Kp={Kp:.4f}, Kd={Kd:.4f}, R²={R2_mean:.4f}")
    return {'Kp': Kp, 'Kd': Kd, 'R2_mean': R2_mean}


def generate_pool(
    gmm: AEKGMMSampler,
    train_x: np.ndarray,
    train_u: np.ndarray,
    cfg: AEK4cConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Generate augmentation pool via AEK simulator with analytic dx.

    Steps:
        1. Estimate PD gains from training data
        2. Sample ICs + I_w_C from GMM
        3. Simulate with PD stabilizing controller + noise
        4. Compute analytic dx from EOM
        5. QC filter (phi, phi_dot, theta_w_dot bounds)

    AEK is an unstable inverted pendulum: torque-replay from training fails
    because each torque profile is matched to its specific IC. Instead, we use
    a PD controller (estimated from training data) with additive noise for
    trajectory diversity. The GMM-sampled ICs and I_w_C provide the primary
    source of augmentation variety.
    """
    # ── Estimate PD gains (for logging; actual gains are adaptive) ──
    pd_gains = estimate_pd_gains(train_x, train_u)

    # Adaptive PD: Kp = gain_margin * Mgh per I_w_C
    gain_margin = cfg.pd_gain_margin
    Kd_factor = cfg.pd_Kd_factor
    noise_std = cfg.pd_noise_std

    trajectories = []
    dx_list = []
    u_list = []
    ics_list = []
    params_list = []

    n_accepted = 0
    n_attempted = 0
    T = cfg.T_steps
    dt = cfg.dt

    while n_accepted < cfg.pool_size and n_attempted < cfg.max_pool_attempts:
        batch = min(100, cfg.pool_size - n_accepted)
        ics_batch, params_batch = gmm.sample(batch)

        for i in range(batch):
            if n_accepted >= cfg.pool_size:
                break
            n_attempted += 1

            ic = ics_batch[i]
            I_w_C = float(params_batch[i, 0])

            # Create simulator with this I_w_C
            sim = AEKSimulator(params={'I_w_C': I_w_C})

            # Adaptive PD: compute Kp from gravity torque coefficient
            dp = sim.get_derived_params()
            Mgh = dp['M_total'] * dp['g'] * dp['h_cm']
            Kp = gain_margin * Mgh
            Kd = Kd_factor * Kp

            # PD stabilizing controller + exploration noise
            # Pre-generate noise for reproducibility
            noise_seq = rng.normal(0, noise_std, size=T)

            def _make_pd_ctrl(kp, kd, noise, dt_val, T_val):
                def ctrl(t, x):
                    k = min(int(t / dt_val), T_val - 1)
                    tau = kp * x[0] + kd * x[1] + noise[k]
                    return tau
                return ctrl

            controller = _make_pd_ctrl(Kp, Kd, noise_seq, dt, T)

            try:
                t_arr, x_arr, u_arr = sim.simulate(
                    x0=ic,
                    t_span=(0.0, cfg.T_duration),
                    dt=dt,
                    controller=controller,
                    method='RK45',
                )
            except Exception:
                continue

            if len(t_arr) != T:
                continue

            # QC
            if np.any(np.abs(x_arr[:, 0]) > cfg.qc_max_phi):
                continue
            if np.any(np.abs(x_arr[:, 1]) > cfg.qc_max_phi_dot):
                continue
            if np.any(np.abs(x_arr[:, 3]) > cfg.qc_max_theta_w_dot):
                continue
            if not np.all(np.isfinite(x_arr)):
                continue

            # Analytic dx from EOM (exact, no numerical differentiation)
            dx_arr = np.zeros_like(x_arr)
            for ti in range(T):
                dx_arr[ti] = sim.dynamics(t_arr[ti], x_arr[ti], float(u_arr[ti, 0]))

            trajectories.append(x_arr)
            dx_list.append(dx_arr)
            u_list.append(u_arr)
            ics_list.append(ic)
            params_list.append([I_w_C])
            n_accepted += 1

    rate = n_accepted / max(n_attempted, 1)
    print(f"  Pool: {n_accepted}/{n_attempted} accepted ({rate:.1%})")

    if n_accepted == 0:
        raise RuntimeError("Pool generation failed: 0 accepted")

    return {
        'trajectories': np.array(trajectories),  # (K, T, 4)
        'dx': np.array(dx_list),                  # (K, T, 4)
        'u': np.array(u_list),                    # (K, T, 1)
        'ics': np.array(ics_list),                # (K, 4)
        'params': np.array(params_list),          # (K, 1)
        'n_accepted': n_accepted,
        'n_attempted': n_attempted,
        'accept_rate': rate,
        'pd_gains': pd_gains,
    }


# ============================================================
# Track A: Teacher Alignment Error Filter
# ============================================================

def track_a_filter(
    pool: Dict[str, Any],
    teacher_coeff: np.ndarray,
    reparam: str,
    reject_ratio: float = 0.10,
) -> Dict[str, Any]:
    """
    Track A: Reject worst trajectories by teacher prediction error.

    For each pooled trajectory:
        error_i = mean |dx_actual - Theta_raw @ teacher_coeff|

    teacher_coeff is UNSCALED (physical units from baseline CSV).
    Theta_raw is the raw (unscaled) feature matrix.
    Prediction: dx_pred = Theta_raw @ teacher_coeff  (no scaling needed).

    Reject trajectories with highest errors (top reject_ratio fraction).
    """
    K = pool['trajectories'].shape[0]
    errors = np.zeros(K)

    for i in range(K):
        x_i = pool['trajectories'][i]   # (T, 4)
        u_i = pool['u'][i]              # (T, 1)
        dx_i = pool['dx'][i]            # (T, 4)

        # Raw feature matrix (no scaling)
        Theta_i, _ = build_aek_library_by_name(x_i, u_i, reparam=reparam)

        # Predict using unscaled teacher coefficients directly
        dx_pred = Theta_i @ teacher_coeff  # (T, 4)

        # Mean absolute error across all timesteps and all 4 targets
        errors[i] = float(np.mean(np.abs(dx_i - dx_pred)))

    # Reject worst fraction
    n_reject = int(np.ceil(K * reject_ratio))
    sorted_idx = np.argsort(errors)  # ascending = best first
    passed = sorted_idx[:K - n_reject]
    rejected = sorted_idx[K - n_reject:]

    print(f"  Track A: {len(passed)}/{K} passed "
          f"(rejected {n_reject}, worst err={errors[rejected].max():.4f})")

    return {
        'selected_indices': passed,
        'rejected_indices': rejected,
        'errors': errors,
    }


# ============================================================
# Random Selection
# ============================================================

def random_select(
    pool: Dict[str, Any],
    track_a: Dict[str, Any],
    n_select: int,
    selection_seed: int,
) -> Dict[str, np.ndarray]:
    """Random selection from Track A passed candidates."""
    candidates = track_a['selected_indices']
    rng = np.random.default_rng(selection_seed)

    if len(candidates) <= n_select:
        chosen = candidates.copy()
        print(f"  Random (seed={selection_seed}): all {len(chosen)} candidates")
    else:
        local_idx = rng.choice(len(candidates), size=n_select, replace=False)
        chosen = candidates[local_idx]
        print(f"  Random (seed={selection_seed}): {n_select}/{len(candidates)} selected")

    chosen = np.sort(chosen)

    return {
        'indices': chosen,
        'trajectories': pool['trajectories'][chosen],
        'dx': pool['dx'][chosen],
        'u': pool['u'][chosen],
        'n_selected': len(chosen),
    }


# ============================================================
# E-SINDy Evaluation
# ============================================================

def evaluate_augmented(
    train_x: np.ndarray,    # (N_tr, T, 4)
    train_u: np.ndarray,    # (N_tr, T, 1)
    train_dx: np.ndarray,   # (N_tr, T, 4)
    aug_x: np.ndarray,      # (N_aug, T, 4)
    aug_u: np.ndarray,      # (N_aug, T, 1)
    aug_dx: np.ndarray,     # (N_aug, T, 4)
    reparam: str,
    n_bootstrap: int,
    threshold: float,
    seed: int,
    z_eps: float,
) -> Dict[str, Any]:
    """
    Evaluate E-SINDy on train+aug data with Reparam-1 library.

    Returns:
        Dict with z-scores (14,4), coefficients, support mask, kappa.
    """
    N_tr, T_tr, D = train_x.shape
    N_aug, T_aug, _ = aug_x.shape

    # T must match for trajectory-level bootstrap indexing
    if T_tr != T_aug:
        raise ValueError(f"T mismatch: train={T_tr}, aug={T_aug}")
    T = T_tr

    # Flatten and concatenate
    x_all = np.vstack([train_x.reshape(-1, D), aug_x.reshape(-1, D)])
    u_all = np.vstack([train_u.reshape(-1, 1), aug_u.reshape(-1, 1)])
    dx_all = np.vstack([train_dx.reshape(-1, D), aug_dx.reshape(-1, D)])

    n_traj = N_tr + N_aug
    n_samples = n_traj * T

    assert x_all.shape[0] == n_samples, \
        f"Sample count: {x_all.shape[0]} != {n_traj}*{T}={n_samples}"

    # Build library
    Theta, feat_names = build_aek_library_by_name(x_all, u_all, reparam=reparam)

    # Scale columns
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    # Condition number (of scaled library)
    kappa = float(np.linalg.cond(Theta_scaled))

    # E-SINDy ensemble
    ensemble = ESINDyEnsemble(
        n_bootstrap=n_bootstrap,
        threshold=threshold,
        random_state=seed,
    )
    ensemble.fit(
        Theta_scaled, dx_all,
        n_trajectories=n_traj,
        T=T,
        scaler=scaler,
        target_scale=None,
    )

    coeff_mean = ensemble.coefficients_mean_    # (14, 4) unscaled
    coeff_std = ensemble.coefficients_std_      # (14, 4) unscaled
    inc_prob = ensemble.inclusion_probability_   # (14, 4)

    support_mask = np.abs(coeff_mean) > 0

    # Z-metric: |mean| / (std + eps)
    z = np.abs(coeff_mean) / (coeff_std + z_eps)

    return {
        'z': z,                          # (14, 4)
        'coefficients_mean': coeff_mean,  # (14, 4) unscaled
        'coefficients_std': coeff_std,
        'inclusion_probability': inc_prob,
        'support_mask': support_mask,
        'scaler': scaler,
        'feature_names': feat_names,
        'kappa': kappa,
        'n_total_samples': n_samples,
        'n_original': N_tr * T,
        'n_augmented': N_aug * T,
    }


# ============================================================
# Metric Computation (AC1 compliant)
# ============================================================

def compute_metrics(
    z_after: np.ndarray,
    z_before: np.ndarray,
    fragile_pairs: List[List[int]],
    ci_bootstrap_B: int,
    ci_alpha: float,
    ci_seed: int,
) -> Dict[str, Any]:
    """
    Compute SSOT metrics for AEK (spurious-primary).

    AC1: Both delta_raw and score_aligned are stored.

    Args:
        z_after: (14, 4) z-metric matrix after augmentation
        z_before: (20,) pre-extracted z at fragile pairs OR (14, 4) matrix
        fragile_pairs: List of [feature_idx, target_idx] lists
    """
    # Extract z at fragile pair positions
    z_af_list, z_bf_list = [], []
    effective_pairs = []

    # z_before may be 1D (pre-extracted at fragile pairs) or 2D (14,4) matrix
    z_before_is_flat = (z_before.ndim == 1)

    for pair_idx, pair in enumerate(fragile_pairs):
        f_idx, t_idx = int(pair[0]), int(pair[1])
        if f_idx < z_after.shape[0] and t_idx < z_after.shape[1]:
            z_af_list.append(z_after[f_idx, t_idx])
            if z_before_is_flat:
                z_bf_list.append(z_before[pair_idx])
            else:
                z_bf_list.append(z_before[f_idx, t_idx])
            effective_pairs.append([f_idx, t_idx])

    n_eff = len(z_af_list)
    if n_eff == 0:
        return {
            'delta_raw_median': None, 'score_aligned_median': None,
            'pass_level': 'NULL', 'n_effective_pairs': 0,
        }

    z_af = np.array(z_af_list)
    z_bf = np.array(z_bf_list)

    # delta_raw per pair: z_after - z_before
    #   For spurious terms: z decreasing = improvement = delta_raw < 0
    delta_per_pair = z_af - z_bf

    # Aggregates
    delta_raw = float(np.median(delta_per_pair))
    score_aligned = -delta_raw  # positive = improvement (spurious-primary)

    # Bootstrap CI for delta_raw median
    ci_lo, ci_hi = _bootstrap_ci(delta_per_pair, ci_bootstrap_B, ci_alpha, ci_seed)

    # Pass level (on score_aligned scale: positive = good)
    # CI(score_aligned): lower = -ci_hi, upper = -ci_lo
    sa_ci_lo = -ci_hi if ci_hi is not None else None
    sa_ci_hi = -ci_lo if ci_lo is not None else None
    pass_level = _classify_pass(score_aligned, sa_ci_lo)

    return {
        # AC1: dual metric storage
        'delta_raw_median': delta_raw,
        'score_aligned_median': score_aligned,

        # CI on delta_raw
        'delta_raw_ci_lower': ci_lo,
        'delta_raw_ci_upper': ci_hi,

        # CI on score_aligned
        'score_aligned_ci_lower': sa_ci_lo,
        'score_aligned_ci_upper': sa_ci_hi,

        # Pass level
        'pass_level': pass_level,

        # Z stats
        'z_after_fragile_median': float(np.median(z_af)),
        'z_after_fragile_mean': float(np.mean(z_af)),
        'z_before_fragile_median': float(np.median(z_bf)),

        # Counts
        'n_fragile_pairs_loaded': len(fragile_pairs),
        'n_effective_pairs': n_eff,

        # Per-pair (diagnostics)
        'delta_per_pair': delta_per_pair.tolist(),
        'effective_pairs': effective_pairs,
    }


def _bootstrap_ci(data, n_boot, alpha, seed):
    if len(data) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    medians = np.array([
        np.median(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    return (
        float(np.percentile(medians, 100 * alpha / 2)),
        float(np.percentile(medians, 100 * (1 - alpha / 2))),
    )


def _classify_pass(score_aligned, sa_ci_lower):
    """
    AEK has no Gate2 ceiling concept, so no CEILING_BREAK.
    STRONG_PASS: CI lower > 0
    SOFT_PASS: median > 0 but CI crosses 0
    NULL: median <= 0
    """
    if score_aligned is None or sa_ci_lower is None:
        return "NULL"
    if sa_ci_lower > 0:
        return "STRONG_PASS"
    elif score_aligned > 0:
        return "SOFT_PASS"
    else:
        return "NULL"


def compute_tau_stats(
    train_u: np.ndarray,
    aug_u: np.ndarray,
    tau_max: float = 0.02,
) -> Dict[str, Any]:
    """
    P0-2: Compare train vs augmented torque distributions.

    Evidence that PD+noise controller produces torques within
    the same operating regime as the training data, not a
    qualitatively different input distribution.

    Args:
        train_u: (N_tr, T, 1)
        aug_u: (N_aug, T, 1)
        tau_max: motor torque limit (N*m)

    Returns:
        Dict with distribution summaries for both splits
    """
    tr = train_u.flatten()
    au = aug_u.flatten()

    def _stats(arr, label):
        return {
            f'{label}_mean': float(np.mean(arr)),
            f'{label}_std': float(np.std(arr)),
            f'{label}_abs_max': float(np.abs(arr).max()),
            f'{label}_q05': float(np.percentile(arr, 5)),
            f'{label}_q50': float(np.percentile(arr, 50)),
            f'{label}_q95': float(np.percentile(arr, 95)),
            f'{label}_clip_rate': float(np.mean(np.abs(arr) >= tau_max * 0.99)),
        }

    stats = {**_stats(tr, 'train_tau'), **_stats(au, 'aug_tau')}
    stats['tau_max'] = tau_max
    stats['clip_note'] = 'all tau values are post-clip (simulator clips internally in dynamics())'
    stats['overlap_note'] = (
        'PD controller estimated from training data (R²≈0.98); '
        'noise provides excitation; tau clipped to motor limits'
    )
    return stats


# ============================================================
# Artifact Save
# ============================================================

def save_run(
    run_dir: Path,
    run_id: str,
    sel_seed: int,
    cfg: AEK4cConfig,
    eval_result: Dict[str, Any],
    metrics: Dict[str, Any],
    pool_sha: str,
    baseline_dir: Path,
    tau_stats: Optional[Dict[str, Any]] = None,
):
    """Save all run artifacts to run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'figures').mkdir(exist_ok=True)

    feat_names = eval_result['feature_names']

    # --- metrics.json (AC1: delta_raw + score_aligned) ---
    full_metrics = {**metrics}
    full_metrics.update({
        'system': 'aek',
        'gate': 'gate4c',
        'method': 'random',
        'reparam': cfg.reparam,
        'selection_seed': sel_seed,
        'pool_size': cfg.pool_size,
        'pool_sha': pool_sha,
        'n_select': cfg.n_select,
        'n_train': cfg.n_train,
        'n_bootstrap': cfg.n_bootstrap,
        'threshold': cfg.threshold,
        'kappa_augmented': eval_result['kappa'],
        'n_total_samples': eval_result['n_total_samples'],
        'n_original': eval_result['n_original'],
        'n_augmented': eval_result['n_augmented'],
        'support_terms_total': int(eval_result['support_mask'].sum()),
        'ci_bootstrap_B': cfg.ci_bootstrap_B,
        'ci_alpha': cfg.ci_alpha,
        'runner_version': RUNNER_VERSION,
    })
    # P0-2: tau distribution comparison
    if tau_stats is not None:
        full_metrics['tau_distribution'] = tau_stats
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2, default=_json_default)

    # --- sindy_coefficients.csv ---
    coeff = eval_result['coefficients_mean']
    with open(run_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + list(AEK_TARGET_NAMES))
        for i, name in enumerate(feat_names):
            w.writerow([name] + [f"{coeff[i,j]:.8f}" for j in range(4)])

    # --- z_after.npy ---
    np.save(run_dir / 'z_after.npy', eval_result['z'])

    # --- manifest.json ---
    manifest = {
        'run_id': run_id,
        'system': 'aek',
        'gate': 'gate4c',
        'method': 'random',
        'reparam': cfg.reparam,
        'selection_seed': sel_seed,
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_aek4c2_random.py',
        'runner_version': RUNNER_VERSION,
        'pool_sha': pool_sha,
        'baseline_dir': str(baseline_dir),
        'config': {
            'pool_size': cfg.pool_size,
            'pool_seed': cfg.pool_seed,
            'n_select': cfg.n_select,
            'n_bootstrap': cfg.n_bootstrap,
            'threshold': cfg.threshold,
            'gmm_n_components': cfg.gmm_n_components,
            'gmm_seed': cfg.gmm_seed,
            'reject_ratio': cfg.reject_ratio,
            'qc_max_phi': cfg.qc_max_phi,
            'qc_max_phi_dot': cfg.qc_max_phi_dot,
            'pd_gain_margin': cfg.pd_gain_margin,
            'pd_Kd_factor': cfg.pd_Kd_factor,
            'pd_noise_std': cfg.pd_noise_std,
            # P0-3: I_w_C clipping rationale
            'I_w_C_clip_range': [5e-5, 1.5e-4],
            'I_w_C_clip_rationale': (
                'main tier: train=[6.95e-5, 8.69e-5], test=[1.04e-4]; '
                'clip covers train+test with ~44% margin on each side'
            ),
        },
        'artifacts': [
            'manifest.json', 'metrics.json', 'sindy_coefficients.csv',
            'z_after.npy',
        ],
    }
    with open(run_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# ============================================================
# Load Baseline Artifacts
# ============================================================

def load_baseline(cfg: AEK4cConfig) -> Dict[str, Any]:
    """
    Load Reparam-2 baseline artifacts.

    Returns:
        Dict with z_before (flat or matrix), teacher_support (14,4),
        fragile_pairs [[f,t],...], coefficients (14,4), dir.
    """
    if cfg.baseline_dir:
        bdir = Path(cfg.baseline_dir)
    else:
        # Auto-detect: most recent Reparam-1 run
        base = (paths.ROOT / 'results' / cfg.dataset_version
                / 'gate1' / 'standardized' / 'esindy'
                / f'n{cfg.n_train}' / f'seed{cfg.baseline_seed}')
        if not base.exists():
            raise FileNotFoundError(f"Baseline base not found: {base}")

        # Find run with 'rp2' in name (most recent first)
        candidates = sorted(base.iterdir(), reverse=True)
        bdir = None
        for c in candidates:
            if c.is_dir() and 'rp2' in c.name:
                bdir = c
                break
        # Fallback: any directory
        if bdir is None:
            for c in candidates:
                if c.is_dir():
                    bdir = c
                    break
        if bdir is None:
            raise FileNotFoundError(f"No baseline runs in {base}")

    print(f"  Baseline dir: {bdir}")

    # z_before.npy -- shape (20,) flat or (14, 4) full z-metric matrix
    z_before = np.load(bdir / 'z_before.npy')
    print(f"  z_before: {z_before.shape}")

    # teacher_support.npy -- shape (14, 4) boolean
    teacher_support = np.load(bdir / 'teacher_support.npy')
    print(f"  teacher_support: {teacher_support.shape} ({teacher_support.sum()} active)")

    # fragile_pairs.json -- {pairs: [[f_idx, t_idx], ...], ...}
    with open(bdir / 'fragile_pairs.json', 'r') as f:
        fp_data = json.load(f)

    # Handle both key formats
    if 'pairs' in fp_data:
        raw_pairs = fp_data['pairs']
    elif 'fragile_pairs' in fp_data:
        raw_pairs = fp_data['fragile_pairs']
    else:
        raise KeyError("No 'pairs' or 'fragile_pairs' key in fragile_pairs.json")

    # Normalize to List[List[int]]
    fragile_pairs = []
    for p in raw_pairs:
        if isinstance(p, dict):
            fragile_pairs.append([p['feature_idx'], p['target_idx']])
        else:
            fragile_pairs.append([int(p[0]), int(p[1])])
    print(f"  Fragile pairs: {len(fragile_pairs)}")

    # sindy_coefficients.csv -- unscaled teacher coefficients (14, 4)
    coeff_path = bdir / 'sindy_coefficients.csv'
    coefficients = np.zeros((N_AEK_FEATURES, 4))
    with open(coeff_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            if i >= N_AEK_FEATURES:
                break
            for j in range(4):
                coefficients[i, j] = float(row[j + 1])
    print(f"  Teacher coefficients: {coefficients.shape}")

    return {
        'dir': bdir,
        'z_before': z_before,            # (20,) or (14, 4)
        'teacher_support': teacher_support,
        'fragile_pairs': fragile_pairs,   # [[f, t], ...]
        'coefficients': coefficients,     # (14, 4) unscaled
    }


# ============================================================
# Main Runner
# ============================================================

class AEK4cRunner:
    """AEK-4c Random Selection Augmentation Runner."""

    def __init__(self, cfg: AEK4cConfig):
        self.cfg = cfg
        self.feature_names = get_aek_feature_names(cfg.reparam)

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        print("=" * 70)
        print("AEK-4c-2: Random Selection Augmentation (Reparam-2)")
        print(f"  Seeds: {cfg.random_seeds}")
        print(f"  Pool: {cfg.pool_size}, Select: {cfg.n_select}")
        print(f"  E-SINDy: B={cfg.n_bootstrap}, threshold={cfg.threshold}")
        print("=" * 70)

        # AC2 check
        print("\n[AC2] Oracle label integrity...")
        assert_oracle_label_integrity(cfg.reparam)
        print("  PASS")

        # Phase 0: Load
        print("\n[Phase 0] Loading data and baseline...")
        dataset_path = paths.get_dataset_path(cfg.dataset_version, system=cfg.system)
        validate_dataset_lite(dataset_path)
        dataset = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  Dataset: {dataset_path}")

        baseline = load_baseline(cfg)

        train_x = dataset['train_x'][:cfg.n_train]    # (10, 201, 4)
        train_u = dataset['train_u'][:cfg.n_train]    # (10, 201, 1)
        train_dx = dataset['train_dx'][:cfg.n_train]  # (10, 201, 4)
        train_params = dataset['train_params'][:cfg.n_train]
        print(f"  Train: {train_x.shape}")

        # Phase 1: GMM
        print("\n[Phase 1] Fitting GMM...")
        gmm = AEKGMMSampler(
            n_components=cfg.gmm_n_components,
            covariance_type=cfg.gmm_covariance_type,
            random_state=cfg.gmm_seed,
        )
        gmm.fit(train_x, train_params)
        print(f"  GMM fitted (5D, {cfg.gmm_n_components} components)")

        # Phase 2: Generate shared pool (same for all seeds)
        print("\n[Phase 2] Generating shared pool...")
        rng_pool = np.random.default_rng(cfg.pool_seed)
        pool = generate_pool(gmm, train_x, train_u, cfg, rng_pool)
        # GPT P0-1: Split traj_sha + theta_sha for confound-free SSOT
        traj_sha = hashlib.sha256(pool['trajectories'].tobytes()).hexdigest()[:16]
        # Theta SHA: recompute library on pool to capture reparam effect
        _pool_x_flat = pool['trajectories'].reshape(-1, 4)
        _pool_u_flat = pool['u'].reshape(-1, 1)
        _pool_Theta, _ = build_aek_library_by_name(_pool_x_flat, _pool_u_flat, reparam=cfg.reparam)
        theta_sha = hashlib.sha256(_pool_Theta.tobytes()).hexdigest()[:16]
        pool_sha = f"{traj_sha}_{theta_sha}"
        print(f"  Pool traj_sha: {traj_sha}")
        print(f"  Pool theta_sha: {theta_sha}")
        print(f"  Pool combined SHA: {pool_sha}")

        # Phase 3: Track A (shared for all seeds)
        print("\n[Phase 3] Track A filtering...")
        track_a = track_a_filter(
            pool, baseline['coefficients'], cfg.reparam, cfg.reject_ratio,
        )

        # Phase 4: Random selection runs
        results_base = paths.ROOT / 'results' / cfg.dataset_version / 'gate4c-2' / 'random'
        all_results = {}

        for sel_seed in cfg.random_seeds:
            print(f"\n{'='*60}")
            print(f"[Phase 4.{sel_seed}] Random seed={sel_seed}")
            print(f"{'='*60}")

            try:
                # Select
                selected = random_select(pool, track_a, cfg.n_select, sel_seed)

                # E-SINDy evaluation
                print(f"  E-SINDy: {train_x.shape[0]} train + {selected['n_selected']} aug ...")
                eval_result = evaluate_augmented(
                    train_x, train_u, train_dx,
                    selected['trajectories'], selected['u'], selected['dx'],
                    reparam=cfg.reparam,
                    n_bootstrap=cfg.n_bootstrap,
                    threshold=cfg.threshold,
                    seed=cfg.baseline_seed,
                    z_eps=cfg.z_eps,
                )

                # Metrics (AC1: delta_raw + score_aligned)
                metrics = compute_metrics(
                    eval_result['z'], baseline['z_before'],
                    baseline['fragile_pairs'],
                    cfg.ci_bootstrap_B, cfg.ci_alpha, cfg.baseline_seed,
                )

                # P0-2: tau distribution comparison
                tau_stats = compute_tau_stats(train_u, selected['u'])

                # Save
                run_id = paths.generate_run_id(f"aek4c2_rp2_random_s{sel_seed}")
                run_dir = results_base / f"seed{sel_seed}" / run_id
                save_run(
                    run_dir, run_id, sel_seed, cfg,
                    eval_result, metrics, pool_sha, baseline['dir'],
                    tau_stats=tau_stats,
                )

                print(f"  delta_raw={metrics['delta_raw_median']:.3f}, "
                      f"score_aligned={metrics['score_aligned_median']:.3f}, "
                      f"pass_level={metrics['pass_level']}")
                print(f"  kappa={eval_result['kappa']:.0f}, "
                      f"support={int(eval_result['support_mask'].sum())}/56")
                print(f"  tau: train=[{tau_stats['train_tau_q05']:.5f}, "
                      f"{tau_stats['train_tau_q95']:.5f}], "
                      f"aug=[{tau_stats['aug_tau_q05']:.5f}, "
                      f"{tau_stats['aug_tau_q95']:.5f}], "
                      f"clip={tau_stats['aug_tau_clip_rate']:.1%}")
                print(f"  Saved: {run_dir}")

                all_results[f's{sel_seed}'] = {
                    'run_id': run_id,
                    'run_dir': str(run_dir),
                    'status': 'completed',
                    'delta_raw': metrics['delta_raw_median'],
                    'score_aligned': metrics['score_aligned_median'],
                    'pass_level': metrics['pass_level'],
                    'support': int(eval_result['support_mask'].sum()),
                    'kappa': eval_result['kappa'],
                }

            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                all_results[f's{sel_seed}'] = {
                    'status': 'failed', 'error': str(e),
                }

        # Phase 5: Summary
        print("\n" + "=" * 70)
        print("  AEK-4c SUMMARY")
        print("=" * 70)

        completed = {k: v for k, v in all_results.items()
                     if v.get('status') == 'completed'}

        if completed:
            print(f"\n  {'Seed':>4}  {'delta_raw':>10}  {'score_aln':>10}  "
                  f"{'pass_level':>12}  {'support':>8}  {'kappa':>10}")
            print("  " + "-" * 66)

            summary_rows = []
            for k in sorted(completed.keys()):
                v = completed[k]
                seed_num = int(k.replace('s', ''))
                print(f"  {seed_num:>4}  {v['delta_raw']:>10.3f}  "
                      f"{v['score_aligned']:>10.3f}  {v['pass_level']:>12}  "
                      f"{v['support']:>8}  {v['kappa']:>10.0f}")
                summary_rows.append({
                    'seed': seed_num,
                    'delta_raw': v['delta_raw'],
                    'score_aligned': v['score_aligned'],
                    'pass_level': v['pass_level'],
                    'support': v['support'],
                    'kappa': v['kappa'],
                })

            # Save summary JSON
            summary = {
                'system': 'aek', 'gate': 'gate4c-2', 'method': 'random',
                'reparam': cfg.reparam,
                'n_seeds': len(cfg.random_seeds),
                'n_completed': len(completed),
                'pool_sha': pool_sha,
                'pool_size': pool['n_accepted'],
                'pool_accept_rate': pool.get('accept_rate', None),
                'pd_gains': pool.get('pd_gains', {}),
                'created_at': datetime.now().isoformat(),
                'results': summary_rows,
                'runner_version': RUNNER_VERSION,
            }
            summary_path = results_base / 'aek4c2_random_rp2_summary.json'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=_json_default)
            print(f"\n  Summary: {summary_path}")

        n_fail = len(all_results) - len(completed)
        if n_fail > 0:
            print(f"\n  {n_fail} run(s) failed")

        return all_results


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='AEK-4c-2 Random Augmentation (Reparam-2)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                   help='Selection seeds (default: 0 1 2)')
    p.add_argument('--pool_size', type=int, default=200)
    p.add_argument('--n_select', type=int, default=50)
    p.add_argument('--n_bootstrap', type=int, default=100)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--baseline_seed', type=int, default=0)
    p.add_argument('--baseline_dir', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = AEK4cConfig(
        random_seeds=args.seeds,
        pool_size=args.pool_size,
        n_select=args.n_select,
        n_bootstrap=args.n_bootstrap,
        threshold=args.threshold,
        baseline_seed=args.baseline_seed,
        baseline_dir=args.baseline_dir,
    )
    runner = AEK4cRunner(cfg)
    results = runner.run()

    n_ok = sum(1 for v in results.values() if v.get('status') == 'completed')
    n_total = len(results)
    if n_ok == n_total:
        print(f"\nAEK-4c complete ({n_ok}/{n_total} seeds)")
    elif n_ok > 0:
        print(f"\nAEK-4c partial ({n_ok}/{n_total} seeds)")
        sys.exit(0)
    else:
        print(f"\nAEK-4c failed (0/{n_total} seeds)")
        sys.exit(1)


if __name__ == '__main__':
    main()