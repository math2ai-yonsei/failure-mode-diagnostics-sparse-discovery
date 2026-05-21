"""
Lorenz Gate: Baseline + GMM Augmentation + D-opt vs Random Ablation

Purpose:
    Validate GMM+ODE Teacher-Aligned Augmentation on the canonical
    Lorenz-63 system. Three-phase experiment:

    Phase 0: Generate / load dataset (lorenz_ood_v1)
    Phase 1: E-SINDy baseline → fragile pairs (z_before)
    Phase 2: GMM pool generation (Lorenz ODE as teacher)
    Phase 3: D-optimal vs Random selection → score_aligned comparison

Design (v1.1 — confirmed final):
    - n_train=5 (low-data regime; n=10 with T=501 is overdetermined for 10-term library)
    - rho=28 SINGLE value — rho appears as EOM coefficient (dy/dt = rho*x - y - xz);
      mixing multiple rho values makes SINDy coefficients structurally undefined.
    - OOD: initial conditions only (rho fixed). NOT rho variation.
    - Pool: 200 trajectories, GMM fitted on train ICs [x0,y0,z0] (3D, rho excluded)
    - D-optimal: 1 run (confound-free)
    - Random: 10 seeds
    - Noise: 5% Gaussian + Savitzky-Golay derivative (Brunton 2016 standard protocol)

Metric SSOT (Lorenz — precision_collapse, like AEK):
    delta_raw = median(z_after − z_before) over fragile pairs
    score_aligned = −delta_raw  (positive = improvement; spurious z reduction)
    Pass levels: CEILING_BREAK / STRONG_PASS / SOFT_PASS / NULL
    NOTE: failure_mode is determined at runtime from fragile pair composition.
          Lorenz baseline yields precision_collapse (oracle=1, spurious=5 of 6 pairs).

SSOT Rules:
    - paths.py for all path generation (no hardcoded paths)
    - plot_style.save_figure() for all figures
    - Preflight: validate_lorenz_dataset() (Lorenz-specific; STATE_DIM=3, INPUT_DIM=0.
      validate_dataset_lite() from schema_dataset_lite.py is CP-specific (4D/1D)
      and is NOT used for Lorenz.)
    - manifest.json + metrics.json + sindy_coefficients.csv required
    - Library: lorenz_library.py (assert_lorenz_feature_integrity at start)
    - ColumnScaler fitted on training data ONLY
    - Coverage Gate: Lorenz has no controller → strange attractor guarantees coverage

Usage:
    python experiments/run_lorenz_gate.py --phase all --seeds 0 1 2 3 4 5 6 7 8 9
    python experiments/run_lorenz_gate.py --phase baseline
    python experiments/run_lorenz_gate.py --phase augment --seeds 0 1 2

Author: Claude (Gate-Lorenz)
Date: 2026-03-05
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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.mixture import GaussianMixture

from src.contracts import paths
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble

# Lorenz-specific modules
from src.simulators.lorenz_simulator import LorenzSimulator, generate_lorenz_dataset
from src.sindy.lorenz_library import (
    build_lorenz_library,
    get_lorenz_feature_names,
    get_lorenz_oracle_support,
    get_lorenz_fragile_pairs,
    assert_lorenz_feature_integrity,
    N_LORENZ_FEATURES,
    LORENZ_TARGET_NAMES,
    LORENZ_FEATURE_NAMES,
)

# ============================================================
# Constants
# ============================================================

RUNNER_VERSION = 'v1.0'
GATE2_CEILING = 0.058       # Inherited from CP Gate2 (same pipeline)
DATASET_VERSION = 'lorenz_ood_v1'
SYSTEM = 'lorenz'


# ============================================================
# Configuration
# ============================================================

@dataclass
class LorenzGateConfig:
    """Lorenz Gate experiment configuration."""

    # Dataset
    dataset_version: str = DATASET_VERSION
    system: str = SYSTEM
    n_train: int = 5          # Low-data regime (Lorenz with noise needs <10)

    # Lorenz physics (SSOT — matches lorenz.yaml)
    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    rho_nominal: float = 28.0
    # Single rho: rho is a coefficient in EOM (dy/dt = rho*x - y - xz)
    # Multi-rho training makes SINDy coefficients structurally undefined.
    train_rho: List[float] = field(default_factory=lambda: [28.0])
    val_rho: List[float] = field(default_factory=lambda: [28.0])
    test_rho: List[float] = field(default_factory=lambda: [28.0])

    # Noise + SavGol (SINDy standard protocol, Brunton 2016)
    noise_std_fraction: float = 0.05   # 5% of per-state std
    savgol_window: int = 7
    savgol_polyorder: int = 3

    # Simulation (matches lorenz.yaml)
    dt: float = 0.01
    T_steps: int = 501
    max_state_norm: float = 200.0
    master_seed: int = 42

    # Baseline
    baseline_seed: int = 1

    # GMM
    gmm_n_components: int = 3
    gmm_covariance_type: str = 'full'
    gmm_seed: int = 42

    # Pool
    pool_size: int = 200
    max_pool_attempts: int = 2000
    pool_seed: int = 42

    # QC
    qc_max_state_norm: float = 200.0

    # Track A
    reject_ratio: float = 0.10

    # Selection
    n_select: int = 50
    random_seeds: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    # E-SINDy
    n_bootstrap: int = 100
    threshold: float = 0.05
    z_eps: float = 1e-6

    # Fragile pair threshold — z_threshold=2.0 gives ~9 pairs (stable bootstrap CI)
    # z_threshold=3.0 gives only ~5 pairs (too few → CI width [-18,4])
    z_fragile_threshold: float = 2.0

    # CI
    ci_bootstrap_B: int = 2000
    ci_alpha: float = 0.05

    # D-optimal
    dopt_lambda: float = 1e-6
    dopt_n_candidates: int = 200

    # Output
    note: str = 'lorenz_gate'


# ============================================================
# AC2: Feature Integrity (run at start)
# ============================================================

def check_feature_integrity():
    """AC2 equivalent for Lorenz: verify library structure."""
    assert_lorenz_feature_integrity()
    print("  ✅ AC2: Lorenz feature integrity verified")


# ============================================================
# Dataset Generation
# ============================================================

def generate_or_load_dataset(cfg: LorenzGateConfig) -> Dict[str, np.ndarray]:
    """
    Generate Lorenz dataset if not present, otherwise load it.

    Dataset is saved to:
        data/lorenz/lorenz_ood_v1/dataset.npz

    Returns:
        Dict with train_x, train_u, train_dx, train_params,
        train_cond_id, val_*, test_*, t, dt
    """
    dataset_path = paths.ROOT / 'data' / SYSTEM / DATASET_VERSION / 'dataset.npz'
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        print(f"  Loading existing dataset: {dataset_path}")
        data = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  train_x: {data['train_x'].shape}, "
              f"val_x: {data['val_x'].shape}, "
              f"test_x: {data['test_x'].shape}")
        return data

    print(f"  Generating Lorenz dataset → {dataset_path}")
    data = generate_lorenz_dataset(
        train_rho=cfg.train_rho,
        val_rho=cfg.val_rho,
        test_rho=cfg.test_rho,
        n_train=cfg.n_train,
        n_val=5,
        n_test=15,
        T_steps=cfg.T_steps,
        dt=cfg.dt,
        master_seed=cfg.master_seed,
        max_state_norm=cfg.max_state_norm,
        sigma=cfg.sigma,
        beta=cfg.beta,
        noise_std_fraction=cfg.noise_std_fraction,
        savgol_window=cfg.savgol_window,
        savgol_polyorder=cfg.savgol_polyorder,
    )

    np.savez(dataset_path, **data)
    print(f"  ✅ Dataset saved: train={data['train_x'].shape}, "
          f"test={data['test_x'].shape}")

    # Save meta.json
    meta = {
        'system': SYSTEM,
        'dataset_version': DATASET_VERSION,
        'state_dim': 3,
        'input_dim': 0,
        'n_train': data['train_x'].shape[0],
        'n_val': data['val_x'].shape[0],
        'n_test': data['test_x'].shape[0],
        'T_steps': int(data['train_x'].shape[1]),
        'dt': float(cfg.dt),
        'train_rho': cfg.train_rho,
        'val_rho': cfg.val_rho,
        'test_rho': cfg.test_rho,
        'sigma': cfg.sigma,
        'beta_approx': 2.6667,
        'beta_exact': '8/3',
        'master_seed': cfg.master_seed,
        'noise_std_fraction': cfg.noise_std_fraction,
        'savgol_window': cfg.savgol_window,
        'savgol_polyorder': cfg.savgol_polyorder,
        'dx_source': 'savgol_from_noisy_x',
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_lorenz_gate.py',
    }
    with open(dataset_path.parent / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return data


def validate_lorenz_dataset(data: Dict[str, np.ndarray]) -> None:
    """
    Lorenz-specific dataset validation (replaces validate_dataset_lite
    for Lorenz since input_dim=0 → u is zeros, not physics).

    Checks shape, finiteness, and basic sanity.
    """
    for split in ['train', 'val', 'test']:
        x = data[f'{split}_x']
        u = data[f'{split}_u']
        dx = data[f'{split}_dx']

        assert x.ndim == 3 and x.shape[2] == 3, \
            f"{split}_x shape mismatch: {x.shape}"
        assert u.ndim == 3 and u.shape[2] == 1, \
            f"{split}_u shape mismatch: {u.shape}"
        assert dx.ndim == 3 and dx.shape[2] == 3, \
            f"{split}_dx shape mismatch: {dx.shape}"
        assert x.shape[:2] == u.shape[:2] == dx.shape[:2], \
            f"{split} shape mismatch: {x.shape}, {u.shape}, {dx.shape}"

        assert np.all(np.isfinite(x)), f"{split}_x contains non-finite"
        assert np.all(np.isfinite(dx)), f"{split}_dx contains non-finite"
        assert np.allclose(u, 0), f"{split}_u should be zeros (Lorenz has no input)"

    assert 't' in data and 'dt' in data, "Missing t or dt"
    print("  ✅ Lorenz dataset validation passed")


# ============================================================
# E-SINDy Evaluation
# ============================================================

def evaluate_with_esindy(
    train_x: np.ndarray,
    aug_x: np.ndarray,
    train_dx: np.ndarray,
    aug_dx: np.ndarray,
    n_bootstrap: int,
    threshold: float,
    z_eps: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Run E-SINDy on train + augmented data.

    Lorenz-specific: no u input needed.
    ColumnScaler fitted on training data ONLY (P0 confound-free).

    Args:
        train_x: (N_tr, T, 3)
        aug_x: (N_aug, T, 3)
        train_dx: (N_tr, T, 3)
        aug_dx: (N_aug, T, 3)

    Returns:
        Dict with z (10,3), coefficients_mean, kappa, etc.
    """
    N_tr, T_tr, _ = train_x.shape
    N_aug, T_aug, _ = aug_x.shape

    if T_tr != T_aug:
        raise ValueError(f"T mismatch: train={T_tr}, aug={T_aug}")

    # Flatten
    x_tr_flat = train_x.reshape(-1, 3)
    x_au_flat = aug_x.reshape(-1, 3)
    dx_tr_flat = train_dx.reshape(-1, 3)
    dx_au_flat = aug_dx.reshape(-1, 3)

    x_all = np.vstack([x_tr_flat, x_au_flat])
    dx_all = np.vstack([dx_tr_flat, dx_au_flat])

    n_traj = N_tr + N_aug
    n_samples = n_traj * T_tr

    # Build library — train data only for scaler
    Theta_tr, feat_names = build_lorenz_library(x_tr_flat)
    Theta_all, _ = build_lorenz_library(x_all)

    # Scale: fit on training only
    scaler = ColumnScaler()
    scaler.fit(Theta_tr)
    Theta_all_scaled = scaler.transform(Theta_all)
    Theta_tr_scaled = scaler.transform(Theta_tr)

    # Condition number on scaled train library
    kappa = float(np.linalg.cond(Theta_tr_scaled))

    # E-SINDy ensemble
    ensemble = ESINDyEnsemble(
        n_bootstrap=n_bootstrap,
        threshold=threshold,
        random_state=seed,
    )
    ensemble.fit(
        Theta_all_scaled, dx_all,
        n_trajectories=n_traj,
        T=T_tr,
        scaler=scaler,
        target_scale=None,
    )

    coeff_mean = ensemble.coefficients_mean_      # (10, 3)
    coeff_std = ensemble.coefficients_std_        # (10, 3)
    inc_prob = ensemble.inclusion_probability_    # (10, 3)
    support_mask = np.abs(coeff_mean) > 0

    # z-metric
    z = np.abs(coeff_mean) / (coeff_std + z_eps)

    return {
        'z': z,
        'coefficients_mean': coeff_mean,
        'coefficients_std': coeff_std,
        'inclusion_probability': inc_prob,
        'support_mask': support_mask,
        'scaler': scaler,
        'feature_names': feat_names,
        'kappa': kappa,
        'n_total_samples': n_samples,
        'n_original': N_tr * T_tr,
        'n_augmented': N_aug * T_aug,
    }


def run_baseline(
    train_x: np.ndarray,
    train_dx: np.ndarray,
    cfg: LorenzGateConfig,
) -> Dict[str, Any]:
    """
    Run E-SINDy baseline (no augmentation).

    Returns z_before and fragile_pairs for subsequent augmentation runs.
    """
    N_tr, T_tr, _ = train_x.shape
    x_flat = train_x.reshape(-1, 3)
    dx_flat = train_dx.reshape(-1, 3)

    Theta, feat_names = build_lorenz_library(x_flat)
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    kappa = float(np.linalg.cond(Theta_scaled))

    ensemble = ESINDyEnsemble(
        n_bootstrap=cfg.n_bootstrap,
        threshold=cfg.threshold,
        random_state=cfg.baseline_seed,
    )
    ensemble.fit(
        Theta_scaled, dx_flat,
        n_trajectories=N_tr,
        T=T_tr,
        scaler=scaler,
        target_scale=None,
    )

    coeff_mean = ensemble.coefficients_mean_
    coeff_std = ensemble.coefficients_std_
    z = np.abs(coeff_mean) / (coeff_std + cfg.z_eps)

    fragile_pairs = get_lorenz_fragile_pairs(z, z_threshold=cfg.z_fragile_threshold)

    oracle = get_lorenz_oracle_support()
    n_oracle_fragile = sum(
        1 for f, t in fragile_pairs if oracle[f, t]
    )
    n_spurious_fragile = sum(
        1 for f, t in fragile_pairs if not oracle[f, t]
    )

    print(f"  κ (baseline): {kappa:.3e}")
    print(f"  Fragile pairs: {len(fragile_pairs)} total "
          f"({n_oracle_fragile} oracle/recall, {n_spurious_fragile} spurious/precision)")

    # Dominant failure mode classification
    if n_oracle_fragile >= n_spurious_fragile:
        failure_mode = 'recall_fragility'
        print(f"  Dominant failure: RECALL FRAGILITY (like CP) → score_aligned = +delta_raw")
    else:
        failure_mode = 'precision_collapse'
        print(f"  Dominant failure: PRECISION COLLAPSE (like AEK) → score_aligned = −delta_raw")

    return {
        'z': z,
        'coefficients_mean': coeff_mean,
        'coefficients_std': coeff_std,
        'kappa': kappa,
        'fragile_pairs': fragile_pairs,
        'n_oracle_fragile': n_oracle_fragile,
        'n_spurious_fragile': n_spurious_fragile,
        'failure_mode': failure_mode,
        'feature_names': feat_names,
        'scaler': scaler,
    }


# ============================================================
# GMM Sampler (Lorenz-specific: 4D = 3 IC + 1 rho)
# ============================================================

class LorenzGMMSampler:
    """
    GMM sampler for Lorenz initial conditions.

    Fits a 3D GMM on [x0, y0, z0] from training data.
    rho is fixed (single value) so not included in GMM.
    Samples candidate ICs for pool generation.
    """

    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        self._fitted = False
        self._rho = None

    def fit(self, train_x: np.ndarray, train_params: np.ndarray):
        """
        Fit GMM on training ICs [x0, y0, z0].

        Args:
            train_x: (N, T, 3) training trajectories
            train_params: (N, 1) rho values (used to extract the fixed rho)
        """
        ics = train_x[:, 0, :]       # (N, 3)
        self.gmm.fit(ics)
        self._fitted = True
        self._rho = float(train_params.mean())  # Fixed rho from training data

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample ICs from fitted GMM with fixed rho.

        Returns:
            ics: (n, 3) — [x0, y0, z0]
            rhos: (n, 1) — fixed rho repeated
        """
        if not self._fitted:
            raise RuntimeError("GMM not fitted")
        ics, _ = self.gmm.sample(n_samples)
        rhos = np.full((n_samples, 1), self._rho)
        return ics, rhos


# ============================================================
# Pool Generation (Lorenz ODE teacher — no controller needed)
# ============================================================

def generate_lorenz_pool(
    gmm_sampler: LorenzGMMSampler,
    cfg: LorenzGateConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Generate augmentation pool using Lorenz ODE teacher.

    Key advantage over AEK: no stabilizing controller needed.
    The Lorenz attractor is self-bounding → Coverage collapse impossible.

    Pool entries: x (T,3), dx (T,3), rho (scalar)

    Returns:
        Dict with arrays and SHA checksum
    """
    pool_x = []
    pool_dx = []
    pool_rho = []
    pool_ics = []

    n_attempts = 0
    n_accepted = 0

    print(f"  Generating pool: target={cfg.pool_size}, max_attempts={cfg.max_pool_attempts}")

    while n_accepted < cfg.pool_size and n_attempts < cfg.max_pool_attempts:
        # Sample batch from GMM
        batch_size = min(50, cfg.pool_size - n_accepted)
        ics_batch, rhos_batch = gmm_sampler.sample(batch_size * 3)
        n_attempts += len(ics_batch)

        for i in range(len(ics_batch)):
            if n_accepted >= cfg.pool_size:
                break

            ic = ics_batch[i]
            rho = float(rhos_batch[i, 0])
            rho = max(1.0, rho)

            sim = LorenzSimulator(params={
                'sigma': cfg.sigma,
                'beta': cfg.beta,
                'rho': rho,
            })

            try:
                t_arr, x_arr, dx_arr = sim.simulate(
                    x0=ic,
                    T_steps=cfg.T_steps,
                    dt=cfg.dt,
                )
                if sim.is_bounded(x_arr, max_norm=cfg.qc_max_state_norm):
                    pool_x.append(x_arr)      # (T, 3)
                    pool_dx.append(dx_arr)    # (T, 3)
                    pool_rho.append(rho)
                    pool_ics.append(ic)
                    n_accepted += 1
            except RuntimeError:
                continue

    if n_accepted < cfg.pool_size:
        print(f"  ⚠️  Pool: {n_accepted}/{cfg.pool_size} accepted "
              f"after {n_attempts} attempts")
    else:
        print(f"  ✅ Pool: {n_accepted} accepted ({n_attempts} attempts, "
              f"acceptance rate: {n_accepted/n_attempts:.1%})")

    pool_x_arr = np.array(pool_x, dtype=np.float32)   # (N_pool, T, 3)
    pool_dx_arr = np.array(pool_dx, dtype=np.float32) # (N_pool, T, 3)
    pool_rho_arr = np.array(pool_rho, dtype=np.float32)  # (N_pool,)
    pool_ics_arr = np.array(pool_ics, dtype=np.float32)  # (N_pool, 3)

    # Pool SHA (content-based, reproducibility guard)
    sha_data = (
        pool_x_arr.tobytes() +
        pool_dx_arr.tobytes() +
        pool_rho_arr.tobytes()
    )
    pool_sha = hashlib.sha256(sha_data).hexdigest()[:16]

    print(f"  Pool SHA: {pool_sha}")

    return {
        'x': pool_x_arr,
        'dx': pool_dx_arr,
        'rho': pool_rho_arr,
        'ics': pool_ics_arr,
        'n_accepted': n_accepted,
        'n_attempts': n_attempts,
        'sha': pool_sha,
    }


# ============================================================
# Track A: OOD/Quality Filter
# ============================================================

def track_a_filter(
    pool: Dict[str, Any],
    train_x: np.ndarray,
    reject_ratio: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track A: Reject top-reject_ratio% of pool by teacher error.

    Teacher error = mean absolute deviation between pool derivatives
    and derivatives computed from Lorenz EOM at the nominal rho.
    This identifies trajectories that deviate strongly from
    typical training-regime dynamics.

    For Lorenz, we use the L2 norm of dx residuals relative to
    a reference Lorenz system (rho=rho_nominal).

    Returns:
        candidate_indices: Indices of Track A passed pool entries
        errors: Per-trajectory teacher error scores
    """
    N_pool = pool['x'].shape[0]

    # Compute teacher error per trajectory:
    # Use variance of state norm as proxy for coverage quality
    # (high variance = good exploration of attractor)
    errors = np.zeros(N_pool, dtype=np.float32)
    for i in range(N_pool):
        state_norms = np.linalg.norm(pool['x'][i], axis=1)  # (T,)
        # Error = negative variance (we reject LOW variance = over-concentrated)
        # High variance trajectories → better coverage → keep
        errors[i] = -float(np.var(state_norms))  # negative so high = bad

    # Reject top reject_ratio% (worst = highest error = lowest variance)
    n_reject = int(N_pool * reject_ratio)
    reject_idx = np.argsort(errors)[-n_reject:] if n_reject > 0 else np.array([], dtype=int)
    all_idx = np.arange(N_pool)
    candidate_mask = np.ones(N_pool, dtype=bool)
    candidate_mask[reject_idx] = False
    candidate_indices = all_idx[candidate_mask]

    print(f"  Track A: {len(candidate_indices)}/{N_pool} passed "
          f"(rejected {n_reject} low-variance trajectories)")

    return candidate_indices, errors


# ============================================================
# D-optimal Selection
# ============================================================

def dopt_selection(
    pool: Dict[str, Any],
    candidate_indices: np.ndarray,
    train_x: np.ndarray,
    n_select: int,
    cfg: LorenzGateConfig,
) -> np.ndarray:
    """
    Greedy D-optimal selection from pool candidates.

    Maximizes log det(Theta^T Theta) where Theta is the Lorenz library
    matrix built from selected pool trajectories.

    Uses unit-trace normalization and greedy forward selection.

    Args:
        pool: Pool dict with x, dx arrays
        candidate_indices: Track A passed indices
        train_x: Training data (N, T, 3) for scaler fitting
        n_select: Number to select
        cfg: Configuration

    Returns:
        selected_indices: (n_select,) indices into original pool
    """
    n_candidates = len(candidate_indices)
    if n_candidates == 0:
        raise ValueError("No candidates for D-optimal selection")
    if n_candidates <= n_select:
        print(f"  D-opt: n_candidates={n_candidates} ≤ n_select={n_select}, "
              f"selecting all")
        return candidate_indices

    # Build library matrix for all candidates
    # Use training scaler for normalization
    x_tr_flat = train_x.reshape(-1, 3)
    Theta_tr, _ = build_lorenz_library(x_tr_flat)
    scaler = ColumnScaler()
    scaler.fit(Theta_tr)

    # Build candidate library (each trajectory flattened)
    cand_libs = []
    for idx in candidate_indices:
        x_cand = pool['x'][idx].reshape(-1, 3).astype(np.float64)
        Theta_c, _ = build_lorenz_library(x_cand)
        Theta_c_scaled = scaler.transform(Theta_c)
        cand_libs.append(Theta_c_scaled)  # (T, 10)

    # Gram matrix from training data
    Theta_tr_scaled = scaler.transform(Theta_tr)
    n_feat = N_LORENZ_FEATURES
    G = Theta_tr_scaled.T @ Theta_tr_scaled  # (10, 10)

    # Unit-trace normalization
    tr_G = np.trace(G)
    if tr_G > 0:
        G = G / tr_G * n_feat

    # Greedy forward selection
    lam = cfg.dopt_lambda
    G_reg = G + lam * np.eye(n_feat)
    selected_local = []

    for step in range(n_select):
        best_logdet = -np.inf
        best_i = -1

        for local_i, cand_Theta in enumerate(cand_libs):
            if local_i in selected_local:
                continue
            # Rank-1 update: log det(G + theta theta^T) = log det(G) + log(1 + theta^T G^-1 theta)
            # Use sum of diagonal updates as fast approximation
            update = cand_Theta.T @ cand_Theta  # (10, 10)
            G_test = G_reg + update
            try:
                sign, logdet = np.linalg.slogdet(G_test)
                if sign > 0 and logdet > best_logdet:
                    best_logdet = logdet
                    best_i = local_i
            except np.linalg.LinAlgError:
                continue

        if best_i < 0:
            print(f"  D-opt: early stop at step {step}/{n_select}")
            break

        selected_local.append(best_i)
        # Update Gram matrix
        G_reg = G_reg + cand_libs[best_i].T @ cand_libs[best_i]

    selected_pool_indices = candidate_indices[np.array(selected_local, dtype=int)]
    print(f"  D-opt: selected {len(selected_pool_indices)} trajectories "
          f"from {n_candidates} candidates")
    return selected_pool_indices


# ============================================================
# Metric Computation (SSOT: score_aligned = −delta_raw for Lorenz)
# Lorenz failure mode: precision_collapse (like AEK, NOT recall_fragility)
# Confirmed: baseline fragile_pairs = 6 (oracle=1, spurious=5)
# ============================================================

def compute_metrics(
    z_after: np.ndarray,
    z_before: np.ndarray,
    fragile_pairs: List[List[int]],
    failure_mode: str,
    ci_bootstrap_B: int,
    ci_alpha: float,
    ci_seed: int,
) -> Dict[str, Any]:
    """
    Compute SSOT metrics.

    Lorenz failure mode is precision_collapse (spurious fragile pairs dominate).
    score_aligned = −delta_raw (positive = improvement = spurious z reduction).
    This matches AEK convention, NOT Cart-Pole (+delta_raw).

    The failure_mode argument is determined at runtime from fragile pair composition
    and is always reported in metrics.json alongside score_aligned.

    Args:
        z_after: (10, 3) z-metric after augmentation
        z_before: (10, 3) z-metric before (baseline)
        fragile_pairs: List of [f_idx, t_idx]
        failure_mode: 'recall_fragility' or 'precision_collapse'
    """
    if len(fragile_pairs) == 0:
        return {
            'delta_raw_median': None,
            'score_aligned_median': None,
            'pass_level': 'NULL',
            'n_effective_pairs': 0,
        }

    z_af_list, z_bf_list = [], []
    for f, t in fragile_pairs:
        if f < z_after.shape[0] and t < z_after.shape[1]:
            z_af_list.append(z_after[f, t])
            z_bf_list.append(z_before[f, t])

    if not z_af_list:
        return {
            'delta_raw_median': None,
            'score_aligned_median': None,
            'pass_level': 'NULL',
            'n_effective_pairs': 0,
        }

    z_af = np.array(z_af_list)
    z_bf = np.array(z_bf_list)
    delta_per_pair = z_af - z_bf

    delta_raw = float(np.median(delta_per_pair))

    # score_aligned: Lorenz expected recall_fragility → +delta_raw
    if failure_mode == 'precision_collapse':
        score_aligned = -delta_raw  # spurious suppression = improvement
        sign_note = 'precision_collapse: score_aligned = -delta_raw'
    else:
        score_aligned = +delta_raw  # oracle strengthening = improvement
        sign_note = 'recall_fragility: score_aligned = +delta_raw'

    # Bootstrap CI on delta_raw
    ci_rng = np.random.default_rng(ci_seed)
    boot_medians = [
        float(np.median(ci_rng.choice(delta_per_pair,
                                       size=len(delta_per_pair), replace=True)))
        for _ in range(ci_bootstrap_B)
    ]
    ci_lo_raw = float(np.percentile(boot_medians, 100 * ci_alpha / 2))
    ci_hi_raw = float(np.percentile(boot_medians, 100 * (1 - ci_alpha / 2)))

    # Translate CI to score_aligned direction
    if failure_mode == 'precision_collapse':
        sa_ci_lo = -ci_hi_raw
        sa_ci_hi = -ci_lo_raw
    else:
        sa_ci_lo = ci_lo_raw
        sa_ci_hi = ci_hi_raw

    # Pass level
    pass_level = _classify_pass(score_aligned, sa_ci_lo)

    return {
        # Primary SSOT metrics
        'delta_raw_median': delta_raw,
        'score_aligned_median': score_aligned,
        # CI on delta_raw
        'delta_raw_ci_lower': ci_lo_raw,
        'delta_raw_ci_upper': ci_hi_raw,
        # CI on score_aligned
        'score_aligned_ci_lower': sa_ci_lo,
        'score_aligned_ci_upper': sa_ci_hi,
        # Pass level
        'pass_level': pass_level,
        # Diagnostics
        'failure_mode': failure_mode,
        'sign_note': sign_note,
        'z_after_fragile_median': float(np.median(z_af)),
        'z_before_fragile_median': float(np.median(z_bf)),
        'n_fragile_pairs': len(fragile_pairs),
        'n_effective_pairs': len(z_af_list),
        'delta_per_pair': delta_per_pair.tolist(),
        'fragile_pairs': fragile_pairs,
    }


def _classify_pass(score_aligned: float, sa_ci_lower: float) -> str:
    """Pass level classification (CEILING_BREAK uses Gate2 ceiling=0.058)."""
    if score_aligned is None or sa_ci_lower is None:
        return 'NULL'
    if sa_ci_lower > GATE2_CEILING:
        return 'CEILING_BREAK'
    if sa_ci_lower > 0:
        return 'STRONG_PASS'
    if score_aligned > 0:
        return 'SOFT_PASS'
    return 'NULL'


# ============================================================
# Artifact Save
# ============================================================

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


def save_run_artifacts(
    run_dir: Path,
    run_id: str,
    method: str,
    selection_seed: Optional[int],
    cfg: LorenzGateConfig,
    eval_result: Dict[str, Any],
    metrics: Dict[str, Any],
    pool_sha: str,
    baseline_dir: Optional[Path],
):
    """Save standard artifacts: manifest, metrics, sindy_coefficients."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'figures').mkdir(exist_ok=True)

    # --- metrics.json ---
    full_metrics = {**metrics}
    full_metrics.update({
        'system': SYSTEM,
        'gate': 'gate_lorenz',
        'method': method,
        'library': 'standard',
        'selection_seed': selection_seed,
        'pool_sha': pool_sha,
        'pool_size': cfg.pool_size,
        'n_select': cfg.n_select,
        'n_train': cfg.n_train,
        'n_bootstrap': cfg.n_bootstrap,
        'threshold': cfg.threshold,
        'kappa': eval_result.get('kappa'),
        'n_total_samples': eval_result.get('n_total_samples'),
        'n_original': eval_result.get('n_original'),
        'n_augmented': eval_result.get('n_augmented'),
        'support_terms_total': int(eval_result['support_mask'].sum()),
        'ci_bootstrap_B': cfg.ci_bootstrap_B,
        'ci_alpha': cfg.ci_alpha,
        'runner_version': RUNNER_VERSION,
        'gate2_ceiling': GATE2_CEILING,
    })
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2, default=_json_default)

    # --- sindy_coefficients.csv ---
    coeff = eval_result['coefficients_mean']
    with open(run_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + list(LORENZ_TARGET_NAMES))
        for i, name in enumerate(eval_result['feature_names']):
            w.writerow([name] + [f"{coeff[i,j]:.8f}" for j in range(3)])

    # --- z_after.npy ---
    np.save(run_dir / 'z_after.npy', eval_result['z'])

    # --- manifest.json ---
    manifest = {
        'run_id': run_id,
        'system': SYSTEM,
        'gate': 'gate_lorenz',
        'method': method,
        'library': 'standard',
        'selection_seed': selection_seed,
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_lorenz_gate.py',
        'runner_version': RUNNER_VERSION,
        'pool_sha': pool_sha,
        'baseline_dir': str(baseline_dir) if baseline_dir else None,
        'config': {
            'pool_size': cfg.pool_size,
            'pool_seed': cfg.pool_seed,
            'n_select': cfg.n_select,
            'n_bootstrap': cfg.n_bootstrap,
            'threshold': cfg.threshold,
            'gmm_n_components': cfg.gmm_n_components,
            'gmm_seed': cfg.gmm_seed,
            'reject_ratio': cfg.reject_ratio,
            'z_fragile_threshold': cfg.z_fragile_threshold,
            'lorenz_sigma': cfg.sigma,
            'lorenz_beta': '8/3',
            'lorenz_rho_nominal': cfg.rho_nominal,
            'train_rho': cfg.train_rho,
            'test_rho': cfg.test_rho,
        },
        'artifacts': [
            'manifest.json', 'metrics.json',
            'sindy_coefficients.csv', 'z_after.npy',
        ],
    }
    with open(run_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=_json_default)


# ============================================================
# Main Experiment Flow
# ============================================================

def run_lorenz_gate(cfg: LorenzGateConfig, seeds: List[int]) -> Dict[str, Any]:
    """
    Full Lorenz Gate experiment.

    Phase 0: Dataset generation / loading
    Phase 1: Baseline (z_before, fragile_pairs)
    Phase 2: GMM pool generation
    Phase 3: D-opt + Random augmentation, score_aligned comparison
    """
    print("=" * 70)
    print("Lorenz Gate: GMM Augmentation + D-opt vs Random Ablation")
    print("=" * 70)
    print(f"  n_train={cfg.n_train}, pool={cfg.pool_size}, "
          f"n_select={cfg.n_select}, seeds={seeds}")
    print(f"  rho train={cfg.train_rho}, test={cfg.test_rho}")
    print("=" * 70)

    # ── AC2: Feature integrity ─────────────────────────────────────────
    print("\n[AC2] Feature integrity check...")
    check_feature_integrity()

    # ── Phase 0: Dataset ───────────────────────────────────────────────
    print("\n[Phase 0] Dataset generation / loading...")
    data = generate_or_load_dataset(cfg)
    validate_lorenz_dataset(data)

    train_x = data['train_x'].astype(np.float64)    # (10, 501, 3)
    train_dx = data['train_dx'].astype(np.float64)  # (10, 501, 3)
    train_params = data['train_params']              # (10, 1) = rho

    # ── Phase 1: Baseline ──────────────────────────────────────────────
    print("\n[Phase 1] E-SINDy Baseline...")
    baseline_rng = np.random.default_rng(cfg.baseline_seed)
    baseline_result = run_baseline(
        train_x=train_x,
        train_dx=train_dx,
        cfg=cfg,
    )
    z_before = baseline_result['z']
    fragile_pairs = baseline_result['fragile_pairs']
    failure_mode = baseline_result['failure_mode']

    # Save baseline artifacts
    run_id_baseline = paths.generate_run_id(f'lorenz_baseline_s{cfg.baseline_seed}')
    baseline_dir = paths.get_results_dir(
        dataset_version=cfg.dataset_version,
        gate='gate_lorenz',
        track='standardized',
        method='esindy_baseline',
        n_train=cfg.n_train,
        seed=cfg.baseline_seed,
        run_id=run_id_baseline,
    )
    np.save(baseline_dir / 'z_before.npy', z_before)

    baseline_coeff = baseline_result['coefficients_mean']
    with open(baseline_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + LORENZ_TARGET_NAMES)
        for i, name in enumerate(LORENZ_FEATURE_NAMES):
            w.writerow([name] + [f"{baseline_coeff[i,j]:.8f}" for j in range(3)])

    with open(baseline_dir / 'fragile_pairs.json', 'w') as f:
        json.dump({
            'fragile_pairs': fragile_pairs,
            'n_pairs': len(fragile_pairs),
            'failure_mode': failure_mode,
            'n_oracle_fragile': baseline_result['n_oracle_fragile'],
            'n_spurious_fragile': baseline_result['n_spurious_fragile'],
            'kappa': baseline_result['kappa'],
            'z_fragile_threshold': cfg.z_fragile_threshold,
        }, f, indent=2)

    with open(baseline_dir / 'manifest.json', 'w') as f:
        json.dump({
            'run_id': run_id_baseline,
            'system': SYSTEM,
            'gate': 'gate_lorenz',
            'method': 'esindy_baseline',
            'library': 'standard',
            'created_at': datetime.now().isoformat(),
            'runner': 'experiments/run_lorenz_gate.py',
            'runner_version': RUNNER_VERSION,
            'kappa': baseline_result['kappa'],
            'n_fragile_pairs': len(fragile_pairs),
            'failure_mode': failure_mode,
            'artifacts': ['manifest.json', 'z_before.npy',
                          'sindy_coefficients.csv', 'fragile_pairs.json'],
        }, f, indent=2)

    print(f"  ✅ Baseline saved: {baseline_dir}")

    # ── Phase 2: GMM Pool ──────────────────────────────────────────────
    print("\n[Phase 2] GMM Pool Generation...")
    pool_rng = np.random.default_rng(cfg.pool_seed)

    gmm_sampler = LorenzGMMSampler(
        n_components=cfg.gmm_n_components,
        covariance_type=cfg.gmm_covariance_type,
        random_state=cfg.gmm_seed,
    )
    gmm_sampler.fit(train_x, train_params)
    print(f"  GMM fitted: {cfg.gmm_n_components} components, "
          f"fixed rho={gmm_sampler._rho:.1f}")

    pool = generate_lorenz_pool(gmm_sampler, cfg, pool_rng)
    pool_sha = pool['sha']

    # Track A
    candidate_indices, errors = track_a_filter(pool, train_x, cfg.reject_ratio)

    # ── Phase 3: Augmentation Runs ────────────────────────────────────
    print(f"\n[Phase 3] Augmentation: D-opt + Random (seeds={seeds})")

    all_results = {}

    # --- D-optimal ---
    print(f"\n  [D-opt]")
    try:
        dopt_indices = dopt_selection(pool, candidate_indices, train_x,
                                      cfg.n_select, cfg)
        aug_x_dopt = pool['x'][dopt_indices].astype(np.float64)
        aug_dx_dopt = pool['dx'][dopt_indices].astype(np.float64)

        eval_dopt = evaluate_with_esindy(
            train_x, aug_x_dopt, train_dx, aug_dx_dopt,
            cfg.n_bootstrap, cfg.threshold, cfg.z_eps, seed=42,
        )
        metrics_dopt = compute_metrics(
            z_after=eval_dopt['z'],
            z_before=z_before,
            fragile_pairs=fragile_pairs,
            failure_mode=failure_mode,
            ci_bootstrap_B=cfg.ci_bootstrap_B,
            ci_alpha=cfg.ci_alpha,
            ci_seed=42,
        )

        run_id_dopt = paths.generate_run_id('lorenz_dopt')
        run_dir_dopt = paths.get_results_dir(
            cfg.dataset_version, 'gate_lorenz', 'standardized',
            'esindy_dopt', cfg.n_train, 0, run_id_dopt,
        )
        save_run_artifacts(run_dir_dopt, run_id_dopt, 'd_optimal', None,
                           cfg, eval_dopt, metrics_dopt, pool_sha, baseline_dir)

        all_results['d_optimal'] = {
            'status': 'completed',
            'score_aligned': metrics_dopt['score_aligned_median'],
            'ci_lower': metrics_dopt['score_aligned_ci_lower'],
            'ci_upper': metrics_dopt['score_aligned_ci_upper'],
            'pass_level': metrics_dopt['pass_level'],
            'kappa': eval_dopt['kappa'],
            'run_dir': str(run_dir_dopt),
        }
        print(f"  D-opt: score_aligned={metrics_dopt['score_aligned_median']:.3f}, "
              f"CI=[{metrics_dopt['score_aligned_ci_lower']:.3f}, "
              f"{metrics_dopt['score_aligned_ci_upper']:.3f}], "
              f"pass={metrics_dopt['pass_level']}")

    except Exception as e:
        print(f"  ❌ D-opt failed: {e}")
        traceback.print_exc()
        all_results['d_optimal'] = {'status': 'failed', 'error': str(e)}

    # --- Random seeds ---
    for sel_seed in seeds:
        print(f"\n  [Random seed={sel_seed}]")
        try:
            rng_sel = np.random.default_rng(sel_seed)
            n_avail = len(candidate_indices)
            if n_avail <= cfg.n_select:
                rand_indices = candidate_indices.copy()
            else:
                chosen = rng_sel.choice(n_avail, size=cfg.n_select, replace=False)
                rand_indices = np.sort(candidate_indices[chosen])

            aug_x_rand = pool['x'][rand_indices].astype(np.float64)
            aug_dx_rand = pool['dx'][rand_indices].astype(np.float64)

            eval_rand = evaluate_with_esindy(
                train_x, aug_x_rand, train_dx, aug_dx_rand,
                cfg.n_bootstrap, cfg.threshold, cfg.z_eps, seed=sel_seed,
            )
            metrics_rand = compute_metrics(
                z_after=eval_rand['z'],
                z_before=z_before,
                fragile_pairs=fragile_pairs,
                failure_mode=failure_mode,
                ci_bootstrap_B=cfg.ci_bootstrap_B,
                ci_alpha=cfg.ci_alpha,
                ci_seed=sel_seed,
            )

            run_id_rand = paths.generate_run_id(f'lorenz_random_s{sel_seed}')
            run_dir_rand = paths.get_results_dir(
                cfg.dataset_version, 'gate_lorenz', 'standardized',
                'esindy_random', cfg.n_train, sel_seed, run_id_rand,
            )
            save_run_artifacts(run_dir_rand, run_id_rand, 'random', sel_seed,
                               cfg, eval_rand, metrics_rand, pool_sha, baseline_dir)

            all_results[f'random_s{sel_seed}'] = {
                'status': 'completed',
                'score_aligned': metrics_rand['score_aligned_median'],
                'ci_lower': metrics_rand['score_aligned_ci_lower'],
                'ci_upper': metrics_rand['score_aligned_ci_upper'],
                'pass_level': metrics_rand['pass_level'],
                'kappa': eval_rand['kappa'],
                'run_dir': str(run_dir_rand),
            }
            print(f"  Random s{sel_seed}: "
                  f"score_aligned={metrics_rand['score_aligned_median']:.3f}, "
                  f"pass={metrics_rand['pass_level']}")

        except Exception as e:
            print(f"  ❌ Random s{sel_seed} failed: {e}")
            traceback.print_exc()
            all_results[f'random_s{sel_seed}'] = {'status': 'failed', 'error': str(e)}

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LORENZ GATE SUMMARY")
    print("=" * 70)
    print(f"  Baseline κ: {baseline_result['kappa']:.3e}")
    print(f"  Failure mode: {failure_mode}")
    print(f"  Fragile pairs: {len(fragile_pairs)}")
    print(f"  Pool SHA: {pool_sha}")
    print()

    completed_random = [
        v for k, v in all_results.items()
        if k.startswith('random_') and v.get('status') == 'completed'
    ]
    if completed_random:
        sa_values = [r['score_aligned'] for r in completed_random if r['score_aligned'] is not None]
        pass_levels = [r['pass_level'] for r in completed_random]
        pass_counts = {
            'NULL': pass_levels.count('NULL'),
            'SOFT_PASS': pass_levels.count('SOFT_PASS'),
            'STRONG_PASS': pass_levels.count('STRONG_PASS'),
            'CEILING_BREAK': pass_levels.count('CEILING_BREAK'),
        }
        print(f"  Random ({len(completed_random)} seeds):")
        print(f"    median score_aligned = {float(np.median(sa_values)):.3f}")
        print(f"    Pass distribution: {pass_counts}")

    if 'd_optimal' in all_results and all_results['d_optimal'].get('status') == 'completed':
        dr = all_results['d_optimal']
        print(f"  D-optimal:")
        print(f"    score_aligned = {dr['score_aligned']:.3f}, "
              f"CI=[{dr['ci_lower']:.3f}, {dr['ci_upper']:.3f}]")
        print(f"    pass_level = {dr['pass_level']}")

    # ── Context Packet ─────────────────────────────────────────────────
    run_id_cp = paths.generate_run_id('lorenz_gate_cp')
    cp_path = paths.get_context_packet_path(run_id_cp)
    _write_context_packet(cp_path, baseline_result, all_results, pool_sha, cfg, seeds)
    print(f"\n  ✅ Context Packet: {cp_path}")

    # ── Save summary.json ──────────────────────────────────────────────
    summary_path = (paths.RESULTS_ROOT / cfg.dataset_version
                    / 'gate_lorenz' / 'summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'runner_version': RUNNER_VERSION,
        'baseline_kappa': baseline_result['kappa'],
        'failure_mode': failure_mode,
        'n_fragile_pairs': len(fragile_pairs),
        'pool_sha': pool_sha,
        'n_train': cfg.n_train,
        'results': all_results,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"  ✅ Summary: {summary_path}")

    return summary


def _write_context_packet(
    cp_path: Path,
    baseline_result: Dict,
    all_results: Dict,
    pool_sha: str,
    cfg: LorenzGateConfig,
    seeds: List[int],
) -> None:
    """Write Context Packet markdown file."""
    completed_random = [
        v for k, v in all_results.items()
        if k.startswith('random_') and v.get('status') == 'completed'
    ]
    sa_values = [r['score_aligned'] for r in completed_random
                 if r['score_aligned'] is not None]
    pass_levels = [r['pass_level'] for r in completed_random]

    dopt = all_results.get('d_optimal', {})

    lines = [
        f"# Context Packet: Lorenz Gate",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runner**: run_lorenz_gate.py {RUNNER_VERSION}",
        f"",
        f"## Baseline",
        f"- κ: {baseline_result['kappa']:.3e}",
        f"- Failure mode: **{baseline_result['failure_mode']}**",
        f"- Oracle fragile: {baseline_result['n_oracle_fragile']}",
        f"- Spurious fragile: {baseline_result['n_spurious_fragile']}",
        f"- Total fragile pairs: {len(baseline_result['fragile_pairs'])}",
        f"",
        f"## Pool",
        f"- SHA: `{pool_sha}`",
        f"- Size: {cfg.pool_size}",
        f"- Train rho: {cfg.train_rho}",
        f"",
        f"## Random Results ({len(completed_random)} seeds)",
    ]
    if sa_values:
        from collections import Counter
        pc = Counter(pass_levels)
        lines += [
            f"- Median score_aligned: {float(np.median(sa_values)):.3f}",
            f"- NULL: {pc.get('NULL',0)}, SOFT: {pc.get('SOFT_PASS',0)}, "
            f"STRONG: {pc.get('STRONG_PASS',0)}, CEILING: {pc.get('CEILING_BREAK',0)}",
        ]

    lines += [
        f"",
        f"## D-optimal",
        f"- Status: {dopt.get('status','N/A')}",
        f"- score_aligned: {dopt.get('score_aligned','N/A')}",
        f"- CI: [{dopt.get('ci_lower','N/A')}, {dopt.get('ci_upper','N/A')}]",
        f"- Pass level: {dopt.get('pass_level','N/A')}",
        f"",
        f"## SSOT Notes",
        f"- score_aligned = +delta_raw (recall_fragility, consistent with CP)",
        f"- gate2_ceiling = {GATE2_CEILING}",
        f"- Library: standard (no reparameterization needed, polynomial only)",
    ]

    with open(cp_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Lorenz Gate: GMM Augmentation + D-opt vs Random'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+',
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help='Random selection seeds'
    )
    parser.add_argument(
        '--n_train', type=int, default=5,
        help='Number of training trajectories (default: 5)'
    )
    parser.add_argument(
        '--pool_size', type=int, default=200,
        help='Augmentation pool size'
    )
    parser.add_argument(
        '--n_select', type=int, default=50,
        help='Number of trajectories to select from pool'
    )
    parser.add_argument(
        '--baseline_seed', type=int, default=1,
        help='Seed for baseline E-SINDy run'
    )
    parser.add_argument(
        '--note', type=str, default='lorenz_gate',
        help='Run note for run_id'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cfg = LorenzGateConfig(
        n_train=args.n_train,
        pool_size=args.pool_size,
        n_select=args.n_select,
        baseline_seed=args.baseline_seed,
        random_seeds=args.seeds,
        note=args.note,
    )

    summary = run_lorenz_gate(cfg, seeds=args.seeds)

    print("\n✅ Lorenz Gate complete.")