"""
Silverbox Gate: Baseline + GMM Augmentation + D-opt vs Random Ablation

Purpose:
    Validate GMM+Duffing Teacher-Aligned Augmentation on the Silverbox
    real-world electronic benchmark. Three-phase experiment:

    Phase 0: Generate / load dataset (silverbox_v1_n3)
    Phase 1: E-SINDy baseline → fragile pairs (z_before)
    Phase 2: GMM pool generation (Duffing ODE as teacher, u=0)
    Phase 3: D-optimal vs Random selection → score_aligned comparison

Design:
    - Real measured data: SNLS80mV (multisine excitation, 610.35 Hz)
    - W=500 steps (0.82s per window), temporal split 70/30
    - n_train=3 (extreme low-data regime; n=10/5 trivially identified, κ≈3.8)
    - State: [x1=y, x2=dy/dt] via SavGol (window=11, order=3)
    - Library: 8-term Duffing {1, x1, x2, x1², x1·x2, x2², x1³, u}
    - ColumnScaler: MANDATORY (x1/x2 scale ratio ≈ 300×)
    - Pool: 200 trajectories, GMM fitted on train ICs [x1_0, x2_0] (2D)
    - Teacher: Duffing ODE with u=0 (free oscillation from GMM ICs)
      NOTE: u=0 means pool has no input-direction augmentation
    - D-optimal: 1 run (confound-free)
    - Random: 10 seeds

Metric SSOT (Silverbox — failure_mode determined at runtime):
    delta_raw = median(z_after − z_before) over fragile pairs
    score_aligned = +delta_raw if recall_fragility
                  = −delta_raw if precision_collapse OR mixed
    Pass levels: CEILING_BREAK / STRONG_PASS / SOFT_PASS / NULL
    NOTE: Runtime baseline (n_train=3) confirms failure_mode=mixed
          (x1² spurious z=11.7 dominant; x1³ recall z=0.02).
          score_aligned = −delta_raw (precision-collapse dominant).

Teacher Parameters SSOT (exploration-fixed defaults — NOT re-fitted at runtime):
    k=-93282.91, k3=-44580.43, c=-2.9527, b=3100.43
    See silverbox_simulator.py module docstring for rationale.

SSOT Rules:
    - paths.py for all path generation (no hardcoded paths)
    - plot_style.save_figure() for all figures
    - validate_silverbox_dataset() preflight guard (state_dim=2, input_dim=1)
    - manifest.json + metrics.json + sindy_coefficients.csv required per run
    - baseline_meta.json required at baseline phase completion
    - Library: silverbox_library.py (assert_silverbox_feature_integrity at start)
    - ColumnScaler fitted on training data ONLY
    - build_silverbox_library(x, u) — always pass both x AND u

Usage:
    python experiments/run_silverbox_gate.py --phase all --seeds 0 1 2 3 4 5 6 7 8 9
    python experiments/run_silverbox_gate.py --phase baseline
    python experiments/run_silverbox_gate.py --phase augment --seeds 0 1 2

Author: Claude (Gate-Silverbox)
Date: 2026-03-10
Runner version: v1.3 (pool state+dx → SavGol for §3.3 consistency; previously analytic)
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
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture

from src.contracts import paths
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble

# Silverbox-specific modules
from src.simulators.silverbox_simulator import (
    SilverboxDuffingTeacher,
    generate_silverbox_dataset,
    validate_silverbox_dataset,
    DUFFING_K_DEFAULT, DUFFING_K3_DEFAULT, DUFFING_C_DEFAULT, DUFFING_B_DEFAULT,
)
from src.sindy.silverbox_library import (
    build_silverbox_library,
    get_silverbox_feature_names,
    get_silverbox_oracle_support,
    get_silverbox_fragile_pairs,
    assert_silverbox_feature_integrity,
    compute_silverbox_kappa,
    N_SILVERBOX_FEATURES,
    SILVERBOX_TARGET_NAMES,
    SILVERBOX_FEATURE_NAMES,
)

# ============================================================
# Constants
# ============================================================

RUNNER_VERSION  = 'v1.3'  # v1.3: pool state+dx → SavGol (논문 §3.3 일관성)
GATE2_CEILING   = 0.058       # Inherited from CP Gate2 (same pipeline)
DATASET_VERSION = 'silverbox_v1_n3'   # n_train=3 low-data regime (n=10 identified trivially)
SYSTEM          = 'silverbox'


# ============================================================
# Configuration
# ============================================================

@dataclass
class SilverboxGateConfig:
    """Silverbox Gate experiment configuration."""

    # Dataset
    dataset_version: str = DATASET_VERSION
    system: str = SYSTEM
    n_train: int = 3     # Low-data regime: n=10/5 identified trivially (κ=3.23),
                         # n=3 yields 3 fragile pairs (mixed: x1³ recall + x1²/const spurious)
    n_val: int = 3
    n_test: int = 20

    # Window / SavGol
    W: int = 500                # Window length (time steps); T ≈ 0.82 s
    savgol_window: int = 11     # SavGol filter window (must be odd)
    savgol_polyorder: int = 3   # SavGol polynomial order
    train_fraction: float = 0.70

    # Dataset generation
    master_seed: int = 42

    # Baseline
    baseline_seed: int = 1

    # GMM
    gmm_n_components: int = 2    # n_train=3 ICs → keep components ≤ n_train
    gmm_covariance_type: str = 'full'
    gmm_seed: int = 42

    # Pool
    pool_size: int = 200
    max_pool_attempts: int = 2000
    pool_seed: int = 42
    max_state_norm: float = 300.0  # ≈ 5× max observed ||[x1,x2]||

    # Selection
    n_select: int = 50
    random_seeds: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    # E-SINDy
    n_bootstrap: int = 100
    threshold: float = 0.05
    z_eps: float = 1e-6

    # Fragile pair threshold
    z_fragile_threshold: float = 2.0

    # CI bootstrap
    ci_bootstrap_B: int = 2000
    ci_alpha: float = 0.05

    # D-optimal
    dopt_lambda: float = 1e-6
    dopt_n_candidates: int = 200

    # Output
    note: str = 'silverbox_gate'


# ============================================================
# AC2: Feature Integrity
# ============================================================

def check_feature_integrity():
    """AC2 equivalent for Silverbox: verify library structure."""
    assert_silverbox_feature_integrity()
    print("  ✅ AC2: Silverbox feature integrity verified")


# ============================================================
# Dataset Generation / Loading
# ============================================================

def generate_or_load_dataset(cfg: SilverboxGateConfig) -> Dict[str, np.ndarray]:
    """
    Generate Silverbox dataset if not present, otherwise load it.

    Dataset saved to: data/silverbox/silverbox_v1_n3/dataset.npz

    NOTE: Duffing teacher parameters are NOT stored in the dataset.
    Teacher always uses exploration-fixed DUFFING_*_DEFAULT constants.
    See module-level docstring for rationale.

    Returns:
        Dict with train_x, train_u, train_dx, train_params,
        train_cond_id, val_*, test_*, t, dt
    """
    dataset_path = (paths.ROOT / 'data' / SYSTEM /
                    DATASET_VERSION / 'dataset.npz')
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        print(f"  Loading existing dataset: {dataset_path}")
        data = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  train_x: {data['train_x'].shape}, "
              f"val_x: {data['val_x'].shape}, "
              f"test_x: {data['test_x'].shape}")
        return data

    print(f"  Generating Silverbox dataset → {dataset_path}")
    print("  (Downloading Silverbox data if not cached — may take 1-2 min)")
    data = generate_silverbox_dataset(
        n_train=cfg.n_train,
        n_val=cfg.n_val,
        n_test=cfg.n_test,
        W=cfg.W,
        savgol_window=cfg.savgol_window,
        savgol_polyorder=cfg.savgol_polyorder,
        train_fraction=cfg.train_fraction,
        master_seed=cfg.master_seed,
    )

    np.savez(dataset_path, **data)
    print(f"  ✅ Dataset saved: train={data['train_x'].shape}, "
          f"test={data['test_x'].shape}, dt={float(data['dt']):.6f}s")

    # Save meta.json — teacher params documented as exploration-fixed SSOT
    meta = {
        'system': SYSTEM,
        'dataset_version': DATASET_VERSION,
        'state_dim': 2,
        'input_dim': 1,
        'n_train': data['train_x'].shape[0],
        'n_val':   data['val_x'].shape[0],
        'n_test':  data['test_x'].shape[0],
        'W':    int(data['train_x'].shape[1]),
        'dt':   float(data['dt']),
        'W_seconds': float(data['dt']) * int(data['train_x'].shape[1]),
        'savgol_window': cfg.savgol_window,
        'savgol_polyorder': cfg.savgol_polyorder,
        'train_fraction': cfg.train_fraction,
        'master_seed': cfg.master_seed,
        'teacher_params_ssot': {
            'provenance': 'exploration-fixed defaults (2026-03-10 least-squares on 5000 pts)',
            'k':  DUFFING_K_DEFAULT,
            'k3': DUFFING_K3_DEFAULT,
            'c':  DUFFING_C_DEFAULT,
            'b':  DUFFING_B_DEFAULT,
            'note': 'n_train-level re-fitting disabled — numerically unstable at n=3',
        },
        'notes': (
            'Real Silverbox measurement data. SNLS80mV multisine excitation. '
            'State estimated via SavGol filter. '
            'Duffing teacher uses exploration-fixed DEFAULT params (not re-fitted from train data).'
        ),
    }
    meta_path = dataset_path.parent / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ meta.json saved")

    return data


# ============================================================
# SHA-256 Pool Fingerprint
# ============================================================

def compute_pool_sha(pool_x: np.ndarray, pool_u: np.ndarray) -> str:
    """
    Compute SHA-256 fingerprint of augmentation pool.

    Args:
        pool_x: Pool state trajectories (pool_size, W, 2)
        pool_u: Pool input trajectories (pool_size, W, 1)

    Returns:
        First 16 hex characters of SHA-256
    """
    combined = (pool_x.astype(np.float32).tobytes() +
                pool_u.astype(np.float32).tobytes())
    return hashlib.sha256(combined).hexdigest()[:16]


# ============================================================
# E-SINDy Baseline
# ============================================================

def run_esindy_baseline(
    cfg: SilverboxGateConfig,
    data: Dict[str, np.ndarray],
    scaler: ColumnScaler,
) -> Dict:
    """
    Phase 1: Run E-SINDy baseline on training data.

    Fits bootstrap ensemble to training data and computes z-matrix
    (coefficient stability metric) for fragile pair identification.

    Args:
        cfg: Experiment configuration
        data: Dataset dict
        scaler: ColumnScaler fitted on training data

    Returns:
        Dict with z_matrix, fragile_pairs, failure_mode, ensemble, kappa
    """
    rng_base = np.random.default_rng(cfg.baseline_seed)

    # Flatten training data
    train_x_flat  = data['train_x'].reshape(-1, 2).astype(np.float64)
    train_u_flat  = data['train_u'].reshape(-1, 1).astype(np.float64)
    train_dx_flat = data['train_dx'].reshape(-1, 2).astype(np.float64)

    # Build and scale library
    Theta_raw, feat_names = build_silverbox_library(train_x_flat, train_u_flat)
    Theta_scaled = scaler.transform(Theta_raw)

    # Compute condition number
    kappa = float(np.linalg.cond(Theta_scaled))
    print(f"  κ (scaled Silverbox library) = {kappa:.3e}")

    # E-SINDy ensemble
    n_train = data['train_x'].shape[0]
    T_steps = data['train_x'].shape[1]
    ensemble = ESINDyEnsemble(
        n_bootstrap=cfg.n_bootstrap,
        threshold=cfg.threshold,
        random_state=cfg.baseline_seed,
    )
    ensemble.fit(
        Theta_scaled, train_dx_flat,
        n_trajectories=n_train,
        T=T_steps,
        scaler=scaler,
        target_scale=None,
    )

    # z-matrix: |mean_b(ξ)| / (std_b(ξ) + ε)
    z_matrix = compute_z_matrix(ensemble, cfg.z_eps)
    print(f"  z_matrix shape: {z_matrix.shape} "
          f"(min={z_matrix.min():.3f}, max={z_matrix.max():.3f})")

    # Fragile pairs
    fragile_pairs = get_silverbox_fragile_pairs(
        z_matrix, z_threshold=cfg.z_fragile_threshold
    )
    print(f"  Fragile pairs (z_threshold={cfg.z_fragile_threshold}): "
          f"{len(fragile_pairs)}")

    # Diagnose failure mode
    oracle = get_silverbox_oracle_support()
    failure_mode = diagnose_failure_mode(fragile_pairs, oracle, feat_names)
    print(f"  Failure mode: {failure_mode}")

    return {
        'z_matrix':     z_matrix,
        'fragile_pairs': fragile_pairs,
        'failure_mode': failure_mode,
        'ensemble':     ensemble,
        'kappa':        kappa,
        'feat_names':   feat_names,
        'Theta_scaled': Theta_scaled,
    }


def compute_z_matrix(
    ensemble: 'ESINDyEnsemble',
    z_eps: float,
) -> np.ndarray:
    """
    Compute z-metric matrix from ensemble coefficients.

    z(i,j) = |mean_b(ξ_b[i,j])| / (std_b(ξ_b[i,j]) + ε)

    Uses ensemble.coefficients_mean_ and coefficients_std_ (unscaled).

    Args:
        ensemble: Fitted ESINDyEnsemble
        z_eps: Small constant to avoid division by zero

    Returns:
        z_matrix: (N_SILVERBOX_FEATURES, 2) float array
    """
    mean_coef = np.abs(ensemble.coefficients_mean_)  # (n_features, n_targets)
    std_coef  = ensemble.coefficients_std_            # (n_features, n_targets)
    z_matrix  = mean_coef / (std_coef + z_eps)
    return z_matrix


def diagnose_failure_mode(
    fragile_pairs: List,
    oracle: np.ndarray,
    feat_names: List[str],
) -> str:
    """
    Determine whether fragile pairs are mostly recall (oracle missed)
    or precision (spurious) failures.

    Returns:
        'recall_fragility', 'precision_collapse', or 'mixed'
    """
    if len(fragile_pairs) == 0:
        return 'none'

    n_recall   = sum(1 for fp in fragile_pairs if oracle[fp[0], fp[1]])
    n_spurious = sum(1 for fp in fragile_pairs if not oracle[fp[0], fp[1]])

    total = len(fragile_pairs)
    if n_spurious / total >= 0.70:
        return 'precision_collapse'
    elif n_recall / total >= 0.70:
        return 'recall_fragility'
    else:
        return 'mixed'


# ============================================================
# Pool Generation (Duffing Teacher)
# ============================================================

def generate_pool(
    cfg: SilverboxGateConfig,
    data: Dict[str, np.ndarray],
    duffing_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Phase 2: Generate augmentation pool using Duffing teacher.

    GMM is fitted on training IC pairs [x1_0, x2_0].
    Teacher integrates Duffing ODE (u=0) from GMM-sampled ICs.

    Args:
        cfg: Experiment configuration
        data: Dataset dict (used for training IC extraction)
        duffing_params: Fitted Duffing params (k, k3, c). If None, use defaults.

    Returns:
        pool_x:    (pool_size, W, 2)
        pool_dx:   (pool_size, W, 2)
        pool_u:    (pool_size, W, 1) — zeros
        pool_sha:  SHA-256 fingerprint (16 hex chars)
    """
    dt = float(data['dt'])
    W  = int(data['train_x'].shape[1])

    # Build Duffing teacher — always use DEFAULT exploration-fitted parameters.
    # Rationale: n_train=3 re-fitting is unreliable (too few data points).
    # Default params (k=-93283, k3=-44580, c=-2.95, b=3100) were fitted on
    # 5000 points from the full signal and are physically validated (c < 0).
    teacher_params = {
        'k':  DUFFING_K_DEFAULT,
        'k3': DUFFING_K3_DEFAULT,
        'c':  DUFFING_C_DEFAULT,
    }

    teacher = SilverboxDuffingTeacher(params=teacher_params)
    print(f"  Teacher Duffing params: k={teacher_params['k']:.2f}, "
          f"k3={teacher_params['k3']:.2f}, c={teacher_params['c']:.4f}")

    # Extract training ICs: first time step of each training window
    # IC = [x1[0], x2[0]] for each training trajectory
    train_ics = data['train_x'][:, 0, :]  # (n_train, 2)
    print(f"  GMM fit on {len(train_ics)} training ICs (IC space: 2D)")
    print(f"  IC x1_0: mean={train_ics[:,0].mean():.4f}  "
          f"std={train_ics[:,0].std():.4f}")
    print(f"  IC x2_0: mean={train_ics[:,1].mean():.4f}  "
          f"std={train_ics[:,1].std():.4f}")

    # Fit GMM on training ICs
    gmm = GaussianMixture(
        n_components=cfg.gmm_n_components,
        covariance_type=cfg.gmm_covariance_type,
        random_state=cfg.gmm_seed,
    )
    gmm.fit(train_ics)

    # Sample ICs from GMM and integrate Duffing teacher
    rng_pool = np.random.default_rng(cfg.pool_seed)
    pool_x_list  = []
    pool_dx_list = []
    n_attempts   = 0
    n_reject     = 0

    print(f"  Generating {cfg.pool_size} pool trajectories "
          f"(max_attempts={cfg.max_pool_attempts})...")

    while len(pool_x_list) < cfg.pool_size:
        n_attempts += 1
        if n_attempts > cfg.max_pool_attempts:
            raise RuntimeError(
                f"Pool generation failed: only {len(pool_x_list)}/{cfg.pool_size} "
                f"trajectories after {cfg.max_pool_attempts} attempts. "
                f"Check Duffing parameters or increase max_pool_attempts."
            )

        # Sample IC from GMM
        ic_sample, _ = gmm.sample(1)
        ic = ic_sample[0]  # (2,)

        # Integrate Duffing teacher
        result = teacher.generate_trajectory(
            rng=rng_pool,
            ic=ic,
            W=W,
            dt=dt,
            max_state_norm=cfg.max_state_norm,
            max_attempts=1,
        )

        if result is None:
            n_reject += 1
            continue

        # v1.3: SavGol state+derivative from x1 only (논문 §3.3 일관성)
        # Training pipeline: x2=SavGol(y,deriv=1), dx2=SavGol(y,deriv=2)
        # Pool pipeline: identical — reconstruct x2/dx from x1 via SavGol.
        x_ode = result['x']               # (W, 2) from ODE [x1, x2]
        x1_clean = x_ode[:, 0]            # position only (= "raw y")

        x2_savgol = savgol_filter(
            x1_clean, window_length=cfg.savgol_window,
            polyorder=cfg.savgol_polyorder, deriv=1, delta=dt,
        )
        dx1_savgol = x2_savgol            # kinematic identity: dx1/dt = x2
        dx2_savgol = savgol_filter(
            x1_clean, window_length=cfg.savgol_window,
            polyorder=cfg.savgol_polyorder, deriv=2, delta=dt,
        )

        x_processed  = np.column_stack([x1_clean, x2_savgol])    # (W, 2)
        dx_processed = np.column_stack([dx1_savgol, dx2_savgol])  # (W, 2)

        pool_x_list.append(x_processed)
        pool_dx_list.append(dx_processed)

    pool_x  = np.array(pool_x_list,  dtype=np.float32)   # (P, W, 2)
    pool_dx = np.array(pool_dx_list, dtype=np.float32)   # (P, W, 2)
    pool_u  = np.zeros((cfg.pool_size, W, 1), dtype=np.float32)

    pool_sha = compute_pool_sha(pool_x, pool_u)

    accept_rate = cfg.pool_size / n_attempts
    print(f"  ✅ Pool generated: {cfg.pool_size} trajectories, "
          f"acceptance={accept_rate:.1%}, reject={n_reject}, SHA={pool_sha}")

    return pool_x, pool_dx, pool_u, pool_sha


# ============================================================
# Augmented E-SINDy
# ============================================================

def run_esindy_augmented(
    cfg: SilverboxGateConfig,
    data: Dict[str, np.ndarray],
    pool_x: np.ndarray,
    pool_dx: np.ndarray,
    pool_u: np.ndarray,
    selected_indices: np.ndarray,
    scaler: ColumnScaler,
    seed: int,
) -> Dict:
    """
    Run E-SINDy with augmented data (training + selected pool trajectories).

    Args:
        cfg: Configuration
        data: Dataset dict
        pool_x: Full pool states (P, W, 2)
        pool_dx: Full pool derivatives (P, W, 2)
        pool_u: Full pool inputs (P, W, 1) — zeros
        selected_indices: Which pool trajectories to include
        scaler: ColumnScaler fitted on training data ONLY
        seed: E-SINDy bootstrap seed

    Returns:
        Dict with z_matrix_after, ensemble_aug
    """
    # Training data
    train_x_flat  = data['train_x'].reshape(-1, 2).astype(np.float64)
    train_u_flat  = data['train_u'].reshape(-1, 1).astype(np.float64)
    train_dx_flat = data['train_dx'].reshape(-1, 2).astype(np.float64)

    # Selected pool data
    sel_x  = pool_x[selected_indices].reshape(-1, 2).astype(np.float64)
    sel_u  = pool_u[selected_indices].reshape(-1, 1).astype(np.float64)
    sel_dx = pool_dx[selected_indices].reshape(-1, 2).astype(np.float64)

    # Concatenate training + augmented
    aug_x_flat  = np.concatenate([train_x_flat,  sel_x],  axis=0)
    aug_u_flat  = np.concatenate([train_u_flat,  sel_u],  axis=0)
    aug_dx_flat = np.concatenate([train_dx_flat, sel_dx], axis=0)

    # Build library (scaler fitted on training only — SSOT rule)
    Theta_raw, _ = build_silverbox_library(aug_x_flat, aug_u_flat)
    Theta_scaled = scaler.transform(Theta_raw)

    # E-SINDy ensemble
    n_train_orig = data['train_x'].shape[0]
    T_steps = data['train_x'].shape[1]
    n_aug = len(selected_indices)
    n_total_traj = n_train_orig + n_aug

    ensemble_aug = ESINDyEnsemble(
        n_bootstrap=cfg.n_bootstrap,
        threshold=cfg.threshold,
        random_state=seed,
    )
    ensemble_aug.fit(
        Theta_scaled, aug_dx_flat,
        n_trajectories=n_total_traj,
        T=T_steps,
        scaler=scaler,
        target_scale=None,
    )

    z_matrix_after = compute_z_matrix(ensemble_aug, cfg.z_eps)

    return {
        'z_matrix_after': z_matrix_after,
        'ensemble_aug':   ensemble_aug,
    }


# ============================================================
# Selection Methods
# ============================================================

def select_random(
    cfg: SilverboxGateConfig,
    pool_x: np.ndarray,
    pool_u: np.ndarray,
    seed: int,
    scaler: ColumnScaler,
) -> np.ndarray:
    """
    Random selection: sample n_select indices uniformly from pool.

    Args:
        cfg: Configuration
        pool_x: Pool states (P, W, 2)
        pool_u: Pool inputs (P, W, 1)
        seed: Random seed
        scaler: Unused (kept for interface consistency)

    Returns:
        selected_indices: (n_select,) array of pool indices
    """
    rng = np.random.default_rng(seed)
    P = pool_x.shape[0]
    return rng.choice(P, size=cfg.n_select, replace=False)


def select_doptimal(
    cfg: SilverboxGateConfig,
    data: Dict[str, np.ndarray],
    pool_x: np.ndarray,
    pool_u: np.ndarray,
    fragile_pairs: List[List[int]],
    scaler: ColumnScaler,
) -> np.ndarray:
    """
    D-optimal selection: maximize |det(X^T X)| over fragile pair features.

    NOTE: For pool with u=0, the u column (feature 7) in the library
    is zero for all pool trajectories. D-optimal selects based on
    x1, x2, and cross-term diversity only. This is a known limitation
    of the u=0 teacher design.

    Args:
        cfg: Configuration
        data: Dataset dict (for training data Theta)
        pool_x: Pool states (P, W, 2)
        pool_u: Pool inputs (P, W, 1) — zeros for teacher
        fragile_pairs: List of [feature_idx, target_idx] from baseline
        scaler: ColumnScaler fitted on training data

    Returns:
        selected_indices: (n_select,) array of pool indices
    """
    P = pool_x.shape[0]

    # Build pool library matrices (per trajectory)
    # Extract feature rows for fragile feature indices only
    fragile_feature_indices = sorted(set(fp[0] for fp in fragile_pairs))
    if len(fragile_feature_indices) == 0:
        print("  D-opt WARNING: no fragile pairs found → falling back to random (seed=0)")
        rng = np.random.default_rng(0)
        return rng.choice(P, size=cfg.n_select, replace=False)

    print(f"  D-opt fragile feature indices: {fragile_feature_indices}")

    # Compute per-trajectory Fisher information contribution
    pool_info = []
    for i in range(P):
        x_traj = pool_x[i].astype(np.float64)   # (W, 2)
        u_traj = pool_u[i].astype(np.float64)   # (W, 1)
        Theta_raw, _ = build_silverbox_library(x_traj, u_traj)
        Theta_scaled = scaler.transform(Theta_raw)
        Theta_frag   = Theta_scaled[:, fragile_feature_indices]  # (W, n_frag)
        XtX = Theta_frag.T @ Theta_frag  # (n_frag, n_frag)
        info = float(np.linalg.slogdet(XtX + cfg.dopt_lambda * np.eye(len(fragile_feature_indices)))[1])
        pool_info.append(info)

    pool_info = np.array(pool_info)

    # Greedy forward selection (D-optimal greedy)
    # Start with training data Fisher matrix
    train_x_flat = data['train_x'].reshape(-1, 2).astype(np.float64)
    train_u_flat = data['train_u'].reshape(-1, 1).astype(np.float64)
    Theta_train_raw, _ = build_silverbox_library(train_x_flat, train_u_flat)
    Theta_train_scaled = scaler.transform(Theta_train_raw)
    Theta_train_frag   = Theta_train_scaled[:, fragile_feature_indices]
    M_current = Theta_train_frag.T @ Theta_train_frag  # Current information matrix

    selected = []
    remaining = list(range(P))

    for _ in range(min(cfg.n_select, P)):
        best_idx   = None
        best_logdet = -np.inf

        # Subsample candidates for speed
        n_cand = min(cfg.dopt_n_candidates, len(remaining))
        candidates = remaining[:n_cand]

        for idx in candidates:
            x_traj = pool_x[idx].astype(np.float64)
            u_traj = pool_u[idx].astype(np.float64)
            Theta_raw, _ = build_silverbox_library(x_traj, u_traj)
            Theta_scaled = scaler.transform(Theta_raw)
            Theta_frag   = Theta_scaled[:, fragile_feature_indices]
            M_candidate  = M_current + Theta_frag.T @ Theta_frag
            reg_M        = M_candidate + cfg.dopt_lambda * np.eye(len(fragile_feature_indices))
            logdet       = float(np.linalg.slogdet(reg_M)[1])
            if logdet > best_logdet:
                best_logdet = logdet
                best_idx    = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

        # Update M_current
        x_traj = pool_x[best_idx].astype(np.float64)
        u_traj = pool_u[best_idx].astype(np.float64)
        Theta_raw, _ = build_silverbox_library(x_traj, u_traj)
        Theta_scaled = scaler.transform(Theta_raw)
        Theta_frag   = Theta_scaled[:, fragile_feature_indices]
        M_current   += Theta_frag.T @ Theta_frag

    # Pad with random if not enough selected
    if len(selected) < cfg.n_select:
        rng = np.random.default_rng(0)
        extra = [i for i in range(P) if i not in selected]
        selected.extend(
            rng.choice(extra, size=cfg.n_select - len(selected), replace=False).tolist()
        )

    return np.array(selected[:cfg.n_select])


# ============================================================
# Score Computation
# ============================================================

def compute_delta_raw(
    z_before: np.ndarray,
    z_after: np.ndarray,
    fragile_pairs: List[List[int]],
) -> float:
    """
    Compute delta_raw = median(z_after − z_before) over fragile pairs.

    Args:
        z_before: (8, 2) z-matrix before augmentation
        z_after:  (8, 2) z-matrix after augmentation
        fragile_pairs: List of [feature_idx, target_idx]

    Returns:
        delta_raw (float)
    """
    if len(fragile_pairs) == 0:
        return 0.0
    diffs = [z_after[fp[0], fp[1]] - z_before[fp[0], fp[1]]
             for fp in fragile_pairs]
    return float(np.median(diffs))


def compute_score_aligned(
    delta_raw: float,
    failure_mode: str,
) -> float:
    """
    Convert delta_raw to score_aligned (positive = improvement).

    Convention:
        recall_fragility  → +delta_raw (want z to increase → recall recovered)
        precision_collapse→ -delta_raw (want z to decrease → spurious removed)
        mixed             → -delta_raw (precision_collapse dominant in Silverbox n=3;
                            x1² spurious z=11.7 >> x1³ recall z=0.02)

    Args:
        delta_raw: Raw score (median z_after − z_before)
        failure_mode: 'recall_fragility', 'precision_collapse', or 'mixed'

    Returns:
        score_aligned
    """
    if failure_mode == 'recall_fragility':
        return delta_raw
    else:
        # precision_collapse and mixed: -delta_raw
        # For mixed, precision collapse is dominant (x1² z=11.7 >> x1³ z=0.02)
        return -delta_raw


def determine_pass_level(
    score_aligned: float,
    ci_lower: float,
) -> str:
    """Determine pass level from score_aligned and CI lower bound."""
    if ci_lower > GATE2_CEILING:
        return 'CEILING_BREAK'
    elif ci_lower > 0:
        return 'STRONG_PASS'
    elif score_aligned > 0:
        return 'SOFT_PASS'
    else:
        return 'NULL'


def bootstrap_ci(
    scores: np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap CI for median score_aligned.

    Args:
        scores: Array of score_aligned values from random seeds
        B: Bootstrap resamples
        alpha: CI level (1-alpha CI)
        seed: Random seed

    Returns:
        (ci_lower, ci_upper) — (1-alpha) confidence interval
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    medians = [np.median(rng.choice(scores, size=n, replace=True))
               for _ in range(B)]
    ci_lower = float(np.percentile(medians, 100 * alpha / 2))
    ci_upper = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


# ============================================================
# Results Saving
# ============================================================

def save_manifest(
    results_dir: Path,
    cfg: SilverboxGateConfig,
    run_id: str,
    pool_sha: str,
    method: str,
    seed: int,
    baseline_result: Dict,
    data: Dict,
    runner_version: str = RUNNER_VERSION,
) -> None:
    """Save manifest.json to results directory."""
    dt = float(data['dt'])
    manifest = {
        'run_id':           run_id,
        'runner_version':   runner_version,
        'timestamp':        datetime.now().isoformat(),
        'system':           SYSTEM,
        'gate':             'gate4e2',
        'dataset_version':  cfg.dataset_version,
        'library_version':  'Silverbox-Duffing-8term-v1.0',
        'n_train':          cfg.n_train,
        'W':                cfg.W,
        'dt':               dt,
        'W_seconds':        cfg.W * dt,
        'method':           method,
        'baseline_seed':    cfg.baseline_seed,
        'augment_seed':     seed,
        'n_select':         cfg.n_select,
        'pool_size':        cfg.pool_size,
        'pool_sha':         pool_sha,
        'gmm_source':       'train_ICs_2D_[x1_0,x2_0]',
        'gmm_n_components': cfg.gmm_n_components,
        'teacher':          'Duffing_ODE_u_equals_0',
        # P0-1 SSOT: teacher uses exploration-fixed defaults, NOT data-fitted params
        'teacher_params': {
            'provenance': 'exploration-fixed defaults (2026-03-10)',
            'k':  DUFFING_K_DEFAULT,
            'k3': DUFFING_K3_DEFAULT,
            'c':  DUFFING_C_DEFAULT,
            'b':  DUFFING_B_DEFAULT,
        },
        'fragile_pairs':    baseline_result['fragile_pairs'],
        'failure_mode':     baseline_result['failure_mode'],
        'n_fragile':        len(baseline_result['fragile_pairs']),
        'kappa':            baseline_result['kappa'],
        'z_fragile_threshold': cfg.z_fragile_threshold,
        'n_bootstrap':      cfg.n_bootstrap,
        'threshold':        cfg.threshold,
    }
    with open(results_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)


def save_metrics(
    results_dir: Path,
    delta_raw: float,
    score_aligned: float,
    pass_level: str,
    ci_lower: Optional[float],
    ci_upper: Optional[float],
    z_before: np.ndarray,
    z_after: np.ndarray,
    fragile_pairs: List,
    failure_mode: str,
    method: str,
    seed: int,
    pool_sha: str,
) -> None:
    """Save metrics.json to results directory."""
    metrics = {
        'delta_raw':     delta_raw,
        'score_aligned': score_aligned,
        'pass_level':    pass_level,
        'ci_lower':      ci_lower,
        'ci_upper':      ci_upper,
        'failure_mode':  failure_mode,
        'n_fragile':     len(fragile_pairs),
        'method':        method,
        'seed':          seed,
        'pool_sha':      pool_sha,
        'sign_convention': (
            '+delta_raw (recall_fragility)' if failure_mode == 'recall_fragility'
            else '-delta_raw (precision_collapse or mixed)'
        ),
        'z_before_fragile': [
            float(z_before[fp[0], fp[1]]) for fp in fragile_pairs
        ],
        'z_after_fragile': [
            float(z_after[fp[0], fp[1]]) for fp in fragile_pairs
        ],
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


def save_sindy_coefficients(
    results_dir: Path,
    ensemble: 'ESINDyEnsemble',
    feat_names: List[str],
    target_names: List[str] = None,
) -> None:
    """Save sindy_coefficients.csv to results directory."""
    if target_names is None:
        target_names = SILVERBOX_TARGET_NAMES

    mean_coef = ensemble.coefficients_mean_  # (n_features, n_targets)
    std_coef  = ensemble.coefficients_std_

    with open(results_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['feature']
        for t in target_names:
            header.extend([f'{t}_mean', f'{t}_std'])
        writer.writerow(header)
        for i, fname in enumerate(feat_names):
            row = [fname]
            for j in range(len(target_names)):
                row.extend([f'{mean_coef[i,j]:.6f}', f'{std_coef[i,j]:.6f}'])
            writer.writerow(row)


def save_context_packet(
    run_id: str,
    cfg: SilverboxGateConfig,
    baseline_result: Dict,
    pool_sha: str,
    random_results: List[Dict],
    dopt_result: Optional[Dict],
    data: Dict,
) -> None:
    """Save context packet for next session."""
    cp_dir = paths.get_context_packet_dir()
    cp_path = cp_dir / f'CP_{run_id}.md'

    lines = [
        f"# Context Packet — Gate4e-2 Silverbox",
        f"**run_id**: {run_id}",
        f"**date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**runner**: run_silverbox_gate.py {RUNNER_VERSION}",
        "",
        "## Dataset",
        f"- system: {SYSTEM}, dataset_version: {cfg.dataset_version}",
        f"- n_train={cfg.n_train}, W={cfg.W}, dt={float(data['dt']):.6f}s",
        f"- W_seconds={cfg.W * float(data['dt']):.3f}s",
        f"- savgol: window={cfg.savgol_window}, polyorder={cfg.savgol_polyorder}",
        "",
        "## Duffing Parameters (exploration-fixed SSOT — NOT re-fitted from train data)",
        f"- k  = {DUFFING_K_DEFAULT:.4f}",
        f"- k3 = {DUFFING_K3_DEFAULT:.4f}",
        f"- c  = {DUFFING_C_DEFAULT:.4f}  (c < 0 → positive damping ✅)",
        f"- b  = {DUFFING_B_DEFAULT:.4f}",
        f"- provenance: least-squares fit on 5000 pts, 2026-03-10 exploration",
        f"- NOTE: teacher u=0 (free oscillation; no input-direction augmentation)",
        "",
        "## Baseline",
        f"- κ = {baseline_result['kappa']:.3e}",
        f"- failure_mode = {baseline_result['failure_mode']}",
        f"- n_fragile = {len(baseline_result['fragile_pairs'])}",
        f"- fragile_pairs = {baseline_result['fragile_pairs']}",
        "",
        "## Pool",
        f"- pool_size = {cfg.pool_size}, pool_sha = `{pool_sha}`",
        f"- teacher = Duffing ODE u=0 (exploration-fixed params)",
        f"- GMM: n_components={cfg.gmm_n_components}, IC=[x1_0, x2_0] (2D)",
        "",
        "## Results Summary",
    ]

    failure_mode = baseline_result['failure_mode']
    sign_str = '+delta_raw' if failure_mode == 'recall_fragility' else '−delta_raw'

    if random_results:
        scores = [r['score_aligned'] for r in random_results]
        median_score = float(np.median(scores))
        null_count  = sum(1 for r in random_results if r['pass_level'] == 'NULL')
        soft_count  = sum(1 for r in random_results if r['pass_level'] == 'SOFT_PASS')
        lines.extend([
            f"### Random (10 seeds)",
            f"- score_aligned (median) = {median_score:.3f}",
            f"- NULL: {null_count}/10, SOFT_PASS: {soft_count}/10",
            f"- failure_mode → score_aligned = {sign_str}",
        ])

    if dopt_result is not None:
        lines.extend([
            f"### D-optimal",
            f"- score_aligned = {dopt_result['score_aligned']:.3f}",
            f"- pass_level = {dopt_result['pass_level']}",
            f"- ci = [{dopt_result.get('ci_lower', 'N/A')}, {dopt_result.get('ci_upper', 'N/A')}]",
        ])

    lines.extend([
        "",
        "## SSOT Reference",
        "| Artifact | SHA |",
        "|----------|-----|",
        f"| Silverbox pool | `{pool_sha}` |",
        "",
        "## Next Steps",
        "- Compare Silverbox failure mode with AEK/Lorenz",
        "- Update Gate4_Paper1_Results_Tables with Silverbox results",
        "- Cross-review with GPT if SOFT_PASS or better achieved",
    ])

    cp_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  ✅ Context Packet saved: {cp_path}")


# ============================================================
# Phase Runners
# ============================================================

def run_phase_baseline(
    cfg: SilverboxGateConfig,
) -> Tuple[Dict, Dict, ColumnScaler, str]:
    """
    Phase 0+1: Load dataset and run E-SINDy baseline.

    Returns:
        data, baseline_result, scaler, run_id
    """
    print("\n" + "="*60)
    print("Phase 0: Dataset Generation / Loading")
    print("="*60)
    data = generate_or_load_dataset(cfg)
    validate_silverbox_dataset(data)

    print("\n" + "="*60)
    print("Phase 1: E-SINDy Baseline")
    print("="*60)

    # Fit ColumnScaler on training data ONLY (SSOT rule)
    train_x_flat = data['train_x'].reshape(-1, 2).astype(np.float64)
    train_u_flat = data['train_u'].reshape(-1, 1).astype(np.float64)
    Theta_raw, _ = build_silverbox_library(train_x_flat, train_u_flat)
    scaler = ColumnScaler()
    scaler.fit(Theta_raw)
    print(f"  ColumnScaler fitted on training data "
          f"(n_train={cfg.n_train}, W={cfg.W})")

    baseline_result = run_esindy_baseline(cfg, data, scaler)

    run_id = paths.generate_run_id(note=cfg.note)
    results_dir_base = paths.get_results_dir(
        dataset_version=cfg.dataset_version,
        gate='gate4e2',
        track='ablation',
        method='baseline',
        n_train=cfg.n_train,
        seed=cfg.baseline_seed,
        run_id=run_id,
    )

    # Save baseline coefficients
    save_sindy_coefficients(
        results_dir_base,
        baseline_result['ensemble'],
        baseline_result['feat_names'],
    )

    # P0-2: Save baseline_meta.json — audit trail for fragile pairs, failure_mode, κ
    baseline_meta = {
        'run_id':          run_id,
        'runner_version':  RUNNER_VERSION,
        'timestamp':       datetime.now().isoformat(),
        'system':          SYSTEM,
        'gate':            'gate4e2',
        'dataset_version': cfg.dataset_version,
        'n_train':         cfg.n_train,
        'W':               cfg.W,
        'dt':              float(data['dt']),
        'kappa':           baseline_result['kappa'],
        'failure_mode':    baseline_result['failure_mode'],
        'n_fragile':       len(baseline_result['fragile_pairs']),
        'fragile_pairs':   baseline_result['fragile_pairs'],
        'z_fragile_threshold': cfg.z_fragile_threshold,
        'feat_names':      baseline_result['feat_names'],
        'z_matrix_shape':  list(baseline_result['z_matrix'].shape),
        'pool_sha':        None,  # pool not yet generated at baseline phase
        'sign_convention': (
            '+delta_raw' if baseline_result['failure_mode'] == 'recall_fragility'
            else '-delta_raw (precision_collapse or mixed)'
        ),
    }
    with open(results_dir_base / 'baseline_meta.json', 'w') as f:
        json.dump(baseline_meta, f, indent=2)
    print(f"  ✅ baseline_meta.json saved: {results_dir_base / 'baseline_meta.json'}")

    # GPT P0 fix: z_before.npy + fragile_pairs.json + manifest.json (Lorenz-level audit trail)
    np.save(results_dir_base / 'z_before.npy', baseline_result['z_matrix'])

    _oracle = get_silverbox_oracle_support()
    with open(results_dir_base / 'fragile_pairs.json', 'w') as f:
        json.dump({
            'fragile_pairs':    baseline_result['fragile_pairs'],
            'n_pairs':          len(baseline_result['fragile_pairs']),
            'failure_mode':     baseline_result['failure_mode'],
            'n_oracle_fragile': sum(1 for fp in baseline_result['fragile_pairs']
                                    if _oracle[fp[0], fp[1]]),
            'n_spurious_fragile': sum(1 for fp in baseline_result['fragile_pairs']
                                      if not _oracle[fp[0], fp[1]]),
            'kappa':            baseline_result['kappa'],
            'z_fragile_threshold': cfg.z_fragile_threshold,
        }, f, indent=2)

    with open(results_dir_base / 'manifest.json', 'w') as f:
        json.dump({
            'run_id':          run_id,
            'system':          SYSTEM,
            'gate':            'gate4e2',
            'method':          'esindy_baseline',
            'runner_version':  RUNNER_VERSION,
            'created_at':      datetime.now().isoformat(),
            'kappa':           baseline_result['kappa'],
            'failure_mode':    baseline_result['failure_mode'],
            'n_fragile_pairs': len(baseline_result['fragile_pairs']),
            'artifacts': [
                'manifest.json', 'z_before.npy',
                'sindy_coefficients.csv', 'fragile_pairs.json',
                'baseline_meta.json',
            ],
        }, f, indent=2)

    print(f"  ✅ z_before.npy + fragile_pairs.json + manifest.json saved")

    print(f"\n  Baseline fragile pairs ({len(baseline_result['fragile_pairs'])}): "
          f"{baseline_result['fragile_pairs']}")
    print(f"  Failure mode: {baseline_result['failure_mode']}")
    print(f"  κ: {baseline_result['kappa']:.3e}")

    return data, baseline_result, scaler, run_id


def run_phase_augment(
    cfg: SilverboxGateConfig,
    data: Dict,
    baseline_result: Dict,
    scaler: ColumnScaler,
    run_id: str,
    seeds: List[int],
    run_dopt: bool = True,
) -> Tuple[List[Dict], Optional[Dict], str]:
    """
    Phase 2+3: Pool generation and D-opt vs Random augmentation.

    Returns:
        random_results, dopt_result, pool_sha
    """
    print("\n" + "="*60)
    print("Phase 2: GMM Pool Generation (Duffing teacher, u=0)")
    print("="*60)

    pool_x, pool_dx, pool_u, pool_sha = generate_pool(cfg, data)

    print("\n" + "="*60)
    print("Phase 3: Augmentation — Random")
    print("="*60)

    fragile_pairs = baseline_result['fragile_pairs']
    z_before      = baseline_result['z_matrix']
    failure_mode  = baseline_result['failure_mode']

    if len(fragile_pairs) == 0:
        print("  WARNING: No fragile pairs found. "
              "Augmentation may not have meaningful effect.")

    random_results = []
    for seed in seeds:
        print(f"\n  [Random seed={seed}]")

        selected_idx = select_random(cfg, pool_x, pool_u, seed, scaler)

        aug_result = run_esindy_augmented(
            cfg, data, pool_x, pool_dx, pool_u,
            selected_idx, scaler, seed=seed,
        )

        z_after     = aug_result['z_matrix_after']
        delta_raw   = compute_delta_raw(z_before, z_after, fragile_pairs)
        score_al    = compute_score_aligned(delta_raw, failure_mode)

        # Save per-seed results
        res_dir = paths.get_results_dir(
            dataset_version=cfg.dataset_version,
            gate='gate4e2',
            track='ablation',
            method='random',
            n_train=cfg.n_train,
            seed=seed,
            run_id=run_id,
        )
        pass_level = determine_pass_level(score_al, -999.0)  # No CI per-seed
        save_manifest(res_dir, cfg, run_id, pool_sha, 'random', seed,
                      baseline_result, data)
        save_metrics(res_dir, delta_raw, score_al, pass_level,
                     None, None, z_before, z_after,
                     fragile_pairs, failure_mode, 'random', seed, pool_sha)
        save_sindy_coefficients(res_dir, aug_result['ensemble_aug'],
                                baseline_result['feat_names'])

        result = {
            'seed': seed, 'delta_raw': delta_raw,
            'score_aligned': score_al, 'pass_level': pass_level,
        }
        random_results.append(result)
        print(f"    delta_raw={delta_raw:.3f}, "
              f"score_aligned={score_al:.3f}, "
              f"pass={pass_level}")

    # Summary statistics for Random
    if random_results:
        scores_rand = [r['score_aligned'] for r in random_results]
        median_rand = float(np.median(scores_rand))
        null_count  = sum(1 for r in random_results if r['pass_level'] == 'NULL')
        soft_count  = sum(1 for r in random_results if r['pass_level'] == 'SOFT_PASS')
        ci_lo, ci_hi = bootstrap_ci(
            np.array(scores_rand),
            B=cfg.ci_bootstrap_B, alpha=cfg.ci_alpha
        )
        summary_pass = determine_pass_level(median_rand, ci_lo)
        print(f"\n  ── Random Summary ──")
        print(f"  score_aligned median = {median_rand:.3f}")
        print(f"  CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  NULL: {null_count}/10, SOFT: {soft_count}/10")
        print(f"  Summary pass level: {summary_pass}")

    # D-optimal
    dopt_result = None
    if run_dopt:
        print("\n" + "="*60)
        print("Phase 3: Augmentation — D-optimal")
        print("="*60)

        selected_dopt = select_doptimal(
            cfg, data, pool_x, pool_u, fragile_pairs, scaler
        )
        aug_result_dopt = run_esindy_augmented(
            cfg, data, pool_x, pool_dx, pool_u,
            selected_dopt, scaler, seed=cfg.baseline_seed,
        )

        z_after_dopt  = aug_result_dopt['z_matrix_after']
        delta_raw_dopt = compute_delta_raw(z_before, z_after_dopt, fragile_pairs)
        score_al_dopt  = compute_score_aligned(delta_raw_dopt, failure_mode)

        # Bootstrap CI for D-opt (single run — use per-fragile-pair bootstrap)
        dopt_scores = np.array([
            compute_score_aligned(
                float(z_after_dopt[fp[0], fp[1]] - z_before[fp[0], fp[1]]),
                failure_mode
            )
            for fp in fragile_pairs
        ]) if fragile_pairs else np.array([score_al_dopt])

        ci_lo_d, ci_hi_d = bootstrap_ci(
            dopt_scores, B=cfg.ci_bootstrap_B, alpha=cfg.ci_alpha
        )
        pass_level_dopt = determine_pass_level(score_al_dopt, ci_lo_d)

        res_dir_dopt = paths.get_results_dir(
            dataset_version=cfg.dataset_version,
            gate='gate4e2',
            track='ablation',
            method='doptimal',
            n_train=cfg.n_train,
            seed=0,
            run_id=run_id,
        )
        save_manifest(res_dir_dopt, cfg, run_id, pool_sha, 'doptimal', 0,
                      baseline_result, data)
        save_metrics(res_dir_dopt, delta_raw_dopt, score_al_dopt,
                     pass_level_dopt, ci_lo_d, ci_hi_d,
                     z_before, z_after_dopt, fragile_pairs, failure_mode,
                     'doptimal', 0, pool_sha)
        save_sindy_coefficients(res_dir_dopt, aug_result_dopt['ensemble_aug'],
                                baseline_result['feat_names'])

        dopt_result = {
            'delta_raw': delta_raw_dopt,
            'score_aligned': score_al_dopt,
            'pass_level': pass_level_dopt,
            'ci_lower': ci_lo_d,
            'ci_upper': ci_hi_d,
        }
        print(f"  D-opt score_aligned = {score_al_dopt:.3f}, "
              f"CI=[{ci_lo_d:.3f}, {ci_hi_d:.3f}], "
              f"pass={pass_level_dopt}")

    return random_results, dopt_result, pool_sha


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Silverbox Gate4e-2: E-SINDy + GMM Augmentation'
    )
    parser.add_argument(
        '--phase', choices=['all', 'baseline', 'augment'],
        default='all', help='Which phase(s) to run'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help='Random seeds for augmentation'
    )
    parser.add_argument(
        '--no-dopt', action='store_true',
        help='Skip D-optimal (run Random only)'
    )
    parser.add_argument(
        '--n-train', type=int, default=3,
        help='Number of training windows'
    )
    args = parser.parse_args()

    print("="*60)
    print(f"Gate4e-2: Silverbox Feasibility Sprint")
    print(f"Runner: {RUNNER_VERSION}  |  Date: {datetime.now():%Y-%m-%d %H:%M}")
    print("="*60)

    # Feature integrity check (AC2)
    print("\nAC2 Feature Integrity Check:")
    check_feature_integrity()

    cfg = SilverboxGateConfig(n_train=args.n_train)

    if args.phase in ('all', 'baseline'):
        data, baseline_result, scaler, run_id = run_phase_baseline(cfg)

    if args.phase in ('all', 'augment'):
        if args.phase == 'augment':
            # Reload baseline if augment-only
            data, baseline_result, scaler, run_id = run_phase_baseline(cfg)

        random_results, dopt_result, pool_sha = run_phase_augment(
            cfg, data, baseline_result, scaler, run_id,
            seeds=args.seeds,
            run_dopt=(not args.no_dopt),
        )

        # Save context packet
        save_context_packet(
            run_id, cfg, baseline_result, pool_sha,
            random_results, dopt_result, data,
        )

        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        if random_results:
            scores = [r['score_aligned'] for r in random_results]
            null_n = sum(1 for r in random_results if r['pass_level'] == 'NULL')
            soft_n = sum(1 for r in random_results if r['pass_level'] == 'SOFT_PASS')
            print(f"  Random: median={np.median(scores):.3f}, "
                  f"NULL={null_n}/10, SOFT={soft_n}/10")
        if dopt_result:
            print(f"  D-opt:  score={dopt_result['score_aligned']:.3f}, "
                  f"CI=[{dopt_result['ci_lower']:.3f}, {dopt_result['ci_upper']:.3f}], "
                  f"pass={dopt_result['pass_level']}")
        print(f"  Pool SHA: {pool_sha}")
        print(f"  Failure mode: {baseline_result['failure_mode']}")
        print(f"  κ: {baseline_result['kappa']:.3e}")

    elif args.phase == 'baseline':
        print("\n  Baseline-only mode complete.")


if __name__ == '__main__':
    main()