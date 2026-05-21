"""
Lynx-Hare Gate: Baseline + GMM Augmentation + D-opt vs Random Ablation

Purpose:
    Validate GMM+ODE Teacher-Aligned Augmentation on the real-world
    Lynx-Hare (Lotka-Volterra) dataset.

    4th benchmark for EAAI paper (Gate4e):
    - Real measured data (Hudson Bay fur trade records, 1900-1920)
    - E-SINDy lineage: same dataset as Fasel et al. (2022)
    - Role: public real-data lineage validation

    Three-phase experiment:
    Phase 0: Load / save dataset (lynxhare_v1)
    Phase 1: E-SINDy baseline → fragile pairs (z_before)
    Phase 2: GMM pool generation (LV ODE as teacher)
    Phase 3: D-optimal vs Random selection → score_aligned comparison

Design (v1.1 — sliding window):
    - n_train=3 sliding windows (window_size=7, stride=2) from the real 21-pt series.
      21-pt series → 8 pseudo-trajectories available; first 3 used for training.
      This gives valid trajectory-level bootstrap (n_traj=3 ≥ 2).
    - GMM: 2D, fitted on the 21 UNIQUE raw observations (H_t, L_t), t=0..20.
      NOT on the flattened training windows (which would over-represent 1900-1910).
      Raw unique points capture the full attractor geometry (rise + peak + fall).
    - ODE teacher: Lotka-Volterra with parameters estimated from real data
    - D-optimal: 1 run (confound-free)
    - Random: 10 seeds

Metric SSOT (Lynx-Hare — failure mode detected at runtime):
    z(i,j) = |mean_b(ξ_b[i,j])| / (std_b(ξ_b[i,j]) + ε)
    delta_raw = median(z_after − z_before) over fragile pairs
    score_aligned: determined by failure mode (runtime)
        recall_fragility   → +delta_raw  (like CP)
        precision_collapse → −delta_raw  (like AEK, Lorenz)

    Pass levels: CEILING_BREAK / STRONG_PASS / SOFT_PASS / NULL
    GATE2_CEILING = 0.058 (inherited from CP Gate2)

SSOT Rules:
    - paths.py for all path generation
    - plot_style.save_figure() for all figures
    - validate_lynxhare_dataset() (custom, STATE_DIM=2, INPUT_DIM=0)
    - manifest.json + metrics.json + sindy_coefficients.csv required
    - ColumnScaler fitted on training data ONLY
    - GMM fitted on raw 21 unique observations (NOT on training windows)

Usage:
    python experiments/run_lynxhare_gate.py --seeds 0 1 2 3 4 5 6 7 8 9
    python experiments/run_lynxhare_gate.py --phase baseline
    python experiments/run_lynxhare_gate.py --phase augment --seeds 0 1 2

Author: Claude (Gate-LynxHare)
Date: 2026-03-09
Runner version: v1.2 (pool dx → SavGol for §3.3 consistency; previously analytic)
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
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture

from src.contracts import paths
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble

# Lynx-Hare specific modules
from src.simulators.lynxhare_simulator import (
    generate_lynxhare_dataset,
    simulate_lv_trajectory,
    compute_lv_derivatives,
    T_REAL, DT_REAL, STATE_DIM, INPUT_DIM,
    WINDOW_SIZE, WINDOW_STRIDE, N_WINDOWS,
)
from src.sindy.lynxhare_library import (
    build_lynxhare_library,
    get_lynxhare_feature_names,
    get_lynxhare_oracle_support,
    get_lynxhare_fragile_pairs,
    assert_lynxhare_feature_integrity,
    N_LYNXHARE_FEATURES,
    LYNXHARE_TARGET_NAMES,
    LYNXHARE_FEATURE_NAMES,
)

# ============================================================
# Constants
# ============================================================

RUNNER_VERSION = 'v1.2'  # v1.2: pool dx → SavGol (§3.3 consistency; previously analytic)
GATE2_CEILING = 0.058       # CP Gate2 ceiling (SSOT)
DATASET_VERSION = 'lynxhare_v1'
SYSTEM = 'lynxhare'


# ============================================================
# Configuration
# ============================================================

@dataclass
class LynxHareGateConfig:
    """Lynx-Hare Gate experiment configuration."""

    # Dataset
    dataset_version: str = DATASET_VERSION
    system: str = SYSTEM
    n_train: int = 3        # Sliding windows from real data (low-data regime)
    window_size: int = WINDOW_SIZE    # 7 years per pseudo-trajectory
    window_stride: int = WINDOW_STRIDE  # stride=2 → 8 windows available

    # SavGol derivative for real data (applied to full 21-pt series)
    savgol_window: int = 5
    savgol_polyorder: int = 3

    # Simulation
    dt: float = DT_REAL          # 1.0 year
    T_steps: int = WINDOW_SIZE   # 7 time steps per pseudo-trajectory
    max_state: float = 500.0     # QC bound (thousands)

    # Baseline
    baseline_seed: int = 1

    # GMM: fitted on state observations (2D: H, L)
    gmm_n_components: int = 3
    gmm_covariance_type: str = 'full'
    gmm_seed: int = 42

    # Pool
    pool_size: int = 200
    max_pool_attempts: int = 3000
    pool_seed: int = 42

    # QC thresholds (physical bounds for LV)
    qc_min_state: float = 0.1    # H, L > 0.1 thousand
    qc_max_state: float = 500.0  # H, L < 500 thousand

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

    # Fragile pair threshold
    z_fragile_threshold: float = 2.0

    # CI
    ci_bootstrap_B: int = 2000
    ci_alpha: float = 0.05

    # D-optimal
    dopt_lambda: float = 1e-6
    dopt_n_candidates: int = 200

    # Output
    note: str = 'lynxhare_gate'


# ============================================================
# Dataset
# ============================================================

def generate_or_load_dataset(cfg: LynxHareGateConfig) -> Dict[str, np.ndarray]:
    """Load or generate Lynx-Hare dataset."""
    dataset_path = paths.ROOT / 'data' / SYSTEM / DATASET_VERSION / 'dataset.npz'
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        print(f"  Loading existing dataset: {dataset_path}")
        data = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  train_x: {data['train_x'].shape}, "
              f"val_x: {data['val_x'].shape}, "
              f"test_x: {data['test_x'].shape}")
        return data

    print(f"  Generating Lynx-Hare dataset → {dataset_path}")
    data = generate_lynxhare_dataset(
        n_train=cfg.n_train,
        n_val=5,
        n_test=15,
        window_size=cfg.window_size,
        window_stride=cfg.window_stride,
        dt=cfg.dt,
        savgol_window=cfg.savgol_window,
        savgol_polyorder=cfg.savgol_polyorder,
        master_seed=42,
        max_state=cfg.max_state,
    )

    np.savez(dataset_path, **data)
    print(f"  ✅ Dataset saved: train={data['train_x'].shape}, "
          f"test={data['test_x'].shape}")

    # Save meta.json
    meta = {
        'system': SYSTEM,
        'dataset_version': DATASET_VERSION,
        'state_dim': STATE_DIM,
        'input_dim': INPUT_DIM,
        'n_train': int(data['train_x'].shape[0]),
        'n_val': int(data['val_x'].shape[0]),
        'n_test': int(data['test_x'].shape[0]),
        'T_steps': int(data['train_x'].shape[1]),
        'window_size': int(cfg.window_size),
        'window_stride': int(cfg.window_stride),
        'n_windows_available': int(data.get('n_windows_available', N_WINDOWS)),
        'dt': float(cfg.dt),
        'data_source': 'Hudson Bay fur trade records 1900-1920 (sliding window)',
        'reference': 'Elton & Nicholson (1942), MacLulich (1937)',
        'esindy_reference': 'Fasel et al. (2022, Proc. R. Soc. A)',
        'lv_alpha': float(data['lv_alpha']),
        'lv_beta':  float(data['lv_beta']),
        'lv_gamma': float(data['lv_gamma']),
        'lv_delta': float(data['lv_delta']),
        'dx_source': 'savgol_full_series (train) / analytic_lv (val/test)',
        'savgol_window': cfg.savgol_window,
        'savgol_polyorder': cfg.savgol_polyorder,
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_lynxhare_gate.py',
    }
    with open(dataset_path.parent / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return data


def validate_lynxhare_dataset(data: Dict[str, np.ndarray]) -> None:
    """
    Lynx-Hare specific dataset validation.
    STATE_DIM=2, INPUT_DIM=0 (u stored as zeros for schema compat).
    """
    for split in ['train', 'val', 'test']:
        x  = data[f'{split}_x']
        u  = data[f'{split}_u']
        dx = data[f'{split}_dx']

        assert x.ndim == 3 and x.shape[2] == 2, \
            f"{split}_x shape mismatch: {x.shape} (expected (..., 2))"
        assert u.ndim == 3 and u.shape[2] == 1, \
            f"{split}_u shape mismatch: {u.shape}"
        assert dx.ndim == 3 and dx.shape[2] == 2, \
            f"{split}_dx shape mismatch: {dx.shape}"
        assert x.shape[:2] == u.shape[:2] == dx.shape[:2], \
            f"{split} shape mismatch across arrays"

        assert np.all(np.isfinite(x)),  f"{split}_x contains non-finite"
        assert np.all(np.isfinite(dx)), f"{split}_dx contains non-finite"
        assert np.allclose(u, 0), \
            f"{split}_u should be zeros (Lynx-Hare has no control input)"

    assert 't' in data and 'dt' in data, "Missing t or dt in dataset"
    print("  ✅ Lynx-Hare dataset validation passed")


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
    Run E-SINDy on train + augmented Lynx-Hare data.

    ColumnScaler fitted on training data ONLY (confound-free).

    Args:
        train_x: (N_tr, T, 2)
        aug_x:   (N_aug, T, 2)
        train_dx:(N_tr, T, 2)
        aug_dx:  (N_aug, T, 2)

    Returns:
        Dict with z (6,2), coefficients_mean, kappa, etc.
    """
    N_tr, T_tr, _ = train_x.shape
    N_aug, T_aug, _ = aug_x.shape

    if T_tr != T_aug:
        raise ValueError(f"T mismatch: train={T_tr}, aug={T_aug}")

    x_tr_flat  = train_x.reshape(-1, 2)
    x_au_flat  = aug_x.reshape(-1, 2)
    dx_tr_flat = train_dx.reshape(-1, 2)
    dx_au_flat = aug_dx.reshape(-1, 2)

    x_all  = np.vstack([x_tr_flat, x_au_flat])
    dx_all = np.vstack([dx_tr_flat, dx_au_flat])

    n_traj   = N_tr + N_aug
    n_samples = n_traj * T_tr

    # Build library — scaler fitted on train only
    Theta_tr, feat_names = build_lynxhare_library(x_tr_flat)
    Theta_all, _ = build_lynxhare_library(x_all)

    scaler = ColumnScaler()
    scaler.fit(Theta_tr)
    Theta_all_scaled = scaler.transform(Theta_all)
    Theta_tr_scaled  = scaler.transform(Theta_tr)

    kappa = float(np.linalg.cond(Theta_tr_scaled))

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

    coeff_mean = ensemble.coefficients_mean_      # (6, 2)
    coeff_std  = ensemble.coefficients_std_       # (6, 2)
    inc_prob   = ensemble.inclusion_probability_  # (6, 2)
    support_mask = np.abs(coeff_mean) > 0
    z = np.abs(coeff_mean) / (coeff_std + z_eps)

    return {
        'z':                 z,
        'coefficients_mean': coeff_mean,
        'coefficients_std':  coeff_std,
        'inclusion_probability': inc_prob,
        'support_mask':      support_mask,
        'scaler':            scaler,
        'feature_names':     feat_names,
        'kappa':             kappa,
        'n_total_samples':   n_samples,
        'n_original':        N_tr * T_tr,
        'n_augmented':       N_aug * T_aug,
        'n_traj_total':      n_traj,
    }


def run_baseline(
    train_x: np.ndarray,
    train_dx: np.ndarray,
    cfg: LynxHareGateConfig,
) -> Dict[str, Any]:
    """Run E-SINDy baseline (no augmentation). Returns z_before + fragile_pairs."""
    N_tr, T_tr, _ = train_x.shape
    x_flat  = train_x.reshape(-1, 2)
    dx_flat = train_dx.reshape(-1, 2)

    Theta, feat_names = build_lynxhare_library(x_flat)
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    kappa = float(np.linalg.cond(Theta_scaled))

    # n_train >= 2 guaranteed by sliding window design (WINDOW_SIZE=7, stride=2 → 8 windows)
    if N_tr < 2:
        print(f"  ⚠️  n_traj={N_tr} < 2 — bootstrap variance may be unreliable")

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
    coeff_std  = ensemble.coefficients_std_
    z = np.abs(coeff_mean) / (coeff_std + cfg.z_eps)

    fragile_pairs, failure_mode = get_lynxhare_fragile_pairs(
        z, z_threshold=cfg.z_fragile_threshold
    )

    oracle = get_lynxhare_oracle_support()
    n_oracle_fragile   = sum(1 for f, t in fragile_pairs if oracle[f, t])
    n_spurious_fragile = sum(1 for f, t in fragile_pairs if not oracle[f, t])

    print(f"  κ (baseline): {kappa:.3e}")
    print(f"  Fragile pairs: {len(fragile_pairs)} total "
          f"({n_oracle_fragile} oracle/recall, "
          f"{n_spurious_fragile} spurious/precision)")

    if failure_mode == 'recall_fragility':
        print(f"  Dominant failure: RECALL FRAGILITY → score_aligned = +delta_raw")
    else:
        print(f"  Dominant failure: PRECISION COLLAPSE → score_aligned = −delta_raw")

    return {
        'z':                z,
        'coefficients_mean': coeff_mean,
        'coefficients_std':  coeff_std,
        'kappa':            kappa,
        'fragile_pairs':    fragile_pairs,
        'n_oracle_fragile': n_oracle_fragile,
        'n_spurious_fragile': n_spurious_fragile,
        'failure_mode':     failure_mode,
        'feature_names':    feat_names,
        'scaler':           scaler,
    }


# ============================================================
# Metrics
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
    Compute score_aligned and pass_level.

    Metric SSOT:
        delta_raw      = median(z_after - z_before)  over fragile pairs
        score_aligned  = +delta_raw  if recall_fragility
                       = −delta_raw  if precision_collapse
        pass_level     from score_aligned (see GATE2_CEILING)
    """
    if not fragile_pairs:
        return {
            'delta_raw_median':         0.0,
            'score_aligned_median':     0.0,
            'delta_raw_ci_lower':       0.0,
            'delta_raw_ci_upper':       0.0,
            'score_aligned_ci_lower':   0.0,
            'score_aligned_ci_upper':   0.0,
            'pass_level':               'NULL',
            'n_fragile_pairs':          0,
            'failure_mode':             failure_mode,
            'warning':                  'No fragile pairs detected; score undefined',
        }

    deltas = np.array([
        z_after[f, t] - z_before[f, t]
        for f, t in fragile_pairs
    ])
    delta_raw_median = float(np.median(deltas))

    # score_aligned sign convention
    sign = +1.0 if failure_mode == 'recall_fragility' else -1.0
    score_aligned_median = sign * delta_raw_median

    # Bootstrap CI
    rng_ci = np.random.default_rng(ci_seed)
    boot_scores = []
    for _ in range(ci_bootstrap_B):
        idx = rng_ci.integers(0, len(deltas), size=len(deltas))
        boot_scores.append(sign * float(np.median(deltas[idx])))
    boot_scores = np.array(boot_scores)

    alpha_half = ci_alpha / 2.0
    ci_lower = float(np.quantile(boot_scores, alpha_half))
    ci_upper = float(np.quantile(boot_scores, 1 - alpha_half))

    # Pass level
    if ci_lower > GATE2_CEILING:
        pass_level = 'CEILING_BREAK'
    elif ci_lower > 0:
        pass_level = 'STRONG_PASS'
    elif score_aligned_median > 0:
        pass_level = 'SOFT_PASS'
    else:
        pass_level = 'NULL'

    return {
        'delta_raw_median':       delta_raw_median,
        'score_aligned_median':   score_aligned_median,
        'delta_raw_ci_lower':     ci_lower if failure_mode == 'recall_fragility' else -ci_upper,
        'delta_raw_ci_upper':     ci_upper if failure_mode == 'recall_fragility' else -ci_lower,
        'score_aligned_ci_lower': ci_lower,
        'score_aligned_ci_upper': ci_upper,
        'pass_level':             pass_level,
        'n_fragile_pairs':        len(fragile_pairs),
        'failure_mode':           failure_mode,
    }


# ============================================================
# GMM Sampler (Lynx-Hare specific: 2D state observations)
# ============================================================

class LynxHareGMMSampler:
    """
    GMM sampler for Lynx-Hare initial conditions.

    Design difference from Lorenz:
        Lorenz: GMM fitted on initial conditions [x0, y0, z0] from N trajectories.
        Lynx-Hare: GMM fitted on the 21 UNIQUE raw observations (H_t, L_t), t=0..20.
                   The raw series covers all phases of the population cycle
                   (rise 1900-1906, peak 1907-1910, fall 1911-1920).
                   Using training windows would over-represent 1900-1910 (rise phase only).
    """

    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, raw_observations: np.ndarray):
        """
        Fit GMM on raw unique state observations.

        Args:
            raw_observations: (T_raw, 2) array of [H_t, L_t], t=0..T_raw-1.
                              Must be the 21 unique real data points (NOT training windows).
                              Passed from get_lynxhare_data() in generate_pool().
        """
        if raw_observations.ndim != 2 or raw_observations.shape[1] != 2:
            raise ValueError(
                f"raw_observations must be (T, 2), got {raw_observations.shape}. "
                "Pass the full raw series (21 unique points), not training windows."
            )
        self.gmm.fit(raw_observations)
        self._fitted = True

    def sample_ics(self, n_samples: int, rng: np.random.Generator,
                   min_state: float = 0.5,
                   max_state: float = 300.0) -> np.ndarray:
        """
        Sample candidate initial conditions from fitted GMM.

        Args:
            n_samples: Number of ICs to sample
            rng: Random generator
            min_state: Minimum H or L value (physical: > 0)
            max_state: Maximum H or L value

        Returns:
            ics: (n_samples, 2) array of [H0, L0]
        """
        if not self._fitted:
            raise RuntimeError("GMM not fitted yet. Call fit() first.")

        ics = []
        attempts = 0
        while len(ics) < n_samples and attempts < n_samples * 50:
            attempts += 1
            # Sample from GMM and add small jitter
            sample, _ = self.gmm.sample(1)
            H0, L0 = float(sample[0, 0]), float(sample[0, 1])

            # Add small random perturbation (10% of magnitude)
            jitter_H = float(rng.normal(0, abs(H0) * 0.1 + 0.1))
            jitter_L = float(rng.normal(0, abs(L0) * 0.1 + 0.1))
            H0 += jitter_H
            L0 += jitter_L

            if H0 < min_state or L0 < min_state:
                continue
            if H0 > max_state or L0 > max_state:
                continue

            ics.append([H0, L0])

        if len(ics) < n_samples:
            raise RuntimeError(
                f"GMM sampler: could not generate {n_samples} valid ICs "
                f"(got {len(ics)} after {attempts} attempts)"
            )

        return np.array(ics[:n_samples])


# ============================================================
# Pool Generation
# ============================================================

def generate_pool(
    train_x: np.ndarray,
    lv_params: Dict[str, float],
    cfg: LynxHareGateConfig,
) -> Dict[str, Any]:
    """
    Generate augmentation pool using GMM + LV ODE teacher.

    GMM fitting source (SSOT):
        The GMM is fitted on the 21 UNIQUE raw observations from the
        full Hudson Bay series (1900-1920). NOT on the training windows,
        which would over-represent the 1900-1910 rising phase.
        Raw unique points are imported directly from get_lynxhare_data().

    Steps:
        1. Fit GMM on 21 raw unique state observations (full attractor coverage)
        2. Sample candidate ICs from GMM
        3. Simulate LV ODE from each IC for T_steps=window_size
        4. QC filter: accept physically valid trajectories
        5. Compute SHA256 of pool

    Args:
        train_x: (n_train, window_size, 2) training windows (NOT used for GMM fit)
        lv_params: Estimated LV parameters
        cfg: Configuration

    Returns:
        Dict with 'x' (N_pool, T, 2), 'dx' (N_pool, T, 2), 'sha',
                 'gmm_source' (provenance string)
    """
    # GMM fitted on raw unique observations — NOT on training windows
    from src.simulators.lynxhare_simulator import get_lynxhare_data
    raw = get_lynxhare_data()
    raw_obs = np.column_stack([raw['H'], raw['L']])   # (21, 2), all unique
    print(f"  Fitting GMM on {len(raw_obs)} raw unique observations (full 1900-1920 series)...")

    gmm_sampler = LynxHareGMMSampler(
        n_components=cfg.gmm_n_components,
        covariance_type=cfg.gmm_covariance_type,
        random_state=cfg.gmm_seed,
    )
    gmm_sampler.fit(raw_obs)   # (21, 2) — all unique points

    rng = np.random.default_rng(cfg.pool_seed)

    pool_x, pool_dx = [], []
    n_rejected = 0
    attempts = 0

    while len(pool_x) < cfg.pool_size and attempts < cfg.max_pool_attempts:
        attempts += 1

        # Sample IC from GMM
        ic = gmm_sampler.sample_ics(1, rng,
                                     min_state=cfg.qc_min_state,
                                     max_state=cfg.qc_max_state)
        H0, L0 = float(ic[0, 0]), float(ic[0, 1])

        # v1.2: Simulate LV for T_REAL=21 steps (matching real data length),
        # then apply SavGol and extract center window of window_size=7.
        # Training pipeline: SavGol on full 21-pt series → window to 7 pts.
        # Pool pipeline must match: simulate 21 → SavGol → center 7-pt window.
        x_full = simulate_lv_trajectory(
            H0, L0, lv_params,
            T_steps=T_REAL,      # 21 (full series length)
            dt=cfg.dt,
            max_state=cfg.qc_max_state,
        )
        if x_full is None:
            n_rejected += 1
            continue

        # QC on full trajectory
        if np.any(x_full < cfg.qc_min_state) or np.any(x_full > cfg.qc_max_state):
            n_rejected += 1
            continue

        # SavGol on full 21-pt trajectory (same params as training: w=5, p=3)
        dx_full = np.zeros_like(x_full)
        for s in range(2):
            dx_full[:, s] = savgol_filter(
                x_full[:, s],
                window_length=cfg.savgol_window,
                polyorder=cfg.savgol_polyorder,
                deriv=1,
                delta=cfg.dt,
            )

        if not np.all(np.isfinite(dx_full)):
            n_rejected += 1
            continue

        # Extract center window of window_size points (avoids edge effects)
        center_start = (T_REAL - cfg.T_steps) // 2   # (21-7)//2 = 7
        center_end = center_start + cfg.T_steps
        x_traj = x_full[center_start:center_end]     # (7, 2)
        dx_traj = dx_full[center_start:center_end]    # (7, 2)

        pool_x.append(x_traj)
        pool_dx.append(dx_traj)

    if len(pool_x) < cfg.pool_size:
        raise RuntimeError(
            f"Pool generation failed: got {len(pool_x)}/{cfg.pool_size} "
            f"after {attempts} attempts ({n_rejected} rejected)"
        )

    pool_x  = np.stack(pool_x[:cfg.pool_size], axis=0)   # (pool_size, T, 2)
    pool_dx = np.stack(pool_dx[:cfg.pool_size], axis=0)

    reject_rate = n_rejected / max(attempts, 1)
    print(f"  Pool: {len(pool_x)} trajectories "
          f"(reject_rate={reject_rate:.2%}, attempts={attempts})")

    # Compute SHA256
    pool_sha = hashlib.sha256(pool_x.tobytes()).hexdigest()[:16]
    print(f"  Pool SHA: {pool_sha}")

    return {
        'x':          pool_x,
        'dx':         pool_dx,
        'sha':        pool_sha,
        'n_attempts': attempts,
        'n_rejected': n_rejected,
        'reject_rate': reject_rate,
        'gmm_source': 'raw_21_unique_obs_1900-1920',  # provenance SSOT
    }


# ============================================================
# D-Optimal Selection
# ============================================================

def d_optimal_selection(
    train_x: np.ndarray,
    pool_x: np.ndarray,
    n_select: int,
    dopt_lambda: float = 1e-6,
    dopt_n_candidates: int = 200,
) -> np.ndarray:
    """
    D-optimal selection: maximize log det(X^T X) on combined library.

    Args:
        train_x: (N_tr, T, 2) training states
        pool_x: (N_pool, T, 2) pool states
        n_select: Number of pool trajectories to select
        dopt_lambda: Ridge regularization for numerical stability
        dopt_n_candidates: Pool indices to select from (confound-free)

    Returns:
        selected_indices: (n_select,) indices into pool_x
    """
    x_tr_flat = train_x.reshape(-1, 2)
    Theta_tr, _ = build_lynxhare_library(x_tr_flat)
    scaler = ColumnScaler()
    Theta_tr_scaled = scaler.fit_transform(Theta_tr)

    # Use up to dopt_n_candidates from pool
    n_cand = min(dopt_n_candidates, len(pool_x))
    cand_idx = np.arange(n_cand)

    # Current Gram matrix (training only)
    G = Theta_tr_scaled.T @ Theta_tr_scaled + dopt_lambda * np.eye(N_LYNXHARE_FEATURES)

    selected = []
    remaining = list(cand_idx)

    for _ in range(n_select):
        if not remaining:
            break

        best_idx = None
        best_logdet = -np.inf

        for ci in remaining:
            x_cand_flat = pool_x[ci].reshape(-1, 2)
            Theta_c, _ = build_lynxhare_library(x_cand_flat)
            Theta_c_scaled = scaler.transform(Theta_c)
            G_new = G + Theta_c_scaled.T @ Theta_c_scaled
            sign, ld = np.linalg.slogdet(G_new)
            if sign > 0 and ld > best_logdet:
                best_logdet = ld
                best_idx = ci

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        x_sel_flat = pool_x[best_idx].reshape(-1, 2)
        Theta_s, _ = build_lynxhare_library(x_sel_flat)
        Theta_s_scaled = scaler.transform(Theta_s)
        G = G + Theta_s_scaled.T @ Theta_s_scaled

    # If not enough, fill with random from remaining
    if len(selected) < n_select and remaining:
        needed = n_select - len(selected)
        extra = np.random.choice(remaining, size=min(needed, len(remaining)), replace=False)
        selected.extend(extra.tolist())

    return np.array(selected[:n_select])


# ============================================================
# Artifact Saving
# ============================================================

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_run_artifacts(
    run_dir: Path,
    run_id: str,
    method: str,
    sel_seed: int,
    cfg: LynxHareGateConfig,
    eval_result: Dict,
    metrics: Dict,
    pool_sha: str,
    baseline_dir: Path,
) -> None:
    """Save manifest.json, metrics.json, sindy_coefficients.csv."""
    # manifest.json
    manifest = {
        'run_id': run_id,
        'runner_version': RUNNER_VERSION,
        'system': SYSTEM,
        'gate': 'gate_lynxhare',
        'method': method,
        'selection_seed': sel_seed,
        'n_train': cfg.n_train,
        'n_select': cfg.n_select,
        'pool_sha': pool_sha,
        'pool_size': cfg.pool_size,
        'n_bootstrap': cfg.n_bootstrap,
        'threshold': cfg.threshold,
        'baseline_seed': cfg.baseline_seed,          # confound-free provenance
        'library_version': 'standard_polynomial_deg2',  # 6-term LV library
        'gmm_source': 'raw_21_unique_obs_1900-1920', # GMM fit provenance
        'window_size': cfg.window_size,
        'window_stride': cfg.window_stride,
        'baseline_dir': str(baseline_dir),
        'timestamp': datetime.now().isoformat(),
        'note': cfg.note,
    }
    with open(run_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=_json_default)

    # metrics.json
    metrics_out = {
        **metrics,
        'system': SYSTEM,
        'gate': 'gate_lynxhare',
        'method': method,
        'selection_seed': sel_seed,
        'pool_sha': pool_sha,
        'n_train': cfg.n_train,
        'n_bootstrap': cfg.n_bootstrap,
        'threshold': cfg.threshold,
        'kappa': eval_result.get('kappa'),
        'n_total_samples': eval_result.get('n_total_samples'),
        'n_original': eval_result.get('n_original'),
        'n_augmented': eval_result.get('n_augmented'),
        'n_traj_total': eval_result.get('n_traj_total'),
    }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2, default=_json_default)

    # sindy_coefficients.csv
    coeff = eval_result['coefficients_mean']
    feat_names = eval_result['feature_names']
    with open(run_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature'] + LYNXHARE_TARGET_NAMES)
        for i, name in enumerate(feat_names):
            writer.writerow([name] + [f'{coeff[i, j]:.6f}' for j in range(2)])


# ============================================================
# Main Gate Runner
# ============================================================

def run_lynxhare_gate(
    cfg: LynxHareGateConfig,
    seeds: List[int],
    phase: str = 'all',
) -> Dict:
    """
    Run full Lynx-Hare gate experiment.

    Phases:
        'baseline' — Phase 0+1 only
        'augment'  — Phase 2+3 (requires saved baseline)
        'all'      — All phases
    """
    print("=" * 70)
    print("LYNX-HARE GATE")
    print(f"  Runner:  {RUNNER_VERSION}")
    print(f"  Dataset: {cfg.dataset_version}")
    print(f"  Phase:   {phase}")
    print(f"  Seeds:   {seeds}")
    print("=" * 70)

    # ── AC2: Feature integrity check ────────────────────────────────────
    print("\n[AC2] Feature integrity check...")
    assert_lynxhare_feature_integrity()
    print("  ✅ AC2: Lynx-Hare feature integrity verified")

    # ── Phase 0: Dataset ────────────────────────────────────────────────
    print("\n[Phase 0] Dataset...")
    data = generate_or_load_dataset(cfg)
    validate_lynxhare_dataset(data)

    train_x  = data['train_x']   # (n_train, window_size, 2)
    train_dx = data['train_dx']  # (n_train, window_size, 2)

    lv_params = {
        'alpha': float(data['lv_alpha']),
        'beta':  float(data['lv_beta']),
        'gamma': float(data['lv_gamma']),
        'delta': float(data['lv_delta']),
    }
    print(f"  LV params: α={lv_params['alpha']:.4f}, β={lv_params['beta']:.4f}, "
          f"γ={lv_params['gamma']:.4f}, δ={lv_params['delta']:.4f}")

    if phase == 'baseline':
        seeds = []  # skip augmentation

    # ── Phase 1: Baseline ───────────────────────────────────────────────
    print("\n[Phase 1] E-SINDy baseline...")
    baseline_result = run_baseline(train_x, train_dx, cfg)

    fragile_pairs = baseline_result['fragile_pairs']
    failure_mode  = baseline_result['failure_mode']
    z_before      = baseline_result['z']

    if not fragile_pairs:
        print("  ⚠️  No fragile pairs detected at z_threshold="
              f"{cfg.z_fragile_threshold}. "
              "Consider lowering z_fragile_threshold or checking data quality.")

    # Save baseline
    run_id_bl = paths.generate_run_id('lynxhare_baseline')
    baseline_dir = paths.get_results_dir(
        cfg.dataset_version, 'gate_lynxhare', 'standardized',
        'esindy_baseline', cfg.n_train, cfg.baseline_seed, run_id_bl,
    )
    coeff_bl = baseline_result['coefficients_mean']
    feat_names = baseline_result['feature_names']
    with open(baseline_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature'] + LYNXHARE_TARGET_NAMES)
        for i, name in enumerate(feat_names):
            writer.writerow([name] + [f'{coeff_bl[i, j]:.6f}' for j in range(2)])

    baseline_meta = {
        'run_id': run_id_bl, 'kappa': baseline_result['kappa'],
        'failure_mode': failure_mode,
        'n_fragile_pairs': len(fragile_pairs),
        'n_oracle_fragile': baseline_result['n_oracle_fragile'],
        'n_spurious_fragile': baseline_result['n_spurious_fragile'],
        'fragile_pairs': fragile_pairs,
        'lv_params': lv_params,
        'timestamp': datetime.now().isoformat(),
    }
    with open(baseline_dir / 'baseline_meta.json', 'w') as f:
        json.dump(baseline_meta, f, indent=2, default=_json_default)

    # GPT P0 fix: z_before.npy + fragile_pairs.json + manifest.json (Lorenz-level audit trail)
    np.save(baseline_dir / 'z_before.npy', z_before)

    with open(baseline_dir / 'fragile_pairs.json', 'w') as f:
        json.dump({
            'fragile_pairs':      fragile_pairs,
            'n_pairs':            len(fragile_pairs),
            'failure_mode':       failure_mode,
            'n_oracle_fragile':   baseline_result['n_oracle_fragile'],
            'n_spurious_fragile': baseline_result['n_spurious_fragile'],
            'kappa':              baseline_result['kappa'],
            'z_fragile_threshold': cfg.z_fragile_threshold,
        }, f, indent=2, default=_json_default)

    with open(baseline_dir / 'manifest.json', 'w') as f:
        json.dump({
            'run_id':          run_id_bl,
            'system':          SYSTEM,
            'gate':            'gate_lynxhare',
            'method':          'esindy_baseline',
            'runner_version':  RUNNER_VERSION,
            'created_at':      datetime.now().isoformat(),
            'kappa':           baseline_result['kappa'],
            'failure_mode':    failure_mode,
            'n_fragile_pairs': len(fragile_pairs),
            'artifacts': [
                'manifest.json', 'z_before.npy',
                'sindy_coefficients.csv', 'fragile_pairs.json',
                'baseline_meta.json',
            ],
        }, f, indent=2, default=_json_default)

    print(f"  ✅ Baseline saved: {baseline_dir} (z_before.npy + fragile_pairs.json + manifest.json added)")

    if phase == 'baseline':
        return {'baseline': baseline_result}

    # ── Phase 2: Pool Generation ────────────────────────────────────────
    print("\n[Phase 2] GMM pool generation...")
    pool = generate_pool(train_x, lv_params, cfg)
    pool_sha = pool['sha']

    # ── Phase 3: Selection + Evaluation ─────────────────────────────────
    print("\n[Phase 3] D-optimal + Random selection...")
    all_results = {}

    # ── D-optimal ───────────────────────────────────────────────────────
    print("  Running D-optimal selection...")
    try:
        dopt_indices = d_optimal_selection(
            train_x, pool['x'],
            n_select=cfg.n_select,
            dopt_lambda=cfg.dopt_lambda,
            dopt_n_candidates=min(cfg.dopt_n_candidates, cfg.pool_size),
        )

        aug_x_dopt  = pool['x'][dopt_indices]
        aug_dx_dopt = pool['dx'][dopt_indices]

        eval_dopt = evaluate_with_esindy(
            train_x, aug_x_dopt, train_dx, aug_dx_dopt,
            cfg.n_bootstrap, cfg.threshold, cfg.z_eps, seed=0,
        )
        metrics_dopt = compute_metrics(
            z_after=eval_dopt['z'], z_before=z_before,
            fragile_pairs=fragile_pairs, failure_mode=failure_mode,
            ci_bootstrap_B=cfg.ci_bootstrap_B, ci_alpha=cfg.ci_alpha, ci_seed=0,
        )

        run_id_dopt = paths.generate_run_id('lynxhare_dopt')
        run_dir_dopt = paths.get_results_dir(
            cfg.dataset_version, 'gate_lynxhare', 'standardized',
            'esindy_dopt', cfg.n_train, 0, run_id_dopt,
        )
        save_run_artifacts(run_dir_dopt, run_id_dopt, 'd_optimal', 0,
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
        print(f"  D-optimal: score_aligned={metrics_dopt['score_aligned_median']:.3f}, "
              f"pass={metrics_dopt['pass_level']}")

    except Exception as e:
        print(f"  ❌ D-optimal failed: {e}")
        traceback.print_exc()
        all_results['d_optimal'] = {'status': 'failed', 'error': str(e)}

    # ── Random ──────────────────────────────────────────────────────────
    for sel_seed in seeds:
        print(f"  Running Random seed={sel_seed}...")
        try:
            rng_sel = np.random.default_rng(sel_seed + 1000)
            n_avail = len(pool['x'])
            candidate_indices = np.arange(n_avail)

            if len(candidate_indices) >= cfg.n_select:
                chosen = rng_sel.choice(n_avail, size=cfg.n_select, replace=False)
                rand_indices = np.sort(chosen)
            else:
                rand_indices = candidate_indices

            aug_x_rand  = pool['x'][rand_indices]
            aug_dx_rand = pool['dx'][rand_indices]

            eval_rand = evaluate_with_esindy(
                train_x, aug_x_rand, train_dx, aug_dx_rand,
                cfg.n_bootstrap, cfg.threshold, cfg.z_eps, seed=sel_seed,
            )
            metrics_rand = compute_metrics(
                z_after=eval_rand['z'], z_before=z_before,
                fragile_pairs=fragile_pairs, failure_mode=failure_mode,
                ci_bootstrap_B=cfg.ci_bootstrap_B, ci_alpha=cfg.ci_alpha,
                ci_seed=sel_seed,
            )

            run_id_rand = paths.generate_run_id(f'lynxhare_random_s{sel_seed}')
            run_dir_rand = paths.get_results_dir(
                cfg.dataset_version, 'gate_lynxhare', 'standardized',
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
    print("LYNX-HARE GATE SUMMARY")
    print("=" * 70)
    print(f"  Baseline κ:    {baseline_result['kappa']:.3e}")
    print(f"  Failure mode:  {failure_mode}")
    print(f"  Fragile pairs: {len(fragile_pairs)}")
    print(f"  Pool SHA:      {pool_sha}")
    print()

    completed_random = [
        v for k, v in all_results.items()
        if k.startswith('random_') and v.get('status') == 'completed'
    ]
    if completed_random:
        sa_vals = [r['score_aligned'] for r in completed_random
                   if r['score_aligned'] is not None]
        pass_levels = [r['pass_level'] for r in completed_random]
        pc = Counter(pass_levels)
        print(f"  Random ({len(completed_random)} seeds):")
        print(f"    median score_aligned = {float(np.median(sa_vals)):.3f}")
        print(f"    NULL:{pc.get('NULL',0)}  SOFT:{pc.get('SOFT_PASS',0)}  "
              f"STRONG:{pc.get('STRONG_PASS',0)}  CEILING:{pc.get('CEILING_BREAK',0)}")

    if 'd_optimal' in all_results and all_results['d_optimal'].get('status') == 'completed':
        dr = all_results['d_optimal']
        print(f"  D-optimal:")
        print(f"    score_aligned = {dr['score_aligned']:.3f}, "
              f"CI=[{dr['ci_lower']:.3f}, {dr['ci_upper']:.3f}]")
        print(f"    pass_level = {dr['pass_level']}")

    # ── Context Packet ──────────────────────────────────────────────────
    run_id_cp = paths.generate_run_id('lynxhare_gate_cp')
    cp_path = paths.get_context_packet_path(run_id_cp)
    _write_context_packet(cp_path, baseline_result, all_results, pool_sha, cfg, seeds)
    print(f"\n  ✅ Context Packet: {cp_path}")

    # ── Summary JSON ────────────────────────────────────────────────────
    summary_path = (paths.RESULTS_ROOT / cfg.dataset_version
                    / 'gate_lynxhare' / 'summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'runner_version': RUNNER_VERSION,
        'baseline_kappa': baseline_result['kappa'],
        'failure_mode': failure_mode,
        'n_fragile_pairs': len(fragile_pairs),
        'pool_sha': pool_sha,
        'n_train': cfg.n_train,
        'lv_params': lv_params,
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
    cfg: LynxHareGateConfig,
    seeds: List[int],
) -> None:
    """Write Context Packet markdown."""
    completed_random = [
        v for k, v in all_results.items()
        if k.startswith('random_') and v.get('status') == 'completed'
    ]
    sa_values = [r['score_aligned'] for r in completed_random
                 if r['score_aligned'] is not None]
    pass_levels = [r['pass_level'] for r in completed_random]
    dopt = all_results.get('d_optimal', {})

    lines = [
        f"# Context Packet: Lynx-Hare Gate",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runner**: run_lynxhare_gate.py {RUNNER_VERSION}",
        f"",
        f"## Baseline",
        f"- κ: {baseline_result['kappa']:.3e}",
        f"- Failure mode: **{baseline_result['failure_mode']}**",
        f"- Oracle fragile: {baseline_result['n_oracle_fragile']}",
        f"- Spurious fragile: {baseline_result['n_spurious_fragile']}",
        f"- Total fragile pairs: {len(baseline_result['fragile_pairs'])}",
        f"- n_train: {cfg.n_train} (sliding windows: window={cfg.window_size}, stride={cfg.window_stride})",
        f"",
        f"## Pool",
        f"- SHA: `{pool_sha}`",
        f"- Size: {cfg.pool_size}",
        f"- GMM: fitted on 21 raw unique observations (full 1900-1920, all attractor phases)",
        f"",
        f"## Random Results ({len(completed_random)} seeds)",
    ]
    if sa_values:
        from collections import Counter
        pc = Counter(pass_levels)
        lines += [
            f"- Median score_aligned: {float(np.median(sa_values)):.3f}",
            f"- NULL:{pc.get('NULL',0)}, SOFT:{pc.get('SOFT_PASS',0)}, "
            f"STRONG:{pc.get('STRONG_PASS',0)}, CEILING:{pc.get('CEILING_BREAK',0)}",
        ]
    lines += [
        f"",
        f"## D-optimal",
        f"- Status: {dopt.get('status', 'N/A')}",
        f"- score_aligned: {dopt.get('score_aligned', 'N/A')}",
        f"- CI: [{dopt.get('ci_lower','N/A')}, {dopt.get('ci_upper','N/A')}]",
        f"- Pass level: {dopt.get('pass_level', 'N/A')}",
        f"",
        f"## SSOT Notes",
        f"- score_aligned direction: runtime-detected ({baseline_result['failure_mode']})",
        f"- gate2_ceiling = {GATE2_CEILING}",
        f"- Library: standard polynomial degree-2 (no reparameterization)",
        f"- Data: Hudson Bay fur trade 1900-1920 (same as Fasel et al. 2022)",
    ]
    with open(cp_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Lynx-Hare Gate: GMM Augmentation + D-opt vs Random'
    )
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'baseline', 'augment'])
    parser.add_argument('--pool_size', type=int, default=200)
    parser.add_argument('--n_select', type=int, default=50)
    parser.add_argument('--baseline_seed', type=int, default=1)
    parser.add_argument('--note', type=str, default='lynxhare_gate')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = LynxHareGateConfig(
        pool_size=args.pool_size,
        n_select=args.n_select,
        baseline_seed=args.baseline_seed,
        random_seeds=args.seeds,
        note=args.note,
    )
    summary = run_lynxhare_gate(cfg, seeds=args.seeds, phase=args.phase)
    print("\n✅ Lynx-Hare Gate complete.")