"""
AEK-3: E-SINDy Baseline Runner for AEK Self-balancing Motorcycle

Purpose:
    Establish E-SINDy baseline on AEK dataset (no augmentation).
    Produces teacher_support, fragile_pairs, z_before for future Gate3/4 use.

Design:
    - Uses AEK-specific library (14 terms from aek.yaml, post-GPT review)
    - Supports Standard and Reparam-1 (cos(phi)→cos(phi)-1) libraries
    - Analytic dx from dataset (no Savitzky-Golay needed)
    - Threshold sweep on val split to select optimal threshold
    - Oracle comparison for precision/recall/fragile pair identification

Output structure (SSOT paths.py):
    results/aek_ood_v1/gate1/standardized/esindy/n10/seed{S}/{run_id}/
        manifest.json
        metrics.json
        sindy_coefficients.csv
        teacher_support.npy
        stable_core_mask.npy
        fragile_pairs.json
        z_before.npy
        oracle_coefficients.npy
        figures/
            F01_coefficient_heatmap.png/.pdf
            F02_support_comparison.png/.pdf

Usage (PowerShell, copy-paste ready):
    python experiments/run_aek3_baseline.py --seed 0
    python experiments/run_aek3_baseline.py --seed 1
    python experiments/run_aek3_baseline.py --seed 0 --reparam reparam1
    python experiments/run_aek3_baseline.py --seed 0 --n_bootstrap 100

Author: Claude (AEK-3)
Date: 2026-02-07
Updated: 2026-02-10 (Gate4c Reparam-1 support)
"""

import sys
from pathlib import Path

# Project root (experiments/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import yaml
import numpy as np
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Project SSOT
from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.contracts.plot_style import (
    create_figure, save_figure, setup_style, get_color, COLORS
)

# SINDy modules
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble

# AEK library (updated with Reparam-1 support)
from src.sindy.aek_library import (
    build_aek_library,
    build_aek_library_by_name,
    get_aek_feature_names,
    get_aek_oracle_support,
    get_aek_oracle_coefficients,
    AEK_FEATURE_NAMES,
    AEK_REPARAM1_FEATURE_NAMES,
    AEK_TARGET_NAMES,
    N_AEK_FEATURES,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class AEK3Config:
    """AEK-3 E-SINDy Baseline Configuration."""
    # Dataset
    dataset_version: str = 'aek_ood_v1'
    system: str = 'aek'
    n_train: int = 10

    # E-SINDy
    n_bootstrap: int = 100
    threshold_candidates: Tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5
    )
    default_threshold: float = 0.05

    # Z-metric
    z_eps: float = 1e-6

    # Seed
    seed: int = 0

    # YAML path
    config_path: str = 'configs/systems/aek.yaml'

    # Output
    gate: str = 'gate1'
    track: str = 'standardized'
    method: str = 'esindy'
    note: str = 'aek3_baseline'

    # Library parameterization: 'standard' or 'reparam1'
    reparam: str = 'standard'


# ============================================================
# Utility Functions
# ============================================================

def flatten_trajectories(
    x: np.ndarray, u: np.ndarray, dx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Flatten (N, T, D) trajectory arrays to (N*T, D).

    Returns:
        x_flat, u_flat, dx_flat, n_traj, T
    """
    n_traj, T, state_dim = x.shape
    x_flat = x.reshape(-1, state_dim)
    u_flat = u.reshape(-1, u.shape[-1])
    dx_flat = dx.reshape(-1, state_dim)
    return x_flat, u_flat, dx_flat, n_traj, T


def compute_z_metric(
    coefficients_mean: np.ndarray,
    coefficients_std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute z-metric: |mean| / (std + eps).

    Args:
        coefficients_mean: shape (n_features, n_targets)
        coefficients_std: shape (n_features, n_targets)
        eps: numerical stability

    Returns:
        z: shape (n_features, n_targets)
    """
    return np.abs(coefficients_mean) / (coefficients_std + eps)


def identify_fragile_pairs(
    support_mask: np.ndarray,
    oracle_support: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """
    Identify stable-core and fragile terms by comparing with oracle.

    Definitions:
        stable_core: terms where E-SINDy and oracle AGREE (both on or both off)
        fragile: terms where they DISAGREE
            - fragile_dynamics: oracle ON but E-SINDy OFF (recall failure)
            - fragile_spurious: oracle OFF but E-SINDy ON (precision failure)

    Args:
        support_mask: E-SINDy support, shape (n_features, n_targets)
        oracle_support: Oracle support, shape (n_features, n_targets)
        feature_names: List of feature name strings (Standard or Reparam-1)

    Returns:
        Dict with masks and pair lists
    """
    n_features, n_targets = support_mask.shape

    # Stable core: agreement between E-SINDy and oracle
    stable_core = support_mask == oracle_support

    # Fragile: disagreement
    fragile = ~stable_core

    # Sub-classify
    fragile_dynamics = oracle_support & ~support_mask      # missed real terms
    fragile_spurious = ~oracle_support & support_mask       # false positives

    # Build fragile pairs list
    pairs = []
    for f_idx in range(n_features):
        for t_idx in range(n_targets):
            if fragile_dynamics[f_idx, t_idx]:
                pairs.append({
                    'feature_idx': int(f_idx),
                    'target_idx': int(t_idx),
                    'feature_name': feature_names[f_idx],
                    'target_name': AEK_TARGET_NAMES[t_idx],
                    'type': 'dynamics',
                })
            elif fragile_spurious[f_idx, t_idx]:
                pairs.append({
                    'feature_idx': int(f_idx),
                    'target_idx': int(t_idx),
                    'feature_name': feature_names[f_idx],
                    'target_name': AEK_TARGET_NAMES[t_idx],
                    'type': 'spurious',
                })

    return {
        'stable_core_mask': stable_core,
        'fragile_mask': fragile,
        'fragile_dynamics_mask': fragile_dynamics,
        'fragile_spurious_mask': fragile_spurious,
        'fragile_pairs': pairs,
        'n_stable': int(stable_core.sum()),
        'n_fragile': int(fragile.sum()),
        'n_fragile_dynamics': int(fragile_dynamics.sum()),
        'n_fragile_spurious': int(fragile_spurious.sum()),
    }


# ============================================================
# AEK-3 Runner
# ============================================================

class AEK3Runner:
    """AEK-3 E-SINDy Baseline Runner."""

    def __init__(self, config: AEK3Config):
        self.cfg = config
        self.dataset = None
        self.aek_config = None
        self.results_dir = None
        self.run_id = None
        # Feature names for current parameterization
        self._feature_names = get_aek_feature_names(config.reparam)

    def run(self) -> Dict:
        """Execute full AEK-3 baseline pipeline."""
        print("=" * 70)
        print("AEK-3: E-SINDy Baseline for AEK Self-balancing Motorcycle")
        print(f"  Library: {self.cfg.reparam}")
        print("=" * 70)

        # Phase 0: Setup
        self._setup()

        # Phase 1: Load data
        self._load_data()

        # Phase 2: Build library + fit E-SINDy
        results = self._fit_esindy()

        # Phase 3: Oracle comparison + fragile pairs
        analysis = self._analyze_results(results)

        # Phase 4: Save outputs
        self._save_outputs(results, analysis)

        # Phase 5: Generate figures
        self._generate_figures(results, analysis)

        # Phase 6: Context packet
        self._save_context_packet(results, analysis)

        print(f"\n{'='*70}")
        print("AEK-3 Baseline Complete!")
        print(f"{'='*70}")
        print(f"  Results: {self.results_dir}")
        print(f"  Run ID:  {self.run_id}")

        return {
            'status': 'success',
            'run_id': self.run_id,
            'results_dir': str(self.results_dir),
            **analysis['summary'],
        }

    # ----------------------------------------------------------
    # Phase 0: Setup
    # ----------------------------------------------------------
    def _setup(self):
        """Validate dataset and create output directory."""
        cfg = self.cfg

        # Validate dataset
        dataset_path = paths.get_dataset_path(cfg.dataset_version, system=cfg.system)
        print(f"\n[Phase 0] Setup")
        print(f"  Dataset: {dataset_path}")

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        validate_dataset_lite(dataset_path)
        print(f"  Preflight: PASS")

        # Load AEK config
        config_path = paths.ROOT / cfg.config_path
        if not config_path.exists():
            raise FileNotFoundError(f"AEK config not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.aek_config = yaml.safe_load(f)
        print(f"  Config: {config_path}")
        print(f"  Reparam: {cfg.reparam}")

        # Create run_id and results dir
        self.run_id = paths.generate_run_id(cfg.note)
        self.results_dir = paths.get_results_dir(
            dataset_version=cfg.dataset_version,
            gate=cfg.gate,
            track=cfg.track,
            method=cfg.method,
            n_train=cfg.n_train,
            seed=cfg.seed,
            run_id=self.run_id,
        )
        print(f"  Run ID: {self.run_id}")
        print(f"  Output: {self.results_dir}")

    # ----------------------------------------------------------
    # Phase 1: Load data
    # ----------------------------------------------------------
    def _load_data(self):
        """Load AEK dataset."""
        print(f"\n[Phase 1] Loading dataset...")
        dataset_path = paths.get_dataset_path(
            self.cfg.dataset_version, system=self.cfg.system
        )
        self.dataset = dict(np.load(dataset_path, allow_pickle=True))

        for split in ['train', 'val', 'test']:
            shape = self.dataset[f'{split}_x'].shape
            print(f"  {split}: {shape}")

    # ----------------------------------------------------------
    # Phase 2: Fit E-SINDy
    # ----------------------------------------------------------
    def _fit_esindy(self) -> Dict:
        """Build library and fit E-SINDy on training data."""
        print(f"\n[Phase 2] E-SINDy fitting...")
        cfg = self.cfg

        # Flatten training data
        train_x = self.dataset['train_x']              # §3.3: clean x + SavGol dx
        train_u = self.dataset['train_u']
        train_dx = self.dataset['train_dx_savgol']    # §3.3: SavGol consistency
        x_flat, u_flat, dx_flat, n_traj, T = flatten_trajectories(
            train_x, train_u, train_dx
        )
        print(f"  Flattened: {x_flat.shape[0]} samples "
              f"({n_traj} traj x {T} steps)")

        # Build AEK library (standard or reparam1)
        Theta, feature_names = build_aek_library_by_name(
            x_flat, u_flat, reparam=cfg.reparam
        )
        self._feature_names = feature_names  # cache for save/figures
        print(f"  Library: {Theta.shape[1]} features ({cfg.reparam})")

        # Scale features
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        col_stds = Theta_scaled.std(axis=0)
        print(f"  Scaled: column std range "
              f"[{col_stds.min():.4f}, {col_stds.max():.4f}]")

        # Prepare val data for threshold sweep
        val_x_flat, val_u_flat, val_dx_flat, n_val, T_val = flatten_trajectories(
            self.dataset['val_x'], self.dataset['val_u'], self.dataset['val_dx_savgol']
        )
        Theta_val, _ = build_aek_library_by_name(
            val_x_flat, val_u_flat, reparam=cfg.reparam
        )
        Theta_val_scaled = scaler.transform(Theta_val)

        # Threshold sweep (R²-based, per-target normalized)
        # Why R² not MSE: AEK targets span 8 orders of magnitude in variance,
        # so raw MSE is dominated by theta_w_ddot. Per-target R² normalizes this.
        print(f"\n  Threshold sweep ({len(cfg.threshold_candidates)} candidates)...")
        best_threshold = cfg.default_threshold
        best_val_score = -np.inf
        sweep_results = []

        for thr in cfg.threshold_candidates:
            ens = ESINDyEnsemble(
                n_bootstrap=cfg.n_bootstrap,
                threshold=thr,
                random_state=cfg.seed,
            )
            ens.fit(
                Theta_scaled, dx_flat,
                n_trajectories=n_traj, T=T,
                scaler=scaler, target_scale=None,
            )

            # Val prediction: convert unscaled coeff to scaled space
            # Theta_scaled[:, i] = Theta_raw[:, i] / scale[i]
            # => coeff_scaled[i, :] = coeff_unscaled[i, :] * scale[i]
            coeff_unscaled = ens.coefficients_mean_
            scale = scaler.scale_
            coeff_scaled = coeff_unscaled * scale[:, None]
            dx_pred = Theta_val_scaled @ coeff_scaled

            # Per-target R²
            ss_res = np.sum((val_dx_flat - dx_pred) ** 2, axis=0)  # (4,)
            ss_tot = np.sum(
                (val_dx_flat - val_dx_flat.mean(axis=0)) ** 2, axis=0
            )  # (4,)
            # Guard: if ss_tot ~ 0 (constant target), set R²=1 if ss_res~0
            r2_per_target = np.where(
                ss_tot > 1e-30,
                1.0 - ss_res / ss_tot,
                np.where(ss_res < 1e-30, 1.0, -np.inf),
            )
            val_r2_mean = float(np.mean(r2_per_target))
            val_mse = float(np.mean((dx_pred - val_dx_flat) ** 2))

            # Sparsity (fraction of zero terms)
            support = np.abs(coeff_unscaled) > 0
            sparsity = 1.0 - support.sum() / support.size

            # Score: mean R² (higher = better), sparsity tiebreak
            score = val_r2_mean + 0.01 * sparsity

            sweep_results.append({
                'threshold': float(thr),
                'val_mse': val_mse,
                'val_r2_mean': float(val_r2_mean),
                'val_r2_per_target': [float(r) for r in r2_per_target],
                'sparsity': float(sparsity),
                'n_active': int(support.sum()),
                'score': float(score),
            })

            if score > best_val_score:
                best_val_score = score
                best_threshold = thr

            r2_str = ', '.join(f'{r:.4f}' for r in r2_per_target)
            print(f"    thr={thr:.3f}: R²_mean={val_r2_mean:.4f} "
                  f"[{r2_str}], "
                  f"active={int(support.sum())}/{support.size}")

        print(f"\n  Selected threshold: {best_threshold}")

        # Final fit with best threshold
        ensemble = ESINDyEnsemble(
            n_bootstrap=cfg.n_bootstrap,
            threshold=best_threshold,
            random_state=cfg.seed,
        )
        ensemble.fit(
            Theta_scaled, dx_flat,
            n_trajectories=n_traj, T=T,
            scaler=scaler, target_scale=None,
        )

        # Extract results
        coeff_mean = ensemble.coefficients_mean_   # unscaled
        coeff_std = ensemble.coefficients_std_
        inclusion_prob = ensemble.inclusion_probability_
        support_mask = np.abs(coeff_mean) > 0

        print(f"\n  Final E-SINDy results:")
        print(f"    Active terms: {support_mask.sum()}/{support_mask.size}")
        print(f"    Coeff range: [{coeff_mean.min():.6f}, {coeff_mean.max():.6f}]")

        return {
            'ensemble': ensemble,
            'scaler': scaler,
            'feature_names': feature_names,
            'coefficients_mean': coeff_mean,
            'coefficients_std': coeff_std,
            'inclusion_probability': inclusion_prob,
            'support_mask': support_mask,
            'best_threshold': best_threshold,
            'sweep_results': sweep_results,
            'n_traj': n_traj,
            'T': T,
            'Theta_raw': Theta,              # for condition number
            'Theta_scaled': Theta_scaled,     # for condition number
        }

    # ----------------------------------------------------------
    # Phase 3: Oracle comparison + fragile pairs
    # ----------------------------------------------------------
    def _analyze_results(self, results: Dict) -> Dict:
        """Compare E-SINDy results with oracle."""
        print(f"\n[Phase 3] Oracle comparison...")

        coeff_mean = results['coefficients_mean']
        coeff_std = results['coefficients_std']
        support_mask = results['support_mask']

        # Oracle support (same for Standard and Reparam-1)
        oracle_support = get_aek_oracle_support()

        # Oracle coefficients (nominal parameters from YAML)
        phys = self.aek_config
        oracle_coeff = get_aek_oracle_coefficients(
            M_total=phys['dynamics']['M_total_kg'],
            g=phys['physics']['g'],
            h_cm=phys['dynamics']['h_cm_m'],
            I_p=phys['inertia']['I_p'],
            I_w_C=phys['inertia']['I_w_C'],
        )

        # Print oracle coefficients for verification
        print(f"\n  Oracle coefficients (nominal):")
        print(f"    phi_ddot:     sin(phi) = {oracle_coeff[4,1]:.4f}, "
              f"tau = {oracle_coeff[3,1]:.4f}")
        print(f"    theta_w_ddot: sin(phi) = {oracle_coeff[4,3]:.4f}, "
              f"tau = {oracle_coeff[3,3]:.4f}")

        # Fragile pairs (use active feature names)
        fragile_info = identify_fragile_pairs(
            support_mask, oracle_support, self._feature_names
        )

        print(f"\n  Oracle active terms: {oracle_support.sum()}")
        print(f"  E-SINDy active terms: {support_mask.sum()}")
        print(f"  Stable core: {fragile_info['n_stable']}")
        print(f"  Fragile total: {fragile_info['n_fragile']}")
        print(f"    - dynamics (recall failure): "
              f"{fragile_info['n_fragile_dynamics']}")
        print(f"    - spurious (precision failure): "
              f"{fragile_info['n_fragile_spurious']}")

        # Z-metric
        z = compute_z_metric(coeff_mean, coeff_std, eps=self.cfg.z_eps)

        # z_before: z-metric values at fragile pairs
        fragile_mask = fragile_info['fragile_mask']
        z_before = z[fragile_mask]

        print(f"\n  Z-metric (fragile pairs):")
        if len(z_before) > 0:
            print(f"    count: {len(z_before)}")
            print(f"    median: {np.median(z_before):.3f}")
            print(f"    mean: {np.mean(z_before):.3f}")
            print(f"    range: [{z_before.min():.3f}, {z_before.max():.3f}]")
        else:
            print(f"    No fragile pairs (perfect oracle agreement)")

        # Precision / Recall per target
        precision_recall = []
        for t_idx in range(4):
            esindy_on = support_mask[:, t_idx].sum()
            oracle_on = oracle_support[:, t_idx].sum()
            true_pos = (support_mask[:, t_idx] & oracle_support[:, t_idx]).sum()

            precision = true_pos / max(esindy_on, 1)
            recall = true_pos / max(oracle_on, 1)

            precision_recall.append({
                'target': AEK_TARGET_NAMES[t_idx],
                'target_idx': int(t_idx),
                'esindy_active': int(esindy_on),
                'oracle_active': int(oracle_on),
                'true_positive': int(true_pos),
                'precision': float(precision),
                'recall': float(recall),
            })
            print(f"\n  Target {t_idx} ({AEK_TARGET_NAMES[t_idx]}):")
            print(f"    E-SINDy={esindy_on}, Oracle={oracle_on}, TP={true_pos}")
            print(f"    Precision={precision:.2f}, Recall={recall:.2f}")

        # ---- GPT P0-B/P0-C/P1 diagnostics ----

        # Feature column stds (raw and scaled)
        Theta_raw = results['Theta_raw']
        Theta_scaled = results['Theta_scaled']
        feature_stds_raw = Theta_raw.std(axis=0).tolist()
        feature_stds_scaled = Theta_scaled.std(axis=0).tolist()
        scaler_scale = results['scaler'].scale_.tolist()

        # Condition number (P0-C defense)
        cond_full = float(np.linalg.cond(Theta_scaled))
        # Also without constant column (idx 0) for cleaner diagnostic
        cond_no_const = float(np.linalg.cond(Theta_scaled[:, 1:]))
        print(f"\n  Condition number (Theta_scaled): {cond_full:.1f}")
        print(f"  Condition number (no constant):  {cond_no_const:.1f}")

        # Oracle recovery error (P1: coefficient accuracy at oracle-active positions)
        oracle_active_mask = oracle_support  # (14, 4) bool
        oracle_recovery = {}
        if oracle_active_mask.sum() > 0:
            coeff_at_oracle = coeff_mean[oracle_active_mask]
            oracle_at_oracle = oracle_coeff[oracle_active_mask]
            abs_error = np.abs(coeff_at_oracle - oracle_at_oracle)
            rel_error = abs_error / (np.abs(oracle_at_oracle) + 1e-12)
            oracle_recovery = {
                'max_abs_error': float(abs_error.max()),
                'mean_abs_error': float(abs_error.mean()),
                'max_rel_error': float(rel_error.max()),
                'mean_rel_error': float(rel_error.mean()),
                'per_term': [],
            }
            for f_idx in range(N_AEK_FEATURES):
                for t_idx in range(4):
                    if oracle_active_mask[f_idx, t_idx]:
                        oracle_recovery['per_term'].append({
                            'feature': self._feature_names[f_idx],
                            'target': AEK_TARGET_NAMES[t_idx],
                            'oracle_coeff': float(oracle_coeff[f_idx, t_idx]),
                            'esindy_coeff': float(coeff_mean[f_idx, t_idx]),
                            'abs_error': float(abs(coeff_mean[f_idx, t_idx]
                                                    - oracle_coeff[f_idx, t_idx])),
                            'rel_error': float(abs(coeff_mean[f_idx, t_idx]
                                                    - oracle_coeff[f_idx, t_idx])
                                               / (abs(oracle_coeff[f_idx, t_idx]) + 1e-12)),
                        })
            print(f"\n  Oracle recovery error:")
            print(f"    mean |error|: {oracle_recovery['mean_abs_error']:.4f}")
            print(f"    mean rel error: {oracle_recovery['mean_rel_error']:.4f}")
            for term in oracle_recovery['per_term']:
                print(f"    {term['feature']:>15s} → {term['target']}: "
                      f"oracle={term['oracle_coeff']:.4f}, "
                      f"esindy={term['esindy_coeff']:.4f}, "
                      f"rel_err={term['rel_error']:.4f}")

        # Summary dict
        summary = {
            'n_fragile': fragile_info['n_fragile'],
            'n_fragile_dynamics': fragile_info['n_fragile_dynamics'],
            'n_fragile_spurious': fragile_info['n_fragile_spurious'],
            'n_stable_core': fragile_info['n_stable'],
            'z_before_median': float(np.median(z_before)) if len(z_before) > 0 else None,
            'z_before_mean': float(np.mean(z_before)) if len(z_before) > 0 else None,
            'best_threshold': float(results['best_threshold']),
            'condition_number_theta_scaled': cond_full,
            'condition_number_no_constant': cond_no_const,
            'oracle_recovery': oracle_recovery,
        }

        return {
            'oracle_support': oracle_support,
            'oracle_coeff': oracle_coeff,
            'fragile_info': fragile_info,
            'z': z,
            'z_before': z_before,
            'precision_recall': precision_recall,
            'summary': summary,
            'feature_stds_raw': feature_stds_raw,
            'feature_stds_scaled': feature_stds_scaled,
            'scaler_scale': scaler_scale,
        }

    # ----------------------------------------------------------
    # Phase 4: Save outputs
    # ----------------------------------------------------------
    def _save_outputs(self, results: Dict, analysis: Dict):
        """Save all required outputs."""
        print(f"\n[Phase 4] Saving outputs...")
        rd = self.results_dir

        # 1. sindy_coefficients.csv (UNSCALED, physical units)
        coeff = results['coefficients_mean']
        csv_path = rd / 'sindy_coefficients.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['feature'] + AEK_TARGET_NAMES
            writer.writerow(header)
            for i, name in enumerate(self._feature_names):
                row = [name] + [f"{coeff[i, j]:.8f}" for j in range(4)]
                writer.writerow(row)
        print(f"  sindy_coefficients.csv (unscaled)")

        # 1b. sindy_coefficients_scaled.csv (SCALED, for reviewer defense)
        scale = results['scaler'].scale_
        coeff_scaled = coeff * scale[:, None]
        csv_scaled_path = rd / 'sindy_coefficients_scaled.csv'
        with open(csv_scaled_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['feature'] + AEK_TARGET_NAMES
            writer.writerow(header)
            for i, name in enumerate(self._feature_names):
                row = [name] + [f"{coeff_scaled[i, j]:.8f}" for j in range(4)]
                writer.writerow(row)
        print(f"  sindy_coefficients_scaled.csv (scaled)")

        # 2. teacher_support.npy
        np.save(rd / 'teacher_support.npy', results['support_mask'])
        print(f"  teacher_support.npy ({results['support_mask'].shape})")

        # 3. stable_core_mask.npy
        np.save(rd / 'stable_core_mask.npy',
                analysis['fragile_info']['stable_core_mask'])
        print(f"  stable_core_mask.npy")

        # 4. z_before.npy
        np.save(rd / 'z_before.npy', analysis['z_before'])
        print(f"  z_before.npy ({len(analysis['z_before'])} fragile pairs)")

        # 5. oracle_coefficients.npy
        np.save(rd / 'oracle_coefficients.npy', analysis['oracle_coeff'])
        print(f"  oracle_coefficients.npy")

        # 6. fragile_pairs.json
        fp_data = {
            'system': 'aek',
            'reparam': self.cfg.reparam,
            'n_features': N_AEK_FEATURES,
            'n_targets': 4,
            'feature_names': list(self._feature_names),
            'target_names': list(AEK_TARGET_NAMES),
            'n_fragile': analysis['fragile_info']['n_fragile'],
            'n_fragile_dynamics': analysis['fragile_info']['n_fragile_dynamics'],
            'n_fragile_spurious': analysis['fragile_info']['n_fragile_spurious'],
            'pairs': analysis['fragile_info']['fragile_pairs'],
        }
        with open(rd / 'fragile_pairs.json', 'w', encoding='utf-8') as f:
            json.dump(fp_data, f, indent=2, ensure_ascii=False)
        print(f"  fragile_pairs.json")

        # 7. metrics.json
        metrics = {
            'system': 'aek',
            'gate': 'gate1',
            'seed': self.cfg.seed,
            'reparam': self.cfg.reparam,
            'n_train': self.cfg.n_train,
            'n_bootstrap': self.cfg.n_bootstrap,
            'best_threshold': float(results['best_threshold']),
            'n_library_terms': N_AEK_FEATURES,
            'n_active_terms': int(results['support_mask'].sum()),
            'n_oracle_terms': int(analysis['oracle_support'].sum()),
            **{k: v for k, v in analysis['summary'].items()
               if k != 'oracle_recovery'},
            'precision_recall': analysis['precision_recall'],
            'threshold_sweep': results['sweep_results'],
            # GPT P0-B: feature scale diagnostics
            'diagnostics': {
                'feature_column_stds_raw': dict(zip(
                    self._feature_names, analysis['feature_stds_raw'])),
                'feature_column_stds_scaled': dict(zip(
                    self._feature_names, analysis['feature_stds_scaled'])),
                'scaler_scale': dict(zip(
                    self._feature_names, analysis['scaler_scale'])),
                # GPT P0-C: condition number
                'condition_number_theta_scaled': analysis['summary'][
                    'condition_number_theta_scaled'],
                'condition_number_no_constant': analysis['summary'][
                    'condition_number_no_constant'],
                # GPT P1: oracle recovery error
                'oracle_recovery': analysis['summary'].get(
                    'oracle_recovery', {}),
            },
        }
        with open(rd / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  metrics.json")

        # 8. manifest.json
        manifest = {
            'run_id': self.run_id,
            'system': 'aek',
            'gate': 'gate1',
            'runner': 'run_aek3_baseline.py',
            'runner_version': 'v1.2',
            'created_at': datetime.now().isoformat(),
            'config': {
                'dataset_version': self.cfg.dataset_version,
                'seed': self.cfg.seed,
                'n_train': self.cfg.n_train,
                'n_bootstrap': self.cfg.n_bootstrap,
                'best_threshold': float(results['best_threshold']),
                'library_size': N_AEK_FEATURES,
                'library_terms': list(self._feature_names),
                'reparam': self.cfg.reparam,
            },
            'outputs': [
                'manifest.json',
                'metrics.json',
                'sindy_coefficients.csv',
                'sindy_coefficients_scaled.csv',
                'teacher_support.npy',
                'stable_core_mask.npy',
                'z_before.npy',
                'oracle_coefficients.npy',
                'fragile_pairs.json',
            ],
        }
        with open(rd / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  manifest.json")

    # ----------------------------------------------------------
    # Phase 5: Figures
    # ----------------------------------------------------------
    def _generate_figures(self, results: Dict, analysis: Dict):
        """Generate diagnostic figures."""
        print(f"\n[Phase 5] Generating figures...")
        fig_dir = self.results_dir / 'figures'

        setup_style()

        # F01: Coefficient heatmap (E-SINDy vs Oracle)
        self._plot_coefficient_heatmap(results, analysis, fig_dir)

        # F02: Support comparison
        self._plot_support_comparison(results, analysis, fig_dir)

    def _plot_coefficient_heatmap(self, results, analysis, fig_dir):
        """Coefficient heatmap: E-SINDy mean vs Oracle."""
        import matplotlib.pyplot as plt

        coeff = results['coefficients_mean']
        oracle = analysis['oracle_coeff']

        fig, (ax1, ax2) = create_figure('double', nrows=1, ncols=2)

        # Shared color scale
        vmax = max(np.abs(coeff).max(), np.abs(oracle).max())
        if vmax < 1e-10:
            vmax = 1.0  # prevent degenerate case

        target_labels = [r"$\dot\phi$", r"$\ddot\phi$",
                         r"$\dot\theta_w$", r"$\ddot\theta_w$"]

        # E-SINDy
        im1 = ax1.imshow(coeff, aspect='auto', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
        ax1.set_title('E-SINDy (learned)')
        ax1.set_ylabel('Feature')
        ax1.set_xlabel('Target')
        ax1.set_yticks(range(N_AEK_FEATURES))
        ax1.set_yticklabels(self._feature_names, fontsize=7)
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(target_labels, fontsize=9)

        # Oracle
        im2 = ax2.imshow(oracle, aspect='auto', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
        ax2.set_title('Oracle (analytic)')
        ax2.set_xlabel('Target')
        ax2.set_yticks(range(N_AEK_FEATURES))
        ax2.set_yticklabels(self._feature_names, fontsize=7)
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(target_labels, fontsize=9)

        fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label='Coefficient')
        reparam_label = f', {self.cfg.reparam}' if self.cfg.reparam != 'standard' else ''
        fig.suptitle(f'AEK E-SINDy Baseline (seed={self.cfg.seed}, '
                     f'thr={results["best_threshold"]:.3f}{reparam_label})',
                     fontsize=11)
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)

        save_figure(fig, fig_dir, 'F01_coefficient_heatmap')

    def _plot_support_comparison(self, results, analysis, fig_dir):
        """Support comparison: E-SINDy vs Oracle with fragile highlighting."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        support = results['support_mask']
        oracle = analysis['oracle_support']

        # Combined visualization
        # 0=both off, 1=oracle only (dynamics fragile),
        # 2=esindy only (spurious), 3=both on (agreement)
        combined = np.zeros(support.shape, dtype=float)
        both_on = support & oracle
        oracle_only = oracle & ~support
        esindy_only = support & ~oracle

        combined[oracle_only] = 1    # recall failure (red)
        combined[esindy_only] = 2    # precision failure (orange)
        combined[both_on] = 3        # correct (green)

        fig, ax = create_figure('single')

        cmap = plt.cm.colors.ListedColormap(
            ['#f0f0f0', '#e74c3c', '#f39c12', '#27ae60']
        )
        ax.imshow(combined, aspect='auto', cmap=cmap,
                  vmin=-0.5, vmax=3.5)

        target_labels = [r"$\dot\phi$", r"$\ddot\phi$",
                         r"$\dot\theta_w$", r"$\ddot\theta_w$"]

        ax.set_title(f'Support: E-SINDy vs Oracle (seed={self.cfg.seed})')
        ax.set_ylabel('Feature')
        ax.set_xlabel('Target')
        ax.set_yticks(range(N_AEK_FEATURES))
        ax.set_yticklabels(self._feature_names, fontsize=7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(target_labels, fontsize=9)

        # Legend
        legend_elements = [
            Patch(facecolor='#f0f0f0', edgecolor='gray', label='Both OFF'),
            Patch(facecolor='#e74c3c', label='Recall failure (dynamics)'),
            Patch(facecolor='#f39c12', label='Precision failure (spurious)'),
            Patch(facecolor='#27ae60', label='Agreement'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6,
                  framealpha=0.9)

        fig.tight_layout()
        save_figure(fig, fig_dir, 'F02_support_comparison')

    # ----------------------------------------------------------
    # Phase 6: Context Packet
    # ----------------------------------------------------------
    def _save_context_packet(self, results: Dict, analysis: Dict):
        """Generate context packet for cross-review."""
        cp_path = paths.get_context_packet_path(self.run_id)

        reparam_info = f" ({self.cfg.reparam})" if self.cfg.reparam != 'standard' else ''

        lines = [
            f"# Context Packet: {self.run_id}",
            f"",
            f"**System**: AEK Self-balancing Motorcycle",
            f"**Gate**: Gate1 (E-SINDy Baseline{reparam_info})",
            f"**Seed**: {self.cfg.seed}",
            f"**Date**: {datetime.now().isoformat()}",
            f"**Runner**: experiments/run_aek3_baseline.py v1.2",
            f"**Reparam**: {self.cfg.reparam}",
            f"",
            f"## Configuration",
            f"",
            f"- Library: {N_AEK_FEATURES} terms{reparam_info}",
            f"- n_train: {self.cfg.n_train}",
            f"- n_bootstrap: {self.cfg.n_bootstrap}",
            f"- Threshold (val-selected): {results['best_threshold']}",
            f"- Threshold selection: per-target R² mean (NOT MSE)",
            f"",
            f"## Results Summary",
            f"",
            f"- Active terms: {int(results['support_mask'].sum())} / "
            f"{results['support_mask'].size}",
            f"- Oracle terms: {int(analysis['oracle_support'].sum())}",
            f"- Fragile pairs: {analysis['fragile_info']['n_fragile']}",
            f"  - Dynamics (recall failure): "
            f"{analysis['fragile_info']['n_fragile_dynamics']}",
            f"  - Spurious (precision failure): "
            f"{analysis['fragile_info']['n_fragile_spurious']}",
            f"- Stable core: {analysis['fragile_info']['n_stable']}",
            f"- Z-before median: {analysis['summary']['z_before_median']}",
            f"",
            f"## Diagnostics (GPT P0-B/P0-C/P1)",
            f"",
            f"- Condition number (Theta_scaled): "
            f"{analysis['summary']['condition_number_theta_scaled']:.1f}",
            f"- Condition number (no constant): "
            f"{analysis['summary']['condition_number_no_constant']:.1f}",
            f"",
        ]

        # Oracle recovery error
        oracle_rec = analysis['summary'].get('oracle_recovery', {})
        if oracle_rec:
            lines += [
                f"## Oracle Recovery Error",
                f"",
                f"- Mean |error|: {oracle_rec['mean_abs_error']:.4f}",
                f"- Mean rel error: {oracle_rec['mean_rel_error']:.4f}",
                f"",
            ]
            for term in oracle_rec.get('per_term', []):
                lines.append(
                    f"  - {term['feature']} → {term['target']}: "
                    f"oracle={term['oracle_coeff']:.4f}, "
                    f"esindy={term['esindy_coeff']:.4f}, "
                    f"rel_err={term['rel_error']:.4f}"
                )
            lines.append("")

        lines.append(f"## Precision/Recall per Target")
        lines.append(f"")

        for pr in analysis['precision_recall']:
            lines.append(
                f"- {pr['target']}: P={pr['precision']:.2f}, "
                f"R={pr['recall']:.2f} "
                f"(E-SINDy={pr['esindy_active']}, Oracle={pr['oracle_active']})"
            )

        lines += [
            f"",
            f"## Fragile Pairs",
            f"",
        ]
        if analysis['fragile_info']['fragile_pairs']:
            for pair in analysis['fragile_info']['fragile_pairs']:
                lines.append(
                    f"- [{pair['type']}] {pair['feature_name']} -> "
                    f"{pair['target_name']}"
                )
        else:
            lines.append("- None (perfect oracle agreement)")

        lines += [
            f"",
            f"## Oracle Coefficients (Nominal)",
            f"",
            f"- phi_ddot: sin(phi) = "
            f"{analysis['oracle_coeff'][4,1]:.4f}, "
            f"tau = {analysis['oracle_coeff'][3,1]:.4f}",
            f"- theta_w_ddot: sin(phi) = "
            f"{analysis['oracle_coeff'][4,3]:.4f}, "
            f"tau = {analysis['oracle_coeff'][3,3]:.4f}",
            f"",
            f"## Output Directory",
            f"",
            f"`{self.results_dir}`",
            f"",
            f"## Next Steps",
            f"",
            f"- AEK-4c: Reparam-1 augmentation + D-optimal/Random ablation",
            f"- Compare condition number vs Standard baseline",
        ]

        with open(cp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Context Packet: {cp_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='AEK-3: E-SINDy Baseline for AEK System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (PowerShell, copy-paste ready):
  python experiments/run_aek3_baseline.py --seed 0
  python experiments/run_aek3_baseline.py --seed 1
  python experiments/run_aek3_baseline.py --seed 0 --reparam reparam1
  python experiments/run_aek3_baseline.py --seed 0 --n_bootstrap 200
        """
    )
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--n_bootstrap', type=int, default=100,
                        help='Bootstrap iterations (default: 100)')
    parser.add_argument('--dataset_version', type=str, default='aek_ood_v1',
                        help='Dataset version (default: aek_ood_v1)')
    parser.add_argument('--note', type=str, default='aek3_baseline',
                        help='Run note (default: aek3_baseline)')
    parser.add_argument('--reparam', type=str, default='standard',
                        choices=['standard', 'reparam1'],
                        help='Library parameterization (default: standard)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-update note for reparam1
    note = args.note
    if args.reparam == 'reparam1' and 'reparam' not in note:
        note = note.replace('baseline', 'baseline_rp1')

    config = AEK3Config(
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        dataset_version=args.dataset_version,
        note=note,
        reparam=args.reparam,
    )
    runner = AEK3Runner(config)
    result = runner.run()

    if result['status'] == 'success':
        print("\nNext steps:")
        print("  1. Review fragile_pairs.json")
        print("  2. Run with seed=1 for reproducibility")
        print("  3. GPT cross-review with context packet")
    else:
        print(f"\nFailed: {result.get('error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()