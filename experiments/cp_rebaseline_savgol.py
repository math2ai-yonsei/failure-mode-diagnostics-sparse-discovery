"""
CP Day3 Rebaseline: Recompute z_before with SavGol Derivatives

Purpose:
    The original Day3 (phase35) baseline computed z_before using analytic dx
    (dataset['train_dx']). Paper §3.3 requires all derivative estimation to
    use SavGol. This script recomputes z_before using dataset['train_dx_savgol']
    and produces a new Day3-compatible artifact directory.

What it produces:
    A new directory in phase35 with:
    - z_before.npy (recomputed with SavGol dx)
    - teacher_support.npy (copied from original — oracle, dx-independent)
    - fragile_pairs.json (recomputed from new z_before)
    - manifest.json (preserving gate1_artifacts for teacher coeff loading)

Usage (PowerShell, copy-paste ready):
    python experiments/cp_rebaseline_savgol.py --seed 0 --original_day3_run_id 20260110_004647_nogit_day6_r3_B100
    python experiments/cp_rebaseline_savgol.py --seed 1 --original_day3_run_id 20260202_165724_nogit_day5_ctrl250_B100_seed1

Author: Claude (SavGol consistency patch)
Date: 2026-03-11
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import hashlib
import shutil
import numpy as np
from datetime import datetime

from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.sindy.library import SINDyLibrary
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble


# =============================================================================
# Constants (matching Gate3 SSOT)
# =============================================================================

DEFAULT_BOOTSTRAP_B = 20
DEFAULT_THRESHOLD = 0.05
DEFAULT_TAU_SUPPORT = 0.5
DEFAULT_Z0 = 2.0
DEFAULT_EPS = 1e-12
DYNAMICS_TARGET_INDICES = [1, 3]  # x_ddot, theta_ddot
TARGET_NAMES = ['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CP Day3 Rebaseline: z_before with SavGol dx')
    parser.add_argument('--seed', type=int, required=True,
                        help='Seed (0 or 1)')
    parser.add_argument('--original_day3_run_id', type=str, required=True,
                        help='Original Day3 run_id to copy teacher artifacts from')
    parser.add_argument('--bootstrap_B', type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--n_train', type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("  CP Day3 Rebaseline: z_before with SavGol Derivatives")
    print("=" * 70)

    # --- Paths ---
    project_root = _PROJECT_ROOT
    dataset_version = 'cartpole_ood_v1'
    track = 'standardized'
    method = 'stable_core'

    original_day3_dir = (
        project_root / 'results' / dataset_version / 'phase35' /
        track / method / f'n{args.n_train}' / f'seed{args.seed}' /
        args.original_day3_run_id
    )
    print(f"\n[1] Original Day3: {original_day3_dir}")

    if not original_day3_dir.exists():
        raise FileNotFoundError(f"Original Day3 not found: {original_day3_dir}")

    # --- Create new rebaseline directory ---
    run_id = paths.generate_run_id('savgol_rebaseline')
    new_day3_dir = (
        project_root / 'results' / dataset_version / 'phase35' /
        track / method / f'n{args.n_train}' / f'seed{args.seed}' /
        run_id
    )
    new_day3_dir.mkdir(parents=True, exist_ok=True)
    print(f"  New Day3: {new_day3_dir}")

    # --- Load original manifest (for gate1_artifacts / teacher_run_id) ---
    original_manifest_path = original_day3_dir / 'manifest.json'
    if original_manifest_path.exists():
        with open(original_manifest_path, 'r', encoding='utf-8') as f:
            original_manifest = json.load(f)
        print(f"  Original manifest loaded")
    else:
        # Minimal fallback
        original_manifest = {}
        print(f"  ⚠️  No original manifest found")

    # --- Copy teacher_support (oracle, dx-independent) ---
    teacher_support_src = original_day3_dir / 'teacher_support.npy'
    if teacher_support_src.exists():
        teacher_support = np.load(teacher_support_src)
        np.save(new_day3_dir / 'teacher_support.npy', teacher_support)
        print(f"\n[2] teacher_support copied: {teacher_support.shape}")
    else:
        raise FileNotFoundError(f"teacher_support.npy not found in {original_day3_dir}")

    # --- Load dataset with SavGol dx ---
    print(f"\n[3] Loading dataset with SavGol dx...")
    dataset_path = project_root / 'data' / 'cartpole' / dataset_version / 'dataset.npz'
    validate_dataset_lite(dataset_path)
    dataset = dict(np.load(dataset_path, allow_pickle=True))

    assert 'train_dx_savgol' in dataset, \
        "Dataset missing 'train_dx_savgol' key. Run dataset preprocessing first."

    train_x = dataset['train_x'][:args.n_train]                # (N, T, 4) — clean x
    train_u = dataset['train_u'][:args.n_train]                 # (N, T, 1)
    train_dx = dataset['train_dx_savgol'][:args.n_train]        # (N, T, 4) — SavGol dx
    print(f"  train_x: {train_x.shape}")
    print(f"  train_dx (SavGol): {train_dx.shape}")

    # Verify it's different from analytic
    train_dx_analytic = dataset['train_dx'][:args.n_train]
    diff = np.abs(train_dx - train_dx_analytic)
    print(f"  SavGol vs analytic: max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")

    # --- Fit E-SINDy baseline ---
    print(f"\n[4] Fitting E-SINDy baseline (B={args.bootstrap_B}, thr={args.threshold})")

    N_train, T, D = train_x.shape
    n_targets = len(TARGET_NAMES)

    # Build library (same as Gate3: SINDyLibrary gate0_min)
    library = SINDyLibrary(config='gate0_min')
    x_flat = train_x.reshape(-1, D)
    u_flat = train_u.reshape(-1, 1)
    Theta = library.fit_transform(x_flat, u_flat)
    dx_flat = train_dx.reshape(-1, n_targets)

    feature_names = library.get_feature_names()
    n_features = len(feature_names)
    print(f"  Library: {n_features} features")
    print(f"  Theta: {Theta.shape}, dx: {dx_flat.shape}")

    # Scale
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    # Condition number
    kappa = float(np.linalg.cond(Theta_scaled))
    print(f"  κ (scaled): {kappa:.1f}")

    # E-SINDy
    ensemble = ESINDyEnsemble(
        n_bootstrap=args.bootstrap_B,
        threshold=args.threshold,
        random_state=args.seed,
    )
    ensemble.fit(
        Theta_scaled, dx_flat,
        n_trajectories=N_train,
        T=T,
        scaler=scaler,
        target_scale=None,
    )

    coef_mean = ensemble.coefficients_mean_
    coef_std = ensemble.coefficients_std_
    inc_prob = ensemble.inclusion_probability_

    # z-scores
    z_before = np.abs(coef_mean) / (coef_std + DEFAULT_EPS)
    print(f"  z_before shape: {z_before.shape}")
    print(f"  z_before range: [{z_before.min():.3f}, {z_before.max():.3f}]")

    # --- Compute fragile pairs ---
    print(f"\n[5] Computing fragile pairs (tau={DEFAULT_TAU_SUPPORT}, z0={DEFAULT_Z0})")

    support_mask = inc_prob >= DEFAULT_TAU_SUPPORT
    fragile_pairs = []

    for t_idx in range(n_targets):
        for f_idx in range(n_features):
            is_support = support_mask[f_idx, t_idx]
            if is_support and z_before[f_idx, t_idx] < DEFAULT_Z0:
                fragile_pairs.append([int(f_idx), int(t_idx)])

    # Also identify dynamics-only fragile pairs (for Gate3 filtering)
    dynamics_fragile = [
        [f, t] for f, t in fragile_pairs if t in DYNAMICS_TARGET_INDICES
    ]

    print(f"  Total fragile pairs: {len(fragile_pairs)}")
    print(f"  Dynamics-only fragile: {len(dynamics_fragile)}")
    print(f"  Active terms: {support_mask.sum()}/{n_features * n_targets}")

    # --- Save artifacts ---
    print(f"\n[6] Saving artifacts...")

    # z_before.npy
    np.save(new_day3_dir / 'z_before.npy', z_before)
    print(f"  ✅ z_before.npy: {z_before.shape}")

    # fragile_pairs.json
    fp_data = {
        'pairs': fragile_pairs,
        'dynamics_pairs': dynamics_fragile,
        'n_total': len(fragile_pairs),
        'n_dynamics': len(dynamics_fragile),
        'z0': DEFAULT_Z0,
        'tau_support': DEFAULT_TAU_SUPPORT,
        'dx_source': 'train_dx_savgol',
    }
    with open(new_day3_dir / 'fragile_pairs.json', 'w') as f:
        json.dump(fp_data, f, indent=2)
    print(f"  ✅ fragile_pairs.json: {len(fragile_pairs)} pairs")

    # Compute SHA for teacher_support
    with open(new_day3_dir / 'teacher_support.npy', 'rb') as f:
        ts_sha = hashlib.sha256(f.read()).hexdigest()

    # manifest.json — preserve gate1_artifacts from original
    manifest = {
        'run_id': run_id,
        'type': 'savgol_rebaseline',
        'original_day3_run_id': args.original_day3_run_id,
        'seed': args.seed,
        'n_train': args.n_train,
        'bootstrap_B': args.bootstrap_B,
        'threshold': args.threshold,
        'dx_source': 'train_dx_savgol',
        'teacher_support_sha256': ts_sha,
        'n_features': n_features,
        'n_targets': n_targets,
        'n_fragile_total': len(fragile_pairs),
        'n_fragile_dynamics': len(dynamics_fragile),
        'kappa_scaled': kappa,
        'created_at': datetime.now().isoformat(),
        # Preserve gate1_artifacts from original manifest
        'gate1_artifacts': original_manifest.get('gate1_artifacts', {}),
    }
    with open(new_day3_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  ✅ manifest.json")

    # Also copy stable_core_mask if it exists
    stable_core_src = original_day3_dir / 'stable_core_mask.npy'
    if stable_core_src.exists():
        # Recompute stable_core_mask with new z_before
        stable_core_mask = support_mask & (z_before >= DEFAULT_Z0)
        np.save(new_day3_dir / 'stable_core_mask.npy', stable_core_mask)
        print(f"  ✅ stable_core_mask.npy (recomputed)")

    # Copy oracle_coefficients if exists
    oracle_src = original_day3_dir / 'oracle_coefficients.npy'
    if oracle_src.exists():
        shutil.copy2(oracle_src, new_day3_dir / 'oracle_coefficients.npy')
        print(f"  ✅ oracle_coefficients.npy (copied)")

    # Copy inc_prob_before if needed by downstream
    np.save(new_day3_dir / 'inc_prob_before.npy', inc_prob)
    print(f"  ✅ inc_prob_before.npy")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"  ✅ CP Day3 Rebaseline Complete!")
    print(f"{'=' * 70}")
    print(f"  New run_id: {run_id}")
    print(f"  Path: {new_day3_dir}")
    print(f"  z_before: median={np.median(z_before):.3f}, "
          f"max={z_before.max():.3f}")
    print(f"  Fragile pairs: {len(fragile_pairs)} total, "
          f"{len(dynamics_fragile)} dynamics")
    print(f"")
    print(f"  Next steps:")
    print(f"    Gate3: python experiments/run_gate3_v2.py "
          f"--day3_run_id {run_id} --seed {args.seed} ...")
    print(f"    Gate4: python experiments/run_gate4_ablation.py "
          f"--day3_source {new_day3_dir.relative_to(project_root)} ...")


if __name__ == '__main__':
    main()
