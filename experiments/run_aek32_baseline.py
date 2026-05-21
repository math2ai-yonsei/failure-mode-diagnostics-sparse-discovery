"""
AEK-3.2: Re-baseline for Reparam-2

Purpose:
    Generate baseline artifacts (z_before, teacher_support, fragile_pairs,
    sindy_coefficients) for the Reparam-2 library. These artifacts are
    consumed by the Gate4c-2 augmentation runners (Random, D-optimal).

    No augmentation — E-SINDy on training data only.

Artifacts produced (per seed):
    results/aek_ood_v1/gate1/standardized/esindy/n10/seed{S}/{run_id}/
    ├── manifest.json
    ├── metrics.json         (kappa, support counts)
    ├── sindy_coefficients.csv  (14×4 unscaled teacher coefficients)
    ├── z_before.npy         (14×4 z-metric matrix)
    ├── teacher_support.npy  (14×4 boolean)
    └── fragile_pairs.json   (feature_index primary key)

Library: Reparam-2 (build_aek_library_v3 equivalent)
    #5:  cos(phi) → cos(phi)-1        (inherited from RP1)
    #13: sin(phi)*tau → (cos(phi)-1)*tau  (★ RP2 new)

Usage:
    python experiments/run_aek32_baseline.py
    python experiments/run_aek32_baseline.py --seeds 0 1

Author: Claude (Gate4c Phase 2)
Date: 2026-03-03
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

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

# ============================================================
# Constants
# ============================================================

RUNNER_VERSION = 'v1.0'
DATASET_VERSION = 'aek_ood_v1'
SYSTEM = 'aek'
N_TRAIN = 10
REPARAM = 'reparam2'

# E-SINDy settings (match Gate4c runners)
N_BOOTSTRAP = 100
THRESHOLD = 0.05
Z_EPS = 1e-6

# Fragile pair threshold: z < TAU_SUPPORT → fragile
TAU_SUPPORT = 2.0


# ============================================================
# AC2: Feature Name Integrity
# ============================================================

def assert_oracle_label_integrity():
    """Verify oracle-relevant feature indices match expected names."""
    names = get_aek_feature_names(REPARAM)
    checks = {
        0: '1', 1: 'phi_dot', 2: 'theta_w_dot',
        3: 'tau', 4: 'sin(phi)',
        5: 'cos(phi)-1',       # RP1 inherited
        13: '(cos(phi)-1)*tau', # RP2 new
    }
    for idx, expected in checks.items():
        actual = names[idx]
        assert actual == expected, \
            f"AC2 FAIL: idx {idx} expected '{expected}', got '{actual}'"

    # Oracle terms (#3 tau, #4 sin(phi)) must be in oracle support
    oracle = get_aek_oracle_support()
    assert oracle[3, 1] and oracle[3, 3], "AC2: tau not in oracle"
    assert oracle[4, 1] and oracle[4, 3], "AC2: sin(phi) not in oracle"
    # #5 and #13 must NOT be in oracle
    assert not oracle[5, :].any(), "AC2: #5 in oracle"
    assert not oracle[13, :].any(), "AC2: #13 in oracle"
    assert len(names) == N_AEK_FEATURES
    print("  AC2: Feature name integrity ✅")


# ============================================================
# Baseline E-SINDy
# ============================================================

def run_baseline_esindy(
    train_x: np.ndarray,    # (N, T, 4)
    train_u: np.ndarray,    # (N, T, 1)
    train_dx: np.ndarray,   # (N, T, 4)
    seed: int,
) -> Dict[str, Any]:
    """
    Run E-SINDy on training data only (no augmentation).

    Returns dict with z-metric matrix, coefficients, support, kappa.
    """
    N, T, D = train_x.shape

    # Flatten
    x_flat = train_x.reshape(-1, D)
    u_flat = train_u.reshape(-1, 1)
    dx_flat = train_dx.reshape(-1, D)

    # Build library
    Theta, feat_names = build_aek_library_by_name(x_flat, u_flat, reparam=REPARAM)
    print(f"  Theta: {Theta.shape}, features: {len(feat_names)}")

    # Scale
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    # Condition number
    kappa_with = float(np.linalg.cond(Theta_scaled))

    # Without constant
    non_const = ~scaler.constant_mask_
    if non_const.any():
        kappa_without = float(np.linalg.cond(Theta_scaled[:, non_const]))
    else:
        kappa_without = float('inf')

    print(f"  κ₂(with const):    {kappa_with:.4e} (log10={np.log10(kappa_with):.2f})")
    print(f"  κ₂(without const): {kappa_without:.4e} (log10={np.log10(kappa_without):.2f})")

    # Column stds diagnostic
    col_stds = np.std(Theta, axis=0)
    print(f"  Column stds range: [{col_stds.min():.4e}, {col_stds.max():.4e}]")
    print(f"  #13 ((cos(phi)-1)*tau) std: {col_stds[13]:.4e}")

    # E-SINDy
    ensemble = ESINDyEnsemble(
        n_bootstrap=N_BOOTSTRAP,
        threshold=THRESHOLD,
        random_state=seed,
    )
    ensemble.fit(
        Theta_scaled, dx_flat,
        n_trajectories=N,
        T=T,
        scaler=scaler,
        target_scale=None,
    )

    coeff_mean = ensemble.coefficients_mean_     # (14, 4) unscaled
    coeff_std = ensemble.coefficients_std_        # (14, 4) unscaled
    inc_prob = ensemble.inclusion_probability_    # (14, 4)

    # Z-metric: |mean| / (std + eps)
    z = np.abs(coeff_mean) / (coeff_std + Z_EPS)

    # Teacher support: any bootstrap detected nonzero
    teacher_support = (inc_prob > 0).astype(bool)

    return {
        'z': z,                          # (14, 4)
        'coefficients_mean': coeff_mean,
        'coefficients_std': coeff_std,
        'inclusion_probability': inc_prob,
        'teacher_support': teacher_support,
        'feature_names': feat_names,
        'kappa_with_const': kappa_with,
        'kappa_without_const': kappa_without,
        'column_stds': col_stds.tolist(),
        'scaler_scales': scaler.scale_.tolist(),
        'n_samples': int(x_flat.shape[0]),
    }


# ============================================================
# Fragile Pair Identification
# ============================================================

def identify_fragile_pairs(
    z: np.ndarray,           # (14, 4)
    teacher_support: np.ndarray,  # (14, 4) bool
    oracle_support: np.ndarray,   # (14, 4) bool
    feature_names: List[str],
    tau: float = TAU_SUPPORT,
) -> Dict[str, Any]:
    """
    Identify fragile (feature, target) pairs.

    A pair is fragile if:
        - Teacher includes it (inc_prob > 0) OR oracle includes it
        - z-score < tau (unstable identification)

    Classification:
        - dynamics: oracle ON, E-SINDy OFF → recall failure
        - spurious: oracle OFF, E-SINDy ON → precision failure
    """
    pairs = []
    n_dynamics = 0
    n_spurious = 0

    for f_idx in range(z.shape[0]):
        for t_idx in range(z.shape[1]):
            is_oracle = bool(oracle_support[f_idx, t_idx])
            is_teacher = bool(teacher_support[f_idx, t_idx])
            z_val = float(z[f_idx, t_idx])

            # Fragile if active (oracle or teacher) and z < tau
            if (is_oracle or is_teacher) and z_val < tau:
                if is_oracle and not is_teacher:
                    pair_type = 'dynamics'
                    n_dynamics += 1
                elif not is_oracle and is_teacher:
                    pair_type = 'spurious'
                    n_spurious += 1
                elif is_oracle and is_teacher:
                    # Both: if z < tau, it's a dynamics issue (underdetection)
                    pair_type = 'dynamics'
                    n_dynamics += 1
                else:
                    continue

                pairs.append({
                    'feature_idx': f_idx,
                    'target_idx': t_idx,
                    'feature_name': feature_names[f_idx],
                    'target_name': AEK_TARGET_NAMES[t_idx],
                    'type': pair_type,
                    'z_score': round(z_val, 4),
                })

    result = {
        'system': SYSTEM,
        'reparam': REPARAM,
        'n_features': N_AEK_FEATURES,
        'n_targets': 4,
        'feature_names': feature_names,
        'target_names': list(AEK_TARGET_NAMES),
        'tau_support': tau,
        'n_fragile': len(pairs),
        'n_fragile_dynamics': n_dynamics,
        'n_fragile_spurious': n_spurious,
        'pairs': pairs,
    }

    print(f"  Fragile pairs: {len(pairs)} "
          f"(dynamics={n_dynamics}, spurious={n_spurious})")

    return result


# ============================================================
# Save Artifacts
# ============================================================

def save_baseline(
    run_dir: Path,
    run_id: str,
    seed: int,
    esindy_result: Dict[str, Any],
    fragile_data: Dict[str, Any],
):
    """Save all baseline artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'figures').mkdir(exist_ok=True)

    feat_names = esindy_result['feature_names']

    # z_before.npy — full (14, 4) matrix
    np.save(run_dir / 'z_before.npy', esindy_result['z'])

    # teacher_support.npy — (14, 4) boolean
    np.save(run_dir / 'teacher_support.npy', esindy_result['teacher_support'])

    # sindy_coefficients.csv
    coeff = esindy_result['coefficients_mean']
    with open(run_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + list(AEK_TARGET_NAMES))
        for i, name in enumerate(feat_names):
            w.writerow([name] + [f"{coeff[i,j]:.8f}" for j in range(4)])

    # fragile_pairs.json
    with open(run_dir / 'fragile_pairs.json', 'w') as f:
        json.dump(fragile_data, f, indent=2)

    # metrics.json
    metrics = {
        'system': SYSTEM,
        'gate': 'gate1',
        'method': 'esindy_baseline',
        'reparam': REPARAM,
        'baseline_seed': seed,
        'n_train': N_TRAIN,
        'n_bootstrap': N_BOOTSTRAP,
        'threshold': THRESHOLD,
        'kappa_with_const': esindy_result['kappa_with_const'],
        'kappa_without_const': esindy_result['kappa_without_const'],
        'log10_kappa_with': float(np.log10(esindy_result['kappa_with_const'])),
        'log10_kappa_without': float(np.log10(esindy_result['kappa_without_const'])),
        'n_fragile': fragile_data['n_fragile'],
        'n_fragile_dynamics': fragile_data['n_fragile_dynamics'],
        'n_fragile_spurious': fragile_data['n_fragile_spurious'],
        'support_terms_total': int(esindy_result['teacher_support'].sum()),
        'column_stds': esindy_result['column_stds'],
        'scaler_scales': esindy_result['scaler_scales'],
        'feature_names': feat_names,
        'runner_version': RUNNER_VERSION,
    }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    # manifest.json
    manifest = {
        'run_id': run_id,
        'system': SYSTEM,
        'gate': 'gate1',
        'method': 'esindy_baseline',
        'reparam': REPARAM,
        'baseline_seed': seed,
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_aek32_baseline.py',
        'runner_version': RUNNER_VERSION,
        'config': {
            'n_train': N_TRAIN,
            'n_bootstrap': N_BOOTSTRAP,
            'threshold': THRESHOLD,
            'z_eps': Z_EPS,
            'tau_support': TAU_SUPPORT,
        },
        'artifacts': [
            'manifest.json', 'metrics.json', 'sindy_coefficients.csv',
            'z_before.npy', 'teacher_support.npy', 'fragile_pairs.json',
        ],
    }
    with open(run_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=_json_default)

    print(f"  Saved to: {run_dir}")


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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='AEK-3.2 Re-baseline (Reparam-2)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1],
                        help='Baseline seeds (default: 0 1)')
    args = parser.parse_args()

    print("=" * 70)
    print("  AEK-3.2: Re-baseline for Reparam-2")
    print(f"  Library: {REPARAM}")
    print(f"  Seeds: {args.seeds}")
    print(f"  E-SINDy: B={N_BOOTSTRAP}, threshold={THRESHOLD}")
    print("=" * 70)

    # AC2 check
    print("\n[AC2] Oracle label integrity...")
    assert_oracle_label_integrity()

    # Load dataset
    print("\n[Load] Dataset...")
    dataset_path = paths.get_dataset_path(DATASET_VERSION, system=SYSTEM)
    validate_dataset_lite(dataset_path)
    dataset = dict(np.load(dataset_path, allow_pickle=True))
    print(f"  Dataset: {dataset_path}")

    train_x = dataset['train_x']    # (N, T, 4)
    train_u = dataset['train_u']    # (N, T, 1)
    train_dx = dataset['train_dx']  # (N, T, 4)
    print(f"  train_x: {train_x.shape}")

    # Oracle support (same for all reparams)
    oracle_support = get_aek_oracle_support()
    print(f"  Oracle support: {oracle_support.sum()} active terms")

    # Run for each seed
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"  Baseline seed={seed}")
        print(f"{'='*70}")

        # E-SINDy baseline
        print("\n[E-SINDy] Running baseline (train only, no augmentation)...")
        result = run_baseline_esindy(train_x, train_u, train_dx, seed=seed)

        # Fragile pairs
        print("\n[Fragile] Identifying fragile pairs...")
        fragile_data = identify_fragile_pairs(
            z=result['z'],
            teacher_support=result['teacher_support'],
            oracle_support=oracle_support,
            feature_names=result['feature_names'],
        )

        # Save
        run_id = paths.generate_run_id(note=f'aek32_baseline_rp2')
        run_dir = paths.get_results_dir(
            dataset_version=DATASET_VERSION,
            gate='gate1',
            track='standardized',
            method='esindy',
            n_train=N_TRAIN,
            seed=seed,
            run_id=run_id,
        )

        print(f"\n[Save] Saving artifacts...")
        save_baseline(run_dir, run_id, seed, result, fragile_data)

        # Summary
        print(f"\n[Summary] seed={seed}")
        print(f"  κ₂(with const):    {result['kappa_with_const']:.4e}")
        print(f"  κ₂(without const): {result['kappa_without_const']:.4e}")
        print(f"  Fragile: {fragile_data['n_fragile']} "
              f"(dyn={fragile_data['n_fragile_dynamics']}, "
              f"spur={fragile_data['n_fragile_spurious']})")
        print(f"  Teacher support: {result['teacher_support'].sum()} terms")
        print(f"  Run dir: {run_dir}")

    print(f"\n{'='*70}")
    print(f"  AEK-3.2 Re-baseline COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
