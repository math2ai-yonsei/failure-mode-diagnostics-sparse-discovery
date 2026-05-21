"""
R2-0: Fail-fast Diagnostic for Reparam-2

Purpose:
    Before investing in re-baseline and augmentation runs, verify that
    Reparam-2 (#5: cos(phi)->cos(phi)-1, #13: sin(phi)*tau->(cos(phi)-1)*tau)
    achieves acceptable identifiability metrics without introducing
    pathological near-zero columns.

    Cost: ~0 (no augmentation, uses existing training data only)

Fail-fast Criteria (GPT/Claude consensus):
    AC1: corr(phi*tau, new#13) <= 0.995  (cancellation pair broken)
    AC2: std(new#13) >= 1e-6 AND max(scale)/median(scale) <= 1e4
    AC3: log10(kappa_RP2) <= log10(kappa_RP1) + 0.5  (no severe degradation)

If any criterion fails:
    RP2-A is abandoned. Fallback: RP2-C (remove #13, 13-term library).

Reports:
    - All 3 library variants (Standard, RP1, RP2) side-by-side
    - kappa with/without constant
    - Full feature column stds
    - Pairwise correlations for cancellation pairs

Usage:
    python experiments/run_r2_0_failfast.py

Author: Claude (Gate4c Phase 2)
Date: 2026-03-03
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import numpy as np
from datetime import datetime

from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.sindy.optimizer import ColumnScaler
from src.sindy.aek_library import (
    build_aek_library_by_name,
    get_aek_feature_names,
    N_AEK_FEATURES,
)


# ============================================================
# Configuration
# ============================================================

DATASET_VERSION = 'aek_ood_v1'
SYSTEM = 'aek'
N_TRAIN = 10

# Fail-fast thresholds (GPT/Claude consensus)
CORR_THRESHOLD = 0.995       # AC1: target pair correlation upper bound
STD_MIN = 1e-6               # AC2: minimum column std
SCALE_RATIO_MAX = 1e4        # AC2: max(scale)/median(scale)
KAPPA_MARGIN_LOG10 = 0.5     # AC3: allowed kappa increase vs RP1

# Feature indices for cancellation pair
IDX_PHI_TAU = 10             # phi*tau
IDX_NEW_13 = 13              # sin(phi)*tau (Std/RP1) or (cos(phi)-1)*tau (RP2)


# ============================================================
# Diagnostic Functions
# ============================================================

def compute_diagnostics(
    Theta: np.ndarray,
    feature_names: list,
    label: str,
) -> dict:
    """
    Compute full diagnostic suite for a library variant.

    Returns dict with:
        kappa_with_const, kappa_without_const,
        column_stds, column_stds_dict,
        scale_ratio, scaler_scales,
        corr_10_13, corr_matrix_key_pairs
    """
    N, P = Theta.shape
    print(f"\n{'='*60}")
    print(f"  Diagnostics: {label}")
    print(f"  Theta shape: {Theta.shape}")
    print(f"{'='*60}")

    # --- Column stds ---
    col_stds = np.std(Theta, axis=0)
    print(f"\n  [Column Stds]")
    for i, (name, std) in enumerate(zip(feature_names, col_stds)):
        flag = " ⚠️ NEAR-ZERO" if std < STD_MIN else ""
        print(f"    #{i:2d} {name:22s}: {std:.6e}{flag}")

    # --- ColumnScaler ---
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)
    scales = scaler.scale_.copy()
    const_mask = scaler.constant_mask_.copy()

    non_const_scales = scales[~const_mask]
    if len(non_const_scales) > 0:
        scale_ratio = float(np.max(non_const_scales) / np.median(non_const_scales))
    else:
        scale_ratio = 1.0

    print(f"\n  [ColumnScaler]")
    print(f"    Constant columns: {np.where(const_mask)[0].tolist()}")
    print(f"    Scale range: [{np.min(scales):.4e}, {np.max(scales):.4e}]")
    print(f"    Scale ratio (max/median, non-const): {scale_ratio:.1f}")
    flag_sr = " ⚠️ EXCEEDS THRESHOLD" if scale_ratio > SCALE_RATIO_MAX else " ✅"
    print(f"    Threshold: {SCALE_RATIO_MAX:.0e}{flag_sr}")

    # --- Condition number with constant ---
    kappa_with = float(np.linalg.cond(Theta_scaled))
    print(f"\n  [Condition Number]")
    print(f"    κ₂(Θ_scaled, with constant):    {kappa_with:.4e}  (log10={np.log10(kappa_with):.2f})")

    # --- Condition number without constant ---
    non_const_idx = np.where(~const_mask)[0]
    if len(non_const_idx) > 0:
        Theta_no_const = Theta_scaled[:, non_const_idx]
        kappa_without = float(np.linalg.cond(Theta_no_const))
    else:
        kappa_without = float('inf')
    print(f"    κ₂(Θ_scaled, without constant): {kappa_without:.4e}  (log10={np.log10(kappa_without):.2f})")

    # --- Correlation: phi*tau (#10) vs #13 ---
    col_10 = Theta[:, IDX_PHI_TAU]
    col_13 = Theta[:, IDX_NEW_13]

    if np.std(col_10) > 1e-15 and np.std(col_13) > 1e-15:
        corr_10_13 = float(np.corrcoef(col_10, col_13)[0, 1])
    else:
        corr_10_13 = float('nan')

    print(f"\n  [Cancellation Pair Correlation]")
    print(f"    corr({feature_names[IDX_PHI_TAU]}, {feature_names[IDX_NEW_13]}): {corr_10_13:.6f}")
    if not np.isnan(corr_10_13):
        flag_corr = " ✅ BROKEN" if abs(corr_10_13) <= CORR_THRESHOLD else " ⚠️ STILL HIGH"
        print(f"    Threshold: |corr| <= {CORR_THRESHOLD}{flag_corr}")

    # --- Additional key correlations ---
    key_pairs = [
        (0, 5, '1 vs #5'),           # constant vs cos-variant
        (IDX_PHI_TAU, IDX_NEW_13, f'{feature_names[IDX_PHI_TAU]} vs {feature_names[IDX_NEW_13]}'),
    ]
    corr_dict = {}
    print(f"\n  [Key Correlations]")
    for i, j, desc in key_pairs:
        ci, cj = Theta[:, i], Theta[:, j]
        if np.std(ci) > 1e-15 and np.std(cj) > 1e-15:
            c = float(np.corrcoef(ci, cj)[0, 1])
        else:
            c = float('nan')
        corr_dict[f"corr_{i}_{j}"] = c
        print(f"    #{i} vs #{j} ({desc}): {c:.6f}")

    return {
        'label': label,
        'kappa_with_const': kappa_with,
        'kappa_without_const': kappa_without,
        'log10_kappa_with': float(np.log10(kappa_with)),
        'log10_kappa_without': float(np.log10(kappa_without)),
        'column_stds': col_stds.tolist(),
        'column_stds_dict': {n: float(s) for n, s in zip(feature_names, col_stds)},
        'scale_ratio': scale_ratio,
        'scaler_scales': scales.tolist(),
        'constant_mask': const_mask.tolist(),
        'corr_10_13': corr_10_13,
        'correlations': corr_dict,
        'feature_names': feature_names,
    }


def check_fail_fast(diag_rp1: dict, diag_rp2: dict) -> dict:
    """
    Evaluate R2-0 fail-fast criteria.

    Returns:
        dict with ac1, ac2, ac3 pass/fail and overall verdict
    """
    print(f"\n{'='*60}")
    print(f"  FAIL-FAST CRITERIA CHECK")
    print(f"{'='*60}")

    # AC1: corr(phi*tau, new#13) <= 0.995
    corr_val = abs(diag_rp2['corr_10_13'])
    ac1_pass = corr_val <= CORR_THRESHOLD
    print(f"\n  AC1: |corr(phi*tau, (cos(phi)-1)*tau)| = {corr_val:.6f}")
    print(f"       Threshold: <= {CORR_THRESHOLD}")
    print(f"       Verdict: {'✅ PASS' if ac1_pass else '❌ FAIL'}")

    # AC2: std >= 1e-6 AND scale_ratio <= 1e4
    std_13 = diag_rp2['column_stds'][IDX_NEW_13]
    std_ok = std_13 >= STD_MIN
    sr_ok = diag_rp2['scale_ratio'] <= SCALE_RATIO_MAX
    ac2_pass = std_ok and sr_ok
    print(f"\n  AC2a: std(#13) = {std_13:.6e}")
    print(f"        Threshold: >= {STD_MIN:.0e}")
    print(f"        Verdict: {'✅ PASS' if std_ok else '❌ FAIL'}")
    print(f"  AC2b: scale_ratio = {diag_rp2['scale_ratio']:.1f}")
    print(f"        Threshold: <= {SCALE_RATIO_MAX:.0e}")
    print(f"        Verdict: {'✅ PASS' if sr_ok else '❌ FAIL'}")
    print(f"  AC2 overall: {'✅ PASS' if ac2_pass else '❌ FAIL'}")

    # AC3: log10(kappa_RP2) <= log10(kappa_RP1) + 0.5
    log_rp1 = diag_rp1['log10_kappa_with']
    log_rp2 = diag_rp2['log10_kappa_with']
    delta_log = log_rp2 - log_rp1
    ac3_pass = delta_log <= KAPPA_MARGIN_LOG10
    print(f"\n  AC3: log10(κ_RP2) = {log_rp2:.2f}, log10(κ_RP1) = {log_rp1:.2f}")
    print(f"       Δlog10(κ) = {delta_log:+.2f}")
    print(f"       Threshold: <= +{KAPPA_MARGIN_LOG10}")
    print(f"       Verdict: {'✅ PASS' if ac3_pass else '❌ FAIL'}")

    # Overall
    overall = ac1_pass and ac2_pass and ac3_pass
    print(f"\n  {'='*40}")
    print(f"  OVERALL VERDICT: {'✅ GO — proceed with RP2-A' if overall else '❌ NO-GO — switch to fallback'}")
    if not overall:
        fails = []
        if not ac1_pass:
            fails.append("AC1 (correlation still high)")
        if not ac2_pass:
            fails.append("AC2 (near-zero/scale blowup)")
        if not ac3_pass:
            fails.append("AC3 (kappa degradation)")
        print(f"  Failed: {', '.join(fails)}")
        print(f"  Recommended: RP2-C (remove #13, 13-term library)")
    print(f"  {'='*40}")

    return {
        'ac1_pass': ac1_pass,
        'ac1_value': float(corr_val),
        'ac2_pass': ac2_pass,
        'ac2_std_13': float(std_13),
        'ac2_scale_ratio': diag_rp2['scale_ratio'],
        'ac3_pass': ac3_pass,
        'ac3_delta_log10_kappa': float(delta_log),
        'overall_pass': overall,
        'verdict': 'GO' if overall else 'NO-GO',
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  R2-0: Fail-fast Diagnostic for Reparam-2")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load dataset ──
    dataset_path = paths.get_dataset_path(DATASET_VERSION, system=SYSTEM)
    print(f"\n  Dataset: {dataset_path}")
    validate_dataset_lite(dataset_path)
    dataset = dict(np.load(dataset_path, allow_pickle=True))

    train_x = dataset['train_x']    # (N, T, 4)
    train_u = dataset['train_u']    # (N, T, 1)
    N, T, D = train_x.shape
    print(f"  train_x: {train_x.shape}, train_u: {train_u.shape}")

    # Flatten for library
    x_flat = train_x.reshape(-1, D)   # (N*T, 4)
    u_flat = train_u.reshape(-1, 1)   # (N*T, 1)
    print(f"  Flattened: x={x_flat.shape}, u={u_flat.shape}")

    # ── Quick data range check ──
    phi = x_flat[:, 0]
    tau = u_flat[:, 0]
    print(f"\n  [Data Ranges]")
    print(f"    phi:  [{phi.min():.6f}, {phi.max():.6f}] rad")
    print(f"    |phi| max: {np.abs(phi).max():.6f} rad ({np.degrees(np.abs(phi).max()):.2f} deg)")
    print(f"    tau:  [{tau.min():.6f}, {tau.max():.6f}] N*m")

    # ── Build all 3 library variants ──
    results = {}

    for reparam in ['standard', 'reparam1', 'reparam2']:
        Theta, names = build_aek_library_by_name(x_flat, u_flat, reparam=reparam)
        diag = compute_diagnostics(Theta, names, label=reparam)
        results[reparam] = diag

    # ── Cross-comparison table ──
    print(f"\n{'='*60}")
    print(f"  CROSS-COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<35s} {'Standard':>12s} {'Reparam-1':>12s} {'Reparam-2':>12s}")
    print(f"  {'-'*71}")

    for key, fmt in [
        ('log10_kappa_with', '{:.2f}'),
        ('log10_kappa_without', '{:.2f}'),
        ('scale_ratio', '{:.1f}'),
        ('corr_10_13', '{:.6f}'),
    ]:
        vals = [fmt.format(results[r][key]) for r in ['standard', 'reparam1', 'reparam2']]
        print(f"  {key:<35s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s}")

    # Std of #13
    for r in ['standard', 'reparam1', 'reparam2']:
        std_13 = results[r]['column_stds'][IDX_NEW_13]
        name_13 = results[r]['feature_names'][IDX_NEW_13]
        print(f"  std(#13={name_13}): {std_13:.4e} [{r}]")

    # Δlog10(κ) vs Standard
    log_std = results['standard']['log10_kappa_with']
    for r in ['reparam1', 'reparam2']:
        delta = results[r]['log10_kappa_with'] - log_std
        print(f"  Δlog10(κ) {r} vs Standard: {delta:+.2f}")

    # ── Fail-fast check (RP1 vs RP2) ──
    ff = check_fail_fast(results['reparam1'], results['reparam2'])

    # ── Save results ──
    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'run_r2_0_failfast.py',
        'dataset': DATASET_VERSION,
        'n_train': N_TRAIN,
        'n_samples': int(x_flat.shape[0]),
        'data_ranges': {
            'phi_min': float(phi.min()),
            'phi_max': float(phi.max()),
            'phi_abs_max': float(np.abs(phi).max()),
            'tau_min': float(tau.min()),
            'tau_max': float(tau.max()),
        },
        'thresholds': {
            'corr': CORR_THRESHOLD,
            'std_min': STD_MIN,
            'scale_ratio_max': SCALE_RATIO_MAX,
            'kappa_margin_log10': KAPPA_MARGIN_LOG10,
        },
        'diagnostics': {
            r: {
                'kappa_with_const': results[r]['kappa_with_const'],
                'kappa_without_const': results[r]['kappa_without_const'],
                'log10_kappa_with': results[r]['log10_kappa_with'],
                'log10_kappa_without': results[r]['log10_kappa_without'],
                'scale_ratio': results[r]['scale_ratio'],
                'corr_10_13': results[r]['corr_10_13'],
                'column_stds': results[r]['column_stds'],
                'feature_names': results[r]['feature_names'],
            }
            for r in ['standard', 'reparam1', 'reparam2']
        },
        'fail_fast': ff,
    }

    out_dir = paths.RESULTS_ROOT / DATASET_VERSION / 'gate4c' / 'diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'r2_0_failfast.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"  R2-0 COMPLETE — Verdict: {ff['verdict']}")
    print(f"{'='*60}")

    return ff['overall_pass']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
