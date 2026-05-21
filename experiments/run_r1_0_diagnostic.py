"""
Gate4c Phase R1-0: Fail-fast Diagnostic — Standard vs Reparam-1 Library

Purpose:
    Compare condition number and collinearity diagnostics between
    Standard AEK library (14 terms) and Reparam-1 library (cos(phi) → cos(phi)-1).
    No augmentation, no E-SINDy fitting. Pure matrix diagnostics only.

Reparam-1 변환 규칙:
    Index #5: cos(phi) → cos(phi)-1  (단일 변환)
    나머지 13개 feature 동일

합격 기준:
    Primary: Δlog₁₀(κ) ≥ 2 (100× 이상 감소)
    Auxiliary: corr(1, cos(phi)) vs corr(1, cos(phi)-1) 유의미 하락

산출물:
    diagnostics_r1_0.json  — 모든 수치 기록
    콘솔 Go/No-go 판정

Usage (PowerShell, copy-paste ready):
    python experiments/run_r1_0_diagnostic.py

Author: Claude (Gate4c R1-0)
Date: 2026-02-10
"""

import sys
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from src.sindy.optimizer import ColumnScaler
from src.sindy.aek_library import (
    AEK_FEATURE_NAMES,
    N_AEK_FEATURES,
)


# =============================================================================
# Reparam-1 Feature Names (SSOT)
# =============================================================================

REPARAM1_FEATURE_NAMES: List[str] = [
    '1',                # 0: constant
    'phi_dot',          # 1
    'theta_w_dot',      # 2
    'tau',              # 3: Oracle — 변경 금지
    'sin(phi)',         # 4: Oracle — 변경 금지
    'cos(phi)-1',       # 5: ★ Reparam-1 핵심 변환
    'phi_dot^2',        # 6
    'theta_w_dot^2',    # 7
    'tau^2',            # 8
    'phi*phi_dot',      # 9
    'phi*tau',          # 10
    'phi_dot*tau',      # 11
    'theta_w_dot*tau',  # 12
    'sin(phi)*tau',     # 13
]


# =============================================================================
# Library Builders
# =============================================================================

def build_standard_library(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Build Standard AEK library (14 terms). Matches aek_library.py exactly."""
    N = x.shape[0]
    phi = x[:, 0]
    phi_dot = x[:, 1]
    theta_w_dot = x[:, 3]
    tau = u[:, 0]

    Theta = np.column_stack([
        np.ones(N),             # 0: 1
        phi_dot,                # 1: phi_dot
        theta_w_dot,            # 2: theta_w_dot
        tau,                    # 3: tau
        np.sin(phi),            # 4: sin(phi)
        np.cos(phi),            # 5: cos(phi)
        phi_dot ** 2,           # 6: phi_dot^2
        theta_w_dot ** 2,       # 7: theta_w_dot^2
        tau ** 2,               # 8: tau^2
        phi * phi_dot,          # 9: phi*phi_dot
        phi * tau,              # 10: phi*tau
        phi_dot * tau,          # 11: phi_dot*tau
        theta_w_dot * tau,      # 12: theta_w_dot*tau
        np.sin(phi) * tau,      # 13: sin(phi)*tau
    ])
    return Theta


def build_reparam1_library(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Build Reparam-1 AEK library (14 terms). Only #5 changed."""
    N = x.shape[0]
    phi = x[:, 0]
    phi_dot = x[:, 1]
    theta_w_dot = x[:, 3]
    tau = u[:, 0]

    Theta = np.column_stack([
        np.ones(N),             # 0: 1
        phi_dot,                # 1: phi_dot
        theta_w_dot,            # 2: theta_w_dot
        tau,                    # 3: tau
        np.sin(phi),            # 4: sin(phi)
        np.cos(phi) - 1.0,     # 5: cos(phi)-1  ★ REPARAM-1
        phi_dot ** 2,           # 6: phi_dot^2
        theta_w_dot ** 2,       # 7: theta_w_dot^2
        tau ** 2,               # 8: tau^2
        phi * phi_dot,          # 9: phi*phi_dot
        phi * tau,              # 10: phi*tau
        phi_dot * tau,          # 11: phi_dot*tau
        theta_w_dot * tau,      # 12: theta_w_dot*tau
        np.sin(phi) * tau,      # 13: sin(phi)*tau
    ])
    return Theta


# =============================================================================
# Diagnostic Functions
# =============================================================================

def compute_pairwise_corr(Theta: np.ndarray, i: int, j: int) -> float:
    """Compute absolute Pearson correlation between columns i and j."""
    a = Theta[:, i]
    b = Theta[:, j]
    # Handle constant columns
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.abs(np.corrcoef(a, b)[0, 1]))


def compute_condition_number(Theta: np.ndarray) -> float:
    """Compute 2-norm condition number."""
    return float(np.linalg.cond(Theta, 2))


def compute_feature_stds(Theta: np.ndarray) -> np.ndarray:
    """Compute column-wise standard deviations."""
    return np.std(Theta, axis=0)


def scale_and_compute_kappa(
    Theta: np.ndarray,
    with_constant: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Apply ColumnScaler and compute condition number.

    Args:
        Theta: Raw feature matrix (N, F)
        with_constant: If True, include constant column. If False, remove it.

    Returns:
        kappa: Condition number of scaled matrix
        scales: Column scales used
    """
    if not with_constant:
        # Remove constant column (index 0)
        Theta = Theta[:, 1:]

    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)
    kappa = compute_condition_number(Theta_scaled)
    return kappa, scaler.scale_


# =============================================================================
# Main Diagnostic
# =============================================================================

def run_diagnostic(dataset_path: str) -> Dict:
    """
    Run Phase R1-0 diagnostic.

    Args:
        dataset_path: Path to AEK dataset.npz

    Returns:
        Dict with all diagnostic results + Go/No-go verdict
    """
    print("=" * 70)
    print("  Gate4c Phase R1-0: Standard vs Reparam-1 Diagnostic")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------
    print("\n[1] Loading dataset...")
    ds = np.load(dataset_path, allow_pickle=True)

    train_x = ds['train_x']    # (N, T, 4)
    train_u = ds['train_u']    # (N, T, 1)

    N_traj, T, state_dim = train_x.shape
    print(f"    train_x: {train_x.shape}  (N={N_traj}, T={T}, D={state_dim})")
    print(f"    train_u: {train_u.shape}")

    # Flatten to (N*T, D)
    x_flat = train_x.reshape(-1, state_dim)
    u_flat = train_u.reshape(-1, train_u.shape[-1])
    N_samples = x_flat.shape[0]
    print(f"    Flattened: {N_samples} samples")

    # Quick phi stats
    phi_vals = x_flat[:, 0]
    print(f"    phi range: [{phi_vals.min():.4f}, {phi_vals.max():.4f}] rad "
          f"({np.degrees(phi_vals.min()):.2f}° ~ {np.degrees(phi_vals.max()):.2f}°)")

    # ----------------------------------------------------------
    # 2. Build both libraries
    # ----------------------------------------------------------
    print("\n[2] Building feature matrices...")
    Theta_std = build_standard_library(x_flat, u_flat)
    Theta_rp1 = build_reparam1_library(x_flat, u_flat)
    print(f"    Standard:  Theta shape = {Theta_std.shape}")
    print(f"    Reparam-1: Theta shape = {Theta_rp1.shape}")

    # Verify only column 5 differs
    diff_mask = ~np.isclose(Theta_std, Theta_rp1, atol=1e-15)
    diff_cols = np.where(diff_mask.any(axis=0))[0]
    print(f"    Columns that differ: {diff_cols.tolist()} "
          f"(expected: [5] only)")
    assert list(diff_cols) == [5], \
        f"Only column 5 should differ, but got {diff_cols}"

    # ----------------------------------------------------------
    # 3. Feature column stds comparison
    # ----------------------------------------------------------
    print("\n[3] Feature column standard deviations...")
    stds_std = compute_feature_stds(Theta_std)
    stds_rp1 = compute_feature_stds(Theta_rp1)

    print(f"\n    {'#':>2}  {'Feature (Std)':>20}  {'std_Std':>12}  "
          f"{'Feature (RP1)':>20}  {'std_RP1':>12}  {'Ratio':>8}")
    print("    " + "-" * 90)
    for i in range(N_AEK_FEATURES):
        ratio = stds_rp1[i] / stds_std[i] if stds_std[i] > 1e-15 else float('nan')
        marker = " ★" if i == 5 else ""
        print(f"    {i:2d}  {AEK_FEATURE_NAMES[i]:>20}  {stds_std[i]:12.6e}  "
              f"{REPARAM1_FEATURE_NAMES[i]:>20}  {stds_rp1[i]:12.6e}  "
              f"{ratio:8.4f}{marker}")

    # ----------------------------------------------------------
    # 4. Collinearity diagnostics
    # ----------------------------------------------------------
    print("\n[4] Collinearity diagnostics...")

    # 4a. corr(1, cos(phi)) vs corr(1, cos(phi)-1)
    corr_1_cos_std = compute_pairwise_corr(Theta_std, 0, 5)
    corr_1_cos_rp1 = compute_pairwise_corr(Theta_rp1, 0, 5)
    print(f"    corr(1, cos(phi))   [Standard]:  {corr_1_cos_std:.6f}")
    print(f"    corr(1, cos(phi)-1) [Reparam-1]: {corr_1_cos_rp1:.6f}")
    print(f"    Δ|corr|: {corr_1_cos_std - corr_1_cos_rp1:.6f}")

    # 4b. corr(phi*tau, sin(phi)*tau) — should be similar (no change)
    corr_phi_tau_std = compute_pairwise_corr(Theta_std, 10, 13)
    corr_phi_tau_rp1 = compute_pairwise_corr(Theta_rp1, 10, 13)
    print(f"\n    corr(phi*tau, sin(phi)*tau) [Standard]:  {corr_phi_tau_std:.6f}")
    print(f"    corr(phi*tau, sin(phi)*tau) [Reparam-1]: {corr_phi_tau_rp1:.6f}")
    print(f"    (Expected: similar — these features unchanged)")

    # 4c. Full correlation matrix top-5 highest pairs
    print("\n    Top-5 highest |corr| pairs:")
    for label, Theta, names in [
        ("Standard", Theta_std, AEK_FEATURE_NAMES),
        ("Reparam-1", Theta_rp1, REPARAM1_FEATURE_NAMES),
    ]:
        C = np.corrcoef(Theta.T)
        # Extract upper triangle
        pairs = []
        for i in range(N_AEK_FEATURES):
            for j in range(i + 1, N_AEK_FEATURES):
                pairs.append((abs(C[i, j]), i, j))
        pairs.sort(reverse=True)
        print(f"\n    [{label}]")
        for rank, (corr_val, i, j) in enumerate(pairs[:5]):
            print(f"      {rank+1}. |corr({names[i]}, {names[j]})| = {corr_val:.6f}")

    # ----------------------------------------------------------
    # 5. Condition number (κ₂) comparison
    # ----------------------------------------------------------
    print("\n[5] Condition number κ₂(Θ_scaled)...")

    # 5a. With constant column
    kappa_std_wc, scales_std_wc = scale_and_compute_kappa(Theta_std, with_constant=True)
    kappa_rp1_wc, scales_rp1_wc = scale_and_compute_kappa(Theta_rp1, with_constant=True)

    # 5b. Without constant column
    kappa_std_nc, scales_std_nc = scale_and_compute_kappa(Theta_std, with_constant=False)
    kappa_rp1_nc, scales_rp1_nc = scale_and_compute_kappa(Theta_rp1, with_constant=False)

    print(f"\n    {'Variant':>25}  {'Standard':>15}  {'Reparam-1':>15}  "
          f"{'log10(Std)':>10}  {'log10(RP1)':>10}  {'Δlog10':>8}")
    print("    " + "-" * 90)

    log_std_wc = np.log10(kappa_std_wc)
    log_rp1_wc = np.log10(kappa_rp1_wc)
    delta_wc = log_std_wc - log_rp1_wc
    print(f"    {'With constant':>25}  {kappa_std_wc:15.4e}  {kappa_rp1_wc:15.4e}  "
          f"{log_std_wc:10.2f}  {log_rp1_wc:10.2f}  {delta_wc:8.2f}")

    log_std_nc = np.log10(kappa_std_nc)
    log_rp1_nc = np.log10(kappa_rp1_nc)
    delta_nc = log_std_nc - log_rp1_nc
    print(f"    {'Without constant':>25}  {kappa_std_nc:15.4e}  {kappa_rp1_nc:15.4e}  "
          f"{log_std_nc:10.2f}  {log_rp1_nc:10.2f}  {delta_nc:8.2f}")

    # ----------------------------------------------------------
    # 6. Near-zero column risk check
    # ----------------------------------------------------------
    print("\n[6] Near-zero column risk (cos(phi)-1 at small angles)...")
    col5_std_val = stds_std[5]
    col5_rp1_val = stds_rp1[5]
    print(f"    std(cos(phi))   = {col5_std_val:.6e}")
    print(f"    std(cos(phi)-1) = {col5_rp1_val:.6e}")

    # Check if ColumnScaler treats cos(phi)-1 as constant
    is_near_constant = col5_rp1_val < 1e-10
    print(f"    ColumnScaler constant threshold: 1e-10")
    print(f"    cos(phi)-1 treated as constant? {is_near_constant}")
    if is_near_constant:
        print("    ⚠️  WARNING: cos(phi)-1 has near-zero std, ColumnScaler sets scale=1.0")
        print("    This means the column won't be effectively scaled.")

    # ----------------------------------------------------------
    # 7. Go/No-go verdict
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Primary criterion: Δlog₁₀(κ) ≥ 2 (with constant)
    primary_pass = delta_wc >= 2.0
    print(f"\n  Primary: Δlog₁₀(κ) with constant = {delta_wc:.2f}  "
          f"(threshold ≥ 2.0)  → {'✅ PASS' if primary_pass else '❌ FAIL'}")

    # Also check without constant
    primary_pass_nc = delta_nc >= 2.0
    print(f"  Primary: Δlog₁₀(κ) no constant   = {delta_nc:.2f}  "
          f"(threshold ≥ 2.0)  → {'✅ PASS' if primary_pass_nc else '❌ FAIL'}")

    # Auxiliary: corr(1, cos) drop
    aux_pass = (corr_1_cos_std - corr_1_cos_rp1) > 0.1
    print(f"\n  Auxiliary: Δ|corr(1, col5)| = {corr_1_cos_std - corr_1_cos_rp1:.4f}  "
          f"(directional)  → {'✅ PASS' if aux_pass else '⚠️  Marginal'}")

    # Near-zero risk
    print(f"\n  Near-zero risk: {'⚠️  YES' if is_near_constant else '✅ No risk'}")

    # Overall
    overall_go = primary_pass or primary_pass_nc
    print(f"\n  {'=' * 40}")
    print(f"  OVERALL: {'🟢 GO → Proceed to Phase 1' if overall_go else '🔴 NO-GO → Switch to Reparam-2'}")
    print(f"  {'=' * 40}")

    # ----------------------------------------------------------
    # 8. Save diagnostics JSON
    # ----------------------------------------------------------
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'R1-0',
        'dataset': str(dataset_path),
        'n_train_trajectories': int(N_traj),
        'T_per_trajectory': int(T),
        'n_samples': int(N_samples),
        'phi_range_rad': [float(phi_vals.min()), float(phi_vals.max())],
        'phi_range_deg': [float(np.degrees(phi_vals.min())),
                          float(np.degrees(phi_vals.max()))],

        'feature_stds': {
            'standard': {name: float(v) for name, v in
                         zip(AEK_FEATURE_NAMES, stds_std)},
            'reparam1': {name: float(v) for name, v in
                         zip(REPARAM1_FEATURE_NAMES, stds_rp1)},
        },

        'collinearity': {
            'corr_1_cos_phi_standard': float(corr_1_cos_std),
            'corr_1_cos_phi_minus1_reparam1': float(corr_1_cos_rp1),
            'delta_corr_1_col5': float(corr_1_cos_std - corr_1_cos_rp1),
            'corr_phi_tau_sinphi_tau_standard': float(corr_phi_tau_std),
            'corr_phi_tau_sinphi_tau_reparam1': float(corr_phi_tau_rp1),
        },

        'condition_number': {
            'with_constant': {
                'standard': float(kappa_std_wc),
                'reparam1': float(kappa_rp1_wc),
                'log10_standard': float(log_std_wc),
                'log10_reparam1': float(log_rp1_wc),
                'delta_log10': float(delta_wc),
            },
            'without_constant': {
                'standard': float(kappa_std_nc),
                'reparam1': float(kappa_rp1_nc),
                'log10_standard': float(log_std_nc),
                'log10_reparam1': float(log_rp1_nc),
                'delta_log10': float(delta_nc),
            },
        },

        'near_zero_risk': {
            'cos_phi_std': float(col5_std_val),
            'cos_phi_minus1_std': float(col5_rp1_val),
            'treated_as_constant': bool(is_near_constant),
        },

        'verdict': {
            'primary_pass_with_constant': bool(primary_pass),
            'primary_pass_no_constant': bool(primary_pass_nc),
            'auxiliary_corr_pass': bool(aux_pass),
            'near_zero_risk': bool(is_near_constant),
            'overall': 'GO' if overall_go else 'NO_GO',
        },
    }

    # Save
    out_dir = Path('results/aek_ood_v1/gate4c/r1_0_diagnostic')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'diagnostics_r1_0.json'

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Saved: {out_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Gate4c R1-0: Standard vs Reparam-1 Diagnostic')
    parser.add_argument(
        '--dataset', type=str,
        default='data/aek/aek_ood_v1/dataset.npz',
        help='Path to AEK dataset.npz')
    args = parser.parse_args()

    run_diagnostic(args.dataset)


if __name__ == '__main__':
    main()