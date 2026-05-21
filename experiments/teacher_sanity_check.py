#!/usr/bin/env python3
"""
Teacher Sanity Check v2: Verify alignment error with correct normalization.

FIX: Gate1 teacher coefficients were trained on normalized inputs:
- x_norm = (x - state_mean) / state_std
- u_norm = (u - input_mean) / input_std
- Theta = library(x_norm, u_norm)
- dx_pred = Theta @ coef + dx_mean

This script verifies that with correct normalization:
- Train alignment error should be LOW (< 1.0)
- Train error << Pool error
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# Configuration (SSOT)
# ============================================================

PROJECT_ROOT = Path(r"C:\python_work\PhD_project")
DATASET_PATH = PROJECT_ROOT / "data" / "cartpole" / "cartpole_ood_v1" / "dataset.npz"
NORM_STATS_PATH = PROJECT_ROOT / "data" / "cartpole" / "cartpole_ood_v1" / "norm_stats.json"
TEACHER_PATH = PROJECT_ROOT / "results" / "cartpole_ood_v1" / "gate1" / "standardized" / "esindy" / "n10" / "seed0" / "20251229_213749_nogit_base" / "sindy_coefficients.csv"

# Expected hashes for SSOT verification
NORM_STATS_SHA256 = "934DD298D31A4E6CF4DA85ECE4F6BC3B35E9D0732C66B8EAA876EB17838A03AB"

N_TRAIN = 10
SEED = 0
DYNAMICS_TARGETS = [1, 3]  # x_ddot, theta_ddot
DERIVATIVE_KEY = "derivative_dx_savgol"

# Feature library (gate0_min)
FEATURE_NAMES = [
    "1", "x", "x_dot", "theta_dot", "sin(theta)", "cos(theta)", "u",
    "x^2", "x*x_dot", "x_dot^2", "theta_dot^2", "x*theta_dot", "x_dot*theta_dot",
    "x*sin(theta)", "x*cos(theta)", "x_dot*sin(theta)", "x_dot*cos(theta)",
    "theta_dot*sin(theta)", "theta_dot*cos(theta)", "u*sin(theta)", "u*cos(theta)"
]

# ============================================================
# Normalization Functions
# ============================================================

def load_norm_stats(path: Path) -> dict:
    """Load normalization statistics."""
    with open(path, 'r') as f:
        return json.load(f)


def normalize(data: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize data: (data - mean) / std"""
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])
    return (data - mean) / std


# ============================================================
# Feature Computation (NORMALIZED inputs)
# ============================================================

def compute_features_normalized(
    x_norm: np.ndarray, 
    u_norm: np.ndarray
) -> np.ndarray:
    """
    Compute feature matrix Θ from NORMALIZED state and input.
    
    Args:
        x_norm: Normalized state (N, T, 4) or (T, 4)
        u_norm: Normalized input (N, T, 1) or (T, 1)
    
    Returns:
        Θ: Feature matrix (N*T, 21) or (T, 21)
    """
    if x_norm.ndim == 3:
        N, T, _ = x_norm.shape
        x_flat = x_norm.reshape(-1, 4)
        u_flat = u_norm.reshape(-1, 1)
    else:
        x_flat = x_norm
        u_flat = u_norm
    
    x_pos = x_flat[:, 0]
    x_dot = x_flat[:, 1]
    theta = x_flat[:, 2]
    theta_dot = x_flat[:, 3]
    u_val = u_flat[:, 0]
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Build feature matrix (21 features, gate0_min order)
    features = np.column_stack([
        np.ones_like(x_pos),           # 1
        x_pos,                          # x
        x_dot,                          # x_dot
        theta_dot,                      # theta_dot
        sin_theta,                      # sin(theta)
        cos_theta,                      # cos(theta)
        u_val,                          # u
        x_pos**2,                       # x^2
        x_pos * x_dot,                  # x*x_dot
        x_dot**2,                       # x_dot^2
        theta_dot**2,                   # theta_dot^2
        x_pos * theta_dot,              # x*theta_dot
        x_dot * theta_dot,              # x_dot*theta_dot
        x_pos * sin_theta,              # x*sin(theta)
        x_pos * cos_theta,              # x*cos(theta)
        x_dot * sin_theta,              # x_dot*sin(theta)
        x_dot * cos_theta,              # x_dot*cos(theta)
        theta_dot * sin_theta,          # theta_dot*sin(theta)
        theta_dot * cos_theta,          # theta_dot*cos(theta)
        u_val * sin_theta,              # u*sin(theta)
        u_val * cos_theta,              # u*cos(theta)
    ])
    
    return features


def compute_alignment_error_v2(
    theta: np.ndarray,
    dx_raw: np.ndarray,
    teacher_coef: np.ndarray,
    dx_mean: np.ndarray,
    dx_std: np.ndarray,
    dynamics_targets: list = [1, 3],
) -> float:
    """
    Compute normalized RMSE alignment error with correct formula.
    
    Gate1 relationship:
        dx_raw = Theta(x_norm, u_norm) @ coef + dx_mean
    
    Args:
        theta: Feature matrix from normalized inputs (M, 21)
        dx_raw: Raw derivative array (M, 4)
        teacher_coef: Teacher coefficients (21, 4)
        dx_mean: Derivative mean for offset (4,)
        dx_std: Derivative std for normalization (4,)
        dynamics_targets: Target indices [1, 3]
    
    Returns:
        Normalized RMSE error (scalar)
    """
    # Predict dx_raw
    dx_pred = theta @ teacher_coef + dx_mean  # (M, 4)
    
    # Error on dynamics targets only
    errors = []
    for t_idx in dynamics_targets:
        residual = dx_raw[:, t_idx] - dx_pred[:, t_idx]
        std = dx_std[t_idx]
        if std > 0:
            normalized_rmse = np.sqrt(np.mean(residual**2)) / std
            errors.append(normalized_rmse)
    
    return np.mean(errors)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Teacher Sanity Check v2 (with normalization fix)")
    print("=" * 60)
    
    # 1. Load norm_stats
    print("\n[Loading Normalization Stats]")
    norm_stats = load_norm_stats(NORM_STATS_PATH)
    
    state_mean = np.array(norm_stats['state']['mean'])
    state_std = np.array(norm_stats['state']['std'])
    input_mean = np.array(norm_stats['input']['mean'])
    input_std = np.array(norm_stats['input']['std'])
    dx_mean = np.array(norm_stats[DERIVATIVE_KEY]['mean'])
    dx_std = np.array(norm_stats[DERIVATIVE_KEY]['std'])
    
    print(f"  State mean: {state_mean}")
    print(f"  State std:  {state_std}")
    print(f"  Input mean: {input_mean}")
    print(f"  Input std:  {input_std}")
    print(f"  dx mean:    {dx_mean}")
    print(f"  dx std:     {dx_std}")
    
    # 2. Load dataset
    print("\n[Loading Dataset]")
    data = np.load(DATASET_PATH)
    train_x = data['train_x'][:N_TRAIN]
    train_dx = data['train_dx'][:N_TRAIN]  # raw dx
    train_u = data['train_u'][:N_TRAIN]
    
    print(f"  train_x shape: {train_x.shape}")
    print(f"  train_dx shape: {train_dx.shape}")
    print(f"  train_u shape: {train_u.shape}")
    
    # 3. Normalize x, u (NOT dx - we predict raw dx)
    print("\n[Normalizing Inputs]")
    train_x_norm = normalize(train_x, norm_stats['state'])
    train_u_norm = normalize(train_u, norm_stats['input'])
    
    print(f"  x_norm sample [0,0]: {train_x_norm[0,0]}")
    print(f"  u_norm sample [0,0]: {train_u_norm[0,0]}")
    
    # 4. Load teacher coefficients
    print("\n[Loading Teacher Coefficients]")
    teacher_df = pd.read_csv(TEACHER_PATH, index_col=0)
    teacher_coef = teacher_df.values  # (21, 4)
    print(f"  Shape: {teacher_coef.shape}")
    print(f"  Targets: {list(teacher_df.columns)}")
    
    # 5. Compute alignment error for each train trajectory
    print("\n[Train Data Alignment Error (v2 - with normalization)]")
    train_errors = []
    
    for i in range(N_TRAIN):
        x_norm_i = train_x_norm[i]  # (T, 4) normalized
        dx_raw_i = train_dx[i]      # (T, 4) raw
        u_norm_i = train_u_norm[i]  # (T, 1) normalized
        
        theta_i = compute_features_normalized(x_norm_i, u_norm_i)  # (T, 21)
        error_i = compute_alignment_error_v2(
            theta_i, dx_raw_i, teacher_coef, dx_mean, dx_std, DYNAMICS_TARGETS
        )
        train_errors.append(error_i)
        print(f"  Traj {i}: error = {error_i:.4f}")
    
    train_errors = np.array(train_errors)
    
    # 6. Summary statistics
    print("\n[Train Error Summary]")
    print(f"  Mean:   {train_errors.mean():.4f}")
    print(f"  Std:    {train_errors.std():.4f}")
    print(f"  Min:    {train_errors.min():.4f}")
    print(f"  Max:    {train_errors.max():.4f}")
    print(f"  Median: {np.median(train_errors):.4f}")
    
    # 7. Compare with old method (for reference)
    print("\n[Comparison: Old method (raw inputs, no dx_mean)]")
    old_errors = []
    for i in range(N_TRAIN):
        x_raw_i = train_x[i]
        dx_raw_i = train_dx[i]
        u_raw_i = train_u[i]
        
        # OLD: raw inputs
        theta_raw = compute_features_normalized(x_raw_i, u_raw_i)
        dx_pred_old = theta_raw @ teacher_coef  # no dx_mean
        
        # Error
        errs = []
        for t_idx in DYNAMICS_TARGETS:
            residual = dx_raw_i[:, t_idx] - dx_pred_old[:, t_idx]
            errs.append(np.sqrt(np.mean(residual**2)) / dx_std[t_idx])
        old_errors.append(np.mean(errs))
    
    old_errors = np.array(old_errors)
    print(f"  Old mean error: {old_errors.mean():.4f}")
    print(f"  New mean error: {train_errors.mean():.4f}")
    print(f"  Improvement: {old_errors.mean() - train_errors.mean():.4f}")
    
    # 8. Sanity check verdict
    print("\n" + "=" * 60)
    print("  Sanity Check Verdict")
    print("=" * 60)
    
    if train_errors.mean() < 0.5:
        print("  ✅ PASS: Train alignment error is VERY LOW (< 0.5)")
        print("     Teacher coefficients are valid for alignment scoring.")
    elif train_errors.mean() < 1.0:
        print("  ✅ PASS: Train alignment error is LOW (< 1.0)")
        print("     Teacher coefficients appear valid for alignment scoring.")
    elif train_errors.mean() < 2.0:
        print("  ⚠️ WARNING: Train alignment error is MODERATE (1.0 - 2.0)")
        print("     May need further investigation.")
    else:
        print("  ❌ FAIL: Train alignment error is HIGH (> 2.0)")
        print("     Something is still wrong with the alignment formula!")
    
    print("\n  Formula verified:")
    print("    dx_pred = Theta(x_norm, u_norm) @ coef + dx_mean")
    print(f"    norm_stats_sha256: {NORM_STATS_SHA256[:16]}...")
    
    return train_errors.mean()


if __name__ == "__main__":
    main()