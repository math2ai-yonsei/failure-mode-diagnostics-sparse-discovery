"""
S04-A: Normalization Utilities

Computes and applies normalization statistics for Cart-Pole dataset.
Statistics are computed from train split only (no data leakage).

Usage:
    from src.data.normalization import compute_norm_stats, normalize, denormalize
    
    # Compute stats from train data
    stats = compute_norm_stats(train_x, train_u, train_dx, train_dx_savgol)
    
    # Normalize
    x_norm = normalize(x, stats['state'])
    
    # Denormalize
    x_orig = denormalize(x_norm, stats['state'])
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

from src.contracts import paths


# =============================================================================
# Core Normalization Functions
# =============================================================================

def compute_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute mean and std from data.
    
    Args:
        data: Array of shape (N, T, D) or (N, D)
    
    Returns:
        Dict with 'mean' and 'std', each of shape (D,)
    """
    # Flatten to (N*T, D) or (N, D)
    if data.ndim == 3:
        flat = data.reshape(-1, data.shape[-1])
    elif data.ndim == 2:
        flat = data
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    
    # Prevent division by zero (use 1.0 for constant features)
    std = np.where(std < 1e-8, 1.0, std)
    
    return {
        'mean': mean.astype(np.float64),
        'std': std.astype(np.float64)
    }


def normalize(
    data: np.ndarray,
    stats: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Normalize data using precomputed statistics.
    
    Formula: (data - mean) / std
    
    Args:
        data: Array of shape (..., D)
        stats: Dict with 'mean' and 'std' of shape (D,)
    
    Returns:
        Normalized array, same shape as input
    """
    mean = np.asarray(stats['mean'])
    std = np.asarray(stats['std'])
    return (data - mean) / std


def denormalize(
    data: np.ndarray,
    stats: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Inverse of normalize.
    
    Formula: data * std + mean
    
    Args:
        data: Normalized array of shape (..., D)
        stats: Dict with 'mean' and 'std' of shape (D,)
    
    Returns:
        Denormalized array, same shape as input
    """
    mean = np.asarray(stats['mean'])
    std = np.asarray(stats['std'])
    return data * std + mean


# =============================================================================
# Dataset-Level Functions
# =============================================================================

def compute_norm_stats(
    train_x: np.ndarray,
    train_u: np.ndarray,
    train_dx: Optional[np.ndarray] = None,
    train_dx_savgol: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute normalization statistics from train split.
    
    Args:
        train_x: State trajectories (N, T, 4)
        train_u: Input trajectories (N, T, 1)
        train_dx: Analytic derivatives (N, T, 4), optional
        train_dx_savgol: Savgol derivatives (N, T, 4), optional
    
    Returns:
        Dict with 'state', 'input', 'derivative_dx', 'derivative_dx_savgol'
        Each contains 'mean' and 'std' arrays
    """
    stats = {
        'state': compute_stats(train_x),
        'input': compute_stats(train_u),
    }
    
    if train_dx is not None:
        stats['derivative_dx'] = compute_stats(train_dx)
    
    if train_dx_savgol is not None:
        stats['derivative_dx_savgol'] = compute_stats(train_dx_savgol)
    
    return stats


def normalize_dataset(
    x: np.ndarray,
    u: np.ndarray,
    dx: Optional[np.ndarray],
    stats: Dict[str, Dict[str, np.ndarray]],
    derivative_key: str = 'derivative_dx'
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Normalize a dataset split using precomputed statistics.
    
    Args:
        x: State trajectories (N, T, 4)
        u: Input trajectories (N, T, 1)
        dx: Derivative trajectories (N, T, 4), optional
        stats: Normalization statistics dict
        derivative_key: 'derivative_dx' or 'derivative_dx_savgol'
    
    Returns:
        (x_norm, u_norm, dx_norm) tuple
    
    Raises:
        KeyError: If dx is provided but derivative_key not in stats (fail-fast)
    """
    x_norm = normalize(x, stats['state'])
    u_norm = normalize(u, stats['input'])
    
    dx_norm = None
    if dx is not None:
        if derivative_key not in stats:
            raise KeyError(
                f"derivative_key '{derivative_key}' not found in stats. "
                f"Available keys: {list(stats.keys())}"
            )
        dx_norm = normalize(dx, stats[derivative_key])
    
    return x_norm, u_norm, dx_norm


def denormalize_dataset(
    x_norm: np.ndarray,
    u_norm: np.ndarray,
    dx_norm: Optional[np.ndarray],
    stats: Dict[str, Dict[str, np.ndarray]],
    derivative_key: str = 'derivative_dx'
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Denormalize a dataset split.
    
    Args:
        x_norm: Normalized state trajectories
        u_norm: Normalized input trajectories
        dx_norm: Normalized derivative trajectories, optional
        stats: Normalization statistics dict
        derivative_key: 'derivative_dx' or 'derivative_dx_savgol'
    
    Returns:
        (x, u, dx) tuple in original scale
    
    Raises:
        KeyError: If dx_norm is provided but derivative_key not in stats (fail-fast)
    """
    x = denormalize(x_norm, stats['state'])
    u = denormalize(u_norm, stats['input'])
    
    dx = None
    if dx_norm is not None:
        if derivative_key not in stats:
            raise KeyError(
                f"derivative_key '{derivative_key}' not found in stats. "
                f"Available keys: {list(stats.keys())}"
            )
        dx = denormalize(dx_norm, stats[derivative_key])
    
    return x, u, dx


# =============================================================================
# I/O Functions
# =============================================================================

def stats_to_json_serializable(stats: Dict) -> Dict:
    """Convert numpy arrays to lists for JSON serialization."""
    result = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            result[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = value
    return result


def stats_from_json(json_stats: Dict) -> Dict:
    """Convert JSON-loaded stats back to numpy arrays."""
    result = {}
    for key, value in json_stats.items():
        if isinstance(value, dict) and 'mean' in value:
            result[key] = {
                'mean': np.array(value['mean'], dtype=np.float64),
                'std': np.array(value['std'], dtype=np.float64)
            }
        else:
            result[key] = value
    return result


def save_norm_stats(
    stats: Dict,
    dataset_version: str,
    system: str = 'cartpole'
) -> Path:
    """
    Save normalization statistics to norm_stats.json.
    
    Args:
        stats: Normalization statistics dict
        dataset_version: e.g., 'cartpole_ood_v1'
        system: System name
    
    Returns:
        Path to saved file
    """
    save_path = paths.get_norm_stats_path(dataset_version, system)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    output = {
        'created_at': datetime.now().isoformat(),
        'computed_from': 'train',
        **stats_to_json_serializable(stats)
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Saved: {save_path}")
    return save_path


def load_norm_stats(
    dataset_version: str,
    system: str = 'cartpole'
) -> Dict:
    """
    Load normalization statistics from norm_stats.json.
    
    Args:
        dataset_version: e.g., 'cartpole_ood_v1'
        system: System name
    
    Returns:
        Dict with normalization statistics (numpy arrays)
    """
    load_path = paths.get_norm_stats_path(dataset_version, system)
    
    if not load_path.exists():
        raise FileNotFoundError(f"norm_stats.json not found: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        json_stats = json.load(f)
    
    return stats_from_json(json_stats)


# =============================================================================
# Convenience Function: Generate from Dataset
# =============================================================================

def generate_norm_stats_from_dataset(
    dataset_version: str,
    system: str = 'cartpole',
    save: bool = True
) -> Dict:
    """
    Load dataset.npz, compute norm_stats from train split, optionally save.
    
    Args:
        dataset_version: e.g., 'cartpole_ood_v1'
        system: System name
        save: Whether to save norm_stats.json
    
    Returns:
        Normalization statistics dict
    """
    dataset_path = paths.get_dataset_path(dataset_version, system)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"  Loading dataset: {dataset_path}")
    data = np.load(dataset_path)
    
    # Extract train split
    train_x = data['train_x']
    train_u = data['train_u']
    train_dx = data['train_dx']
    
    # Check for dx_savgol
    train_dx_savgol = data.get('train_dx_savgol', None)
    if train_dx_savgol is None:
        print("  ⚠️ train_dx_savgol not found in dataset")
    
    # Compute stats
    print("  Computing normalization statistics from train split...")
    stats = compute_norm_stats(train_x, train_u, train_dx, train_dx_savgol)
    
    # Print summary
    print("\n  Normalization Statistics Summary:")
    print("  " + "-" * 50)
    
    state_names = ['x', 'x_dot', 'theta', 'theta_dot']
    print("  State:")
    for i, name in enumerate(state_names):
        m, s = stats['state']['mean'][i], stats['state']['std'][i]
        print(f"    {name:12s}: mean={m:+.4f}, std={s:.4f}")
    
    print("  Input:")
    print(f"    {'u':12s}: mean={stats['input']['mean'][0]:+.4f}, std={stats['input']['std'][0]:.4f}")
    
    if 'derivative_dx' in stats:
        print("  Derivative (dx - analytic):")
        deriv_names = ['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']
        for i, name in enumerate(deriv_names):
            m, s = stats['derivative_dx']['mean'][i], stats['derivative_dx']['std'][i]
            print(f"    {name:12s}: mean={m:+.4f}, std={s:.4f}")
    
    if 'derivative_dx_savgol' in stats:
        print("  Derivative (dx_savgol - numeric):")
        for i, name in enumerate(deriv_names):
            m, s = stats['derivative_dx_savgol']['mean'][i], stats['derivative_dx_savgol']['std'][i]
            print(f"    {name:12s}: mean={m:+.4f}, std={s:.4f}")
    
    print("  " + "-" * 50)
    
    if save:
        save_norm_stats(stats, dataset_version, system)
    
    return stats


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_normalization(
    data: np.ndarray,
    stats: Dict[str, np.ndarray],
    rtol: float = 0.01,
    atol: float = 0.01
) -> Dict[str, bool]:
    """
    Validate that normalized data has mean≈0, std≈1.
    
    Args:
        data: Original data (N, T, D)
        stats: Normalization statistics
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Dict with validation results
    """
    normalized = normalize(data, stats)
    
    if normalized.ndim == 3:
        flat = normalized.reshape(-1, normalized.shape[-1])
    else:
        flat = normalized
    
    actual_mean = np.mean(flat, axis=0)
    actual_std = np.std(flat, axis=0)
    
    mean_ok = np.allclose(actual_mean, 0, rtol=rtol, atol=atol)
    std_ok = np.allclose(actual_std, 1, rtol=rtol, atol=atol)
    
    return {
        'mean_ok': mean_ok,
        'std_ok': std_ok,
        'actual_mean': actual_mean,
        'actual_std': actual_std
    }


def validate_inverse(
    data: np.ndarray,
    stats: Dict[str, np.ndarray],
    rtol: float = 1e-10,
    atol: float = 1e-10
) -> bool:
    """
    Validate that denormalize(normalize(data)) ≈ data.
    
    Args:
        data: Original data
        stats: Normalization statistics
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        True if inverse is accurate
    """
    normalized = normalize(data, stats)
    recovered = denormalize(normalized, stats)
    return np.allclose(data, recovered, rtol=rtol, atol=atol)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  normalization.py 검증")
    print("=" * 60)
    
    # Test with synthetic data
    print("\n[합성 데이터 테스트]")
    np.random.seed(42)
    
    # Create data with known statistics
    N, T, D = 50, 101, 4
    data = np.random.randn(N, T, D) * 2 + 3  # mean≈3, std≈2
    
    stats = compute_stats(data)
    print(f"  Original: mean≈{data.mean():.2f}, std≈{data.std():.2f}")
    print(f"  Computed stats: mean={stats['mean']}, std={stats['std']}")
    
    # Normalize
    normalized = normalize(data, stats)
    print(f"  Normalized: mean≈{normalized.mean():.4f}, std≈{normalized.std():.4f}")
    
    # Validate
    val_result = validate_normalization(data, stats)
    print(f"  mean≈0: {val_result['mean_ok']}, std≈1: {val_result['std_ok']}")
    
    # Inverse
    recovered = denormalize(normalized, stats)
    inverse_ok = validate_inverse(data, stats)
    print(f"  Inverse accurate: {inverse_ok}")
    
    # Test with real dataset if exists
    print("\n[실제 데이터셋 테스트]")
    try:
        stats = generate_norm_stats_from_dataset('cartpole_ood_v1', save=False)
        print("  ✅ 실제 데이터셋 통계 계산 성공")
    except FileNotFoundError as e:
        print(f"  ⚠️ 데이터셋 없음: {e}")
    
    print("\n" + "=" * 60)
    print("  ✅ normalization.py 검증 완료")
    print("=" * 60)