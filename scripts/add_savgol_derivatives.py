"""
S03: Add Savgol Derivatives to Existing Dataset

Adds dx_savgol fields to existing dataset.npz without regenerating trajectories.

Usage:
    python scripts/add_savgol_derivatives.py --version cartpole_ood_v1
"""
import argparse
import json
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.utils.derivatives import (
    compute_derivatives_savgol,
    validate_derivatives,
    check_wrap_boundary_issues,
    SAVGOL_CONFIG
)


def add_savgol_derivatives(version: str, system: str = 'cartpole') -> Path:
    """
    Add Savgol derivatives to existing dataset.
    
    Args:
        version: Dataset version (e.g., 'cartpole_ood_v1')
        system: System name
    
    Returns:
        Path to updated dataset
    """
    # Load existing dataset
    npz_path = paths.get_dataset_path(version, system=system)
    meta_path = paths.get_meta_path(version, system=system)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    
    print("=" * 60)
    print(f"  S03: Adding Savgol Derivatives")
    print("=" * 60)
    print(f"\n  Dataset: {npz_path}")
    print(f"  Savgol config: window={SAVGOL_CONFIG['window']}, polyorder={SAVGOL_CONFIG['polyorder']}")
    
    # Load data
    data = dict(np.load(npz_path))
    dt = float(data['dt'])
    
    print(f"\n  Processing splits...")
    
    comparison_results = {}
    wrap_check_results = {}
    
    for split in ['train', 'val', 'test']:
        x = data[f'{split}_x']
        dx_analytic = data[f'{split}_dx']
        
        print(f"\n    {split}: {x.shape[0]} trajectories, T={x.shape[1]}")
        
        # Compute Savgol derivatives
        dx_savgol = compute_derivatives_savgol(x, dt, theta_idx=2)
        
        # Store in data dict
        data[f'{split}_dx_savgol'] = dx_savgol
        
        # Validation: compare with analytic
        metrics = validate_derivatives(
            dx_savgol, dx_analytic,
            state_names=['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']
        )
        comparison_results[split] = metrics
        
        # Check wrap boundary
        wrap_check = check_wrap_boundary_issues(x, dx_savgol, theta_idx=2)
        wrap_check_results[split] = wrap_check
        
        # Print summary
        print(f"      Savgol vs Analytic:")
        for name, m in metrics.items():
            print(f"        {name}: RMSE={m['rmse']:.6f}, R²={m['r2']:.4f}")
        
        if wrap_check['has_issues']:
            print(f"      ⚠️  Wrap boundary issues detected: {wrap_check['n_boundary_spikes']}")
        else:
            print(f"      ✅ No wrap boundary issues")
    
    # Save updated dataset
    print(f"\n  Saving updated dataset...")
    np.savez_compressed(npz_path, **data)
    print(f"  ✅ Saved: {npz_path}")
    
    # Update metadata
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    else:
        meta = {}
    
    meta['s03_savgol'] = {
        'added_at': datetime.now().isoformat(),
        'savgol_config': SAVGOL_CONFIG,
        'comparison_results': {
            split: {k: v for k, v in metrics.items()}
            for split, metrics in comparison_results.items()
        },
        'wrap_check': wrap_check_results
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Updated: {meta_path}")
    
    # Final validation
    print(f"\n  Running Preflight validation...")
    validate_dataset_lite(npz_path)
    print(f"  ✅ Preflight passed")
    
    return npz_path


def verify_savgol_added(version: str, system: str = 'cartpole') -> bool:
    """
    Verify that Savgol derivatives were added correctly.
    
    Args:
        version: Dataset version
        system: System name
    
    Returns:
        True if verification passed
    """
    npz_path = paths.get_dataset_path(version, system=system)
    data = np.load(npz_path)
    
    print("\n" + "=" * 60)
    print("  S03 Verification")
    print("=" * 60)
    
    all_passed = True
    
    for split in ['train', 'val', 'test']:
        key = f'{split}_dx_savgol'
        x_key = f'{split}_x'
        
        if key not in data:
            print(f"  ❌ Missing: {key}")
            all_passed = False
            continue
        
        dx_savgol = data[key]
        x = data[x_key]
        
        # Check shape
        if dx_savgol.shape != x.shape:
            print(f"  ❌ {key}: shape mismatch {dx_savgol.shape} vs {x.shape}")
            all_passed = False
            continue
        
        # Check no NaN
        if np.isnan(dx_savgol).any():
            print(f"  ❌ {key}: contains NaN")
            all_passed = False
            continue
        
        # Check non-constant
        if dx_savgol.std() == 0:
            print(f"  ❌ {key}: constant values")
            all_passed = False
            continue
        
        print(f"  ✅ {key}: shape={dx_savgol.shape}, std={dx_savgol.std():.4f}")
    
    if all_passed:
        print("\n  ✅ S03 Verification PASSED")
    else:
        print("\n  ❌ S03 Verification FAILED")
    
    return all_passed


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Add Savgol derivatives to existing dataset'
    )
    parser.add_argument(
        '--version', '-v',
        type=str,
        default='cartpole_ood_v1',
        help='Dataset version (default: cartpole_ood_v1)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify, do not add derivatives'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_savgol_added(args.version)
    else:
        add_savgol_derivatives(args.version)
        verify_savgol_added(args.version)


if __name__ == '__main__':
    main()