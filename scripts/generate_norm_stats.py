#!/usr/bin/env python
"""
S04-A: Generate Normalization Statistics

Computes norm_stats.json from train split of dataset.npz.
Also updates meta.json with norm_stats reference.

Usage:
    python scripts/generate_norm_stats.py --version cartpole_ood_v1
    python scripts/generate_norm_stats.py --version cartpole_ood_v1 --validate
"""
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import paths
from src.data.normalization import (
    generate_norm_stats_from_dataset,
    load_norm_stats,
    normalize,
    validate_normalization,
    validate_inverse
)


def update_meta_json(
    dataset_version: str,
    system: str = 'cartpole'
) -> None:
    """
    Add norm_stats section to meta.json.
    
    Records which keys were actually used for computing statistics.
    """
    meta_path = paths.get_meta_path(dataset_version, system)
    dataset_path = paths.get_dataset_path(dataset_version, system)
    
    if not meta_path.exists():
        print(f"  ⚠️ meta.json not found: {meta_path}")
        return
    
    # Check which keys actually exist in dataset
    data = np.load(dataset_path)
    actual_keys = []
    for key in ['train_x', 'train_u', 'train_dx', 'train_dx_savgol']:
        if key in data:
            actual_keys.append(key)
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # Add norm_stats section with actual keys
    meta['s04_norm_stats'] = {
        'added_at': datetime.now().isoformat(),
        'computed_from': 'train',
        'keys_used': actual_keys,
        'norm_stats_path': str(paths.get_norm_stats_path(dataset_version, system))
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Updated: {meta_path}")


def validate_norm_stats(
    dataset_version: str,
    system: str = 'cartpole'
) -> bool:
    """
    Validate normalization statistics.
    
    Checks:
    1. norm_stats.json exists and loads correctly
    2. train_x normalized has mean≈0, std≈1
    3. Inverse transformation is accurate
    
    Returns:
        True if all checks pass
    """
    print("\n" + "=" * 60)
    print("  Normalization Validation")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Load stats
    print("\n[1] Loading norm_stats.json...")
    try:
        stats = load_norm_stats(dataset_version, system)
        print("  ✅ norm_stats.json loaded successfully")
    except FileNotFoundError as e:
        print(f"  ❌ Failed to load: {e}")
        return False
    
    # 2. Load dataset
    print("\n[2] Loading dataset...")
    dataset_path = paths.get_dataset_path(dataset_version, system)
    data = np.load(dataset_path)
    
    # 3. Validate train_x normalization
    print("\n[3] Validating train_x normalization (should have mean≈0, std≈1)...")
    train_x = data['train_x']
    result = validate_normalization(train_x, stats['state'])
    
    print(f"  Actual mean: {result['actual_mean']}")
    print(f"  Actual std:  {result['actual_std']}")
    
    if result['mean_ok'] and result['std_ok']:
        print("  ✅ train_x normalization: PASS")
    else:
        print("  ❌ train_x normalization: FAIL")
        all_passed = False
    
    # 4. Validate inverse
    print("\n[4] Validating inverse transformation...")
    if validate_inverse(train_x, stats['state']):
        print("  ✅ Inverse (denormalize ∘ normalize = identity): PASS")
    else:
        print("  ❌ Inverse transformation: FAIL")
        all_passed = False
    
    # 5. Validate train_u
    print("\n[5] Validating train_u normalization...")
    train_u = data['train_u']
    result_u = validate_normalization(train_u, stats['input'])
    
    if result_u['mean_ok'] and result_u['std_ok']:
        print("  ✅ train_u normalization: PASS")
    else:
        print("  ❌ train_u normalization: FAIL")
        all_passed = False
    
    # 6. Validate derivatives
    print("\n[6] Validating derivative normalization...")
    
    if 'derivative_dx' in stats:
        train_dx = data['train_dx']
        result_dx = validate_normalization(train_dx, stats['derivative_dx'])
        if result_dx['mean_ok'] and result_dx['std_ok']:
            print("  ✅ train_dx (analytic) normalization: PASS")
        else:
            print("  ❌ train_dx (analytic) normalization: FAIL")
            all_passed = False
    
    if 'derivative_dx_savgol' in stats:
        if 'train_dx_savgol' in data:
            train_dx_savgol = data['train_dx_savgol']
            result_dxs = validate_normalization(train_dx_savgol, stats['derivative_dx_savgol'])
            if result_dxs['mean_ok'] and result_dxs['std_ok']:
                print("  ✅ train_dx_savgol (numeric) normalization: PASS")
            else:
                print("  ❌ train_dx_savgol (numeric) normalization: FAIL")
                all_passed = False
        else:
            print("  ⚠️ train_dx_savgol not in dataset, skipping")
    
    # 7. Check val/test are NOT exactly 0,1 (no leakage)
    print("\n[7] Checking for data leakage (val/test should NOT have exact mean=0, std=1)...")
    
    val_x = data['val_x']
    val_norm = normalize(val_x, stats['state'])
    val_mean = val_norm.reshape(-1, 4).mean(axis=0)
    val_std = val_norm.reshape(-1, 4).std(axis=0)
    
    print(f"  val_x normalized mean: {val_mean}")
    print(f"  val_x normalized std:  {val_std}")
    
    # val/test should have some deviation from 0/1 (otherwise might be leakage)
    if not np.allclose(val_mean, 0, atol=0.001) or not np.allclose(val_std, 1, atol=0.001):
        print("  ✅ No data leakage detected (val has natural deviation)")
    else:
        print("  ⚠️ Warning: val_x seems too close to mean=0, std=1 - check for leakage")
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("  ❌ SOME VALIDATION CHECKS FAILED")
    print("=" * 60)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Generate normalization statistics from dataset'
    )
    parser.add_argument(
        '--version', '-v',
        type=str,
        default='cartpole_ood_v1',
        help='Dataset version (default: cartpole_ood_v1)'
    )
    parser.add_argument(
        '--system', '-s',
        type=str,
        default='cartpole',
        help='System name (default: cartpole)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after generation'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation (skip generation)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"  S04-A: Normalization Statistics Generation")
    print(f"  Dataset: {args.version}")
    print("=" * 60)
    
    if not args.validate_only:
        # Generate norm_stats.json
        print("\n[Step 1] Generating norm_stats.json...")
        stats = generate_norm_stats_from_dataset(
            args.version, 
            args.system, 
            save=True
        )
        
        # Update meta.json
        print("\n[Step 2] Updating meta.json...")
        update_meta_json(args.version, args.system)
    
    # Validate if requested
    if args.validate or args.validate_only:
        success = validate_norm_stats(args.version, args.system)
        if not success:
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  ✅ S04-A Complete!")
    print("=" * 60)
    
    # Show file locations
    print(f"\n  Files:")
    print(f"    {paths.get_norm_stats_path(args.version, args.system)}")
    print(f"    {paths.get_meta_path(args.version, args.system)}")


if __name__ == '__main__':
    main()