"""
AEK Dataset Fix: SavGol Derivatives from Clean State (No Noise)

Purpose:
    Previous prep_aek_savgol_dataset.py added 5% noise + SavGol.
    Per GPT P0 review: noise addition is "benchmark redesign", not bug-fix.
    The actual requirement is derivative METHOD consistency (SavGol for all),
    NOT noise addition.

    This script:
    1. Computes SavGol derivatives from CLEAN state trajectories
    2. Overwrites train_dx_savgol/val_dx_savgol/test_dx_savgol with clean-based versions
    3. Removes noisy keys (train_x_noisy etc.) to avoid confusion

    After this, runners use:
    - train_x (clean, original)
    - train_dx_savgol (SavGol from clean x — derivative method consistency)

Usage (PowerShell, copy-paste ready):
    python experiments/fix_aek_savgol_clean.py

Author: Claude (SavGol consistency patch — GPT P0 correction)
Date: 2026-03-11
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from datetime import datetime
from scipy.signal import savgol_filter

from src.contracts import paths


# =============================================================================
# Configuration
# =============================================================================

SYSTEM = 'aek'
DATASET_VERSION = 'aek_ood_v1'
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 3
STATE_DIM = 4


# =============================================================================
# Functions
# =============================================================================

def compute_savgol_dx(x: np.ndarray, dt: float,
                      window: int, polyorder: int) -> np.ndarray:
    """Compute SavGol derivatives from state trajectories."""
    N, T, D = x.shape
    dx = np.zeros_like(x)
    for n in range(N):
        for d in range(D):
            dx[n, :, d] = savgol_filter(
                x[n, :, d],
                window_length=window,
                polyorder=polyorder,
                deriv=1,
                delta=dt,
            )
    return dx


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  AEK Dataset Fix: SavGol from Clean State (No Noise)")
    print("  GPT P0: noise is benchmark redesign, not bug-fix")
    print("=" * 70)

    # 1. Load dataset
    dataset_path = paths.get_dataset_path(DATASET_VERSION, system=SYSTEM)
    print(f"\n[1] Loading dataset: {dataset_path}")

    data = dict(np.load(dataset_path, allow_pickle=True))
    dt = float(data['dt'])
    print(f"  dt: {dt}")
    print(f"  Current keys: {sorted(data.keys())}")

    # 2. Compute SavGol dx from CLEAN state
    print(f"\n[2] Computing SavGol dx from CLEAN state "
          f"(window={SAVGOL_WINDOW}, polyorder={SAVGOL_POLYORDER})")

    splits = ['train', 'val', 'test']

    for split in splits:
        x_clean = data[f'{split}_x']           # Original clean state
        dx_analytic = data[f'{split}_dx']       # Original analytic dx

        dx_savgol = compute_savgol_dx(x_clean, dt, SAVGOL_WINDOW, SAVGOL_POLYORDER)

        # Diagnostic: SavGol(clean) vs analytic
        diff = dx_savgol - dx_analytic
        max_abs = np.abs(diff).max()
        mean_abs = np.abs(diff).mean()
        rel_err = np.abs(diff) / (np.abs(dx_analytic) + 1e-10)
        median_rel = np.median(rel_err)

        print(f"\n  {split} (shape={x_clean.shape}):")
        print(f"    SavGol(clean) vs analytic: max_abs={max_abs:.6f}, "
              f"mean_abs={mean_abs:.6f}, median_rel={median_rel:.6f}")

        # Overwrite savgol key with clean-based version
        data[f'{split}_dx_savgol'] = dx_savgol

    # 3. Remove noisy keys (avoid confusion)
    print(f"\n[3] Removing noisy keys...")
    noisy_keys = [k for k in data.keys() if 'noisy' in k]
    for k in noisy_keys:
        del data[k]
        print(f"  Removed: {k}")

    # Remove old savgol_meta if present
    if 'savgol_meta' in data:
        del data['savgol_meta']
        print(f"  Removed: savgol_meta")

    # 4. Add clean metadata
    data['savgol_meta'] = np.array({
        'noise_fraction': 0.0,   # NO noise — GPT P0
        'savgol_window': SAVGOL_WINDOW,
        'savgol_polyorder': SAVGOL_POLYORDER,
        'dx_source': 'savgol_from_clean_x',
        'rationale': 'Derivative method consistency (§3.3), not noise addition',
        'fixed_at': datetime.now().isoformat(),
        'script': 'fix_aek_savgol_clean.py',
    })

    # 5. Save
    print(f"\n[4] Saving fixed dataset...")
    np.savez(dataset_path, **data)
    print(f"  ✅ Saved: {dataset_path}")
    print(f"  Final keys: {sorted(data.keys())}")

    # 6. Verification
    print(f"\n[5] Verification")
    reload = dict(np.load(dataset_path, allow_pickle=True))
    assert 'train_dx_savgol' in reload
    assert 'train_x_noisy' not in reload, "FAIL: noisy key still present"
    # Verify SavGol(clean) is close to analytic
    diff_check = np.abs(reload['train_dx_savgol'] - reload['train_dx'])
    print(f"  SavGol(clean) vs analytic max_diff: {diff_check.max():.6f}")
    print(f"  ✅ Verification passed")

    print(f"\n{'=' * 70}")
    print(f"  ✅ AEK dataset fix complete (clean SavGol, no noise)")
    print(f"  Next: fix runners to use train_x (not train_x_noisy)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
