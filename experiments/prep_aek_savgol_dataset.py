"""
AEK Dataset Preprocessing: Add Measurement Noise + SavGol Derivatives

Purpose:
    AEK dataset currently has analytic (clean) dx only.
    Paper §3.3/§4.1 claims: "5% Gaussian measurement noise + SavGol window=7, polyorder=3"
    This script adds noise + SavGol dx keys to the existing dataset.npz,
    matching the protocol used by CP, Lorenz, Silverbox, and Lynx-Hare.

What it does:
    1. Loads existing AEK dataset.npz
    2. Adds 5% Gaussian noise to state measurements (deterministic seed=2026)
    3. Computes SavGol derivatives from noisy states
    4. Saves NEW keys alongside originals (non-destructive):
       - train_x_noisy, val_x_noisy, test_x_noisy
       - train_dx_savgol, val_dx_savgol, test_dx_savgol
    5. Creates backup of original dataset

Noise protocol (matches Lorenz/CP):
    noise_std = noise_fraction * std(x_clean, axis=(0,1))
    x_noisy = x_clean + N(0, noise_std) per state variable

SavGol parameters (from paper §4.1):
    window_length = 7
    polyorder = 3
    AEK dt = 0.01 (201 steps, 2.0s) — window=7 is appropriate

Usage (PowerShell, copy-paste ready):
    python experiments/prep_aek_savgol_dataset.py

Author: Claude (SavGol consistency patch)
Date: 2026-03-11
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import shutil
from datetime import datetime
from scipy.signal import savgol_filter

from src.contracts import paths


# =============================================================================
# Configuration
# =============================================================================

SYSTEM = 'aek'
DATASET_VERSION = 'aek_ood_v1'
NOISE_FRACTION = 0.05       # 5% Gaussian noise (same as Lorenz/CP)
NOISE_SEED = 2026           # Deterministic, reproducible
SAVGOL_WINDOW = 7           # Paper §4.1
SAVGOL_POLYORDER = 3        # Paper §4.1
STATE_DIM = 4               # [phi, phi_dot, theta_w, theta_w_dot]


# =============================================================================
# Functions
# =============================================================================

def add_noise(x_clean: np.ndarray, noise_fraction: float,
              rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian measurement noise to state trajectories.

    Protocol (matches Lorenz runner v1.1):
        noise_std_per_state = noise_fraction * std(x_clean) across all (traj, time)
        x_noisy = x_clean + N(0, noise_std_per_state)

    Args:
        x_clean: (N, T, D) clean state trajectories
        noise_fraction: fraction of signal std (e.g. 0.05 for 5%)
        rng: numpy random Generator for reproducibility

    Returns:
        x_noisy: (N, T, D) noisy state trajectories
    """
    N, T, D = x_clean.shape

    # Compute std per state across all trajectories and timesteps
    x_flat = x_clean.reshape(-1, D)
    signal_std = np.std(x_flat, axis=0)  # (D,)

    noise_std = noise_fraction * signal_std
    print(f"  Signal std per state: {signal_std}")
    print(f"  Noise std per state:  {noise_std}")

    # Add noise
    noise = rng.normal(0, 1, size=x_clean.shape) * noise_std[np.newaxis, np.newaxis, :]
    x_noisy = x_clean + noise

    return x_noisy


def compute_savgol_dx(x: np.ndarray, dt: float,
                      window: int, polyorder: int) -> np.ndarray:
    """
    Compute SavGol derivatives from (possibly noisy) state trajectories.

    Args:
        x: (N, T, D) state trajectories
        dt: time step
        window: SavGol window length
        polyorder: SavGol polynomial order

    Returns:
        dx: (N, T, D) SavGol-estimated derivatives
    """
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


def validate_savgol_vs_analytic(dx_savgol: np.ndarray,
                                dx_analytic: np.ndarray,
                                label: str) -> None:
    """Print diagnostic comparison between SavGol and analytic dx."""
    diff = dx_savgol - dx_analytic
    max_abs = np.abs(diff).max()
    mean_abs = np.abs(diff).mean()
    rel_err = np.abs(diff) / (np.abs(dx_analytic) + 1e-10)
    median_rel = np.median(rel_err)

    print(f"  {label}: max_abs_diff={max_abs:.4f}, "
          f"mean_abs_diff={mean_abs:.4f}, median_rel_err={median_rel:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  AEK Dataset Preprocessing: Add Noise + SavGol Derivatives")
    print("=" * 70)

    # 1. Load dataset
    dataset_path = paths.get_dataset_path(DATASET_VERSION, system=SYSTEM)
    print(f"\n[1] Loading dataset: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = dict(np.load(dataset_path, allow_pickle=True))
    print(f"  Existing keys: {list(data.keys())}")
    print(f"  train_x: {data['train_x'].shape}")
    print(f"  dt: {float(data['dt'])}")

    dt = float(data['dt'])
    assert abs(dt - 0.01) < 1e-6, f"Expected dt=0.01, got {dt}"

    # Check if already preprocessed
    if 'train_dx_savgol' in data:
        print("\n  ⚠️  Dataset already has 'train_dx_savgol' key!")
        print("  Skipping preprocessing (delete key to re-run)")
        return

    # 2. Backup original
    backup_path = dataset_path.with_suffix('.npz.bak_analytic')
    if not backup_path.exists():
        print(f"\n[2] Creating backup: {backup_path}")
        shutil.copy2(dataset_path, backup_path)
        print(f"  ✅ Backup saved")
    else:
        print(f"\n[2] Backup already exists: {backup_path}")

    # 3. Add measurement noise
    print(f"\n[3] Adding {NOISE_FRACTION*100:.0f}% Gaussian noise (seed={NOISE_SEED})")
    rng = np.random.default_rng(NOISE_SEED)

    splits = ['train', 'val', 'test']
    noisy_data = {}

    for split in splits:
        x_clean = data[f'{split}_x']
        print(f"\n  --- {split} (shape={x_clean.shape}) ---")

        x_noisy = add_noise(x_clean, NOISE_FRACTION, rng)
        noisy_data[f'{split}_x_noisy'] = x_noisy

        # Sanity: noise magnitude
        noise_mag = np.abs(x_noisy - x_clean)
        print(f"  Noise magnitude: mean={noise_mag.mean():.6f}, "
              f"max={noise_mag.max():.6f}")

    # 4. Compute SavGol derivatives from noisy states
    print(f"\n[4] Computing SavGol derivatives "
          f"(window={SAVGOL_WINDOW}, polyorder={SAVGOL_POLYORDER})")

    savgol_data = {}

    for split in splits:
        x_noisy = noisy_data[f'{split}_x_noisy']
        dx_analytic = data[f'{split}_dx']

        dx_savgol = compute_savgol_dx(x_noisy, dt, SAVGOL_WINDOW, SAVGOL_POLYORDER)
        savgol_data[f'{split}_dx_savgol'] = dx_savgol

        validate_savgol_vs_analytic(dx_savgol, dx_analytic, split)

    # 5. Save augmented dataset
    print(f"\n[5] Saving augmented dataset")

    # Merge new keys into existing data
    for key, val in noisy_data.items():
        data[key] = val
    for key, val in savgol_data.items():
        data[key] = val

    # Add metadata
    data['savgol_meta'] = np.array({
        'noise_fraction': NOISE_FRACTION,
        'noise_seed': NOISE_SEED,
        'savgol_window': SAVGOL_WINDOW,
        'savgol_polyorder': SAVGOL_POLYORDER,
        'preprocessed_at': datetime.now().isoformat(),
        'script': 'prep_aek_savgol_dataset.py',
    })

    np.savez(dataset_path, **data)
    print(f"  ✅ Saved: {dataset_path}")
    print(f"  New keys: {[k for k in data.keys() if 'noisy' in k or 'savgol' in k]}")

    # 6. Verification
    print(f"\n[6] Verification")
    reload = dict(np.load(dataset_path, allow_pickle=True))
    assert 'train_dx_savgol' in reload, "FAIL: train_dx_savgol not found"
    assert 'train_x_noisy' in reload, "FAIL: train_x_noisy not found"
    assert reload['train_dx_savgol'].shape == reload['train_dx'].shape, \
        "FAIL: shape mismatch"
    print(f"  ✅ All verification checks passed")

    print(f"\n{'=' * 70}")
    print(f"  ✅ AEK dataset preprocessing complete")
    print(f"  Next: re-run run_aek3_baseline.py with --use_savgol flag")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
