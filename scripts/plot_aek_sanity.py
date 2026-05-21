"""
AEK Dataset Sanity Check Plots

Generates 3 sanity plots from the AEK dataset:
1. phi(t)      - Lean angle trajectories (train/test overlay)
2. theta_w(t)  - Wheel angle trajectories
3. tau(t)      - Motor torque input

Uses plot_style SSOT (save_figure only, no seaborn).

Usage:
    python scripts/plot_aek_sanity.py
    python scripts/plot_aek_sanity.py --version aek_ood_v1
"""
import argparse
import numpy as np
from pathlib import Path

from src.contracts import paths
from src.contracts.plot_style import (
    create_figure, save_figure, setup_style, get_color, COLORS
)


def plot_aek_sanity(version: str = 'aek_ood_v1'):
    """Generate sanity check plots for AEK dataset."""

    # Load dataset
    npz_path = paths.get_dataset_path(version, system='aek')
    print(f"  Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    t = data['t']
    train_x = data['train_x']   # (N, T, 4)
    train_u = data['train_u']   # (N, T, 1)
    test_x = data['test_x']
    test_u = data['test_u']

    # Output directory
    fig_dir = npz_path.parent / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Train: {train_x.shape}, Test: {test_x.shape}")
    print(f"  T: {len(t)} steps, dt={t[1]-t[0]:.4f}s, duration={t[-1]:.2f}s")

    setup_style()

    # Colors
    c_train = get_color('train')
    c_test = get_color('test')

    # =========================================================================
    # Plot 1: phi(t) - Lean angle
    # =========================================================================
    fig, ax = create_figure('wide')

    for i in range(train_x.shape[0]):
        ax.plot(t, np.rad2deg(train_x[i, :, 0]),
                color=c_train, alpha=0.4, linewidth=0.8,
                label='Train (ID)' if i == 0 else None)

    for i in range(test_x.shape[0]):
        ax.plot(t, np.rad2deg(test_x[i, :, 0]),
                color=c_test, alpha=0.4, linewidth=0.8,
                label='Test (OOD)' if i == 0 else None)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\phi$ (deg)')
    ax.set_title(f'AEK Lean Angle — {version}')
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='-')

    save_figure(fig, fig_dir, 'sanity_phi')

    # =========================================================================
    # Plot 2: theta_w(t) - Wheel angle (relative)
    # =========================================================================
    fig, ax = create_figure('wide')

    for i in range(train_x.shape[0]):
        ax.plot(t, train_x[i, :, 2],
                color=c_train, alpha=0.4, linewidth=0.8,
                label='Train (ID)' if i == 0 else None)

    for i in range(test_x.shape[0]):
        ax.plot(t, test_x[i, :, 2],
                color=c_test, alpha=0.4, linewidth=0.8,
                label='Test (OOD)' if i == 0 else None)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\theta_w$ (rad)')
    ax.set_title(f'AEK Wheel Angle (relative) — {version}')
    ax.legend(loc='upper right')

    save_figure(fig, fig_dir, 'sanity_theta_w')

    # =========================================================================
    # Plot 3: tau(t) - Motor torque
    # =========================================================================
    fig, ax = create_figure('wide')

    for i in range(train_u.shape[0]):
        ax.plot(t, train_u[i, :, 0] * 1000,  # Convert to mN*m for readability
                color=c_train, alpha=0.4, linewidth=0.8,
                label='Train (ID)' if i == 0 else None)

    for i in range(test_u.shape[0]):
        ax.plot(t, test_u[i, :, 0] * 1000,
                color=c_test, alpha=0.4, linewidth=0.8,
                label='Test (OOD)' if i == 0 else None)

    # tau_max lines
    tau_max_mNm = 20.0  # 0.02 N*m = 20 mN*m
    ax.axhline(y=tau_max_mNm, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=-tau_max_mNm, color='k', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\tau$ (mN$\cdot$m)')
    ax.set_title(f'AEK Motor Torque — {version}')
    ax.legend(loc='upper right')

    save_figure(fig, fig_dir, 'sanity_tau')

    # Summary statistics
    print(f"\n  === Dataset Summary ===")
    print(f"  phi  range: [{np.rad2deg(train_x[:,:,0].min()):.2f}, "
          f"{np.rad2deg(train_x[:,:,0].max()):.2f}] deg (train)")
    print(f"  phi  range: [{np.rad2deg(test_x[:,:,0].min()):.2f}, "
          f"{np.rad2deg(test_x[:,:,0].max()):.2f}] deg (test)")
    print(f"  tau  range: [{train_u[:,:,0].min()*1000:.2f}, "
          f"{train_u[:,:,0].max()*1000:.2f}] mN*m (train)")
    print(f"  |tau|<=tau_max: {(np.abs(train_u).max() <= 0.02 + 1e-10) and (np.abs(test_u).max() <= 0.02 + 1e-10)}")

    print(f"\n  ✅ Sanity plots saved to: {fig_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AEK dataset sanity plots')
    parser.add_argument('--version', '-v', type=str, default='aek_ood_v1')
    args = parser.parse_args()
    plot_aek_sanity(args.version)