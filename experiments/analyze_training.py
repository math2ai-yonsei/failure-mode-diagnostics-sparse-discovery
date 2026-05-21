"""
Analyze training data: torque profiles and phi trajectories.
Goal: understand what controller keeps phi small, and what approach
we should use for pool generation.
"""
import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

data_path = _PROJECT_ROOT / "data" / "aek" / "aek_ood_v1" / "dataset.npz"
ds = np.load(data_path, allow_pickle=True)

train_x = ds['train_x']     # (N, T, 4)
train_u = ds['train_u']     # (N, T, 1)
train_params = ds['train_params']

N, T, _ = train_x.shape
dt = float(ds['dt'])

print("=" * 60)
print("Training Data Analysis")
print("=" * 60)

# ── Phi trajectories ──
print(f"\n── Phi trajectories (N={N}, T={T}) ──")
for i in range(N):
    phi = train_x[i, :, 0]
    phi_dot = train_x[i, :, 1]
    tw_dot = train_x[i, :, 3]
    u = train_u[i, :, 0]
    print(f"  traj[{i}]: phi=[{phi.min():.4f}, {phi.max():.4f}], "
          f"|phi|_max={np.abs(phi).max():.4f}, "
          f"|phi_dot|_max={np.abs(phi_dot).max():.2f}, "
          f"|tw_dot|_max={np.abs(tw_dot).max():.1f}, "
          f"|u|_max={np.abs(u).max():.6f}")

# ── Torque statistics ──
print(f"\n── Torque profiles ──")
all_u = train_u.reshape(-1)
print(f"  Overall: min={all_u.min():.6f}, max={all_u.max():.6f}, "
      f"mean={all_u.mean():.6f}, std={all_u.std():.6f}")

# ── Check if torques look like PD control ──
print(f"\n── PD control check ──")
print("  If tau ≈ -Kp*phi - Kd*phi_dot, we can estimate Kp, Kd")
for i in range(min(3, N)):
    phi = train_x[i, :, 0]
    phi_dot = train_x[i, :, 1]
    tau = train_u[i, :, 0]
    # Least squares: tau = a*phi + b*phi_dot
    A = np.column_stack([phi, phi_dot])
    result = np.linalg.lstsq(A, tau, rcond=None)
    coeffs = result[0]
    residual = tau - A @ coeffs
    r2 = 1 - np.var(residual) / np.var(tau) if np.var(tau) > 0 else 0
    print(f"  traj[{i}]: Kp≈{-coeffs[0]:.4f}, Kd≈{-coeffs[1]:.4f}, R²={r2:.4f}")

# ── Training phi max across all time ──
print(f"\n── Training phi extremes ──")
all_phi = train_x[:, :, 0]
print(f"  Global phi max: {all_phi.max():.6f}")
print(f"  Global phi min: {all_phi.min():.6f}")
print(f"  Global |phi| max: {np.abs(all_phi).max():.6f}")

# ── Check val/test data ──
for split in ['val', 'test']:
    x = ds.get(f'{split}_x')
    if x is not None:
        phi_s = x[:, :, 0]
        print(f"  {split}: |phi|_max={np.abs(phi_s).max():.6f}, "
              f"N={x.shape[0]}, T={x.shape[1]}")