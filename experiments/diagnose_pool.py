"""
Diagnose why generate_pool() rejects all trajectories.
Tracks rejection reasons and prints sample ICs/simulation results.
"""
import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.mixture import GaussianMixture
from src.simulators.aek_simulator import AEKSimulator

# ── Load data ──
data_path = _PROJECT_ROOT / "data" / "aek" / "aek_ood_v1" / "dataset.npz"
ds = np.load(data_path, allow_pickle=True)

train_x = ds['train_x']     # (N, T, 4)
train_u = ds['train_u']     # (N, T, 1)
train_params = ds['train_params']  # (N, ...)

N, T_data, _ = train_x.shape
dt = float(ds['dt'])
t_total = (T_data - 1) * dt

print("=" * 60)
print("AEK Pool Generation Diagnostic")
print("=" * 60)
print(f"  train_x shape: {train_x.shape}")
print(f"  train_u shape: {train_u.shape}")
print(f"  train_params shape: {train_params.shape}")
print(f"  dt: {dt}, T_data: {T_data}, t_total: {t_total:.3f}")

# ── Training IC statistics ──
ics = train_x[:, 0, :]  # (N, 4)
print(f"\n── Training ICs (N={N}) ──")
labels = ['phi', 'phi_dot', 'theta_w', 'theta_w_dot']
for j, lbl in enumerate(labels):
    col = ics[:, j]
    print(f"  {lbl:15s}: min={col.min():.6f}, max={col.max():.6f}, "
          f"mean={col.mean():.6f}, std={col.std():.6f}")

params_flat = train_params.reshape(N, -1)
print(f"\n── Training params shape: {params_flat.shape} ──")
for j in range(params_flat.shape[1]):
    col = params_flat[:, j]
    print(f"  param[{j}]: min={col.min():.6f}, max={col.max():.6f}, "
          f"mean={col.mean():.6f}, std={col.std():.6f}")

# ── Fit GMM ──
data_5d = np.hstack([ics, params_flat[:, :1]])
print(f"\n── GMM fit data: {data_5d.shape} ──")

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(data_5d)
print(f"  GMM converged: {gmm.converged_}")

# ── Sample and compare ──
rng = np.random.default_rng(42)
samples, _ = gmm.sample(20)
sample_ics = samples[:, :4]
sample_params = np.abs(samples[:, 4:5])
sample_params = np.clip(sample_params, 1e-6, 1e-2)

print(f"\n── GMM Samples (20) ──")
for j, lbl in enumerate(labels):
    col = sample_ics[:, j]
    print(f"  {lbl:15s}: min={col.min():.6f}, max={col.max():.6f}, "
          f"mean={col.mean():.6f}")
print(f"  {'I_w_C':15s}: min={sample_params.min():.6f}, max={sample_params.max():.6f}")

# ── Try simulating 20 samples ──
print(f"\n── Simulation test (20 samples) ──")
QC_PHI = 1.5
QC_PHI_DOT = 50.0
QC_THETA_W_DOT = 500.0

counts = {
    'exception': 0,
    'len_mismatch': 0,
    'qc_phi': 0,
    'qc_phi_dot': 0,
    'qc_theta_w_dot': 0,
    'nan_inf': 0,
    'accepted': 0,
}

for i in range(20):
    ic = sample_ics[i]
    I_w_C = float(sample_params[i, 0])
    sim = AEKSimulator(params={'I_w_C': I_w_C})

    u_idx = int(rng.integers(0, N))
    u_profile = train_u[u_idx]

    def _make_ctrl(u_prof, dt_val, T_val):
        def ctrl(t, x):
            k = min(int(t / dt_val), T_val - 1)
            return float(u_prof[k, 0])
        return ctrl

    controller = _make_ctrl(u_profile, dt, T_data)

    try:
        t_arr, x_arr, u_arr = sim.simulate(
            x0=ic,
            t_span=(0.0, t_total),
            dt=dt,
            controller=controller,
            method='RK45',
        )
    except Exception as e:
        counts['exception'] += 1
        print(f"  [{i:2d}] EXCEPTION: {type(e).__name__}: {e}")
        continue

    if len(t_arr) != T_data:
        counts['len_mismatch'] += 1
        print(f"  [{i:2d}] LEN_MISMATCH: got {len(t_arr)}, expected {T_data}")
        continue

    phi_max = np.abs(x_arr[:, 0]).max()
    phi_dot_max = np.abs(x_arr[:, 1]).max()
    tw_dot_max = np.abs(x_arr[:, 3]).max()
    has_nan = not np.all(np.isfinite(x_arr))

    reason = None
    if has_nan:
        reason = 'nan_inf'
    elif phi_max > QC_PHI:
        reason = 'qc_phi'
    elif phi_dot_max > QC_PHI_DOT:
        reason = 'qc_phi_dot'
    elif tw_dot_max > QC_THETA_W_DOT:
        reason = 'qc_theta_w_dot'

    if reason:
        counts[reason] += 1
        print(f"  [{i:2d}] REJECT ({reason}): phi_max={phi_max:.3f}, "
              f"phi_dot_max={phi_dot_max:.1f}, tw_dot_max={tw_dot_max:.1f}, "
              f"nan={has_nan}")
    else:
        counts['accepted'] += 1
        print(f"  [{i:2d}] ACCEPTED: phi_max={phi_max:.3f}, "
              f"phi_dot_max={phi_dot_max:.1f}, tw_dot_max={tw_dot_max:.1f}")

print(f"\n── Summary ──")
for k, v in counts.items():
    print(f"  {k:20s}: {v}")

# ── Also check: does sim.simulate return the right format? ──
print(f"\n── Quick format check with training IC[0] ──")
ic0 = ics[0]
sim0 = AEKSimulator(params={'I_w_C': float(params_flat[0, 0])})
u0 = train_u[0]
ctrl0 = _make_ctrl(u0, dt, T_data)
try:
    t_arr, x_arr, u_arr = sim0.simulate(
        x0=ic0, t_span=(0.0, t_total), dt=dt,
        controller=ctrl0, method='RK45',
    )
    print(f"  t_arr: shape={np.array(t_arr).shape}, type={type(t_arr)}")
    print(f"  x_arr: shape={np.array(x_arr).shape}, type={type(x_arr)}")
    print(f"  u_arr: shape={np.array(u_arr).shape}, type={type(u_arr)}")
    print(f"  len(t_arr)={len(t_arr)}, expected T_data={T_data}")
except Exception as e:
    print(f"  EXCEPTION: {e}")