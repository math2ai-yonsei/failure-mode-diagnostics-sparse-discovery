"""Quick verification: PD controller pool generation works."""
import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.mixture import GaussianMixture
from src.simulators.aek_simulator import AEKSimulator

ds = np.load(_PROJECT_ROOT / "data" / "aek" / "aek_ood_v1" / "dataset.npz", allow_pickle=True)
train_x = ds['train_x']     # (10, 201, 4)
train_u = ds['train_u']     # (10, 201, 1)
train_params = ds['train_params']  # (10, 1)
N, T, _ = train_x.shape
dt = float(ds['dt'])
t_total = (T - 1) * dt

# ── Estimate PD gains ──
kp_list, kd_list = [], []
for i in range(N):
    A = np.column_stack([train_x[i, :, 0], train_x[i, :, 1]])
    coeffs, _, _, _ = np.linalg.lstsq(A, train_u[i, :, 0], rcond=None)
    kp_list.append(coeffs[0])
    kd_list.append(coeffs[1])
Kp = float(np.median(kp_list))
Kd = float(np.median(kd_list))
print(f"PD gains: Kp={Kp:.4f}, Kd={Kd:.4f}")

# ── GMM ──
ics = train_x[:, 0, :]
params_flat = train_params.reshape(N, -1)[:, :1]
data_5d = np.hstack([ics, params_flat])
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(data_5d)

# ── Generate 20 trajectories with PD controller ──
rng = np.random.default_rng(42)
samples, _ = gmm.sample(20)
sample_ics = samples[:, :4]
sample_params = np.clip(np.abs(samples[:, 4:5]), 1e-6, 1e-2)

QC_PHI = 0.30
QC_PHI_DOT = 5.0
QC_TW_DOT = 500.0
NOISE_STD = 0.002

counts = {'exception': 0, 'len_mismatch': 0, 'qc_phi': 0, 'qc_phi_dot': 0,
          'qc_theta_w_dot': 0, 'nan_inf': 0, 'accepted': 0}

print(f"\n── PD Controller Test (20 samples, QC phi<{QC_PHI}, phi_dot<{QC_PHI_DOT}) ──")
for i in range(20):
    ic = sample_ics[i]
    I_w_C = float(sample_params[i, 0])
    sim = AEKSimulator(params={'I_w_C': I_w_C})

    noise_seq = rng.normal(0, NOISE_STD, size=T)

    def _make_pd_ctrl(kp, kd, noise, dt_val, T_val):
        def ctrl(t, x):
            k = min(int(t / dt_val), T_val - 1)
            return kp * x[0] + kd * x[1] + noise[k]
        return ctrl

    controller = _make_pd_ctrl(Kp, Kd, noise_seq, dt, T)
    try:
        t_arr, x_arr, u_arr = sim.simulate(
            x0=ic, t_span=(0.0, t_total), dt=dt,
            controller=controller, method='RK45')
    except Exception as e:
        counts['exception'] += 1
        print(f"  [{i:2d}] EXCEPTION: {e}")
        continue

    if len(t_arr) != T:
        counts['len_mismatch'] += 1
        continue

    phi_max = np.abs(x_arr[:, 0]).max()
    pd_max = np.abs(x_arr[:, 1]).max()
    td_max = np.abs(x_arr[:, 3]).max()
    has_nan = not np.all(np.isfinite(x_arr))

    reason = None
    if has_nan: reason = 'nan_inf'
    elif phi_max > QC_PHI: reason = 'qc_phi'
    elif pd_max > QC_PHI_DOT: reason = 'qc_phi_dot'
    elif td_max > QC_TW_DOT: reason = 'qc_theta_w_dot'

    if reason:
        counts[reason] += 1
        print(f"  [{i:2d}] REJECT ({reason}): phi_max={phi_max:.4f}, "
              f"phi_dot_max={pd_max:.2f}, tw_dot_max={td_max:.1f}")
    else:
        counts['accepted'] += 1
        print(f"  [{i:2d}] ACCEPTED: phi_max={phi_max:.4f}, "
              f"phi_dot_max={pd_max:.2f}, tw_dot_max={td_max:.1f}")

print(f"\n── Summary ──")
for k, v in counts.items():
    print(f"  {k:20s}: {v}")
print(f"\n  Accept rate: {counts['accepted']}/20 = {counts['accepted']/20:.0%}")