"""
Diagnose: what Kp does each I_w_C value need?
The gravity torque coefficient = M_total * g * h_cm depends on I_w_C.
For stability: Kp > M_total * g * h_cm  (linearized condition)
"""
import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.mixture import GaussianMixture
from src.simulators.aek_simulator import AEKSimulator

ds = np.load(_PROJECT_ROOT / "data" / "aek" / "aek_ood_v1" / "dataset.npz", allow_pickle=True)
train_x = ds['train_x']
train_params = ds['train_params'].reshape(-1)
N = train_x.shape[0]

print("=" * 60)
print("Adaptive PD Analysis")
print("=" * 60)

# ── Nominal gravity torque coefficient ──
print("\n── Gravity torque coefficient per I_w_C ──")
print(f"  {'I_w_C':>12s}  {'M_total':>8s}  {'h_cm':>8s}  {'I_p':>10s}  "
      f"{'Mgh':>8s}  {'Mgh/Ip':>8s}  {'Kp_min':>8s}")

test_iwc = [5e-5, 6.95e-5, 8.69e-5, 1.04e-4, 1.5e-4, 3e-4, 1e-3, 1.5e-3]
for iwc in test_iwc:
    sim = AEKSimulator(params={'I_w_C': iwc})
    dp = sim.get_derived_params()
    Mgh = dp['M_total'] * dp['g'] * dp['h_cm']
    ratio = Mgh / dp['I_p']
    print(f"  {iwc:12.6f}  {dp['M_total']:8.4f}  {dp['h_cm']:8.5f}  "
          f"{dp['I_p']:10.7f}  {Mgh:8.5f}  {ratio:8.3f}  Kp>{Mgh:.4f}")

# ── Training I_w_C values ──
print(f"\n── Training I_w_C values (N={N}) ──")
for i, p in enumerate(train_params):
    sim = AEKSimulator(params={'I_w_C': float(p)})
    dp = sim.get_derived_params()
    Mgh = dp['M_total'] * dp['g'] * dp['h_cm']
    print(f"  train[{i}]: I_w_C={p:.6f}, Mgh={Mgh:.5f}")

# ── What GMM produces ──
ics = train_x[:, 0, :]
params_flat = train_params.reshape(-1, 1)
data_5d = np.hstack([ics, params_flat])
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(data_5d)

samples, _ = gmm.sample(100)
gmm_iwc = np.abs(samples[:, 4])
gmm_iwc_clipped = np.clip(gmm_iwc, 1e-6, 1e-2)
print(f"\n── GMM I_w_C distribution (100 samples, current clip [1e-6, 1e-2]) ──")
print(f"  min={gmm_iwc_clipped.min():.6f}, max={gmm_iwc_clipped.max():.6f}, "
      f"mean={gmm_iwc_clipped.mean():.6f}, median={np.median(gmm_iwc_clipped):.6f}")
print(f"  In training range [6.9e-5, 8.7e-5]: "
      f"{np.sum((gmm_iwc_clipped >= 5e-5) & (gmm_iwc_clipped <= 1.1e-4))}/100")

# ── Proposed fix: tighter clipping ──
ood_range = [5e-5, 1.5e-4]  # covers train + test with margin
gmm_iwc_tight = np.clip(gmm_iwc, ood_range[0], ood_range[1])
print(f"\n── With tight clip [{ood_range[0]:.0e}, {ood_range[1]:.0e}] ──")
print(f"  min={gmm_iwc_tight.min():.6f}, max={gmm_iwc_tight.max():.6f}, "
      f"mean={gmm_iwc_tight.mean():.6f}")

# ── Test adaptive PD with tight clip ──
print(f"\n── Adaptive PD + tight clip simulation test (20 samples) ──")
rng = np.random.default_rng(42)
from src.simulators.aek_simulator import AEKSimulator
T = train_x.shape[1]
dt = float(ds['dt'])
t_total = (T - 1) * dt

samples20, _ = gmm.sample(20)
n_accept = 0
gain_margin = 3.0  # Kp = gain_margin * Mgh
Kd_factor = 0.15   # Kd = Kd_factor * Kp (damping ratio)

for i in range(20):
    ic = samples20[i, :4]
    raw_iwc = abs(samples20[i, 4])
    iwc = np.clip(raw_iwc, ood_range[0], ood_range[1])
    
    sim = AEKSimulator(params={'I_w_C': float(iwc)})
    dp = sim.get_derived_params()
    Mgh = dp['M_total'] * dp['g'] * dp['h_cm']
    
    Kp = gain_margin * Mgh
    Kd = Kd_factor * Kp
    noise = rng.normal(0, 0.001, size=T)
    
    def _make_ctrl(kp, kd, ns, dt_v, T_v):
        def ctrl(t, x):
            k = min(int(t / dt_v), T_v - 1)
            return kp * x[0] + kd * x[1] + ns[k]
        return ctrl
    
    controller = _make_ctrl(Kp, Kd, noise, dt, T)
    try:
        t_arr, x_arr, u_arr = sim.simulate(
            x0=ic, t_span=(0.0, t_total), dt=dt,
            controller=controller, method='RK45')
    except Exception as e:
        print(f"  [{i:2d}] EXCEPTION: {e}")
        continue
    
    phi_max = np.abs(x_arr[:, 0]).max()
    pd_max = np.abs(x_arr[:, 1]).max()
    tag = "ACCEPTED" if phi_max < 0.30 else "REJECT"
    if phi_max < 0.30:
        n_accept += 1
    print(f"  [{i:2d}] {tag}: I_w_C={iwc:.6f}, Kp={Kp:.3f}, Kd={Kd:.4f}, "
          f"Mgh={Mgh:.4f}, phi_max={phi_max:.4f}, phi_dot_max={pd_max:.2f}")

print(f"\n  Accept rate: {n_accept}/20 = {n_accept/20:.0%}")