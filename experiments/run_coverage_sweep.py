"""
Coverage Sweep: PD Controller Parameter Exploration

Purpose:
    Test whether modifying PD controller parameters can restore
    augmented data coverage to acceptable levels.

    GPT/Claude consensus (2026-03-04):
    "PD/excitation 2점 스윕으로 인과를 확정한다."

Design (2 settings + baseline):
    Baseline: gain_margin=3.0, noise_std=0.001  (current, coverage collapse)
    Setting 1 "Relaxed": gain_margin=1.5, noise_std=0.003
        - Lower gain → phi wanders more before tau saturation
        - Higher noise → more perturbation
    Setting 2 "Dither": gain_margin=3.0, noise_std=0.001, + reference dither
        - Keep stability but force phi oscillation via sinusoidal reference
        - tau = Kp*(phi - A*sin(2πft)) + Kd*phi_dot + noise
        - A=0.015 rad, f=0.5 Hz

Coverage Gate (GPT-specified):
    std_ratio(sin(phi)) ≥ 0.7  AND  std_ratio(theta_w_dot²) ≥ 0.5

Output:
    Console report + results JSON
    If a setting passes → can be used for full augmentation run

Usage:
    python experiments/run_coverage_sweep.py

Author: Claude (Gate4c Phase 2, Coverage Diagnosis)
Date: 2026-03-04
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from sklearn.mixture import GaussianMixture

from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.sindy.aek_library import (
    build_aek_library_by_name,
    get_aek_feature_names,
    N_AEK_FEATURES,
)
from src.simulators.aek_simulator import AEKSimulator


# ============================================================
# Configuration
# ============================================================

DATASET_VERSION = 'aek_ood_v1'
SYSTEM = 'aek'
N_TRAIN = 10
REPARAM = 'reparam1'  # Use RP1 for coverage sweep (no Layer3 confound)

POOL_SIZE = 200
POOL_SEED = 42
GMM_N_COMPONENTS = 3
GMM_SEED = 42

# Simulation
DT = 0.01
T_DURATION = 2.0
T_STEPS = 201
MAX_POOL_ATTEMPTS = 5000

# QC thresholds (from runner)
QC_MAX_PHI = 0.30
QC_MAX_PHI_DOT = 5.0
QC_MAX_THETA_W_DOT = 500.0

# Coverage Gate (GPT-specified)
GATE_SIN_PHI_STD_RATIO = 0.70
GATE_TWD2_STD_RATIO = 0.50

# ============================================================
# PD Settings to sweep
# ============================================================

PD_SETTINGS = {
    'dither_r1': {
        'label': 'Round 1 Dither (reference)',
        'gain_margin': 3.0,
        'Kd_factor': 0.15,
        'noise_std': 0.001,
        'dither_amplitude': 0.015,
        'dither_freq': 0.5,
    },
    'dither_plus': {
        'label': 'Dither + Relaxed (combined)',
        'gain_margin': 1.5,
        'Kd_factor': 0.15,
        'noise_std': 0.003,
        'dither_amplitude': 0.015,
        'dither_freq': 0.5,
    },
    'dither_strong': {
        'label': 'Strong Dither (A=0.025, f=1.0Hz, noise=0.003)',
        'gain_margin': 2.0,
        'Kd_factor': 0.15,
        'noise_std': 0.003,
        'dither_amplitude': 0.025,  # rad (~1.43 deg)
        'dither_freq': 1.0,         # Hz (faster oscillation)
    },
}


# ============================================================
# GMM Sampler
# ============================================================

class AEKGMMSampler:
    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, train_x, train_params):
        ics = train_x[:, 0, :]
        params = train_params.reshape(-1, 1) if train_params.ndim == 1 else train_params
        data_5d = np.hstack([ics, params])
        self.gmm.fit(data_5d)
        self._fitted = True

    def sample(self, n_samples):
        samples, _ = self.gmm.sample(n_samples)
        ics = samples[:, :4]
        params = samples[:, 4:5]
        params = np.abs(params)
        params = np.clip(params, 5e-5, 1.5e-4)
        return ics, params


# ============================================================
# Pool Generation (parameterized PD)
# ============================================================

def generate_pool_with_settings(
    gmm: AEKGMMSampler,
    pd_settings: dict,
    pool_seed: int = POOL_SEED,
) -> Dict[str, Any]:
    """
    Generate pool with specified PD controller settings.

    Supports:
        - gain_margin: PD proportional gain multiplier
        - Kd_factor: derivative gain = Kd_factor * Kp
        - noise_std: additive torque noise
        - dither_amplitude: sinusoidal reference amplitude (rad)
        - dither_freq: sinusoidal reference frequency (Hz)
    """
    gain_margin = pd_settings['gain_margin']
    Kd_factor = pd_settings['Kd_factor']
    noise_std = pd_settings['noise_std']
    dither_amp = pd_settings['dither_amplitude']
    dither_freq = pd_settings['dither_freq']

    rng = np.random.default_rng(pool_seed)
    trajectories, dx_list, u_list = [], [], []
    n_accepted, n_attempted = 0, 0

    while n_accepted < POOL_SIZE and n_attempted < MAX_POOL_ATTEMPTS:
        batch = min(100, POOL_SIZE - n_accepted)
        ics_batch, params_batch = gmm.sample(batch)

        for i in range(batch):
            if n_accepted >= POOL_SIZE:
                break
            n_attempted += 1
            ic = ics_batch[i]
            I_w_C = float(params_batch[i, 0])

            sim = AEKSimulator(params={'I_w_C': I_w_C})
            dp = sim.get_derived_params()
            Mgh = dp['M_total'] * dp['g'] * dp['h_cm']
            Kp = gain_margin * Mgh
            Kd = Kd_factor * Kp

            noise_seq = rng.normal(0, noise_std, size=T_STEPS)

            # Pre-compute dither reference if applicable
            if dither_amp > 0:
                t_grid = np.linspace(0, T_DURATION, T_STEPS)
                # Random phase per trajectory for diversity
                phase = rng.uniform(0, 2 * np.pi)
                dither_ref = dither_amp * np.sin(2 * np.pi * dither_freq * t_grid + phase)
            else:
                dither_ref = np.zeros(T_STEPS)

            def _make_ctrl(kp, kd, noise, dref, dt_val, T_val):
                def ctrl(t, x):
                    k = min(int(t / dt_val), T_val - 1)
                    phi_error = x[0] - dref[k]  # track dither reference
                    tau = kp * phi_error + kd * x[1] + noise[k]
                    return tau
                return ctrl

            controller = _make_ctrl(Kp, Kd, noise_seq, dither_ref, DT, T_STEPS)

            try:
                t_arr, x_arr, u_arr = sim.simulate(
                    x0=ic, t_span=(0.0, T_DURATION), dt=DT,
                    controller=controller, method='RK45',
                )
            except Exception:
                continue

            if len(t_arr) != T_STEPS:
                continue
            if np.any(np.abs(x_arr[:, 0]) > QC_MAX_PHI):
                continue
            if np.any(np.abs(x_arr[:, 1]) > QC_MAX_PHI_DOT):
                continue
            if np.any(np.abs(x_arr[:, 3]) > QC_MAX_THETA_W_DOT):
                continue
            if not np.all(np.isfinite(x_arr)):
                continue

            dx_arr = np.zeros_like(x_arr)
            for ti in range(T_STEPS):
                dx_arr[ti] = sim.dynamics(t_arr[ti], x_arr[ti], float(u_arr[ti, 0]))

            trajectories.append(x_arr)
            dx_list.append(dx_arr)
            u_list.append(u_arr)
            n_accepted += 1

    accept_rate = n_accepted / max(n_attempted, 1)
    print(f"  Pool: {n_accepted}/{n_attempted} accepted ({accept_rate:.1%})")

    if n_accepted < POOL_SIZE:
        print(f"  ⚠️ WARNING: Only {n_accepted}/{POOL_SIZE} trajectories accepted!")

    traj_arr = np.array(trajectories) if trajectories else np.empty((0, T_STEPS, 4))
    u_arr_all = np.array(u_list) if u_list else np.empty((0, T_STEPS, 1))

    return {
        'trajectories': traj_arr,
        'dx': np.array(dx_list) if dx_list else np.empty((0, T_STEPS, 4)),
        'u': u_arr_all,
        'n_accepted': n_accepted,
        'n_attempted': n_attempted,
        'accept_rate': accept_rate,
    }


# ============================================================
# Coverage Measurement
# ============================================================

def measure_coverage(
    train_x: np.ndarray,
    train_u: np.ndarray,
    aug_x: np.ndarray,
    aug_u: np.ndarray,
    reparam: str,
) -> Dict[str, Any]:
    """
    Measure coverage metrics for a pool vs training data.

    Returns feature-level and state-level coverage ratios.
    """
    # State coverage
    tr_flat = train_x.reshape(-1, 4)
    au_flat = aug_x.reshape(-1, 4)
    tr_tau = train_u.flatten()
    au_tau = aug_u.flatten()

    state_names = ['phi', 'phi_dot', 'theta_w', 'theta_w_dot']
    state_coverage = {}
    for i, name in enumerate(state_names):
        tr_std = float(np.std(tr_flat[:, i]))
        au_std = float(np.std(au_flat[:, i]))
        state_coverage[name] = {
            'train_std': tr_std,
            'aug_std': au_std,
            'std_ratio': au_std / max(tr_std, 1e-15),
        }

    # tau
    tr_tau_std = float(np.std(tr_tau))
    au_tau_std = float(np.std(au_tau))
    state_coverage['tau'] = {
        'train_std': tr_tau_std,
        'aug_std': au_tau_std,
        'std_ratio': au_tau_std / max(tr_tau_std, 1e-15),
    }

    # tau IQR ratio (bulk coverage)
    tr_iqr = float(np.percentile(tr_tau, 75) - np.percentile(tr_tau, 25))
    au_iqr = float(np.percentile(au_tau, 75) - np.percentile(au_tau, 25))
    state_coverage['tau_iqr_ratio'] = au_iqr / max(tr_iqr, 1e-15)

    # Theta column coverage
    tr_x_flat = train_x.reshape(-1, 4)
    tr_u_flat = train_u.reshape(-1, 1)
    au_x_flat = aug_x.reshape(-1, 4)
    au_u_flat = aug_u.reshape(-1, 1)

    Theta_tr, names = build_aek_library_by_name(tr_x_flat, tr_u_flat, reparam=reparam)
    Theta_au, _ = build_aek_library_by_name(au_x_flat, au_u_flat, reparam=reparam)

    theta_coverage = {}
    for j in range(len(names)):
        tr_std = float(np.std(Theta_tr[:, j]))
        au_std = float(np.std(Theta_au[:, j]))
        theta_coverage[names[j]] = {
            'idx': j,
            'train_std': tr_std,
            'aug_std': au_std,
            'std_ratio': au_std / max(tr_std, 1e-15),
        }

    # Gate features
    sin_phi_ratio = theta_coverage['sin(phi)']['std_ratio']
    twd2_ratio = theta_coverage['theta_w_dot^2']['std_ratio']

    return {
        'state': state_coverage,
        'theta': theta_coverage,
        'feature_names': names,
        'gate_sin_phi_std_ratio': sin_phi_ratio,
        'gate_twd2_std_ratio': twd2_ratio,
    }


def check_coverage_gate(coverage: Dict) -> Dict[str, Any]:
    """
    Check Coverage Gate (GPT-specified thresholds).

    Gate: std_ratio(sin(phi)) ≥ 0.7 AND std_ratio(theta_w_dot²) ≥ 0.5
    """
    sin_r = coverage['gate_sin_phi_std_ratio']
    twd2_r = coverage['gate_twd2_std_ratio']

    g1 = sin_r >= GATE_SIN_PHI_STD_RATIO
    g2 = twd2_r >= GATE_TWD2_STD_RATIO
    overall = g1 and g2

    return {
        'sin_phi_std_ratio': sin_r,
        'sin_phi_gate': g1,
        'sin_phi_threshold': GATE_SIN_PHI_STD_RATIO,
        'twd2_std_ratio': twd2_r,
        'twd2_gate': g2,
        'twd2_threshold': GATE_TWD2_STD_RATIO,
        'overall_pass': overall,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  COVERAGE SWEEP: PD Controller Parameter Exploration")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Library: {REPARAM} (RP1, no Layer3 confound)")
    print(f"  Settings: {list(PD_SETTINGS.keys())}")
    print("=" * 70)

    # Load dataset
    dataset_path = paths.get_dataset_path(DATASET_VERSION, system=SYSTEM)
    validate_dataset_lite(dataset_path)
    dataset = dict(np.load(dataset_path, allow_pickle=True))

    train_x = dataset['train_x'][:N_TRAIN]
    train_u = dataset['train_u'][:N_TRAIN]
    train_params = dataset['train_params'][:N_TRAIN]
    print(f"  Train: {train_x.shape}")

    # Fit GMM (once, shared across all settings)
    gmm = AEKGMMSampler(
        n_components=GMM_N_COMPONENTS,
        covariance_type='full',
        random_state=GMM_SEED,
    )
    gmm.fit(train_x, train_params)
    print(f"  GMM fitted")

    # Sweep
    all_results = {}

    for setting_name, pd_cfg in PD_SETTINGS.items():
        print(f"\n{'='*70}")
        print(f"  Setting: {setting_name} — {pd_cfg['label']}")
        print(f"  gain_margin={pd_cfg['gain_margin']}, "
              f"noise_std={pd_cfg['noise_std']}, "
              f"dither_amp={pd_cfg['dither_amplitude']}")
        print(f"{'='*70}")

        # Generate pool
        print(f"\n  [Pool Generation]")
        pool = generate_pool_with_settings(gmm, pd_cfg, pool_seed=POOL_SEED)

        if pool['n_accepted'] < 50:
            print(f"  ❌ SKIP: Only {pool['n_accepted']} trajectories (need ≥50)")
            all_results[setting_name] = {
                'status': 'insufficient_pool',
                'n_accepted': pool['n_accepted'],
                'accept_rate': pool['accept_rate'],
            }
            continue

        # Pool SHA
        traj_sha = hashlib.sha256(pool['trajectories'].tobytes()).hexdigest()[:16]
        print(f"  traj_sha: {traj_sha}")

        # Measure coverage
        print(f"\n  [Coverage Measurement]")
        coverage = measure_coverage(
            train_x, train_u,
            pool['trajectories'], pool['u'],
            REPARAM,
        )

        # Print state coverage
        print(f"\n  {'Variable':<15s} {'std_ratio':>10s}")
        print(f"  {'-'*25}")
        for var in ['phi', 'phi_dot', 'theta_w', 'theta_w_dot', 'tau']:
            r = coverage['state'][var]['std_ratio']
            flag = " ⚠️" if r < 0.5 else (" ✅" if r >= 0.7 else "")
            print(f"  {var:<15s} {r:>10.2f}{flag}")

        # Print Theta coverage (key features only)
        print(f"\n  {'Feature':<22s} {'std_ratio':>10s}")
        print(f"  {'-'*32}")
        key_features = ['sin(phi)', 'cos(phi)-1', 'theta_w_dot^2',
                        'phi*tau', 'theta_w_dot*tau']
        for feat in key_features:
            if feat in coverage['theta']:
                r = coverage['theta'][feat]['std_ratio']
                flag = " ⚠️" if r < 0.5 else (" ✅" if r >= 0.7 else "")
                print(f"  {feat:<22s} {r:>10.2f}{flag}")

        # Coverage Gate
        gate = check_coverage_gate(coverage)
        print(f"\n  [Coverage Gate]")
        print(f"  sin(phi) std_ratio: {gate['sin_phi_std_ratio']:.2f} "
              f"(threshold ≥ {gate['sin_phi_threshold']}) "
              f"{'✅ PASS' if gate['sin_phi_gate'] else '❌ FAIL'}")
        print(f"  theta_w_dot² std_ratio: {gate['twd2_std_ratio']:.2f} "
              f"(threshold ≥ {gate['twd2_threshold']}) "
              f"{'✅ PASS' if gate['twd2_gate'] else '❌ FAIL'}")
        print(f"  Overall: {'✅ PASS' if gate['overall_pass'] else '❌ FAIL'}")

        all_results[setting_name] = {
            'status': 'completed',
            'label': pd_cfg['label'],
            'pd_config': pd_cfg,
            'n_accepted': pool['n_accepted'],
            'accept_rate': pool['accept_rate'],
            'traj_sha': traj_sha,
            'state_coverage': {
                k: v['std_ratio'] for k, v in coverage['state'].items()
                if isinstance(v, dict) and 'std_ratio' in v
            },
            'gate': gate,
            'theta_coverage_summary': {
                feat: coverage['theta'][feat]['std_ratio']
                for feat in key_features if feat in coverage['theta']
            },
        }

    # ── Final Comparison ──
    print(f"\n{'='*70}")
    print(f"  COVERAGE SWEEP — COMPARISON TABLE")
    print(f"{'='*70}")

    print(f"\n  {'Setting':<12s} {'Accept%':>8s} {'phi':>6s} {'phi_d':>6s} "
          f"{'tw_d':>6s} {'sin_φ':>6s} {'tw_d²':>6s} {'Gate':>6s}")
    print(f"  {'-'*62}")

    for name, res in all_results.items():
        if res['status'] != 'completed':
            print(f"  {name:<12s} {'SKIP':>8s}")
            continue
        sc = res['state_coverage']
        g = res['gate']
        print(f"  {name:<12s} {res['accept_rate']:>7.1%} "
              f"{sc.get('phi', 0):>6.2f} {sc.get('phi_dot', 0):>6.2f} "
              f"{sc.get('theta_w_dot', 0):>6.2f} "
              f"{g['sin_phi_std_ratio']:>6.2f} {g['twd2_std_ratio']:>6.2f} "
              f"{'PASS' if g['overall_pass'] else 'FAIL':>6s}")

    # ── Save ──
    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'run_coverage_sweep.py',
        'reparam': REPARAM,
        'pool_size': POOL_SIZE,
        'pool_seed': POOL_SEED,
        'coverage_gate': {
            'sin_phi_std_ratio': GATE_SIN_PHI_STD_RATIO,
            'twd2_std_ratio': GATE_TWD2_STD_RATIO,
        },
        'results': all_results,
    }

    out_dir = paths.RESULTS_ROOT / DATASET_VERSION / 'gate4c-2' / 'diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'coverage_sweep_r2.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n  Results saved: {out_path}")

    # ── Recommendation ──
    passed = [name for name, res in all_results.items()
              if res.get('status') == 'completed'
              and res.get('gate', {}).get('overall_pass', False)]

    print(f"\n{'='*70}")
    if passed:
        print(f"  ✅ GATE PASSED: {passed}")
        print(f"  → Use these settings for full augmentation run")
    else:
        print(f"  ❌ NO SETTING PASSED COVERAGE GATE")
        print(f"  → Consider more aggressive excitation or wider QC bounds")
    print(f"{'='*70}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


if __name__ == '__main__':
    main()
