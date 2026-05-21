"""
Coverage Diagnostic: Train vs Augmented Data Quality Analysis

Purpose:
    Investigate why AEK augmentation fails downstream despite κ improvements.
    Hypothesis: Augmented data quality/coverage is the bottleneck, not library.

    GPT/Claude consensus (2026-03-03):
    "선택지 C 우선 — coverage 진단 3종 즉시 실행 (비용 0)"

Diagnostics (GPT-specified):
    D1: tau distribution comparison (train vs aug)
    D2: Theta column-wise coverage ratio (aug range / train range)
    D3: Fragile pair coverage hole analysis
    D4: Seed asymmetry analysis (#13 column behavior per baseline seed)

Output:
    results/aek_ood_v1/gate4c-2/diagnostics/coverage_diagnostic.json
    Console report with all findings

Usage:
    python experiments/run_coverage_diagnostic.py

Author: Claude (Gate4c Phase 2)
Date: 2026-03-03
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
from src.sindy.optimizer import ColumnScaler
from src.sindy.aek_library import (
    build_aek_library_by_name,
    get_aek_feature_names,
    get_aek_oracle_support,
    AEK_TARGET_NAMES,
    N_AEK_FEATURES,
)
from src.simulators.aek_simulator import AEKSimulator


# ============================================================
# Configuration (match run_aek4c2_random.py exactly)
# ============================================================

DATASET_VERSION = 'aek_ood_v1'
SYSTEM = 'aek'
N_TRAIN = 10
REPARAM = 'reparam2'

# Pool generation (must match runner for reproducibility)
POOL_SIZE = 200
POOL_SEED = 42
GMM_N_COMPONENTS = 3
GMM_SEED = 42

# QC thresholds (from runner)
QC_MAX_PHI = 0.30
QC_MAX_PHI_DOT = 5.0
QC_MAX_THETA_W_DOT = 500.0

# PD controller (from runner)
PD_GAIN_MARGIN = 3.0
PD_KD_FACTOR = 0.15
PD_NOISE_STD = 0.001

# Simulation
DT = 0.01
T_DURATION = 2.0
T_STEPS = 201
MAX_POOL_ATTEMPTS = 5000


# ============================================================
# GMM Sampler (copied from runner for standalone execution)
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
# Pool Generation (copied from runner)
# ============================================================

def generate_pool(gmm, train_x, train_u):
    """Generate pool — deterministic with POOL_SEED."""
    # Estimate PD gains
    N = train_x.shape[0]
    kp_list, kd_list = [], []
    for i in range(N):
        phi = train_x[i, :, 0]
        phi_dot = train_x[i, :, 1]
        tau = train_u[i, :, 0]
        A = np.column_stack([phi, phi_dot])
        coeffs, _, _, _ = np.linalg.lstsq(A, tau, rcond=None)
        kp_list.append(coeffs[0])
        kd_list.append(coeffs[1])

    rng = np.random.default_rng(POOL_SEED)
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
            Kp = PD_GAIN_MARGIN * Mgh
            Kd = PD_KD_FACTOR * Kp

            noise_seq = rng.normal(0, PD_NOISE_STD, size=T_STEPS)

            def _make_ctrl(kp, kd, noise, dt_val, T_val):
                def ctrl(t, x):
                    k = min(int(t / dt_val), T_val - 1)
                    return kp * x[0] + kd * x[1] + noise[k]
                return ctrl

            controller = _make_ctrl(Kp, Kd, noise_seq, DT, T_STEPS)

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

    print(f"  Pool: {n_accepted}/{n_attempted} accepted")

    # Verify SHA
    traj_arr = np.array(trajectories)
    sha = hashlib.sha256(traj_arr.tobytes()).hexdigest()[:16]
    print(f"  Pool traj_sha: {sha}")
    assert sha == '87594090343bee29', \
        f"Pool SHA mismatch! Expected 87594090343bee29, got {sha}"

    return {
        'trajectories': traj_arr,
        'dx': np.array(dx_list),
        'u': np.array(u_list),
    }


# ============================================================
# D1: Tau Distribution Comparison
# ============================================================

def diagnose_tau_distribution(train_u, aug_u):
    """D1: Compare torque distributions in detail."""
    print(f"\n{'='*60}")
    print(f"  D1: TAU DISTRIBUTION COMPARISON")
    print(f"{'='*60}")

    tr = train_u.flatten()
    au = aug_u.flatten()

    def _stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'abs_max': float(np.abs(arr).max()),
            'range': float(arr.max() - arr.min()),
            'q05': float(np.percentile(arr, 5)),
            'q25': float(np.percentile(arr, 25)),
            'q50': float(np.percentile(arr, 50)),
            'q75': float(np.percentile(arr, 75)),
            'q95': float(np.percentile(arr, 95)),
            'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }

    ts = _stats(tr)
    aus = _stats(au)

    print(f"\n  {'Metric':<20s} {'Train':>12s} {'Aug':>12s} {'Ratio':>8s}")
    print(f"  {'-'*52}")
    for key in ['mean', 'std', 'range', 'abs_max', 'q05', 'q50', 'q95', 'iqr']:
        t_val = ts[key]
        a_val = aus[key]
        if abs(t_val) > 1e-10:
            ratio = a_val / t_val
        else:
            ratio = float('nan')
        print(f"  {key:<20s} {t_val:>12.6f} {a_val:>12.6f} {ratio:>8.2f}")

    # Coverage ratio
    range_ratio = aus['range'] / max(ts['range'], 1e-15)
    std_ratio = aus['std'] / max(ts['std'], 1e-15)

    print(f"\n  Range coverage: aug/train = {range_ratio:.2%}")
    print(f"  Std coverage:   aug/train = {std_ratio:.2%}")

    verdict = "NARROW" if range_ratio < 0.5 else ("COMPARABLE" if range_ratio < 0.8 else "ADEQUATE")
    print(f"  Verdict: {verdict}")

    return {
        'train': ts, 'aug': aus,
        'range_ratio': range_ratio,
        'std_ratio': std_ratio,
        'verdict': verdict,
    }


# ============================================================
# D2: Theta Column-wise Coverage
# ============================================================

def diagnose_theta_coverage(train_x, train_u, aug_x, aug_u, reparam):
    """D2: Compare feature matrix coverage column by column."""
    print(f"\n{'='*60}")
    print(f"  D2: THETA COLUMN-WISE COVERAGE")
    print(f"{'='*60}")

    # Flatten
    tr_x = train_x.reshape(-1, 4)
    tr_u = train_u.reshape(-1, 1)
    au_x = aug_x.reshape(-1, 4)
    au_u = aug_u.reshape(-1, 1)

    Theta_tr, names = build_aek_library_by_name(tr_x, tr_u, reparam=reparam)
    Theta_au, _ = build_aek_library_by_name(au_x, au_u, reparam=reparam)

    n_features = len(names)
    results = []

    print(f"\n  {'#':>3s} {'Feature':<22s} {'tr_std':>10s} {'au_std':>10s} "
          f"{'std_ratio':>10s} {'tr_range':>10s} {'au_range':>10s} {'rng_ratio':>10s}")
    print(f"  {'-'*96}")

    for j in range(n_features):
        tr_col = Theta_tr[:, j]
        au_col = Theta_au[:, j]

        tr_std = float(np.std(tr_col))
        au_std = float(np.std(au_col))
        tr_range = float(tr_col.max() - tr_col.min())
        au_range = float(au_col.max() - au_col.min())

        std_ratio = au_std / max(tr_std, 1e-15)
        rng_ratio = au_range / max(tr_range, 1e-15)

        flag = ""
        if std_ratio < 0.3:
            flag = " ⚠️ VERY_NARROW"
        elif std_ratio < 0.5:
            flag = " ⚠️ NARROW"

        print(f"  {j:>3d} {names[j]:<22s} {tr_std:>10.4e} {au_std:>10.4e} "
              f"{std_ratio:>10.2f} {tr_range:>10.4e} {au_range:>10.4e} "
              f"{rng_ratio:>10.2f}{flag}")

        results.append({
            'feature_idx': j,
            'feature_name': names[j],
            'train_std': tr_std,
            'aug_std': au_std,
            'std_ratio': std_ratio,
            'train_range': tr_range,
            'aug_range': au_range,
            'range_ratio': rng_ratio,
        })

    # Summary
    std_ratios = [r['std_ratio'] for r in results if r['train_std'] > 1e-10]
    narrow = [r for r in results if r['std_ratio'] < 0.5 and r['train_std'] > 1e-10]

    print(f"\n  Median std_ratio: {np.median(std_ratios):.2f}")
    print(f"  Narrow columns (std_ratio < 0.5): {len(narrow)}")
    for r in narrow:
        print(f"    #{r['feature_idx']} {r['feature_name']}: std_ratio={r['std_ratio']:.2f}")

    return {
        'columns': results,
        'median_std_ratio': float(np.median(std_ratios)),
        'n_narrow': len(narrow),
        'narrow_features': [r['feature_name'] for r in narrow],
    }


# ============================================================
# D3: State-space Coverage (phi, phi_dot, tau, theta_w_dot)
# ============================================================

def diagnose_state_coverage(train_x, train_u, aug_x, aug_u):
    """D3: Compare state variable distributions."""
    print(f"\n{'='*60}")
    print(f"  D3: STATE-SPACE COVERAGE")
    print(f"{'='*60}")

    state_names = ['phi', 'phi_dot', 'theta_w', 'theta_w_dot']
    tr_flat = train_x.reshape(-1, 4)
    au_flat = aug_x.reshape(-1, 4)
    tr_tau = train_u.flatten()
    au_tau = aug_u.flatten()

    all_vars = list(state_names) + ['tau']
    all_tr = [tr_flat[:, i] for i in range(4)] + [tr_tau]
    all_au = [au_flat[:, i] for i in range(4)] + [au_tau]

    results = []
    print(f"\n  {'Variable':<15s} {'tr_std':>10s} {'au_std':>10s} "
          f"{'ratio':>8s} {'tr_range':>12s} {'au_range':>12s} {'rng_ratio':>8s}")
    print(f"  {'-'*73}")

    for name, tr, au in zip(all_vars, all_tr, all_au):
        tr_std = float(np.std(tr))
        au_std = float(np.std(au))
        tr_range = float(tr.max() - tr.min())
        au_range = float(au.max() - au.min())
        std_r = au_std / max(tr_std, 1e-15)
        rng_r = au_range / max(tr_range, 1e-15)

        flag = " ⚠️" if std_r < 0.5 else ""
        print(f"  {name:<15s} {tr_std:>10.4e} {au_std:>10.4e} "
              f"{std_r:>8.2f} {tr_range:>12.4e} {au_range:>12.4e} "
              f"{rng_r:>8.2f}{flag}")

        results.append({
            'variable': name,
            'train_std': tr_std, 'aug_std': au_std, 'std_ratio': std_r,
            'train_range': tr_range, 'aug_range': au_range, 'range_ratio': rng_r,
        })

    return {'variables': results}


# ============================================================
# D4: Seed Asymmetry — Baseline #13 Behavior
# ============================================================

def diagnose_seed_asymmetry():
    """D4: Compare baseline seed=0 vs seed=1 for #13 column behavior."""
    print(f"\n{'='*60}")
    print(f"  D4: SEED ASYMMETRY — #13 COLUMN ANALYSIS")
    print(f"{'='*60}")

    results = {}

    for seed in [0, 1]:
        base = (paths.ROOT / 'results' / DATASET_VERSION
                / 'gate1' / 'standardized' / 'esindy'
                / f'n{N_TRAIN}' / f'seed{seed}')

        # Find RP2 baseline
        candidates = sorted(base.iterdir(), reverse=True)
        bdir = None
        for c in candidates:
            if c.is_dir() and 'rp2' in c.name:
                bdir = c
                break

        if bdir is None:
            print(f"  seed={seed}: No RP2 baseline found")
            continue

        print(f"\n  seed={seed}: {bdir.name}")

        # Load artifacts
        z_before = np.load(bdir / 'z_before.npy')  # (14, 4)
        teacher_support = np.load(bdir / 'teacher_support.npy')

        # Load teacher coefficients
        coeff = np.zeros((N_AEK_FEATURES, 4))
        with open(bdir / 'sindy_coefficients.csv', 'r') as f:
            import csv
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if i >= N_AEK_FEATURES:
                    break
                for j in range(4):
                    coeff[i, j] = float(row[j + 1])

        # Load fragile pairs
        with open(bdir / 'fragile_pairs.json', 'r') as f:
            fp_data = json.load(f)

        # #13 analysis
        feat_names = get_aek_feature_names(REPARAM)
        print(f"  #13 = {feat_names[13]}")
        print(f"  Teacher coeff #13:")
        for t in range(4):
            active = "ON" if teacher_support[13, t] else "off"
            print(f"    target {t} ({AEK_TARGET_NAMES[t]}): "
                  f"coeff={coeff[13, t]:.6e}, z={z_before[13, t]:.4f}, "
                  f"support={active}")

        # Check which pair is dynamics fragile=1
        dyn_pairs = [p for p in fp_data['pairs'] if p['type'] == 'dynamics']
        spur_pairs = [p for p in fp_data['pairs'] if p['type'] == 'spurious']
        print(f"  Fragile: {len(fp_data['pairs'])} "
              f"(dyn={len(dyn_pairs)}, spur={len(spur_pairs)})")
        if dyn_pairs:
            print(f"  Dynamics fragile pairs:")
            for dp in dyn_pairs:
                print(f"    #{dp['feature_idx']} {dp['feature_name']} → "
                      f"target {dp['target_idx']} ({dp['target_name']}), "
                      f"z={dp['z_score']:.4f}")

        # #13 in fragile pairs?
        fp_13 = [p for p in fp_data['pairs'] if p['feature_idx'] == 13]
        if fp_13:
            print(f"  #13 fragile entries:")
            for p in fp_13:
                print(f"    target {p['target_idx']}: type={p['type']}, z={p['z_score']:.4f}")

        # Overall teacher coefficient magnitudes for cross-terms
        print(f"  Cross-term teacher coefficients (max |coeff| across targets):")
        for idx in [9, 10, 11, 12, 13]:
            max_abs = np.max(np.abs(coeff[idx, :]))
            print(f"    #{idx:2d} {feat_names[idx]:22s}: {max_abs:.4e}")

        results[f'seed{seed}'] = {
            'coeff_13': coeff[13, :].tolist(),
            'z_13': z_before[13, :].tolist(),
            'support_13': teacher_support[13, :].tolist(),
            'n_fragile': len(fp_data['pairs']),
            'n_dynamics': len(dyn_pairs),
            'n_spurious': len(spur_pairs),
            'dynamics_pairs': dyn_pairs,
            'fragile_13': fp_13,
        }

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  COVERAGE DIAGNOSTIC: Train vs Augmented")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Library: {REPARAM}")
    print("=" * 60)

    # Load dataset
    dataset_path = paths.get_dataset_path(DATASET_VERSION, system=SYSTEM)
    validate_dataset_lite(dataset_path)
    dataset = dict(np.load(dataset_path, allow_pickle=True))

    train_x = dataset['train_x'][:N_TRAIN]
    train_u = dataset['train_u'][:N_TRAIN]
    train_dx = dataset['train_dx'][:N_TRAIN]
    train_params = dataset['train_params'][:N_TRAIN]
    print(f"  Train: {train_x.shape}")

    # Regenerate pool (deterministic)
    print("\n[Pool] Regenerating (pool_seed=42)...")
    gmm = AEKGMMSampler(
        n_components=GMM_N_COMPONENTS,
        covariance_type='full',
        random_state=GMM_SEED,
    )
    gmm.fit(train_x, train_params)
    pool = generate_pool(gmm, train_x, train_u)

    aug_x = pool['trajectories']  # (200, 201, 4)
    aug_u = pool['u']             # (200, 201, 1)

    # === D1: Tau distribution ===
    d1 = diagnose_tau_distribution(train_u, aug_u)

    # === D2: Theta column coverage ===
    d2 = diagnose_theta_coverage(train_x, train_u, aug_x, aug_u, REPARAM)

    # === D3: State-space coverage ===
    d3 = diagnose_state_coverage(train_x, train_u, aug_x, aug_u)

    # === D4: Seed asymmetry ===
    d4 = diagnose_seed_asymmetry()

    # === Save ===
    output = {
        'timestamp': datetime.now().isoformat(),
        'script': 'run_coverage_diagnostic.py',
        'reparam': REPARAM,
        'n_train': N_TRAIN,
        'pool_size': POOL_SIZE,
        'd1_tau_distribution': d1,
        'd2_theta_coverage': d2,
        'd3_state_coverage': d3,
        'd4_seed_asymmetry': d4,
    }

    out_dir = paths.RESULTS_ROOT / DATASET_VERSION / 'gate4c-2' / 'diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'coverage_diagnostic.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n  Results saved: {out_path}")

    # === Final Summary ===
    print(f"\n{'='*60}")
    print(f"  COVERAGE DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    print(f"\n  D1 (tau):     {d1['verdict']} "
          f"(range ratio={d1['range_ratio']:.2%}, "
          f"std ratio={d1['std_ratio']:.2%})")
    print(f"  D2 (Theta):   median std_ratio={d2['median_std_ratio']:.2f}, "
          f"narrow columns={d2['n_narrow']}")
    if d2['narrow_features']:
        print(f"                {d2['narrow_features']}")
    print(f"  D4 (seed):    see above for #13 coefficient details")

    print(f"\n{'='*60}")


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
