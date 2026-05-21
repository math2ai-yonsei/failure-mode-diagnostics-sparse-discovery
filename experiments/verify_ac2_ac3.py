"""
Gate4c Pre-flight Verification: AC2 + AC3

GPT Cross-Review Acceptance Conditions (2026-02-10):

AC2: Oracle/Spurious 의미 고정 Assert
  - feature_names 인덱스가 oracle 의미와 일치하는지 검증
  - Standard, Reparam-1 모두 체크

AC3: Dataset–Oracle EOM 일치성 1회 체크
  - dataset.npz의 train_dx와 aek.yaml EOM으로 계산한 dx 비교
  - max abs error 로그 남기기

Usage:
  python experiments/verify_ac2_ac3.py

Output:
  Console log + results/aek_ood_v1/gate4c/ac_verification.json
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import yaml
import numpy as np
from datetime import datetime

from src.contracts import paths
from src.sindy.aek_library import (
    AEK_FEATURE_NAMES,
    AEK_REPARAM1_FEATURE_NAMES,
    AEK_TARGET_NAMES,
    N_AEK_FEATURES,
    get_aek_oracle_support,
    get_aek_oracle_coefficients,
    get_aek_feature_names,
)


def verify_ac2():
    """
    AC2: Oracle/Spurious 의미 고정 Assert.

    Oracle terms are defined by PHYSICAL MEANING, not just index:
      - tau (motor torque) must be at index 3
      - sin(phi) (gravity coupling) must be at index 4

    For each parameterization, verify that the feature at each oracle-relevant
    index has the expected name. This prevents silent breakage if library
    order is ever changed.
    """
    print("=" * 70)
    print("  AC2: Oracle/Spurious Label Integrity Check")
    print("=" * 70)

    results = {}
    all_pass = True

    # Define expected mappings (index → expected name)
    # These are the oracle-relevant features that MUST be at specific indices
    oracle_index_map = {
        0: '1',            # constant (not oracle, but structural)
        1: 'phi_dot',      # oracle for target 0
        2: 'theta_w_dot',  # oracle for target 2
        3: 'tau',          # oracle for targets 1, 3
        4: 'sin(phi)',     # oracle for targets 1, 3
    }

    # Parameterization-specific checks
    reparam_specific = {
        'standard': {5: 'cos(phi)'},
        'reparam1': {5: 'cos(phi)-1'},
    }

    for reparam in ['standard', 'reparam1']:
        print(f"\n  [{reparam}]")
        names = get_aek_feature_names(reparam)
        checks = []

        # Common oracle checks
        for idx, expected in oracle_index_map.items():
            actual = names[idx]
            ok = actual == expected
            status = "✅" if ok else "❌ FAIL"
            print(f"    idx {idx}: expected='{expected}', actual='{actual}' {status}")
            checks.append({
                'index': idx, 'expected': expected,
                'actual': actual, 'pass': ok,
            })
            if not ok:
                all_pass = False

        # Reparam-specific check
        for idx, expected in reparam_specific[reparam].items():
            actual = names[idx]
            ok = actual == expected
            status = "✅" if ok else "❌ FAIL"
            print(f"    idx {idx}: expected='{expected}', actual='{actual}' {status}")
            checks.append({
                'index': idx, 'expected': expected,
                'actual': actual, 'pass': ok,
            })
            if not ok:
                all_pass = False

        # Verify total count
        ok_count = len(names) == N_AEK_FEATURES
        print(f"    n_features: expected={N_AEK_FEATURES}, actual={len(names)} "
              f"{'✅' if ok_count else '❌ FAIL'}")
        if not ok_count:
            all_pass = False

        # Verify oracle support is consistent
        oracle_support = get_aek_oracle_support()
        # Oracle terms: (feature_idx, target_idx) pairs
        oracle_pairs = []
        for f_idx in range(N_AEK_FEATURES):
            for t_idx in range(4):
                if oracle_support[f_idx, t_idx]:
                    oracle_pairs.append((f_idx, t_idx, names[f_idx],
                                        AEK_TARGET_NAMES[t_idx]))

        print(f"    Oracle support ({len(oracle_pairs)} terms):")
        for f_idx, t_idx, f_name, t_name in oracle_pairs:
            print(f"      [{f_idx},{t_idx}] {f_name} → {t_name}")

        # Key check: oracle terms must NOT include cos(phi) or cos(phi)-1
        oracle_feature_indices = set(f_idx for f_idx, _, _, _ in oracle_pairs)
        if 5 in oracle_feature_indices:
            print(f"    ❌ CRITICAL: Index 5 ({names[5]}) is in oracle support!")
            print(f"       cos(phi)/cos(phi)-1 should NOT be oracle term!")
            all_pass = False
        else:
            print(f"    ✅ Index 5 ({names[5]}) correctly NOT in oracle support")

        results[reparam] = {
            'checks': checks,
            'n_features_ok': ok_count,
            'oracle_pairs': [
                {'feature_idx': f, 'target_idx': t,
                 'feature_name': fn, 'target_name': tn}
                for f, t, fn, tn in oracle_pairs
            ],
            'idx5_not_in_oracle': 5 not in oracle_feature_indices,
        }

    print(f"\n  AC2 Overall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    return all_pass, results


def verify_ac3():
    """
    AC3: Dataset–Oracle EOM 일치성 1회 체크.

    Load dataset.npz train split, compute dx from EOM using aek.yaml
    nominal parameters, compare with stored train_dx.

    Note: Dataset may use OOD I_w_C values per trajectory.
    We use train_params to get the actual I_w_C for each trajectory.
    """
    print("\n" + "=" * 70)
    print("  AC3: Dataset–Oracle EOM Consistency Check")
    print("=" * 70)

    # Load dataset
    dataset_path = paths.get_dataset_path('aek_ood_v1', system='aek')
    print(f"\n  Dataset: {dataset_path}")
    if not dataset_path.exists():
        print(f"  ❌ Dataset not found!")
        return False, {'error': 'dataset_not_found'}

    data = dict(np.load(dataset_path, allow_pickle=True))

    # Load YAML for base parameters
    yaml_path = paths.ROOT / 'configs' / 'systems' / 'aek.yaml'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        aek_cfg = yaml.safe_load(f)

    # Base parameters (fixed across trajectories)
    m_r = aek_cfg['physics']['rod']['mass_kg']
    R = aek_cfg['physics']['wheel']['radius_m']
    r = aek_cfg['physics']['rod']['cross_section_radius_m']
    l = aek_cfg['physics']['rod']['length_m']
    g = aek_cfg['physics']['g']
    l_AB = l / 2.0
    l_AC = l

    # Rod inertia about pivot (fixed)
    I_r_B = (1.0 / 12.0) * m_r * (3.0 * r**2 + l**2)
    I_r_A = I_r_B + m_r * l_AB**2

    train_x = data['train_x']    # (N, T, 4)
    train_u = data['train_u']    # (N, T, 1)
    train_dx = data['train_dx']  # (N, T, 4)

    # Check if train_params exists (per-trajectory OOD parameters)
    has_params = 'train_params' in data
    if has_params:
        train_params = data['train_params']  # (N,) or (N, k)
        print(f"  train_params shape: {train_params.shape}")
    else:
        print(f"  ⚠️ No train_params found, using nominal I_w_C")

    # Check train_cond_id for OOD condition info
    has_cond_id = 'train_cond_id' in data
    if has_cond_id:
        train_cond_id = data['train_cond_id']
        print(f"  train_cond_id shape: {train_cond_id.shape}")
        print(f"  unique cond_ids: {np.unique(train_cond_id)}")

    N, T, _ = train_x.shape
    print(f"  train_x: ({N}, {T}, 4)")

    # Compute dx from EOM for each trajectory
    max_abs_errors = []
    per_traj_results = []

    for i in range(N):
        # Get I_w_C for this trajectory
        if has_params:
            if train_params.ndim == 1:
                I_w_C_i = float(train_params[i])
            else:
                I_w_C_i = float(train_params[i, 0])
        else:
            I_w_C_i = aek_cfg['inertia']['I_w_C']

        # Recompute derived quantities for this I_w_C
        m_w = 2.0 * I_w_C_i / R**2
        M_total = m_r + m_w
        h_cm = (m_r * l_AB + m_w * l_AC) / M_total
        I_p = I_r_A + m_w * l_AC**2

        # Compute dx from EOM for all timesteps
        phi = train_x[i, :, 0]
        phi_dot = train_x[i, :, 1]
        theta_w_dot = train_x[i, :, 3]
        tau = train_u[i, :, 0]

        dx_computed = np.zeros((T, 4))
        dx_computed[:, 0] = phi_dot
        phi_ddot = (M_total * g * h_cm * np.sin(phi) - tau) / I_p
        dx_computed[:, 1] = phi_ddot
        dx_computed[:, 2] = theta_w_dot
        dx_computed[:, 3] = tau / I_w_C_i - phi_ddot

        # Compare with stored dx
        abs_err = np.abs(train_dx[i] - dx_computed)
        max_err_this = float(abs_err.max())
        max_abs_errors.append(max_err_this)

        # Per-target max errors
        per_target_max = [float(abs_err[:, t].max()) for t in range(4)]

        per_traj_results.append({
            'traj_idx': int(i),
            'I_w_C': float(I_w_C_i),
            'max_abs_error': max_err_this,
            'per_target_max': per_target_max,
        })

        status = "✅" if max_err_this < 1e-10 else "⚠️"
        print(f"    Traj {i}: I_w_C={I_w_C_i:.6e}, max|err|={max_err_this:.2e} {status}")

    overall_max = max(max_abs_errors)
    # Threshold: numerical precision (float64 ~1e-15, but ODE integration might
    # introduce ~1e-12 level differences)
    threshold = 1e-8
    ac3_pass = overall_max < threshold

    print(f"\n  Overall max |error|: {overall_max:.2e}")
    print(f"  Threshold: {threshold:.0e}")
    print(f"  AC3 Overall: {'✅ PASS' if ac3_pass else '❌ FAIL'}")

    if not ac3_pass:
        print(f"\n  ⚠️ Error exceeds threshold!")
        print(f"  Possible causes:")
        print(f"    1. Dataset dx was computed by different EOM")
        print(f"    2. Different I_w_C values than train_params")
        print(f"    3. Numerical integration artifacts")

    results = {
        'dataset_path': str(dataset_path),
        'n_trajectories': N,
        'overall_max_abs_error': overall_max,
        'threshold': threshold,
        'pass': ac3_pass,
        'per_trajectory': per_traj_results,
    }
    return ac3_pass, results


def main():
    print("=" * 70)
    print("  Gate4c Pre-flight: AC2 + AC3 Verification")
    print(f"  Date: {datetime.now().isoformat()}")
    print("=" * 70)

    ac2_pass, ac2_results = verify_ac2()
    ac3_pass, ac3_results = verify_ac3()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  AC2 (Oracle Label Integrity):     {'✅ PASS' if ac2_pass else '❌ FAIL'}")
    print(f"  AC3 (Dataset–EOM Consistency):    {'✅ PASS' if ac3_pass else '❌ FAIL'}")

    overall = ac2_pass and ac3_pass
    print(f"\n  Overall: {'🟢 ALL PASS — Gate4c cleared' if overall else '🔴 FAIL — Fix before Gate4c'}")

    # Save results
    out_dir = paths.ROOT / 'results' / 'aek_ood_v1' / 'gate4c'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'ac_verification.json'

    output = {
        'verified_at': datetime.now().isoformat(),
        'script': 'experiments/verify_ac2_ac3.py',
        'ac2': {'pass': ac2_pass, 'details': ac2_results},
        'ac3': {'pass': ac3_pass, 'details': ac3_results},
        'overall_pass': overall,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✅ Saved: {out_path}")


if __name__ == '__main__':
    main()