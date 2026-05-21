"""
Gate4d: Cart-Pole Reparam-1 D-optimal Augmentation Runner

Purpose:
    이식 Step 2 — Gate4d Reparam-1 library로 D-optimal augmentation 실행.
    AEK에서 검증된 D-opt 전략을 CP에 이식하여 일반성(generality) 확인.

    Gate4a Standard D-opt 결과 (reference):
        median = +0.424, ci_lower = 0.052 → STRONG_PASS

    Gate4d Reparam-1 D-opt 목적:
        ① κ 개선 확인 (Reparam-1 vs Standard)
        ② score_aligned 비교: Reparam-1 ≥ Standard 0.424?
        ③ 논문 Claim: "Reparam-1은 CP에서도 identifiability를 개선"

Design:
    1. Load CP Reparam-1 baseline (run_gate4d_cp_reparam_baseline.py 출력)
    2. Fit GMM on training ICs+params (Gate3 인프라 재사용)
    3. Generate pool via Gate3 PoolGenerator (CP simulator, analytic dx)
    4. Track A: teacher alignment error top-10% 제거
    5. D-optimal selection: greedy logdet on fragile features (Reparam-1 space)
    6. E-SINDy evaluation (Reparam-1 library)
    7. Compute delta_raw + score_aligned

Metric SSOT (CP — dynamics-primary, recall fragility):
    delta_raw = median(z_after − z_before) over fragile pairs
    score_aligned = +delta_raw  (양수 = 개선)
    ← AEK와 반대 부호! (AEK: score_aligned = −delta_raw)

Confound-free:
    동일 pool, Reparam-1 library에서 Random vs D-opt 비교 가능하도록
    pool_sha 기록. Random runner는 같은 pool에서 실행.

Usage:
    python experiments/run_gate4d_cp_reparam_dopt.py \\
        --baseline_dir results/cartpole_ood_v1/gate4d/standardized/esindy_baseline_rp1/n10/seed42/<run_id>

Author: Claude (Gate4d)
Date: 2026-03-04
Runner version: v1.0
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import hashlib
import csv
import traceback
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble
from src.sindy.cp_library import (
    build_cp_library_by_name,
    get_cp_feature_names,
    assert_cp_feature_integrity,
    N_CP_FEATURES,
    CP_TARGET_NAMES,
)

# Gate3 인프라 재사용 (CP Pool 생성)
from experiments.run_gate3_v2 import (
    Gate3Config,
    GMMProposalSampler,
    PoolGenerator,
    track_a_selection,
    create_rng_streams,
    GATE3_CONFIG,
    DEFAULT_TARGET_NAMES,
)

# D-opt 공유 로직 재사용 (AEK D-opt runner에서 import)
from experiments.run_aek4c_dopt import (
    compute_fragile_feature_sets,
    compute_pool_theta_per_traj,
    compute_gram_contributions,
    greedy_dopt_selection,
)


# ============================================================
# Constants
# ============================================================

RUNNER_VERSION = 'v1.0_gate4d_cp_dopt'

DATASET_VERSION = 'cartpole_ood_v1'
SYSTEM = 'cartpole'

# CP dynamics target indices (x_ddot=1, theta_ddot=3)
CP_DYNAMICS_TARGETS = [1, 3]

# Gate4a Standard D-opt reference
GATE4A_STD_DOPT_MEDIAN = 0.424
GATE4A_STD_DOPT_CI_LOWER = 0.052

# Gate2 ceiling (CP)
GATE2_CEILING = 0.058


def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ============================================================
# Configuration
# ============================================================

@dataclass
class Gate4dCPDoptConfig:
    """Gate4d CP Reparam-1 D-optimal configuration."""
    # Library
    reparam: str = 'reparam1'

    # Pool generation (same as Gate3/Gate4a)
    pool_size: int = 2000
    pool_seed: int = 42
    baseline_seed: int = 42   # for E-SINDy eval seed

    # Selection
    n_select: int = 200
    reject_ratio: float = 0.10

    # D-optimal
    dopt_lambda: float = 1e-6
    dopt_gram_energy_mode: str = 'unit_trace'
    dopt_trace_power: float = 1.0
    dopt_use_teacher_intersection: bool = True   # CP: recall fragile → teacher-active
    dopt_dynamics_targets: List[int] = field(default_factory=lambda: CP_DYNAMICS_TARGETS)

    # E-SINDy
    n_train: int = 10
    n_bootstrap: int = 100
    threshold: float = 0.05
    z_eps: float = 1e-6

    # CI
    ci_bootstrap_B: int = 2000
    ci_alpha: float = 0.05

    # Dataset
    dataset_version: str = DATASET_VERSION


# ============================================================
# Load Baseline Artifacts
# ============================================================

def load_cp_baseline(baseline_dir: Path) -> Dict[str, Any]:
    """
    Load Gate4d CP Reparam-1 baseline artifacts.

    Expected files (from run_gate4d_cp_reparam_baseline.py):
        fragile_pairs.json  → fragile_pairs, z_before
        z_before.npy        → z_before array
        z_full.npy          → z_full (21×4)
        support_mask.npy    → support_mask (21×4)
        sindy_coefficients.csv → teacher coefficients
        metrics.json        → kappa, metadata

    Returns:
        Dict with all baseline artifacts
    """
    baseline_dir = Path(baseline_dir)
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline dir not found: {baseline_dir}")

    # fragile_pairs.json
    fp_path = baseline_dir / 'fragile_pairs.json'
    if not fp_path.exists():
        raise FileNotFoundError(f"fragile_pairs.json not found in {baseline_dir}")
    with open(fp_path) as f:
        fp_data = json.load(f)

    fragile_pairs = fp_data['fragile_pairs']
    z_before = np.load(baseline_dir / 'z_before.npy')
    z_full = np.load(baseline_dir / 'z_full.npy')
    support_mask = np.load(baseline_dir / 'support_mask.npy')

    # Teacher coefficients
    coeff_path = baseline_dir / 'sindy_coefficients.csv'
    teacher_coefficients = np.zeros((N_CP_FEATURES, 4))
    with open(coeff_path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for i, row in enumerate(reader):
            for j in range(4):
                teacher_coefficients[i, j] = float(row[j + 1])

    # Teacher support (from support_mask, same as ensemble)
    teacher_support = support_mask.astype(bool)

    # Metrics
    with open(baseline_dir / 'metrics.json') as f:
        metrics = json.load(f)

    kappa_baseline = metrics.get('kappa', None)
    reparam = metrics.get('reparam', 'reparam1')

    print(f"  Baseline loaded from: {baseline_dir}")
    print(f"  Library: {metrics.get('library_version', reparam)}")
    print(f"  κ (baseline): {kappa_baseline:.3e}" if kappa_baseline else "")
    print(f"  Fragile pairs: {len(fragile_pairs)}")
    print(f"  z_before shape: {z_before.shape}")

    return {
        'dir': baseline_dir,
        'fragile_pairs': fragile_pairs,
        'z_before': z_before,
        'z_full': z_full,
        'support_mask': support_mask,
        'teacher_support': teacher_support,
        'teacher_coefficients': teacher_coefficients,
        'kappa_baseline': kappa_baseline,
        'reparam': reparam,
        'metrics': metrics,
    }


# ============================================================
# Pool Theta (trajectory-level, 3D)
# ============================================================

def compute_pool_theta_cp(
    pool_traj: np.ndarray,    # (N_cand, T, 4)
    pool_u: np.ndarray,       # (N_cand, T, 1)
    train_x: np.ndarray,      # (N_tr, T, 4)
    train_u: np.ndarray,      # (N_tr, T, 1)
    reparam: str,
) -> Tuple[np.ndarray, ColumnScaler]:
    """
    Compute scaled library features per pool trajectory (3D).

    Scaler is fitted on training data (same as E-SINDy evaluation).

    Returns:
        Theta_scaled: (N_cand, T, n_features) — scaled per training scaler
        scaler: ColumnScaler fitted on training Theta
    """
    N_tr, T, D = train_x.shape
    N_cand = pool_traj.shape[0]

    # Fit scaler on training data
    x_flat = train_x.reshape(-1, D)
    u_flat = train_u.reshape(-1, 1)
    Theta_train, _ = build_cp_library_by_name(x_flat, u_flat, reparam=reparam)
    scaler = ColumnScaler()
    scaler.fit(Theta_train)

    # Apply to each pool trajectory
    Theta_scaled = np.zeros((N_cand, T, N_CP_FEATURES))
    for i in range(N_cand):
        x_i = pool_traj[i]    # (T, 4)
        u_i = pool_u[i]       # (T, 1)
        Th_i, _ = build_cp_library_by_name(x_i, u_i, reparam=reparam)
        Theta_scaled[i] = scaler.transform(Th_i)

    return Theta_scaled, scaler


# ============================================================
# E-SINDy Evaluation (train + augmentation)
# ============================================================

def evaluate_augmented_cp(
    train_x: np.ndarray,      # (N_tr, T, 4)
    train_u: np.ndarray,      # (N_tr, T, 1)
    train_dx: np.ndarray,     # (N_tr, T, 4)
    aug_traj: np.ndarray,     # (N_aug, T, 4)
    aug_u: np.ndarray,        # (N_aug, T, 1)
    aug_dx: np.ndarray,       # (N_aug, T, 4)
    reparam: str,
    n_bootstrap: int = 100,
    threshold: float = 0.05,
    seed: int = 42,
    z_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    E-SINDy evaluation with training + augmented data (CP, Reparam-1).

    Returns:
        Dict with z (n_fragile,), z_full (21,4), support_mask, coefficients_mean, kappa
    """
    N_tr, T, D = train_x.shape
    N_aug = aug_traj.shape[0]

    # Concatenate
    all_x  = np.concatenate([train_x,  aug_traj], axis=0)
    all_u  = np.concatenate([train_u,  aug_u],    axis=0)
    all_dx = np.concatenate([train_dx, aug_dx],   axis=0)

    x_flat  = all_x.reshape(-1, D)
    u_flat  = all_u.reshape(-1, 1)
    dx_flat = all_dx.reshape(-1, D)

    # Build library
    Theta, feat_names = build_cp_library_by_name(x_flat, u_flat, reparam=reparam)

    # Scale
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    # Condition number
    kappa = float(np.linalg.cond(Theta_scaled))

    n_traj = N_tr + N_aug
    T_steps = T

    # E-SINDy
    ensemble = ESINDyEnsemble(
        n_bootstrap=n_bootstrap,
        threshold=threshold,
        random_state=seed,
    )
    ensemble.fit(
        Theta_scaled, dx_flat,
        n_trajectories=n_traj,
        T=T_steps,
        scaler=scaler,
        target_scale=None,
    )

    coeff_mean = ensemble.coefficients_mean_    # (21, 4) unscaled
    coeff_std  = ensemble.coefficients_std_     # (21, 4) unscaled
    support    = np.abs(coeff_mean) > 0         # (21, 4)

    z_full = np.abs(coeff_mean) / (coeff_std + z_eps)  # (21, 4)

    return {
        'z_full': z_full,
        'support_mask': support,
        'coefficients_mean': coeff_mean,
        'kappa': kappa,
        'feature_names': feat_names,
        'n_train': N_tr,
        'n_aug': N_aug,
        'n_total_samples': n_traj * T_steps,
        'reparam': reparam,
    }


# ============================================================
# Metrics (CP — score_aligned = +delta_raw)
# ============================================================

def compute_metrics_cp(
    z_after_full: np.ndarray,   # (21, 4)
    z_before: np.ndarray,       # (n_fragile,)
    fragile_pairs: List,
    ci_B: int = 2000,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute delta_raw and score_aligned for CP (recall fragility).

    CP SSOT: score_aligned = +delta_raw (양수 = improvement)

    Fragile pairs: recall fragility (teacher-active, low z)
    delta_raw = median(z_after − z_before) over fragile pairs
    """
    # Extract z_after for fragile pairs
    z_after = np.array([
        float(z_after_full[p[0], p[1]]) for p in fragile_pairs
    ])

    delta_per_pair = z_after - z_before   # positive = improvement
    delta_raw_median = float(np.median(delta_per_pair))

    # CP: score_aligned = +delta_raw
    score_aligned_median = delta_raw_median  # +delta_raw

    # Bootstrap CI on score_aligned
    rng = np.random.default_rng(seed)
    n = len(delta_per_pair)
    if n > 0:
        bootstrap_medians = np.array([
            np.median(rng.choice(delta_per_pair, size=n, replace=True))
            for _ in range(ci_B)
        ])
        ci_lower = float(np.percentile(bootstrap_medians, 100 * ci_alpha / 2))
        ci_upper = float(np.percentile(bootstrap_medians, 100 * (1 - ci_alpha / 2)))
    else:
        ci_lower = ci_upper = 0.0

    # Pass level (same thresholds as Gate3/4a)
    if ci_lower > GATE2_CEILING:
        pass_level = 'CEILING_BREAK'
    elif ci_lower > 0:
        pass_level = 'STRONG_PASS'
    elif score_aligned_median > 0:
        pass_level = 'SOFT_PASS'
    else:
        pass_level = 'NULL'

    return {
        'delta_raw_median': delta_raw_median,
        'score_aligned_median': score_aligned_median,
        'score_aligned_ci_lower': ci_lower,
        'score_aligned_ci_upper': ci_upper,
        'pass_level': pass_level,
        'n_fragile_pairs': n,
        'z_after_fragile': z_after.tolist(),
        'z_before_fragile': z_before.tolist(),
        'delta_per_pair': delta_per_pair.tolist(),
        'metric_sign_convention': 'CP: score_aligned = +delta_raw',
    }


# ============================================================
# D-optimal Selection (CP wrapper)
# ============================================================

def dopt_select_cp(
    pool: Dict[str, Any],
    track_a: Dict[str, Any],
    fragile_pairs: List,
    teacher_support: np.ndarray,
    reparam: str,
    train_x: np.ndarray,
    train_u: np.ndarray,
    cfg: Gate4dCPDoptConfig,
) -> Dict[str, Any]:
    """
    D-optimal selection for CP (Reparam-1 library).

    Differences from AEK:
        - use_teacher_intersection=True (CP fragile = recall, teacher-active)
        - 21-term library (vs 14-term AEK)
        - CP dynamics targets = [1, 3] (x_ddot, theta_ddot)
    """
    print(f"\n[D-optimal Selection] n_select={cfg.n_select}")

    candidates = track_a['selected_indices']
    N_cand = len(candidates)
    print(f"  Track A candidates: {N_cand}")

    # Step 1: Fragile feature sets per dynamics target
    # CP: teacher_intersection=True → only teacher-active fragile features
    F_by_target = {}
    for t in cfg.dopt_dynamics_targets:
        t_fragile_features = set()
        for p in fragile_pairs:
            fi, ti = p[0], p[1]
            if ti == t:
                # Teacher intersection check
                if cfg.dopt_use_teacher_intersection:
                    if teacher_support[fi, ti]:
                        t_fragile_features.add(fi)
                else:
                    t_fragile_features.add(fi)
        F_by_target[t] = np.array(sorted(t_fragile_features), dtype=int)

    target_label = {1: 'd(x_dot)/dt', 3: 'd(theta_dot)/dt'}
    for t, F_t in F_by_target.items():
        print(f"  F_{target_label.get(t, t)}: {len(F_t)} features → {F_t.tolist()}")

    # Step 2: Scaled pool library (3D: N_cand, T, n_features)
    print(f"\n  Computing scaled library features for {N_cand} candidates...")
    pool_traj_cand = pool['trajectories'][candidates]  # (N_cand, T, 4)
    pool_u_cand    = pool['u'][candidates]             # (N_cand, T, 1)

    Theta_scaled, _ = compute_pool_theta_cp(
        pool_traj_cand, pool_u_cand,
        train_x, train_u,
        reparam=reparam,
    )

    # Step 3: Gram contributions
    G_by_target, _ = compute_gram_contributions(
        Theta_scaled, F_by_target,
        gram_energy_mode=cfg.dopt_gram_energy_mode,
        trace_power=cfg.dopt_trace_power,
    )

    # Step 4: Greedy D-optimal selection
    print(f"\n  Greedy D-optimal selection...")
    selected_pool_indices, selection_trace = greedy_dopt_selection(
        G_by_target=G_by_target,
        candidate_pool_indices=candidates,
        n_select=cfg.n_select,
        lambda_reg=cfg.dopt_lambda,
    )

    # Validate selection count
    n_selected = len(selected_pool_indices)
    print(f"  Selected: {n_selected} trajectories")

    # Build dopt_spec for manifest
    dopt_spec = {
        'lambda': cfg.dopt_lambda,
        'gram_energy_mode': cfg.dopt_gram_energy_mode,
        'trace_power': cfg.dopt_trace_power,
        'use_teacher_intersection': cfg.dopt_use_teacher_intersection,
        'dynamics_targets': cfg.dopt_dynamics_targets,
        'F_by_target': {str(t): F_t.tolist() for t, F_t in F_by_target.items()},
        'n_candidates': N_cand,
        'n_selected': n_selected,
        'spec_hash': hashlib.sha256(
            json.dumps({'F_by_target': {
                str(k): v.tolist() for k, v in F_by_target.items()
            }}, sort_keys=True).encode()
        ).hexdigest()[:16],
    }

    return {
        'pool_indices': selected_pool_indices,
        'trajectories': pool['trajectories'][selected_pool_indices],
        'u': pool['u'][selected_pool_indices],
        'dx': pool['dx'][selected_pool_indices],
        'params': pool['params'][selected_pool_indices],
        'selection_trace': selection_trace,
        'dopt_spec': dopt_spec,
    }


# ============================================================
# Save Artifacts
# ============================================================

def save_gate4d_dopt_run(
    run_dir: Path,
    run_id: str,
    cfg: Gate4dCPDoptConfig,
    eval_result: Dict[str, Any],
    metrics: Dict[str, Any],
    pool_sha: str,
    traj_sha: str,
    baseline_dir: Path,
    dopt_spec: Dict[str, Any],
    selection_trace: List,
    sensitivity_results: Optional[List] = None,
    sign_stable: Optional[bool] = None,
):
    """Save Gate4d D-opt artifacts (SSOT-compliant)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'figures').mkdir(exist_ok=True)

    feat_names = eval_result['feature_names']
    coeff = eval_result['coefficients_mean']

    # --- metrics.json ---
    full_metrics = {**metrics}
    full_metrics.update({
        'system': 'cartpole',
        'gate': 'gate4d',
        'method': 'dopt_reparam1',
        'library_version': 'Reparam-1',
        'reparam': cfg.reparam,
        'baseline_seed': cfg.baseline_seed,
        'pool_sha': pool_sha,
        'pool_sha_definition': 'sha256(trajectories_bytes)[:16]',
        'traj_sha': traj_sha,
        'n_select': cfg.n_select,
        'n_train': cfg.n_train,
        'n_bootstrap': cfg.n_bootstrap,
        'threshold': cfg.threshold,
        'kappa_augmented': eval_result['kappa'],
        'n_total_samples': eval_result['n_total_samples'],
        'n_original': eval_result['n_train'],
        'n_augmented': eval_result['n_aug'],
        'support_terms_total': int(eval_result['support_mask'].sum()),
        'dopt_lambda': cfg.dopt_lambda,
        'dopt_gram_energy_mode': cfg.dopt_gram_energy_mode,
        'dopt_use_teacher_intersection': cfg.dopt_use_teacher_intersection,
        'dopt_spec_hash': dopt_spec.get('spec_hash', ''),
        'metric_convention': 'CP: score_aligned = +delta_raw',
        'gate4a_reference_dopt_median': GATE4A_STD_DOPT_MEDIAN,
        'runner_version': RUNNER_VERSION,
    })
    if sensitivity_results is not None:
        all_sa = ([metrics['score_aligned_median']]
                  + [r['score_aligned_median'] for r in sensitivity_results])
        full_metrics['eval_seed_sensitivity'] = {
            'score_aligned_values': all_sa,
            'sign_stable': sign_stable,
        }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2, default=_json_default)

    # --- sindy_coefficients.csv ---
    with open(run_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + list(CP_TARGET_NAMES))
        for i, name in enumerate(feat_names):
            w.writerow([name] + [f"{coeff[i,j]:.8f}" for j in range(4)])

    # --- z_after.npy ---
    np.save(run_dir / 'z_after.npy', eval_result['z_full'])

    # --- dopt_spec.json ---
    with open(run_dir / 'dopt_spec.json', 'w') as f:
        json.dump(dopt_spec, f, indent=2, default=_json_default)

    # --- dopt_selection_trace.json ---
    with open(run_dir / 'dopt_selection_trace.json', 'w') as f:
        json.dump(selection_trace, f, indent=2, default=_json_default)

    # --- manifest.json ---
    manifest = {
        'run_id': run_id,
        'system': 'cartpole',
        'gate': 'gate4d',
        'method': 'dopt_reparam1',
        'library_id': 'RP1',
        'library_version': 'Reparam-1',
        'reparam': cfg.reparam,
        'baseline_seed': cfg.baseline_seed,
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_gate4d_cp_reparam_dopt.py',
        'runner_version': RUNNER_VERSION,
        'pool_sha': pool_sha,
        'traj_sha': traj_sha,
        'pool_sha_definition': 'sha256(trajectories_bytes)[:16]',
        'baseline_dir': str(baseline_dir),
        'config': {
            'pool_size': cfg.pool_size,
            'pool_seed': cfg.pool_seed,
            'n_select': cfg.n_select,
            'n_bootstrap': cfg.n_bootstrap,
            'threshold': cfg.threshold,
            'dopt_lambda': cfg.dopt_lambda,
            'dopt_gram_energy_mode': cfg.dopt_gram_energy_mode,
            'dopt_use_teacher_intersection': cfg.dopt_use_teacher_intersection,
            'dopt_dynamics_targets': cfg.dopt_dynamics_targets,
        },
        'artifacts': [
            'manifest.json', 'metrics.json', 'sindy_coefficients.csv',
            'z_after.npy', 'dopt_spec.json', 'dopt_selection_trace.json',
        ],
    }
    with open(run_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=_json_default)


# ============================================================
# Main Runner
# ============================================================

def run_gate4d_cp_dopt(
    baseline_dir: str,
    cfg: Optional[Gate4dCPDoptConfig] = None,
) -> Dict[str, Any]:
    """Gate4d: CP Reparam-1 D-optimal augmentation."""

    if cfg is None:
        cfg = Gate4dCPDoptConfig()

    print("=" * 70)
    print("Gate4d: Cart-Pole Reparam-1 D-optimal Augmentation")
    print(f"  Library: Reparam-1 (21-term, cos(theta)→cos(theta)-1)")
    print(f"  Pool: {cfg.pool_size}, n_select: {cfg.n_select}")
    print(f"  seed: {cfg.baseline_seed}, pool_seed: {cfg.pool_seed}")
    print(f"  use_teacher_intersection: {cfg.dopt_use_teacher_intersection}")
    print(f"  Metric: score_aligned = +delta_raw (CP: recall fragility)")
    print("=" * 70)

    # ── AC2 ──
    print("\n[AC2] CP library integrity check...")
    assert_cp_feature_integrity('reparam1')

    # ── Phase 0: Load dataset ──
    print("\n[Phase 0] Loading CP dataset...")
    dataset_path = paths.get_dataset_path(cfg.dataset_version, system=SYSTEM)
    validate_dataset_lite(dataset_path)
    dataset = dict(np.load(dataset_path, allow_pickle=True))
    train_x  = dataset['train_x'][:cfg.n_train]
    train_u  = dataset['train_u'][:cfg.n_train]
    train_dx = dataset['train_dx'][:cfg.n_train]
    print(f"  Train shape: {train_x.shape}")

    # ── Phase 1: Load baseline ──
    print("\n[Phase 1] Loading CP Reparam-1 baseline...")
    baseline = load_cp_baseline(Path(baseline_dir))

    if baseline['reparam'] != 'reparam1':
        raise ValueError(
            f"Baseline is '{baseline['reparam']}', expected 'reparam1'. "
            "Run run_gate4d_cp_reparam_baseline.py first."
        )

    fragile_pairs = baseline['fragile_pairs']
    z_before = baseline['z_before']
    teacher_support = baseline['teacher_support']
    teacher_coefficients = baseline['teacher_coefficients']

    # ── Phase 2: GMM + Pool ──
    print(f"\n[Phase 2] GMM fitting + pool generation "
          f"(pool_size={cfg.pool_size}, seed={cfg.pool_seed})...")

    train_params = dataset['train_params'][:cfg.n_train]
    rng_streams = create_rng_streams(cfg.baseline_seed, pool_seed=cfg.pool_seed)

    gmm_sampler = GMMProposalSampler(
        n_components=GATE3_CONFIG['gmm_n_components'],
        covariance_type=GATE3_CONFIG['gmm_covariance_type'],
        random_state=cfg.pool_seed,
    )
    gmm_sampler.fit(train_x, train_params)

    pool_generator = PoolGenerator(
        gmm_sampler=gmm_sampler,
        train_u=train_u,
        config=GATE3_CONFIG,
        fixed_physics=GATE3_CONFIG['fixed_physics'],
        seed=cfg.baseline_seed,
        rng=rng_streams['pool'],
    )
    pool = pool_generator.generate_pool(
        target_n_accept=cfg.pool_size,
        max_attempts=GATE3_CONFIG.get('max_pool_attempts', cfg.pool_size * 5),
    )
    print(f"  Pool size: {len(pool['trajectories'])}")

    # Pool SHA (trajectories bytes)
    pool_sha = hashlib.sha256(
        pool['trajectories'].tobytes()
    ).hexdigest()[:16]
    traj_sha = pool_sha   # same definition
    print(f"  Pool SHA: {pool_sha}")
    print(f"  (SHA definition: sha256(trajectories_bytes)[:16])")

    # ── Phase 3: Track A ──
    print(f"\n[Phase 3] Track A filter (reject top {cfg.reject_ratio*100:.0f}%)...")

    # Compute norm_stats from training data
    x_flat_tr = train_x.reshape(-1, 4)
    dx_flat_tr = train_dx.reshape(-1, 4)
    u_flat_tr  = train_u.reshape(-1, 1)
    norm_stats = {
        'state': {
            'mean': x_flat_tr.mean(axis=0).tolist(),
            'std': x_flat_tr.std(axis=0).tolist(),
        },
        'derivative_dx_savgol': {
            'mean': dx_flat_tr.mean(axis=0).tolist(),
            'std': dx_flat_tr.std(axis=0).tolist(),
        },
        'input': {
            'mean': float(u_flat_tr.mean()),
            'std': float(u_flat_tr.std()),
        },
    }
    dx_std = np.array(norm_stats['derivative_dx_savgol']['std'])

    feature_names_rp1 = get_cp_feature_names('reparam1')

    track_a = track_a_selection(
        pool=pool,
        teacher_coefficients=teacher_coefficients,
        feature_names=feature_names_rp1,
        target_names=list(DEFAULT_TARGET_NAMES),
        dx_std=dx_std,
        norm_stats=norm_stats,
        reject_ratio=cfg.reject_ratio,
        n_select=cfg.n_select,
        dynamics_target_indices=CP_DYNAMICS_TARGETS,
    )
    n_track_a = len(track_a['selected_indices'])
    print(f"  Track A passed: {n_track_a} / {len(pool['trajectories'])}")

    # ── Phase 4: D-optimal selection ──
    print(f"\n[Phase 4] D-optimal selection...")
    selected = dopt_select_cp(
        pool=pool,
        track_a=track_a,
        fragile_pairs=fragile_pairs,
        teacher_support=teacher_support,
        reparam=cfg.reparam,
        train_x=train_x,
        train_u=train_u,
        cfg=cfg,
    )
    n_selected = len(selected['pool_indices'])
    print(f"  Final selection: {n_selected} trajectories")

    # ── Phase 5: E-SINDy evaluation ──
    print(f"\n[Phase 5] E-SINDy evaluation (Reparam-1, n_bootstrap={cfg.n_bootstrap})...")
    eval_result = evaluate_augmented_cp(
        train_x, train_u, train_dx,
        selected['trajectories'],
        selected['u'],
        selected['dx'],
        reparam=cfg.reparam,
        n_bootstrap=cfg.n_bootstrap,
        threshold=cfg.threshold,
        seed=cfg.baseline_seed,
        z_eps=cfg.z_eps,
    )
    print(f"  κ (augmented): {eval_result['kappa']:.3e}")
    print(f"  Support active: {int(eval_result['support_mask'].sum())}/84")

    # ── Phase 6: Metrics ──
    metrics = compute_metrics_cp(
        eval_result['z_full'],
        z_before,
        fragile_pairs,
        ci_B=cfg.ci_bootstrap_B,
        ci_alpha=cfg.ci_alpha,
        seed=cfg.baseline_seed,
    )
    sa = metrics['score_aligned_median']
    print(f"\n  delta_raw:     {metrics['delta_raw_median']:.3f}")
    print(f"  score_aligned: {sa:.3f}  (positive = improvement)")
    print(f"  CI:            [{metrics['score_aligned_ci_lower']:.3f}, "
          f"{metrics['score_aligned_ci_upper']:.3f}]")
    print(f"  pass_level:    {metrics['pass_level']}")
    print(f"\n  [Gate4a Standard D-opt reference: median={GATE4A_STD_DOPT_MEDIAN}, "
          f"ci_lower={GATE4A_STD_DOPT_CI_LOWER} → STRONG_PASS]")

    # ── Phase 6b: Eval seed sensitivity ──
    print(f"\n[Phase 6b] Eval seed sensitivity...")
    sensitivity_results = []
    for es in [1, 2]:
        ev_es = evaluate_augmented_cp(
            train_x, train_u, train_dx,
            selected['trajectories'],
            selected['u'],
            selected['dx'],
            reparam=cfg.reparam,
            n_bootstrap=cfg.n_bootstrap,
            threshold=cfg.threshold,
            seed=es,
            z_eps=cfg.z_eps,
        )
        met_es = compute_metrics_cp(
            ev_es['z_full'], z_before, fragile_pairs,
            ci_B=cfg.ci_bootstrap_B, ci_alpha=cfg.ci_alpha, seed=es,
        )
        sensitivity_results.append({
            'eval_seed': es,
            'score_aligned_median': met_es['score_aligned_median'],
            'pass_level': met_es['pass_level'],
            'kappa': ev_es['kappa'],
        })
        print(f"  eval_seed={es}: score_aligned={met_es['score_aligned_median']:.3f}, "
              f"pass={met_es['pass_level']}")

    all_sa = [sa] + [r['score_aligned_median'] for r in sensitivity_results]
    sign_stable = all(s > 0 for s in all_sa) or all(s < 0 for s in all_sa)
    sign_label = ('STABLE (all +)' if all(s > 0 for s in all_sa)
                  else 'STABLE (all -)' if all(s < 0 for s in all_sa)
                  else 'UNSTABLE (mixed)')
    print(f"  score_aligned: {[f'{s:.3f}' for s in all_sa]}")
    print(f"  Sign stable: {sign_label}")

    # ── Phase 7: Save artifacts ──
    print(f"\n[Phase 7] Saving artifacts...")
    run_id = paths.generate_run_id(f"gate4d_cp_dopt_s{cfg.baseline_seed}")
    run_dir = paths.get_results_dir(
        dataset_version=cfg.dataset_version,
        gate='gate4d',
        track='standardized',
        method='dopt_reparam1',
        n_train=cfg.n_train,
        seed=cfg.baseline_seed,
        run_id=run_id,
    )
    save_gate4d_dopt_run(
        run_dir=run_dir,
        run_id=run_id,
        cfg=cfg,
        eval_result=eval_result,
        metrics=metrics,
        pool_sha=pool_sha,
        traj_sha=traj_sha,
        baseline_dir=Path(baseline_dir),
        dopt_spec=selected['dopt_spec'],
        selection_trace=selected['selection_trace'],
        sensitivity_results=sensitivity_results,
        sign_stable=sign_stable,
    )
    print(f"  Artifacts saved: {run_dir}")

    # ── Context Packet ──
    cp_path = paths.get_context_packet_path(run_id)
    cp_content = (
        f"# Context Packet: {run_id}\n\n"
        f"**System**: Cart-Pole | **Gate**: 4d | **Method**: D-opt Reparam-1\n"
        f"**Library**: Reparam-1 (cos(theta)→cos(theta)-1, 21-term)\n"
        f"**Baseline seed**: {cfg.baseline_seed}\n"
        f"**Created**: {datetime.now().isoformat()}\n\n"
        f"## Results\n\n"
        f"- delta_raw: {metrics['delta_raw_median']:.3f}\n"
        f"- score_aligned: {sa:.3f}  (positive = improvement, +delta_raw)\n"
        f"- CI(score_aligned): [{metrics['score_aligned_ci_lower']:.3f}, "
        f"{metrics['score_aligned_ci_upper']:.3f}]\n"
        f"- pass_level: {metrics['pass_level']}\n"
        f"- κ (augmented): {eval_result['kappa']:.3e}\n"
        f"- Pool SHA: {pool_sha}\n\n"
        f"## Eval Seed Sensitivity\n\n"
        f"- score_aligned: {[f'{s:.3f}' for s in all_sa]}\n"
        f"- Sign stable: {sign_stable} ({sign_label})\n\n"
        f"## Comparison\n\n"
        f"| | Standard D-opt (Gate4a) | Reparam-1 D-opt (Gate4d) |\n"
        f"|---|---|---|\n"
        f"| Library | Standard | Reparam-1 |\n"
        f"| median | {GATE4A_STD_DOPT_MEDIAN:.3f} | {sa:.3f} |\n"
        f"| pass_level | STRONG_PASS | {metrics['pass_level']} |\n\n"
        f"## Metric Convention\n\n"
        f"CP: score_aligned = +delta_raw (recall fragility — AEK와 반대)\n\n"
        f"## Artifacts\n\n"
        f"- Run dir: {run_dir}\n"
        f"- Baseline dir: {baseline_dir}\n"
    )
    with open(cp_path, 'w', encoding='utf-8') as f:
        f.write(cp_content)
    print(f"  Context Packet: {cp_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  GATE4d CP REPARAM-1 D-OPT SUMMARY")
    print("=" * 70)
    print(f"\n  Library:        Reparam-1 (cos(theta)→cos(theta)-1, 21-term)")
    print(f"  score_aligned:  {sa:.3f}  [{metrics['pass_level']}]")
    print(f"  CI:             [{metrics['score_aligned_ci_lower']:.3f}, "
          f"{metrics['score_aligned_ci_upper']:.3f}]")
    print(f"  κ (aug):        {eval_result['kappa']:.3e}")
    print(f"\n  [Standard D-opt reference: median=0.424, STRONG_PASS]")
    print(f"\n  Sign stable: {sign_label}")
    print(f"  score_aligned all seeds: {[f'{s:.3f}' for s in all_sa]}")
    print(f"\n  Run dir: {run_dir}")
    print("=" * 70)

    return {
        'status': 'completed',
        'run_id': run_id,
        'run_dir': str(run_dir),
        'score_aligned': sa,
        'pass_level': metrics['pass_level'],
        'ci_lower': metrics['score_aligned_ci_lower'],
        'ci_upper': metrics['score_aligned_ci_upper'],
        'kappa_augmented': eval_result['kappa'],
        'pool_sha': pool_sha,
        'sign_stable': sign_stable,
        'eval_seed_sensitivity': all_sa,
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Gate4d: CP Reparam-1 D-optimal augmentation'
    )
    p.add_argument(
        '--baseline_dir', type=str, required=True,
        help='Path to Gate4d CP Reparam-1 baseline run dir '
             '(from run_gate4d_cp_reparam_baseline.py)',
    )
    p.add_argument('--n_train', type=int, default=10)
    p.add_argument('--n_bootstrap', type=int, default=100)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--pool_size', type=int, default=2000)
    p.add_argument('--n_select', type=int, default=200)
    p.add_argument('--baseline_seed', type=int, default=42)
    p.add_argument('--pool_seed', type=int, default=42)
    p.add_argument(
        '--no_teacher_intersection', action='store_true',
        help='Disable teacher intersection (default: True for CP recall fragility)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Gate4dCPDoptConfig(
        n_train=args.n_train,
        n_bootstrap=args.n_bootstrap,
        threshold=args.threshold,
        pool_size=args.pool_size,
        n_select=args.n_select,
        baseline_seed=args.baseline_seed,
        pool_seed=args.pool_seed,
        dopt_use_teacher_intersection=not args.no_teacher_intersection,
    )
    try:
        result = run_gate4d_cp_dopt(args.baseline_dir, cfg)
        print(f"\nGate4d D-opt complete: "
              f"score_aligned={result['score_aligned']:.3f}, "
              f"pass={result['pass_level']}")
    except Exception as e:
        print(f"\nGate4d D-opt FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()