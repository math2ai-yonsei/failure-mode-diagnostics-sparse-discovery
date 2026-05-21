"""
AEK-4c: D-optimal Selection Augmentation Runner (Reparam-1)

Purpose:
    Test whether D-optimal selection (FIM-based) improves E-SINDy
    precision on AEK with Reparam-1 library, compared to Random baseline.

Design:
    1. Load AEK baseline artifacts (z_before, teacher_support, fragile_pairs)
    2. Fit 3-component GMM on training ICs+params (5D)
    3. Generate pool via AEK simulator (analytic dx)
    4. Track A: reject top-10% teacher alignment error
    5. D-optimal selection: greedy logdet on scaled fragile features
    6. E-SINDy evaluation on train+aug data
    7. Compute delta_raw + score_aligned (AC1 compliant)
    8. Assert feature name integrity (AC2 compliant)

D-optimal specifics:
    - ColumnScaler fitted on training Theta (approximate regression scaling)
    - Gram contributions per trajectory, per target (F_by_target)
    - use_teacher_intersection=False (all 20 fragile pairs are spurious;
      teacher intersection would give empty F_t)
    - gram_energy_mode='unit_trace' (energy-neutral across trajectories)
    - Deterministic: single run, no seed loop

Metric SSOT (AEK — spurious-primary):
    delta_raw = median(z_after − z_before) over fragile pairs
    score_aligned = −delta_raw  (positive = improvement)
    Both stored in metrics.json (AC1).

Usage:
    python experiments/run_aek4c_dopt.py
    python experiments/run_aek4c_dopt.py --baseline_seed 0

Author: Claude (Gate4c)
Date: 2026-03-03
Runner version: v1.1 (GPT review: SSOT path, CP gen, eval_seed sensitivity)
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
from src.sindy.aek_library import (
    build_aek_library_by_name,
    get_aek_feature_names,
    get_aek_oracle_support,
    AEK_TARGET_NAMES,
    N_AEK_FEATURES,
)
from src.simulators.aek_simulator import AEKSimulator

# Import shared functions from random runner
from experiments.run_aek4c_random import (
    AEK4cConfig,
    AEKGMMSampler,
    generate_pool,
    track_a_filter,
    evaluate_augmented,
    compute_metrics,
    compute_tau_stats,
    load_baseline,
    assert_oracle_label_integrity,
    _json_default,
)


# ============================================================
# Constants
# ============================================================

RUNNER_VERSION = 'v1.1'


# ============================================================
# Configuration
# ============================================================

@dataclass
class AEK4cDoptConfig(AEK4cConfig):
    """AEK-4c D-optimal experiment configuration (extends Random config)."""
    # D-optimal parameters
    dopt_lambda: float = 1e-6
    dopt_gram_energy_mode: str = 'unit_trace'
    dopt_trace_power: float = 1.0
    # AEK: all fragile pairs are spurious → no teacher intersection
    dopt_use_teacher_intersection: bool = False
    # Dynamics targets with fragile pairs
    dopt_dynamics_targets: List[int] = field(default_factory=lambda: [1, 3])

    # Override note for output naming
    note: str = 'aek4c_dopt'


# ============================================================
# D-optimal Selection Functions
# ============================================================

def compute_fragile_feature_sets(
    fragile_pairs: List[List[int]],
    dynamics_target_indices: List[int] = [1, 3],
) -> Dict[int, np.ndarray]:
    """
    Compute fragile feature sets per target (F_t).

    AEK-specific: No teacher intersection (all fragile = spurious,
    teacher intersection would give empty sets).

    Args:
        fragile_pairs: [[feature_idx, target_idx], ...]
        dynamics_target_indices: [1, 3] for phi_ddot, theta_w_ddot

    Returns:
        F_by_target: {target_idx: np.array of feature indices}
    """
    F_by_target = {}
    for t in dynamics_target_indices:
        fragile_t = set(f for f, target in fragile_pairs if target == t)
        F_by_target[t] = np.array(sorted(fragile_t), dtype=int)
    return F_by_target


def compute_pool_theta_per_traj(
    pool: Dict[str, Any],
    candidate_indices: np.ndarray,
    reparam: str,
    train_x: np.ndarray,
    train_u: np.ndarray,
) -> Tuple[np.ndarray, ColumnScaler]:
    """
    Compute scaled library features per candidate trajectory.

    Steps:
        1. Fit ColumnScaler on training data Theta
        2. For each candidate, compute Theta and scale

    Why train-only scaler:
        E-SINDy fits scaler on combined (train+aug), but we don't
        know the selection yet. Train-only scaler approximates
        the scaling landscape for D-optimal candidate ranking.

    Args:
        pool: Pool dict with 'trajectories' and 'u'
        candidate_indices: Track A passed indices into pool
        reparam: 'reparam1'
        train_x: (N_tr, T, 4)
        train_u: (N_tr, T, 1)

    Returns:
        Theta_scaled: (N_cand, T, 14) scaled features per trajectory
        scaler: fitted ColumnScaler (for diagnostics)
    """
    N_tr, T, D = train_x.shape

    # Fit scaler on training data
    x_flat = train_x.reshape(-1, D)
    u_flat = train_u.reshape(-1, 1)
    Theta_train, _ = build_aek_library_by_name(x_flat, u_flat, reparam=reparam)
    scaler = ColumnScaler()
    scaler.fit(Theta_train)
    print(f"  ColumnScaler fitted on training data ({N_tr} traj × {T} steps)")
    print(f"  Scale factors: min={scaler.scale_.min():.6f}, "
          f"max={scaler.scale_.max():.6f}")
    print(f"  Constant columns: {scaler.constant_mask_.sum()}")

    # Compute and scale per candidate trajectory
    pool_x = pool['trajectories'][candidate_indices]  # (N_cand, T, 4)
    pool_u = pool['u'][candidate_indices]              # (N_cand, T, 1)
    N_cand = len(candidate_indices)

    Theta_list = []
    for i in range(N_cand):
        Theta_i, _ = build_aek_library_by_name(
            pool_x[i], pool_u[i], reparam=reparam,
        )
        Theta_i_scaled = scaler.transform(Theta_i)  # (T, 14)
        Theta_list.append(Theta_i_scaled)

    Theta_scaled = np.array(Theta_list)  # (N_cand, T, 14)
    print(f"  Theta_scaled: {Theta_scaled.shape}")
    return Theta_scaled, scaler


def compute_gram_contributions(
    Theta: np.ndarray,
    F_by_target: Dict[int, np.ndarray],
    gram_energy_mode: str = 'unit_trace',
    trace_power: float = 1.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute Gram matrix contributions G_i per candidate per target.

    G_i = Theta_F[i]^T @ Theta_F[i]  (summed over time)

    Energy normalization (gram_energy_mode):
        'raw': no normalization
        'unit_trace': G_i /= trace(G_i) + eps
        'trace_power': G_i /= trace(G_i)^p + eps

    Args:
        Theta: (N_cand, T, n_features) scaled library features
        F_by_target: {target_idx: feature indices}
        gram_energy_mode: energy normalization mode
        trace_power: power p for trace_power mode

    Returns:
        G_by_target: {target_idx: (N_cand, |F_t|, |F_t|)}
        trace_by_target: {target_idx: (N_cand,)} trace values
    """
    N_cand, T, n_features = Theta.shape
    G_by_target = {}
    trace_by_target = {}
    eps = 1e-8

    for t, F_t in F_by_target.items():
        n_F = len(F_t)
        if n_F == 0:
            G_by_target[t] = np.zeros((N_cand, 1, 1))
            trace_by_target[t] = np.zeros(N_cand)
            continue

        # Extract fragile features: (N_cand, T, |F_t|)
        Theta_F = Theta[:, :, F_t]

        G = np.zeros((N_cand, n_F, n_F))
        traces = np.zeros(N_cand)

        for i in range(N_cand):
            G[i] = Theta_F[i].T @ Theta_F[i]  # (n_F, n_F)
            traces[i] = np.trace(G[i])

            if gram_energy_mode == 'unit_trace':
                G[i] = G[i] / (traces[i] + eps)
            elif gram_energy_mode == 'trace_power':
                G[i] = G[i] / (traces[i] ** trace_power + eps)

        G_by_target[t] = G
        trace_by_target[t] = traces

    return G_by_target, trace_by_target


def greedy_dopt_selection(
    G_by_target: Dict[int, np.ndarray],
    candidate_pool_indices: np.ndarray,
    n_select: int,
    lambda_reg: float = 1e-6,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Greedy D-optimal selection maximizing sum of logdet over targets.

    Δ_i = Σ_t [logdet(G_t + G_{i,t} + λI) - logdet(G_t + λI)]

    Tie-break: smallest pool_idx (deterministic).

    Args:
        G_by_target: {target_idx: (N_cand, |F_t|, |F_t|)}
        candidate_pool_indices: (N_cand,) pool indices
        n_select: number to select
        lambda_reg: regularization for logdet

    Returns:
        selected_pool_indices: (n_select,) pool indices
        selection_trace: list of step-level diagnostics
    """
    N_cand = len(candidate_pool_indices)

    if N_cand <= n_select:
        return candidate_pool_indices.copy(), [
            {'note': 'all_candidates_selected', 'n_cand': N_cand}
        ]

    # Initialize cumulative Gram (λI per target)
    G_cumulative = {}
    for t, G_t in G_by_target.items():
        n_F = G_t.shape[1]
        G_cumulative[t] = lambda_reg * np.eye(n_F)

    def safe_logdet(M):
        try:
            sign, logdet = np.linalg.slogdet(M)
            if sign <= 0:
                return -np.inf
            return logdet
        except Exception:
            return -np.inf

    def current_total_logdet():
        return sum(safe_logdet(G_cumulative[t]) for t in G_cumulative)

    selection_trace = []
    selected_local = []
    remaining_mask = np.ones(N_cand, dtype=bool)

    for step in range(n_select):
        if not remaining_mask.any():
            break

        remaining_local = np.where(remaining_mask)[0]
        current_logdet = current_total_logdet()

        # Compute delta for each remaining candidate
        deltas = np.full(len(remaining_local), -np.inf)

        for idx, local_i in enumerate(remaining_local):
            new_logdet = 0.0
            for t, G_t in G_cumulative.items():
                G_new = G_t + G_by_target[t][local_i]
                new_logdet += safe_logdet(G_new)
            deltas[idx] = new_logdet - current_logdet

        # Best candidate
        best_idx = np.argmax(deltas)
        best_local_idx = remaining_local[best_idx]
        best_delta = deltas[best_idx]

        # Tie-break: smallest pool_idx
        tie_mask = np.abs(deltas - best_delta) < 1e-12
        if tie_mask.sum() > 1:
            tie_local = remaining_local[tie_mask]
            tie_pool_idx = candidate_pool_indices[tie_local]
            best_tie = np.argmin(tie_pool_idx)
            best_local_idx = tie_local[best_tie]
            best_delta = deltas[tie_mask][best_tie]

        # Update cumulative Gram
        for t in G_cumulative:
            G_cumulative[t] = G_cumulative[t] + G_by_target[t][best_local_idx]

        selected_local.append(best_local_idx)
        remaining_mask[best_local_idx] = False

        new_total = current_total_logdet()

        selection_trace.append({
            'step': step,
            'local_idx': int(best_local_idx),
            'pool_idx': int(candidate_pool_indices[best_local_idx]),
            'delta_logdet': float(best_delta),
            'cumulative_logdet': float(new_total),
            'n_remaining': int(remaining_mask.sum()),
        })

        if step % 10 == 0 or step == n_select - 1:
            print(f"    D-opt step {step}: Δlogdet={best_delta:.4f}, "
                  f"cumulative={new_total:.4f}")

    selected_pool_indices = candidate_pool_indices[np.array(selected_local)]
    return selected_pool_indices, selection_trace


def dopt_select(
    pool: Dict[str, Any],
    track_a: Dict[str, Any],
    fragile_pairs: List[List[int]],
    reparam: str,
    train_x: np.ndarray,
    train_u: np.ndarray,
    n_select: int,
    cfg: AEK4cDoptConfig,
) -> Dict[str, Any]:
    """
    D-optimal selection pipeline (AEK-specific).

    Steps:
        1. Compute F_by_target from fragile pairs
        2. Compute scaled Theta per Track A trajectory
        3. Compute Gram contributions per target
        4. Greedy D-optimal selection
        5. Return selected trajectories + diagnostics

    Args:
        pool: Pool dict
        track_a: Track A filter result
        fragile_pairs: [[f_idx, t_idx], ...]
        reparam: 'reparam1'
        train_x, train_u: training data for scaler fitting
        n_select: number to select
        cfg: D-optimal config

    Returns:
        Dict with selected trajectories, diagnostics, metadata
    """
    print(f"\n[D-optimal Selection] n_select={n_select}")
    print(f"  lambda={cfg.dopt_lambda}, "
          f"gram_mode={cfg.dopt_gram_energy_mode}, "
          f"trace_power={cfg.dopt_trace_power}")

    candidates = track_a['selected_indices']
    N_cand = len(candidates)
    print(f"  Track A candidates: {N_cand}")

    # Step 1: Fragile feature sets
    F_by_target = compute_fragile_feature_sets(
        fragile_pairs, cfg.dopt_dynamics_targets,
    )
    target_label = {1: 'd(phi_dot)/dt', 3: 'd(theta_w_dot)/dt'}
    for t, F_t in F_by_target.items():
        print(f"  F_{target_label.get(t, t)}: {len(F_t)} features — {F_t.tolist()}")

    # Step 2: Scaled library features per candidate
    print(f"\n  Computing scaled library features...")
    Theta_scaled, train_scaler = compute_pool_theta_per_traj(
        pool, candidates, reparam, train_x, train_u,
    )

    # Step 3: Gram contributions
    print(f"\n  Computing Gram contributions "
          f"(mode={cfg.dopt_gram_energy_mode})...")
    G_by_target, trace_by_target = compute_gram_contributions(
        Theta_scaled, F_by_target,
        gram_energy_mode=cfg.dopt_gram_energy_mode,
        trace_power=cfg.dopt_trace_power,
    )
    for t, G_t in G_by_target.items():
        traces = trace_by_target[t]
        print(f"  G_{target_label.get(t, t)}: shape={G_t.shape}, "
              f"trace range=[{traces.min():.4f}, {traces.max():.4f}]")

    # Step 4: Greedy selection
    print(f"\n  Running greedy D-opt selection...")
    selected_pool_indices, selection_trace = greedy_dopt_selection(
        G_by_target=G_by_target,
        candidate_pool_indices=candidates,
        n_select=n_select,
        lambda_reg=cfg.dopt_lambda,
    )

    selected_sorted = np.sort(selected_pool_indices)
    n_selected = len(selected_pool_indices)
    print(f"\n  Selected: {n_selected} trajectories")

    # Build D-optimal spec (for reproducibility)
    dopt_spec = {
        'version': RUNNER_VERSION,
        'method': 'd_optimal',
        'objective': 'sum_logdet_by_target',
        'dynamics_target_indices': cfg.dopt_dynamics_targets,
        'use_teacher_intersection': cfg.dopt_use_teacher_intersection,
        'lambda_reg': cfg.dopt_lambda,
        'gram_energy_mode': cfg.dopt_gram_energy_mode,
        'trace_power': cfg.dopt_trace_power,
        'tie_break': 'pool_idx_asc',
        'F_by_target': {str(t): F_t.tolist() for t, F_t in F_by_target.items()},
        'n_candidates': N_cand,
        'n_selected': n_selected,
        'scaler_scales': train_scaler.scale_.tolist(),
        'scaler_constant_mask': train_scaler.constant_mask_.tolist(),
    }
    spec_str = json.dumps(dopt_spec, sort_keys=True)
    dopt_spec['spec_hash'] = hashlib.sha256(spec_str.encode()).hexdigest()[:16]
    dopt_spec['selected_pool_indices'] = selected_sorted.tolist()

    return {
        'indices': selected_sorted,
        'trajectories': pool['trajectories'][selected_sorted],
        'dx': pool['dx'][selected_sorted],
        'u': pool['u'][selected_sorted],
        'n_selected': n_selected,
        'dopt_spec': dopt_spec,
        'selection_trace': selection_trace,
        'train_scaler': train_scaler,
    }


# ============================================================
# Save D-optimal Run Artifacts
# ============================================================

def save_dopt_run(
    run_dir: Path,
    run_id: str,
    cfg: AEK4cDoptConfig,
    eval_result: Dict[str, Any],
    metrics: Dict[str, Any],
    pool_sha: str,
    baseline_dir: Path,
    dopt_spec: Dict[str, Any],
    selection_trace: List[Dict],
    tau_stats: Optional[Dict[str, Any]] = None,
    sensitivity_results: Optional[List[Dict]] = None,
    sign_stable: Optional[bool] = None,
):
    """Save all D-optimal run artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'figures').mkdir(exist_ok=True)

    feat_names = eval_result['feature_names']

    # --- metrics.json (AC1: delta_raw + score_aligned) ---
    full_metrics = {**metrics}
    full_metrics.update({
        'system': 'aek',
        'gate': 'gate4c',
        'method': 'd_optimal',
        'reparam': cfg.reparam,
        'selection_seed': None,  # deterministic
        'pool_size': cfg.pool_size,
        'pool_sha': pool_sha,
        'n_select': cfg.n_select,
        'n_train': cfg.n_train,
        'n_bootstrap': cfg.n_bootstrap,
        'threshold': cfg.threshold,
        'kappa_augmented': eval_result['kappa'],
        'n_total_samples': eval_result['n_total_samples'],
        'n_original': eval_result['n_original'],
        'n_augmented': eval_result['n_augmented'],
        'support_terms_total': int(eval_result['support_mask'].sum()),
        'ci_bootstrap_B': cfg.ci_bootstrap_B,
        'ci_alpha': cfg.ci_alpha,
        'runner_version': RUNNER_VERSION,
        # D-optimal specific
        'dopt_lambda': cfg.dopt_lambda,
        'dopt_gram_energy_mode': cfg.dopt_gram_energy_mode,
        'dopt_trace_power': cfg.dopt_trace_power,
        'dopt_use_teacher_intersection': cfg.dopt_use_teacher_intersection,
        'dopt_spec_hash': dopt_spec.get('spec_hash', ''),
    })
    if tau_stats is not None:
        full_metrics['tau_distribution'] = tau_stats
    if sensitivity_results is not None:
        full_metrics['eval_seed_sensitivity'] = {
            'seeds_tested': [0] + [r['eval_seed'] for r in sensitivity_results],
            'score_aligned_values': (
                [metrics['score_aligned_median']]
                + [r['score_aligned_median'] for r in sensitivity_results]
            ),
            'sign_stable': sign_stable,
            'results': sensitivity_results,
        }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2, default=_json_default)

    # --- sindy_coefficients.csv ---
    coeff = eval_result['coefficients_mean']
    with open(run_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + list(AEK_TARGET_NAMES))
        for i, name in enumerate(feat_names):
            w.writerow([name] + [f"{coeff[i,j]:.8f}" for j in range(4)])

    # --- z_after.npy ---
    np.save(run_dir / 'z_after.npy', eval_result['z'])

    # --- dopt_spec.json ---
    with open(run_dir / 'dopt_spec.json', 'w') as f:
        json.dump(dopt_spec, f, indent=2, default=_json_default)

    # --- dopt_selection_trace.json ---
    with open(run_dir / 'dopt_selection_trace.json', 'w') as f:
        json.dump(selection_trace, f, indent=2, default=_json_default)

    # --- manifest.json ---
    manifest = {
        'run_id': run_id,
        'system': 'aek',
        'gate': 'gate4c',
        'method': 'd_optimal',
        'reparam': cfg.reparam,
        'selection_seed': None,
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_aek4c_dopt.py',
        'runner_version': RUNNER_VERSION,
        'pool_sha': pool_sha,
        'baseline_dir': str(baseline_dir),
        'config': {
            'pool_size': cfg.pool_size,
            'pool_seed': cfg.pool_seed,
            'n_select': cfg.n_select,
            'n_bootstrap': cfg.n_bootstrap,
            'threshold': cfg.threshold,
            'gmm_n_components': cfg.gmm_n_components,
            'gmm_seed': cfg.gmm_seed,
            'reject_ratio': cfg.reject_ratio,
            'qc_max_phi': cfg.qc_max_phi,
            'qc_max_phi_dot': cfg.qc_max_phi_dot,
            'pd_gain_margin': cfg.pd_gain_margin,
            'pd_Kd_factor': cfg.pd_Kd_factor,
            'pd_noise_std': cfg.pd_noise_std,
            'I_w_C_clip_range': [5e-5, 1.5e-4],
            # D-optimal params
            'dopt_lambda': cfg.dopt_lambda,
            'dopt_gram_energy_mode': cfg.dopt_gram_energy_mode,
            'dopt_trace_power': cfg.dopt_trace_power,
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

class AEK4cDoptRunner:
    """AEK-4c D-optimal Selection Augmentation Runner."""

    def __init__(self, cfg: AEK4cDoptConfig):
        self.cfg = cfg
        self.feature_names = get_aek_feature_names(cfg.reparam)

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        print("=" * 70)
        print("AEK-4c: D-optimal Selection Augmentation (Reparam-1)")
        print(f"  Pool: {cfg.pool_size}, Select: {cfg.n_select}")
        print(f"  E-SINDy: B={cfg.n_bootstrap}, threshold={cfg.threshold}")
        print(f"  D-opt: lambda={cfg.dopt_lambda}, "
              f"gram={cfg.dopt_gram_energy_mode}")
        print("=" * 70)

        # AC2 check
        print("\n[AC2] Oracle label integrity...")
        assert_oracle_label_integrity(cfg.reparam)
        print("  PASS")

        # Phase 0: Load
        print("\n[Phase 0] Loading data and baseline...")
        dataset_path = paths.get_dataset_path(
            cfg.dataset_version, system=cfg.system,
        )
        validate_dataset_lite(dataset_path)
        dataset = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  Dataset: {dataset_path}")

        baseline = load_baseline(cfg)

        train_x = dataset['train_x'][:cfg.n_train]    # (10, 201, 4) §3.3
        train_u = dataset['train_u'][:cfg.n_train]           # (10, 201, 1)
        train_dx = dataset['train_dx_savgol'][:cfg.n_train]  # (10, 201, 4) §3.3
        train_params = dataset['train_params'][:cfg.n_train]
        print(f"  Train: {train_x.shape}")

        # Phase 1: GMM
        print("\n[Phase 1] Fitting GMM...")
        gmm = AEKGMMSampler(
            n_components=cfg.gmm_n_components,
            covariance_type=cfg.gmm_covariance_type,
            random_state=cfg.gmm_seed,
        )
        gmm.fit(train_x, train_params)
        print(f"  GMM fitted (5D, {cfg.gmm_n_components} components)")

        # Phase 2: Generate shared pool (same seed as Random)
        print("\n[Phase 2] Generating shared pool...")
        rng_pool = np.random.default_rng(cfg.pool_seed)
        pool = generate_pool(gmm, train_x, train_u, cfg, rng_pool)
        pool_sha = hashlib.sha256(
            pool['trajectories'].tobytes(),
        ).hexdigest()[:16]
        print(f"  Pool SHA: {pool_sha}")

        EXPECTED_SHA = '87594090343bee29'
        if pool_sha != EXPECTED_SHA:
            print(f"  ⚠️ SHA MISMATCH: expected {EXPECTED_SHA}, got {pool_sha}")
            print(f"  Pool may differ from Random baseline — proceed with caution")
        else:
            print(f"  ✅ SHA matches Random baseline")

        # Phase 3: Track A (shared filter)
        print("\n[Phase 3] Track A filtering...")
        track_a = track_a_filter(
            pool, baseline['coefficients'], cfg.reparam, cfg.reject_ratio,
        )

        # Phase 4: D-optimal selection
        print("\n[Phase 4] D-optimal selection...")
        selected = dopt_select(
            pool=pool,
            track_a=track_a,
            fragile_pairs=baseline['fragile_pairs'],
            reparam=cfg.reparam,
            train_x=train_x,
            train_u=train_u,
            n_select=cfg.n_select,
            cfg=cfg,
        )

        # Phase 5: E-SINDy evaluation
        print(f"\n[Phase 5] E-SINDy evaluation...")
        print(f"  {train_x.shape[0]} train + {selected['n_selected']} aug")
        eval_result = evaluate_augmented(
            train_x, train_u, train_dx,
            selected['trajectories'], selected['u'], selected['dx'],
            reparam=cfg.reparam,
            n_bootstrap=cfg.n_bootstrap,
            threshold=cfg.threshold,
            seed=cfg.baseline_seed,
            z_eps=cfg.z_eps,
        )

        # Phase 6: Metrics (AC1)
        print(f"\n[Phase 6] Computing metrics...")
        metrics = compute_metrics(
            eval_result['z'], baseline['z_before'],
            baseline['fragile_pairs'],
            cfg.ci_bootstrap_B, cfg.ci_alpha, cfg.baseline_seed,
        )

        # Tau distribution
        tau_stats = compute_tau_stats(train_u, selected['u'])

        # Phase 6b: eval_seed sensitivity (AC3 — sign stability check)
        print(f"\n[Phase 6b] Eval seed sensitivity check...")
        sensitivity_seeds = [1, 2]
        sensitivity_results = []
        for es in sensitivity_seeds:
            eval_es = evaluate_augmented(
                train_x, train_u, train_dx,
                selected['trajectories'], selected['u'], selected['dx'],
                reparam=cfg.reparam,
                n_bootstrap=cfg.n_bootstrap,
                threshold=cfg.threshold,
                seed=es,
                z_eps=cfg.z_eps,
            )
            met_es = compute_metrics(
                eval_es['z'], baseline['z_before'],
                baseline['fragile_pairs'],
                cfg.ci_bootstrap_B, cfg.ci_alpha, es,
            )
            sensitivity_results.append({
                'eval_seed': es,
                'delta_raw_median': met_es['delta_raw_median'],
                'score_aligned_median': met_es['score_aligned_median'],
                'pass_level': met_es['pass_level'],
                'kappa': eval_es['kappa'],
            })
            print(f"  eval_seed={es}: score_aligned="
                  f"{met_es['score_aligned_median']:.3f}, "
                  f"pass={met_es['pass_level']}")

        # Check sign stability
        all_sa = [metrics['score_aligned_median']] + [
            r['score_aligned_median'] for r in sensitivity_results
        ]
        sign_stable = all(s < 0 for s in all_sa)
        print(f"  Sign stability: {'STABLE (all negative)' if sign_stable else 'UNSTABLE'}")
        print(f"  score_aligned values: {[f'{s:.3f}' for s in all_sa]}")

        # Phase 7: Save
        print(f"\n[Phase 7] Saving artifacts...")
        run_id = paths.generate_run_id("aek4c_dopt")
        run_dir = paths.get_results_dir(
            dataset_version=cfg.dataset_version,
            gate='gate4c',
            track='standardized',
            method='d_optimal',
            n_train=cfg.n_train,
            seed=cfg.baseline_seed,
            run_id=run_id,
        )

        save_dopt_run(
            run_dir=run_dir,
            run_id=run_id,
            cfg=cfg,
            eval_result=eval_result,
            metrics=metrics,
            pool_sha=pool_sha,
            baseline_dir=baseline['dir'],
            dopt_spec=selected['dopt_spec'],
            selection_trace=selected['selection_trace'],
            tau_stats=tau_stats,
            sensitivity_results=sensitivity_results,
            sign_stable=sign_stable,
        )

        # Context Packet (SSOT: CP_{run_id}.md)
        cp_path = paths.get_context_packet_path(run_id)
        cp_content = (
            f"# Context Packet: {run_id}\n\n"
            f"**System**: AEK | **Gate**: 4c | **Method**: D-optimal\n"
            f"**Library**: Reparam-1 (14-term)\n"
            f"**Created**: {datetime.now().isoformat()}\n\n"
            f"## Results\n\n"
            f"- delta_raw: {metrics['delta_raw_median']:.3f}\n"
            f"- score_aligned: {metrics['score_aligned_median']:.3f}\n"
            f"- pass_level: {metrics['pass_level']}\n"
            f"- CI(score_aligned): [{metrics['score_aligned_ci_lower']:.3f}, "
            f"{metrics['score_aligned_ci_upper']:.3f}]\n"
            f"- kappa_augmented: {eval_result['kappa']:.0f}\n"
            f"- support: {int(eval_result['support_mask'].sum())}/56\n"
            f"- pool_sha: {pool_sha}\n\n"
            f"## Eval Seed Sensitivity (AC3)\n\n"
            f"- Seeds tested: {[0] + [r['eval_seed'] for r in sensitivity_results]}\n"
            f"- score_aligned: {[f'{s:.3f}' for s in all_sa]}\n"
            f"- Sign stable: {sign_stable}\n\n"
            f"## D-optimal Config\n\n"
            f"- lambda: {cfg.dopt_lambda}\n"
            f"- gram_energy_mode: {cfg.dopt_gram_energy_mode}\n"
            f"- use_teacher_intersection: {cfg.dopt_use_teacher_intersection}\n"
            f"- F_by_target: see dopt_spec.json\n\n"
            f"## Artifacts\n\n"
            f"- Run dir: {run_dir}\n"
            f"- Baseline: {baseline['dir']}\n"
        )
        with open(cp_path, 'w', encoding='utf-8') as f:
            f.write(cp_content)
        print(f"  Context Packet: {cp_path}")

        # Summary
        print("\n" + "=" * 70)
        print("  AEK-4c D-OPTIMAL SUMMARY")
        print("=" * 70)
        print(f"  delta_raw    = {metrics['delta_raw_median']:.3f}")
        print(f"  score_aligned= {metrics['score_aligned_median']:.3f}")
        print(f"  pass_level   = {metrics['pass_level']}")
        print(f"  CI(score_aln)= [{metrics['score_aligned_ci_lower']:.3f}, "
              f"{metrics['score_aligned_ci_upper']:.3f}]")
        print(f"  kappa_aug    = {eval_result['kappa']:.0f}")
        print(f"  support      = {int(eval_result['support_mask'].sum())}/56")
        print(f"  pool_sha     = {pool_sha}")
        print(f"  tau: train=[{tau_stats['train_tau_q05']:.5f}, "
              f"{tau_stats['train_tau_q95']:.5f}], "
              f"aug=[{tau_stats['aug_tau_q05']:.5f}, "
              f"{tau_stats['aug_tau_q95']:.5f}]")
        print(f"  Saved: {run_dir}")
        print(f"\n  [Eval Seed Sensitivity — AC3]")
        print(f"  score_aligned by eval_seed: {[f'{s:.3f}' for s in all_sa]}")
        print(f"  Sign stable: {'YES (all negative)' if sign_stable else 'NO'}")
        print("=" * 70)

        # Compare with Random baseline
        print("\n  [Comparison with Random baseline]")
        print(f"  Random (10-seed): score_aligned median = -0.650, "
              f"10/10 NULL")
        sa = metrics['score_aligned_median']
        pl = metrics['pass_level']
        if sa > 0:
            print(f"  D-optimal:        score_aligned = {sa:.3f} — "
                  f"IMPROVEMENT over Random")
        else:
            print(f"  D-optimal:        score_aligned = {sa:.3f} — "
                  f"still negative")
        print(f"  D-optimal pass_level: {pl}")

        return {
            'run_id': run_id,
            'run_dir': str(run_dir),
            'status': 'completed',
            'delta_raw': metrics['delta_raw_median'],
            'score_aligned': metrics['score_aligned_median'],
            'pass_level': metrics['pass_level'],
            'support': int(eval_result['support_mask'].sum()),
            'kappa': eval_result['kappa'],
            'pool_sha': pool_sha,
            'eval_seed_sensitivity': {
                'sign_stable': sign_stable,
                'score_aligned_values': all_sa,
            },
        }


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='AEK-4c D-optimal Augmentation')
    p.add_argument('--pool_size', type=int, default=200)
    p.add_argument('--n_select', type=int, default=50)
    p.add_argument('--n_bootstrap', type=int, default=100)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--baseline_seed', type=int, default=0)
    p.add_argument('--baseline_dir', type=str, default='')
    p.add_argument('--dopt_lambda', type=float, default=1e-6)
    p.add_argument('--dopt_gram_mode', type=str, default='unit_trace',
                   choices=['raw', 'unit_trace', 'trace_power'])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = AEK4cDoptConfig(
        pool_size=args.pool_size,
        n_select=args.n_select,
        n_bootstrap=args.n_bootstrap,
        threshold=args.threshold,
        baseline_seed=args.baseline_seed,
        baseline_dir=args.baseline_dir,
        dopt_lambda=args.dopt_lambda,
        dopt_gram_energy_mode=args.dopt_gram_mode,
    )

    runner = AEK4cDoptRunner(cfg)
    try:
        result = runner.run()
        print(f"\nAEK-4c D-optimal complete: {result['pass_level']}")
    except Exception as e:
        print(f"\nAEK-4c D-optimal FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()