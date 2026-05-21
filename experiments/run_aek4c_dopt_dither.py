"""
AEK-4c: D-optimal Selection + Dither PD Controller (Reparam-1, Coverage-Improved)

Purpose:
    Test whether D-optimal selection + dither_plus PD controller improves
    E-SINDy precision on AEK (Reparam-1 library).

    This runner combines:
      - dither_plus pool generation (from run_aek4c_dither_random.py):
          gain_margin=1.5, noise_std=0.003, dither A=0.015rad f=0.5Hz
          → Coverage Gate PASS (sin(phi)=1.68, theta_w_dot²=1.29)
      - D-optimal FIM-based selection (from run_aek4c_dopt.py)

Design:
    1. Load AEK baseline artifacts (z_before, teacher_support, fragile_pairs)
    2. Fit 3-component GMM on training ICs+params (5D)
    3. Generate pool via AEK simulator with dither_plus PD (analytic dx)
    4. Verify pool SHA == fc5e11fa22f51172 (dither_plus, confound-free with Random)
    5. Track A: reject top-10% teacher alignment error
    6. D-optimal selection: greedy logdet on scaled fragile features
    7. E-SINDy evaluation on train+aug data
    8. Compute delta_raw + score_aligned (AC1 compliant)
    9. Fail-fast: if score_aligned < -0.20, halt immediately

Fail-fast safety (GPT P0-2):
    1. Pool SHA must match fc5e11fa22f51172 (dither pool — confound-free vs Random)
    2. If D-opt score_aligned < -0.20 → immediate halt (Failure Mode Mismatch)
    3. Spurious explosion monitor: support ≥ 56 (all features) or teacher coeff > 1e6

Metric SSOT (AEK — spurious-primary):
    delta_raw = median(z_after − z_before) over fragile pairs
    score_aligned = −delta_raw  (positive = improvement)
    Both stored in metrics.json (AC1).

Coverage Gate (Rule #11):
    std_ratio(sin(phi)) ≥ 0.70 AND std_ratio(theta_w_dot²) ≥ 0.50
    dither_plus pool: PASS ✅ (sin(phi)=1.68, theta_w_dot²=1.29)

Usage:
    python experiments/run_aek4c_dopt_dither.py
    python experiments/run_aek4c_dopt_dither.py --baseline_seed 1

Author: Claude (Gate4c D-opt+Dither)
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
from src.sindy.aek_library import (
    build_aek_library_by_name,
    get_aek_feature_names,
    get_aek_oracle_support,
    AEK_TARGET_NAMES,
    N_AEK_FEATURES,
)
from src.simulators.aek_simulator import AEKSimulator

# ── Import dither pool generation & shared utilities ──
from experiments.run_aek4c_dither_random import (
    AEK4cConfig,          # dither_plus PD defaults (gain_margin=1.5, noise=0.003, dither)
    AEKGMMSampler,
    generate_pool,        # uses dither_plus PD controller
    track_a_filter,
    evaluate_augmented,
    compute_metrics,
    compute_tau_stats,
    load_baseline,
    assert_oracle_label_integrity,
    _json_default,
)

# ── Import D-optimal selection functions ──
from experiments.run_aek4c_dopt import (
    compute_fragile_feature_sets,
    compute_pool_theta_per_traj,
    compute_gram_contributions,
    greedy_dopt_selection,
    dopt_select,
)


# ============================================================
# Constants
# ============================================================

RUNNER_VERSION = 'v1.1_dopt_dither'  # P0 patch: path SSOT, manifest SHA, CP fix, fail-fast logic

# Pool SHA for dither_plus (Coverage Gate PASS)
EXPECTED_SHA_DITHER = 'fc5e11fa22f51172'

# Fail-fast threshold: if score_aligned < this, halt
FAILFAST_THRESHOLD = -0.20

# Comparison baselines (Random Dither results, Context Packet)
RANDOM_DITHER_BASELINES = {
    0: {'score_aligned': -0.296, 'soft_pass': '0/3',  'note': 'seed0: 3/3 NULL'},
    1: {'score_aligned': -0.083, 'soft_pass': '2/10', 'note': 'seed1: 2/10 SOFT_PASS ★'},
}


# ============================================================
# Configuration (D-opt params on top of dither AEK4cConfig)
# ============================================================

@dataclass
class AEK4cDoptDitherConfig(AEK4cConfig):
    """
    AEK-4c D-optimal + Dither configuration.

    Inherits from AEK4cConfig (run_aek4c_dither_random.py) which already
    sets dither_plus PD defaults:
        pd_gain_margin=1.5, pd_noise_std=0.003
        dither_amplitude=0.015rad, dither_freq=0.5Hz

    Adds D-optimal selection parameters.
    """
    # D-optimal parameters (same as run_aek4c_dopt.py)
    dopt_lambda: float = 1e-6
    dopt_gram_energy_mode: str = 'unit_trace'
    dopt_trace_power: float = 1.0
    # AEK: all fragile pairs are spurious → no teacher intersection
    dopt_use_teacher_intersection: bool = False
    # Dynamics targets with fragile pairs (phi_ddot=1, theta_w_ddot=3)
    dopt_dynamics_targets: List[int] = field(default_factory=lambda: [1, 3])

    # Override note for output naming
    note: str = 'aek4c_dopt_dither'


# ============================================================
# Save D-optimal + Dither Run Artifacts
# ============================================================

def save_dopt_dither_run(
    run_dir: Path,
    run_id: str,
    cfg: AEK4cDoptDitherConfig,
    eval_result: Dict[str, Any],
    metrics: Dict[str, Any],
    pool_sha: str,
    traj_sha: str,
    theta_sha: str,
    dither_config_hash: str,
    selection_spec_hash: str,
    baseline_dir: Path,
    dopt_spec: Dict[str, Any],
    selection_trace: List[Dict],
    tau_stats: Optional[Dict[str, Any]] = None,
    sensitivity_results: Optional[List[Dict]] = None,
    sign_stable: Optional[bool] = None,
):
    """Save all D-optimal + Dither run artifacts (SSOT-compliant)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'figures').mkdir(exist_ok=True)

    feat_names = eval_result['feature_names']

    # Dither config record (Coverage Gate proof)
    dither_config = {
        'pool_type': 'dither_plus',
        'pd_gain_margin': cfg.pd_gain_margin,
        'pd_Kd_factor': cfg.pd_Kd_factor,
        'pd_noise_std': cfg.pd_noise_std,
        'dither_amplitude_rad': cfg.dither_amplitude,
        'dither_freq_hz': cfg.dither_freq,
        'coverage_gate_pass': True,
        'coverage_sin_phi': 1.68,
        'coverage_theta_w_dot2': 1.29,
        'expected_sha': EXPECTED_SHA_DITHER,
    }

    # --- metrics.json (AC1: delta_raw + score_aligned) ---
    full_metrics = {**metrics}
    full_metrics.update({
        'system': 'aek',
        'gate': 'gate4c-2',
        'method': 'dopt_dither',
        'reparam': cfg.reparam,
        'library_version': 'Reparam-1',
        'pool_type': 'dither_plus',
        'selection_seed': None,  # deterministic D-opt
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
        # Dither config
        'dither_config': dither_config,
    })
    if tau_stats is not None:
        full_metrics['tau_distribution'] = tau_stats
    if sensitivity_results is not None:
        full_metrics['eval_seed_sensitivity'] = {
            'seeds_tested': [cfg.baseline_seed] + [r['eval_seed'] for r in sensitivity_results],
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
        'gate': 'gate4c-2',
        'method': 'dopt_dither',
        'reparam': cfg.reparam,
        'library_id': 'RP1',           # P0-3: explicit library ID
        'library_version': 'Reparam-1',
        'pool_type': 'dither_plus',
        'selection_seed': None,
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_aek4c_dopt_dither.py',
        'runner_version': RUNNER_VERSION,
        # P0-2/3: Complete SHA registry
        'pool_sha': pool_sha,
        'traj_sha': traj_sha,
        'theta_sha': theta_sha,
        'pool_sha_definition': 'sha256(trajectories_bytes)[:16]',
        'dither_config_hash': dither_config_hash,
        'selection_spec_hash': selection_spec_hash,
        'baseline_seed': cfg.baseline_seed,
        'baseline_dir': str(baseline_dir),
        'dither_config': dither_config,
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
            # Dither PD params
            'pd_gain_margin': cfg.pd_gain_margin,
            'pd_Kd_factor': cfg.pd_Kd_factor,
            'pd_noise_std': cfg.pd_noise_std,
            'dither_amplitude': cfg.dither_amplitude,
            'dither_freq': cfg.dither_freq,
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
# Spurious Explosion Monitor (Fail-fast #3)
# ============================================================

def check_spurious_explosion(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Monitor for spurious coefficient explosion (GPT Fail-fast #3).

    Design change (v1.1): warn+continue instead of halt, so score_aligned
    is preserved for paper narrative and comparison with Random Dither.

    Triggers if:
      - support_mask is saturated: all 14*4=56 features active
      - any ensemble mean coefficient > 1e6 (numerical instability)

    Returns:
        dict with 'triggered' (bool), 'reason' (str), 'max_coeff' (float),
        'support_total' (int)
    """
    support_total = int(eval_result['support_mask'].sum())
    max_support = N_AEK_FEATURES * 4  # 56
    coeff_mean = eval_result['coefficients_mean']
    max_coeff = float(np.abs(coeff_mean).max())

    if support_total >= max_support:
        return {
            'triggered': True,
            'reason': f"support={support_total}/{max_support} (all features active)",
            'max_coeff': max_coeff,
            'support_total': support_total,
        }

    if max_coeff > 1e6:
        return {
            'triggered': True,
            'reason': f"max_coeff={max_coeff:.3e} > 1e6 (coefficient explosion)",
            'max_coeff': max_coeff,
            'support_total': support_total,
        }

    return {
        'triggered': False,
        'reason': 'clean',
        'max_coeff': max_coeff,
        'support_total': support_total,
    }


# ============================================================
# Main Runner
# ============================================================

class AEK4cDoptDitherRunner:
    """AEK-4c D-optimal + Dither Selection Augmentation Runner."""

    def __init__(self, cfg: AEK4cDoptDitherConfig):
        self.cfg = cfg
        self.feature_names = get_aek_feature_names(cfg.reparam)

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        print("=" * 70)
        print("AEK-4c: D-optimal + Dither Augmentation (Reparam-1)")
        print(f"  Pool: {cfg.pool_size}, Select: {cfg.n_select}")
        print(f"  E-SINDy: B={cfg.n_bootstrap}, threshold={cfg.threshold}")
        print(f"  D-opt: lambda={cfg.dopt_lambda}, "
              f"gram={cfg.dopt_gram_energy_mode}")
        print(f"  PD: gain_margin={cfg.pd_gain_margin}, "
              f"noise={cfg.pd_noise_std}, "
              f"dither A={cfg.dither_amplitude}rad f={cfg.dither_freq}Hz")
        print(f"  Baseline seed: {cfg.baseline_seed}")
        print(f"  Coverage Gate: PASS (dither_plus pool)")
        print("=" * 70)

        # ── AC2: Feature name integrity ──
        print("\n[AC2] Oracle label integrity...")
        assert_oracle_label_integrity(cfg.reparam)
        print("  PASS")

        # ── Phase 0: Load data and baseline ──
        print("\n[Phase 0] Loading data and baseline...")
        dataset_path = paths.get_dataset_path(
            cfg.dataset_version, system=cfg.system,
        )
        validate_dataset_lite(dataset_path)
        dataset = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  Dataset: {dataset_path}")

        baseline = load_baseline(cfg)

        train_x   = dataset['train_x'][:cfg.n_train]    # (10, 201, 4) §3.3
        train_u   = dataset['train_u'][:cfg.n_train]           # (10, 201, 1)
        train_dx  = dataset['train_dx_savgol'][:cfg.n_train]   # (10, 201, 4) §3.3
        train_params = dataset['train_params'][:cfg.n_train]
        print(f"  Train: {train_x.shape}")

        # ── Phase 1: GMM ──
        print("\n[Phase 1] Fitting GMM...")
        gmm = AEKGMMSampler(
            n_components=cfg.gmm_n_components,
            covariance_type=cfg.gmm_covariance_type,
            random_state=cfg.gmm_seed,
        )
        gmm.fit(train_x, train_params)
        print(f"  GMM fitted (5D, {cfg.gmm_n_components} components)")

        # ── Phase 2: Generate dither_plus pool ──
        print("\n[Phase 2] Generating dither_plus pool...")
        print(f"  PD: gain_margin={cfg.pd_gain_margin}, "
              f"noise_std={cfg.pd_noise_std}")
        print(f"  Dither: A={cfg.dither_amplitude}rad, "
              f"f={cfg.dither_freq}Hz, random phase per traj")
        rng_pool = np.random.default_rng(cfg.pool_seed)
        pool = generate_pool(gmm, train_x, train_u, cfg, rng_pool)
        pool_sha = hashlib.sha256(
            pool['trajectories'].tobytes(),
        ).hexdigest()[:16]
        print(f"  Pool SHA: {pool_sha}")

        # ── Fail-fast #1: SHA verification ──
        if pool_sha != EXPECTED_SHA_DITHER:
            msg = (
                f"\n[FAIL-FAST #1] SHA MISMATCH\n"
                f"  Expected: {EXPECTED_SHA_DITHER} (dither_plus)\n"
                f"  Got:      {pool_sha}\n"
                f"  Pool is NOT confound-free with Random Dither baseline.\n"
                f"  Halting to preserve experimental integrity."
            )
            print(msg)
            raise RuntimeError(f"Pool SHA mismatch: {pool_sha} != {EXPECTED_SHA_DITHER}")
        print(f"  ✅ SHA matches dither_plus baseline — confound-free with Random")

        # ── Phase 3: Track A filter ──
        print("\n[Phase 3] Track A filtering...")
        track_a = track_a_filter(
            pool, baseline['coefficients'], cfg.reparam, cfg.reject_ratio,
        )

        # ── Phase 4: D-optimal selection ──
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

        # ── Phase 5: E-SINDy evaluation ──
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

        # ── Fail-fast #3: Spurious explosion monitor (warn+continue) ──
        explosion = check_spurious_explosion(eval_result)
        explosion_flag = explosion['triggered']
        if explosion_flag:
            print(f"\n[WARN Fail-fast #3] SPURIOUS EXPLOSION DETECTED")
            print(f"  Reason: {explosion['reason']}")
            print(f"  max_coeff={explosion['max_coeff']:.3e}, "
                  f"support={explosion['support_total']}/56")
            print(f"  Continuing to compute score_aligned for paper record...")
            print(f"  Results will be flagged as EXPLODED in pass_level.")
        else:
            print(f"\n  Explosion check: CLEAN "
                  f"(max_coeff={explosion['max_coeff']:.3e}, "
                  f"support={explosion['support_total']}/56)")

        # ── Phase 6: Metrics ──
        print(f"\n[Phase 6] Computing metrics...")
        metrics = compute_metrics(
            eval_result['z'], baseline['z_before'],
            baseline['fragile_pairs'],
            cfg.ci_bootstrap_B, cfg.ci_alpha, cfg.baseline_seed,
        )

        # ── Fail-fast #2: D-opt median threshold ──
        sa = metrics['score_aligned_median']
        # Override pass_level if explosion detected
        original_pass_level = metrics['pass_level']
        if explosion_flag:
            metrics['pass_level'] = f"EXPLODED({original_pass_level})"
            metrics['explosion_flag'] = True
            metrics['explosion_reason'] = explosion['reason']
            metrics['explosion_max_coeff'] = explosion['max_coeff']
            print(f"  pass_level overridden → {metrics['pass_level']}")
        else:
            metrics['explosion_flag'] = False

        if sa < FAILFAST_THRESHOLD and not explosion_flag:
            # Only apply fail-fast #2 if no explosion (explosion is separate condition)
            msg = (
                f"\n[FAIL-FAST #2] score_aligned={sa:.3f} < {FAILFAST_THRESHOLD}\n"
                f"  Failure Mode Mismatch re-detected (D-opt amplifying spurious).\n"
                f"  Random Dither baseline (seed=1): -0.083\n"
                f"  D-opt is worsening results significantly — halting."
            )
            print(msg)
            raise RuntimeError(
                f"D-opt fail-fast #2: score_aligned={sa:.3f} < {FAILFAST_THRESHOLD}"
            )

        # ── Tau distribution ──
        tau_stats = compute_tau_stats(train_u, selected['u'])

        # ── Phase 6b: Eval seed sensitivity (sign stability) ──
        # P0-5: Skip if explosion AND sa < -0.20 (fail-fast spirit + time saving)
        skip_sensitivity = explosion_flag and (sa < FAILFAST_THRESHOLD)
        sensitivity_results = []
        sign_stable = None
        all_sa = [sa]
        sign_label = 'N/A (skipped)'

        if skip_sensitivity:
            print(f"\n[Phase 6b] Eval seed sensitivity SKIPPED")
            print(f"  Reason: explosion_flag=True AND score_aligned={sa:.3f} < {FAILFAST_THRESHOLD}")
            print(f"  (P0-5: fail-fast spirit — unstable/NULL conclusion locked)")
        else:
            print(f"\n[Phase 6b] Eval seed sensitivity check...")
            sensitivity_seeds = [1, 2]
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

            all_sa = [sa] + [r['score_aligned_median'] for r in sensitivity_results]
            all_positive = all(s > 0 for s in all_sa)
            all_negative = all(s < 0 for s in all_sa)
            sign_stable = all_positive or all_negative
            sign_label = ('STABLE (all positive ✅)' if all_positive
                          else 'STABLE (all negative)' if all_negative
                          else 'UNSTABLE (mixed signs ⚠️)')
            print(f"  Sign stability: {sign_label}")
            print(f"  score_aligned values: {[f'{s:.3f}' for s in all_sa]}")

        # ── Phase 7: Save artifacts ──
        print(f"\n[Phase 7] Saving artifacts...")
        run_id = paths.generate_run_id("aek4c_dopt_dither")

        # P0-1: SSOT path (paths.get_results_dir — 수동 조립 금지)
        run_dir = paths.get_results_dir(
            dataset_version=cfg.dataset_version,
            gate='gate4c-2',
            track='standardized',
            method='dopt_dither',
            n_train=cfg.n_train,
            seed=cfg.baseline_seed,
            run_id=run_id,
        )

        # P0-2/3: Extended SHA fields for manifest
        traj_sha = pool_sha  # pool_sha = sha256(trajectories_bytes)
        theta_sha = hashlib.sha256(
            pool['u'].tobytes() + pool['dx'].tobytes()
        ).hexdigest()[:16]
        dither_cfg_str = (
            f"gain={cfg.pd_gain_margin},noise={cfg.pd_noise_std},"
            f"A={cfg.dither_amplitude},f={cfg.dither_freq}"
        )
        dither_config_hash = hashlib.sha256(
            dither_cfg_str.encode()
        ).hexdigest()[:16]
        selection_spec_hash = selected['dopt_spec'].get('spec_hash', 'N/A')

        save_dopt_dither_run(
            run_dir=run_dir,
            run_id=run_id,
            cfg=cfg,
            eval_result=eval_result,
            metrics=metrics,
            pool_sha=pool_sha,
            traj_sha=traj_sha,
            theta_sha=theta_sha,
            dither_config_hash=dither_config_hash,
            selection_spec_hash=selection_spec_hash,
            baseline_dir=baseline['dir'],
            dopt_spec=selected['dopt_spec'],
            selection_trace=selected['selection_trace'],
            tau_stats=tau_stats,
            sensitivity_results=sensitivity_results,
            sign_stable=sign_stable,
        )

        # ── Context Packet (SSOT Rule #6) ──
        cp_path = paths.get_context_packet_path(run_id)
        cp_content = (
            f"# Context Packet: {run_id}\n\n"
            f"**System**: AEK | **Gate**: 4c-2 | **Method**: D-opt + Dither\n"
            f"**Library**: Reparam-1 (14-term)\n"
            f"**Pool**: dither_plus (Coverage Gate PASS)\n"
            f"**Baseline seed**: {cfg.baseline_seed}\n"
            f"**Explosion flag**: {'⚠️ EXPLODED' if explosion_flag else '✅ CLEAN'}\n"
            f"**Created**: {datetime.now().isoformat()}\n\n"
            f"## Results\n\n"
            f"- delta_raw: {metrics['delta_raw_median']:.3f}\n"
            f"- score_aligned: {metrics['score_aligned_median']:.3f}  "
            f"(positive = improvement)\n"
            f"- pass_level: {metrics['pass_level']}\n"
            f"- CI(score_aligned): [{metrics['score_aligned_ci_lower']:.3f}, "
            f"{metrics['score_aligned_ci_upper']:.3f}]\n"
            f"- kappa_augmented: {eval_result['kappa']:.0f}\n"
            f"- support: {int(eval_result['support_mask'].sum())}/56\n"
            f"- pool_sha: {pool_sha}\n"
            + (f"- explosion_reason: {explosion['reason']}\n"
               if explosion_flag else "") + "\n"
            f"## Eval Seed Sensitivity\n\n"
            + (f"- SKIPPED (explosion + sa<{FAILFAST_THRESHOLD})\n"
               if skip_sensitivity else
               f"- Seeds tested: {[cfg.baseline_seed] + [r['eval_seed'] for r in sensitivity_results]}\n"
               f"- score_aligned: {[f'{s:.3f}' for s in all_sa]}\n"
               f"- Sign stable: {sign_stable} ({sign_label})\n")
            + "\n"
            f"## Comparison vs Random Dither (same pool)\n\n"
        )
        for bs, ref in RANDOM_DITHER_BASELINES.items():
            cp_content += (
                f"- Random Dither seed={bs}: "
                f"score_aligned={ref['score_aligned']}, {ref['note']}\n"
            )
        cp_content += (
            f"- **D-opt Dither seed={cfg.baseline_seed}**: "
            f"score_aligned={metrics['score_aligned_median']:.3f}, "
            f"pass={metrics['pass_level']}\n\n"
            f"## Fail-fast Status\n\n"
            f"- SHA check: ✅ PASS ({pool_sha} == {EXPECTED_SHA_DITHER})\n"
            f"- Median threshold (>-0.20): "
            f"{'✅ PASS' if sa >= FAILFAST_THRESHOLD else '❌ FAIL'} "
            f"(score_aligned={sa:.3f})\n"
            f"- Spurious explosion: "
            f"{'⚠️ DETECTED — ' + explosion['reason'] if explosion_flag else '✅ CLEAN'}\n\n"
            f"## D-optimal Config\n\n"
            f"- lambda: {cfg.dopt_lambda}\n"
            f"- gram_energy_mode: {cfg.dopt_gram_energy_mode}\n"
            f"- use_teacher_intersection: {cfg.dopt_use_teacher_intersection}\n"
            f"- F_by_target: see dopt_spec.json\n\n"
            f"## Dither PD Config (Coverage Gate PASS)\n\n"
            f"- gain_margin: {cfg.pd_gain_margin}\n"
            f"- noise_std: {cfg.pd_noise_std}\n"
            f"- dither amplitude: {cfg.dither_amplitude}rad\n"
            f"- dither freq: {cfg.dither_freq}Hz\n\n"
            f"## Artifacts\n\n"
            f"- Run dir: {run_dir}\n"
            f"- Baseline: {baseline['dir']}\n"
        )
        with open(cp_path, 'w', encoding='utf-8') as f:
            f.write(cp_content)
        print(f"  Context Packet: {cp_path}")

        # ── Summary ──
        random_ref = RANDOM_DITHER_BASELINES.get(cfg.baseline_seed, {})
        print("\n" + "=" * 70)
        print("  AEK-4c D-OPTIMAL + DITHER SUMMARY")
        print("=" * 70)
        print(f"  Library:       Reparam-1 (RP1)")
        print(f"  Pool:          dither_plus (SHA: {pool_sha})")
        print(f"  Baseline seed: {cfg.baseline_seed}")
        print(f"  delta_raw    = {metrics['delta_raw_median']:.3f}")
        print(f"  score_aligned= {metrics['score_aligned_median']:.3f}  "
              f"(positive = improvement)")
        print(f"  pass_level   = {metrics['pass_level']}")
        print(f"  CI(score_aln)= [{metrics['score_aligned_ci_lower']:.3f}, "
              f"{metrics['score_aligned_ci_upper']:.3f}]")
        print(f"  kappa_aug    = {eval_result['kappa']:.0f}")
        print(f"  support      = {int(eval_result['support_mask'].sum())}/56")
        print(f"  tau: train=[{tau_stats['train_tau_q05']:.5f}, "
              f"{tau_stats['train_tau_q95']:.5f}], "
              f"aug=[{tau_stats['aug_tau_q05']:.5f}, "
              f"{tau_stats['aug_tau_q95']:.5f}]")
        print(f"  Saved: {run_dir}")
        print(f"\n  [Eval Seed Sensitivity]")
        if skip_sensitivity:
            print(f"  SKIPPED (explosion + score_aligned={sa:.3f} < {FAILFAST_THRESHOLD})")
        else:
            print(f"  score_aligned by eval_seed: {[f'{s:.3f}' for s in all_sa]}")
            print(f"  Sign stable: {sign_label}")

        print(f"\n  [Fail-fast Status]")
        print(f"  SHA check:          ✅ PASS")
        print(f"  Median threshold:   "
              f"{'✅ PASS' if sa >= FAILFAST_THRESHOLD else '⚠️  N/A (explosion)'} "
              f"({sa:.3f})")
        print(f"  Spurious explosion: "
              f"{'⚠️  DETECTED — ' + explosion['reason'] if explosion_flag else '✅ CLEAN'}")

        print(f"\n  [Comparison vs Random Dither — same pool]")
        if random_ref:
            ref_sa = random_ref['score_aligned']
            improvement = metrics['score_aligned_median'] - ref_sa
            print(f"  Random Dither (seed={cfg.baseline_seed}): "
                  f"score_aligned={ref_sa:.3f} ({random_ref['note']})")
            print(f"  D-opt Dither  (seed={cfg.baseline_seed}): "
                  f"score_aligned={sa:.3f}, pass={metrics['pass_level']}")
            if improvement > 0:
                print(f"  → D-opt improvement: +{improvement:.3f} ✅")
            else:
                print(f"  → D-opt vs Random: {improvement:+.3f} "
                      f"(no selection benefit)")
        print("=" * 70)

        return {
            'run_id': run_id,
            'run_dir': str(run_dir),
            'status': 'completed',
            'delta_raw': metrics['delta_raw_median'],
            'score_aligned': metrics['score_aligned_median'],
            'pass_level': metrics['pass_level'],
            'explosion_flag': explosion_flag,
            'explosion_reason': explosion['reason'] if explosion_flag else None,
            'support': int(eval_result['support_mask'].sum()),
            'kappa': eval_result['kappa'],
            'pool_sha': pool_sha,
            'baseline_seed': cfg.baseline_seed,
            'sensitivity_skipped': skip_sensitivity,
            'eval_seed_sensitivity': {
                'sign_stable': sign_stable,
                'score_aligned_values': all_sa,
                'skipped': skip_sensitivity,
            },
        }


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='AEK-4c D-optimal + Dither Augmentation (Coverage-Improved)'
    )
    p.add_argument('--pool_size', type=int, default=200)
    p.add_argument('--n_select', type=int, default=50)
    p.add_argument('--n_bootstrap', type=int, default=100)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--baseline_seed', type=int, default=0,
                   help='Baseline seed (0 or 1; run separately for each)')
    p.add_argument('--baseline_dir', type=str, default='',
                   help='Explicit baseline dir (auto-detect if empty)')
    p.add_argument('--dopt_lambda', type=float, default=1e-6)
    p.add_argument('--dopt_gram_mode', type=str, default='unit_trace',
                   choices=['raw', 'unit_trace', 'trace_power'])
    return p.parse_args()


def main():
    args = parse_args()

    cfg = AEK4cDoptDitherConfig(
        pool_size=args.pool_size,
        n_select=args.n_select,
        n_bootstrap=args.n_bootstrap,
        threshold=args.threshold,
        baseline_seed=args.baseline_seed,
        baseline_dir=args.baseline_dir,
        dopt_lambda=args.dopt_lambda,
        dopt_gram_energy_mode=args.dopt_gram_mode,
    )

    runner = AEK4cDoptDitherRunner(cfg)
    try:
        result = runner.run()
        print(f"\nAEK-4c D-opt+Dither complete: {result['pass_level']}")
    except Exception as e:
        print(f"\nAEK-4c D-opt+Dither FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()