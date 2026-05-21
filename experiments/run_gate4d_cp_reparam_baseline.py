"""
Gate4d: Cart-Pole Reparam-1 Baseline (κ Comparison + fragile_pairs)

Purpose:
    이식 Step 1 — AEK에서 검증된 Reparam 전략(cos(theta)→cos(theta)-1)을
    Cart-Pole에 이식하여 κ 개선과 fragile_pairs를 확인.

    AEK 결과 요약:
        Standard:  κ ≈ 4.7×10⁹
        Reparam-1: κ ≈ 4.5×10⁴  (Δlog₁₀ = 5.02)

    CP에서의 기대:
        Standard:  κ = ? (훈련 데이터 기반)
        Reparam-1: κ = ? (cos(theta)→cos(theta)-1로 collinearity 해소)

Design:
    1. Load CP training dataset (n_train=10)
    2. Run E-SINDy (Standard, 21-term) → z_std, fragile_pairs_std, κ_std
    3. Run E-SINDy (Reparam-1, 21-term) → z_rp1, fragile_pairs_rp1, κ_rp1
    4. κ comparison table (Δlog₁₀)
    5. fragile_pairs comparison (count + index sets)
    6. Save Reparam-1 baseline artifacts (for Gate4d D-opt runner)

Metric SSOT (CP — dynamics-primary, recall fragility):
    delta_raw = median(z_after − z_before) over fragile pairs
    score_aligned = +delta_raw  (양수 = 개선)
    [Note: CP는 +delta_raw, AEK는 −delta_raw]

Fragile pair definition (CP):
    (feature_idx, target_idx) where:
        teacher_support[i, j] = True   AND
        z_before[pair_index] < Z_FRAGILE_THRESHOLD (= 1.0)
    → Recall fragility: 활성화 되어야 하나 불안정하게 회복

Usage:
    python experiments/run_gate4d_cp_reparam_baseline.py
    python experiments/run_gate4d_cp_reparam_baseline.py --baseline_seed 1

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
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

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
    CP_REPARAM1_COS_INDICES,
)


# ============================================================
# Constants
# ============================================================

RUNNER_VERSION = 'v1.0_gate4d_baseline'

# Fragile pair threshold: z < this → fragile (inconsistently recovered)
Z_FRAGILE_THRESHOLD = 1.0

# Dataset (same as Gate4a)
DATASET_VERSION = 'cartpole_ood_v1'
SYSTEM = 'cartpole'

# Gate4a Standard reference (from SSOT)
GATE4A_STANDARD_DOPT_MEDIAN = 0.424   # STRONG_PASS
GATE4A_STANDARD_RANDOM_MEDIANS = {'NULL': 3, 'SOFT': 6, 'CEILING': 1}


def _json_default(obj):
    """JSON serialization fallback."""
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
# E-SINDy Evaluation (CP baseline — no augmentation)
# ============================================================

def run_esindy_baseline(
    train_x: np.ndarray,    # (N, T, 4)
    train_u: np.ndarray,    # (N, T, 1)
    train_dx: np.ndarray,   # (N, T, 4)
    reparam: str,
    n_bootstrap: int = 100,
    threshold: float = 0.05,
    seed: int = 42,
    z_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Run E-SINDy on training data only (no augmentation).

    Returns:
        Dict with z (N_fragile_candidates,), z_full (21,4),
        support_mask (21,4), coefficients_mean (21,4),
        kappa, feature_names, n_total_samples
    """
    N_tr, T, D = train_x.shape

    # Flatten
    x_flat = train_x.reshape(-1, D)           # (N*T, 4)
    u_flat = train_u.reshape(-1, 1)           # (N*T, 1)
    dx_flat = train_dx.reshape(-1, D)         # (N*T, 4)

    n_samples = x_flat.shape[0]

    # Build library
    Theta, feat_names = build_cp_library_by_name(x_flat, u_flat, reparam=reparam)

    # Scale columns
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    # Condition number
    kappa = float(np.linalg.cond(Theta_scaled))

    # E-SINDy ensemble
    n_traj = N_tr
    T_steps = T
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

    # Ensemble statistics (attribute names match esindy.py)
    coeff_mean = ensemble.coefficients_mean_    # (21, 4) unscaled
    coeff_std  = ensemble.coefficients_std_     # (21, 4) unscaled
    support    = np.abs(coeff_mean) > 0         # (21, 4)

    # Z-score per (feature, target)
    z_full = np.abs(coeff_mean) / (coeff_std + z_eps)  # (21, 4)

    return {
        'z_full': z_full,                  # (21, 4) — all pairs
        'support_mask': support,           # (21, 4)
        'coefficients_mean': coeff_mean,   # (21, 4)
        'coefficients_std': coeff_std,     # (21, 4)
        'kappa': kappa,
        'feature_names': feat_names,
        'n_total_samples': n_samples,
        'reparam': reparam,
    }


# ============================================================
# Fragile Pair Computation (CP — recall fragility)
# ============================================================

def compute_fragile_pairs_cp(
    result: Dict[str, Any],
    z_threshold: float = Z_FRAGILE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute fragile pairs from E-SINDy baseline results (CP).

    Fragile pair definition (recall fragility):
        (i, j) where teacher_support[i,j]=True AND z_full[i,j] < z_threshold

    Teacher support = E-SINDy ensemble support mask.
    z < threshold means high coefficient variance relative to mean
    → feature is inconsistently recovered.

    Args:
        result: Output of run_esindy_baseline()
        z_threshold: z-score threshold below which pairs are fragile

    Returns:
        Dict with fragile_pairs, z_before, z_full, n_fragile
    """
    z_full = result['z_full']         # (21, 4)
    support = result['support_mask']  # (21, 4)

    # Fragile = teacher-active AND low z-score
    fragile_mask = support & (z_full < z_threshold)  # (21, 4)

    fragile_pairs = []
    z_before_list = []

    for i in range(N_CP_FEATURES):
        for j in range(4):
            if fragile_mask[i, j]:
                fragile_pairs.append([i, j])
                z_before_list.append(float(z_full[i, j]))

    # Sort by target then feature for determinism
    order = sorted(range(len(fragile_pairs)),
                   key=lambda k: (fragile_pairs[k][1], fragile_pairs[k][0]))
    fragile_pairs = [fragile_pairs[k] for k in order]
    z_before_list = [z_before_list[k] for k in order]

    z_before = np.array(z_before_list)

    return {
        'fragile_pairs': fragile_pairs,
        'z_before': z_before,
        'z_full': z_full,
        'n_fragile': len(fragile_pairs),
        'fragile_mask': fragile_mask,
        'z_threshold': z_threshold,
    }


# ============================================================
# Teacher coefficients (for Track A)
# ============================================================

def get_teacher_coefficients(result: Dict[str, Any]) -> np.ndarray:
    """
    Extract teacher (ensemble mean) coefficients, unscaled.

    Note: ESINDy stores coeff in scaled space. We return the raw
    mean coefficients (in physical units) for Track A alignment check.
    """
    return result['coefficients_mean'].copy()  # (21, 4)


# ============================================================
# Save Baseline Artifacts
# ============================================================

def save_baseline_artifacts(
    run_dir: Path,
    run_id: str,
    reparam: str,
    result: Dict[str, Any],
    fragile_info: Dict[str, Any],
    n_train: int,
    baseline_seed: int,
    dataset_version: str,
):
    """Save baseline artifacts in Gate1-compatible format (for D-opt runner loading)."""
    run_dir.mkdir(parents=True, exist_ok=True)

    feat_names = result['feature_names']
    coeff = result['coefficients_mean']

    # --- fragile_pairs.json ---
    fp_path = run_dir / 'fragile_pairs.json'
    fp_data = {
        'fragile_pairs': fragile_info['fragile_pairs'],
        'z_before': fragile_info['z_before'].tolist(),
        'n_fragile': fragile_info['n_fragile'],
        'z_threshold': fragile_info['z_threshold'],
        'reparam': reparam,
        'baseline_seed': baseline_seed,
        'n_train': n_train,
        'created_at': datetime.now().isoformat(),
    }
    with open(fp_path, 'w') as f:
        json.dump(fp_data, f, indent=2, default=_json_default)

    # --- z_before.npy ---
    np.save(run_dir / 'z_before.npy', fragile_info['z_before'])

    # --- z_full.npy (21×4, all pairs) ---
    np.save(run_dir / 'z_full.npy', fragile_info['z_full'])

    # --- support_mask.npy ---
    np.save(run_dir / 'support_mask.npy', result['support_mask'])

    # --- sindy_coefficients.csv ---
    with open(run_dir / 'sindy_coefficients.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + list(CP_TARGET_NAMES))
        for i, name in enumerate(feat_names):
            w.writerow([name] + [f"{coeff[i,j]:.8f}" for j in range(4)])

    # --- metrics.json ---
    metrics = {
        'system': 'cartpole',
        'gate': 'gate4d',
        'phase': 'baseline',
        'reparam': reparam,
        'library_version': 'Reparam-1' if reparam == 'reparam1' else 'Standard',
        'baseline_seed': baseline_seed,
        'n_train': n_train,
        'dataset_version': dataset_version,
        'kappa': result['kappa'],
        'n_fragile': fragile_info['n_fragile'],
        'z_threshold': fragile_info['z_threshold'],
        'support_active': int(result['support_mask'].sum()),
        'n_total_samples': result['n_total_samples'],
        'runner_version': RUNNER_VERSION,
        'created_at': datetime.now().isoformat(),
    }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    # --- manifest.json ---
    manifest = {
        'run_id': run_id,
        'system': 'cartpole',
        'gate': 'gate4d',
        'method': 'esindy_baseline',
        'reparam': reparam,
        'library_version': 'Reparam-1' if reparam == 'reparam1' else 'Standard',
        'baseline_seed': baseline_seed,
        'created_at': datetime.now().isoformat(),
        'runner': 'experiments/run_gate4d_cp_reparam_baseline.py',
        'runner_version': RUNNER_VERSION,
        'artifacts': [
            'manifest.json', 'metrics.json', 'sindy_coefficients.csv',
            'fragile_pairs.json', 'z_before.npy', 'z_full.npy',
            'support_mask.npy',
        ],
    }
    with open(run_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=_json_default)

    return fp_path


# ============================================================
# Main Runner
# ============================================================

class Gate4dCPReparamBaselineRunner:
    """Gate4d: CP Standard vs Reparam-1 baseline comparison."""

    def __init__(
        self,
        n_train: int = 10,
        n_bootstrap: int = 100,
        threshold: float = 0.05,
        baseline_seed: int = 42,
        z_fragile_threshold: float = Z_FRAGILE_THRESHOLD,
    ):
        self.n_train = n_train
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.baseline_seed = baseline_seed
        self.z_fragile_threshold = z_fragile_threshold

    def run(self) -> Dict[str, Any]:

        print("=" * 70)
        print("Gate4d: Cart-Pole Reparam-1 Baseline (κ Comparison)")
        print(f"  n_train={self.n_train}, B={self.n_bootstrap}, "
              f"threshold={self.threshold}, seed={self.baseline_seed}")
        print(f"  Fragile threshold: z < {self.z_fragile_threshold}")
        print("=" * 70)

        # ── AC2: Feature integrity ──
        print("\n[AC2] CP library integrity checks...")
        assert_cp_feature_integrity('standard')
        assert_cp_feature_integrity('reparam1')

        # ── Phase 0: Load dataset ──
        print("\n[Phase 0] Loading CP dataset...")
        dataset_path = paths.get_dataset_path(DATASET_VERSION, system=SYSTEM)
        validate_dataset_lite(dataset_path)
        dataset = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  Dataset: {dataset_path}")

        train_x  = dataset['train_x'][:self.n_train]   # (10, T, 4)
        train_u  = dataset['train_u'][:self.n_train]   # (10, T, 1)
        train_dx = dataset['train_dx'][:self.n_train]  # (10, T, 4)

        N_tr, T, D = train_x.shape
        print(f"  Train: {train_x.shape} (N={N_tr}, T={T}, D={D})")
        print(f"  dt: {float(dataset['dt']):.4f}s, "
              f"T_steps: {T}")

        # Theta_sha for pool identification later
        x_flat = train_x.reshape(-1, D)
        u_flat = train_u.reshape(-1, 1)
        theta_std, _ = build_cp_library_by_name(x_flat, u_flat, 'standard')
        theta_rp1, _ = build_cp_library_by_name(x_flat, u_flat, 'reparam1')
        train_theta_sha = hashlib.sha256(theta_std.tobytes()).hexdigest()[:16]
        print(f"  Train Theta SHA (Standard): {train_theta_sha}")

        # ── Phase 1: Standard E-SINDy baseline ──
        print(f"\n[Phase 1] E-SINDy baseline — Standard library (21-term)...")
        result_std = run_esindy_baseline(
            train_x, train_u, train_dx,
            reparam='standard',
            n_bootstrap=self.n_bootstrap,
            threshold=self.threshold,
            seed=self.baseline_seed,
        )
        frag_std = compute_fragile_pairs_cp(result_std, self.z_fragile_threshold)

        print(f"  κ (Standard):  {result_std['kappa']:.3e}")
        print(f"  Support active: {int(result_std['support_mask'].sum())}/84")
        print(f"  Fragile pairs:  {frag_std['n_fragile']}")

        # ── Phase 2: Reparam-1 E-SINDy baseline ──
        print(f"\n[Phase 2] E-SINDy baseline — Reparam-1 library (21-term)...")
        result_rp1 = run_esindy_baseline(
            train_x, train_u, train_dx,
            reparam='reparam1',
            n_bootstrap=self.n_bootstrap,
            threshold=self.threshold,
            seed=self.baseline_seed,
        )
        frag_rp1 = compute_fragile_pairs_cp(result_rp1, self.z_fragile_threshold)

        print(f"  κ (Reparam-1): {result_rp1['kappa']:.3e}")
        print(f"  Support active: {int(result_rp1['support_mask'].sum())}/84")
        print(f"  Fragile pairs:  {frag_rp1['n_fragile']}")

        # ── Phase 3: κ Comparison ──
        kappa_std = result_std['kappa']
        kappa_rp1 = result_rp1['kappa']
        delta_log10 = np.log10(kappa_std) - np.log10(kappa_rp1)

        print(f"\n[Phase 3] κ Comparison")
        print(f"  {'Library':<15} {'κ':>12}  {'log₁₀(κ)':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Standard':<15} {kappa_std:>12.3e}  {np.log10(kappa_std):>10.2f}")
        print(f"  {'Reparam-1':<15} {kappa_rp1:>12.3e}  {np.log10(kappa_rp1):>10.2f}")
        print(f"  {'Δlog₁₀(κ)':<15} {'':>12}  {delta_log10:>+10.2f}")

        if delta_log10 >= 2.0:
            kappa_verdict = f"SIGNIFICANT_IMPROVEMENT (Δlog₁₀={delta_log10:.2f} ≥ 2.0)"
        elif delta_log10 >= 0.5:
            kappa_verdict = f"MODERATE_IMPROVEMENT (Δlog₁₀={delta_log10:.2f})"
        elif delta_log10 >= 0:
            kappa_verdict = f"MARGINAL_IMPROVEMENT (Δlog₁₀={delta_log10:.2f})"
        else:
            kappa_verdict = f"NO_IMPROVEMENT (Δlog₁₀={delta_log10:.2f})"
        print(f"\n  Verdict: {kappa_verdict}")
        print(f"  [AEK reference: Δlog₁₀=5.02 (4.7×10⁹→4.5×10⁴)]")

        # ── Phase 4: Fragile Pairs Comparison ──
        print(f"\n[Phase 4] Fragile Pairs Comparison (z < {self.z_fragile_threshold})")
        print(f"  Standard:  {frag_std['n_fragile']} fragile pairs")
        print(f"  Reparam-1: {frag_rp1['n_fragile']} fragile pairs")

        # Which features are fragile in each
        def _fp_set(frag):
            return set((p[0], p[1]) for p in frag['fragile_pairs'])

        fp_std_set = _fp_set(frag_std)
        fp_rp1_set = _fp_set(frag_rp1)
        common = fp_std_set & fp_rp1_set
        std_only = fp_std_set - fp_rp1_set
        rp1_only = fp_rp1_set - fp_std_set

        print(f"\n  Overlap analysis:")
        print(f"  Common:       {len(common)} pairs (fragile in both)")
        print(f"  Std-only:     {len(std_only)} pairs (resolved by Reparam)")
        print(f"  Reparam-only: {len(rp1_only)} pairs (new fragile in Reparam)")

        # ── Phase 5: Save Reparam-1 artifacts (for D-opt runner) ──
        print(f"\n[Phase 5] Saving Reparam-1 baseline artifacts...")
        run_id = paths.generate_run_id(f"gate4d_cp_rp1_baseline_s{self.baseline_seed}")
        run_dir = paths.get_results_dir(
            dataset_version=DATASET_VERSION,
            gate='gate4d',
            track='standardized',
            method='esindy_baseline_rp1',
            n_train=self.n_train,
            seed=self.baseline_seed,
            run_id=run_id,
        )
        save_baseline_artifacts(
            run_dir, run_id, 'reparam1',
            result_rp1, frag_rp1,
            self.n_train, self.baseline_seed, DATASET_VERSION,
        )
        print(f"  Reparam-1 baseline saved: {run_dir}")

        # Also save Standard baseline for reference
        run_id_std = paths.generate_run_id(f"gate4d_cp_std_baseline_s{self.baseline_seed}")
        run_dir_std = paths.get_results_dir(
            dataset_version=DATASET_VERSION,
            gate='gate4d',
            track='standardized',
            method='esindy_baseline_std',
            n_train=self.n_train,
            seed=self.baseline_seed,
            run_id=run_id_std,
        )
        save_baseline_artifacts(
            run_dir_std, run_id_std, 'standard',
            result_std, frag_std,
            self.n_train, self.baseline_seed, DATASET_VERSION,
        )
        print(f"  Standard baseline saved:  {run_dir_std}")

        # ── Context Packet ──
        cp_path = paths.get_context_packet_path(run_id)
        cp_content = (
            f"# Context Packet: {run_id}\n\n"
            f"**System**: Cart-Pole | **Gate**: 4d | **Phase**: Reparam-1 Baseline\n"
            f"**Library**: Standard vs Reparam-1 (21-term)\n"
            f"**Baseline seed**: {self.baseline_seed}\n"
            f"**Created**: {datetime.now().isoformat()}\n\n"
            f"## κ Comparison\n\n"
            f"| Library | κ | log₁₀(κ) |\n"
            f"|---------|---|----------|\n"
            f"| Standard  | {kappa_std:.3e} | {np.log10(kappa_std):.2f} |\n"
            f"| Reparam-1 | {kappa_rp1:.3e} | {np.log10(kappa_rp1):.2f} |\n"
            f"| Δlog₁₀(κ) | | {delta_log10:+.2f} |\n\n"
            f"**Verdict**: {kappa_verdict}\n\n"
            f"## Fragile Pairs (z < {self.z_fragile_threshold})\n\n"
            f"- Standard:  {frag_std['n_fragile']} pairs\n"
            f"- Reparam-1: {frag_rp1['n_fragile']} pairs\n"
            f"- Common: {len(common)} | Std-only: {len(std_only)} | "
            f"Reparam-only: {len(rp1_only)}\n\n"
            f"## Reparam-1 Fragile Pairs Detail\n\n"
            f"Pairs: {frag_rp1['fragile_pairs']}\n\n"
            f"## Artifacts\n\n"
            f"- RP1 baseline dir: {run_dir}\n"
            f"- Std baseline dir: {run_dir_std}\n\n"
            f"## Next Step (Gate4d D-opt)\n\n"
            f"- Load RP1 baseline from: {run_dir}\n"
            f"- Run D-opt augmentation with Reparam-1 library\n"
            f"- Compare vs Standard D-opt (median=0.424, STRONG_PASS)\n"
        )
        with open(cp_path, 'w', encoding='utf-8') as f:
            f.write(cp_content)
        print(f"  Context Packet: {cp_path}")

        # ── Final Summary ──
        print("\n" + "=" * 70)
        print("  GATE4d CP REPARAM-1 BASELINE SUMMARY")
        print("=" * 70)
        print(f"\n  κ Comparison (training data, n_train={self.n_train}):")
        print(f"    Standard:    {kappa_std:.4e}")
        print(f"    Reparam-1:   {kappa_rp1:.4e}")
        print(f"    Δlog₁₀(κ):  {delta_log10:+.2f}  [{kappa_verdict}]")
        print(f"\n  Fragile Pairs (z < {self.z_fragile_threshold}):")
        print(f"    Standard:    {frag_std['n_fragile']}")
        print(f"    Reparam-1:   {frag_rp1['n_fragile']}")
        print(f"\n  Reparam-1 RP1 baseline dir:")
        print(f"    {run_dir}")
        print(f"\n  [AEK reference: Δlog₁₀=5.02]")
        print(f"\n  Next: run_gate4d_cp_reparam_dopt.py --baseline_dir <above>")
        print("=" * 70)

        return {
            'status': 'completed',
            'kappa_standard': kappa_std,
            'kappa_reparam1': kappa_rp1,
            'delta_log10_kappa': float(delta_log10),
            'kappa_verdict': kappa_verdict,
            'n_fragile_standard': frag_std['n_fragile'],
            'n_fragile_reparam1': frag_rp1['n_fragile'],
            'baseline_dir_rp1': str(run_dir),
            'baseline_dir_std': str(run_dir_std),
            'run_id': run_id,
        }


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Gate4d: CP Reparam-1 Baseline (κ comparison)'
    )
    p.add_argument('--n_train', type=int, default=10)
    p.add_argument('--n_bootstrap', type=int, default=100)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--baseline_seed', type=int, default=42,
                   help='E-SINDy ensemble seed (default: 42)')
    p.add_argument('--z_fragile', type=float, default=1.0,
                   help='Fragile pair z-score threshold (default: 1.0)')
    return p.parse_args()


def main():
    args = parse_args()
    runner = Gate4dCPReparamBaselineRunner(
        n_train=args.n_train,
        n_bootstrap=args.n_bootstrap,
        threshold=args.threshold,
        baseline_seed=args.baseline_seed,
        z_fragile_threshold=args.z_fragile,
    )
    try:
        result = runner.run()
        print(f"\nGate4d baseline complete: Δlog₁₀(κ)={result['delta_log10_kappa']:+.2f}")
    except Exception as e:
        print(f"\nGate4d baseline FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()