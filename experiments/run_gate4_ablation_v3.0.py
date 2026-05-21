"""
Gate4 D-optimal Ablation Runner (Production Version)

목적: D-optimal selection의 인과적 기여 검증 (Confound-free 설계)

설계 원칙:
- 동일 Pool, 다른 Selection → 순수 Selection 효과 측정
- Pool: seed=1, pool_seed=42, pool_size=2000 (고정)
- D-optimal: 1 run
- Random: 3 runs (selection_seed=0,1,2)

핵심 구현:
- Pool 생성 1회 (공유)
- Track A 필터링 1회 (공유)
- Selection만 다르게 (D-optimal vs Random)
- E-SINDy 평가 각각 수행

산출물:
- ablation_summary.json: 4 runs 비교
- 각 run별: manifest.json, metrics.json, comparison_gen.json

Author: Claude (Gate4 Ablation)
Date: 2026-02-04
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings
import traceback

import numpy as np

# 프로젝트 모듈 import
from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.sindy.library import SINDyLibrary
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble
from src.utils.derivatives import compute_derivatives_savgol

# Gate3 Runner에서 핵심 함수들 import
from experiments.run_gate3_v2 import (
    Gate3Config,
    Gate3TreatRunner,
    GMMProposalSampler,
    PoolGenerator,
    track_a_selection,
    track_b_dopt_selection,
    final_selection,
    evaluate_with_esindy,
    create_rng_streams,
    generate_run_id,
    GATE3_CONFIG,
    CONTROL_EQUIVALENCE,
    DEFAULT_TARGET_NAMES,
    DEFAULT_TAU_SUPPORT,
    DEFAULT_Z0,
    DEFAULT_EPS,
)


# ============================================================
# Ablation Configuration
# ============================================================

@dataclass
class AblationConfig:
    """D-optimal Ablation 실험 설정"""
    # Fixed Pool Settings (Confound-free)
    seed: int = 1                    # Base seed
    pool_seed: int = 42              # GMM fitting + pool generation seed
    pool_size: int = 2000            # Target pool size
    
    # Selection Settings
    n_select: int = 200              # Number to select from pool
    reject_ratio: float = 0.10       # Track A reject ratio
    
    # D-optimal Settings
    dopt_lambda: float = 1e-6
    dopt_pre_gate_mode: str = 'error'
    dopt_gram_energy_mode: str = 'unit_trace'
    dopt_selection_variant: str = 'greedy'
    dopt_use_teacher_intersection: bool = True
    dopt_top_m_ratio: float = 2.0
    dopt_alpha: float = 0.5
    dopt_trace_power: float = 1.0
    dopt_topL_L: int = 25
    
    # Random Selection Seeds
    random_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    
    # Paths
    dataset_version: str = "cartpole_ood_v1"
    dataset_path: str = ""
    fragile_pairs_source: str = ""
    day3_source: str = ""           # Day3 baseline (teacher_support, norm_stats 등)
    
    # Output
    results_base: str = "results/cartpole_ood_v1/gate4/ablation/d_optimal_vs_random"
    note: str = "ablation"
    
    # E-SINDy Settings
    bootstrap_B: int = 100
    threshold: float = 0.05
    n_train: int = 10
    
    # SSOT Metrics Settings (Gate3/4 aligned)
    ci_bootstrap_B: int = 2000       # Bootstrap samples for CI computation
    ci_alpha: float = 0.05           # CI significance level (95% CI)
    gate2_ceiling: float = 0.058     # Gate2 ceiling for pass_level classification


# ============================================================
# Random Selection with Explicit Seed
# ============================================================

def random_selection_with_seed(
    pool: Dict[str, np.ndarray],
    track_a_result: Dict[str, Any],
    n_select: int,
    selection_seed: int,
) -> Dict[str, Any]:
    """
    Random selection with explicit selection seed.
    
    Confound-free: Uses the same pool, only selection randomness differs.
    Uses Track A filtered candidates (same as D-optimal).
    
    Args:
        pool: Generated pool (동일 pool_seed)
        track_a_result: Track A filtering result
        n_select: Number to select
        selection_seed: Random seed for selection only
        
    Returns:
        selected: Dict with trajectories, indices, stats
    """
    # Use Track A filtered candidates (same filtering as D-optimal)
    candidate_indices = track_a_result['selected_indices']
    all_errors = track_a_result['errors']
    n_pool = len(pool['trajectories'])
    
    print(f"  Random selection (seed={selection_seed}): {len(candidate_indices)} candidates from {n_pool} pool")
    
    # Create RNG with selection_seed
    rng = np.random.default_rng(selection_seed)
    
    n_available = len(candidate_indices)
    
    if n_available < n_select:
        print(f"  ⚠️ Only {n_available} available, selecting all")
        final_indices = candidate_indices.copy()
    else:
        # Random selection from Track A passed candidates
        chosen = rng.choice(n_available, size=n_select, replace=False)
        final_indices = candidate_indices[chosen]
    
    # Sort for deterministic ordering (canonical)
    final_indices = np.sort(final_indices)
    
    # Extract selected data
    result = {
        'trajectories': pool['trajectories'][final_indices],
        'dx': pool['dx'][final_indices],
        'params': pool['params'][final_indices],
        'ic': pool['ic'][final_indices],
        'u': pool['u'][final_indices],
        'u_indices': pool['u_indices'][final_indices],
        'errors': all_errors[final_indices] if len(all_errors) > 0 else np.zeros(len(final_indices)),
        'original_indices': final_indices.copy(),
        'stats': {
            'n_pool': n_pool,
            'n_available': n_available,
            'n_selected': len(final_indices),
            'selection_mode': 'random',
            'selection_seed': selection_seed,
        }
    }
    
    if len(all_errors) > 0 and len(final_indices) > 0:
        selected_errors = all_errors[final_indices]
        result['stats'].update({
            'error_mean_selected': float(selected_errors.mean()),
            'error_std_selected': float(selected_errors.std()),
        })
        print(f"  Selected {len(final_indices)} trajectories")
        print(f"  Error range: [{selected_errors.min():.4f}, {selected_errors.max():.4f}]")
    
    return result


# ============================================================
# Ablation Run Result
# ============================================================

@dataclass
class AblationRunResult:
    """Single ablation run result"""
    run_id: str
    selection_method: str
    selection_seed: Optional[int]
    results_dir: Path
    status: str
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# ============================================================
# Ablation Runner
# ============================================================

class AblationRunner:
    """D-optimal vs Random Ablation Runner"""
    
    def __init__(self, config: AblationConfig):
        self.cfg = config
        self.results_base = Path(config.results_base)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.project_root = _PROJECT_ROOT
        
        # Shared artifacts (generated/loaded once)
        self.dataset = None
        self.pool = None
        self.track_a_result = None
        self.teacher_support = None
        self.teacher_coefficients = None
        self.norm_stats = None
        self.fragile_pairs = None
        self.feature_names = None
        self.target_names = None
        self.dx_std = None
        
        # GMM sampler
        self.gmm_sampler = None
        
        # RNG streams
        self.rng_streams = None
        
    def run_all(self) -> Dict[str, Any]:
        """
        Run all ablation experiments.
        
        1. Load dataset and Day3 artifacts
        2. Generate shared pool (once)
        3. Run Track A selection (once)
        4. Run D-optimal selection
        5. Run Random selections (3 seeds)
        6. Generate summary
        """
        print("=" * 70)
        print("Gate4 D-optimal Ablation Study (Confound-free)")
        print("=" * 70)
        print(f"Pool: seed={self.cfg.seed}, pool_seed={self.cfg.pool_seed}, size={self.cfg.pool_size}")
        print(f"Selection: D-optimal (1) vs Random ({len(self.cfg.random_seeds)} seeds)")
        print(f"Results: {self.results_base}")
        print("=" * 70)
        
        all_results = {}
        
        try:
            # 1. Setup
            print("\n[Phase 0] Setup and validation...")
            self._setup()
            
            # 2. Load artifacts
            print("\n[Phase 1] Loading dataset and Day3 artifacts...")
            self._load_artifacts()
            
            # 3. Generate shared pool
            print("\n[Phase 2] Generating shared pool...")
            self._generate_shared_pool()
            
            # 4. Track A selection (shared)
            print("\n[Phase 3] Track A selection...")
            self._run_track_a()
            
            # 5. D-optimal selection
            print("\n[Phase 4] D-optimal selection + evaluation...")
            dopt_result = self._run_dopt_selection()
            all_results['d_optimal'] = dopt_result
            
            # 6. Random selections
            for seed in self.cfg.random_seeds:
                print(f"\n[Phase 5.{seed}] Random selection (seed={seed}) + evaluation...")
                random_result = self._run_random_selection(seed)
                all_results[f'random_s{seed}'] = random_result
            
            # 7. Generate summary
            print("\n[Phase 6] Generating summary...")
            summary = self._generate_summary(all_results)
            
            return summary
            
        except Exception as e:
            print(f"\n❌ Ablation failed: {e}")
            traceback.print_exc()
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': self.timestamp,
            }
    
    def _setup(self):
        """Setup directories and RNG streams."""
        # Create results directory
        self.results_base.mkdir(parents=True, exist_ok=True)
        
        # Create RNG streams with pool_seed separation
        self.rng_streams = create_rng_streams(self.cfg.seed, pool_seed=self.cfg.pool_seed)
        
        print(f"  Results base: {self.results_base}")
        print(f"  RNG seed: {self.cfg.seed}, pool_seed: {self.cfg.pool_seed}")
    
    def _load_artifacts(self):
        """Load dataset and Day3/Gate1 artifacts."""
        cfg = self.cfg
        
        # Validate and load dataset
        dataset_path = Path(cfg.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        validate_dataset_lite(dataset_path)
        self.dataset = dict(np.load(dataset_path, allow_pickle=True))
        print(f"  ✅ Dataset loaded: {dataset_path}")
        
        # Load Day3 artifacts
        day3_path = Path(cfg.day3_source)
        if not day3_path.exists():
            day3_path = self.project_root / cfg.day3_source
        
        if not day3_path.exists():
            raise FileNotFoundError(f"Day3 source not found: {cfg.day3_source}")
        
        # Teacher support (npy format)
        teacher_support_path = day3_path / 'teacher_support.npy'
        if teacher_support_path.exists():
            self.teacher_support = np.load(teacher_support_path)
            print(f"  ✅ Teacher support loaded: {self.teacher_support.shape}")
        else:
            raise FileNotFoundError(f"teacher_support.npy not found in {day3_path}")
        
        # Load manifest to get Gate1 teacher_run_id
        manifest_path = day3_path / 'manifest.json'
        teacher_run_id = ''
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                day3_manifest = json.load(f)
            gate1_info = day3_manifest.get('gate1_artifacts', {})
            teacher_run_id = gate1_info.get('teacher_run_id', '')
            print(f"  Gate1 teacher_run_id: {teacher_run_id}")
        
        # Load teacher coefficients from Gate1 (csv format)
        gate1_coef_path = None
        if teacher_run_id:
            # Try to find Gate1 path from day3_source structure
            # day3_source: results/.../phase35/standardized/stable_core/n10/seed1/run_id/
            # gate1: results/.../gate1/standardized/esindy/n10/seed1/{teacher_run_id}/
            day3_parts = day3_path.parts
            if 'phase35' in day3_parts:
                idx = day3_parts.index('phase35')
                # Gate1 uses 'esindy' track, not 'stable_core'
                base_path = Path(*day3_parts[:idx])
                # Extract seed folder (e.g., 'seed1')
                seed_folder = None
                for part in day3_parts[idx+1:]:
                    if part.startswith('seed'):
                        seed_folder = part
                        break
                
                # Find n_train folder (e.g., 'n10')
                n_folder = None
                for part in day3_parts[idx+1:]:
                    if part.startswith('n') and part[1:].isdigit():
                        n_folder = part
                        break
                
                if seed_folder and n_folder:
                    gate1_coef_path = base_path / 'gate1' / 'standardized' / 'esindy' / n_folder / seed_folder / teacher_run_id / 'sindy_coefficients.csv'
        
        if gate1_coef_path and gate1_coef_path.exists():
            # Load sindy_coefficients.csv and convert to numpy array
            import csv
            with open(gate1_coef_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                rows = list(reader)
            # CSV format: feature, x_dot, x_ddot, theta_dot, theta_ddot
            n_features = len(rows)
            n_targets = 4
            self.teacher_coefficients = np.zeros((n_features, n_targets))
            for i, row in enumerate(rows):
                for j in range(n_targets):
                    self.teacher_coefficients[i, j] = float(row[j + 1])
            print(f"  ✅ Teacher coefficients loaded from Gate1: {self.teacher_coefficients.shape}")
            print(f"     Source: {gate1_coef_path}")
        else:
            # Fallback: compute from teacher_support (approximate)
            print(f"  ⚠️ Gate1 coefficients not found at: {gate1_coef_path}")
            print(f"     Using teacher_support as coefficient mask")
            self.teacher_coefficients = self.teacher_support.astype(float)
        
        # Feature/target names from library
        library = SINDyLibrary(config='gate0_min')
        self.feature_names = library.get_feature_names()
        self.target_names = DEFAULT_TARGET_NAMES
        
        # Norm stats: compute from dataset (no separate file)
        train_x = self.dataset['train_x'][:cfg.n_train]
        train_dx = self.dataset['train_dx'][:cfg.n_train]
        train_u = self.dataset['train_u'][:cfg.n_train]
        
        # Compute normalization statistics
        x_flat = train_x.reshape(-1, 4)
        dx_flat = train_dx.reshape(-1, 4)
        u_flat = train_u.reshape(-1, 1)
        
        self.norm_stats = {
            'state': {
                'mean': x_flat.mean(axis=0).tolist(),
                'std': x_flat.std(axis=0).tolist(),
            },
            'derivative_dx_savgol': {
                'mean': dx_flat.mean(axis=0).tolist(),
                'std': dx_flat.std(axis=0).tolist(),
            },
            'input': {
                'mean': float(u_flat.mean()),
                'std': float(u_flat.std()),
            },
        }
        self.dx_std = np.array(self.norm_stats['derivative_dx_savgol']['std'])
        print(f"  ✅ Norm stats computed from dataset")
        print(f"  dx_std: {self.dx_std}")
        
        # Load fragile pairs
        fragile_path = Path(cfg.fragile_pairs_source)
        if not fragile_path.exists():
            fragile_path = self.project_root / cfg.fragile_pairs_source
        
        if fragile_path.exists():
            with open(fragile_path, 'r') as f:
                fragile_data = json.load(f)
            if 'pairs' in fragile_data:
                self.fragile_pairs = [tuple(p) for p in fragile_data['pairs']]
            elif 'fragile_pairs' in fragile_data:
                self.fragile_pairs = [tuple(p) for p in fragile_data['fragile_pairs']]
            print(f"  ✅ Fragile pairs loaded: n={len(self.fragile_pairs)}")
        else:
            raise FileNotFoundError(f"Fragile pairs not found: {cfg.fragile_pairs_source}")
        
        # Load z_before (control reference) for aug_pure calculation
        z_before_path = day3_path / 'z_before.npy'
        if z_before_path.exists():
            self.z_before = np.load(z_before_path)
            # Compute hash for SSOT verification
            with open(z_before_path, 'rb') as f:
                self._z_before_sha = hashlib.sha256(f.read()).hexdigest()[:16]
            print(f"  ✅ z_before loaded: {self.z_before.shape}, SHA={self._z_before_sha}")
            
            # Extract z_before for fragile pairs
            self.fragile_z_before = []
            for feature_idx, target_idx in self.fragile_pairs:
                if feature_idx < self.z_before.shape[0] and target_idx < self.z_before.shape[1]:
                    self.fragile_z_before.append(self.z_before[feature_idx, target_idx])
            self.fragile_z_before = np.array(self.fragile_z_before)
            print(f"  ✅ fragile_z_before extracted: n={len(self.fragile_z_before)}")
        else:
            raise FileNotFoundError(f"z_before.npy not found in {day3_path}")
    
    def _generate_shared_pool(self):
        """Generate pool that will be shared across all selection methods."""
        cfg = self.cfg
        
        # Get training data
        train_x = self.dataset['train_x'][:cfg.n_train]
        train_params = self.dataset['train_params'][:cfg.n_train]
        train_u = self.dataset['train_u'][:cfg.n_train]
        
        print(f"  Training data: n={cfg.n_train}, T={train_x.shape[1]}")
        
        # Fit GMM sampler
        self.gmm_sampler = GMMProposalSampler(
            n_components=GATE3_CONFIG['gmm_n_components'],
            covariance_type=GATE3_CONFIG['gmm_covariance_type'],
            random_state=cfg.pool_seed,  # Use pool_seed for GMM fitting
        )
        self.gmm_sampler.fit(train_x, train_params)
        
        # Compute GMM fit hash for verification
        gmm_params = self.gmm_sampler.get_params_dict()
        gmm_json = json.dumps({
            'weights': gmm_params.get('weights', []),
            'means': gmm_params.get('means', []),
        }, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        gmm_fit_sha = hashlib.sha256(gmm_json.encode()).hexdigest()[:16]
        print(f"  GMM fit hash: {gmm_fit_sha}...")
        
        # Generate pool
        pool_generator = PoolGenerator(
            gmm_sampler=self.gmm_sampler,
            train_u=train_u,
            config=GATE3_CONFIG,
            fixed_physics=GATE3_CONFIG['fixed_physics'],
            seed=cfg.seed,
            rng=self.rng_streams['pool'],
        )
        
        self.pool = pool_generator.generate_pool(
            target_n_accept=cfg.pool_size,
            max_attempts=GATE3_CONFIG['max_pool_attempts'],
        )
        
        n_generated = len(self.pool['trajectories'])
        print(f"  ✅ Pool generated: {n_generated} trajectories")
        
        # Save pool hash for SSOT verification
        pool_hash_data = {
            'n_trajectories': n_generated,
            'ic_mean': self.pool['ic'].mean(axis=0).tolist(),
            'params_mean': self.pool['params'].mean(axis=0).tolist(),
        }
        pool_json = json.dumps(pool_hash_data, sort_keys=True)
        pool_sha = hashlib.sha256(pool_json.encode()).hexdigest()[:16]
        print(f"  Pool hash: {pool_sha}...")
        
        self._pool_sha = pool_sha
        self._gmm_fit_sha = gmm_fit_sha
    
    def _run_track_a(self):
        """Run Track A selection (shared across all selection methods)."""
        cfg = self.cfg
        
        self.track_a_result = track_a_selection(
            pool=self.pool,
            teacher_coefficients=self.teacher_coefficients,
            feature_names=self.feature_names,
            target_names=self.target_names,
            dx_std=self.dx_std,
            norm_stats=self.norm_stats,
            reject_ratio=cfg.reject_ratio,
            n_select=cfg.n_select,
            dynamics_target_indices=[1, 3],  # x_ddot, theta_ddot
        )
        
        n_passed = len(self.track_a_result['selected_indices'])
        n_pool = len(self.pool['trajectories'])
        print(f"  ✅ Track A: {n_passed}/{n_pool} passed ({n_passed/n_pool:.1%})")
    
    def _run_dopt_selection(self) -> AblationRunResult:
        """Run D-optimal selection on shared pool."""
        cfg = self.cfg
        
        run_id = f"{self.timestamp}_nogit_ablation_dopt"
        results_dir = self.results_base / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  run_id: {run_id}")
        
        try:
            # Create SINDy library
            library = SINDyLibrary(config='gate0_min')
            
            # D-optimal selection
            selected = track_b_dopt_selection(
                pool=self.pool,
                track_a_result=self.track_a_result,
                fragile_pairs=self.fragile_pairs,
                teacher_support=self.teacher_support,
                norm_stats=self.norm_stats,
                library=library,
                n_select=cfg.n_select,
                top_m_ratio=cfg.dopt_top_m_ratio,
                lambda_reg=cfg.dopt_lambda,
                use_teacher_intersection=cfg.dopt_use_teacher_intersection,
                dynamics_target_indices=[1, 3],
                pre_gate_mode=cfg.dopt_pre_gate_mode,
                alpha=cfg.dopt_alpha,
                gram_energy_mode=cfg.dopt_gram_energy_mode,
                trace_power=cfg.dopt_trace_power,
                selection_variant=cfg.dopt_selection_variant,
                topL_L=cfg.dopt_topL_L,
                rng=self.rng_streams['select'],
                results_dir=results_dir,
            )
            
            print(f"  Selected: {selected['stats']['n_selected']} trajectories")
            
            # E-SINDy evaluation
            metrics = self._evaluate_selection(selected, results_dir, 'd_optimal')
            
            # Save artifacts
            self._save_run_artifacts(run_id, results_dir, 'd_optimal', None, selected, metrics)
            
            return AblationRunResult(
                run_id=run_id,
                selection_method='d_optimal',
                selection_seed=None,
                results_dir=results_dir,
                status='completed',
                metrics=metrics,
            )
            
        except Exception as e:
            traceback.print_exc()
            return AblationRunResult(
                run_id=run_id,
                selection_method='d_optimal',
                selection_seed=None,
                results_dir=results_dir,
                status='failed',
                error_message=str(e),
            )
    
    def _run_random_selection(self, selection_seed: int) -> AblationRunResult:
        """Run random selection with specific seed on shared pool."""
        cfg = self.cfg
        
        run_id = f"{self.timestamp}_nogit_ablation_random_s{selection_seed}"
        results_dir = self.results_base / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  run_id: {run_id}")
        print(f"  selection_seed: {selection_seed}")
        
        try:
            # Random selection with explicit seed
            selected = random_selection_with_seed(
                pool=self.pool,
                track_a_result=self.track_a_result,
                n_select=cfg.n_select,
                selection_seed=selection_seed,
            )
            
            print(f"  Selected: {selected['stats']['n_selected']} trajectories")
            
            # E-SINDy evaluation
            metrics = self._evaluate_selection(selected, results_dir, 'random')
            
            # Save artifacts
            self._save_run_artifacts(run_id, results_dir, 'random', selection_seed, selected, metrics)
            
            return AblationRunResult(
                run_id=run_id,
                selection_method='random',
                selection_seed=selection_seed,
                results_dir=results_dir,
                status='completed',
                metrics=metrics,
            )
            
        except Exception as e:
            traceback.print_exc()
            return AblationRunResult(
                run_id=run_id,
                selection_method='random',
                selection_seed=selection_seed,
                results_dir=results_dir,
                status='failed',
                error_message=str(e),
            )
    
    def _evaluate_selection(
        self, 
        selected: Dict[str, Any], 
        results_dir: Path,
        method_name: str,
    ) -> Dict[str, Any]:
        """Evaluate selected trajectories with E-SINDy and compute SSOT metrics."""
        cfg = self.cfg
        
        # Get training data
        train_x = self.dataset['train_x'][:cfg.n_train]
        train_dx = self.dataset['train_dx'][:cfg.n_train]
        train_u = self.dataset['train_u'][:cfg.n_train]
        
        # Augmented data
        aug_x = selected['trajectories']
        aug_dx = selected['dx']
        aug_u = selected['u']
        
        print(f"  E-SINDy evaluation: {train_x.shape[0]} orig + {aug_x.shape[0]} aug")
        
        # Evaluate with E-SINDy
        eval_result = evaluate_with_esindy(
            train_x=train_x,
            train_dx=train_dx,
            train_u=train_u,
            aug_x=aug_x,
            aug_dx=aug_dx,
            aug_u=aug_u,
            feature_names=self.feature_names,
            target_names=self.target_names,
            bootstrap_B=cfg.bootstrap_B,
            threshold=cfg.threshold,
            seed=cfg.seed,
            tau_support=DEFAULT_TAU_SUPPORT,
            z0=DEFAULT_Z0,
            eps=DEFAULT_EPS,
        )
        
        # Compute metrics for fragile pairs
        z_after = eval_result['z_scores']
        
        # Extract z-scores for fragile pairs
        fragile_z_after = []
        for feature_idx, target_idx in self.fragile_pairs:
            if feature_idx < z_after.shape[0] and target_idx < z_after.shape[1]:
                fragile_z_after.append(z_after[feature_idx, target_idx])
        
        fragile_z_after = np.array(fragile_z_after)
        n_fragile_pairs_effective = len(fragile_z_after)
        
        # ============================================================
        # SSOT Metrics: aug_pure + CI + pass_level
        # ============================================================
        
        # Compute aug_pure = z_after - z_before (for fragile pairs)
        fragile_aug_pure = fragile_z_after - self.fragile_z_before[:n_fragile_pairs_effective]
        
        # Primary metric: median_aug_pure
        median_aug_pure = float(np.median(fragile_aug_pure)) if n_fragile_pairs_effective > 0 else None
        
        # Bootstrap CI for median_aug_pure
        ci_lower, ci_upper = self._compute_bootstrap_ci(
            fragile_aug_pure, 
            n_bootstrap=cfg.ci_bootstrap_B,
            alpha=cfg.ci_alpha,
            seed=cfg.seed,
        )
        
        # Pass level classification (Gate3 SSOT definition)
        pass_level = self._classify_pass_level(median_aug_pure, ci_lower, cfg.gate2_ceiling)
        
        # ============================================================
        # Build metrics dict (SSOT-aligned)
        # ============================================================
        
        metrics = {
            'method': method_name,
            
            # Primary SSOT metrics
            'median_aug_pure': median_aug_pure,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pass_level': pass_level,
            
            # Control reference (for SSOT chain verification)
            'z_before_sha': self._z_before_sha,
            
            # Secondary metrics (for backward compatibility)
            'z_after_median': float(np.median(fragile_z_after)) if n_fragile_pairs_effective > 0 else None,
            'z_after_mean': float(np.mean(fragile_z_after)) if n_fragile_pairs_effective > 0 else None,
            'z_after_std': float(np.std(fragile_z_after)) if n_fragile_pairs_effective > 0 else None,
            
            # Promotion metrics
            'n_above_z0': int((fragile_z_after >= DEFAULT_Z0).sum()) if n_fragile_pairs_effective > 0 else 0,
            'promotion_rate': float((fragile_z_after >= DEFAULT_Z0).sum() / n_fragile_pairs_effective) if n_fragile_pairs_effective > 0 else 0.0,
            
            # Pair tracking
            'n_pairs_loaded': len(self.fragile_pairs),
            'n_pairs_effective': n_fragile_pairs_effective,
            
            # Sample counts
            'n_total_samples': eval_result['n_total'],
            'n_original': eval_result['n_original'],
            'n_augmented': eval_result['n_augmented'],
            
            # Config parameters (for audit)
            'bootstrap_B': cfg.bootstrap_B,
            'ci_bootstrap_B': cfg.ci_bootstrap_B,
            'ci_alpha': cfg.ci_alpha,
            'gate2_ceiling': cfg.gate2_ceiling,
        }
        
        # Add E-SINDy support summary (term-level)
        support_mask = eval_result['support_mask']
        stable_mask = eval_result['stable_core_mask']
        fragile_mask = eval_result['fragile_pool_mask']
        metrics['support_terms_total'] = int(support_mask.sum())
        metrics['stable_terms_total'] = int(stable_mask.sum())
        metrics['fragile_terms_total'] = int(fragile_mask.sum())
        
        # Save arrays
        np.save(results_dir / 'z_after.npy', z_after)
        np.save(results_dir / 'fragile_z_after.npy', fragile_z_after)
        np.save(results_dir / 'fragile_aug_pure.npy', fragile_aug_pure)
        
        # Print summary
        print(f"  Support terms: {metrics['support_terms_total']}, Stable: {metrics['stable_terms_total']}, Fragile: {metrics['fragile_terms_total']}")
        print(f"  median_aug_pure: {median_aug_pure:.3f}, CI: [{ci_lower:.3f}, {ci_upper:.3f}], pass_level: {pass_level}")
        print(f"  z_after_median: {metrics['z_after_median']:.3f}, promotion_rate: {metrics['promotion_rate']:.1%}")
        print(f"  ✅ Saved: {results_dir}")
        
        return metrics
    
    def _compute_bootstrap_ci(
        self,
        data: np.ndarray,
        n_bootstrap: int,
        alpha: float,
        seed: int,
    ) -> Tuple[float, float]:
        """Compute bootstrap percentile CI for median."""
        if len(data) == 0:
            return (None, None)
        
        rng = np.random.default_rng(seed)
        boot_medians = []
        
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            boot_medians.append(np.median(sample))
        
        boot_medians = np.array(boot_medians)
        ci_lower = float(np.percentile(boot_medians, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
        
        return (ci_lower, ci_upper)
    
    def _classify_pass_level(
        self,
        median_aug_pure: Optional[float],
        ci_lower: Optional[float],
        gate2_ceiling: float,
    ) -> str:
        """Classify pass level based on Gate3 SSOT definition."""
        if median_aug_pure is None or ci_lower is None:
            return "NULL"
        
        if ci_lower > gate2_ceiling:
            return "CEILING_BREAK"
        elif ci_lower > 0:
            return "STRONG_PASS"
        elif median_aug_pure > 0:
            return "SOFT_PASS"
        else:
            return "NULL"
    
    def _save_run_artifacts(
        self,
        run_id: str,
        results_dir: Path,
        method: str,
        selection_seed: Optional[int],
        selected: Dict[str, Any],
        metrics: Dict[str, Any],
    ):
        """Save artifacts for a single run."""
        cfg = self.cfg
        
        # Manifest (SSOT-aligned)
        manifest = {
            'run_id': run_id,
            'experiment': 'gate4_dopt_ablation',
            'design': 'confound_free',
            'timestamp': self.timestamp,
            
            'fixed_conditions': {
                'seed': cfg.seed,
                'pool_seed': cfg.pool_seed,
                'pool_size': cfg.pool_size,
                'pool_sha': self._pool_sha,
                'gmm_fit_sha': self._gmm_fit_sha,
                'z_before_sha': self._z_before_sha,  # Control reference hash
            },
            
            'selection': {
                'method': method,
                'selection_seed': selection_seed,
                'n_select': cfg.n_select,
                'reject_ratio': cfg.reject_ratio,
            },
            
            'evaluation': {
                'bootstrap_B': cfg.bootstrap_B,
                'threshold': cfg.threshold,
                'n_pairs_loaded': len(self.fragile_pairs),
                # SSOT alignment
                'ci_bootstrap_B': cfg.ci_bootstrap_B,
                'ci_alpha': cfg.ci_alpha,
                'gate2_ceiling': cfg.gate2_ceiling,
                'primary_metric': 'median_aug_pure',
                'pass_level_rule': {
                    'CEILING_BREAK': 'ci_lower > gate2_ceiling',
                    'STRONG_PASS': 'ci_lower > 0',
                    'SOFT_PASS': 'median > 0',
                    'NULL': 'otherwise',
                },
            },
            
            'artifacts': [
                'manifest.json',
                'metrics.json',
                'z_after.npy',
                'fragile_z_after.npy',
                'fragile_aug_pure.npy',
                'selected_indices.npy',
            ],
        }
        
        with open(results_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Metrics
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Selected indices
        np.save(results_dir / 'selected_indices.npy', selected['original_indices'])
    
    def _generate_summary(self, all_results: Dict[str, AblationRunResult]) -> Dict[str, Any]:
        """Generate ablation summary comparing D-optimal vs Random (SSOT-aligned)."""
        cfg = self.cfg
        
        # Collect metrics
        dopt_result = all_results['d_optimal']
        random_results = [all_results[f'random_s{s}'] for s in cfg.random_seeds]
        
        # D-optimal metrics
        dopt_metrics = dopt_result.metrics if dopt_result.status == 'completed' else None
        
        # Random metrics (aggregate)
        random_metrics_list = [r.metrics for r in random_results if r.status == 'completed' and r.metrics]
        
        # Build runs list (SSOT schema: runs as list)
        runs_list = []
        
        # D-optimal run
        if dopt_result.status == 'completed' and dopt_metrics:
            runs_list.append({
                'method': 'd_optimal',
                'selection_seed': None,
                'run_id': dopt_result.run_id,
                'status': dopt_result.status,
                # Primary SSOT metrics
                'median_aug_pure': dopt_metrics.get('median_aug_pure'),
                'ci_lower': dopt_metrics.get('ci_lower'),
                'ci_upper': dopt_metrics.get('ci_upper'),
                'pass_level': dopt_metrics.get('pass_level'),
                # Secondary metrics
                'z_after_median': dopt_metrics.get('z_after_median'),
                'promotion_rate': dopt_metrics.get('promotion_rate'),
                'n_above_z0': dopt_metrics.get('n_above_z0'),
            })
        
        # Random runs
        for r in random_results:
            if r.status == 'completed' and r.metrics:
                runs_list.append({
                    'method': 'random',
                    'selection_seed': r.selection_seed,
                    'run_id': r.run_id,
                    'status': r.status,
                    # Primary SSOT metrics
                    'median_aug_pure': r.metrics.get('median_aug_pure'),
                    'ci_lower': r.metrics.get('ci_lower'),
                    'ci_upper': r.metrics.get('ci_upper'),
                    'pass_level': r.metrics.get('pass_level'),
                    # Secondary metrics
                    'z_after_median': r.metrics.get('z_after_median'),
                    'promotion_rate': r.metrics.get('promotion_rate'),
                    'n_above_z0': r.metrics.get('n_above_z0'),
                })
        
        # Compute aggregate statistics
        if random_metrics_list:
            random_aug_pures = [m['median_aug_pure'] for m in random_metrics_list if m.get('median_aug_pure') is not None]
            random_z_medians = [m['z_after_median'] for m in random_metrics_list if m.get('z_after_median') is not None]
            random_promotion_rates = [m['promotion_rate'] for m in random_metrics_list if m.get('promotion_rate') is not None]
            
            # Pass level distribution
            pass_level_counts = {'NULL': 0, 'SOFT_PASS': 0, 'STRONG_PASS': 0, 'CEILING_BREAK': 0}
            for m in random_metrics_list:
                pl = m.get('pass_level', 'NULL')
                if pl in pass_level_counts:
                    pass_level_counts[pl] += 1
        else:
            random_aug_pures = []
            random_z_medians = []
            random_promotion_rates = []
            pass_level_counts = {'NULL': 0, 'SOFT_PASS': 0, 'STRONG_PASS': 0, 'CEILING_BREAK': 0}
        
        # Build summary (SSOT-aligned schema)
        summary = {
            'experiment_type': 'gate4_dopt_ablation',
            'design': 'confound_free',
            'timestamp': self.timestamp,
            
            'fixed_conditions': {
                'seed': cfg.seed,
                'pool_seed': cfg.pool_seed,
                'pool_size': cfg.pool_size,
                'n_select': cfg.n_select,
                'pool_sha': self._pool_sha,
                'gmm_fit_sha': self._gmm_fit_sha,
                'z_before_sha': self._z_before_sha,
                'bootstrap_B': cfg.bootstrap_B,
                'ci_bootstrap_B': cfg.ci_bootstrap_B,
                'ci_alpha': cfg.ci_alpha,
                'gate2_ceiling': cfg.gate2_ceiling,
            },
            
            'runs': runs_list,  # SSOT: list instead of dict
            
            'comparison': {
                'd_optimal': {
                    # Primary SSOT metrics
                    'median_aug_pure': dopt_metrics.get('median_aug_pure') if dopt_metrics else None,
                    'ci_lower': dopt_metrics.get('ci_lower') if dopt_metrics else None,
                    'ci_upper': dopt_metrics.get('ci_upper') if dopt_metrics else None,
                    'pass_level': dopt_metrics.get('pass_level') if dopt_metrics else None,
                    # Secondary
                    'z_after_median': dopt_metrics.get('z_after_median') if dopt_metrics else None,
                    'promotion_rate': dopt_metrics.get('promotion_rate') if dopt_metrics else None,
                },
                'random': {
                    # Primary SSOT metrics
                    'median_aug_pure_mean': float(np.mean(random_aug_pures)) if random_aug_pures else None,
                    'median_aug_pure_std': float(np.std(random_aug_pures)) if random_aug_pures else None,
                    'pass_level_distribution': pass_level_counts,
                    # Secondary
                    'z_after_median_mean': float(np.mean(random_z_medians)) if random_z_medians else None,
                    'z_after_median_std': float(np.std(random_z_medians)) if random_z_medians else None,
                    'promotion_rate_mean': float(np.mean(random_promotion_rates)) if random_promotion_rates else None,
                    'promotion_rate_std': float(np.std(random_promotion_rates)) if random_promotion_rates else None,
                    'n_completed': len(random_metrics_list),
                },
            },
            
            'verdict': self._compute_verdict(dopt_metrics, random_metrics_list),
        }
        
        # Save summary
        summary_path = self.results_base / 'ablation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{'='*70}")
        print("ABLATION SUMMARY (SSOT-aligned)")
        print(f"{'='*70}")
        print(f"[Primary Metric: median_aug_pure]")
        print(f"  D-optimal: {summary['comparison']['d_optimal']['median_aug_pure']:.3f}" if summary['comparison']['d_optimal']['median_aug_pure'] else "  D-optimal: N/A")
        print(f"  Random:    {summary['comparison']['random']['median_aug_pure_mean']:.3f} ± {summary['comparison']['random']['median_aug_pure_std']:.3f}" if summary['comparison']['random']['median_aug_pure_mean'] else "  Random: N/A")
        print(f"\n[Pass Level]")
        print(f"  D-optimal: {summary['comparison']['d_optimal']['pass_level']}")
        print(f"  Random distribution: {pass_level_counts}")
        print(f"\n[Secondary: z_after_median]")
        print(f"  D-optimal: {summary['comparison']['d_optimal']['z_after_median']:.3f}" if summary['comparison']['d_optimal']['z_after_median'] else "  D-optimal: N/A")
        print(f"  Random:    {summary['comparison']['random']['z_after_median_mean']:.3f} ± {summary['comparison']['random']['z_after_median_std']:.3f}" if summary['comparison']['random']['z_after_median_mean'] else "  Random: N/A")
        print(f"\n[Verdict]")
        print(f"  {summary['verdict']['conclusion']}")
        print(f"{'='*70}")
        print(f"\n✅ Summary saved: {summary_path}")
        
        return summary
    
    def _compute_verdict(
        self, 
        dopt_metrics: Optional[Dict], 
        random_metrics_list: List[Dict],
    ) -> Dict[str, Any]:
        """Compute verdict comparing D-optimal vs Random (SSOT-aligned: median_aug_pure)."""
        if not dopt_metrics or not random_metrics_list:
            return {
                'conclusion': 'INCOMPLETE',
                'reason': 'Missing metrics',
            }
        
        # Primary metric: median_aug_pure (SSOT)
        dopt_aug_pure = dopt_metrics.get('median_aug_pure')
        dopt_pass_level = dopt_metrics.get('pass_level', 'NULL')
        
        random_aug_pures = [m.get('median_aug_pure') for m in random_metrics_list if m.get('median_aug_pure') is not None]
        random_pass_levels = [m.get('pass_level', 'NULL') for m in random_metrics_list]
        
        if not random_aug_pures or dopt_aug_pure is None:
            return {
                'conclusion': 'INCOMPLETE',
                'reason': 'Missing median_aug_pure data',
            }
        
        random_mean = np.mean(random_aug_pures)
        random_std = np.std(random_aug_pures)
        
        # D-optimal advantage in median_aug_pure
        dopt_advantage = dopt_aug_pure - random_mean
        
        # Heuristic comparison (NOT a statistical test)
        # Note: With limited random samples, formal significance testing is not meaningful
        advantage_outside_1std = abs(dopt_advantage) > random_std if random_std > 0 else dopt_advantage != 0
        
        # Pass level ordering: NULL < SOFT_PASS < STRONG_PASS < CEILING_BREAK
        pass_level_order = {'NULL': 0, 'SOFT_PASS': 1, 'STRONG_PASS': 2, 'CEILING_BREAK': 3}
        dopt_pl_score = pass_level_order.get(dopt_pass_level, 0)
        random_max_pl_score = max(pass_level_order.get(pl, 0) for pl in random_pass_levels)
        
        # Determine conclusion
        if dopt_aug_pure > random_mean and advantage_outside_1std:
            conclusion = 'DOPT_ADVANTAGE_OBSERVED'
        elif dopt_aug_pure < random_mean and advantage_outside_1std:
            conclusion = 'RANDOM_ADVANTAGE_OBSERVED'
        else:
            conclusion = 'NO_CLEAR_DIFFERENCE'
        
        # Build verdict
        return {
            'conclusion': conclusion,
            # Primary metric comparison
            'dopt_median_aug_pure': float(dopt_aug_pure),
            'random_median_aug_pure_mean': float(random_mean),
            'random_median_aug_pure_std': float(random_std),
            'dopt_advantage': float(dopt_advantage),
            'advantage_outside_1std': bool(advantage_outside_1std),
            # Pass level comparison
            'dopt_pass_level': dopt_pass_level,
            'dopt_pass_level_higher_than_all_random': dopt_pl_score > random_max_pl_score,
            # Note
            'note': f'Heuristic comparison only. With n={len(random_metrics_list)} random samples, formal significance testing is not applicable.',
        }


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Gate4 D-optimal Ablation Study (Confound-free, SSOT-aligned)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (basic - 3 random seeds):
  python experiments/run_gate4_ablation.py ^
    --dataset_path data/cartpole/cartpole_ood_v1/dataset.npz ^
    --day3_source results/cartpole_ood_v1/phase35/standardized/stable_core/n10/seed1/...  ^
    --fragile_pairs_source results/.../fragile_pairs.json ^
    --seed 1 --pool_seed 42 --pool_size 2000

Example (extended - 10 random seeds):
  python experiments/run_gate4_ablation.py ^
    --dataset_path data/cartpole/cartpole_ood_v1/dataset.npz ^
    --day3_source results/.../day3_run ^
    --fragile_pairs_source results/.../fragile_pairs.json ^
    --seed 1 --pool_seed 42 --pool_size 2000 ^
    --random_seeds 0 1 2 3 4 5 6 7 8 9

Confound-free design:
  - Same pool (pool_seed=42, size=2000)
  - Same control reference (z_before from Day3)
  - Different selection only (D-optimal vs Random)
  - Primary metric: median_aug_pure (SSOT-aligned)
        """
    )
    
    # Required
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset.npz')
    parser.add_argument('--day3_source', type=str, required=True,
                        help='Path to Day3 results (teacher_support, z_before, etc.)')
    parser.add_argument('--fragile_pairs_source', type=str, required=True,
                        help='Path to fragile_pairs JSON (for D-optimal)')
    
    # Pool settings (fixed for confound-free)
    parser.add_argument('--seed', type=int, default=1,
                        help='Base seed (default: 1)')
    parser.add_argument('--pool_seed', type=int, default=42,
                        help='Pool generation seed (default: 42)')
    parser.add_argument('--pool_size', type=int, default=2000,
                        help='Target pool size (default: 2000)')
    parser.add_argument('--n_select', type=int, default=200,
                        help='Number to select (default: 200)')
    parser.add_argument('--n_train', type=int, default=10,
                        help='Number of training trajectories (default: 10)')
    
    # Random seeds (expandable for better coverage)
    parser.add_argument('--random_seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='Seeds for random selection (default: 0 1 2). Use more for better coverage: --random_seeds 0 1 2 3 4 5 6 7 8 9')
    
    # Output
    parser.add_argument('--results_base', type=str,
                        default='results/cartpole_ood_v1/gate4/ablation/d_optimal_vs_random',
                        help='Base directory for results')
    
    # E-SINDy settings
    parser.add_argument('--bootstrap_B', type=int, default=100,
                        help='Bootstrap iterations for E-SINDy (default: 100)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='STLSQ threshold (default: 0.05)')
    
    # SSOT Metrics settings (Gate3/4 aligned)
    parser.add_argument('--ci_bootstrap_B', type=int, default=2000,
                        help='Bootstrap samples for CI computation (default: 2000)')
    parser.add_argument('--ci_alpha', type=float, default=0.05,
                        help='CI significance level (default: 0.05 for 95%% CI)')
    parser.add_argument('--gate2_ceiling', type=float, default=0.058,
                        help='Gate2 ceiling for pass_level classification (default: 0.058)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = AblationConfig(
        seed=args.seed,
        pool_seed=args.pool_seed,
        pool_size=args.pool_size,
        n_select=args.n_select,
        n_train=args.n_train,
        random_seeds=args.random_seeds,
        dataset_path=args.dataset_path,
        day3_source=args.day3_source,
        fragile_pairs_source=args.fragile_pairs_source,
        results_base=args.results_base,
        bootstrap_B=args.bootstrap_B,
        threshold=args.threshold,
        # SSOT parameters
        ci_bootstrap_B=args.ci_bootstrap_B,
        ci_alpha=args.ci_alpha,
        gate2_ceiling=args.gate2_ceiling,
    )
    
    runner = AblationRunner(config)
    summary = runner.run_all()
    
    print("\n" + "=" * 70)
    print("Ablation Study Complete")
    print("=" * 70)
    
    if summary.get('status') == 'failed':
        print(f"❌ Failed: {summary.get('error')}")
        sys.exit(1)
    
    print(f"Summary: {config.results_base}/ablation_summary.json")
    print("\nNext steps:")
    print("  1. Review ablation_summary.json")
    print("  2. Compare D-optimal vs Random metrics")
    print("  3. Update Paper1 Ablation section")
    

if __name__ == '__main__':
    main()