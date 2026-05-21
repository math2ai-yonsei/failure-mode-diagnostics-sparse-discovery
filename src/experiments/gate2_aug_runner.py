"""
Gate2: Augmentation Runner

Composes augmentation with Gate1 E-SINDy pipeline.
Ensures fair comparison by:
    1. Using same train subset idx as Gate1 (seed-based)
    2. Augmenting train only (val/test unchanged)
    3. Using same norm_stats (no recomputation)

Pipeline:
    1. Load original dataset
    2. Get Gate1 train subset idx (seed-based)
    3. Apply augmentation to train subset only
    4. Merge original + augmented → temp dataset
    5. Call Gate1ESINDyRunner with dataset_path_override
    6. Compute delta vs Gate1 baseline
    7. Save Gate2-specific artifacts

Usage:
    from src.experiments.gate2_aug_runner import Gate2AugRunner, Gate2Config
    
    config = Gate2Config(n_train=10, seed=0, aug_ratio=1.0)
    runner = Gate2AugRunner(config)
    result = runner.run()
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import tempfile
import shutil

# Contracts (SSOT)
from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.contracts.plot_style import create_figure, save_figure, get_color

# Augmentation
from src.augmentation import PhysicsAugmentor
from src.augmentation.physics_augmentor import PhysicsAugmentorConfig
from src.augmentation.base import get_train_subset_idx

# Gate1 Runner (composition target)
from src.experiments.gate1_esindy_runner import Gate1ESINDyRunner, Gate1Config

# Normalization
from src.data.normalization import load_norm_stats

# Library
from src.sindy.library import get_derivative_key


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Gate2Config:
    """Gate2 Augmentation experiment configuration."""
    
    # Dataset
    dataset_version: str = 'cartpole_ood_v1'
    system: str = 'cartpole'
    
    # Experiment settings (same as Gate1)
    n_train: int = 10
    seed: int = 0
    track: str = 'standardized'
    note: str = 'aug'
    
    # Augmentation settings
    aug_method: str = 'physics_resim'
    aug_ratio: float = 1.0  # n_aug = n_train * aug_ratio
    aug_seed: Optional[int] = None  # If None, use experiment seed
    jitter_mode: str = 'both'  # 'ic_only', 'param_only', 'both', 'random'
    ic_std_scale: float = 1.0
    param_rel_std_scale: float = 1.0
    
    # E-SINDy settings (passed to Gate1)
    n_bootstrap: int = 20
    thresholds: List[float] = field(default_factory=lambda: [
        0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2
    ])
    library_config: str = 'gate0_min'
    ridge_alpha: float = 0.0
    final_fit_split: str = 'train'
    
    # Gate1 baseline reference (for delta computation)
    gate1_baseline_run_id: Optional[str] = None
    
    # Fixed
    gate: str = 'gate2'
    method: str = 'physics_resim'  # Will be set based on aug_method
    
    def __post_init__(self):
        if self.aug_seed is None:
            self.aug_seed = self.seed
        self.method = self.aug_method
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Gate2Config':
        """Create config from dictionary."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Gate2Config':
        """Load config from YAML file."""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


# =============================================================================
# Gate2 Runner
# =============================================================================

class Gate2AugRunner:
    """
    Gate2 Augmentation Runner.
    
    Workflow:
        1. Preflight: Validate dataset
        2. Load data: Get train subset using Gate1-compatible logic
        3. Augment: Apply physics augmentation to train only
        4. Create temp dataset: Merge original + augmented
        5. Run Gate1: Call Gate1ESINDyRunner with augmented data
        6. Compare: Compute delta vs Gate1 baseline
        7. Save: Gate2-specific artifacts
        8. Generate CP: Context Packet
    """
    
    def __init__(self, config: Gate2Config):
        self.config = config
        self.run_id = paths.generate_run_id(config.note)
        
        # Results directory (Gate2 path structure)
        self.results_dir = paths.get_results_dir(
            dataset_version=config.dataset_version,
            gate=config.gate,
            track=config.track,
            method=config.method,
            n_train=config.n_train,
            seed=config.seed,
            run_id=self.run_id
        )
        
        # Dataset path
        self.dataset_path = paths.get_dataset_path(
            config.dataset_version, config.system
        )
        
        # Derivative key based on track
        self.derivative_key = get_derivative_key(config.track)
        
        # State containers
        self.original_data: Dict[str, np.ndarray] = {}
        self.augmented_data: Dict[str, np.ndarray] = {}
        self.aug_result = None
        self.gate1_result: Dict = {}
        self.metrics: Dict = {}
        self.temp_dataset_path: Optional[Path] = None
        self._temp_dir: Optional[Path] = None
    
    def run(self) -> Dict:
        """Execute the full Gate2 pipeline."""
        print("=" * 70)
        print(f"  Gate2 Augmentation Runner: {self.run_id}")
        print("=" * 70)
        
        try:
            # 1. Preflight
            self._preflight()
            
            # 2. Load original data
            self._load_data()
            
            # 3. Apply augmentation
            self._augment_data()
            
            # 4. Create temporary augmented dataset
            self._create_temp_dataset()
            
            # 5. Run Gate1 E-SINDy on augmented data
            self._run_gate1()
            
            # 6. Compute comparison metrics
            self._compute_comparison()
            
            # 7. Generate figures
            self._generate_figures()
            
            # 8. Save artifacts
            self._save_artifacts()
            
            # 9. Generate Context Packet
            self._generate_context_packet()
            
            # Cleanup
            self._cleanup()
            
            print("\n" + "=" * 70)
            print(f"  ✅ Gate2 Complete: {self.run_id}")
            print(f"  Results: {self.results_dir}")
            print("=" * 70)
            
            return {
                'success': True,
                'run_id': self.run_id,
                'results_dir': str(self.results_dir),
                'metrics': self.metrics,
            }
            
        except Exception as e:
            self._cleanup()
            print(f"\n❌ Gate2 Failed: {e}")
            raise
    
    # =========================================================================
    # Pipeline Steps
    # =========================================================================
    
    def _preflight(self) -> None:
        """Step 1: Preflight validation."""
        print("\n[1/9] Preflight Validation")
        print("-" * 50)
        
        validate_dataset_lite(self.dataset_path)
        
        print(f"  Dataset: {self.dataset_path.name}")
        print(f"  Track: {self.config.track}")
        print(f"  Aug method: {self.config.aug_method}")
        print(f"  Aug ratio: {self.config.aug_ratio}")
    
    def _load_data(self) -> None:
        """Step 2: Load dataset and apply train subset selection."""
        print("\n[2/9] Loading Dataset")
        print("-" * 50)
        
        data = np.load(self.dataset_path)
        
        # Store time info
        self.original_data['t'] = data['t']
        self.original_data['dt'] = float(data['dt'])
        self.T = len(self.original_data['t'])
        
        # Load all splits
        for split in ['train', 'val', 'test']:
            self.original_data[f'{split}_x'] = data[f'{split}_x']
            self.original_data[f'{split}_u'] = data[f'{split}_u']
            self.original_data[f'{split}_params'] = data[f'{split}_params']
            self.original_data[f'{split}_cond_id'] = data[f'{split}_cond_id']
            
            # Select derivative based on track
            if self.config.track == 'standardized':
                dx_key = f'{split}_dx_savgol'
            else:
                dx_key = f'{split}_dx'
            self.original_data[f'{split}_dx'] = data[dx_key]
        
        # Get train subset idx (MUST match Gate1 logic exactly)
        n_total_train = self.original_data['train_x'].shape[0]
        self.train_idx = get_train_subset_idx(
            n_total=n_total_train,
            n_train=self.config.n_train,
            seed=self.config.seed
        )
        
        print(f"  Total train: {n_total_train}")
        print(f"  Selected train: {len(self.train_idx)} (seed={self.config.seed})")
        print(f"  Val: {self.original_data['val_x'].shape[0]}")
        print(f"  Test: {self.original_data['test_x'].shape[0]}")
        print(f"  T: {self.T}")
    
    def _augment_data(self) -> None:
        """Step 3: Apply augmentation to train subset."""
        print("\n[3/9] Augmenting Data")
        print("-" * 50)
        
        # Get train subset
        train_x = self.original_data['train_x'][self.train_idx]
        train_u = self.original_data['train_u'][self.train_idx]
        train_params = self.original_data['train_params'][self.train_idx]
        
        # Create augmentor config
        aug_config = PhysicsAugmentorConfig(
            aug_ratio=self.config.aug_ratio,
            seed=self.config.aug_seed,
            dt=self.original_data['dt'],
            T=self.T,
            jitter_mode=self.config.jitter_mode,
        )
        
        # Scale jitter if specified
        if self.config.ic_std_scale != 1.0:
            for k in aug_config.ic_jitter_std:
                aug_config.ic_jitter_std[k] *= self.config.ic_std_scale
        
        if self.config.param_rel_std_scale != 1.0:
            for k in aug_config.param_jitter_rel_std:
                if aug_config.param_jitter_rel_std[k] > 0:
                    aug_config.param_jitter_rel_std[k] *= self.config.param_rel_std_scale
        
        # Create augmentor and augment
        augmentor = PhysicsAugmentor(aug_config)
        self.aug_result = augmentor.augment(train_x, train_u, train_params)
        
        print(f"  Original train: {train_x.shape[0]}")
        print(f"  Augmented: {self.aug_result.n_augmented}")
        print(f"  Total after merge: {train_x.shape[0] + self.aug_result.n_augmented}")
        
        # Report aug types
        from collections import Counter
        type_counts = Counter(self.aug_result.aug_type)
        for atype, count in type_counts.items():
            print(f"    - {atype}: {count}")
    
    def _create_temp_dataset(self) -> None:
        """Step 4: Create temporary augmented dataset.npz."""
        print("\n[4/9] Creating Augmented Dataset")
        print("-" * 50)
        
        # Create temp directory
        self._temp_dir = Path(tempfile.mkdtemp(prefix='gate2_'))
        self.temp_dataset_path = self._temp_dir / 'dataset.npz'
        
        # Get original train subset
        train_x = self.original_data['train_x'][self.train_idx]
        train_u = self.original_data['train_u'][self.train_idx]
        train_dx = self.original_data['train_dx'][self.train_idx]
        train_params = self.original_data['train_params'][self.train_idx]
        train_cond_id = self.original_data['train_cond_id'][self.train_idx]
        
        # Merge original + augmented
        merged_x = np.concatenate([train_x, self.aug_result.x], axis=0)
        merged_u = np.concatenate([train_u, self.aug_result.u], axis=0)
        merged_dx = np.concatenate([train_dx, self.aug_result.dx], axis=0)
        merged_params = np.concatenate([train_params, self.aug_result.params], axis=0)
        
        # For augmented trajectories, assign new cond_id (max + 1 + index)
        # This avoids collision with any existing cond_id
        all_cond_ids = np.concatenate([
            train_cond_id,
            self.original_data['val_cond_id'],
            self.original_data['test_cond_id']
        ])
        max_cond_id = int(np.max(all_cond_ids))
        n_aug = self.aug_result.n_augmented
        aug_cond_id = max_cond_id + 1 + np.arange(n_aug)
        merged_cond_id = np.concatenate([train_cond_id, aug_cond_id], axis=0)
        
        # Build dataset dict
        dataset_dict = {
            't': self.original_data['t'],
            'dt': self.original_data['dt'],
            
            # Merged train
            'train_x': merged_x,
            'train_u': merged_u,
            'train_dx': merged_dx,
            'train_params': merged_params,
            'train_cond_id': merged_cond_id,
            
            # Val/Test unchanged
            'val_x': self.original_data['val_x'],
            'val_u': self.original_data['val_u'],
            'val_dx': self.original_data['val_dx'],
            'val_params': self.original_data['val_params'],
            'val_cond_id': self.original_data['val_cond_id'],
            
            'test_x': self.original_data['test_x'],
            'test_u': self.original_data['test_u'],
            'test_dx': self.original_data['test_dx'],
            'test_params': self.original_data['test_params'],
            'test_cond_id': self.original_data['test_cond_id'],
        }
        
        # Add savgol dx if standardized track
        if self.config.track == 'standardized':
            # For augmented data, the dx IS the analytic (which should match savgol closely)
            # But to maintain consistency, we reuse the dx as dx_savgol
            dataset_dict['train_dx_savgol'] = merged_dx
            dataset_dict['val_dx_savgol'] = self.original_data['val_dx']
            dataset_dict['test_dx_savgol'] = self.original_data['test_dx']
        
        # Save
        np.savez(self.temp_dataset_path, **dataset_dict)
        
        # Validate temp dataset (defensive programming)
        validate_dataset_lite(self.temp_dataset_path)
        
        print(f"  Temp dataset: {self.temp_dataset_path}")
        print(f"  Merged train shape: {merged_x.shape}")
    
    def _run_gate1(self) -> None:
        """Step 5: Run Gate1 E-SINDy on augmented data."""
        print("\n[5/9] Running Gate1 E-SINDy")
        print("-" * 50)
        
        # Create Gate1 config
        gate1_config = Gate1Config(
            dataset_version=self.config.dataset_version,
            system=self.config.system,
            n_train=999999,  # Use all (already subsampled+augmented)
            seed=self.config.seed,
            track=self.config.track,
            note=f"gate2_{self.config.note}",
            n_bootstrap=self.config.n_bootstrap,
            thresholds=self.config.thresholds,
            library_config=self.config.library_config,
            ridge_alpha=self.config.ridge_alpha,
            final_fit_split=self.config.final_fit_split,
        )
        
        # Create runner with dataset override
        runner = Gate1ESINDyRunner(gate1_config)
        
        # Override dataset path
        runner.dataset_path = self.temp_dataset_path
        
        # Override results dir to be under Gate2
        runner.results_dir = self.results_dir
        runner.run_id = self.run_id
        
        # Run (skipping preflight since we already validated)
        print("  Running Gate1 pipeline on augmented data...")
        
        # We'll manually run steps to have more control
        runner._load_data()
        runner._normalize_data()
        runner._build_features()
        runner._threshold_sweep()
        runner._select_best_threshold()
        runner._fit_final_esindy()
        runner._compute_metrics()
        runner._generate_figures()
        
        # Store results
        self.gate1_result = {
            'metrics': runner.metrics,
            'best_threshold': runner.best_threshold_result,
            'ensemble': runner.ensemble,
            'sweep_results': runner.sweep_results,
        }
        
        print(f"  Best threshold: {runner.best_threshold_result['threshold']}")
        print(f"  Test R²: {runner.metrics['splits']['test']['r2_mean']:.4f}")
    
    def _compute_comparison(self) -> None:
        """Step 6: Compute comparison with Gate1 baseline."""
        print("\n[6/9] Computing Comparison")
        print("-" * 50)
        
        # Build metrics
        self.metrics = {
            'run_id': self.run_id,
            'gate': 'gate2',
            'config': {
                'dataset_version': self.config.dataset_version,
                'track': self.config.track,
                'n_train_original': len(self.train_idx),
                'n_train_augmented': self.aug_result.n_augmented,
                'n_train_total': len(self.train_idx) + self.aug_result.n_augmented,
                'aug_method': self.config.aug_method,
                'aug_ratio': self.config.aug_ratio,
                'jitter_mode': self.config.jitter_mode,
                'seed': self.config.seed,
                'aug_seed': self.config.aug_seed,
                'n_bootstrap': self.config.n_bootstrap,
                'best_threshold': self.gate1_result['best_threshold']['threshold'],
            },
            'augmentation': {
                'n_original': self.aug_result.n_original,
                'n_augmented': self.aug_result.n_augmented,
                'aug_types': dict(Counter(self.aug_result.aug_type) 
                                  if hasattr(self, 'aug_result') else {}),
            },
            'splits': self.gate1_result['metrics']['splits'],
            'sparsity': self.gate1_result['metrics']['sparsity'],
        }
        
        # Add delta if baseline specified
        if self.config.gate1_baseline_run_id:
            baseline_metrics = self._load_baseline_metrics()
            if baseline_metrics:
                delta = {
                    'test_r2_delta': (
                        self.metrics['splits']['test']['r2_mean'] -
                        baseline_metrics['splits']['test']['r2_mean']
                    ),
                    'val_r2_delta': (
                        self.metrics['splits']['val']['r2_mean'] -
                        baseline_metrics['splits']['val']['r2_mean']
                    ),
                    'sparsity_delta': (
                        self.metrics['sparsity']['sparsity'] -
                        baseline_metrics['sparsity']['sparsity']
                    ),
                    'baseline_run_id': self.config.gate1_baseline_run_id,
                }
                self.metrics['gate1_delta'] = delta
                
                print(f"  Gate1 baseline: {self.config.gate1_baseline_run_id}")
                print(f"  Test R² delta: {delta['test_r2_delta']:+.4f}")
        
        # Summary
        print(f"  Train R²: {self.metrics['splits']['train']['r2_mean']:.4f}")
        print(f"  Val R²: {self.metrics['splits']['val']['r2_mean']:.4f}")
        print(f"  Test R²: {self.metrics['splits']['test']['r2_mean']:.4f}")
        print(f"  Sparsity: {self.metrics['sparsity']['sparsity']:.1%}")
    
    def _load_baseline_metrics(self) -> Optional[Dict]:
        """Load Gate1 baseline metrics for comparison.
        
        Uses rglob for robust search that handles files mixed with directories.
        """
        try:
            gate1_root = paths.RESULTS_ROOT / self.config.dataset_version / 'gate1'
            
            if not gate1_root.exists():
                print(f"  Warning: Gate1 root not found: {gate1_root}")
                return None
            
            # Use rglob to find metrics.json in any subdirectory matching run_id
            pattern = f"**/*{self.config.gate1_baseline_run_id}*/metrics.json"
            matches = list(gate1_root.rglob(pattern))
            
            if not matches:
                # Fallback: search for run_id directory directly
                for metrics_path in gate1_root.rglob("metrics.json"):
                    if self.config.gate1_baseline_run_id in str(metrics_path):
                        matches.append(metrics_path)
                        break
            
            if matches:
                baseline_path = matches[0]
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_metrics = json.load(f)
                print(f"  ✅ Loaded baseline: {baseline_path.parent.name}")
                return baseline_metrics
            else:
                print(f"  Warning: Baseline run_id not found: {self.config.gate1_baseline_run_id}")
                
        except Exception as e:
            print(f"  Warning: Could not load baseline: {e}")
        
        return None
    
    def _generate_figures(self) -> None:
        """Step 7: Generate Gate2-specific figures."""
        print("\n[7/9] Generating Figures")
        print("-" * 50)
        
        figures_dir = self.results_dir / 'figures'
        
        # F05: Augmentation summary
        self._plot_augmentation_summary(figures_dir)
    
    def _plot_augmentation_summary(self, figures_dir: Path) -> None:
        """F05: Augmentation summary visualization."""
        fig, axes = create_figure('double', nrows=1, ncols=2)
        
        # Left: Aug type distribution
        ax1 = axes[0]
        from collections import Counter
        type_counts = Counter(self.aug_result.aug_type)
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        
        ax1.bar(range(len(types)), counts, color=get_color('proposed'))
        ax1.set_xticks(range(len(types)))
        ax1.set_xticklabels(types, rotation=45, ha='right')
        ax1.set_ylabel('Count')
        ax1.set_title('Augmentation Types')
        
        # Right: Source trajectory distribution
        ax2 = axes[1]
        ax2.hist(self.aug_result.source_idx, bins=min(20, len(self.train_idx)),
                 color=get_color('generated'), alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Source Trajectory Index')
        ax2.set_ylabel('Count')
        ax2.set_title('Source Trajectory Distribution')
        
        fig.tight_layout()
        save_figure(fig, figures_dir, 'F05_augmentation_summary')
    
    def _save_artifacts(self) -> None:
        """Step 8: Save Gate2-specific artifacts."""
        print("\\n[8/9] Saving Artifacts")
        print("-" * 50)
        
        # Determine dx_policy for documentation
        if self.config.track == 'standardized':
            dx_policy = 'mixed_savgol_analytic'
        else:
            dx_policy = 'analytic_only'
        
        # Compute augmentation statistics (P2 enhancement)
        from collections import Counter
        type_counts = Counter(self.aug_result.aug_type)
        
        n_target = self.aug_result.n_augmented
        n_fallback = type_counts.get('original_fallback', 0)
        n_success = n_target - n_fallback
        
        augmentation_stats = {
            'n_target': n_target,
            'n_success': n_success,
            'n_fallback': n_fallback,
            'success_rate': n_success / n_target if n_target > 0 else 0.0,
            'fallback_rate': n_fallback / n_target if n_target > 0 else 0.0,
            'type_counts': dict(type_counts),
        }
        
        # 1. aug_manifest.json
        aug_manifest = {
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'aug_method': self.config.aug_method,
            'aug_ratio': self.config.aug_ratio,
            'aug_seed': self.config.aug_seed,
            'jitter_mode': self.config.jitter_mode,
            'n_original': self.aug_result.n_original,
            'n_augmented': self.aug_result.n_augmented,
            'aug_config': self.aug_result.aug_config,
            'dx_policy': dx_policy,
            'augmentation_stats': augmentation_stats,  # P2 enhancement
        }
        
        with open(self.results_dir / 'aug_manifest.json', 'w', encoding='utf-8') as f:
            json.dump(aug_manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: aug_manifest.json")
        
        # Print augmentation stats summary
        print(f"      Success rate: {augmentation_stats['success_rate']:.1%}")
        if n_fallback > 0:
            print(f"      ⚠️ Fallback: {n_fallback}/{n_target}")
        
        # 2. aug_samples.npz
        np.savez(
            self.results_dir / 'aug_samples.npz',
            x=self.aug_result.x,
            u=self.aug_result.u,
            dx=self.aug_result.dx,
            params=self.aug_result.params,
            source_idx=self.aug_result.source_idx,
            aug_type=np.array(self.aug_result.aug_type, dtype=object),
        )
        print(f"  ✅ Saved: aug_samples.npz")
        
        # 3. metrics.json (with augmentation_stats)
        self.metrics['augmentation_stats'] = augmentation_stats  # P2 enhancement
        with open(self.results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: metrics.json")
        
        # 4. manifest.json (combined)
        manifest = {
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'gate': 'gate2',
            'track': self.config.track,
            'method': self.config.method,
            'augmentation': aug_manifest,
            'config': {
                'dataset_version': self.config.dataset_version,
                'n_train': self.config.n_train,
                'seed': self.config.seed,
                'n_bootstrap': self.config.n_bootstrap,
            },
            'paths': {
                'results_dir': str(self.results_dir),
            }
        }
        
        with open(self.results_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: manifest.json")
    
    def _generate_context_packet(self) -> None:
        """Step 9: Generate Context Packet."""
        print("\n[9/9] Generating Context Packet")
        print("-" * 50)
        
        cp_path = paths.get_context_packet_path(self.run_id)
        
        test_r2 = self.metrics['splits']['test']['r2_mean']
        val_r2 = self.metrics['splits']['val']['r2_mean']
        sparsity = self.metrics['sparsity']['sparsity']
        
        delta_info = ""
        if 'gate1_delta' in self.metrics:
            delta = self.metrics['gate1_delta']
            delta_info = f"""
## Gate1 비교

| 항목 | Delta |
|------|-------|
| Test R² | {delta['test_r2_delta']:+.4f} |
| Val R² | {delta['val_r2_delta']:+.4f} |
| Sparsity | {delta['sparsity_delta']:+.4f} |
| Baseline | {delta['baseline_run_id']} |
"""
        
        content = f"""# Context Packet: {self.run_id}

## 실험 정보

| 항목 | 값 |
|------|-----|
| Gate | gate2 |
| Dataset | {self.config.dataset_version} |
| Track | {self.config.track} |
| Method | {self.config.method} |
| n_train (original) | {len(self.train_idx)} |
| n_train (augmented) | {self.aug_result.n_augmented} |
| n_train (total) | {len(self.train_idx) + self.aug_result.n_augmented} |
| seed | {self.config.seed} |
| aug_seed | {self.config.aug_seed} |
| aug_ratio | {self.config.aug_ratio} |
| jitter_mode | {self.config.jitter_mode} |

## 실행 명령어

```bash
python experiments/run_gate2.py --config configs/experiments/gate2_cartpole.yaml --dataset_version {self.config.dataset_version} --n_train {self.config.n_train} --seed {self.config.seed} --track {self.config.track} --aug_ratio {self.config.aug_ratio} --jitter_mode {self.config.jitter_mode} --note {self.config.note}
```

## Results 경로

```
{self.results_dir}
```

## Metrics 요약

| Split | R² (mean) |
|-------|-----------|
| Train | {self.metrics['splits']['train']['r2_mean']:.4f} |
| Val | {val_r2:.4f} |
| Test | {test_r2:.4f} |

- Sparsity: {sparsity:.1%}
- Best threshold: {self.metrics['config']['best_threshold']}
{delta_info}

## 산출물

- aug_manifest.json ✅
- aug_samples.npz ✅
- manifest.json ✅
- metrics.json ✅
- sindy_coefficients.csv ✅
- F00~F05 figures ✅

## 다음 작업

- [ ] 다른 seed로 재실행
- [ ] aug_ratio 조절 실험
- [ ] jitter_mode 비교
- [ ] gate2_summary.csv 생성

---
*Generated: {datetime.now().isoformat()}*
"""
        
        with open(cp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Saved: {cp_path}")
    
    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass


# Need Counter for augmentation type counting
from collections import Counter


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    config = Gate2Config()
    runner = Gate2AugRunner(config)
    result = runner.run()
    
    print("\n[Result]")
    print(f"  Success: {result['success']}")
    print(f"  Run ID: {result['run_id']}")