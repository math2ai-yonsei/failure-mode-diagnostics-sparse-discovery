"""
S06: Gate0 Runner - SINDy Baseline Pipeline

Implements the complete Gate0 experimental pipeline:
1. Preflight validation
2. Data loading and normalization  
3. Feature matrix construction
4. STLSQ fitting
5. Metrics computation
6. Artifact saving
7. Context Packet generation

Usage:
    from src.experiments.gate0_runner import Gate0Runner
    
    runner = Gate0Runner(config)
    runner.run()

CLI:
    python experiments/run_gate0.py ^
      --config configs/experiments/gate0_cartpole.yaml ^
      --dataset_version cartpole_ood_v1 ^
      --n_train 10 --seed 0 --track standardized --note base
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Contracts (SSOT)
from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.contracts.plot_style import (
    create_figure, save_figure, get_color, setup_style
)

# S04: Normalization
from src.data.normalization import (
    load_norm_stats, normalize_dataset, denormalize
)

# S04: Library
from src.sindy.library import (
    SINDyLibrary, get_derivative_key, get_library_manifest
)

# S05: Optimizer
from src.sindy.optimizer import (
    ColumnScaler, STLSQOptimizer,
    save_coefficients_csv, get_optimizer_manifest, TARGET_NAMES
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Gate0Config:
    """Gate0 experiment configuration."""
    # Dataset
    dataset_version: str = 'cartpole_ood_v1'
    system: str = 'cartpole'
    
    # Experiment settings
    n_train: int = 10
    seed: int = 0
    track: str = 'standardized'  # 'standardized' or 'author_recommended'
    note: str = 'base'
    
    # SINDy settings
    library_config: str = 'gate0_min'
    threshold: float = 0.01
    max_iter: int = 10
    ridge_alpha: float = 0.0
    
    # Output
    gate: str = 'gate0'
    method: str = 'sindy'
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Gate0Config':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Gate0Config':
        """Load config from YAML file."""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


# =============================================================================
# Gate0 Runner
# =============================================================================

class Gate0Runner:
    """
    Gate0 SINDy Baseline Runner.
    
    Executes the complete Gate0 pipeline and generates required artifacts.
    """
    
    def __init__(self, config: Gate0Config):
        self.config = config
        self.run_id = paths.generate_run_id(config.note)
        
        # Results directory (auto-created via paths.py)
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
        
        # State containers (filled during run)
        self.data: Dict[str, np.ndarray] = {}
        self.norm_stats: Dict = {}
        self.library: Optional[SINDyLibrary] = None
        self.scaler: Optional[ColumnScaler] = None
        self.optimizer: Optional[STLSQOptimizer] = None
        self.metrics: Dict = {}
        self.dx_source_key: str = ''  # Tracks which dx array was used
    
    def run(self) -> Dict:
        """
        Execute the full Gate0 pipeline.
        
        Returns:
            Dict with metrics and run information
        """
        print("=" * 70)
        print(f"  Gate0 Runner: {self.run_id}")
        print("=" * 70)
        
        try:
            # 1. Preflight
            self._preflight()
            
            # 2. Load data
            self._load_data()
            
            # 3. Normalize
            self._normalize_data()
            
            # 4. Build features
            self._build_features()
            
            # 5. Fit STLSQ
            self._fit_stlsq()
            
            # 6. Compute metrics
            self._compute_metrics()
            
            # 7. Generate figures
            self._generate_figures()
            
            # 8. Save artifacts
            self._save_artifacts()
            
            # 9. Generate Context Packet
            self._generate_context_packet()
            
            print("\n" + "=" * 70)
            print(f"  ✅ Gate0 Complete: {self.run_id}")
            print(f"  Results: {self.results_dir}")
            print("=" * 70)
            
            return {
                'success': True,
                'run_id': self.run_id,
                'results_dir': str(self.results_dir),
                'metrics': self.metrics
            }
            
        except Exception as e:
            print(f"\n❌ Gate0 Failed: {e}")
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
        print(f"  Derivative key: {self.derivative_key}")
    
    def _load_data(self) -> None:
        """Step 2: Load dataset."""
        print("\n[2/9] Loading Dataset")
        print("-" * 50)
        
        data = np.load(self.dataset_path)
        
        # Load all splits
        for split in ['train', 'val', 'test']:
            self.data[f'{split}_x'] = data[f'{split}_x']
            self.data[f'{split}_u'] = data[f'{split}_u']
            self.data[f'{split}_cond_id'] = data[f'{split}_cond_id']
            self.data[f'{split}_params'] = data[f'{split}_params']
            
            # Select derivative based on track (fail-fast, no silent fallback)
            if self.config.track == 'standardized':
                # Use Savitzky-Golay numeric derivatives
                dx_key = f'{split}_dx_savgol'
                if dx_key not in data:
                    raise ValueError(
                        f"track='standardized' requires '{dx_key}' in dataset. "
                        f"Use track='author_recommended' for analytic derivatives."
                    )
            else:
                # author_recommended: use analytic derivatives
                dx_key = f'{split}_dx'
            
            self.data[f'{split}_dx'] = data[dx_key]
            
            # Record dx source for first split (all splits use same pattern)
            if split == 'train':
                self.dx_source_key = dx_key.replace('train_', '')
        
        self.data['t'] = data['t']
        self.data['dt'] = float(data['dt'])
        
        # Apply n_train limit with seed-based selection (local RNG)
        rng = np.random.default_rng(self.config.seed)
        n_total = self.data['train_x'].shape[0]
        n_use = min(self.config.n_train, n_total)
        
        if n_use < n_total:
            idx = rng.choice(n_total, n_use, replace=False)
            idx = np.sort(idx)
            for key in ['train_x', 'train_u', 'train_dx', 
                        'train_cond_id', 'train_params']:
                self.data[key] = self.data[key][idx]
        
        print(f"  Train: {self.data['train_x'].shape[0]} trajectories")
        print(f"  Val: {self.data['val_x'].shape[0]} trajectories")
        print(f"  Test: {self.data['test_x'].shape[0]} trajectories")
        print(f"  T: {self.data['train_x'].shape[1]}, dt: {self.data['dt']:.4f}")
        print(f"  dx source: {self.dx_source_key}")
    
    def _normalize_data(self) -> None:
        """Step 3: Normalize data using train statistics."""
        print("\n[3/9] Normalizing Data")
        print("-" * 50)
        
        self.norm_stats = load_norm_stats(
            self.config.dataset_version, self.config.system
        )
        
        # Normalize each split
        for split in ['train', 'val', 'test']:
            x_norm, u_norm, dx_norm = normalize_dataset(
                self.data[f'{split}_x'],
                self.data[f'{split}_u'],
                self.data[f'{split}_dx'],
                self.norm_stats,
                self.derivative_key
            )
            self.data[f'{split}_x_norm'] = x_norm
            self.data[f'{split}_u_norm'] = u_norm
            self.data[f'{split}_dx_norm'] = dx_norm
        
        print(f"  Using stats from: train split")
        print(f"  Derivative key: {self.derivative_key}")
    
    def _build_features(self) -> None:
        """Step 4: Build SINDy feature matrices."""
        print("\n[4/9] Building Feature Matrices")
        print("-" * 50)
        
        self.library = SINDyLibrary(config=self.config.library_config)
        
        # Build Theta for each split
        for split in ['train', 'val', 'test']:
            Theta = self.library.fit_transform(
                self.data[f'{split}_x_norm'],
                self.data[f'{split}_u_norm']
            )
            self.data[f'{split}_Theta'] = Theta
        
        print(f"  Library: {self.config.library_config}")
        print(f"  Features: {self.library.n_features}")
        print(f"  Train Theta shape: {self.data['train_Theta'].shape}")
        
        # Scale Theta (train-only fit)
        self.scaler = ColumnScaler()
        self.data['train_Theta_scaled'] = self.scaler.fit_transform(
            self.data['train_Theta']
        )
        self.data['val_Theta_scaled'] = self.scaler.transform(
            self.data['val_Theta']
        )
        self.data['test_Theta_scaled'] = self.scaler.transform(
            self.data['test_Theta']
        )
        
        # Verify constant column preserved
        const_preserved = np.allclose(
            self.data['train_Theta_scaled'][:, 0], 1.0
        )
        print(f"  Constant '1' preserved: {const_preserved}")
    
    def _fit_stlsq(self) -> None:
        """Step 5: Fit STLSQ optimizer."""
        print("\n[5/9] Fitting STLSQ")
        print("-" * 50)
        
        # Flatten dx for regression
        dx_train_flat = self.data['train_dx_norm'].reshape(-1, 4)
        
        self.optimizer = STLSQOptimizer(
            threshold=self.config.threshold,
            max_iter=self.config.max_iter,
            ridge_alpha=self.config.ridge_alpha
        )
        
        self.optimizer.fit(
            self.data['train_Theta_scaled'],
            dx_train_flat
        )
        
        sparsity = self.optimizer.get_sparsity_info()
        print(f"  Threshold: {self.config.threshold}")
        print(f"  Iterations: {sparsity['n_iter']}")
        print(f"  Nonzero: {sparsity['n_nonzero']} / {sparsity['n_total']}")
        print(f"  Sparsity: {sparsity['sparsity']:.1%}")
    
    def _compute_metrics(self) -> None:
        """Step 6: Compute evaluation metrics."""
        print("\n[6/9] Computing Metrics")
        print("-" * 50)
        
        self.metrics = {
            'run_id': self.run_id,
            'config': {
                'dataset_version': self.config.dataset_version,
                'n_train': self.config.n_train,
                'seed': self.config.seed,
                'track': self.config.track,
                'threshold': self.config.threshold,
                'library_config': self.config.library_config,
            },
            'sparsity': self.optimizer.get_sparsity_info(),
            'splits': {}
        }
        
        # Compute R² for each split
        for split in ['train', 'val', 'test']:
            dx_true = self.data[f'{split}_dx_norm'].reshape(-1, 4)
            dx_pred = self.optimizer.predict(
                self.data[f'{split}_Theta_scaled']
            )
            
            # R² per target
            ss_res = np.sum((dx_true - dx_pred) ** 2, axis=0)
            ss_tot = np.sum((dx_true - dx_true.mean(axis=0)) ** 2, axis=0)
            r2 = 1 - ss_res / ss_tot
            
            # RMSE per target
            rmse = np.sqrt(np.mean((dx_true - dx_pred) ** 2, axis=0))
            
            self.metrics['splits'][split] = {
                'r2_per_target': r2.tolist(),
                'r2_mean': float(np.mean(r2)),
                'rmse_per_target': rmse.tolist(),
                'rmse_mean': float(np.mean(rmse)),
                'n_samples': dx_true.shape[0],
            }
            
            print(f"  {split:5s}: R²={np.mean(r2):.4f}, RMSE={np.mean(rmse):.4f}")
    
    def _generate_figures(self) -> None:
        """Step 7: Generate required figures."""
        print("\n[7/9] Generating Figures")
        print("-" * 50)
        
        setup_style()
        figures_dir = self.results_dir / 'figures'
        
        # F00: Condition Distribution
        self._plot_condition_distribution(figures_dir)
        
        # F01: Rollout Example
        self._plot_rollout_example(figures_dir)
        
        # F02: Coefficient Heatmap
        self._plot_coefficient_heatmap(figures_dir)
    
    def _plot_condition_distribution(self, figures_dir: Path) -> None:
        """F00: Plot condition ID distribution across splits."""
        fig, ax = create_figure('wide')
        
        splits = ['train', 'val', 'test']
        colors = [get_color('train'), get_color('val'), get_color('test')]
        
        for i, (split, color) in enumerate(zip(splits, colors)):
            cond_ids = self.data[f'{split}_cond_id']
            unique, counts = np.unique(cond_ids, return_counts=True)
            
            ax.bar(
                unique + i * 0.25 - 0.25,
                counts,
                width=0.25,
                color=color,
                label=f'{split} (n={len(cond_ids)})',
                alpha=0.8
            )
        
        ax.set_xlabel('Condition ID')
        ax.set_ylabel('Count')
        ax.set_title('Trajectory Distribution by Condition')
        ax.legend()
        
        save_figure(fig, figures_dir, 'F00_condition_distribution')
    
    def _plot_rollout_example(self, figures_dir: Path) -> None:
        """F01: Plot example trajectory with prediction."""
        fig, axes = create_figure('large', nrows=2, ncols=2)
        axes = axes.flatten()
        
        t = self.data['t']
        T = len(t)
        
        # Use first validation trajectory (explicit indexing)
        traj_idx = 0
        dx_true = self.data['val_dx_norm'][traj_idx]  # (T, 4)
        
        # Compute Theta for this specific trajectory (safer than slicing)
        x_traj = self.data['val_x_norm'][traj_idx:traj_idx+1]  # (1, T, 4)
        u_traj = self.data['val_u_norm'][traj_idx:traj_idx+1]  # (1, T, 1)
        Theta_traj = self.library.fit_transform(x_traj, u_traj)  # (T, 21)
        Theta_traj_scaled = self.scaler.transform(Theta_traj)
        dx_pred = self.optimizer.predict(Theta_traj_scaled)  # (T, 4)
        
        for i, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
            ax.plot(t, dx_true[:, i], 
                    color=get_color('ground_truth'), 
                    label='True', linewidth=1.5)
            ax.plot(t, dx_pred[:, i], 
                    color=get_color('prediction'), 
                    label='Pred', linestyle='--', linewidth=1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(name)
            ax.legend(loc='upper right')
        
        fig.suptitle('Derivative Prediction (Val Traj #0)', fontsize=11)
        fig.tight_layout()
        
        save_figure(fig, figures_dir, 'F01_rollout_example')
    
    def _plot_coefficient_heatmap(self, figures_dir: Path) -> None:
        """F02: Plot coefficient matrix heatmap."""
        fig, ax = create_figure('large')
        
        # Get unscaled coefficients
        target_scale = np.array(
            self.norm_stats[self.derivative_key]['std']
        )
        coeffs = self.optimizer.get_unscaled_coefficients(
            self.scaler, target_scale
        )
        
        # Heatmap
        im = ax.imshow(
            coeffs.T,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-np.abs(coeffs).max(),
            vmax=np.abs(coeffs).max()
        )
        
        # Labels
        ax.set_xticks(range(len(self.library.feature_names)))
        ax.set_xticklabels(self.library.feature_names, rotation=45, ha='right')
        ax.set_yticks(range(len(TARGET_NAMES)))
        ax.set_yticklabels(TARGET_NAMES)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Target')
        ax.set_title('SINDy Coefficient Matrix (Unscaled)')
        
        # Colorbar
        fig.colorbar(im, ax=ax, label='Coefficient Value')
        fig.tight_layout()
        
        save_figure(fig, figures_dir, 'F02_coeff_heatmap')
    
    def _save_artifacts(self) -> None:
        """Step 8: Save required artifacts."""
        print("\n[8/9] Saving Artifacts")
        print("-" * 50)
        
        # 1. manifest.json
        manifest = {
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'gate': self.config.gate,
            'track': self.config.track,
            'method': self.config.method,
            'config': {
                'dataset_version': self.config.dataset_version,
                'system': self.config.system,
                'n_train': self.config.n_train,
                'seed': self.config.seed,
                'threshold': self.config.threshold,
                'dx_source_key': self.dx_source_key,
            },
            'library': get_library_manifest(self.library, self.config.track),
            'optimizer': get_optimizer_manifest(
                self.optimizer, self.scaler, self.library.feature_names
            ),
            'paths': {
                'results_dir': str(self.results_dir),
                'dataset': str(self.dataset_path),
            }
        }
        
        manifest_path = self.results_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: manifest.json")
        
        # 2. metrics.json
        metrics_path = self.results_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: metrics.json")
        
        # 3. sindy_coefficients.csv
        target_scale = np.array(
            self.norm_stats[self.derivative_key]['std']
        )
        coeffs_unscaled = self.optimizer.get_unscaled_coefficients(
            self.scaler, target_scale
        )
        
        save_coefficients_csv(
            coeffs_unscaled,
            self.library.feature_names,
            TARGET_NAMES,
            self.results_dir / 'sindy_coefficients.csv'
        )
    
    def _generate_context_packet(self) -> None:
        """Step 9: Generate Context Packet for next conversation."""
        print("\n[9/9] Generating Context Packet")
        print("-" * 50)
        
        cp_path = paths.get_context_packet_path(self.run_id)
        
        # Format metrics summary
        train_r2 = self.metrics['splits']['train']['r2_mean']
        val_r2 = self.metrics['splits']['val']['r2_mean']
        test_r2 = self.metrics['splits']['test']['r2_mean']
        sparsity = self.metrics['sparsity']['sparsity']
        
        content = f"""# Context Packet: {self.run_id}

## 실험 정보

| 항목 | 값 |
|------|-----|
| Gate | {self.config.gate} |
| Dataset | {self.config.dataset_version} |
| Track | {self.config.track} |
| Method | {self.config.method} |
| n_train | {self.config.n_train} |
| seed | {self.config.seed} |
| threshold | {self.config.threshold} |

## 실행 명령어

```bash
python experiments/run_gate0.py ^
  --config configs/experiments/gate0_cartpole.yaml ^
  --dataset_version {self.config.dataset_version} ^
  --n_train {self.config.n_train} --seed {self.config.seed} ^
  --track {self.config.track} --note {self.config.note}
```

## Results 경로

```
{self.results_dir}
```

## Metrics 요약

| Split | R² (mean) |
|-------|-----------|
| Train | {train_r2:.4f} |
| Val | {val_r2:.4f} |
| Test | {test_r2:.4f} |

- Sparsity: {sparsity:.1%}
- Nonzero terms: {self.metrics['sparsity']['n_nonzero']} / {self.metrics['sparsity']['n_total']}

## 산출물

- manifest.json ✅
- metrics.json ✅
- sindy_coefficients.csv ✅
- F00_condition_distribution.png/pdf ✅
- F01_rollout_example.png/pdf ✅
- F02_coeff_heatmap.png/pdf ✅

## 다음 작업

- [ ] seed=1로 재실행하여 Gate0 통과 조건(2 seeds) 달성
- [ ] n_train=20으로 실험 확장
- [ ] Gate1 (E-SINDy) 진입 준비

---
*Generated: {datetime.now().isoformat()}*
"""
        
        with open(cp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Saved: {cp_path}")


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Quick test with default config
    config = Gate0Config()
    runner = Gate0Runner(config)
    result = runner.run()
    
    print("\n[Result]")
    print(f"  Success: {result['success']}")
    print(f"  Run ID: {result['run_id']}")