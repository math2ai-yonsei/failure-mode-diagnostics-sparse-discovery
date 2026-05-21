"""
S07: Gate1 E-SINDy Runner

Implements the Gate1 experimental pipeline with:
1. Bootstrap ensemble learning (trajectory-level)
2. Threshold grid search with val-based selection
3. Uncertainty quantification (coefficient std, inclusion probability)
4. Additional figures (F03, F04)

Key Design Decisions (per GPT cross-review):
    - Trajectory-level bootstrap (NOT row-level)
    - Best threshold: val R² → sparsity → uncertainty tie-break
    - Coefficients aggregated in UNSCALED (physical) units
    - Does NOT inherit/call Gate0Runner private methods

Usage:
    from src.experiments.gate1_esindy_runner import Gate1ESINDyRunner, Gate1Config
    
    runner = Gate1ESINDyRunner(config)
    runner.run()

CLI:
    python experiments/run_gate1.py ^
      --config configs/experiments/gate1_cartpole.yaml ^
      --dataset_version cartpole_ood_v1 ^
      --n_train 10 --seed 0 --track standardized ^
      --n_bootstrap 20 --note base
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
from src.data.normalization import load_norm_stats, normalize_dataset

# S04: Library
from src.sindy.library import SINDyLibrary, get_derivative_key, get_library_manifest

# S05: Optimizer
from src.sindy.optimizer import (
    ColumnScaler, save_coefficients_csv, TARGET_NAMES
)

# S07: E-SINDy
from src.sindy.esindy import (
    ESINDyEnsemble, threshold_sweep, select_best_threshold,
    save_coefficients_std_csv, save_inclusion_prob_csv, save_threshold_sweep_csv
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Gate1Config:
    """Gate1 E-SINDy experiment configuration."""
    # Dataset
    dataset_version: str = 'cartpole_ood_v1'
    system: str = 'cartpole'
    
    # Experiment settings
    n_train: int = 10
    seed: int = 0
    track: str = 'standardized'
    note: str = 'base'
    
    # Library settings
    library_config: str = 'gate0_min'
    
    # E-SINDy settings
    n_bootstrap: int = 20
    thresholds: List[float] = field(default_factory=lambda: [
        0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2
    ])
    
    # Threshold selection
    best_threshold_metric: str = 'val_r2_mean'
    tie_break_sparsity: bool = True
    tie_break_uncertainty: bool = True
    tie_tolerance: float = 0.001
    
    # Final fit
    final_fit_split: str = 'train'  # 'train' or 'train_val'
    
    # Optimizer settings
    max_iter: int = 10
    ridge_alpha: float = 0.0
    
    # Inclusion settings
    inclusion_eps: float = 0.0
    inclusion_threshold: float = 0.5
    
    # Fixed
    gate: str = 'gate1'
    method: str = 'esindy'
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Gate1Config':
        """Create config from dictionary."""
        # Handle thresholds list specially
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Gate1Config':
        """Load config from YAML file."""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


# =============================================================================
# Gate1 E-SINDy Runner
# =============================================================================

class Gate1ESINDyRunner:
    """
    Gate1 E-SINDy Runner.
    
    Pipeline:
        1. Preflight validation
        2. Load & normalize data
        3. Build feature matrix
        4. Threshold sweep (train → val evaluation)
        5. Select best threshold
        6. Final E-SINDy fit with best threshold
        7. Compute metrics (train/val/test)
        8. Generate figures (F00-F04)
        9. Save artifacts
        10. Generate Context Packet
    """
    
    def __init__(self, config: Gate1Config):
        self.config = config
        self.run_id = paths.generate_run_id(config.note)
        
        # Results directory
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
        self.data: Dict[str, np.ndarray] = {}
        self.norm_stats: Dict = {}
        self.library: Optional[SINDyLibrary] = None
        self.scaler: Optional[ColumnScaler] = None
        self.ensemble: Optional[ESINDyEnsemble] = None
        self.sweep_results: List[Dict] = []
        self.best_threshold_result: Dict = {}
        self.metrics: Dict = {}
        self.dx_source_key: str = ''
        self.T: int = 0  # Time steps per trajectory
    
    def run(self) -> Dict:
        """Execute the full Gate1 pipeline."""
        print("=" * 70)
        print(f"  Gate1 E-SINDy Runner: {self.run_id}")
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
            
            # 5. Threshold sweep
            self._threshold_sweep()
            
            # 6. Select best & final fit
            self._select_best_threshold()
            self._fit_final_esindy()
            
            # 7. Compute metrics
            self._compute_metrics()
            
            # 8. Generate figures
            self._generate_figures()
            
            # 9. Save artifacts
            self._save_artifacts()
            
            # 10. Context Packet
            self._generate_context_packet()
            
            print("\n" + "=" * 70)
            print(f"  ✅ Gate1 Complete: {self.run_id}")
            print(f"  Results: {self.results_dir}")
            print("=" * 70)
            
            return {
                'success': True,
                'run_id': self.run_id,
                'results_dir': str(self.results_dir),
                'metrics': self.metrics,
                'best_threshold': self.best_threshold_result.get('threshold'),
            }
            
        except Exception as e:
            print(f"\n❌ Gate1 Failed: {e}")
            raise
    
    # =========================================================================
    # Pipeline Steps
    # =========================================================================
    
    def _preflight(self) -> None:
        """Step 1: Preflight validation."""
        print("\n[1/10] Preflight Validation")
        print("-" * 50)
        
        validate_dataset_lite(self.dataset_path)
        
        print(f"  Dataset: {self.dataset_path.name}")
        print(f"  Track: {self.config.track}")
        print(f"  Derivative key: {self.derivative_key}")
        print(f"  n_bootstrap: {self.config.n_bootstrap}")
    
    def _load_data(self) -> None:
        """Step 2: Load dataset."""
        print("\n[2/10] Loading Dataset")
        print("-" * 50)
        
        data = np.load(self.dataset_path)
        
        # Store time info
        self.data['t'] = data['t']
        self.data['dt'] = float(data['dt'])
        self.T = len(self.data['t'])
        
        # Load all splits
        for split in ['train', 'val', 'test']:
            self.data[f'{split}_x'] = data[f'{split}_x']
            self.data[f'{split}_u'] = data[f'{split}_u']
            self.data[f'{split}_cond_id'] = data[f'{split}_cond_id']
            self.data[f'{split}_params'] = data[f'{split}_params']
            
            # Select derivative based on track
            if self.config.track == 'standardized':
                dx_key = f'{split}_dx_savgol'
                if dx_key not in data:
                    raise ValueError(
                        f"track='standardized' requires '{dx_key}' in dataset."
                    )
            else:
                dx_key = f'{split}_dx'
            
            self.data[f'{split}_dx'] = data[dx_key]
            
            if split == 'train':
                self.dx_source_key = dx_key.replace('train_', '')
        
        # Apply n_train limit with seed-based selection
        rng = np.random.default_rng(self.config.seed)
        n_total = self.data['train_x'].shape[0]
        n_use = min(self.config.n_train, n_total)
        
        if n_use < n_total:
            idx = rng.choice(n_total, n_use, replace=False)
            idx = np.sort(idx)
            for key in ['train_x', 'train_u', 'train_dx',
                        'train_cond_id', 'train_params']:
                self.data[key] = self.data[key][idx]
        
        # Store trajectory counts
        self.data['n_train'] = self.data['train_x'].shape[0]
        self.data['n_val'] = self.data['val_x'].shape[0]
        self.data['n_test'] = self.data['test_x'].shape[0]
        
        print(f"  Train: {self.data['n_train']} trajectories")
        print(f"  Val: {self.data['n_val']} trajectories")
        print(f"  Test: {self.data['n_test']} trajectories")
        print(f"  T: {self.T}, dt: {self.data['dt']:.4f}")
    
    def _normalize_data(self) -> None:
        """Step 3: Normalize data."""
        print("\n[3/10] Normalizing Data")
        print("-" * 50)
        
        self.norm_stats = load_norm_stats(
            self.config.dataset_version, self.config.system
        )
        
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
        
        print(f"  Derivative key: {self.derivative_key}")
    
    def _build_features(self) -> None:
        """Step 4: Build SINDy feature matrices."""
        print("\n[4/10] Building Feature Matrices")
        print("-" * 50)
        
        self.library = SINDyLibrary(config=self.config.library_config)
        
        for split in ['train', 'val', 'test']:
            Theta = self.library.fit_transform(
                self.data[f'{split}_x_norm'],
                self.data[f'{split}_u_norm']
            )
            self.data[f'{split}_Theta'] = Theta
        
        print(f"  Library: {self.config.library_config}")
        print(f"  Features: {self.library.n_features}")
        
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
        
        print(f"  Train Theta shape: {self.data['train_Theta_scaled'].shape}")
    
    def _threshold_sweep(self) -> None:
        """Step 5: Threshold sweep with val evaluation."""
        print("\n[5/10] Threshold Sweep")
        print("-" * 50)
        
        # Flatten dx for regression
        dx_train = self.data['train_dx_norm'].reshape(-1, 4)
        dx_val = self.data['val_dx_norm'].reshape(-1, 4)
        
        # Get target scale for unscaling
        target_scale = np.array(self.norm_stats[self.derivative_key]['std'])
        
        self.sweep_results = threshold_sweep(
            Theta_train=self.data['train_Theta_scaled'],
            dx_train=dx_train,
            Theta_val=self.data['val_Theta_scaled'],
            dx_val=dx_val,
            thresholds=self.config.thresholds,
            n_trajectories_train=self.data['n_train'],
            n_trajectories_val=self.data['n_val'],
            T=self.T,
            scaler=self.scaler,
            target_scale=target_scale,
            n_bootstrap=self.config.n_bootstrap,
            random_state=self.config.seed,
        )
        
        print(f"  {'Threshold':>10} {'Train R²':>10} {'Val R²':>10} {'Sparsity':>10}")
        for r in self.sweep_results:
            print(f"  {r['threshold']:>10.4f} {r['train_r2_mean']:>10.4f} "
                  f"{r['val_r2_mean']:>10.4f} {r['sparsity']:>10.1%}")
    
    def _select_best_threshold(self) -> None:
        """Step 6a: Select best threshold."""
        print("\n[6/10] Selecting Best Threshold")
        print("-" * 50)
        
        self.best_threshold_result = select_best_threshold(
            self.sweep_results,
            metric=self.config.best_threshold_metric,
            tie_break_sparsity=self.config.tie_break_sparsity,
            tie_break_uncertainty=self.config.tie_break_uncertainty,
            tie_tolerance=self.config.tie_tolerance,
        )
        
        print(f"  Best threshold: {self.best_threshold_result['threshold']}")
        print(f"  Selection reason: {self.best_threshold_result['best_reason']}")
        print(f"  Val R²: {self.best_threshold_result['val_r2_mean']:.4f}")
        print(f"  Sparsity: {self.best_threshold_result['sparsity']:.1%}")
    
    def _fit_final_esindy(self) -> None:
        """Step 6b: Final E-SINDy fit with best threshold."""
        print("\n  Fitting Final E-SINDy...")
        
        best_thresh = self.best_threshold_result['threshold']
        target_scale = np.array(self.norm_stats[self.derivative_key]['std'])
        
        # Determine training data for final fit
        if self.config.final_fit_split == 'train_val':
            # Combine train + val (optional, for maximum data usage)
            Theta_final = np.vstack([
                self.data['train_Theta_scaled'],
                self.data['val_Theta_scaled']
            ])
            dx_final = np.vstack([
                self.data['train_dx_norm'].reshape(-1, 4),
                self.data['val_dx_norm'].reshape(-1, 4)
            ])
            n_traj_final = self.data['n_train'] + self.data['n_val']
        else:
            # Default: train only
            Theta_final = self.data['train_Theta_scaled']
            dx_final = self.data['train_dx_norm'].reshape(-1, 4)
            n_traj_final = self.data['n_train']
        
        self.ensemble = ESINDyEnsemble(
            n_bootstrap=self.config.n_bootstrap,
            threshold=best_thresh,
            max_iter=self.config.max_iter,
            ridge_alpha=self.config.ridge_alpha,
            random_state=self.config.seed,
            inclusion_eps=self.config.inclusion_eps,
        )
        
        self.ensemble.fit(
            Theta_final, dx_final,
            n_trajectories=n_traj_final,
            T=self.T,
            scaler=self.scaler,
            target_scale=target_scale,
        )
        
        sparsity = self.ensemble.get_sparsity_info(self.config.inclusion_threshold)
        print(f"  Active terms: {sparsity['n_active']} / {sparsity['n_total']}")
        print(f"  Final sparsity: {sparsity['sparsity']:.1%}")
    
    def _compute_metrics(self) -> None:
        """Step 7: Compute evaluation metrics."""
        print("\n[7/10] Computing Metrics")
        print("-" * 50)
        
        target_scale = np.array(self.norm_stats[self.derivative_key]['std'])
        
        self.metrics = {
            'run_id': self.run_id,
            'config': {
                'dataset_version': self.config.dataset_version,
                'n_train': self.config.n_train,
                'seed': self.config.seed,
                'track': self.config.track,
                'n_bootstrap': self.config.n_bootstrap,
                'best_threshold': self.best_threshold_result['threshold'],
                'best_reason': self.best_threshold_result['best_reason'],
                'library_config': self.config.library_config,
            },
            'sparsity': self.ensemble.get_sparsity_info(self.config.inclusion_threshold),
            'splits': {},
        }
        
        for split in ['train', 'val', 'test']:
            dx_true = self.data[f'{split}_dx_norm'].reshape(-1, 4)
            dx_pred = self.ensemble.predict(
                self.data[f'{split}_Theta_scaled'],
                self.scaler,
                target_scale,
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
        """Step 8: Generate required figures."""
        print("\n[8/10] Generating Figures")
        print("-" * 50)
        
        setup_style()
        figures_dir = self.results_dir / 'figures'
        
        # F00-F02: Same as Gate0
        self._plot_condition_distribution(figures_dir)
        self._plot_rollout_example(figures_dir)
        self._plot_coefficient_heatmap(figures_dir)
        
        # F03-F04: Gate1 specific
        self._plot_threshold_sweep(figures_dir)
        self._plot_coefficient_uncertainty(figures_dir)
    
    def _plot_condition_distribution(self, figures_dir: Path) -> None:
        """F00: Condition distribution."""
        fig, ax = create_figure('wide')
        
        splits = ['train', 'val', 'test']
        colors = [get_color('train'), get_color('val'), get_color('test')]
        
        for i, (split, color) in enumerate(zip(splits, colors)):
            cond_ids = self.data[f'{split}_cond_id']
            unique, counts = np.unique(cond_ids, return_counts=True)
            ax.bar(unique + i * 0.25 - 0.25, counts, width=0.25,
                   color=color, label=f'{split} (n={len(cond_ids)})', alpha=0.8)
        
        ax.set_xlabel('Condition ID')
        ax.set_ylabel('Count')
        ax.set_title('Trajectory Distribution by Condition')
        ax.legend()
        
        save_figure(fig, figures_dir, 'F00_condition_distribution')
    
    def _plot_rollout_example(self, figures_dir: Path) -> None:
        """F01: Rollout example with uncertainty band."""
        fig, axes = create_figure('large', nrows=2, ncols=2)
        axes = axes.flatten()
        
        t = self.data['t']
        target_scale = np.array(self.norm_stats[self.derivative_key]['std'])
        
        # Use first validation trajectory
        traj_idx = 0
        dx_true = self.data['val_dx_norm'][traj_idx]  # (T, 4)
        
        # Get prediction from each bootstrap member for uncertainty band
        x_traj = self.data['val_x_norm'][traj_idx:traj_idx+1]
        u_traj = self.data['val_u_norm'][traj_idx:traj_idx+1]
        Theta_traj = self.library.fit_transform(x_traj, u_traj)
        Theta_scaled = self.scaler.transform(Theta_traj)
        
        # Compute predictions from all bootstrap members
        scale_Theta = self.scaler.get_scale_factors()['scale']
        all_preds = []
        for coef_unscaled in self.ensemble._individual_coefficients:
            # Convert unscaled coef back to scaled for prediction
            coef_scaled = coef_unscaled * scale_Theta[:, np.newaxis]
            if target_scale is not None:
                coef_scaled = coef_scaled / target_scale[np.newaxis, :]
            pred = Theta_scaled @ coef_scaled
            all_preds.append(pred)
        
        all_preds = np.stack(all_preds, axis=0)  # (B, T, 4)
        dx_pred_mean = np.mean(all_preds, axis=0)
        dx_pred_std = np.std(all_preds, axis=0)
        
        for i, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
            # True values
            ax.plot(t, dx_true[:, i], color=get_color('ground_truth'),
                    label='True', linewidth=1.5)
            # Mean prediction
            ax.plot(t, dx_pred_mean[:, i], color=get_color('prediction'),
                    label='E-SINDy (mean)', linestyle='--', linewidth=1.5)
            # Uncertainty band (mean ± std)
            ax.fill_between(t,
                           dx_pred_mean[:, i] - dx_pred_std[:, i],
                           dx_pred_mean[:, i] + dx_pred_std[:, i],
                           color=get_color('prediction'), alpha=0.2, label='±1σ')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(name)
            ax.legend(loc='upper right', fontsize=7)
        
        fig.suptitle(f'E-SINDy Prediction with Uncertainty (Val Traj #0, thresh={self.best_threshold_result["threshold"]})', fontsize=10)
        fig.tight_layout()
        
        save_figure(fig, figures_dir, 'F01_rollout_example')
    
    def _plot_coefficient_heatmap(self, figures_dir: Path) -> None:
        """F02: Coefficient heatmap (ensemble mean)."""
        fig, ax = create_figure('large')
        
        coeffs = self.ensemble.coefficients_mean_
        
        im = ax.imshow(
            coeffs.T, aspect='auto', cmap='RdBu_r',
            vmin=-np.abs(coeffs).max(), vmax=np.abs(coeffs).max()
        )
        
        ax.set_xticks(range(len(self.library.feature_names)))
        ax.set_xticklabels(self.library.feature_names, rotation=45, ha='right')
        ax.set_yticks(range(len(TARGET_NAMES)))
        ax.set_yticklabels(TARGET_NAMES)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Target')
        ax.set_title('E-SINDy Coefficient Matrix (Ensemble Mean, Unscaled)')
        
        fig.colorbar(im, ax=ax, label='Coefficient Value')
        fig.tight_layout()
        
        save_figure(fig, figures_dir, 'F02_coeff_heatmap')
    
    def _plot_threshold_sweep(self, figures_dir: Path) -> None:
        """F03: Threshold sweep results."""
        fig, axes = create_figure('wide', nrows=1, ncols=2)
        
        thresholds = [r['threshold'] for r in self.sweep_results]
        train_r2 = [r['train_r2_mean'] for r in self.sweep_results]
        val_r2 = [r['val_r2_mean'] for r in self.sweep_results]
        sparsity = [r['sparsity'] for r in self.sweep_results]
        
        best_thresh = self.best_threshold_result['threshold']
        best_idx = thresholds.index(best_thresh)
        
        # Left: R² vs threshold
        ax1 = axes[0]
        ax1.plot(thresholds, train_r2, 'o-', color=get_color('train'), label='Train R²')
        ax1.plot(thresholds, val_r2, 's-', color=get_color('val'), label='Val R²')
        ax1.axvline(best_thresh, color=get_color('neutral'), linestyle='--', alpha=0.7, label=f'Best: {best_thresh}')
        ax1.scatter([best_thresh], [val_r2[best_idx]], s=100, c=get_color('highlight'), zorder=5, marker='*')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('R²')
        ax1.set_title('R² vs Threshold')
        ax1.legend()
        ax1.set_xscale('symlog', linthresh=1e-4)
        
        # Right: Sparsity vs threshold
        ax2 = axes[1]
        ax2.plot(thresholds, sparsity, 'o-', color=get_color('proposed'))
        ax2.axvline(best_thresh, color=get_color('neutral'), linestyle='--', alpha=0.7)
        ax2.scatter([best_thresh], [sparsity[best_idx]], s=100, c=get_color('highlight'), zorder=5, marker='*')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Sparsity')
        ax2.set_title('Sparsity vs Threshold')
        ax2.set_xscale('symlog', linthresh=1e-4)
        
        fig.tight_layout()
        save_figure(fig, figures_dir, 'F03_threshold_sweep')
    
    def _plot_coefficient_uncertainty(self, figures_dir: Path) -> None:
        """F04: Coefficient uncertainty visualization."""
        fig, axes = create_figure('large', nrows=2, ncols=2)
        axes = axes.flatten()
        
        coef_mean = self.ensemble.coefficients_mean_
        coef_std = self.ensemble.coefficients_std_
        incl_prob = self.ensemble.inclusion_probability_
        feature_names = self.library.feature_names
        
        for target_idx, (ax, target_name) in enumerate(zip(axes, TARGET_NAMES)):
            # Get active features for this target
            active_mask = incl_prob[:, target_idx] > self.config.inclusion_threshold
            
            if active_mask.sum() == 0:
                ax.text(0.5, 0.5, 'No active terms', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(f'{target_name}')
                continue
            
            active_idx = np.where(active_mask)[0]
            n_active = len(active_idx)
            
            means = coef_mean[active_idx, target_idx]
            stds = coef_std[active_idx, target_idx]
            names = [feature_names[i] for i in active_idx]
            
            y_pos = np.arange(n_active)
            ax.barh(y_pos, means, xerr=stds, capsize=3,
                    color=get_color('proposed'), alpha=0.7)
            ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Coefficient')
            ax.set_title(f'{target_name} (n={n_active})')
        
        fig.suptitle('Coefficient Uncertainty (mean ± std)', fontsize=11)
        fig.tight_layout()
        
        save_figure(fig, figures_dir, 'F04_coefficient_uncertainty')
    
    def _save_artifacts(self) -> None:
        """Step 9: Save artifacts."""
        print("\n[9/10] Saving Artifacts")
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
                'n_bootstrap': self.config.n_bootstrap,
                'best_threshold': self.best_threshold_result['threshold'],
                'best_reason': self.best_threshold_result['best_reason'],
                'final_fit_split': self.config.final_fit_split,
                'dx_source_key': self.dx_source_key,
            },
            'library': get_library_manifest(self.library, self.config.track),
            'esindy': {
                'n_bootstrap': self.config.n_bootstrap,
                'bootstrap_unit': 'trajectory',
                'inclusion_eps': self.config.inclusion_eps,
                'inclusion_threshold': self.config.inclusion_threshold,
                'thresholds_tested': self.config.thresholds,
            },
            'paths': {
                'results_dir': str(self.results_dir),
                'dataset': str(self.dataset_path),
            }
        }
        
        with open(self.results_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: manifest.json")
        
        # 2. metrics.json
        with open(self.results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: metrics.json")
        
        # 3. sindy_coefficients.csv (ensemble mean, unscaled)
        save_coefficients_csv(
            self.ensemble.coefficients_mean_,
            self.library.feature_names,
            TARGET_NAMES,
            self.results_dir / 'sindy_coefficients.csv'
        )
        
        # 4. coefficient_std.csv (NEW)
        save_coefficients_std_csv(
            self.ensemble.coefficients_std_,
            self.library.feature_names,
            TARGET_NAMES,
            self.results_dir / 'coefficient_std.csv'
        )
        
        # 5. inclusion_probability.csv (NEW)
        save_inclusion_prob_csv(
            self.ensemble.inclusion_probability_,
            self.library.feature_names,
            TARGET_NAMES,
            self.results_dir / 'inclusion_probability.csv'
        )
        
        # 6. threshold_sweep.csv (NEW)
        save_threshold_sweep_csv(
            self.sweep_results,
            self.results_dir / 'threshold_sweep.csv'
        )
    
    def _generate_context_packet(self) -> None:
        """Step 10: Generate Context Packet."""
        print("\n[10/10] Generating Context Packet")
        print("-" * 50)
        
        cp_path = paths.get_context_packet_path(self.run_id)
        
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
| n_bootstrap | {self.config.n_bootstrap} |
| best_threshold | {self.best_threshold_result['threshold']} |
| selection_reason | {self.best_threshold_result['best_reason']} |

## 실행 명령어

```bash
python experiments/run_gate1.py --config configs/experiments/gate1_cartpole.yaml --dataset_version {self.config.dataset_version} --n_train {self.config.n_train} --seed {self.config.seed} --track {self.config.track} --n_bootstrap {self.config.n_bootstrap} --note {self.config.note}
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
- Active terms: {self.metrics['sparsity']['n_active']} / {self.metrics['sparsity']['n_total']}
- Mean coef std: {self.metrics['sparsity']['mean_coef_std']:.6f}

## 산출물

- manifest.json ✅
- metrics.json ✅
- sindy_coefficients.csv ✅
- coefficient_std.csv ✅
- inclusion_probability.csv ✅
- threshold_sweep.csv ✅
- F00~F04 figures ✅

## 다음 작업

- [ ] 다른 seed로 재실행하여 Gate1 통과 조건 달성
- [ ] n_train=20으로 실험 확장
- [ ] gate1_summary.csv 생성

---
*Generated: {datetime.now().isoformat()}*
"""
        
        with open(cp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Saved: {cp_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    config = Gate1Config()
    runner = Gate1ESINDyRunner(config)
    result = runner.run()
    
    print("\n[Result]")
    print(f"  Success: {result['success']}")
    print(f"  Run ID: {result['run_id']}")
    print(f"  Best threshold: {result['best_threshold']}")