#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Generative Augmentation Runner
====================================
GPT cross-review applied:
- A1: train_indices from Gate1 manifest (mandatory when baseline specified)
- A2: tolerance = 0.002 fixed (configurable via YAML)
- B: teacher required when align_filter_on=True
"""

import sys
import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    from pathlib import Path as PathLib
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, PathLib):
        return str(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)  # Fallback for unknown types




@dataclass
class Gate3Config:
    """Gate3 configuration"""
    gate: str = 'gate3'
    dataset_version: str = 'cartpole_ood_v1'
    system: str = 'cartpole'
    track: str = 'standardized'
    n_train: int = 10
    note: str = 'base'
    data_seed: int = 0
    vae_seed: int = 0
    gen_seed: int = 0
    method: str = 'generative'
    aug_method: str = 'vae_align'
    aug_ratio: float = 1.0
    dx_policy: str = 'savgol'
    n_generate: int = 100
    n_select: int = 10
    gate1_baseline_run_id: Optional[str] = None
    gate2_baseline_run_id: Optional[str] = None
    mismatch_action: str = 'fail'
    align_filter_on: bool = True
    align_mode: str = 'topk'
    insufficient_policy: str = 'fail'
    best_threshold_policy: str = 'val_r2_then_sparsity'
    baseline: str = 'none'
    phase: str = 'phase1'
    dataset_path: Optional[str] = None
    threshold_grid: List[float] = None
    # A2: tolerance from config
    val_r2_tolerance: float = 0.002
    
    def __post_init__(self):
        if self.threshold_grid is None:
            self.threshold_grid = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
    
    @classmethod
    def from_yaml(cls, yaml_path: Path, cli_overrides: Optional[Dict] = None) -> 'Gate3Config':
        with open(yaml_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        seeds = config.get('seeds', {})
        eval_config = config.get('evaluation', {})
        
        instance = cls(
            gate=config.get('gate', 'gate3'),
            dataset_version=config.get('dataset_version', 'cartpole_ood_v1'),
            system=config.get('system', 'cartpole'),
            track=config.get('track', 'standardized'),
            n_train=config.get('n_train', 10),
            note=config.get('note', 'base'),
            data_seed=seeds.get('data', 0),
            vae_seed=seeds.get('vae', 0),
            gen_seed=seeds.get('gen', 0),
            method=config.get('method', 'generative'),
            aug_method=config.get('aug_method', 'vae_align'),
            aug_ratio=config.get('augmentation', {}).get('aug_ratio', 1.0),
            dx_policy=config.get('augmentation', {}).get('dx_policy', 'savgol'),
            n_generate=config.get('generator', {}).get('n_generate', 100),
            n_select=config.get('generator', {}).get('n_select', 10),
            gate1_baseline_run_id=config.get('comparison', {}).get('gate1_baseline_run_id'),
            gate2_baseline_run_id=config.get('comparison', {}).get('gate2_baseline_run_id'),
            mismatch_action=config.get('comparison', {}).get('mismatch_action', 'fail'),
            align_filter_on=config.get('filtering', {}).get('align_filter_on', True),
            align_mode=config.get('filtering', {}).get('align_mode', 'topk'),
            insufficient_policy=config.get('filtering', {}).get('insufficient_policy', 'fail'),
            best_threshold_policy=eval_config.get('best_threshold_policy', 'val_r2_then_sparsity'),
            threshold_grid=eval_config.get('threshold_grid', None),
            val_r2_tolerance=eval_config.get('val_r2_tolerance', 0.002),
        )
        
        if cli_overrides:
            for key, value in cli_overrides.items():
                if value is not None and hasattr(instance, key):
                    setattr(instance, key, value)
        
        return instance
    
    def to_dict(self) -> Dict:
        return {
            'gate': self.gate,
            'dataset_version': self.dataset_version,
            'system': self.system,
            'track': self.track,
            'n_train': self.n_train,
            'note': self.note,
            'seeds': {'data': self.data_seed, 'vae': self.vae_seed, 'gen': self.gen_seed},
            'method': self.method,
            'aug_method': self.aug_method,
            'aug_ratio': self.aug_ratio,
            'dx_policy': self.dx_policy,
            'n_generate': self.n_generate,
            'n_select': self.n_select,
            'gate1_baseline_run_id': self.gate1_baseline_run_id,
            'gate2_baseline_run_id': self.gate2_baseline_run_id,
            'baseline': self.baseline,
            'phase': self.phase,
            'align_filter_on': self.align_filter_on,
            'align_mode': self.align_mode,
            'insufficient_policy': self.insufficient_policy,
            'best_threshold_policy': self.best_threshold_policy,
            'threshold_grid': self.threshold_grid,
            'val_r2_tolerance': self.val_r2_tolerance,
        }


def build_sindy_library(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Build SINDy feature library for Cart-Pole system (21 features)"""
    pos = x[:, 0]
    vel = x[:, 1]
    theta = x[:, 2]
    omega = x[:, 3]
    ctrl = u[:, 0] if u.ndim > 1 else u
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    features = [
        np.ones_like(pos),
        pos, vel, sin_theta, cos_theta, omega, ctrl,
        pos**2, pos * vel, pos * sin_theta, pos * cos_theta, pos * omega, pos * ctrl,
        vel**2, vel * sin_theta, vel * cos_theta, vel * omega, vel * ctrl,
        sin_theta * cos_theta, omega**2, omega * ctrl,
    ]
    
    return np.column_stack(features)


def stlsq_fit(Theta: np.ndarray, dx: np.ndarray, threshold: float = 0.05, 
              max_iter: int = 20) -> np.ndarray:
    """Sequentially Thresholded Least Squares (STLSQ)"""
    n_states = dx.shape[1]
    Xi = np.linalg.lstsq(Theta, dx, rcond=None)[0]
    
    for _ in range(max_iter):
        small_mask = np.abs(Xi) < threshold
        Xi[small_mask] = 0
        
        for j in range(n_states):
            big_idx = np.where(np.abs(Xi[:, j]) >= threshold)[0]
            if len(big_idx) > 0:
                Xi[big_idx, j] = np.linalg.lstsq(
                    Theta[:, big_idx], dx[:, j], rcond=None
                )[0]
    
    return Xi


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared score (per-target mean, Gate1 consistent)"""
    n_targets = y_true.shape[1]
    r2_per_target = []
    
    for i in range(n_targets):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
        if ss_tot < 1e-10:
            r2_per_target.append(0.0)
        else:
            r2_per_target.append(1 - ss_res / ss_tot)
    
    return np.mean(r2_per_target)


def compute_sparsity(Xi: np.ndarray, eps: float = 1e-6) -> float:
    """Compute sparsity ratio"""
    return np.mean(np.abs(Xi) < eps)


def compute_indices_hash(indices: np.ndarray) -> str:
    """Compute hash of train indices for verification"""
    return hashlib.md5(indices.tobytes()).hexdigest()[:8]


def select_best_threshold(sweep_results: List[Dict], policy: str = 'val_r2_then_sparsity',
                          tolerance: float = 0.002) -> Dict:
    """
    Select best threshold with proper policy
    A2: tolerance from config (default 0.002)
    """
    if not sweep_results:
        raise ValueError("Empty sweep results")
    
    if policy == 'val_r2_only':
        return max(sweep_results, key=lambda x: x['val_r2'])
    
    # val_r2_then_sparsity (default)
    sorted_results = sorted(sweep_results, key=lambda x: x['val_r2'], reverse=True)
    best_val_r2 = sorted_results[0]['val_r2']
    
    # A2: Use configurable tolerance (default 0.002)
    tied = [r for r in sorted_results if best_val_r2 - r['val_r2'] <= tolerance]
    
    if len(tied) > 1:
        tied = sorted(tied, key=lambda x: x['sparsity'], reverse=True)
    
    best = tied[0]
    best['selection_reason'] = f"{policy}: val_r2={best['val_r2']:.4f}, sparsity={best['sparsity']:.1%}, tolerance={tolerance}"
    return best


class Gate3Runner:
    """Gate3 Experiment Runner"""
    
    def __init__(self, config: Gate3Config, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.run_id = self._generate_run_id()
        self.results_dir = self._get_results_dir()
        self.dataset = None
        self.vae = None
        self.teacher = None
        self.esindy_coefficients = None
        # A1: Track indices source
        self.train_indices_source = None
        self.train_indices_hash = None
        # dx source key (Gate0/Gate1 consistent)
        self.dx_source_key = None
    
    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        git_sha = 'nogit'
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                git_sha = result.stdout.strip()[:7]
        except Exception:
            pass
        return f"{timestamp}_{git_sha}_{self.config.note}"
    
    def _get_results_dir(self) -> Path:
        return (
            self.project_root / 'results' / self.config.dataset_version /
            self.config.gate / self.config.track / self.config.method /
            f'n{self.config.n_train}' / f'seed{self.config.data_seed}' / self.run_id
        )
    
    def _find_dataset_path(self) -> Path:
        if self.config.dataset_path:
            return Path(self.config.dataset_path)
        
        path1 = self.project_root / 'data' / self.config.dataset_version / 'dataset.npz'
        if path1.exists():
            return path1
        
        path2 = self.project_root / 'data' / 'cartpole' / self.config.dataset_version / 'dataset.npz'
        if path2.exists():
            return path2
        
        raise FileNotFoundError(f"Dataset not found")
    
    def _load_gate1_train_indices(self, n_total: int) -> np.ndarray:
        """
        A1: Load train_indices from Gate1 baseline manifest
        If train_indices not saved, reconstruct from seed (same RNG logic)
        """
        if not self.config.gate1_baseline_run_id:
            return None
        
        gate1_root = self.project_root / 'results' / self.config.dataset_version / 'gate1'
        matches = list(gate1_root.rglob(f'*{self.config.gate1_baseline_run_id}*/manifest.json'))
        
        if not matches:
            raise FileNotFoundError(
                f"Gate1 baseline manifest not found: {self.config.gate1_baseline_run_id}"
            )
        
        with open(matches[0], encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Case 1: train_indices saved in manifest (ideal)
        if 'train_indices' in manifest:
            indices = np.array(manifest['train_indices'])
            print(f"  [A1] Loaded train_indices from Gate1 manifest: {len(indices)} indices")
            self.train_indices_source = 'gate1_manifest'
            return indices
        
        # Case 2: Reconstruct from seed (fallback for older Gate1 runs)
        gate1_seed = manifest.get('config', {}).get('seed', None)
        gate1_n_train = manifest.get('config', {}).get('n_train', None)
        
        if gate1_seed is None or gate1_n_train is None:
            raise ValueError(
                f"Gate1 manifest missing both 'train_indices' and 'config.seed/n_train'. "
                f"Cannot reconstruct indices. Run ID: {self.config.gate1_baseline_run_id}"
            )
        
        # Verify consistency
        if gate1_n_train != self.config.n_train:
            raise ValueError(
                f"n_train mismatch: Gate1={gate1_n_train}, Gate3={self.config.n_train}"
            )
        
        # Reconstruct using same RNG logic as Gate1
        print(f"  [A1] WARNING: train_indices not in Gate1 manifest. Reconstructing from seed={gate1_seed}")
        rng = np.random.default_rng(gate1_seed)
        indices = rng.choice(n_total, gate1_n_train, replace=False)
        
        self.train_indices_source = f'reconstructed_from_gate1_seed={gate1_seed}'
        return indices
    
    def _load_dataset(self) -> Dict:
        dataset_path = self._find_dataset_path()
        print(f"  Dataset path: {dataset_path}")
        
        data = np.load(dataset_path)
        
        train_x_all = data['train_x']
        n_total = train_x_all.shape[0]
        
        # A1: Load indices from Gate1 if baseline specified
        if self.config.gate1_baseline_run_id:
            indices = self._load_gate1_train_indices(n_total)
            # train_indices_source is set inside _load_gate1_train_indices()
        else:
            # Fallback: generate indices from seed
            rng = np.random.default_rng(self.config.data_seed)
            if self.config.n_train < n_total:
                indices = rng.choice(n_total, self.config.n_train, replace=False)
            else:
                indices = np.arange(n_total)
            self.train_indices_source = f'rng(data_seed={self.config.data_seed})'
        
        # Sort indices for consistency
        indices = np.sort(indices)
        self.train_indices_hash = compute_indices_hash(indices)
        
        # Track-based dx key selection (Gate0/Gate1 consistent)
        if self.config.track == 'standardized':
            dx_suffix = '_savgol'
        else:  # author_recommended
            dx_suffix = ''
        
        train_dx_key = f'train_dx{dx_suffix}'
        val_dx_key = f'val_dx{dx_suffix}'
        test_dx_key = f'test_dx{dx_suffix}'
        
        # Verify keys exist
        for key in [train_dx_key, val_dx_key, test_dx_key]:
            if key not in data:
                raise KeyError(f"Dataset missing key: {key}")
        
        self.dx_source_key = f'dx{dx_suffix}'  # For manifest
        
        print(f"  Train indices source: {self.train_indices_source}")
        print(f"  Train indices hash: {self.train_indices_hash}")
        print(f"  Train indices: {indices.tolist()}")
        print(f"  dx_source_key: {self.dx_source_key}")
        
        self.dataset = {
            'train_x': train_x_all[indices],
            'train_u': data['train_u'][indices],
            'train_dx': data[train_dx_key][indices],
            'val_x': data['val_x'],
            'val_u': data['val_u'],
            'val_dx': data[val_dx_key],
            'test_x': data['test_x'],
            'test_u': data['test_u'],
            'test_dx': data[test_dx_key],
            'dt': float(data['dt']),
            'train_indices': indices,
        }
        return self.dataset
    
    def _train_vae(self) -> None:
        from src.generative.vae import TrajectoryVAE, VAEConfig
        
        vae_config = VAEConfig(
            latent_dim=8,
            hidden_dim=128,
            state_dim=4,
            epochs=100,
            batch_size=32,
            lr=1e-3,
            beta_start=0.01,
            beta_end=0.5,
            warmup_epochs=30,
            n_generate=self.config.n_generate,
            n_select=self.config.n_select,
            vae_seed=self.config.vae_seed,
            gen_seed=self.config.gen_seed,
        )
        
        seq_len = self.dataset['train_x'].shape[1]
        self.vae = TrajectoryVAE(vae_config, seq_len=seq_len)
        
        print(f"\n[VAE Training] vae_seed={self.config.vae_seed}")
        self.vae.fit(
            train_x=self.dataset['train_x'],
            val_x=self.dataset['val_x'],
            verbose=True
        )
    
    def _load_teacher(self) -> None:
        if not self.config.gate1_baseline_run_id:
            print("[WARN] No gate1_baseline_run_id provided. Teacher alignment disabled.")
            self.teacher = None
            return
        
        from src.generative.alignment import TeacherAlignment
        
        gate1_root = self.project_root / 'results' / self.config.dataset_version / 'gate1'
        self.teacher = TeacherAlignment.from_gate1_run(
            gate1_results_dir=gate1_root,
            run_id=self.config.gate1_baseline_run_id,
            expected_track=self.config.track,
            expected_n_train=self.config.n_train,
            expected_data_seed=self.config.data_seed,
            mismatch_action=self.config.mismatch_action,
        )
        print(f"[Teacher] Loaded from {self.config.gate1_baseline_run_id}")
    
    def _validate_teacher_requirement(self) -> None:
        """
        B: Validate teacher is available when align filter is on
        """
        # Baselines that don't need teacher
        no_teacher_baselines = ['copy_only', 'noise_aug', 'gen_only', 'random_select']
        
        if self.config.baseline in no_teacher_baselines:
            return  # OK, teacher not needed
        
        # M1 (baseline='none') with align filter needs teacher
        if self.config.align_filter_on and self.config.align_mode in ['topk', 'threshold']:
            if self.teacher is None:
                raise ValueError(
                    f"Teacher alignment is required when align_filter_on=True and "
                    f"baseline='{self.config.baseline}'. "
                    f"Please provide --gate1_baseline_run_id or set align_filter_on=False."
                )
    
    def _run_augmentation(self):
        # B: Validate teacher requirement
        self._validate_teacher_requirement()
        
        from src.augmentation.generative_augmentor import GenerativeAugmentor
        
        augmentor_config = {
            'augmentation': {
                'aug_ratio': self.config.aug_ratio,
                'dx_policy': self.config.dx_policy,
                'derivative_params': {'window': 11, 'polyorder': 3},
            },
            'generator': {
                'n_generate': self.config.n_generate,
                'n_select': self.config.n_select,
                'temperature': 1.0,
            },
            'filtering': {
                'sanity_on': True,
                'dedup_on': True,
                'align_filter_on': self.config.align_filter_on,
                'align_mode': self.config.align_mode,
                'insufficient_policy': self.config.insufficient_policy,
                'topk': self.config.n_select,
            },
            'seeds': {
                'data': self.config.data_seed,
                'vae': self.config.vae_seed,
                'gen': self.config.gen_seed,
            },
        }
        
        augmentor = GenerativeAugmentor(
            config=augmentor_config,
            vae_model=self.vae,
            teacher_alignment=self.teacher,
        )
        
        baseline_method = self.config.baseline
        if baseline_method not in ['none', 'gen_only', 'copy_only', 'noise_aug', 'random_select']:
            baseline_method = 'none'
        
        result = augmentor.augment(
            train_x=self.dataset['train_x'],
            train_u=self.dataset['train_u'],
            train_dx=self.dataset['train_dx'],
            dt=self.dataset['dt'],
            baseline_method=baseline_method,
        )
        
        return result
    
    def _run_esindy(self, train_x: np.ndarray, train_u: np.ndarray, 
                    train_dx: np.ndarray) -> Dict:
        """Run E-SINDy with proper threshold selection policy"""
        print(f"\n[E-SINDy] Running evaluation...")
        print(f"  Train data shape: {train_x.shape}")
        print(f"  Threshold policy: {self.config.best_threshold_policy}")
        print(f"  Val R2 tolerance: {self.config.val_r2_tolerance}")
        
        N, T, state_dim = train_x.shape
        
        x_flat = train_x.reshape(N * T, state_dim)
        u_flat = train_u.reshape(N * T, -1)
        dx_flat = train_dx.reshape(N * T, state_dim)
        
        Theta_train = build_sindy_library(x_flat, u_flat)
        
        val_x_flat = self.dataset['val_x'].reshape(-1, state_dim)
        val_u_flat = self.dataset['val_u'].reshape(-1, 1)
        val_dx_flat = self.dataset['val_dx'].reshape(-1, state_dim)
        Theta_val = build_sindy_library(val_x_flat, val_u_flat)
        
        threshold_grid = self.config.threshold_grid
        print(f"  Threshold grid: {threshold_grid}")
        
        sweep_results = []
        for thresh in threshold_grid:
            Xi = stlsq_fit(Theta_train, dx_flat, threshold=thresh)
            
            dx_pred_train = Theta_train @ Xi
            train_r2 = compute_r2(dx_flat, dx_pred_train)
            
            dx_pred_val = Theta_val @ Xi
            val_r2 = compute_r2(val_dx_flat, dx_pred_val)
            
            sparsity = compute_sparsity(Xi)
            
            sweep_results.append({
                'threshold': thresh,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'sparsity': sparsity,
                'Xi': Xi,
            })
        
        # A2: Use tolerance from config
        best = select_best_threshold(
            sweep_results, 
            self.config.best_threshold_policy,
            tolerance=self.config.val_r2_tolerance
        )
        best_Xi = best['Xi']
        best_threshold = best['threshold']
        
        self.esindy_coefficients = best_Xi
        
        # Final evaluation
        dx_pred_train = Theta_train @ best_Xi
        train_r2 = compute_r2(dx_flat, dx_pred_train)
        
        dx_pred_val = Theta_val @ best_Xi
        val_r2 = compute_r2(val_dx_flat, dx_pred_val)
        
        test_x_flat = self.dataset['test_x'].reshape(-1, state_dim)
        test_u_flat = self.dataset['test_u'].reshape(-1, 1)
        test_dx_flat = self.dataset['test_dx'].reshape(-1, state_dim)
        Theta_test = build_sindy_library(test_x_flat, test_u_flat)
        dx_pred_test = Theta_test @ best_Xi
        test_r2 = compute_r2(test_dx_flat, dx_pred_test)
        
        sparsity = compute_sparsity(best_Xi)
        
        metrics = {
            'train_r2': float(train_r2),
            'val_r2': float(val_r2),
            'test_r2': float(test_r2),
            'sparsity': float(sparsity),
            'best_threshold': float(best_threshold),
            'selection_reason': best.get('selection_reason', ''),
            'n_train_samples': int(N),
            'n_train_points': int(N * T),
        }
        
        print(f"  Train R2: {train_r2:.4f}")
        print(f"  Val R2: {val_r2:.4f}")
        print(f"  Test R2: {test_r2:.4f}")
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  Best threshold: {best_threshold}")
        print(f"  Selection: {best.get('selection_reason', 'N/A')}")
        
        return metrics
    
    def _compute_delta(self, metrics: Dict) -> Dict:
        deltas = {}
        
        if self.config.gate1_baseline_run_id:
            gate1_root = self.project_root / 'results' / self.config.dataset_version / 'gate1'
            matches = list(gate1_root.rglob(f'*{self.config.gate1_baseline_run_id}*/metrics.json'))
            
            if matches:
                with open(matches[0], encoding='utf-8') as f:
                    gate1_metrics = json.load(f)
                
                # Gate1 metrics structure: splits.{split}.r2_mean
                gate1_test_r2 = gate1_metrics.get('splits', {}).get('test', {}).get('r2_mean', 0)
                gate1_val_r2 = gate1_metrics.get('splits', {}).get('val', {}).get('r2_mean', 0)
                gate1_train_r2 = gate1_metrics.get('splits', {}).get('train', {}).get('r2_mean', 0)
                
                deltas['vs_gate1'] = {
                    'test_r2_delta_abs': metrics['test_r2'] - gate1_test_r2,
                    'val_r2_delta_abs': metrics['val_r2'] - gate1_val_r2,
                    'train_r2_delta_abs': metrics['train_r2'] - gate1_train_r2,
                    'gate1_test_r2': gate1_test_r2,
                    'gate1_val_r2': gate1_val_r2,
                    'gate1_train_r2': gate1_train_r2,
                }
        
        return deltas
    
    def _save_artifacts(self, aug_result, metrics: Dict, deltas: Dict) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # A1: Include train_indices tracking in manifest
        manifest = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'gate1_baseline_run_id': self.config.gate1_baseline_run_id,
            'gate2_baseline_run_id': self.config.gate2_baseline_run_id,
            'seed': self.config.data_seed,
            'track': self.config.track,
            'n_train': self.config.n_train,
            # A1: Track indices source and hash
            'train_indices_source': self.train_indices_source,
            'train_indices_hash': self.train_indices_hash,
            'train_indices': self.dataset['train_indices'].tolist() if self.dataset else [],
            # dx source key (Gate0/Gate1 consistent)
            'dx_source_key': self.dx_source_key,
        }
        with open(self.results_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(manifest), f, indent=2)
        
        metrics_full = {**metrics, 'deltas': deltas}
        with open(self.results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(metrics_full), f, indent=2)
        
        if aug_result is not None:
            aug_manifest = aug_result.to_manifest_dict()
            with open(self.results_dir / 'aug_manifest.json', 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(aug_manifest), f, indent=2)
        
        if self.esindy_coefficients is not None:
            import csv
            coef_path = self.results_dir / 'sindy_coefficients.csv'
            feature_names = [
                '1', 'x', 'x_dot', 'sin_theta', 'cos_theta', 'theta_dot', 'u',
                'x^2', 'x*x_dot', 'x*sin', 'x*cos', 'x*theta_dot', 'x*u',
                'x_dot^2', 'x_dot*sin', 'x_dot*cos', 'x_dot*theta_dot', 'x_dot*u',
                'sin*cos', 'theta_dot^2', 'theta_dot*u'
            ]
            with open(coef_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['term', 'dx_0', 'dx_1', 'dx_2', 'dx_3'])
                for i, row in enumerate(self.esindy_coefficients):
                    term_name = feature_names[i] if i < len(feature_names) else f'term_{i}'
                    writer.writerow([term_name] + [f'{v:.6f}' for v in row])
        
        print(f"\n[Artifacts] Saved to {self.results_dir}")
    
    def run(self) -> Dict:
        print("\n" + "="*60)
        print("  Gate3 Generative Augmentation Runner")
        print("="*60)
        print(f"  run_id: {self.run_id}")
        print(f"  track: {self.config.track}")
        print(f"  n_train: {self.config.n_train}")
        print(f"  seeds: data={self.config.data_seed}, vae={self.config.vae_seed}, gen={self.config.gen_seed}")
        print(f"  baseline: {self.config.baseline}")
        print(f"  phase: {self.config.phase}")
        
        if self.config.phase == 'phase-1':
            print("\n[Phase -1] Smoke test with copy_only...")
            self.config.baseline = 'copy_only'
        
        print("\n[Step 1] Loading dataset...")
        self._load_dataset()
        print(f"  train_x: {self.dataset['train_x'].shape}")
        
        print("\n[Step 2] Loading teacher...")
        self._load_teacher()
        
        if self.config.baseline not in ['copy_only', 'noise_aug']:
            print("\n[Step 3] Training VAE...")
            self._train_vae()
        else:
            print(f"\n[Step 3] Skipping VAE training (baseline={self.config.baseline})")
            self.vae = None
        
        if self.config.phase == 'phase0':
            print("\n[Phase 0] VAE training complete. Skipping E-SINDy.")
            self._save_artifacts(None, {}, {})
            return {'status': 'phase0_complete', 'vae_trained': self.vae is not None}
        
        print("\n[Step 4] Running augmentation...")
        aug_result = self._run_augmentation()
        print(f"  n_augmented: {aug_result.n_augmented}")
        print(f"  status: {aug_result.status}")
        
        if not aug_result.success:
            print(f"\n[ERROR] Augmentation failed: {aug_result.status}")
            self._save_artifacts(aug_result, {}, {})
            return {'status': 'augmentation_failed', 'reason': aug_result.status}
        
        print("\n[Step 5] Combining train + augmented data...")
        if aug_result.n_augmented > 0:
            train_x = np.concatenate([self.dataset['train_x'], aug_result.x_aug], axis=0)
            train_u = np.concatenate([self.dataset['train_u'], aug_result.u_aug], axis=0)
            train_dx = np.concatenate([self.dataset['train_dx'], aug_result.dx_aug], axis=0)
        else:
            train_x = self.dataset['train_x']
            train_u = self.dataset['train_u']
            train_dx = self.dataset['train_dx']
        print(f"  Combined train_x: {train_x.shape}")
        
        print("\n[Step 6] Running E-SINDy...")
        metrics = self._run_esindy(train_x, train_u, train_dx)
        
        print("\n[Step 7] Computing deltas...")
        deltas = self._compute_delta(metrics)
        if 'vs_gate1' in deltas:
            print(f"  vs Gate1: dR2 = {deltas['vs_gate1']['test_r2_delta_abs']:+.4f}")
            print(f"  Gate1 Test R2: {deltas['vs_gate1']['gate1_test_r2']:.4f}")
        
        print("\n[Step 8] Saving artifacts...")
        self._save_artifacts(aug_result, metrics, deltas)
        
        print("\n" + "="*60)
        print("  Gate3 Run Complete")
        print("="*60)
        
        return {
            'status': 'success',
            'run_id': self.run_id,
            'results_dir': str(self.results_dir),
            'metrics': metrics,
            'deltas': deltas,
        }