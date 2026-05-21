#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Generative Augmentor
==========================
VAE based trajectory generation + Teacher Alignment filtering.

Lock compliance:
- Lock-3: dx is computed externally (teacher generation prohibited)
- Lock-4: copy prohibited when candidates insufficient
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from pathlib import Path
import json
from datetime import datetime


@dataclass
class GenerativeAugmentationResult:
    """Gate3 augmentation result"""
    x_aug: np.ndarray
    u_aug: np.ndarray
    dx_aug: np.ndarray
    n_original: int
    n_augmented: int
    n_total: int
    success: bool
    status: str
    filtering_stats: Dict
    generator_info: Dict
    teacher_info: Dict
    dx_policy: str
    selected_indices: np.ndarray
    align_scores: np.ndarray
    
    def to_manifest_dict(self) -> Dict:
        """Convert to aug_manifest.json dict"""
        return {
            'version': 'gate3_v1',
            'timestamp': datetime.now().isoformat(),
            'generator': self.generator_info,
            'teacher': self.teacher_info,
            'generation': {
                'n_generated': self.filtering_stats.get('n_input', 0),
                'n_selected': self.n_augmented,
                'selected_indices': self.selected_indices.tolist() if self.selected_indices is not None and len(self.selected_indices) > 0 else [],
            },
            'filtering': self.filtering_stats,
            'dx_policy': self.dx_policy,
            'health_check': {
                'all_finite': bool(np.all(np.isfinite(self.x_aug))) if self.x_aug is not None and len(self.x_aug) > 0 else False,
                'n_original': self.n_original,
                'n_augmented': self.n_augmented,
                'n_total': self.n_total,
            },
            'status': self.status,
            'success': self.success,
        }


class GenerativeAugmentor:
    """
    VAE based trajectory generation + Teacher Alignment filtering
    
    Workflow:
    1. Generate n_generate candidates with VAE
    2. Sanity + Dedup filtering
    3. Compute dx (according to dx_policy)
    4. Compute align score + Top-k selection
    5. Return result
    """
    
    def __init__(self, config: Dict, vae_model=None, teacher_alignment=None):
        """
        Args:
            config: YAML config dict
            vae_model: Trained TrajectoryVAE
            teacher_alignment: TeacherAlignment object
        """
        self.config = config
        self.vae = vae_model
        self.teacher = teacher_alignment
        
        self.aug_config = config.get('augmentation', {})
        self.gen_config = config.get('generator', {})
        self.filter_config = config.get('filtering', {})
        self.seeds = config.get('seeds', {})
        
        self.dx_policy = self.aug_config.get('dx_policy', 'savgol')
        self.derivative_params = self.aug_config.get('derivative_params', {
            'window': 11, 'polyorder': 3
        })
    
    def compute_dx(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Compute dx (Lock-3: teacher generation prohibited)"""
        if self.dx_policy == 'savgol':
            from scipy.signal import savgol_filter
            
            window = self.derivative_params.get('window', 11)
            polyorder = self.derivative_params.get('polyorder', 3)
            
            N, T, state_dim = x.shape
            dx = np.zeros_like(x)
            
            for i in range(N):
                for j in range(state_dim):
                    dx[i, :, j] = savgol_filter(
                        x[i, :, j], window, polyorder, deriv=1, delta=dt
                    )
            
            return dx
        
        elif self.dx_policy == 'central_diff':
            dx = np.zeros_like(x)
            dx[:, 1:-1, :] = (x[:, 2:, :] - x[:, :-2, :]) / (2 * dt)
            dx[:, 0, :] = (x[:, 1, :] - x[:, 0, :]) / dt
            dx[:, -1, :] = (x[:, -1, :] - x[:, -2, :]) / dt
            return dx
        
        else:
            raise ValueError(f"Unknown dx_policy: {self.dx_policy}")
    
    def augment(
        self,
        train_x: np.ndarray,
        train_u: np.ndarray,
        train_dx: np.ndarray,
        dt: float,
        baseline_method: Optional[Literal['none', 'gen_only', 'copy_only', 
                                          'noise_aug', 'random_select']] = 'none',
    ) -> GenerativeAugmentationResult:
        """Perform data augmentation"""
        from src.augmentation.generative_baselines import copy_only_augment, noise_augment
        
        N_train, T, state_dim = train_x.shape
        aug_ratio = self.aug_config.get('aug_ratio', 1.0)
        n_target = int(N_train * aug_ratio)
        
        # Use seed from config (P0-C fix)
        gen_seed = self.seeds.get('gen', 0)
        
        # M3: Copy-only
        if baseline_method == 'copy_only':
            x_aug, u_aug, parent_indices = copy_only_augment(
                train_x, train_u, n_target, seed=gen_seed
            )
            # P3: Copy dx from parent (not recompute)
            dx_aug = train_dx[parent_indices].copy()
            
            return GenerativeAugmentationResult(
                x_aug=x_aug,
                u_aug=u_aug,
                dx_aug=dx_aug,
                n_original=N_train,
                n_augmented=len(x_aug),
                n_total=N_train + len(x_aug),
                success=True,
                status='copy_only',
                filtering_stats={
                    'method': 'copy_only',
                    'parent_indices': parent_indices,
                    'seed': gen_seed,
                },
                generator_info={'type': 'copy_only'},
                teacher_info={},
                dx_policy=self.dx_policy,
                selected_indices=np.arange(len(x_aug)),
                align_scores=np.array([]),
            )
        
        # M4: Noise augmentation
        if baseline_method == 'noise_aug':
            noise_std = self.aug_config.get('noise_std', 0.01)
            x_aug, u_aug, parent_indices = noise_augment(
                train_x, train_u, n_target, 
                noise_std=noise_std, seed=gen_seed
            )
            dx_aug = self.compute_dx(x_aug, dt)
            
            return GenerativeAugmentationResult(
                x_aug=x_aug,
                u_aug=u_aug,
                dx_aug=dx_aug,
                n_original=N_train,
                n_augmented=len(x_aug),
                n_total=N_train + len(x_aug),
                success=True,
                status='noise_aug',
                filtering_stats={
                    'method': 'noise_aug',
                    'parent_indices': parent_indices,
                    'noise_std': noise_std,
                    'seed': gen_seed,
                },
                generator_info={'type': 'noise_aug', 'noise_std': noise_std},
                teacher_info={},
                dx_policy=self.dx_policy,
                selected_indices=np.arange(len(x_aug)),
                align_scores=np.array([]),
            )
        
        # VAE-based generation (M1, M2, M5)
        if self.vae is None:
            raise ValueError("VAE model not provided. Train VAE first.")
        
        n_generate = self.gen_config.get('n_generate', 100)
        temperature = self.gen_config.get('temperature', 1.0)
        
        x_generated = self.vae.sample(n_generate, temperature=temperature, seed=gen_seed)
        
        rng = np.random.default_rng(gen_seed)
        u_indices = rng.choice(N_train, n_generate, replace=True)
        u_generated = train_u[u_indices]
        
        dx_generated = self.compute_dx(x_generated, dt)
        
        # Filtering
        from src.generative.filtering import (
            SanityFilter, DedupFilter, AlignScoreFilter,
            apply_filters, FilterStatus
        )
        
        sanity_filter = SanityFilter()
        dedup_filter = DedupFilter(threshold=self.filter_config.get('dedup_threshold', 0.01))
        
        topk = self.filter_config.get('topk', n_target)
        align_filter = AlignScoreFilter(
            mode=self.filter_config.get('align_mode', 'topk'),
            topk=topk,
            threshold=self.filter_config.get('align_threshold', 0.1)
        )
        
        align_filter_on = self.filter_config.get('align_filter_on', True)
        if baseline_method == 'gen_only':
            align_filter_on = False
        elif baseline_method == 'random_select':
            align_filter_on = False
        
        filter_result = apply_filters(
            x_aug=x_generated,
            dx_aug=dx_generated,
            u_aug=u_generated,
            align_scorer=self.teacher,
            n_target=n_target,
            sanity_filter=sanity_filter,
            dedup_filter=dedup_filter,
            align_filter=align_filter,
            align_filter_on=align_filter_on,
            insufficient_policy=self.filter_config.get('insufficient_policy', 'fail'),
            max_retry=self.filter_config.get('max_retry', 3),
            select_seed=gen_seed,  # P0-C fix
        )
        
        if filter_result.status == FilterStatus.INSUFFICIENT_CANDIDATES:
            if self.filter_config.get('insufficient_policy', 'fail') == 'fail':
                return GenerativeAugmentationResult(
                    x_aug=np.array([]),
                    u_aug=np.array([]),
                    dx_aug=np.array([]),
                    n_original=N_train,
                    n_augmented=0,
                    n_total=N_train,
                    success=False,
                    status='insufficient_candidates',
                    filtering_stats=filter_result.to_dict(),
                    generator_info=self.vae.get_model_info() if hasattr(self.vae, 'get_model_info') else {},
                    teacher_info=self.teacher.get_teacher_info_dict() if self.teacher else {},
                    dx_policy=self.dx_policy,
                    selected_indices=filter_result.selected_indices,
                    align_scores=filter_result.selected_align_scores if filter_result.selected_align_scores is not None else np.array([]),
                )
        
        selected_indices = filter_result.selected_indices
        x_aug = x_generated[selected_indices]
        u_aug = u_generated[selected_indices]
        dx_aug = dx_generated[selected_indices]
        
        return GenerativeAugmentationResult(
            x_aug=x_aug,
            u_aug=u_aug,
            dx_aug=dx_aug,
            n_original=N_train,
            n_augmented=len(x_aug),
            n_total=N_train + len(x_aug),
            success=True,
            status='success' if baseline_method == 'none' else baseline_method,
            filtering_stats=filter_result.to_dict(),
            generator_info=self.vae.get_model_info() if hasattr(self.vae, 'get_model_info') else {},
            teacher_info=self.teacher.get_teacher_info_dict() if self.teacher else {},
            dx_policy=self.dx_policy,
            selected_indices=selected_indices,
            align_scores=filter_result.selected_align_scores if filter_result.selected_align_scores is not None else np.array([]),
        )
    
    def save_manifest(self, result: GenerativeAugmentationResult, output_dir: Path) -> Path:
        """Save aug_manifest.json"""
        manifest = result.to_manifest_dict()
        manifest['filtering']['align_score_definition'] = self.filter_config.get(
            'align_score_definition',
            "mean_t || dx_aug - Theta(x_aug, u_aug) @ Xi_bar ||^2"
        )
        
        output_path = output_dir / 'aug_manifest.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        return output_path