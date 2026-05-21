#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Filtering Module
======================
Lock-4: copy prohibited when candidates insufficient
Lock-3 (P0-C): Use config seed, not hardcoded
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal
from enum import Enum


class FilterStatus(Enum):
    SUCCESS = "success"
    INSUFFICIENT_CANDIDATES = "insufficient_candidates"
    FAILED = "failed"


@dataclass
class FilteringResult:
    """Filtering result"""
    status: FilterStatus
    selected_indices: np.ndarray
    n_input: int
    n_after_sanity: int
    n_after_dedup: int
    n_selected: int
    n_target: int
    reject_reasons: Dict[str, int]
    align_score_stats: Dict[str, float]
    selected_align_scores: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'n_input': self.n_input,
            'n_after_sanity': self.n_after_sanity,
            'n_after_dedup': self.n_after_dedup,
            'n_selected': self.n_selected,
            'n_target': self.n_target,
            'reject_rate': 1.0 - (self.n_selected / self.n_input) if self.n_input > 0 else 0.0,
            'duplicate_rate': (self.n_after_sanity - self.n_after_dedup) / self.n_after_sanity if self.n_after_sanity > 0 else 0.0,
            'reject_reasons': self.reject_reasons,
            'align_score_stats': self.align_score_stats,
            'error_message': self.error_message,
        }


class SanityFilter:
    """Sanity filter (always ON)"""
    
    def __init__(self, state_bounds: Optional[Dict[int, Tuple[float, float]]] = None):
        if state_bounds is None:
            self.state_bounds = {
                0: (-5.0, 5.0),
                1: (-10.0, 10.0),
                2: (-np.pi, np.pi),
                3: (-15.0, 15.0),
            }
        else:
            self.state_bounds = state_bounds
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        N = x.shape[0]
        valid_mask = np.ones(N, dtype=bool)
        reject_counts = {'nan_inf': 0, 'range_violation': 0}
        
        for i in range(N):
            traj = x[i]
            
            if np.any(~np.isfinite(traj)):
                valid_mask[i] = False
                reject_counts['nan_inf'] += 1
                continue
            
            for state_idx, (vmin, vmax) in self.state_bounds.items():
                if state_idx < traj.shape[1]:
                    if np.any(traj[:, state_idx] < vmin) or np.any(traj[:, state_idx] > vmax):
                        valid_mask[i] = False
                        reject_counts['range_violation'] += 1
                        break
        
        return valid_mask, reject_counts


class DedupFilter:
    """Deduplication filter"""
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
    
    def __call__(self, x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        if n_valid <= 1:
            return valid_mask
        
        valid_x = x[valid_indices]
        is_duplicate = np.zeros(n_valid, dtype=bool)
        
        for i in range(n_valid):
            if is_duplicate[i]:
                continue
            for j in range(i + 1, n_valid):
                if is_duplicate[j]:
                    continue
                mse = np.mean((valid_x[i] - valid_x[j])**2)
                if mse < self.threshold:
                    is_duplicate[j] = True
        
        updated_mask = valid_mask.copy()
        for idx, is_dup in zip(valid_indices, is_duplicate):
            if is_dup:
                updated_mask[idx] = False
        
        return updated_mask


class AlignScoreFilter:
    """Alignment score based filter"""
    
    def __init__(self, mode: Literal['topk', 'threshold'] = 'topk',
                 topk: int = 10, threshold: float = 0.1):
        self.mode = mode
        self.topk = topk
        self.threshold = threshold
    
    def __call__(self, scores: np.ndarray, valid_mask: np.ndarray,
                 apply_filter: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_indices]
        n_valid = len(valid_indices)
        
        if not apply_filter:
            return valid_indices, valid_scores
        
        if n_valid == 0:
            return np.array([], dtype=int), np.array([])
        
        if self.mode == 'topk':
            k = min(self.topk, n_valid)
            sorted_idx = np.argsort(valid_scores)[:k]
            selected_local = sorted_idx
        else:
            selected_local = np.where(valid_scores <= self.threshold)[0]
        
        selected_indices = valid_indices[selected_local]
        selected_scores = valid_scores[selected_local]
        
        return selected_indices, selected_scores


def apply_filters(
    x_aug: np.ndarray,
    dx_aug: np.ndarray,
    u_aug: np.ndarray,
    align_scorer,
    n_target: int,
    sanity_filter: Optional[SanityFilter] = None,
    dedup_filter: Optional[DedupFilter] = None,
    align_filter: Optional[AlignScoreFilter] = None,
    align_filter_on: bool = True,
    insufficient_policy: Literal['fail', 'proceed_partial'] = 'fail',
    max_retry: int = 0, # NOTE: Retry not implemented in Phase1. Use 0.
    select_seed: int = 0,  # P0-C: configurable seed
) -> FilteringResult:
    """Full filtering pipeline"""
    N = x_aug.shape[0]
    
    if sanity_filter is None:
        sanity_filter = SanityFilter()
    if dedup_filter is None:
        dedup_filter = DedupFilter()
    if align_filter is None:
        align_filter = AlignScoreFilter(topk=n_target)
    
    reject_reasons = {'nan_inf': 0, 'range_violation': 0, 'duplicate': 0, 'low_align_score': 0, 'random_downselect': 0}
    
    # Step 1: Sanity filter
    valid_mask, sanity_rejects = sanity_filter(x_aug)
    reject_reasons.update(sanity_rejects)
    n_after_sanity = valid_mask.sum()
    
    # Step 2: Dedup filter
    valid_mask = dedup_filter(x_aug, valid_mask)
    n_after_dedup = valid_mask.sum()
    reject_reasons['duplicate'] = n_after_sanity - n_after_dedup
    
    # Step 3: Compute align scores
    if align_scorer is not None and callable(getattr(align_scorer, 'compute_align_score', None)):
        align_scores = align_scorer.compute_align_score(x_aug, dx_aug, u_aug)
    elif callable(align_scorer):
        align_scores = align_scorer(x_aug, dx_aug, u_aug)
    else:
        # No teacher: use zeros (all equal)
        align_scores = np.zeros(N)
    
    valid_scores = align_scores[valid_mask]
    if len(valid_scores) > 0:
        align_score_stats = {
            'mean': float(np.mean(valid_scores)),
            'std': float(np.std(valid_scores)),
            'min': float(np.min(valid_scores)),
            'max': float(np.max(valid_scores)),
            'median': float(np.median(valid_scores)),
        }
    else:
        align_score_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    
    # Step 4: Align score filtering or random selection
    if align_filter_on and align_scorer is not None:
        selected_indices, selected_scores = align_filter(
            align_scores, valid_mask, apply_filter=True
        )
        reject_reasons['low_align_score'] = n_after_dedup - len(selected_indices)
    else:
        # Random selection (M2, M5) - use configurable seed
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > n_target:
            rng = np.random.default_rng(select_seed)
            choice_idx = rng.choice(len(valid_indices), n_target, replace=False)
            selected_indices = valid_indices[choice_idx]
            reject_reasons['random_downselect'] = len(valid_indices) - n_target
        else:
            selected_indices = valid_indices
        selected_scores = align_scores[selected_indices] if len(selected_indices) > 0 else np.array([])
    
    n_selected = len(selected_indices)
    
    # Lock-4: Insufficient candidates
    if n_selected < n_target:
        return FilteringResult(
            status=FilterStatus.INSUFFICIENT_CANDIDATES,
            selected_indices=selected_indices,
            n_input=N,
            n_after_sanity=n_after_sanity,
            n_after_dedup=n_after_dedup,
            n_selected=n_selected,
            n_target=n_target,
            reject_reasons=reject_reasons,
            align_score_stats=align_score_stats,
            selected_align_scores=selected_scores,
            error_message=f"Insufficient candidates: {n_selected}/{n_target}",
        )
    
    return FilteringResult(
        status=FilterStatus.SUCCESS,
        selected_indices=selected_indices,
        n_input=N,
        n_after_sanity=n_after_sanity,
        n_after_dedup=n_after_dedup,
        n_selected=n_selected,
        n_target=n_target,
        reject_reasons=reject_reasons,
        align_score_stats=align_score_stats,
        selected_align_scores=selected_scores,
    )