#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Baseline Augmentation Methods
===================================
M3 (Copy-only), M4 (Noise-Aug), M5 (Random-select) implementations.
"""

import numpy as np
from typing import Tuple, List


def copy_only_augment(train_x: np.ndarray, train_u: np.ndarray,
                      n_aug: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    M3: Copy-only augmentation
    
    Random duplication from original train data
    
    Returns:
        x_aug, u_aug, parent_indices
    """
    rng = np.random.default_rng(seed)
    N_train = train_x.shape[0]
    
    parent_indices = rng.choice(N_train, n_aug, replace=True).tolist()
    
    x_aug = train_x[parent_indices].copy()
    u_aug = train_u[parent_indices].copy()
    
    return x_aug, u_aug, parent_indices


def noise_augment(train_x: np.ndarray, train_u: np.ndarray,
                  n_aug: int, noise_std: float = 0.01, 
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    M4: Noise augmentation (Step8 B2 pattern)
    
    Add Gaussian noise to original data
    
    Returns:
        x_aug, u_aug, parent_indices
    """
    rng = np.random.default_rng(seed)
    N_train = train_x.shape[0]
    
    parent_indices = rng.choice(N_train, n_aug, replace=True).tolist()
    
    x_base = train_x[parent_indices].copy()
    u_base = train_u[parent_indices].copy()
    
    # Add noise to states
    noise = rng.normal(0, noise_std, x_base.shape)
    x_aug = x_base + noise
    
    # u remains unchanged
    u_aug = u_base
    
    return x_aug, u_aug, parent_indices


def random_select_augment(x_generated: np.ndarray, u_generated: np.ndarray,
                          valid_mask: np.ndarray, n_select: int,
                          seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M5: Random selection (ignore align score)
    
    Random selection from sanity/dedup passed samples
    
    Returns:
        x_selected, u_selected, selected_indices
    """
    rng = np.random.default_rng(seed)
    
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)
    
    k = min(n_select, n_valid)
    selected_local = rng.choice(n_valid, k, replace=False)
    selected_indices = valid_indices[selected_local]
    
    x_selected = x_generated[selected_indices]
    u_selected = u_generated[selected_indices]
    
    return x_selected, u_selected, selected_indices