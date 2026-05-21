#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase 2-c Evaluation Script (v3 - MMR + Structure Metrics)
================================================================
Pool에서 선택 (Align/Random/MMR) → E-SINDy 학습 → 평가

Phase 2-c 수정사항:
- MMR 선택 모드 추가 (λ 파라미터)
- lexsort 기반 결정론적 tie-break (모든 모드)
- Teacher 기반 구조 지표 (Jaccard, Correlation, Term Frequency)
- k 부족 시 명시적 에러
- manifest 필수 필드 8개 락인

Usage:
    python src/experiments/evaluate_selection.py --pool_npz ... --teacher_run_dir ... --selection_mode align
    python src/experiments/evaluate_selection.py --pool_npz ... --teacher_run_dir ... --selection_mode mmr --mmr_lambda 0.5 --latent_npz ...
"""

import argparse
import json
import sys
import csv
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.paths import get_dataset_path, RESULTS_ROOT, generate_run_id
from src.contracts.schema_dataset_lite import validate_dataset_lite


def build_library(x, u):
    """Cart-Pole SINDy library (21 features)"""
    N, T, state_dim = x.shape
    x_flat = x.reshape(N * T, state_dim)
    u_flat = u.reshape(N * T, 1) if u.ndim == 3 else u.reshape(N * T, 1)
    
    pos, vel, theta, omega = x_flat[:, 0], x_flat[:, 1], x_flat[:, 2], x_flat[:, 3]
    ctrl = u_flat[:, 0]
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    
    features = [
        np.ones_like(pos), pos, vel, sin_theta, cos_theta, omega, ctrl,
        pos**2, pos*vel, pos*sin_theta, pos*cos_theta, pos*omega, pos*ctrl,
        vel**2, vel*sin_theta, vel*cos_theta, vel*omega, vel*ctrl,
        sin_theta*cos_theta, omega**2, omega*ctrl,
    ]
    return np.column_stack(features)


FEATURE_NAMES = [
    '1', 'x', 'x_dot', 'sin(theta)', 'cos(theta)', 'theta_dot', 'u',
    'x^2', 'x*x_dot', 'x*sin', 'x*cos', 'x*theta_dot', 'x*u',
    'x_dot^2', 'x_dot*sin', 'x_dot*cos', 'x_dot*theta_dot', 'x_dot*u',
    'sin*cos', 'theta_dot^2', 'theta_dot*u',
]


def stlsq(Theta, dx, threshold, max_iter=10):
    """Sequentially Thresholded Least Squares"""
    n_features = Theta.shape[1]
    n_targets = dx.shape[1]
    Xi = np.linalg.lstsq(Theta, dx, rcond=None)[0]
    
    for _ in range(max_iter):
        small_mask = np.abs(Xi) < threshold
        Xi[small_mask] = 0
        
        for j in range(n_targets):
            big_idx = np.where(~small_mask[:, j])[0]
            if len(big_idx) > 0:
                Xi[big_idx, j] = np.linalg.lstsq(Theta[:, big_idx], dx[:, j], rcond=None)[0]
    
    return Xi


def compute_r2(y_true, y_pred):
    """R² score (per-target mean)"""
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0))**2, axis=0)
    r2_per_target = 1 - ss_res / (ss_tot + 1e-10)
    return float(np.mean(r2_per_target))


def esindy_bootstrap(Theta, dx, n_bootstrap=20, threshold=0.05, seed=0):
    """E-SINDy bootstrap ensemble"""
    rng = np.random.default_rng(seed)
    N = Theta.shape[0]
    all_coefs = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(N, N, replace=True)
        Xi = stlsq(Theta[idx], dx[idx], threshold)
        all_coefs.append(Xi)
    
    coef_mean = np.mean(all_coefs, axis=0)
    coef_std = np.std(all_coefs, axis=0)
    
    # Term frequency: 각 term이 non-zero로 나타난 빈도
    term_freq = np.mean([np.abs(c) > 1e-12 for c in all_coefs], axis=0)
    
    return coef_mean, coef_std, term_freq


def select_best_threshold(Theta_train, dx_train, Theta_val, dx_val, 
                          threshold_grid, n_bootstrap=20, tolerance=0.002,
                          bootstrap_seed=42):
    """Validation-based threshold selection"""
    best_val_r2 = -np.inf
    best_threshold = threshold_grid[0]
    best_coef = None
    best_term_freq = None
    
    results = []
    for thresh in threshold_grid:
        coef, _, term_freq = esindy_bootstrap(Theta_train, dx_train, n_bootstrap, thresh, seed=bootstrap_seed)
        
        dx_pred_val = Theta_val @ coef
        val_r2 = compute_r2(dx_val, dx_pred_val)
        
        sparsity = np.mean(np.abs(coef) < 1e-10)
        results.append({'threshold': thresh, 'val_r2': val_r2, 'sparsity': sparsity})
        
        if val_r2 > best_val_r2 + tolerance:
            best_val_r2 = val_r2
            best_threshold = thresh
            best_coef = coef
            best_term_freq = term_freq
        elif abs(val_r2 - best_val_r2) <= tolerance:
            if sparsity > np.mean(np.abs(best_coef) < 1e-10):
                best_threshold = thresh
                best_coef = coef
                best_term_freq = term_freq
    
    return best_threshold, best_coef, best_term_freq, results


def load_gate1_train_indices(teacher_run_dir):
    """Gate1 manifest에서 train_indices 로드 (SSOT 강제)"""
    teacher_dir = Path(teacher_run_dir)
    manifest_path = teacher_dir / 'manifest.json'
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Gate1 manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    train_indices = manifest.get('train_indices')
    if train_indices is None:
        raise ValueError(f"Gate1 manifest missing 'train_indices': {manifest_path}")
    
    train_indices = np.array(train_indices)
    n_train = len(train_indices)
    
    config_n_train = manifest.get('config', {}).get('n_train')
    if config_n_train and config_n_train != n_train:
        raise ValueError(f"n_train mismatch: config={config_n_train}, indices={n_train}")
    
    print(f"  [SSOT] Gate1 train_indices loaded: n_train={n_train}")
    return train_indices, manifest


def load_teacher_coefficients(teacher_run_dir):
    """Gate1 Teacher 계수 로드"""
    coef_path = Path(teacher_run_dir) / 'sindy_coefficients.csv'
    if not coef_path.exists():
        raise FileNotFoundError(f"Teacher coefficients not found: {coef_path}")
    
    with open(coef_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        data_cols = [i for i, h in enumerate(header) if h.startswith('dx_')]
        if not data_cols:
            data_cols = list(range(1, len(header)))
        rows = [[float(row[i]) for i in data_cols] for row in reader]
    return np.array(rows)


# =============================================================================
# Selection Functions with Deterministic Tie-Break (lexsort)
# =============================================================================

def select_align_topk(align_scores, candidate_indices, k):
    """
    Align Top-k selection with deterministic tie-break
    
    Args:
        align_scores: (n_generate,) - lower is better
        candidate_indices: valid candidate indices
        k: number to select
    
    Returns:
        selected_global: (k,) global indices
    """
    n_valid = len(candidate_indices)
    if n_valid < k:
        raise ValueError(f"Insufficient candidates: {n_valid} < k={k}")
    
    valid_scores = align_scores[candidate_indices]
    global_idx = candidate_indices
    
    # lexsort: primary=align_score (lower better), secondary=global_idx (lower first)
    # lexsort sorts by last key first, so (global_idx, scores) → sort by scores, tie-break by global_idx
    order = np.lexsort((global_idx, valid_scores))
    selected_local = order[:k]
    selected_global = candidate_indices[selected_local]
    
    return selected_global


def select_random(candidate_indices, k, seed):
    """
    Random selection with deterministic seed
    
    Args:
        candidate_indices: valid candidate indices
        k: number to select
        seed: random seed
    
    Returns:
        selected_global: (k,) global indices
    """
    n_valid = len(candidate_indices)
    if n_valid < k:
        raise ValueError(f"Insufficient candidates: {n_valid} < k={k}")
    
    rng = np.random.default_rng(seed)
    choice_local = rng.choice(n_valid, k, replace=False)
    selected_global = candidate_indices[choice_local]
    
    return selected_global


def select_mmr(align_scores, latent_mu, candidate_indices, k, lambda_mmr=0.5):
    """
    MMR (Maximal Marginal Relevance) selection with deterministic tie-break
    
    MMR(i) = λ * utility(i) + (1-λ) * diversity(i)
    - utility = -align_score (higher is better)
    - diversity = min distance to already selected (farthest-point style)
    
    Args:
        align_scores: (n_generate,) - lower is better
        latent_mu: (n_generate, latent_dim) - VAE latent means
        candidate_indices: valid candidate indices
        k: number to select
        lambda_mmr: trade-off parameter (1=pure align, 0=pure diversity)
    
    Returns:
        selected_global: (k,) global indices
    """
    n_valid = len(candidate_indices)
    if n_valid < k:
        raise ValueError(f"Insufficient candidates: {n_valid} < k={k}")
    
    valid_idx = candidate_indices
    
    # 1. Utility: -align_score (higher is better), normalize ONCE
    scores = align_scores[valid_idx]
    utility = -scores
    u_min, u_max = utility.min(), utility.max()
    utility_norm = (utility - u_min) / (u_max - u_min + 1e-10)
    
    # 2. Latent vectors: z-score normalize ONCE
    z = latent_mu[valid_idx]
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0) + 1e-10
    z_norm = (z - z_mean) / z_std
    
    # 3. Precompute all pairwise distances for diversity scaling
    # Use median of all pairwise distances as scale reference
    all_dists = np.linalg.norm(z_norm[:, None, :] - z_norm[None, :, :], axis=2)
    np.fill_diagonal(all_dists, np.inf)  # exclude self
    median_dist = np.median(all_dists[all_dists < np.inf])
    
    selected_local = []
    selected_global = []
    
    for step in range(k):
        if step == 0:
            # First selection: pure utility with tie-break
            mmr_scores = utility_norm.copy()
        else:
            # Diversity: min distance to already selected
            z_selected = z_norm[selected_local]
            dists_to_selected = np.linalg.norm(z_norm[:, None, :] - z_selected[None, :, :], axis=2)
            min_dists = dists_to_selected.min(axis=1)
            
            # Normalize diversity using fixed scale (median)
            div_norm = min_dists / (median_dist + 1e-10)
            div_norm = np.clip(div_norm, 0, 2)  # clip extreme values
            div_norm = div_norm / 2  # scale to [0, 1]
            
            # MMR score
            mmr_scores = lambda_mmr * utility_norm + (1 - lambda_mmr) * div_norm
        
        # Mask already selected
        for idx in selected_local:
            mmr_scores[idx] = -np.inf
        
        # Tie-break: lexsort by (-mmr_score, global_idx)
        # Want highest mmr_score, so use -mmr_scores
        # For equal scores, prefer lower global_idx
        order = np.lexsort((valid_idx, -mmr_scores))
        best_local = order[0]
        
        selected_local.append(best_local)
        selected_global.append(valid_idx[best_local])
    
    return np.array(selected_global)


# =============================================================================
# Structure Metrics (Teacher-based)
# =============================================================================

def compute_structure_metrics(pred_coef, teacher_coef, term_freq=None):
    """
    Compute teacher-based structure metrics
    
    Args:
        pred_coef: (n_features, n_targets) predicted coefficients
        teacher_coef: (n_features, n_targets) teacher coefficients
        term_freq: (n_features, n_targets) term frequency from bootstrap (optional)
    
    Returns:
        dict with Jaccard, Correlation, etc.
    """
    # P0: Shape validation (GPT recommendation)
    assert teacher_coef.shape == pred_coef.shape, (
        f"Shape mismatch: teacher {teacher_coef.shape} vs pred {pred_coef.shape}"
    )
    
    # Support definition: abs(coef) > 1e-12
    teacher_support = np.abs(teacher_coef) > 1e-12
    pred_support = np.abs(pred_coef) > 1e-12
    
    # Jaccard similarity
    intersection = np.sum(teacher_support & pred_support)
    union = np.sum(teacher_support | pred_support)
    jaccard = intersection / (union + 1e-10)
    
    # Precision/Recall on support
    tp = np.sum(teacher_support & pred_support)
    fp = np.sum(~teacher_support & pred_support)
    fn = np.sum(teacher_support & ~pred_support)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Coefficient correlation on teacher support
    teacher_flat = teacher_coef[teacher_support]
    pred_flat_on_teacher = pred_coef[teacher_support]
    
    if len(teacher_flat) > 1:
        coef_corr_teacher = np.corrcoef(teacher_flat, pred_flat_on_teacher)[0, 1]
        if np.isnan(coef_corr_teacher):
            coef_corr_teacher = 0.0
    else:
        coef_corr_teacher = 0.0
    
    # Coefficient correlation on intersection support (GPT P0 recommendation)
    intersection_mask = teacher_support & pred_support
    if np.sum(intersection_mask) > 1:
        teacher_inter = teacher_coef[intersection_mask]
        pred_inter = pred_coef[intersection_mask]
        coef_corr_intersection = np.corrcoef(teacher_inter, pred_inter)[0, 1]
        if np.isnan(coef_corr_intersection):
            coef_corr_intersection = 0.0
    else:
        coef_corr_intersection = np.nan  # Too few terms for meaningful correlation
    
    # Coefficient RMSE on teacher support
    if len(teacher_flat) > 0:
        coef_rmse = np.sqrt(np.mean((teacher_flat - pred_flat_on_teacher)**2))
    else:
        coef_rmse = 0.0
    
    # Coefficient RMSE on intersection (scaled comparison)
    if np.sum(intersection_mask) > 0:
        coef_rmse_intersection = np.sqrt(np.mean((teacher_coef[intersection_mask] - pred_coef[intersection_mask])**2))
    else:
        coef_rmse_intersection = np.nan
    
    # Term frequency stats (if provided)
    term_freq_stats = None
    if term_freq is not None:
        # Mean frequency on teacher support
        freq_on_support = term_freq[teacher_support]
        term_freq_stats = {
            'mean_freq_on_support': float(np.mean(freq_on_support)),
            'min_freq_on_support': float(np.min(freq_on_support)),
            'n_stable_terms': int(np.sum(freq_on_support > 0.8)),  # terms appearing in >80% of bootstraps
        }
    
    return {
        'jaccard': float(jaccard),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'coef_correlation': float(coef_corr_teacher),  # on teacher support (original)
        'coef_correlation_intersection': float(coef_corr_intersection) if not np.isnan(coef_corr_intersection) else None,
        'coef_rmse': float(coef_rmse),  # on teacher support
        'coef_rmse_intersection': float(coef_rmse_intersection) if not np.isnan(coef_rmse_intersection) else None,
        'n_teacher_terms': int(np.sum(teacher_support)),
        'n_pred_terms': int(np.sum(pred_support)),
        'n_intersection': int(intersection),
        'term_freq_stats': term_freq_stats,
    }


def file_sha256(filepath):
    """Compute SHA256 hash of file"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except:
        return 'not_found'


def main():
    parser = argparse.ArgumentParser(description='Evaluate selection for Gate3 Phase 2-c')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pool_npz', type=str, required=True)
    parser.add_argument('--pool_json', type=str, required=True)
    parser.add_argument('--teacher_run_dir', type=str, required=True)
    parser.add_argument('--selection_mode', type=str, choices=['align', 'random', 'mmr'], required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--select_seed', type=int, default=0, help='Seed for random selection')
    parser.add_argument('--mmr_lambda', type=float, default=0.5, help='MMR lambda (1=pure align, 0=pure diversity)')
    parser.add_argument('--latent_npz', type=str, default=None, help='Latent embeddings for MMR (optional)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--note', type=str, default='eval')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    dataset_version = config.get('dataset_version', 'cartpole_ood_v1')
    
    # Load pool
    pool_data = np.load(args.pool_npz)
    with open(args.pool_json) as f:
        pool_meta = json.load(f)
    
    x_candidates = pool_data['x_candidates']
    u_candidates = pool_data['u_candidates']
    dx_candidates = pool_data['dx_candidates']
    align_scores = pool_data['align_scores']
    candidate_indices = pool_data['candidate_indices']
    
    n_generate = pool_meta['n_generate']
    n_candidates = len(candidate_indices)
    
    print("=" * 60)
    print("Gate3 Phase 2-c Evaluation (v3 - MMR + Structure)")
    print("=" * 60)
    print(f"Pool: {args.pool_npz}")
    print(f"Teacher: {args.teacher_run_dir}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"k: {args.k}")
    print(f"Candidates: {n_candidates}/{n_generate}")
    
    # SSOT: Load Gate1 train_indices
    train_indices, gate1_manifest = load_gate1_train_indices(args.teacher_run_dir)
    n_train = len(train_indices)
    
    # Load teacher coefficients for structure metrics
    teacher_coef = load_teacher_coefficients(args.teacher_run_dir)
    print(f"  Teacher coefficients loaded: {teacher_coef.shape}")
    
    # Selection
    if args.selection_mode == 'align':
        selected_indices = select_align_topk(align_scores, candidate_indices, args.k)
        selection_params = {'tie_break_rule': 'lexsort_global_idx'}
        print(f"  Align Top-{args.k} selected (lexsort tie-break)")
        
    elif args.selection_mode == 'random':
        selected_indices = select_random(candidate_indices, args.k, args.select_seed)
        selection_params = {'select_seed': args.select_seed}
        print(f"  Random-{args.k} selected (seed={args.select_seed})")
        
    elif args.selection_mode == 'mmr':
        # Load latent embeddings
        if args.latent_npz:
            latent_data = np.load(args.latent_npz)
            latent_mu = latent_data['latent_mu']
        else:
            # Try to find latent in pool directory
            pool_dir = Path(args.pool_npz).parent
            latent_path = pool_dir / f'latent_n{n_generate}_seed{pool_meta.get("gen_seed", 0)}.npz'
            if latent_path.exists():
                latent_data = np.load(latent_path)
                latent_mu = latent_data['latent_mu']
            else:
                raise ValueError(f"Latent embeddings required for MMR. Provide --latent_npz or ensure {latent_path} exists")
        
        selected_indices = select_mmr(align_scores, latent_mu, candidate_indices, args.k, args.mmr_lambda)
        selection_params = {
            'mmr_lambda': args.mmr_lambda,
            'tie_break_rule': 'lexsort_global_idx',
            'latent_source': 'mu',
            'latent_norm': 'zscore_valid',
            'distance_metric': 'l2',
        }
        print(f"  MMR-{args.k} selected (λ={args.mmr_lambda}, lexsort tie-break)")
    
    # Selected data
    x_aug_selected = x_candidates[selected_indices]
    u_aug_selected = u_candidates[selected_indices]
    dx_aug_selected = dx_candidates[selected_indices]
    
    # Load dataset
    dataset_path = get_dataset_path(dataset_version)
    validate_dataset_lite(dataset_path)
    data = np.load(dataset_path)
    
    full_train_x = data['train_x']
    full_train_u = data['train_u']
    full_train_dx = data['train_dx']
    
    train_x = full_train_x[train_indices]
    train_u = full_train_u[train_indices]
    train_dx = full_train_dx[train_indices]
    
    print(f"  [SSOT] Using train subset: {train_x.shape[0]}/{full_train_x.shape[0]} trajectories")
    
    val_x, val_u, val_dx = data['val_x'], data['val_u'], data['val_dx']
    test_x, test_u, test_dx = data['test_x'], data['test_u'], data['test_dx']
    
    # Build libraries
    Theta_train_orig = build_library(train_x, train_u)
    dx_train_orig = train_dx.reshape(-1, 4)
    
    Theta_aug_sel = build_library(x_aug_selected, u_aug_selected)
    dx_aug_sel = dx_aug_selected.reshape(-1, 4)
    
    x_cand = x_candidates[candidate_indices]
    u_cand = u_candidates[candidate_indices]
    dx_cand = dx_candidates[candidate_indices]
    Theta_aug_cand = build_library(x_cand, u_cand)
    dx_aug_cand = dx_cand.reshape(-1, 4)
    
    x_all = np.concatenate([train_x, x_aug_selected], axis=0)
    u_all = np.concatenate([train_u, u_aug_selected], axis=0)
    dx_all = np.concatenate([train_dx, dx_aug_selected], axis=0)
    Theta_train_all = build_library(x_all, u_all)
    dx_train_all = dx_all.reshape(-1, 4)
    
    Theta_val = build_library(val_x, val_u)
    dx_val = val_dx.reshape(-1, 4)
    Theta_test = build_library(test_x, test_u)
    dx_test = test_dx.reshape(-1, 4)
    
    # Training config
    threshold_grid = config.get('evaluation', {}).get('threshold_grid', 
        [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05])
    n_bootstrap = config.get('evaluation', {}).get('n_bootstrap', 20)
    tolerance = config.get('evaluation', {}).get('val_r2_tolerance', 0.002)
    bootstrap_seed = config.get('evaluation', {}).get('bootstrap_seed', 42)
    
    print(f"\nTraining E-SINDy on combined data (n={n_train}+{args.k}={n_train+args.k})...")
    print(f"  Bootstrap seed: {bootstrap_seed}")
    
    best_threshold, coef_mean, term_freq, _ = select_best_threshold(
        Theta_train_all, dx_train_all, Theta_val, dx_val,
        threshold_grid, n_bootstrap, tolerance, bootstrap_seed=bootstrap_seed
    )
    print(f"  Best threshold: {best_threshold}")
    
    # Compute R² metrics
    dx_pred_orig = Theta_train_orig @ coef_mean
    train_r2_original = compute_r2(dx_train_orig, dx_pred_orig)
    
    dx_pred_aug_sel = Theta_aug_sel @ coef_mean
    train_r2_aug_selected = compute_r2(dx_aug_sel, dx_pred_aug_sel)
    
    dx_pred_aug_cand = Theta_aug_cand @ coef_mean
    train_r2_aug_candidates = compute_r2(dx_aug_cand, dx_pred_aug_cand)
    
    dx_pred_all = Theta_train_all @ coef_mean
    train_r2_all = compute_r2(dx_train_all, dx_pred_all)
    
    dx_pred_val = Theta_val @ coef_mean
    val_r2 = compute_r2(dx_val, dx_pred_val)
    
    dx_pred_test = Theta_test @ coef_mean
    test_r2 = compute_r2(dx_test, dx_pred_test)
    
    sparsity = float(np.mean(np.abs(coef_mean) < 1e-10))
    
    # Compute structure metrics
    structure_metrics = compute_structure_metrics(coef_mean, teacher_coef, term_freq)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Train R² (original n={n_train}):  {train_r2_original:.4f}")
    print(f"Train R² (aug_selected k={args.k}): {train_r2_aug_selected:.4f}")
    print(f"Train R² (aug_candidates n={n_candidates}): {train_r2_aug_candidates:.4f}")
    print(f"Train R² (all n={n_train+args.k}):      {train_r2_all:.4f}")
    print(f"Val R²:                   {val_r2:.4f}")
    print(f"Test R²:                  {test_r2:.4f}")
    print(f"Sparsity:                 {sparsity:.1%}")
    print(f"\nStructure Metrics (vs Teacher):")
    print(f"  Jaccard:     {structure_metrics['jaccard']:.4f}")
    print(f"  F1:          {structure_metrics['f1']:.4f}")
    print(f"  Coef Corr (teacher support):   {structure_metrics['coef_correlation']:.4f}")
    if structure_metrics.get('coef_correlation_intersection') is not None:
        print(f"  Coef Corr (intersection):      {structure_metrics['coef_correlation_intersection']:.4f}")
    print(f"  Coef RMSE:   {structure_metrics['coef_rmse']:.4f}")
    print(f"  n_intersection: {structure_metrics['n_intersection']}")
    
    # Selection stats
    selected_scores = align_scores[selected_indices]
    valid_scores_all = align_scores[candidate_indices]
    
    align_stats_selected = {
        'mean': float(np.mean(selected_scores)),
        'std': float(np.std(selected_scores)),
        'min': float(np.min(selected_scores)),
        'max': float(np.max(selected_scores)),
        'count': len(selected_scores),
    }
    
    # Output directory
    n_gen = pool_meta['n_generate']
    if args.selection_mode == 'align':
        run_note = f'n{n_gen}_k{args.k}_align'
    elif args.selection_mode == 'random':
        run_note = f'n{n_gen}_k{args.k}_random_s{args.select_seed}'
    else:  # mmr
        run_note = f'n{n_gen}_k{args.k}_mmr_l{int(args.mmr_lambda*100)}'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_ROOT / dataset_version / 'gate3_phase2c' / 'runs' / run_note
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hashes for SSOT
    train_indices_hash = hashlib.md5(train_indices.tobytes()).hexdigest()[:8]
    selected_indices_hash = hashlib.md5(selected_indices.tobytes()).hexdigest()[:8]
    pool_hash = hashlib.md5(open(args.pool_npz, 'rb').read()).hexdigest()[:8]
    
    # Script hashes
    script_dir = Path(__file__).parent
    script_hashes = {
        'evaluate_selection.py': file_sha256(script_dir / 'evaluate_selection.py'),
        'run_phase2c_core.py': file_sha256(script_dir / 'run_phase2c_core.py'),
        'generate_pool.py': file_sha256(script_dir / 'generate_pool.py'),
        'train_vae.py': file_sha256(script_dir / 'train_vae.py'),
    }
    config_hash = file_sha256(args.config)
    
    # Metrics
    metrics = {
        'train_r2_original': train_r2_original,
        'train_r2_aug_selected': train_r2_aug_selected,
        'train_r2_aug_candidates': train_r2_aug_candidates,
        'train_r2_all': train_r2_all,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'sparsity': sparsity,
        'best_threshold': best_threshold,
        'n_train': n_train,
        'n_aug': args.k,
        'n_total': n_train + args.k,
        'n_valid_candidates': n_candidates,
        'structure_metrics': structure_metrics,
        'selection_info': {
            'mode': args.selection_mode,
            'k': args.k,
            'selected_indices': selected_indices.tolist(),
            'selected_indices_hash': selected_indices_hash,
            'n_candidates': n_candidates,
            'n_generate': n_gen,
            **selection_params,
        },
        'align_score_stats_selected': align_stats_selected,
    }
    
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")
    
    # Manifest (SSOT 8 required fields)
    manifest = {
        'version': 'gate3_phase2c_v3',
        'created_at': datetime.now().isoformat(),
        'gate': 'gate3_phase2c',
        'dataset_version': dataset_version,
        
        # SSOT Required Fields (8개)
        'train_indices_hash': train_indices_hash,
        'bootstrap_seed': bootstrap_seed,
        'select_seed': args.select_seed if args.selection_mode == 'random' else None,
        'vae_seed': pool_meta.get('vae_seed'),
        'pool_seed': pool_meta.get('gen_seed'),
        'align_score_definition': (pool_meta.get('align_score_spec') or {}).get('formula', 'teacher_dx_mse'),
        'tie_break_rule': 'lexsort_global_idx',
        'pool_npz_path': str(args.pool_npz),
        'pool_metadata_json_path': str(args.pool_json),
        'pool_hash': pool_hash,
        
        # Additional info
        'selection_mode': args.selection_mode,
        'k': args.k,
        'mmr_lambda': args.mmr_lambda if args.selection_mode == 'mmr' else None,
        'n_train': n_train,
        'train_indices': train_indices.tolist(),
        'teacher_run_dir': str(args.teacher_run_dir),
        
        'config': {
            'config_path': str(args.config),
            'config_sha256': config_hash,
            'n_bootstrap': n_bootstrap,
            'threshold_grid': threshold_grid,
            'val_r2_tolerance': tolerance,
        },
        'script_sha256': script_hashes,
        'command': ' '.join(sys.argv),
    }
    
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")
    
    # Coefficients
    coef_path = output_dir / 'sindy_coefficients.csv'
    with open(coef_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['term_name', 'dx_0', 'dx_1', 'dx_2', 'dx_3'])
        for i, name in enumerate(FEATURE_NAMES):
            writer.writerow([name] + [f'{coef_mean[i, j]:.6f}' for j in range(4)])
    print(f"Coefficients saved: {coef_path}")
    
    print(f"\nOutput directory: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())