#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase1+ Pool Generation Script
=====================================
VAE에서 후보 생성 → Sanity/Dedup 필터링 → Pool 저장

GPT 필수 조건 반영:
1. Pool metadata JSON 분리 (pickle-free)
2. Index/mask 원본 기준 정렬 (n_generate 길이 고정)
3. align_score_spec 스키마화

Usage:
    python src/experiments/generate_pool.py --vae_path models/vae_seed0.pt --n_generate 100 --gen_seed 0
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.paths import get_dataset_path, RESULTS_ROOT
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.generative.vae import TrajectoryVAE, VAEConfig


def compute_dx_savgol(x, dt, window=11, polyorder=3):
    """Savitzky-Golay 필터로 dx 계산"""
    N, T, state_dim = x.shape
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(state_dim):
            dx[i, :, j] = savgol_filter(x[i, :, j], window, polyorder, deriv=1, delta=dt)
    return dx


def sanity_check(x, state_bounds=None):
    """Sanity filter"""
    if state_bounds is None:
        state_bounds = {0: (-5.0, 5.0), 1: (-10.0, 10.0), 2: (-np.pi, np.pi), 3: (-15.0, 15.0)}
    
    N = x.shape[0]
    sanity_mask = np.ones(N, dtype=bool)
    reject_reasons = {'nan_inf': 0, 'range_violation': 0}
    
    for i in range(N):
        traj = x[i]
        if np.any(~np.isfinite(traj)):
            sanity_mask[i] = False
            reject_reasons['nan_inf'] += 1
            continue
        for state_idx, (vmin, vmax) in state_bounds.items():
            if state_idx < traj.shape[1]:
                if np.any(traj[:, state_idx] < vmin) or np.any(traj[:, state_idx] > vmax):
                    sanity_mask[i] = False
                    reject_reasons['range_violation'] += 1
                    break
    return sanity_mask, reject_reasons


def dedup_filter(x, valid_mask, threshold=0.01):
    """Deduplication filter"""
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
            if mse < threshold:
                is_duplicate[j] = True
    
    updated_mask = valid_mask.copy()
    for idx, is_dup in zip(valid_indices, is_duplicate):
        if is_dup:
            updated_mask[idx] = False
    return updated_mask


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


def compute_align_scores(x, dx, u, teacher_coef):
    """Teacher alignment score 계산 (낮을수록 좋음)"""
    N, T, state_dim = x.shape
    Theta = build_library(x, u)
    dx_flat = dx.reshape(N * T, state_dim)
    dx_pred = Theta @ teacher_coef
    residual = dx_flat - dx_pred
    mse_per_timestep = np.mean(residual**2, axis=1)
    mse_per_traj = mse_per_timestep.reshape(N, T).mean(axis=1)
    return mse_per_traj


def load_teacher_coefficients(teacher_run_dir):
    """Gate1 Teacher 계수 로드"""
    coef_path = Path(teacher_run_dir) / 'sindy_coefficients.csv'
    if not coef_path.exists():
        raise FileNotFoundError(f"Teacher coefficients not found: {coef_path}")
    
    import csv
    with open(coef_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        data_cols = [i for i, h in enumerate(header) if h.startswith('dx_')]
        if not data_cols:
            data_cols = list(range(1, len(header)))
        rows = [[float(row[i]) for i in data_cols] for row in reader]
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate candidate pool for Gate3 Phase1+')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--teacher_run_dir', type=str, required=True)
    parser.add_argument('--n_generate', type=int, required=True)
    parser.add_argument('--gen_seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    dataset_version = config.get('dataset_version', 'cartpole_ood_v1')
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_ROOT / dataset_version / 'gate3_phase1plus' / 'pools'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Gate3 Phase1+ Pool Generation")
    print("=" * 60)
    print(f"VAE path: {args.vae_path}")
    print(f"n_generate: {args.n_generate}")
    print(f"gen_seed: {args.gen_seed}")
    
    # Load VAE
    checkpoint = torch.load(args.vae_path, map_location='cpu', weights_only=False)
    vae_cfg = checkpoint['config']
    vae_config = VAEConfig(
        latent_dim=vae_cfg['latent_dim'],
        hidden_dim=vae_cfg['hidden_dim'],
        state_dim=vae_cfg['state_dim'],
        seq_len=vae_cfg['seq_len'],
        vae_seed=vae_cfg['vae_seed'],
        gen_seed=args.gen_seed,
    )
    vae = TrajectoryVAE(vae_config, seq_len=vae_cfg['seq_len'])
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.data_mean = torch.tensor(checkpoint['data_mean'])
    vae.data_std = torch.tensor(checkpoint['data_std'])
    vae.eval()
    print(f"VAE loaded (latent_dim={vae_cfg['latent_dim']})")
    
    # Load dataset for u templates
    dataset_path = get_dataset_path(dataset_version)
    validate_dataset_lite(dataset_path)
    data = np.load(dataset_path)
    train_u = data['train_u']
    dt = float(data['dt'])
    N_train = train_u.shape[0]
    print(f"Dataset loaded (N_train={N_train}, dt={dt})")
    
    # Load teacher
    teacher_coef = load_teacher_coefficients(args.teacher_run_dir)
    print(f"Teacher loaded (shape={teacher_coef.shape})")
    
    # Generate candidates
    print(f"\nGenerating {args.n_generate} candidates...")
    x_candidates = vae.sample(args.n_generate, temperature=1.0, seed=args.gen_seed)
    
    # Sample u from training data
    rng = np.random.default_rng(args.gen_seed)
    u_indices = rng.choice(N_train, args.n_generate, replace=True)
    u_candidates = train_u[u_indices]
    
    # Compute dx
    dx_candidates = compute_dx_savgol(x_candidates, dt)
    print(f"  x_candidates: {x_candidates.shape}")
    
    # Sanity filter
    sanity_mask, reject_reasons = sanity_check(x_candidates)
    n_sanity = sanity_mask.sum()
    print(f"  Sanity pass: {n_sanity}/{args.n_generate}")
    
    # Dedup filter
    dedup_mask = dedup_filter(x_candidates, sanity_mask)
    n_dedup = dedup_mask.sum()
    n_dup_rejected = n_sanity - n_dedup
    print(f"  Dedup pass: {n_dedup}/{n_sanity} (rejected {n_dup_rejected} duplicates)")
    
    # Compute align scores (for ALL candidates, NaN for invalid)
    align_scores = np.full(args.n_generate, np.nan)
    valid_idx = np.where(dedup_mask)[0]
    if len(valid_idx) > 0:
        valid_scores = compute_align_scores(
            x_candidates[valid_idx], 
            dx_candidates[valid_idx], 
            u_candidates[valid_idx], 
            teacher_coef
        )
        align_scores[valid_idx] = valid_scores
    
    # Candidate indices (sanity+dedup pass)
    candidate_indices = np.where(dedup_mask)[0]
    
    # Align score stats for candidates
    valid_scores = align_scores[candidate_indices]
    align_stats = {
        'mean': float(np.mean(valid_scores)),
        'std': float(np.std(valid_scores)),
        'min': float(np.min(valid_scores)),
        'max': float(np.max(valid_scores)),
        'p10': float(np.percentile(valid_scores, 10)),
        'p50': float(np.percentile(valid_scores, 50)),
        'p90': float(np.percentile(valid_scores, 90)),
        'count': len(valid_scores),
    }
    print(f"  Align scores: mean={align_stats['mean']:.2f}, min={align_stats['min']:.2f}, max={align_stats['max']:.2f}")
    
    # Save pool.npz (arrays only, pickle-free)
    pool_npz_path = output_dir / f'pool_n{args.n_generate}_seed{args.gen_seed}.npz'
    np.savez(
        pool_npz_path,
        x_candidates=x_candidates.astype(np.float32),
        u_candidates=u_candidates.astype(np.float32),
        dx_candidates=dx_candidates.astype(np.float32),
        sanity_mask=sanity_mask,
        dedup_mask=dedup_mask,
        align_scores=align_scores.astype(np.float32),
        candidate_indices=candidate_indices.astype(np.int32),
    )
    print(f"\nPool saved: {pool_npz_path}")
    
    # Compute config hash
    config_hash_inputs = {
        'vae_path': str(args.vae_path),
        'teacher_run_dir': str(args.teacher_run_dir),
        'n_generate': args.n_generate,
        'gen_seed': args.gen_seed,
        'dx_policy': 'savgol',
        'savgol_window': 11,
        'savgol_polyorder': 3,
    }
    config_hash = hashlib.md5(json.dumps(config_hash_inputs, sort_keys=True).encode()).hexdigest()[:8]
    
    # Save pool.json (metadata)
    pool_json = {
        'version': 'gate3_phase1plus_v1',
        'created_at': datetime.now().isoformat(),
        'vae_checkpoint_path': str(args.vae_path),
        'vae_seed': vae_cfg['vae_seed'],
        'gen_seed': args.gen_seed,
        'n_generate': args.n_generate,
        'teacher_run_dir': str(args.teacher_run_dir),
        
        'align_score_spec': {
            'metric_name': 'teacher_dx_mse',
            'direction': 'lower_is_better',
            'computed_on': 'dx',
            'aggregation': 'mean_over_time_then_mean_over_state',
            'formula': 'mean_t || dx_gen - Theta @ Xi_bar ||^2',
        },
        
        'config_hash_inputs': config_hash_inputs,
        'config_hash': config_hash,
        
        'filtering_summary': {
            'n_generated': args.n_generate,
            'n_sanity_pass': int(n_sanity),
            'n_dedup_pass': int(n_dedup),
            'sanity_reject_reasons': reject_reasons,
            'dedup_reject_count': int(n_dup_rejected),
        },
        
        'align_score_stats_candidates': align_stats,
    }
    
    pool_json_path = output_dir / f'pool_n{args.n_generate}_seed{args.gen_seed}.json'
    with open(pool_json_path, 'w') as f:
        json.dump(pool_json, f, indent=2)
    print(f"Metadata saved: {pool_json_path}")
    
    print("\n" + "=" * 60)
    print("Pool Generation Complete!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())