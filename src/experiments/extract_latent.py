#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Latent Extraction Script for MMR Selection
==========================================
기존 VAE 모델과 Pool 데이터에서 latent embeddings 추출

Usage:
    python src/experiments/extract_latent.py --vae_path models/vae_seed0.pt --pool_npz pools/pool_n500_seed0.npz
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def extract_latent_from_vae(vae_path, x_data):
    """
    VAE 모델에서 latent mu 추출
    
    Args:
        vae_path: VAE checkpoint 경로
        x_data: (N, T, state_dim) 궤적 데이터
    
    Returns:
        latent_mu: (N, latent_dim)
    """
    # Load VAE checkpoint (weights_only=False for numpy compatibility in PyTorch 2.6+)
    checkpoint = torch.load(vae_path, map_location='cpu', weights_only=False)
    vae_cfg = checkpoint['config']
    
    # Reconstruct VAE model
    from src.generative.vae import TrajectoryVAE, VAEConfig
    
    vae_config = VAEConfig(
        latent_dim=vae_cfg['latent_dim'],
        hidden_dim=vae_cfg['hidden_dim'],
        state_dim=vae_cfg['state_dim'],
        seq_len=vae_cfg['seq_len'],
        vae_seed=vae_cfg['vae_seed'],
    )
    vae = TrajectoryVAE(vae_config, seq_len=vae_cfg['seq_len'])
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.data_mean = torch.tensor(checkpoint['data_mean'])
    vae.data_std = torch.tensor(checkpoint['data_std'])
    vae.eval()
    
    # Normalize input data
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    x_norm = (x_tensor - vae.data_mean) / (vae.data_std + 1e-8)
    
    # Encode to get mu
    with torch.no_grad():
        mu, log_var = vae.encode(x_norm)
    
    # Handle both tensor and numpy array returns
    if hasattr(mu, 'numpy'):
        mu = mu.numpy()
    
    return mu, vae_cfg


def main():
    parser = argparse.ArgumentParser(description='Extract latent embeddings for MMR')
    parser.add_argument('--vae_path', type=str, required=True, help='VAE checkpoint path')
    parser.add_argument('--pool_npz', type=str, required=True, help='Pool NPZ path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as pool)')
    args = parser.parse_args()
    
    vae_path = Path(args.vae_path)
    pool_npz_path = Path(args.pool_npz)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pool_npz_path.parent
    
    print("=" * 60)
    print("Latent Extraction for MMR")
    print("=" * 60)
    print(f"VAE: {vae_path}")
    print(f"Pool: {pool_npz_path}")
    print(f"Output: {output_dir}")
    
    # Load pool data
    pool_data = np.load(pool_npz_path)
    x_candidates = pool_data['x_candidates']
    n_generate = x_candidates.shape[0]
    print(f"\nLoaded pool: {n_generate} candidates, shape={x_candidates.shape}")
    
    # Extract latent
    print("\nExtracting latent embeddings...")
    latent_mu, vae_cfg = extract_latent_from_vae(vae_path, x_candidates)
    print(f"Latent mu shape: {latent_mu.shape}")
    print(f"Latent dim: {vae_cfg['latent_dim']}")
    
    # Latent statistics
    print(f"\nLatent statistics:")
    print(f"  Mean: {latent_mu.mean():.4f}")
    print(f"  Std:  {latent_mu.std():.4f}")
    print(f"  Min:  {latent_mu.min():.4f}")
    print(f"  Max:  {latent_mu.max():.4f}")
    
    # Extract n_generate and seed from pool filename
    # Expected format: pool_n{N}_seed{S}.npz
    pool_name = pool_npz_path.stem
    parts = pool_name.split('_')
    n_gen = None
    seed = None
    for p in parts:
        if p.startswith('n') and p[1:].isdigit():
            n_gen = int(p[1:])
        elif p.startswith('seed') and p[4:].isdigit():
            seed = int(p[4:])
    
    # Save latent
    if n_gen and seed is not None:
        output_filename = f'latent_n{n_gen}_seed{seed}.npz'
    else:
        output_filename = f'latent_{pool_name}.npz'
    
    output_path = output_dir / output_filename
    
    # SSOT: row_idx는 latent_mu의 행 인덱스 (0~n_generate-1)
    # candidate_indices (valid subset)는 pool_npz에서 로드하여 함께 저장 (GPT P0)
    row_idx = np.arange(n_generate)
    candidate_indices = pool_data.get('candidate_indices', None)
    
    save_dict = {
        'latent_mu': latent_mu.astype(np.float32),
        'row_idx': row_idx,  # latent_mu[i]는 생성 샘플 i에 해당
        'latent_dim': vae_cfg['latent_dim'],
        'n_generate': n_generate,  # 전체 생성 샘플 수
        'vae_path': str(vae_path),
        'pool_npz_path': str(pool_npz_path),
    }
    
    # candidate_indices가 있으면 함께 저장 (MMR selection과의 정합성)
    if candidate_indices is not None:
        save_dict['candidate_indices'] = candidate_indices.astype(np.int64)
        print(f"  candidate_indices included: {len(candidate_indices)}/{n_generate} valid")
    
    np.savez(output_path, **save_dict)
    print(f"\nLatent saved: {output_path}")
    
    # Save metadata JSON
    meta = {
        'created_at': datetime.now().isoformat(),
        'vae_path': str(vae_path),
        'pool_npz_path': str(pool_npz_path),
        'latent_dim': int(vae_cfg['latent_dim']),
        'n_candidates': int(n_generate),
        'latent_stats': {
            'mean': float(latent_mu.mean()),
            'std': float(latent_mu.std()),
            'min': float(latent_mu.min()),
            'max': float(latent_mu.max()),
        },
    }
    
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")
    
    print("\n" + "=" * 60)
    print("Latent Extraction Complete!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())