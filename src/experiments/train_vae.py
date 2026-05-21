#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase1+ VAE Training Script
=================================
VAE 학습 → 모델 저장 (공유 구조의 첫 단계)

Usage:
    python src/experiments/train_vae.py --config configs/experiments/gate3_cartpole.yaml --vae_seed 0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.paths import get_dataset_path, RESULTS_ROOT
from src.contracts.schema_dataset_lite import validate_dataset_lite
from src.generative.vae import TrajectoryVAE, VAEConfig


def main():
    parser = argparse.ArgumentParser(description='Train VAE for Gate3 Phase1+')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--vae_seed', type=int, default=0, help='VAE training seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    dataset_version = config.get('dataset_version', 'cartpole_ood_v1')
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_ROOT / dataset_version / 'gate3_phase1plus' / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Gate3 Phase1+ VAE Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"VAE seed: {args.vae_seed}")
    print(f"Output: {output_dir}")
    print()
    
    # Load dataset
    dataset_path = get_dataset_path(dataset_version)
    print(f"Loading dataset: {dataset_path}")
    
    # Preflight check
    validate_dataset_lite(dataset_path)
    
    data = np.load(dataset_path)
    train_x = data['train_x']
    val_x = data['val_x']
    
    print(f"  train_x shape: {train_x.shape}")
    print(f"  val_x shape: {val_x.shape}")
    
    # Create VAE config
    vae_config = VAEConfig.from_yaml_dict(config)
    vae_config.vae_seed = args.vae_seed
    vae_config.seq_len = train_x.shape[1]
    
    print(f"\nVAE Config:")
    print(f"  latent_dim: {vae_config.latent_dim}")
    print(f"  hidden_dim: {vae_config.hidden_dim}")
    print(f"  epochs: {vae_config.epochs}")
    print(f"  beta_schedule: {vae_config.beta_start} -> {vae_config.beta_end}")
    
    # Create and train VAE
    print("\n" + "=" * 60)
    print("Training VAE...")
    print("=" * 60)
    
    vae = TrajectoryVAE(vae_config, seq_len=train_x.shape[1])
    history = vae.fit(train_x, val_x=val_x, verbose=True)
    
    # Save model
    model_path = output_dir / f'vae_seed{args.vae_seed}.pt'
    torch.save({
        'model_state_dict': vae.state_dict(),
        'config': {
            'latent_dim': vae_config.latent_dim,
            'hidden_dim': vae_config.hidden_dim,
            'state_dim': vae_config.state_dim,
            'seq_len': vae_config.seq_len,
            'epochs': vae_config.epochs,
            'lr': vae_config.lr,
            'beta_start': vae_config.beta_start,
            'beta_end': vae_config.beta_end,
            'warmup_epochs': vae_config.warmup_epochs,
            'vae_seed': args.vae_seed,
        },
        'data_mean': vae.data_mean.numpy(),
        'data_std': vae.data_std.numpy(),
        'train_loss_history': history['train_loss'],
        'val_loss_history': history['val_loss'],
    }, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save manifest
    manifest = {
        'version': 'gate3_phase1plus_v1',
        'created_at': datetime.now().isoformat(),
        'vae_seed': args.vae_seed,
        'config_path': str(args.config),
        'dataset_version': dataset_version,
        'model_path': str(model_path),
        'model_info': vae.get_model_info(),
        'training': {
            'n_train': train_x.shape[0],
            'n_val': val_x.shape[0],
            'seq_len': train_x.shape[1],
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        }
    }
    
    manifest_path = output_dir / f'vae_seed{args.vae_seed}_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")
    
    print("\n" + "=" * 60)
    print("VAE Training Complete!")
    print("=" * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())