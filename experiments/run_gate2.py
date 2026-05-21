#!/usr/bin/env python
"""
Gate2: Augmentation Experiment CLI Entry Point

Runs the Gate2 physics-augmented E-SINDy pipeline.

Usage:
    python experiments/run_gate2.py --config configs/experiments/gate2_cartpole.yaml --dataset_version cartpole_ood_v1 --n_train 10 --seed 0 --track standardized --aug_ratio 1.0 --jitter_mode both --note aug

Arguments:
    --config: YAML config file (optional)
    --dataset_version: Dataset version
    --n_train: Number of training trajectories (before augmentation)
    --seed: Random seed
    --track: 'standardized' or 'author_recommended'
    --aug_ratio: Augmentation ratio (n_aug = n_train * aug_ratio)
    --jitter_mode: 'ic_only', 'param_only', 'both', or 'random'
    --note: Note for run_id
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.experiments.gate2_aug_runner import Gate2AugRunner, Gate2Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gate2 Augmentation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file
    parser.add_argument('--config', '-c', type=Path, default=None)
    
    # Dataset
    parser.add_argument('--dataset_version', '-d', type=str, default=None)
    parser.add_argument('--system', type=str, default=None)
    
    # Experiment settings
    parser.add_argument('--n_train', '-n', type=int, default=None)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--track', '-t', type=str, 
                        choices=['standardized', 'author_recommended'], default=None)
    parser.add_argument('--note', type=str, default=None)
    
    # Augmentation settings
    parser.add_argument('--aug_method', type=str, default=None)
    parser.add_argument('--aug_ratio', '-a', type=float, default=None)
    parser.add_argument('--aug_seed', type=int, default=None)
    parser.add_argument('--jitter_mode', '-j', type=str,
                        choices=['ic_only', 'param_only', 'both', 'random'], default=None)
    parser.add_argument('--ic_std_scale', type=float, default=None)
    parser.add_argument('--param_rel_std_scale', type=float, default=None)
    
    # E-SINDy settings
    parser.add_argument('--n_bootstrap', '-b', type=int, default=None)
    parser.add_argument('--library_config', type=str, default=None)
    parser.add_argument('--ridge_alpha', type=float, default=None)
    parser.add_argument('--final_fit_split', type=str,
                        choices=['train', 'train_val'], default=None)
    
    # Gate1 baseline for comparison
    parser.add_argument('--gate1_baseline_run_id', type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Step 1: Load base config as dict (YAML or defaults)
    import yaml
    if args.config and args.config.exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            base_dict = yaml.safe_load(f) or {}
    else:
        base_dict = {}
    
    # Step 2: Merge CLI overrides into dict (only if explicitly provided)
    cli_overrides = {
        'dataset_version': args.dataset_version,
        'system': args.system,
        'n_train': args.n_train,
        'seed': args.seed,
        'track': args.track,
        'note': args.note,
        'aug_method': args.aug_method,
        'aug_ratio': args.aug_ratio,
        'aug_seed': args.aug_seed,
        'jitter_mode': args.jitter_mode,
        'ic_std_scale': args.ic_std_scale,
        'param_rel_std_scale': args.param_rel_std_scale,
        'n_bootstrap': args.n_bootstrap,
        'library_config': args.library_config,
        'ridge_alpha': args.ridge_alpha,
        'final_fit_split': args.final_fit_split,
        'gate1_baseline_run_id': args.gate1_baseline_run_id,
    }
    
    for key, value in cli_overrides.items():
        if value is not None:
            base_dict[key] = value
    
    # Step 3: Create config from merged dict (__post_init__ runs with final values)
    config = Gate2Config.from_dict(base_dict)
    
    print("\n" + "=" * 70)
    print("  Gate2 Augmentation Configuration")
    print("=" * 70)
    print(f"  Dataset: {config.dataset_version}")
    print(f"  n_train: {config.n_train}")
    print(f"  seed: {config.seed}")
    print(f"  track: {config.track}")
    print(f"  aug_method: {config.aug_method}")
    print(f"  aug_ratio: {config.aug_ratio}")
    print(f"  jitter_mode: {config.jitter_mode}")
    print(f"  n_bootstrap: {config.n_bootstrap}")
    print("=" * 70)
    
    runner = Gate2AugRunner(config)
    result = runner.run()
    
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()