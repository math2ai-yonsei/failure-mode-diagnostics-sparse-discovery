#!/usr/bin/env python
"""
S07: Gate1 E-SINDy CLI Entry Point

Runs the Gate1 E-SINDy ensemble pipeline.

Usage:
    python experiments/run_gate1.py --config configs/experiments/gate1_cartpole.yaml --dataset_version cartpole_ood_v1 --n_train 10 --seed 0 --track standardized --n_bootstrap 20 --note base

Arguments:
    --config: YAML config file (optional)
    --dataset_version: Dataset version
    --n_train: Number of training trajectories
    --seed: Random seed
    --track: 'standardized' or 'author_recommended'
    --n_bootstrap: Number of bootstrap samples
    --note: Note for run_id
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.experiments.gate1_esindy_runner import Gate1ESINDyRunner, Gate1Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gate1 E-SINDy Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # All CLI args default to None for proper override detection
    parser.add_argument('--config', '-c', type=Path, default=None)
    parser.add_argument('--dataset_version', '-d', type=str, default=None)
    parser.add_argument('--system', type=str, default=None)
    parser.add_argument('--n_train', '-n', type=int, default=None)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--track', '-t', type=str, choices=['standardized', 'author_recommended'], default=None)
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--n_bootstrap', '-b', type=int, default=None)
    parser.add_argument('--library_config', type=str, default=None)
    parser.add_argument('--ridge_alpha', type=float, default=None)
    parser.add_argument('--final_fit_split', type=str, choices=['train', 'train_val'], default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Start with default config or YAML
    if args.config and args.config.exists():
        config = Gate1Config.from_yaml(args.config)
    else:
        config = Gate1Config()
    
    # Override with CLI args (only if explicitly provided, i.e., not None)
    cli_overrides = {
        'dataset_version': args.dataset_version,
        'system': args.system,
        'n_train': args.n_train,
        'seed': args.seed,
        'track': args.track,
        'note': args.note,
        'n_bootstrap': args.n_bootstrap,
        'library_config': args.library_config,
        'ridge_alpha': args.ridge_alpha,
        'final_fit_split': args.final_fit_split,
    }
    
    for key, value in cli_overrides.items():
        if value is not None:
            setattr(config, key, value)
    
    print("\n" + "=" * 70)
    print("  Gate1 E-SINDy Configuration")
    print("=" * 70)
    print(f"  Dataset: {config.dataset_version}")
    print(f"  n_train: {config.n_train}")
    print(f"  seed: {config.seed}")
    print(f"  track: {config.track}")
    print(f"  n_bootstrap: {config.n_bootstrap}")
    print(f"  thresholds: {config.thresholds}")
    print("=" * 70)
    
    runner = Gate1ESINDyRunner(config)
    result = runner.run()
    
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()