#!/usr/bin/env python
"""
S06: Gate0 CLI Entry Point

Runs the Gate0 SINDy baseline pipeline.

Usage:
    python experiments/run_gate0.py ^
      --config configs/experiments/gate0_cartpole.yaml ^
      --dataset_version cartpole_ood_v1 ^
      --n_train 10 --seed 0 --track standardized --note base

Arguments:
    --config: YAML config file (optional, defaults used if not provided)
    --dataset_version: Dataset version (e.g., cartpole_ood_v1)
    --n_train: Number of training trajectories
    --seed: Random seed for trajectory selection
    --track: 'standardized' (Savgol dx) or 'author_recommended' (analytic dx)
    --note: Note for run_id (e.g., 'base', 'test')
    --threshold: STLSQ threshold (default: 0.01)
"""
import os
# Set matplotlib backend for headless operation (before any matplotlib imports)
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.experiments.gate0_runner import Gate0Runner, Gate0Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Gate0 SINDy Baseline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run
    python experiments/run_gate0.py --n_train 10 --seed 0
    
    # With config file
    python experiments/run_gate0.py --config configs/experiments/gate0_cartpole.yaml
    
    # Full specification
    python experiments/run_gate0.py ^
      --dataset_version cartpole_ood_v1 ^
      --n_train 10 --seed 0 ^
      --track standardized ^
      --threshold 0.01 ^
      --note baseline
        """
    )
    
    # Config file (optional)
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=None,
        help='YAML config file path'
    )
    
    # Dataset settings
    parser.add_argument(
        '--dataset_version', '-d',
        type=str,
        default='cartpole_ood_v1',
        help='Dataset version (default: cartpole_ood_v1)'
    )
    parser.add_argument(
        '--system',
        type=str,
        default='cartpole',
        help='System name (default: cartpole)'
    )
    
    # Experiment settings
    parser.add_argument(
        '--n_train', '-n',
        type=int,
        default=10,
        help='Number of training trajectories (default: 10)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=0,
        help='Random seed (default: 0)'
    )
    parser.add_argument(
        '--track', '-t',
        type=str,
        choices=['standardized', 'author_recommended'],
        default='standardized',
        help='Experiment track (default: standardized)'
    )
    parser.add_argument(
        '--note',
        type=str,
        default='base',
        help='Note for run_id (default: base)'
    )
    
    # SINDy settings
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='STLSQ threshold (default: 0.01)'
    )
    parser.add_argument(
        '--library_config',
        type=str,
        default='gate0_min',
        help='Library configuration (default: gate0_min)'
    )
    parser.add_argument(
        '--ridge_alpha',
        type=float,
        default=0.0,
        help='Ridge regularization (default: 0.0)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build config
    if args.config and args.config.exists():
        # Load from YAML and override with CLI args
        config = Gate0Config.from_yaml(args.config)
        
        # Override with CLI args if provided
        for key in ['dataset_version', 'system', 'n_train', 'seed', 
                    'track', 'note', 'threshold', 'library_config', 'ridge_alpha']:
            cli_value = getattr(args, key)
            default_value = getattr(Gate0Config(), key)
            if cli_value != default_value:
                setattr(config, key, cli_value)
    else:
        # Build from CLI args
        config = Gate0Config(
            dataset_version=args.dataset_version,
            system=args.system,
            n_train=args.n_train,
            seed=args.seed,
            track=args.track,
            note=args.note,
            threshold=args.threshold,
            library_config=args.library_config,
            ridge_alpha=args.ridge_alpha,
        )
    
    # Print config summary
    print("\n" + "=" * 70)
    print("  Gate0 Configuration")
    print("=" * 70)
    print(f"  Dataset: {config.dataset_version}")
    print(f"  n_train: {config.n_train}")
    print(f"  seed: {config.seed}")
    print(f"  track: {config.track}")
    print(f"  threshold: {config.threshold}")
    print(f"  library: {config.library_config}")
    print("=" * 70)
    
    # Run
    runner = Gate0Runner(config)
    result = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()