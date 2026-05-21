#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 CLI Entry Point
=====================
VAE 기반 궤적 생성 + Teacher Alignment 실험 실행.

Lock-6 준수: --baseline 플래그로 대조군 선택 (단일 진입점)

Usage:
    # Phase -1: Smoke test
    python experiments/run_gate3.py --config configs/experiments/gate3_cartpole.yaml \\
        --dataset_version cartpole_ood_v1 --n_train 10 --data_seed 0 \\
        --phase phase-1 --note smoke
    
    # Phase 1: Full run (제안법)
    python experiments/run_gate3.py --config configs/experiments/gate3_cartpole.yaml \\
        --dataset_version cartpole_ood_v1 --n_train 10 --data_seed 0 \\
        --vae_seed 0 --gen_seed 0 --baseline none \\
        --gate1_baseline_run_id 20251229_213749_nogit_base --note m1_s0_v0
    
    # M2: Gen-only (no align filter)
    python experiments/run_gate3.py ... --baseline gen_only --note m2_s0
    
    # M3: Copy-only
    python experiments/run_gate3.py ... --baseline copy_only --note m3_s0
    
    # M4: Noise augmentation
    python experiments/run_gate3.py ... --baseline noise_aug --note m4_s0
    
    # M5: Random select
    python experiments/run_gate3.py ... --baseline random_select --note m5_s0
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gate3 Generative Augmentation Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--dataset_version', type=str, default='cartpole_ood_v1',
                        help='Dataset version')
    
    # Seeds (Lock-1: CLI → seeds.*)
    parser.add_argument('--data_seed', type=int, default=None,
                        help='Data split seed (→ seeds.data)')
    parser.add_argument('--vae_seed', type=int, default=None,
                        help='VAE training seed (→ seeds.vae)')
    parser.add_argument('--gen_seed', type=int, default=None,
                        help='Generation seed (→ seeds.gen)')
    
    # Experiment settings
    parser.add_argument('--n_train', type=int, default=None,
                        help='Number of training trajectories')
    parser.add_argument('--track', type=str, default=None,
                        choices=['standardized', 'author_recommended'],
                        help='Experiment track')
    parser.add_argument('--aug_ratio', type=float, default=None,
                        help='Augmentation ratio')
    
    # Baseline (Lock-6: 단일 진입점)
    parser.add_argument('--baseline', type=str, default='none',
                        choices=['none', 'gen_only', 'copy_only', 'noise_aug', 'random_select'],
                        help='Baseline method: none=proposed, gen_only=M2, copy_only=M3, '
                             'noise_aug=M4, random_select=M5')
    
    # Phase
    parser.add_argument('--phase', type=str, default='phase1',
                        choices=['phase-1', 'phase0', 'phase1', 'phase2'],
                        help='Experiment phase: phase-1=smoke, phase0=VAE only, '
                             'phase1=full, phase2=align in training')
    
    # Comparison (Lock-2)
    parser.add_argument('--gate1_baseline_run_id', type=str, default=None,
                        help='Gate1 baseline run_id for delta computation')
    parser.add_argument('--gate2_baseline_run_id', type=str, default=None,
                        help='Gate2 baseline run_id for matched comparison')
    
    # Misc
    parser.add_argument('--note', type=str, default='base',
                        help='Note for run_id')
    parser.add_argument('--project_root', type=str, default=None,
                        help='Project root directory (auto-detected if not specified)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Find project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Auto-detect: look for src/ directory
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / 'src').exists() and (parent / 'configs').exists():
                project_root = parent
                break
        else:
            # Fallback
            project_root = Path.cwd()
    
    # Add to path
    sys.path.insert(0, str(project_root))
    
    # Import after path setup
    from src.experiments.gate3_gen_runner import Gate3Config, Gate3Runner
    
    # Build CLI overrides
    cli_overrides = {
        'dataset_version': args.dataset_version,
        'data_seed': args.data_seed,
        'vae_seed': args.vae_seed,
        'gen_seed': args.gen_seed,
        'n_train': args.n_train,
        'track': args.track,
        'aug_ratio': args.aug_ratio,
        'baseline': args.baseline,
        'phase': args.phase,
        'gate1_baseline_run_id': args.gate1_baseline_run_id,
        'gate2_baseline_run_id': args.gate2_baseline_run_id,
        'note': args.note,
    }
    
    # Remove None values
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = Gate3Config.from_yaml(config_path, cli_overrides)
    
    # Create and run
    runner = Gate3Runner(config, project_root)
    
    try:
        result = runner.run()
        
        if result['status'] in ['success', 'phase0_complete']:
            print(f"✅ Success! Results saved to: {result.get('results_dir', 'N/A')}")
        else:
            print(f"⚠️ Completed with status: {result['status']}")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()