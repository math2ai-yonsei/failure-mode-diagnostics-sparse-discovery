#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase1+ Master Runner
===========================
전체 파이프라인 실행: VAE 학습 → Pool 생성 → 18 runs 평가

Usage:
    python src/experiments/run_phase1plus.py --config configs/experiments/gate3_cartpole.yaml --teacher_run_dir <path>
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.paths import RESULTS_ROOT


def run_command(cmd, description):
    """Execute command and check result"""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"CMD: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Run Gate3 Phase1+ full pipeline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--teacher_run_dir', type=str, required=True, help='Gate1 run directory')
    parser.add_argument('--vae_seed', type=int, default=0)
    parser.add_argument('--gen_seed', type=int, default=0)
    parser.add_argument('--k', type=int, default=10, help='Number of samples to select')
    parser.add_argument('--dataset_version', type=str, default='cartpole_ood_v1')
    parser.add_argument('--skip_vae', action='store_true', help='Skip VAE training')
    parser.add_argument('--skip_pool', action='store_true', help='Skip pool generation')
    args = parser.parse_args()
    
    base_dir = RESULTS_ROOT / args.dataset_version / 'gate3_phase1plus'
    models_dir = base_dir / 'models'
    pools_dir = base_dir / 'pools'
    runs_dir = base_dir / 'runs'
    
    print("=" * 70)
    print("Gate3 Phase1+ Master Runner")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Teacher: {args.teacher_run_dir}")
    print(f"Output: {base_dir}")
    print(f"VAE seed: {args.vae_seed}, Gen seed: {args.gen_seed}, k: {args.k}")
    
    n_generate_list = [100, 200, 500]
    random_seeds = [0, 1, 2, 3, 4]
    
    # Step 1: Train VAE
    vae_path = models_dir / f'vae_seed{args.vae_seed}.pt'
    if not args.skip_vae:
        cmd = [
            sys.executable, 'src/experiments/train_vae.py',
            '--config', args.config,
            '--vae_seed', str(args.vae_seed),
            '--output_dir', str(models_dir),
        ]
        if not run_command(cmd, "Train VAE"):
            return 1
    else:
        print(f"\n[SKIP] VAE training (using existing: {vae_path})")
    
    # Step 2: Generate Pools
    if not args.skip_pool:
        for n_gen in n_generate_list:
            cmd = [
                sys.executable, 'src/experiments/generate_pool.py',
                '--config', args.config,
                '--vae_path', str(vae_path),
                '--teacher_run_dir', args.teacher_run_dir,
                '--n_generate', str(n_gen),
                '--gen_seed', str(args.gen_seed),
                '--output_dir', str(pools_dir),
            ]
            if not run_command(cmd, f"Generate Pool n={n_gen}"):
                return 1
    else:
        print("\n[SKIP] Pool generation")
    
    # Step 3: Run evaluations (18 runs)
    results = []
    
    for n_gen in n_generate_list:
        pool_npz = pools_dir / f'pool_n{n_gen}_seed{args.gen_seed}.npz'
        pool_json = pools_dir / f'pool_n{n_gen}_seed{args.gen_seed}.json'
        
        # Align run (1 per pool)
        cmd = [
            sys.executable, 'src/experiments/evaluate_selection.py',
            '--config', args.config,
            '--pool_npz', str(pool_npz),
            '--pool_json', str(pool_json),
            '--teacher_run_dir', args.teacher_run_dir,
            '--selection_mode', 'align',
            '--k', str(args.k),
            '--output_dir', str(runs_dir / f'n{n_gen}_align_t0'),
        ]
        if not run_command(cmd, f"Align n={n_gen}"):
            return 1
        results.append({'n_gen': n_gen, 'mode': 'align', 'seed': 0})
        
        # Random runs (5 per pool)
        for seed in random_seeds:
            cmd = [
                sys.executable, 'src/experiments/evaluate_selection.py',
                '--config', args.config,
                '--pool_npz', str(pool_npz),
                '--pool_json', str(pool_json),
                '--teacher_run_dir', args.teacher_run_dir,
                '--selection_mode', 'random',
                '--k', str(args.k),
                '--select_seed', str(seed),
                '--output_dir', str(runs_dir / f'n{n_gen}_random_s{seed}'),
            ]
            if not run_command(cmd, f"Random n={n_gen} seed={seed}"):
                return 1
            results.append({'n_gen': n_gen, 'mode': 'random', 'seed': seed})
    
    # Summary
    print("\n" + "=" * 70)
    print("Phase1+ Complete!")
    print("=" * 70)
    print(f"Total runs: {len(results)}")
    print(f"Results directory: {base_dir}")
    print("\nRun summary:")
    for r in results:
        print(f"  n={r['n_gen']:3d}, {r['mode']:6s}, seed={r['seed']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())