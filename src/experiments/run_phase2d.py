#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase 2-d Master Runner
==============================
Phase 2-c "Less is More" 발견 검증 및 k 최적점 탐색

Phase 2-d 실험 매트릭스 (9 runs, GPT/Claude 합의):
- k=5 Robustness: pool=200, k=5, Random×5
- k Fine-sweep: pool=200, k={3,7}, Align
- Pool invariance: pool=500, k=5, Align
- Diversity 결론: pool=500, k=40, MMR λ=0.9

Usage:
    python src/experiments/run_phase2d.py --config configs/experiments/gate3_cartpole.yaml --teacher_run_dir <path>
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.paths import RESULTS_ROOT


def run_command(cmd, description):
    """Execute command and check result"""
    print(f"\n{'='*70}")
    print(f"[STEP] {description}")
    print(f"{'='*70}")
    print(f"CMD: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        return False
    return True


def check_pool_candidates(pool_json_path, required_k):
    """Check if pool has enough candidates for k selection"""
    with open(pool_json_path) as f:
        pool_meta = json.load(f)
    
    n_dedup = pool_meta.get('filtering_summary', {}).get('n_dedup_pass', 0)
    if n_dedup < required_k:
        return False, n_dedup
    return True, n_dedup


def check_latent_exists(pool_npz_path, pools_dir):
    """Check if latent file exists"""
    pool_name = Path(pool_npz_path).stem
    parts = pool_name.split('_')
    n_gen = None
    seed = None
    for p in parts:
        if p.startswith('n') and p[1:].isdigit():
            n_gen = int(p[1:])
        elif p.startswith('seed') and p[4:].isdigit():
            seed = int(p[4:])
    
    if n_gen and seed is not None:
        latent_path = pools_dir / f'latent_n{n_gen}_seed{seed}.npz'
    else:
        latent_path = pools_dir / f'latent_{pool_name}.npz'
    
    return latent_path.exists(), latent_path


def main():
    parser = argparse.ArgumentParser(description='Run Gate3 Phase 2-d experiments')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--teacher_run_dir', type=str, required=True)
    parser.add_argument('--dataset_version', type=str, default='cartpole_ood_v1')
    parser.add_argument('--phase1plus_dir', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    # Paths
    if args.phase1plus_dir:
        phase1plus_dir = Path(args.phase1plus_dir)
    else:
        phase1plus_dir = RESULTS_ROOT / args.dataset_version / 'gate3_phase1plus'
    
    phase2d_dir = RESULTS_ROOT / args.dataset_version / 'gate3_phase2d'
    pools_dir = phase1plus_dir / 'pools'
    runs_dir = phase2d_dir / 'runs'
    
    print("=" * 70)
    print("Gate3 Phase 2-d Master Runner")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Teacher: {args.teacher_run_dir}")
    print(f"Phase 1+ dir: {phase1plus_dir}")
    print(f"Phase 2-d dir: {phase2d_dir}")
    print(f"Dry run: {args.dry_run}")
    
    # Pool paths
    pool_200_npz = pools_dir / 'pool_n200_seed0.npz'
    pool_200_json = pools_dir / 'pool_n200_seed0.json'
    pool_500_npz = pools_dir / 'pool_n500_seed0.npz'
    pool_500_json = pools_dir / 'pool_n500_seed0.json'
    
    # Check pools exist
    for p in [pool_200_npz, pool_200_json, pool_500_npz, pool_500_json]:
        if not p.exists():
            print(f"ERROR: Pool file not found: {p}")
            return 1
    
    # Preflight checks
    print("\n[PREFLIGHT] Checking pool candidate counts...")
    ok_200, n_cand_200 = check_pool_candidates(pool_200_json, 7)  # max k=7
    ok_500, n_cand_500 = check_pool_candidates(pool_500_json, 40)  # for MMR
    print(f"  pool=200: {n_cand_200} candidates (need ≥7 for k=7)")
    print(f"  pool=500: {n_cand_500} candidates (need ≥40 for MMR)")
    
    if not ok_200:
        print(f"ERROR: pool=200 has insufficient candidates")
        return 1
    if not ok_500:
        print(f"ERROR: pool=500 has insufficient candidates")
        return 1
    
    # Check latent for MMR
    print("\n[PREFLIGHT] Checking latent for MMR λ=0.9...")
    latent_exists, latent_path = check_latent_exists(pool_500_npz, pools_dir)
    if latent_exists:
        print(f"  ✅ Latent found: {latent_path}")
    else:
        print(f"  ❌ Latent not found: {latent_path}")
        print(f"  Run extract_latent.py first or Phase 2-c runner to generate it.")
        return 1
    
    # ==========================================================================
    # Phase 2-d Experiment Matrix (9 runs, confirmed)
    # ==========================================================================
    
    experiments = []
    
    # --- Group 1: k=5 Robustness (Random×5) ---
    for seed in [0, 1, 2, 3, 4]:
        experiments.append({
            'name': f'n200_k5_random_s{seed}',
            'pool_npz': str(pool_200_npz),
            'pool_json': str(pool_200_json),
            'mode': 'random',
            'k': 5,
            'select_seed': seed,
            'description': f'k=5 robustness: pool=200, Random seed={seed}',
        })
    
    # --- Group 2: k Fine-sweep (pool=200, Align) ---
    experiments.append({
        'name': 'n200_k3_align',
        'pool_npz': str(pool_200_npz),
        'pool_json': str(pool_200_json),
        'mode': 'align',
        'k': 3,
        'description': 'k fine-sweep: pool=200, k=3, Align',
    })
    
    experiments.append({
        'name': 'n200_k7_align',
        'pool_npz': str(pool_200_npz),
        'pool_json': str(pool_200_json),
        'mode': 'align',
        'k': 7,
        'description': 'k fine-sweep: pool=200, k=7, Align',
    })
    
    # --- Group 3: Pool invariance (pool=500, k=5, Align) ---
    experiments.append({
        'name': 'n500_k5_align',
        'pool_npz': str(pool_500_npz),
        'pool_json': str(pool_500_json),
        'mode': 'align',
        'k': 5,
        'description': 'Pool invariance: pool=500, k=5, Align',
    })
    
    # --- Group 4: Diversity conclusion (MMR λ=0.9) ---
    experiments.append({
        'name': 'n500_k40_mmr_l90',
        'pool_npz': str(pool_500_npz),
        'pool_json': str(pool_500_json),
        'mode': 'mmr',
        'k': 40,
        'mmr_lambda': 0.9,
        'description': 'Diversity test: pool=500, k=40, MMR λ=0.9',
    })
    
    # Print plan
    print(f"\n[PLAN] Total {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i:2d}. {exp['description']}")
    
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
    
    # Run experiments
    results = []
    failed = []
    
    for i, exp in enumerate(experiments, 1):
        output_dir = runs_dir / exp['name']
        
        cmd = [
            sys.executable, 'src/experiments/evaluate_selection.py',
            '--config', args.config,
            '--pool_npz', exp['pool_npz'],
            '--pool_json', exp['pool_json'],
            '--teacher_run_dir', args.teacher_run_dir,
            '--selection_mode', exp['mode'],
            '--k', str(exp['k']),
            '--output_dir', str(output_dir),
        ]
        
        # Mode-specific args
        if exp['mode'] == 'random':
            cmd.extend(['--select_seed', str(exp['select_seed'])])
        elif exp['mode'] == 'mmr':
            cmd.extend(['--mmr_lambda', str(exp['mmr_lambda'])])
        
        if args.dry_run:
            print(f"\n[{i}/{len(experiments)}] {exp['name']}")
            print(f"  {' '.join(cmd)}")
            results.append({'name': exp['name'], 'status': 'dry_run'})
        else:
            success = run_command(cmd, f"[{i}/{len(experiments)}] {exp['description']}")
            if success:
                results.append({'name': exp['name'], 'status': 'success'})
            else:
                results.append({'name': exp['name'], 'status': 'failed'})
                failed.append(exp['name'])
    
    # Summary
    print("\n" + "=" * 70)
    print("Phase 2-d Summary")
    print("=" * 70)
    
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_failed = sum(1 for r in results if r['status'] == 'failed')
    
    if args.dry_run:
        print(f"Dry run complete. {len(experiments)} experiments would be executed.")
    else:
        print(f"Success: {n_success}/{len(experiments)}")
        print(f"Failed: {n_failed}/{len(experiments)}")
        
        if failed:
            print(f"\nFailed experiments:")
            for name in failed:
                print(f"  - {name}")
        
        print(f"\nResults directory: {runs_dir}")
    
    # Phase 2-c results to combine
    print("\n[NOTE] Combine with Phase 2-c results for full analysis:")
    print("  - n200_k5_align (Phase 2-c)")
    print("  - n200_k10_align (Phase 2-b)")
    print("  - n200_k20_align (Phase 2-c)")
    print("  - n500_k40_align (Phase 2-c)")
    print("  - n500_k40_mmr_l50 (Phase 2-c)")
    
    # Save summary
    if not args.dry_run:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': args.config,
            'teacher_run_dir': args.teacher_run_dir,
            'experiments': results,
            'purpose': 'Phase 2-d: k=5 robustness + k fine-sweep + pool invariance + MMR λ=0.9',
        }
        
        summary_path = phase2d_dir / 'phase2d_summary.json'
        phase2d_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}")
    
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())