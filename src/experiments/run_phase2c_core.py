#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase 2-c Core Master Runner
==================================
합의된 Core 실험 매트릭스 실행

Core 실험 (10 신규 runs):
- k sweep (Align): pool=200, k={5,20} (k=10은 Phase 2-b 재사용)
- Ceiling (Align): pool=500, k={20,40}
- MMR: pool=500, k=40, λ=0.5
- Random baseline: pool=500, k=40, seed={0,1,2,3,4}

재사용 (Phase 2-b):
- pool=200, k=10, Align
- pool=200, k=10, Random×5

Usage:
    python src/experiments/run_phase2c_core.py --config configs/experiments/gate3_cartpole.yaml --teacher_run_dir <path>
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


def check_latent_for_mmr(pool_npz_path, pools_dir):
    """
    Check if latent file exists for MMR selection
    
    Returns:
        (exists, latent_path)
    """
    pool_name = Path(pool_npz_path).stem  # e.g., pool_n500_seed0
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


def auto_extract_latent(vae_path, pool_npz_path, pools_dir):
    """Auto-extract latent if missing (for MMR)"""
    print(f"\n[AUTO] Extracting latent for MMR...")
    cmd = [
        sys.executable, 'src/experiments/extract_latent.py',
        '--vae_path', str(vae_path),
        '--pool_npz', str(pool_npz_path),
        '--output_dir', str(pools_dir),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run Gate3 Phase 2-c Core experiments')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--teacher_run_dir', type=str, required=True, help='Gate1 run directory')
    parser.add_argument('--dataset_version', type=str, default='cartpole_ood_v1')
    parser.add_argument('--phase1plus_dir', type=str, default=None, help='Phase 1+ results dir (for reuse)')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without executing')
    args = parser.parse_args()
    
    # Paths
    if args.phase1plus_dir:
        phase1plus_dir = Path(args.phase1plus_dir)
    else:
        phase1plus_dir = RESULTS_ROOT / args.dataset_version / 'gate3_phase1plus'
    
    phase2c_dir = RESULTS_ROOT / args.dataset_version / 'gate3_phase2c'
    pools_dir = phase1plus_dir / 'pools'
    runs_dir = phase2c_dir / 'runs'
    
    print("=" * 70)
    print("Gate3 Phase 2-c Core Master Runner")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Teacher: {args.teacher_run_dir}")
    print(f"Phase 1+ dir: {phase1plus_dir}")
    print(f"Phase 2-c dir: {phase2c_dir}")
    print(f"Dry run: {args.dry_run}")
    
    # ==========================================================================
    # Core Experiment Matrix (합의본)
    # ==========================================================================
    
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
    
    # Check candidate counts
    print("\n[PREFLIGHT] Checking pool candidate counts...")
    ok_200, n_cand_200 = check_pool_candidates(pool_200_json, 20)
    ok_500, n_cand_500 = check_pool_candidates(pool_500_json, 40)
    print(f"  pool=200: {n_cand_200} candidates (need ≥20 for k=20)")
    print(f"  pool=500: {n_cand_500} candidates (need ≥40 for k=40)")
    
    if not ok_200:
        print(f"ERROR: pool=200 has insufficient candidates for k=20")
        return 1
    if not ok_500:
        print(f"ERROR: pool=500 has insufficient candidates for k=40")
        return 1
    
    # Check latent for MMR (P0: auto-extract if missing)
    print("\n[PREFLIGHT] Checking latent for MMR...")
    latent_exists, latent_path = check_latent_for_mmr(pool_500_npz, pools_dir)
    if latent_exists:
        print(f"  ✅ Latent found: {latent_path}")
    else:
        print(f"  ⚠️ Latent not found: {latent_path}")
        # Try to auto-extract
        vae_path = phase1plus_dir / 'models' / 'vae_seed0.pt'
        if vae_path.exists():
            print(f"  VAE found: {vae_path}")
            if not args.dry_run:
                if auto_extract_latent(vae_path, pool_500_npz, pools_dir):
                    print(f"  ✅ Latent extracted successfully")
                    # P0: Shape validation (GPT recommendation)
                    latent_exists_now, latent_path_now = check_latent_for_mmr(pool_500_npz, pools_dir)
                    if latent_exists_now:
                        latent_data = np.load(latent_path_now)
                        latent_shape = latent_data['latent_mu'].shape[0]
                        pool_data = np.load(pool_500_npz)
                        pool_n_gen = pool_data['x_candidates'].shape[0]
                        if latent_shape == pool_n_gen:
                            print(f"  ✅ Latent shape verified: {latent_shape} == pool n_generate")
                        else:
                            print(f"  ⚠️ Latent shape mismatch: {latent_shape} != {pool_n_gen}")
                else:
                    print(f"  ❌ Latent extraction failed. MMR experiment will be skipped.")
        else:
            print(f"  ❌ VAE not found: {vae_path}")
            print(f"  MMR experiment will fail without latent.")
    
    # ==========================================================================
    # Define experiments
    # ==========================================================================
    
    experiments = []
    
    # --- Group 1: k sweep on pool=200 (Align only) ---
    # k=10 is reused from Phase 2-b
    
    experiments.append({
        'name': 'n200_k5_align',
        'pool_npz': str(pool_200_npz),
        'pool_json': str(pool_200_json),
        'mode': 'align',
        'k': 5,
        'description': 'k sweep: pool=200, k=5, Align',
    })
    
    experiments.append({
        'name': 'n200_k20_align',
        'pool_npz': str(pool_200_npz),
        'pool_json': str(pool_200_json),
        'mode': 'align',
        'k': 20,
        'description': 'k sweep: pool=200, k=20, Align',
    })
    
    # --- Group 2: Ceiling check on pool=500 (Align) ---
    
    experiments.append({
        'name': 'n500_k20_align',
        'pool_npz': str(pool_500_npz),
        'pool_json': str(pool_500_json),
        'mode': 'align',
        'k': 20,
        'description': 'Ceiling: pool=500, k=20, Align',
    })
    
    experiments.append({
        'name': 'n500_k40_align',
        'pool_npz': str(pool_500_npz),
        'pool_json': str(pool_500_json),
        'mode': 'align',
        'k': 40,
        'description': 'Ceiling: pool=500, k=40, Align',
    })
    
    # --- Group 3: MMR test on pool=500, k=40 ---
    
    experiments.append({
        'name': 'n500_k40_mmr_l50',
        'pool_npz': str(pool_500_npz),
        'pool_json': str(pool_500_json),
        'mode': 'mmr',
        'k': 40,
        'mmr_lambda': 0.5,
        'description': 'MMR: pool=500, k=40, λ=0.5',
    })
    
    # --- Group 4: Random baseline on pool=500, k=40 (5 seeds) ---
    
    for seed in [0, 1, 2, 3, 4]:
        experiments.append({
            'name': f'n500_k40_random_s{seed}',
            'pool_npz': str(pool_500_npz),
            'pool_json': str(pool_500_json),
            'mode': 'random',
            'k': 40,
            'select_seed': seed,
            'description': f'Random baseline: pool=500, k=40, seed={seed}',
        })
    
    # ==========================================================================
    # Print experiment summary
    # ==========================================================================
    
    print(f"\n[PLAN] Total {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i:2d}. {exp['description']}")
    
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
    
    # ==========================================================================
    # Run experiments
    # ==========================================================================
    
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
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("Phase 2-c Core Summary")
    print("=" * 70)
    
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_failed = sum(1 for r in results if r['status'] == 'failed')
    n_dry = sum(1 for r in results if r['status'] == 'dry_run')
    
    if args.dry_run:
        print(f"Dry run complete. {n_dry} experiments would be executed.")
    else:
        print(f"Success: {n_success}/{len(experiments)}")
        print(f"Failed: {n_failed}/{len(experiments)}")
        
        if failed:
            print(f"\nFailed experiments:")
            for name in failed:
                print(f"  - {name}")
        
        print(f"\nResults directory: {runs_dir}")
    
    # Reuse info
    print("\n[NOTE] Phase 2-b results to reuse (not re-run):")
    print("  - n200_k10_align (from n200_align_t0)")
    print("  - n200_k10_random_s{0..4} (from n200_random_s{0..4})")
    
    # Save run summary
    if not args.dry_run:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': args.config,
            'teacher_run_dir': args.teacher_run_dir,
            'experiments': results,
            'reused_from_phase1plus': [
                'n200_align_t0 → n200_k10_align',
                'n200_random_s0~s4 → n200_k10_random_s0~s4',
            ],
        }
        
        summary_path = phase2c_dir / 'phase2c_core_summary.json'
        phase2c_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}")
    
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())