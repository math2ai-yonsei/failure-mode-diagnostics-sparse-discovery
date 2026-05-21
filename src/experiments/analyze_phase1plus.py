#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase1+ Results Analysis
==============================
18 runs 결과 집계 + Align vs Random 비교 리포트 생성

Usage:
    python src/experiments/analyze_phase1plus.py --results_dir results/cartpole_ood_v1/gate3_phase1plus
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import csv

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_run_metrics(run_dir):
    """Load metrics.json from a run directory"""
    metrics_path = run_dir / 'metrics.json'
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Analyze Gate3 Phase1+ results')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    runs_dir = results_dir / 'runs'
    
    if not runs_dir.exists():
        print(f"ERROR: runs directory not found: {runs_dir}")
        return 1
    
    print("=" * 70)
    print("Gate3 Phase1+ Results Analysis")
    print("=" * 70)
    print(f"Results dir: {results_dir}")
    
    # Collect all runs
    all_results = []
    n_generate_values = set()
    
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        metrics = load_run_metrics(run_dir)
        if metrics is None:
            print(f"  [WARN] No metrics.json in {run_dir.name}")
            continue
        
        # Parse run name
        name = run_dir.name
        if '_align_' in name:
            mode = 'align'
            n_gen = int(name.split('_')[0][1:])  # n100 -> 100
            seed = 0
        elif '_random_' in name:
            mode = 'random'
            n_gen = int(name.split('_')[0][1:])
            seed = int(name.split('_s')[1])
        else:
            print(f"  [WARN] Unknown run format: {name}")
            continue
        
        n_generate_values.add(n_gen)
        
        all_results.append({
            'name': name,
            'n_gen': n_gen,
            'mode': mode,
            'seed': seed,
            'test_r2': metrics['test_r2'],
            'val_r2': metrics['val_r2'],
            'train_r2_original': metrics.get('train_r2_original'),
            'train_r2_aug_selected': metrics.get('train_r2_aug_selected'),
            'train_r2_aug_candidates': metrics.get('train_r2_aug_candidates'),
            'train_r2_all': metrics.get('train_r2_all'),
            'sparsity': metrics['sparsity'],
            'best_threshold': metrics['best_threshold'],
        })
    
    print(f"\nTotal runs loaded: {len(all_results)}")
    n_generate_values = sorted(n_generate_values)
    print(f"n_generate values: {n_generate_values}")
    
    # Analysis per n_generate
    print("\n" + "=" * 70)
    print("Results by n_generate")
    print("=" * 70)
    
    summary = []
    
    for n_gen in n_generate_values:
        runs = [r for r in all_results if r['n_gen'] == n_gen]
        align_runs = [r for r in runs if r['mode'] == 'align']
        random_runs = [r for r in runs if r['mode'] == 'random']
        
        align_test_r2 = align_runs[0]['test_r2'] if align_runs else None
        random_test_r2s = [r['test_r2'] for r in random_runs]
        random_mean = np.mean(random_test_r2s) if random_test_r2s else None
        random_std = np.std(random_test_r2s) if random_test_r2s else None
        
        print(f"\n--- n_generate = {n_gen} ---")
        print(f"  Align Test R²:  {align_test_r2:.4f}" if align_test_r2 else "  Align: N/A")
        print(f"  Random Test R²: {random_mean:.4f} ± {random_std:.4f}" if random_mean else "  Random: N/A")
        
        if align_test_r2 and random_mean:
            delta = align_test_r2 - random_mean
            print(f"  Δ (Align - Random): {delta:+.4f}")
            winner = "Align" if delta > 0 else "Random"
            print(f"  Winner: {winner}")
        
        # Train R² breakdown (from align run)
        if align_runs:
            ar = align_runs[0]
            print(f"\n  Train R² breakdown (Align):")
            print(f"    original:      {ar['train_r2_original']:.4f}" if ar['train_r2_original'] else "")
            print(f"    aug_selected:  {ar['train_r2_aug_selected']:.4f}" if ar['train_r2_aug_selected'] else "")
            print(f"    aug_candidates:{ar['train_r2_aug_candidates']:.4f}" if ar['train_r2_aug_candidates'] else "")
            print(f"    all:           {ar['train_r2_all']:.4f}" if ar['train_r2_all'] else "")
        
        summary.append({
            'n_gen': n_gen,
            'align_test_r2': align_test_r2,
            'random_mean': random_mean,
            'random_std': random_std,
            'delta': delta if (align_test_r2 and random_mean) else None,
        })
    
    # Overall summary
    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)
    
    print(f"\n{'n_gen':<8} {'Align':<10} {'Random':<15} {'Delta':<10} {'Winner':<8}")
    print("-" * 55)
    for s in summary:
        align_str = f"{s['align_test_r2']:.4f}" if s['align_test_r2'] else "N/A"
        random_str = f"{s['random_mean']:.4f}±{s['random_std']:.4f}" if s['random_mean'] else "N/A"
        delta_str = f"{s['delta']:+.4f}" if s['delta'] else "N/A"
        winner = "Align" if s['delta'] and s['delta'] > 0 else "Random" if s['delta'] else "N/A"
        print(f"{s['n_gen']:<8} {align_str:<10} {random_str:<15} {delta_str:<10} {winner:<8}")
    
    # Success criteria check
    print("\n" + "=" * 70)
    print("Success Criteria Check")
    print("=" * 70)
    
    all_align_wins = all(s['delta'] and s['delta'] > 0 for s in summary if s['delta'])
    print(f"1차 (필수): Align > Random mean in ALL n_gen: {'✅ PASS' if all_align_wins else '❌ FAIL'}")
    
    # Improvement trend check
    if len(summary) >= 2:
        deltas = [s['delta'] for s in summary if s['delta']]
        improving = all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))
        print(f"2차 (필수): Improvement trend 100→500: {'✅ PASS' if improving else '⚠️ Check manually'}")
    
    # M3 baseline (0.9605 from Phase1-mini)
    m3_baseline = 0.9605
    best_align = max(s['align_test_r2'] for s in summary if s['align_test_r2'])
    print(f"3차 (도전): Best Align ({best_align:.4f}) ≥ M3 ({m3_baseline}): {'✅ PASS' if best_align >= m3_baseline else '❌ FAIL'}")
    
    # Save summary CSV
    output_path = args.output or (results_dir / 'phase1plus_summary.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_gen', 'align_test_r2', 'random_mean', 'random_std', 'delta'])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary saved: {output_path}")
    
    # Save full results
    full_results_path = results_dir / 'phase1plus_full_results.csv'
    with open(full_results_path, 'w', newline='') as f:
        fieldnames = ['name', 'n_gen', 'mode', 'seed', 'test_r2', 'val_r2', 
                      'train_r2_original', 'train_r2_aug_selected', 
                      'train_r2_aug_candidates', 'train_r2_all', 
                      'sparsity', 'best_threshold']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Full results saved: {full_results_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())