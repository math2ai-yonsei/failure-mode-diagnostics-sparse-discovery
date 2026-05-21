#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase 2-c Results Analysis
================================
Core 실험 결과 분석 및 시각화

출력:
- 테이블: k sweep, Selection method comparison
- 그림: F01_k_sweep.png, F02_selection_comparison.png, F03_structure_metrics.png

Usage:
    python src/experiments/analyze_phase2c.py --results_dir results/cartpole_ood_v1/gate3_phase2c
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

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


def parse_run_name(name):
    """
    Parse run directory name to extract parameters
    
    Examples:
        n200_k5_align -> {'pool': 200, 'k': 5, 'mode': 'align'}
        n500_k40_random_s2 -> {'pool': 500, 'k': 40, 'mode': 'random', 'seed': 2}
        n500_k40_mmr_l50 -> {'pool': 500, 'k': 40, 'mode': 'mmr', 'lambda': 0.5}
    """
    parts = name.split('_')
    result = {}
    
    # Pool size
    for p in parts:
        if p.startswith('n') and p[1:].isdigit():
            result['pool'] = int(p[1:])
        elif p.startswith('k') and p[1:].isdigit():
            result['k'] = int(p[1:])
        elif p in ['align', 'random', 'mmr']:
            result['mode'] = p
        elif p.startswith('s') and p[1:].isdigit():
            result['seed'] = int(p[1:])
        elif p.startswith('l') and p[1:].isdigit():
            result['lambda'] = int(p[1:]) / 100
    
    return result


def collect_results(results_dir, include_phase1plus=True, phase1plus_dir=None, phase2d_dir=None):
    """
    Collect all run results from Phase 2-c, Phase 2-d, and optionally Phase 1+
    
    Returns:
        list of dicts with run info and metrics
    """
    results = []
    
    # Phase 2-c runs
    runs_dir = results_dir / 'runs'
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                metrics = load_run_metrics(run_dir)
                if metrics:
                    params = parse_run_name(run_dir.name)
                    results.append({
                        'name': run_dir.name,
                        'source': 'phase2c',
                        **params,
                        'test_r2': metrics.get('test_r2'),
                        'val_r2': metrics.get('val_r2'),
                        'sparsity': metrics.get('sparsity'),
                        'structure_metrics': metrics.get('structure_metrics', {}),
                    })
    
    # Phase 2-d runs
    if phase2d_dir:
        p2d_runs = phase2d_dir / 'runs'
        if p2d_runs.exists():
            for run_dir in sorted(p2d_runs.iterdir()):
                if run_dir.is_dir():
                    metrics = load_run_metrics(run_dir)
                    if metrics:
                        params = parse_run_name(run_dir.name)
                        results.append({
                            'name': run_dir.name,
                            'source': 'phase2d',
                            **params,
                            'test_r2': metrics.get('test_r2'),
                            'val_r2': metrics.get('val_r2'),
                            'sparsity': metrics.get('sparsity'),
                            'structure_metrics': metrics.get('structure_metrics', {}),
                        })
    
    # Phase 1+ runs (for reuse)
    if include_phase1plus and phase1plus_dir:
        p1plus_runs = phase1plus_dir / 'runs'
        if p1plus_runs.exists():
            # Map Phase 1+ naming to Phase 2-c naming
            name_map = {
                'n200_align_t0': {'pool': 200, 'k': 10, 'mode': 'align'},
                'n200_random_s0': {'pool': 200, 'k': 10, 'mode': 'random', 'seed': 0},
                'n200_random_s1': {'pool': 200, 'k': 10, 'mode': 'random', 'seed': 1},
                'n200_random_s2': {'pool': 200, 'k': 10, 'mode': 'random', 'seed': 2},
                'n200_random_s3': {'pool': 200, 'k': 10, 'mode': 'random', 'seed': 3},
                'n200_random_s4': {'pool': 200, 'k': 10, 'mode': 'random', 'seed': 4},
            }
            
            for run_dir in sorted(p1plus_runs.iterdir()):
                if run_dir.is_dir() and run_dir.name in name_map:
                    metrics = load_run_metrics(run_dir)
                    if metrics:
                        params = name_map[run_dir.name]
                        results.append({
                            'name': run_dir.name,
                            'source': 'phase1plus',
                            **params,
                            'test_r2': metrics.get('test_r2'),
                            'val_r2': metrics.get('val_r2'),
                            'sparsity': metrics.get('sparsity'),
                            'structure_metrics': metrics.get('structure_metrics', {}),
                        })
    
    return results


def analyze_k_sweep(results):
    """
    Analyze k sweep results (pool=200, Align)
    
    Returns:
        dict with k -> test_r2 mapping
    """
    k_results = {}
    
    for r in results:
        if r.get('pool') == 200 and r.get('mode') == 'align':
            k = r.get('k')
            if k:
                k_results[k] = {
                    'test_r2': r['test_r2'],
                    'val_r2': r['val_r2'],
                    'sparsity': r['sparsity'],
                }
    
    return dict(sorted(k_results.items()))


def analyze_ceiling(results):
    """
    Analyze ceiling results (pool=500, Align)
    
    Returns:
        dict with k -> test_r2 mapping
    """
    ceiling_results = {}
    
    for r in results:
        if r.get('pool') == 500 and r.get('mode') == 'align':
            k = r.get('k')
            if k:
                ceiling_results[k] = {
                    'test_r2': r['test_r2'],
                    'val_r2': r['val_r2'],
                }
    
    return dict(sorted(ceiling_results.items()))


def analyze_random_baseline(results, pool, k):
    """
    Analyze random baseline results
    
    Returns:
        dict with mean, std, min, max, individual seeds
    """
    random_results = []
    
    for r in results:
        if r.get('pool') == pool and r.get('k') == k and r.get('mode') == 'random':
            random_results.append({
                'seed': r.get('seed'),
                'test_r2': r['test_r2'],
            })
    
    if not random_results:
        return None
    
    test_r2_values = [r['test_r2'] for r in random_results]
    
    return {
        'mean': np.mean(test_r2_values),
        'std': np.std(test_r2_values),
        'min': np.min(test_r2_values),
        'max': np.max(test_r2_values),
        'floor': np.mean(test_r2_values) - np.std(test_r2_values),  # Mean - 1σ
        'n_seeds': len(random_results),
        'seeds': random_results,
    }


def analyze_selection_comparison(results, pool, k):
    """
    Compare Align, MMR, Random at same (pool, k)
    
    Returns:
        dict with mode -> metrics
    """
    comparison = {}
    mmr_results = []  # Store all MMR results with different λ
    
    for r in results:
        if r.get('pool') == pool and r.get('k') == k:
            mode = r.get('mode')
            if mode == 'align':
                comparison['align'] = {
                    'test_r2': r['test_r2'],
                    'structure': r.get('structure_metrics', {}),
                }
            elif mode == 'mmr':
                mmr_results.append({
                    'test_r2': r['test_r2'],
                    'lambda': r.get('lambda'),
                    'structure': r.get('structure_metrics', {}),
                })
    
    # Store MMR results (latest one in 'mmr', all in 'mmr_all')
    if mmr_results:
        # Sort by lambda descending (λ=0.9 before λ=0.5)
        mmr_results.sort(key=lambda x: x.get('lambda', 0) or 0, reverse=True)
        comparison['mmr'] = mmr_results[0]  # Use highest λ as primary
        if len(mmr_results) > 1:
            comparison['mmr_all'] = mmr_results
    
    # Random aggregated
    random_baseline = analyze_random_baseline(results, pool, k)
    if random_baseline:
        comparison['random'] = random_baseline
    
    return comparison


def compute_inversion_frequency(align_r2, random_results):
    """
    Compute inversion frequency: how often random beats align
    
    Returns:
        dict with count, frequency, seeds that beat align
    """
    if not random_results or 'seeds' not in random_results:
        return None
    
    beats_align = [s for s in random_results['seeds'] if s['test_r2'] > align_r2]
    
    return {
        'n_inversions': len(beats_align),
        'n_total': random_results['n_seeds'],
        'frequency': len(beats_align) / random_results['n_seeds'],
        'beating_seeds': beats_align,
    }


def print_analysis_report(results, output_dir=None):
    """
    Print comprehensive analysis report
    """
    print("\n" + "=" * 70)
    print("Gate3 Phase 2-c/d Integrated Analysis Report")
    print("=" * 70)
    
    # 1. k Sweep Analysis (pool=200)
    print("\n### 1. k Sweep Analysis (pool=200, Align)")
    print("-" * 50)
    
    k_sweep = analyze_k_sweep(results)
    if k_sweep:
        print(f"{'k':>4} | {'Test R²':>10} | {'Val R²':>10} | {'Sparsity':>10}")
        print("-" * 50)
        for k, metrics in k_sweep.items():
            print(f"{k:4d} | {metrics['test_r2']:10.4f} | {metrics['val_r2']:10.4f} | {metrics['sparsity']:10.1%}")
        
        # Ceiling improvement
        k_values = sorted(k_sweep.keys())
        if len(k_values) >= 2:
            k_min, k_max = k_values[0], k_values[-1]
            r2_min, r2_max = k_sweep[k_min]['test_r2'], k_sweep[k_max]['test_r2']
            print(f"\nCeiling Δ (k={k_min}→{k_max}): {r2_max - r2_min:+.4f}")
    else:
        print("No k sweep results found")
    
    # 2. Pool-k Interaction (ceiling check)
    print("\n### 2. Ceiling Check (pool=500, Align)")
    print("-" * 50)
    
    ceiling = analyze_ceiling(results)
    if ceiling:
        print(f"{'k':>4} | {'Test R²':>10}")
        print("-" * 30)
        for k, metrics in ceiling.items():
            print(f"{k:4d} | {metrics['test_r2']:10.4f}")
    else:
        print("No ceiling results found")
    
    # 3. Selection Method Comparison (pool=500, k=40)
    print("\n### 3. Selection Method Comparison (pool=500, k=40)")
    print("-" * 50)
    
    comparison = analyze_selection_comparison(results, pool=500, k=40)
    if comparison:
        print(f"{'Mode':>12} | {'Test R²':>10} | {'Jaccard':>10} | {'Coef Corr':>10}")
        print("-" * 55)
        
        # Align
        if 'align' in comparison:
            m = comparison['align']
            jaccard = f"{m['structure'].get('jaccard', 0):.4f}"
            corr = f"{m['structure'].get('coef_correlation', 0):.4f}"
            print(f"{'align':>12} | {m['test_r2']:10.4f} | {jaccard:>10} | {corr:>10}")
        
        # All MMR results
        if 'mmr_all' in comparison:
            for mmr in comparison['mmr_all']:
                lam = mmr.get('lambda', 0.5)
                label = f"mmr λ={lam}"
                jaccard = f"{mmr['structure'].get('jaccard', 0):.4f}"
                corr = f"{mmr['structure'].get('coef_correlation', 0):.4f}"
                print(f"{label:>12} | {mmr['test_r2']:10.4f} | {jaccard:>10} | {corr:>10}")
        elif 'mmr' in comparison:
            mmr = comparison['mmr']
            lam = mmr.get('lambda', 0.5)
            label = f"mmr λ={lam}"
            jaccard = f"{mmr['structure'].get('jaccard', 0):.4f}"
            corr = f"{mmr['structure'].get('coef_correlation', 0):.4f}"
            print(f"{label:>12} | {mmr['test_r2']:10.4f} | {jaccard:>10} | {corr:>10}")
        
        # Random
        if 'random' in comparison:
            m = comparison['random']
            r2_str = f"{m['mean']:.4f}±{m['std']:.3f}"
            print(f"{'random':>12} | {r2_str:>10} | {'-':>10} | {'-':>10}")
        
        # Inversion analysis
        if 'align' in comparison and 'random' in comparison:
            inversion = compute_inversion_frequency(
                comparison['align']['test_r2'], 
                comparison['random']
            )
            if inversion:
                print(f"\nInversion frequency: {inversion['n_inversions']}/{inversion['n_total']} "
                      f"({inversion['frequency']:.0%})")
    else:
        print("No comparison results found")
    
    # 4. Robustness Analysis (floor comparison)
    print("\n### 4. Robustness Analysis")
    print("-" * 50)
    
    # Compare floors at (200, k=5), (200, k=10), and (500, k=40)
    for pool, k in [(200, 5), (200, 10), (500, 40)]:
        random = analyze_random_baseline(results, pool, k)
        align_results = [r for r in results if r.get('pool') == pool and r.get('k') == k and r.get('mode') == 'align']
        
        if random and align_results:
            align_r2 = align_results[0]['test_r2']
            print(f"\nPool={pool}, k={k}:")
            print(f"  Align:         {align_r2:.4f}")
            print(f"  Random Mean:   {random['mean']:.4f} ± {random['std']:.3f}")
            print(f"  Random Floor:  {random['floor']:.4f} (Mean - 1σ)")
            print(f"  Δ (Align vs Floor): {align_r2 - random['floor']:+.4f}")
    
    # 5. Pattern → Cause Mapping
    print("\n### 5. Observed Patterns → Cause Interpretation")
    print("-" * 50)
    
    k_sweep = analyze_k_sweep(results)
    if k_sweep and len(k_sweep) >= 2:
        k_values = sorted(k_sweep.keys())
        r2_values = [k_sweep[k]['test_r2'] for k in k_values]
        
        # Check if ceiling increases with k
        ceiling_increase = r2_values[-1] - r2_values[0]
        
        if ceiling_increase > 0.02:
            print("✅ Pattern: k↑ → ceiling↑")
            print("   → Interpretation: k=10 was bottleneck, larger k helps")
        elif ceiling_increase < -0.01:
            print("⚠️ Pattern: k↑ → ceiling↓")
            print("   → Interpretation: More data hurts (possible overfitting/noise)")
        else:
            print("📊 Pattern: k↑ → ceiling stable")
            print("   → Interpretation: Generator quality is the ceiling, not k")
    
    comparison = analyze_selection_comparison(results, pool=500, k=40)
    if comparison and 'align' in comparison:
        align_r2 = comparison['align']['test_r2']
        
        # Check all MMR results
        if 'mmr_all' in comparison:
            mmr_results = comparison['mmr_all']
            print(f"\n📊 MMR Analysis (pool=500, k=40):")
            for mmr in mmr_results:
                lam = mmr.get('lambda', 0.5)
                diff = mmr['test_r2'] - align_r2
                if diff > 0.01:
                    status = ">"
                elif diff < -0.01:
                    status = "<"
                else:
                    status = "≈"
                print(f"   MMR λ={lam}: {mmr['test_r2']:.4f} {status} Align {align_r2:.4f} (Δ={diff:+.4f})")
            
            # Conclusion based on highest λ (closest to pure align)
            high_lam_mmr = max(mmr_results, key=lambda x: x.get('lambda', 0) or 0)
            if high_lam_mmr['test_r2'] < align_r2 - 0.01:
                print("   → Conclusion: Diversity is harmful, pure alignment is best")
            elif high_lam_mmr['test_r2'] > align_r2 + 0.01:
                print("   → Conclusion: Diversity helps, consider adjusting λ")
            else:
                print("   → Conclusion: λ=0.9 ≈ Align, diversity is neutral at best")
        elif 'mmr' in comparison:
            mmr_r2 = comparison['mmr']['test_r2']
            if mmr_r2 > align_r2 + 0.01:
                print("✅ Pattern: MMR > Align")
                print("   → Interpretation: Diversity helps, collapse was happening")
            elif mmr_r2 < align_r2 - 0.01:
                print("📊 Pattern: MMR < Align")
                print("   → Interpretation: Pure alignment is better, no collapse")
            else:
                print("📊 Pattern: MMR ≈ Align")
                print("   → Interpretation: Diversity neutral, focus on other factors")
    
    print("\n" + "=" * 70)


def save_summary_json(results, output_path):
    """Save analysis summary as JSON"""
    summary = {
        'k_sweep_200': analyze_k_sweep(results),
        'ceiling_500': analyze_ceiling(results),
        'comparison_500_k40': analyze_selection_comparison(results, 500, 40),
        'random_baseline_200_k5': analyze_random_baseline(results, 200, 5),
        'random_baseline_200_k10': analyze_random_baseline(results, 200, 10),
        'random_baseline_500_k40': analyze_random_baseline(results, 500, 40),
    }
    
    # Add inversion analysis
    comp = summary.get('comparison_500_k40', {})
    if 'align' in comp and 'random' in comp:
        summary['inversion_500_k40'] = compute_inversion_frequency(
            comp['align']['test_r2'],
            comp['random']
        )
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Gate3 Phase 2-c/d results')
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Phase 2-c results directory')
    parser.add_argument('--phase1plus_dir', type=str, default=None,
                        help='Phase 1+ directory for reused runs')
    parser.add_argument('--phase2d_dir', type=str, default=None,
                        help='Phase 2-d results directory')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Output JSON path for summary')
    parser.add_argument('--no_phase1plus', action='store_true',
                        help='Do not include Phase 1+ results')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Auto-detect phase1plus directory
    if args.phase1plus_dir:
        phase1plus_dir = Path(args.phase1plus_dir)
    else:
        # Assume same parent directory
        phase1plus_dir = results_dir.parent / 'gate3_phase1plus'
        if not phase1plus_dir.exists():
            phase1plus_dir = None
    
    # Auto-detect phase2d directory
    if args.phase2d_dir:
        phase2d_dir = Path(args.phase2d_dir)
    else:
        # Assume same parent directory
        phase2d_dir = results_dir.parent / 'gate3_phase2d'
        if not phase2d_dir.exists():
            phase2d_dir = None
    
    print(f"Results dir: {results_dir}")
    print(f"Phase 1+ dir: {phase1plus_dir}")
    print(f"Phase 2-d dir: {phase2d_dir}")
    
    # Collect results
    results = collect_results(
        results_dir, 
        include_phase1plus=not args.no_phase1plus,
        phase1plus_dir=phase1plus_dir,
        phase2d_dir=phase2d_dir
    )
    
    print(f"\nCollected {len(results)} runs:")
    for r in results:
        print(f"  {r['name']} (source={r['source']}, pool={r.get('pool')}, k={r.get('k')}, mode={r.get('mode')})")
    
    # Print analysis
    print_analysis_report(results)
    
    # Save JSON
    if args.output_json:
        save_summary_json(results, args.output_json)
    else:
        default_json = results_dir / 'phase2c_analysis.json'
        save_summary_json(results, default_json)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())