#!/usr/bin/env python
"""
Generate gate2_summary.csv from all Gate2 experiment results.

Collects metrics from all Gate2 runs and creates a summary table.
Includes Gate1 baseline comparison delta when available.

Features (v3.1):
    - mtime-based timestamp fallback when run_id parsing fails
    - Latest run selection per combination
    - keep_all option for full history
    - aug_seed column for ablation tracking
    - augmentation_stats tracking (success_rate, fallback)
    - --backfill option: compute missing delta/success from existing files
    - Enhanced backfill: track/seed matching for baseline, samples array parsing

Usage:
    python scripts/generate_gate2_summary.py
    python scripts/generate_gate2_summary.py --dataset_version cartpole_ood_v1 --track standardized
    python scripts/generate_gate2_summary.py --keep_all
    python scripts/generate_gate2_summary.py --backfill  # Fill missing delta/success
"""
import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.contracts import paths


def parse_run_id_timestamp(run_id: str, fallback_mtime: float = None) -> datetime:
    """Extract timestamp from run_id."""
    try:
        parts = run_id.split('_')
        if len(parts) >= 2:
            date_str = parts[0]
            time_str = parts[1]
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        pass
    
    if fallback_mtime is not None:
        return datetime.fromtimestamp(fallback_mtime)
    
    return datetime.min


def load_gate1_summary(dataset_version: str) -> Tuple[Dict[str, Dict], Dict[Tuple, Dict]]:
    """
    Load Gate1 summary CSV.
    
    Returns:
        Tuple of:
        - by_run_id: {run_id: {test_r2, val_r2, sparsity, ...}}
        - by_key: {(track, n_train, seed): {test_r2, val_r2, ...}} - latest run per key
    """
    gate1_summary_path = paths.RESULTS_ROOT / dataset_version / 'gate1' / 'gate1_summary.csv'
    
    by_run_id = {}
    by_key = {}
    
    if not gate1_summary_path.exists():
        return by_run_id, by_key
    
    with open(gate1_summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return by_run_id, by_key
        
        headers = lines[0].strip().split(',')
        for line in lines[1:]:
            values = line.strip().split(',')
            if len(values) != len(headers):
                continue
            
            row = dict(zip(headers, values))
            run_id = row.get('run_id', '')
            
            # Parse numeric values
            for key in ['test_r2', 'val_r2', 'train_r2', 'sparsity']:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
            
            # Parse n_train and seed
            try:
                n_train = int(row.get('n_train', 0))
                seed = int(row.get('seed', 0))
            except ValueError:
                continue
            
            track = row.get('track', '')
            
            if run_id:
                by_run_id[run_id] = row
            
            # Store by key (track, n_train, seed) - keep latest
            key = (track, n_train, seed)
            if key not in by_key:
                by_key[key] = row
            else:
                # Compare timestamps, keep newer
                existing_ts = parse_run_id_timestamp(by_key[key].get('run_id', ''))
                new_ts = parse_run_id_timestamp(run_id)
                if new_ts > existing_ts:
                    by_key[key] = row
    
    return by_run_id, by_key


def compute_augmentation_stats_from_manifest(aug_manifest: Dict) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Compute success_rate, n_fallback, n_target from aug_manifest.
    Handles both new format (augmentation_stats) and old format (samples array).
    
    Returns: (success_rate, n_fallback, n_target)
    """
    # Try augmentation_stats first (new format)
    aug_stats = aug_manifest.get('augmentation_stats', {})
    if aug_stats and 'success_rate' in aug_stats:
        return (
            aug_stats.get('success_rate'),
            aug_stats.get('n_fallback'),
            aug_stats.get('n_target'),
        )
    
    # Try type_counts (intermediate format)
    type_counts = aug_manifest.get('type_counts', {})
    if type_counts:
        n_fallback = type_counts.get('original_fallback', 0)
        n_target = aug_manifest.get('n_augmented', sum(type_counts.values()))
        
        if n_target and n_target > 0:
            success_rate = 1.0 - (n_fallback / n_target)
            return success_rate, n_fallback, n_target
    
    # Try samples array (old format)
    samples = aug_manifest.get('samples', [])
    if samples:
        n_target = len(samples)
        n_fallback = sum(1 for s in samples if s.get('type') == 'original_fallback')
        
        if n_target > 0:
            success_rate = 1.0 - (n_fallback / n_target)
            return success_rate, n_fallback, n_target
    
    # Fallback: use n_augmented if available
    n_augmented = aug_manifest.get('n_augmented')
    if n_augmented is not None:
        # Assume all successful if no failure info
        return 1.0, 0, n_augmented
    
    return None, None, None


def backfill_result(
    result: Dict, 
    gate1_by_run_id: Dict[str, Dict],
    gate1_by_key: Dict[Tuple, Dict],
    run_dir: Path
) -> Dict:
    """
    Fill missing delta and success_rate for a result.
    """
    # === Backfill delta ===
    if result.get('test_r2_delta') is None:
        baseline_run_id = result.get('gate1_baseline_run_id')
        baseline = None
        
        # Method 1: Try to get baseline_run_id from manifest.json
        if not baseline_run_id:
            manifest_path = run_dir / 'manifest.json'
            if manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    baseline_run_id = manifest.get('gate1_baseline_run_id')
                    if baseline_run_id:
                        result['gate1_baseline_run_id'] = baseline_run_id
        
        # Method 2: Lookup by run_id
        if baseline_run_id and baseline_run_id in gate1_by_run_id:
            baseline = gate1_by_run_id[baseline_run_id]
        
        # Method 3: Fallback to track/n_train/seed matching
        if baseline is None:
            key = (result.get('track'), result.get('n_train'), result.get('seed'))
            if key in gate1_by_key:
                baseline = gate1_by_key[key]
                # Record the matched baseline
                result['gate1_baseline_run_id'] = baseline.get('run_id', 'matched_by_key')
        
        # Compute deltas
        if baseline is not None:
            baseline_test_r2 = baseline.get('test_r2')
            baseline_val_r2 = baseline.get('val_r2')
            baseline_sparsity = baseline.get('sparsity')
            
            if baseline_test_r2 is not None and result.get('test_r2') is not None:
                result['test_r2_delta'] = result['test_r2'] - baseline_test_r2
            
            if baseline_val_r2 is not None and result.get('val_r2') is not None:
                result['val_r2_delta'] = result['val_r2'] - baseline_val_r2
            
            if baseline_sparsity is not None and result.get('sparsity') is not None:
                result['sparsity_delta'] = result['sparsity'] - baseline_sparsity
    
    # === Backfill success_rate ===
    if result.get('aug_success_rate') is None:
        aug_manifest_path = run_dir / 'aug_manifest.json'
        if aug_manifest_path.exists():
            with open(aug_manifest_path, 'r', encoding='utf-8') as f:
                aug_manifest = json.load(f)
            
            success_rate, n_fallback, n_target = compute_augmentation_stats_from_manifest(aug_manifest)
            if success_rate is not None:
                result['aug_success_rate'] = success_rate
                result['aug_fallback_count'] = n_fallback
                result['aug_n_target'] = n_target
    
    return result


def collect_gate2_results(
    dataset_version: str = 'cartpole_ood_v1',
    track: str = None,
    keep_all: bool = False,
    backfill: bool = False,
) -> list:
    """Collect all Gate2 experiment results."""
    all_results = []
    
    gate2_root = paths.RESULTS_ROOT / dataset_version / 'gate2'
    
    if not gate2_root.exists():
        print(f"  Warning: Gate2 results not found: {gate2_root}")
        return all_results
    
    # Load Gate1 summary for backfill
    gate1_by_run_id = {}
    gate1_by_key = {}
    if backfill:
        print("  [Backfill] Loading Gate1 summary...")
        gate1_by_run_id, gate1_by_key = load_gate1_summary(dataset_version)
        print(f"  [Backfill] Found {len(gate1_by_run_id)} Gate1 runs by run_id")
        print(f"  [Backfill] Found {len(gate1_by_key)} Gate1 runs by (track, n, seed)")
    
    # Iterate through tracks
    tracks_to_scan = [track] if track else [d.name for d in gate2_root.iterdir() if d.is_dir()]
    
    for track_name in tracks_to_scan:
        track_dir = gate2_root / track_name
        if not track_dir.exists():
            continue
        
        # Iterate through methods
        for method_dir in track_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            
            # Iterate through n_train
            for n_dir in method_dir.iterdir():
                if not n_dir.is_dir() or not n_dir.name.startswith('n'):
                    continue
                n_train = int(n_dir.name[1:])
                
                # Iterate through seeds
                for seed_dir in n_dir.iterdir():
                    if not seed_dir.is_dir() or not seed_dir.name.startswith('seed'):
                        continue
                    seed = int(seed_dir.name[4:])
                    
                    # Iterate through run_ids
                    for run_dir in seed_dir.iterdir():
                        if not run_dir.is_dir():
                            continue
                        
                        metrics_path = run_dir / 'metrics.json'
                        aug_manifest_path = run_dir / 'aug_manifest.json'
                        
                        if not metrics_path.exists():
                            continue
                        
                        mtime = metrics_path.stat().st_mtime
                        
                        with open(metrics_path, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        
                        aug_manifest = {}
                        if aug_manifest_path.exists():
                            with open(aug_manifest_path, 'r', encoding='utf-8') as f:
                                aug_manifest = json.load(f)
                        
                        run_id = metrics.get('run_id', run_dir.name)
                        config = metrics.get('config', {})
                        
                        aug_stats = metrics.get('augmentation_stats', 
                                               aug_manifest.get('augmentation_stats', {}))
                        
                        result = {
                            'dataset_version': dataset_version,
                            'track': track_name,
                            'method': method,
                            'n_train': n_train,
                            'seed': seed,
                            'aug_seed': config.get('aug_seed', aug_manifest.get('aug_seed')),
                            'aug_method': config.get('aug_method', method),
                            'aug_ratio': config.get('aug_ratio', aug_manifest.get('aug_ratio', 1.0)),
                            'jitter_mode': config.get('jitter_mode', aug_manifest.get('jitter_mode', 'both')),
                            'n_train_original': config.get('n_train_original', n_train),
                            'n_train_augmented': config.get('n_train_augmented', 0),
                            'n_train_total': config.get('n_train_total', n_train),
                            'n_bootstrap': config.get('n_bootstrap', 20),
                            'best_threshold': config.get('best_threshold', None),
                            'train_r2': metrics.get('splits', {}).get('train', {}).get('r2_mean', None),
                            'val_r2': metrics.get('splits', {}).get('val', {}).get('r2_mean', None),
                            'test_r2': metrics.get('splits', {}).get('test', {}).get('r2_mean', None),
                            'train_rmse': metrics.get('splits', {}).get('train', {}).get('rmse_mean', None),
                            'val_rmse': metrics.get('splits', {}).get('val', {}).get('rmse_mean', None),
                            'test_rmse': metrics.get('splits', {}).get('test', {}).get('rmse_mean', None),
                            'sparsity': metrics.get('sparsity', {}).get('sparsity', None),
                            'active_terms': metrics.get('sparsity', {}).get('n_active', None),
                            'total_terms': metrics.get('sparsity', {}).get('n_total', None),
                            'mean_coef_std': metrics.get('sparsity', {}).get('mean_coef_std', None),
                            'aug_success_rate': aug_stats.get('success_rate', None),
                            'aug_fallback_count': aug_stats.get('n_fallback', None),
                            'aug_n_target': aug_stats.get('n_target', None),
                            'dx_policy': aug_manifest.get('dx_policy', None),
                            'run_id': run_id,
                            'results_path': str(run_dir),
                            '_timestamp': parse_run_id_timestamp(run_id, mtime),
                            '_run_dir': run_dir,
                        }
                        
                        gate1_delta = metrics.get('gate1_delta', {})
                        result['test_r2_delta'] = gate1_delta.get('test_r2_delta', None)
                        result['val_r2_delta'] = gate1_delta.get('val_r2_delta', None)
                        result['sparsity_delta'] = gate1_delta.get('sparsity_delta', None)
                        result['gate1_baseline_run_id'] = gate1_delta.get('baseline_run_id', None)
                        
                        # Backfill missing values
                        if backfill:
                            result = backfill_result(result, gate1_by_run_id, gate1_by_key, run_dir)
                        
                        all_results.append(result)
    
    # Deduplication
    if not keep_all and all_results:
        print(f"  [Dedup] Before: {len(all_results)} runs")
        
        groups = {}
        for r in all_results:
            key = (
                r['track'], 
                r['method'], 
                r['n_train'], 
                r['seed'],
                r.get('aug_seed'),
                r.get('jitter_mode'),
                r.get('aug_ratio'),
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        deduped = []
        for key, runs in groups.items():
            runs_sorted = sorted(runs, key=lambda x: x['_timestamp'], reverse=True)
            deduped.append(runs_sorted[0])
        
        print(f"  [Dedup] After: {len(deduped)} runs (latest per combination)")
        all_results = deduped
    
    # Remove internal fields
    for r in all_results:
        r.pop('_timestamp', None)
        r.pop('_run_dir', None)
    
    return all_results


def save_summary_csv(results: list, output_path: Path) -> None:
    """Save results to CSV."""
    if not results:
        print("  Warning: No results to save")
        return
    
    results = sorted(results, key=lambda x: (
        x['track'], 
        x['n_train'], 
        x['seed'],
        x.get('aug_seed') if x.get('aug_seed') is not None else 999,
        x.get('jitter_mode', ''),
        x.get('aug_ratio', 0),
    ))
    
    columns = [
        'dataset_version', 'track', 'method', 'n_train', 'seed',
        'aug_seed', 'aug_method', 'aug_ratio', 'jitter_mode',
        'n_train_original', 'n_train_augmented', 'n_train_total',
        'n_bootstrap', 'best_threshold',
        'train_r2', 'val_r2', 'test_r2',
        'train_rmse', 'val_rmse', 'test_rmse',
        'sparsity', 'active_terms', 'total_terms', 'mean_coef_std',
        'aug_success_rate', 'aug_fallback_count', 'aug_n_target',
        'test_r2_delta', 'val_r2_delta', 'sparsity_delta',
        'dx_policy', 'gate1_baseline_run_id', 'run_id', 'results_path'
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(columns) + '\n')
        
        for r in results:
            row = []
            for col in columns:
                val = r.get(col, '')
                if val is None:
                    row.append('')
                elif isinstance(val, float):
                    row.append(f'{val:.6f}')
                else:
                    row.append(str(val))
            f.write(','.join(row) + '\n')
    
    print(f"  ✅ Saved: {output_path}")


def print_summary_table(results: list) -> None:
    """Print summary table to console."""
    if not results:
        return
    
    n_delta_filled = sum(1 for r in results if r.get('test_r2_delta') is not None)
    n_success_filled = sum(1 for r in results if r.get('aug_success_rate') is not None)
    
    sorted_results = sorted(results, key=lambda x: (
        x['track'], 
        x['n_train'], 
        x['seed'],
        x.get('aug_seed') if x.get('aug_seed') is not None else 999,
        x.get('jitter_mode', ''),
        x.get('aug_ratio', 0),
    ))
    
    print("\n" + "=" * 140)
    print("  Gate2 Summary")
    print("=" * 140)
    print(f"  Delta filled: {n_delta_filled}/{len(results)} | Success filled: {n_success_filled}/{len(results)}")
    print("-" * 140)
    print(f"  {'Track':<18} {'N':>3} {'Seed':>4} {'ASeed':>5} {'Jitter':<12} {'Ratio':>5} "
          f"{'Test R2':>8} {'Delta':>8} {'Sparsity':>8} {'Success':>7}")
    print("-" * 140)
    
    for r in sorted_results:
        track = r['track'][:18]
        n = r['n_train']
        seed = r['seed']
        aug_seed = r.get('aug_seed')
        aug_seed_str = str(aug_seed) if aug_seed is not None else 'N/A'
        jitter = r.get('jitter_mode', 'N/A')[:12]
        aug_ratio = r.get('aug_ratio', 1.0)
        test_r2 = f"{r['test_r2']:.4f}" if r['test_r2'] is not None else 'N/A'
        delta = f"{r['test_r2_delta']:+.4f}" if r.get('test_r2_delta') is not None else 'N/A'
        sparsity = f"{r['sparsity']*100:.1f}%" if r.get('sparsity') is not None else 'N/A'
        success = f"{r['aug_success_rate']*100:.0f}%" if r.get('aug_success_rate') is not None else 'N/A'
        
        print(f"  {track:<18} {n:>3} {seed:>4} {aug_seed_str:>5} {jitter:<12} {aug_ratio:>5.1f} "
              f"{test_r2:>8} {delta:>8} {sparsity:>8} {success:>7}")
    
    print("=" * 140)


def main():
    parser = argparse.ArgumentParser(description='Generate Gate2 Summary CSV')
    parser.add_argument('--dataset_version', '-d', type=str, default='cartpole_ood_v1')
    parser.add_argument('--track', '-t', type=str, default=None, help='Specific track (default: all)')
    parser.add_argument('--output', '-o', type=Path, default=None)
    parser.add_argument('--keep_all', action='store_true', help='Keep all runs (no dedup)')
    parser.add_argument('--backfill', action='store_true', 
                        help='Compute missing delta/success from existing files')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  Gate2 Summary Generator (v3.1)")
    print("=" * 60)
    
    print(f"\n[1/2] Collecting Gate2 results...")
    print(f"  Dataset: {args.dataset_version}")
    print(f"  Track: {args.track or 'all'}")
    print(f"  Mode: {'keep_all' if args.keep_all else 'latest_only (default)'}")
    print(f"  Backfill: {'enabled' if args.backfill else 'disabled'}")
    
    results = collect_gate2_results(
        args.dataset_version, 
        args.track, 
        args.keep_all,
        args.backfill,
    )
    print(f"  Final count: {len(results)} experiments")
    
    print_summary_table(results)
    
    print(f"\n[2/2] Saving summary CSV...")
    output_path = args.output or (paths.RESULTS_ROOT / args.dataset_version / 'gate2' / 'gate2_summary.csv')
    save_summary_csv(results, output_path)
    
    print("\n" + "=" * 60)
    print("  ✅ Gate2 Summary Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()