#!/usr/bin/env python
"""
Generate gate1_summary.csv from all Gate1 experiment results.

Collects metrics from all Gate1 runs and creates a summary table.
v2: Latest run selection per (track, method, n_train, seed) combination

Usage:
    python scripts/generate_gate1_summary.py
    python scripts/generate_gate1_summary.py --dataset_version cartpole_ood_v1 --track standardized
    python scripts/generate_gate1_summary.py --keep_all  # Keep all runs (no dedup)
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.contracts import paths


def parse_run_id_timestamp(run_id: str) -> datetime:
    """
    Extract timestamp from run_id (YYYYMMDD_HHMMSS_gitsha_note).
    Returns datetime for comparison, or datetime.min if parsing fails.
    """
    try:
        # run_id format: YYYYMMDD_HHMMSS_gitsha_note
        parts = run_id.split('_')
        if len(parts) >= 2:
            date_str = parts[0]  # YYYYMMDD
            time_str = parts[1]  # HHMMSS
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        pass
    return datetime.min


def collect_gate1_results(
    dataset_version: str = 'cartpole_ood_v1',
    track: str = None,  # None = all tracks
    keep_all: bool = False,  # If True, keep all runs (no dedup)
) -> list:
    """
    Collect all Gate1 experiment results.
    
    Args:
        dataset_version: Dataset version string
        track: Specific track or None for all
        keep_all: If False (default), keep only latest run per combination
    
    Returns:
        List of result dicts
    """
    all_results = []
    
    gate1_root = paths.RESULTS_ROOT / dataset_version / 'gate1'
    
    if not gate1_root.exists():
        print(f"  ⚠️ Gate1 results not found: {gate1_root}")
        return all_results
    
    # Iterate through tracks
    tracks = [track] if track else [d.name for d in gate1_root.iterdir() if d.is_dir()]
    
    for track_name in tracks:
        track_dir = gate1_root / track_name
        if not track_dir.exists():
            continue
        
        # Iterate through methods (esindy, etc.)
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
                        manifest_path = run_dir / 'manifest.json'
                        
                        if not metrics_path.exists():
                            continue
                        
                        # Load metrics
                        with open(metrics_path, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        
                        # Load manifest for additional info
                        manifest = {}
                        if manifest_path.exists():
                            with open(manifest_path, 'r', encoding='utf-8') as f:
                                manifest = json.load(f)
                        
                        run_id = metrics.get('run_id', run_dir.name)
                        
                        # Extract result
                        result = {
                            'dataset_version': dataset_version,
                            'track': track_name,
                            'method': method,
                            'n_train': n_train,
                            'seed': seed,
                            'n_bootstrap': metrics.get('config', {}).get('n_bootstrap', 20),
                            'best_threshold': metrics.get('config', {}).get('best_threshold', None),
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
                            'dx_source_key': manifest.get('config', {}).get('dx_source_key', None),
                            'run_id': run_id,
                            'results_path': str(run_dir),
                            '_timestamp': parse_run_id_timestamp(run_id),  # For sorting
                        }
                        
                        all_results.append(result)
    
    # Deduplication: keep only latest run per (track, method, n_train, seed)
    if not keep_all and all_results:
        print(f"  [Dedup] Before: {len(all_results)} runs")
        
        # Group by combination key
        groups = {}
        for r in all_results:
            key = (r['track'], r['method'], r['n_train'], r['seed'])
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        # Select latest from each group
        deduped = []
        for key, runs in groups.items():
            # Sort by timestamp descending, pick first (latest)
            runs_sorted = sorted(runs, key=lambda x: x['_timestamp'], reverse=True)
            deduped.append(runs_sorted[0])
        
        print(f"  [Dedup] After: {len(deduped)} runs (latest per combination)")
        all_results = deduped
    
    # Remove internal _timestamp field
    for r in all_results:
        r.pop('_timestamp', None)
    
    return all_results


def save_summary_csv(results: list, output_path: Path) -> None:
    """Save results to CSV."""
    if not results:
        print("  ⚠️ No results to save")
        return
    
    # Sort by track, n_train, seed
    results = sorted(results, key=lambda x: (x['track'], x['n_train'], x['seed']))
    
    # Define columns (added results_path)
    columns = [
        'dataset_version', 'track', 'method', 'n_train', 'seed', 'n_bootstrap',
        'best_threshold', 'train_r2', 'val_r2', 'test_r2',
        'train_rmse', 'val_rmse', 'test_rmse',
        'sparsity', 'active_terms', 'total_terms', 'mean_coef_std',
        'dx_source_key', 'run_id', 'results_path'
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        # Header
        f.write(','.join(columns) + '\n')
        
        # Data rows
        for r in results:
            row = []
            for col in columns:
                val = r.get(col, '')
                if val is None:  # Fixed: explicit None check
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
    
    print("\n" + "=" * 100)
    print("  Gate1 Summary")
    print("=" * 100)
    print(f"  {'Track':<20} {'N':>5} {'Seed':>5} {'Test R²':>10} {'Val R²':>10} {'Sparsity':>10} {'Threshold':>10}")
    print("-" * 100)
    
    for r in sorted(results, key=lambda x: (x['track'], x['n_train'], x['seed'])):
        track = r['track'][:20]
        n = r['n_train']
        seed = r['seed']
        # Fixed: explicit None check (0.0 is valid)
        test_r2 = f"{r['test_r2']:.4f}" if r['test_r2'] is not None else 'N/A'
        val_r2 = f"{r['val_r2']:.4f}" if r['val_r2'] is not None else 'N/A'
        sparsity = f"{r['sparsity']*100:.1f}%" if r['sparsity'] is not None else 'N/A'
        threshold = f"{r['best_threshold']:.4f}" if r['best_threshold'] is not None else 'N/A'
        
        print(f"  {track:<20} {n:>5} {seed:>5} {test_r2:>10} {val_r2:>10} {sparsity:>10} {threshold:>10}")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Generate Gate1 Summary CSV')
    parser.add_argument('--dataset_version', '-d', type=str, default='cartpole_ood_v1')
    parser.add_argument('--track', '-t', type=str, default=None, help='Specific track (default: all)')
    parser.add_argument('--output', '-o', type=Path, default=None)
    parser.add_argument('--keep_all', action='store_true', help='Keep all runs (no dedup)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  Gate1 Summary Generator (v2)")
    print("=" * 60)
    
    # Collect results
    print(f"\n[1/2] Collecting Gate1 results...")
    print(f"  Dataset: {args.dataset_version}")
    print(f"  Track: {args.track or 'all'}")
    print(f"  Mode: {'keep_all' if args.keep_all else 'latest_only (default)'}")
    
    results = collect_gate1_results(args.dataset_version, args.track, args.keep_all)
    print(f"  Final count: {len(results)} experiments")
    
    # Print summary
    print_summary_table(results)
    
    # Save CSV
    print(f"\n[2/2] Saving summary CSV...")
    output_path = args.output or (paths.RESULTS_ROOT / args.dataset_version / 'gate1' / 'gate1_summary.csv')
    save_summary_csv(results, output_path)
    
    print("\n" + "=" * 60)
    print("  ✅ Gate1 Summary Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()