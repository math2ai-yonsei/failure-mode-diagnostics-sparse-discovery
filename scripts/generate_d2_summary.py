import json
from pathlib import Path
import csv

base_dir = Path('results/cartpole_ood_v1/gate3/standardized/stable_core/n10/seed0')
output_dir = base_dir / 'diagnostics_d2'

# D2 runs 정의 (D1 reference 2개 + Cross-fit 8개 = 10개)
runs = [
    # D1 reference (fit_seed=0)
    {'fit_seed': 0, 'pool_seed': 42, 'gen_run_id': '20260122_185714_nogit_pool_only_dopt_seed42_v037', 'compare_run_id': '20260122_190758_nogit_pool_only_compare_seed42_v037'},
    {'fit_seed': 0, 'pool_seed': 777, 'gen_run_id': '20260122_232142_nogit_pool_only_dopt_seed777_v037', 'compare_run_id': '20260122_232216_nogit_pool_only_compare_seed777_v037'},
    # D2 cross-fit (fit_seed != 0)
    {'fit_seed': 1, 'pool_seed': 42, 'gen_run_id': '20260127_190331_nogit_d2_fit1_pool42_dopt', 'compare_run_id': '20260127_191558_nogit_d2_compare_fit1_pool42'},
    {'fit_seed': 1, 'pool_seed': 777, 'gen_run_id': '20260127_191632_nogit_d2_fit1_pool777_dopt', 'compare_run_id': '20260127_192859_nogit_d2_compare_fit1_pool777'},
    {'fit_seed': 2, 'pool_seed': 42, 'gen_run_id': '20260127_192928_nogit_d2_fit2_pool42_dopt', 'compare_run_id': '20260127_194125_nogit_d2_compare_fit2_pool42'},
    {'fit_seed': 2, 'pool_seed': 777, 'gen_run_id': '20260127_194149_nogit_d2_fit2_pool777_dopt', 'compare_run_id': '20260127_195414_nogit_d2_compare_fit2_pool777'},
    {'fit_seed': 42, 'pool_seed': 42, 'gen_run_id': '20260127_195443_nogit_d2_fit42_pool42_dopt', 'compare_run_id': '20260127_200730_nogit_d2_compare_fit42_pool42'},
    {'fit_seed': 42, 'pool_seed': 777, 'gen_run_id': '20260127_200755_nogit_d2_fit42_pool777_dopt', 'compare_run_id': '20260127_202003_nogit_d2_compare_fit42_pool777'},
    {'fit_seed': 123, 'pool_seed': 42, 'gen_run_id': '20260127_202031_nogit_d2_fit123_pool42_dopt', 'compare_run_id': '20260127_203401_nogit_d2_compare_fit123_pool42'},
    {'fit_seed': 123, 'pool_seed': 777, 'gen_run_id': '20260127_203430_nogit_d2_fit123_pool777_dopt', 'compare_run_id': '20260127_204635_nogit_d2_compare_fit123_pool777'},
]

results = []
for run in runs:
    gen_dir = base_dir / run['gen_run_id']
    compare_dir = base_dir / run['compare_run_id']
    
    # Load manifest
    manifest_path = gen_dir / 'manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Load gmm_params (may not exist for D1 reference)
    gmm_params_path = gen_dir / 'gmm_params.json'
    gmm_params = {}
    if gmm_params_path.exists():
        with open(gmm_params_path, 'r') as f:
            gmm_params = json.load(f)
    
    # Load comparison_gen
    compare_path = compare_dir / 'comparison_gen.json'
    with open(compare_path, 'r') as f:
        comparison = json.load(f)
    
    # Extract data - comparison uses 'results' not 'gen_results'
    gmm_config = manifest.get('gmm_config', {})
    comp_results = comparison.get('results', [{}])
    gen_result = comp_results[0] if comp_results else {}
    
    # Get gmm_fit_sha256 from multiple sources
    gmm_fit_sha = gmm_config.get('gmm_fit_sha256') or gmm_params.get('gmm_fit_sha256') or manifest.get('pool_source_gmm_fit_sha256') or 'N/A'
    if gmm_fit_sha != 'N/A':
        gmm_fit_sha = gmm_fit_sha[:16]
    
    # Get effective_K
    effective_K = gmm_params.get('n_components') or gmm_config.get('n_components') or 'N/A'
    
    # Get per_pair_aug_pure for max_abs calculation
    per_pair = gen_result.get('per_pair_aug_pure', [])
    if per_pair:
        max_abs = max(abs(p[2]) for p in per_pair)
    else:
        max_abs = 'N/A'
    
    row = {
        'fit_seed': run['fit_seed'],
        'pool_seed': run['pool_seed'],
        'gen_run_id': run['gen_run_id'],
        'compare_run_id': run['compare_run_id'],
        'gmm_fit_sha256': gmm_fit_sha,
        'effective_K': effective_K,
        'bootstrap_seed': manifest.get('bootstrap_seed', 'N/A'),
        'pool_sha256': manifest.get('pool_sha256', 'N/A')[:16] if manifest.get('pool_sha256') else 'N/A',
        'median_aug_pure': gen_result.get('median_aug_pure', 'N/A'),
        'ci95_lower': gen_result.get('ci_lower', 'N/A'),
        'ci95_upper': gen_result.get('ci_upper', 'N/A'),
        'pass_level': gen_result.get('pass_level', 'N/A'),
        'mean_aug_pure': gen_result.get('mean_aug_pure', 'N/A'),
        'std_aug_pure': gen_result.get('std_aug_pure', 'N/A'),
        'max_abs_aug_pure': max_abs,
    }
    results.append(row)
    print(f"fit={run['fit_seed']:3d}, pool={run['pool_seed']:3d}: {row['pass_level']}, median={row['median_aug_pure']}")

# Write runlog_d2.tsv
runlog_path = output_dir / 'runlog_d2.tsv'
fieldnames_runlog = ['fit_seed', 'pool_seed', 'gen_run_id', 'compare_run_id', 'gmm_fit_sha256', 'effective_K', 'bootstrap_seed', 'pool_sha256']
with open(runlog_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames_runlog, delimiter='\t')
    writer.writeheader()
    for row in results:
        writer.writerow({k: row[k] for k in fieldnames_runlog})
print(f'\nSaved: {runlog_path}')

# Write summary_d2_crossfit.csv
summary_path = output_dir / 'summary_d2_crossfit.csv'
fieldnames_summary = ['fit_seed', 'pool_seed', 'gmm_fit_sha256', 'effective_K', 'bootstrap_seed', 'pool_sha256', 'median_aug_pure', 'ci95_lower', 'ci95_upper', 'pass_level', 'mean_aug_pure', 'std_aug_pure', 'max_abs_aug_pure', 'gen_run_id', 'compare_run_id']
with open(summary_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames_summary)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
print(f'Saved: {summary_path}')

# Print summary
print('\n' + '='*60)
print('D2 Cross-fit Summary')
print('='*60)
soft_pass_count = sum(1 for r in results if r['pass_level'] == 'SOFT_PASS')
null_count = sum(1 for r in results if r['pass_level'] == 'NULL')
print(f'SOFT_PASS: {soft_pass_count}/10')
print(f'NULL: {null_count}/10')
print(f'D1 ref (fit_seed=0) SOFT_PASS: {sum(1 for r in results if r["fit_seed"] == 0 and r["pass_level"] == "SOFT_PASS")}/2')
print(f'Cross-fit (fit_seed!=0) SOFT_PASS: {sum(1 for r in results if r["fit_seed"] != 0 and r["pass_level"] == "SOFT_PASS")}/8')
