import json
from pathlib import Path
import csv

base_dir = Path('results/cartpole_ood_v1/gate3/standardized/stable_core/n10/seed0')
output_dir = base_dir / 'diagnostics_d2'

# D2 runs 정의
runs = [
    {'fit_seed': 0, 'pool_seed': 42, 'gen_run_id': '20260122_185714_nogit_pool_only_dopt_seed42_v037', 'compare_run_id': '20260122_190758_nogit_pool_only_compare_seed42_v037'},
    {'fit_seed': 0, 'pool_seed': 777, 'gen_run_id': '20260122_232142_nogit_pool_only_dopt_seed777_v037', 'compare_run_id': '20260122_232216_nogit_pool_only_compare_seed777_v037'},
    {'fit_seed': 1, 'pool_seed': 42, 'gen_run_id': '20260127_190331_nogit_d2_fit1_pool42_dopt', 'compare_run_id': '20260127_191558_nogit_d2_compare_fit1_pool42'},
    {'fit_seed': 1, 'pool_seed': 777, 'gen_run_id': '20260127_191632_nogit_d2_fit1_pool777_dopt', 'compare_run_id': '20260127_192859_nogit_d2_compare_fit1_pool777'},
    {'fit_seed': 2, 'pool_seed': 42, 'gen_run_id': '20260127_192928_nogit_d2_fit2_pool42_dopt', 'compare_run_id': '20260127_194125_nogit_d2_compare_fit2_pool42'},
    {'fit_seed': 2, 'pool_seed': 777, 'gen_run_id': '20260127_194149_nogit_d2_fit2_pool777_dopt', 'compare_run_id': '20260127_195414_nogit_d2_compare_fit2_pool777'},
    {'fit_seed': 42, 'pool_seed': 42, 'gen_run_id': '20260127_195443_nogit_d2_fit42_pool42_dopt', 'compare_run_id': '20260127_200730_nogit_d2_compare_fit42_pool42'},
    {'fit_seed': 42, 'pool_seed': 777, 'gen_run_id': '20260127_200755_nogit_d2_fit42_pool777_dopt', 'compare_run_id': '20260127_202003_nogit_d2_compare_fit42_pool777'},
    {'fit_seed': 123, 'pool_seed': 42, 'gen_run_id': '20260127_202031_nogit_d2_fit123_pool42_dopt', 'compare_run_id': '20260127_203401_nogit_d2_compare_fit123_pool42'},
    {'fit_seed': 123, 'pool_seed': 777, 'gen_run_id': '20260127_203430_nogit_d2_fit123_pool777_dopt', 'compare_run_id': '20260127_204635_nogit_d2_compare_fit123_pool777'},
]

def get_label_mode(gmm_params):
    if not gmm_params:
        return 'N/A'
    weights = gmm_params.get('weights', [])
    if len(weights) >= 2:
        if weights[0] < weights[1]:
            return 'Mode1_AB'
        else:
            return 'Mode2_BA'
    return 'N/A'

results = []
for run in runs:
    gen_dir = base_dir / run['gen_run_id']
    compare_dir = base_dir / run['compare_run_id']
    
    with open(gen_dir / 'manifest.json', 'r') as f:
        manifest = json.load(f)
    
    gmm_params_path = gen_dir / 'gmm_params.json'
    gmm_params = {}
    if gmm_params_path.exists():
        with open(gmm_params_path, 'r') as f:
            gmm_params = json.load(f)
    
    with open(compare_dir / 'comparison_gen.json', 'r') as f:
        comparison = json.load(f)
    
    gmm_config = manifest.get('gmm_config', {})
    comp_results = comparison.get('results', [{}])
    gen_result = comp_results[0] if comp_results else {}
    
    gmm_fit_sha = gmm_config.get('gmm_fit_sha256') or gmm_params.get('gmm_fit_sha256') or 'N/A'
    if gmm_fit_sha != 'N/A':
        gmm_fit_sha = gmm_fit_sha[:16]
    
    effective_K = gmm_params.get('n_components') or gmm_config.get('n_components') or 'N/A'
    
    per_pair = gen_result.get('per_pair_aug_pure', [])
    max_abs = max(abs(p[2]) for p in per_pair) if per_pair else 'N/A'
    
    label_mode = get_label_mode(gmm_params)
    if run['fit_seed'] == 0:
        label_mode = 'Mode2_BA'
    
    row = {
        'fit_seed': run['fit_seed'],
        'pool_seed': run['pool_seed'],
        'label_mode': label_mode,
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
        'gen_run_id': run['gen_run_id'],
        'compare_run_id': run['compare_run_id'],
    }
    results.append(row)
    
    median_str = f"{row['median_aug_pure']:.4f}" if isinstance(row['median_aug_pure'], float) else str(row['median_aug_pure'])
    print(f"fit={run['fit_seed']:3d}, pool={run['pool_seed']:3d}, mode={label_mode}: {row['pass_level']}, median={median_str}")

summary_path = output_dir / 'summary_d2_crossfit_v2.csv'
fieldnames = ['fit_seed', 'pool_seed', 'label_mode', 'gmm_fit_sha256', 'effective_K', 'bootstrap_seed', 'pool_sha256', 'median_aug_pure', 'ci95_lower', 'ci95_upper', 'pass_level', 'mean_aug_pure', 'std_aug_pure', 'max_abs_aug_pure', 'gen_run_id', 'compare_run_id']
with open(summary_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
print(f'\nSaved: {summary_path}')

print('\n' + '='*70)
print('D2 Two-Mode Summary (Label Switching Analysis)')
print('='*70)
mode1 = [r for r in results if r['label_mode'] == 'Mode1_AB']
mode2 = [r for r in results if r['label_mode'] == 'Mode2_BA']
print(f'Mode1_AB (weights=[0.4,0.6]): {len(mode1)} runs, fit_seed={{1,2,42}}')
print(f'Mode2_BA (weights=[0.6,0.4]): {len(mode2)} runs, fit_seed={{0,123}}')
print(f'\nAll SOFT_PASS: {sum(1 for r in results if r["pass_level"] == "SOFT_PASS")}/10')
