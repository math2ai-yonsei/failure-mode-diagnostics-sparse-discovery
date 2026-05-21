#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check filtering stats from M1-M5 aug_manifest.json files"""

import json
from pathlib import Path

base_dir = Path('results/cartpole_ood_v1/gate3/standardized/generative/n10/seed0')

runs = [
    ('M1', '20251231_032608_nogit_m1_vae_align'),
    ('M2', '20251231_032718_nogit_m2_gen_only'),
    ('M3', '20251231_032747_nogit_m3_copy_only'),
    ('M4', '20251231_032810_nogit_m4_noise_aug'),
    ('M5', '20251231_032833_nogit_m5_random_select'),
]

print("=" * 80)
print("M1-M5 Filtering Statistics")
print("=" * 80)

for name, run_id in runs:
    manifest_path = base_dir / run_id / 'aug_manifest.json'
    
    if not manifest_path.exists():
        print(f"\n{name}: aug_manifest.json not found")
        continue
    
    with open(manifest_path, encoding='utf-8') as f:
        m = json.load(f)
    
    f_stats = m.get('filtering', {})
    gen_stats = m.get('generation', {})
    
    print(f"\n{'=' * 40}")
    print(f"{name} ({run_id[:20]}...)")
    print(f"{'=' * 40}")
    print(f"  status: {m.get('status')}")
    print(f"  n_generated: {gen_stats.get('n_generated', f_stats.get('n_input', 'N/A'))}")
    print(f"  n_after_sanity: {f_stats.get('n_after_sanity', 'N/A')}")
    print(f"  n_after_dedup: {f_stats.get('n_after_dedup', 'N/A')}")
    print(f"  n_selected: {f_stats.get('n_selected', gen_stats.get('n_selected', 'N/A'))}")
    print(f"  reject_rate: {f_stats.get('reject_rate', 'N/A')}")
    print(f"  reject_reasons: {f_stats.get('reject_reasons', 'N/A')}")
    
    align_stats = f_stats.get('align_score_stats', {})
    if align_stats:
        print(f"  align_score_stats:")
        print(f"    mean: {align_stats.get('mean', 'N/A')}")
        print(f"    std: {align_stats.get('std', 'N/A')}")
        print(f"    min: {align_stats.get('min', 'N/A')}")
        print(f"    max: {align_stats.get('max', 'N/A')}")
    else:
        print(f"  align_score_stats: N/A")

print("\n" + "=" * 80)
print("Summary Table")
print("=" * 80)
print(f"{'Run':<5} {'n_gen':<8} {'n_sanity':<10} {'n_dedup':<10} {'n_sel':<8} {'reject%':<10}")
print("-" * 60)

for name, run_id in runs:
    manifest_path = base_dir / run_id / 'aug_manifest.json'
    
    if not manifest_path.exists():
        print(f"{name:<5} {'N/A':<8}")
        continue
    
    with open(manifest_path, encoding='utf-8') as f:
        m = json.load(f)
    
    f_stats = m.get('filtering', {})
    gen_stats = m.get('generation', {})
    
    n_gen = gen_stats.get('n_generated', f_stats.get('n_input', 'N/A'))
    n_sanity = f_stats.get('n_after_sanity', 'N/A')
    n_dedup = f_stats.get('n_after_dedup', 'N/A')
    n_sel = f_stats.get('n_selected', gen_stats.get('n_selected', 'N/A'))
    reject = f_stats.get('reject_rate', 'N/A')
    
    if isinstance(reject, float):
        reject = f"{reject:.1%}"
    
    print(f"{name:<5} {str(n_gen):<8} {str(n_sanity):<10} {str(n_dedup):<10} {str(n_sel):<8} {str(reject):<10}")
