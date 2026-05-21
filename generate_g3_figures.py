#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Phase1-mini Representative Figures
=========================================
F01: M1-M5 Test R² Comparison
F02: Filtering Pipeline Flow
F03: Align Score Distribution (Optional)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# SSOT: plot_style 적용
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Output directory
output_dir = Path('results/cartpole_ood_v1/gate3/standardized/generative/n10/seed0/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Data
results = {
    'M1\n(VAE+Align)': {'test_r2': 0.9228, 'color': '#2ecc71', 'hatch': ''},
    'M2\n(gen_only)': {'test_r2': 0.7901, 'color': '#e74c3c', 'hatch': ''},
    'M3\n(copy)': {'test_r2': 0.9605, 'color': '#3498db', 'hatch': '//'},
    'M4\n(noise)': {'test_r2': 0.4866, 'color': '#e74c3c', 'hatch': ''},
    'M5\n(random)': {'test_r2': 0.7841, 'color': '#e74c3c', 'hatch': ''},
    'Gate1\n(baseline)': {'test_r2': 0.8243, 'color': '#95a5a6', 'hatch': '..'},
}

filtering_stats = {
    'M1': {'n_gen': 100, 'n_sanity': 91, 'n_dedup': 91, 'n_selected': 10, 
           'reject_align': 81, 'reject_random': 0},
    'M2': {'n_gen': 100, 'n_sanity': 87, 'n_dedup': 86, 'n_selected': 10,
           'reject_align': 0, 'reject_random': 76},
    'M5': {'n_gen': 100, 'n_sanity': 99, 'n_dedup': 99, 'n_selected': 10,
           'reject_align': 0, 'reject_random': 89},
}

# ============================================================
# F01: Test R² Comparison Bar Chart
# ============================================================
def create_f01():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    values = [results[n]['test_r2'] for n in names]
    colors = [results[n]['color'] for n in names]
    hatches = [results[n]['hatch'] for n in names]
    
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add hatches
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Gate1 baseline line
    ax.axhline(y=0.8243, color='#95a5a6', linestyle='--', linewidth=2, label='Gate1 Baseline')
    
    # Annotations
    ax.annotate('', xy=(0, 0.9228), xytext=(0, 0.8243),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(0.35, 0.87, '+0.0986', fontsize=10, color='green', fontweight='bold')
    
    ax.set_ylabel('Test R²')
    ax.set_title('Gate3 Phase1-mini: M1-M5 Test R² Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='M1 (Proposed: VAE+Align)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='VAE Baselines (M2/M4/M5)'),
        Patch(facecolor='#3498db', edgecolor='black', hatch='//', label='M3 (Copy-only, Upper Bound)'),
        Patch(facecolor='#95a5a6', edgecolor='black', hatch='..', label='Gate1 (No Augmentation)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / 'F01_test_r2_comparison.png')
    fig.savefig(output_dir / 'F01_test_r2_comparison.pdf')
    print(f"✅ Saved: F01_test_r2_comparison.png + .pdf")
    plt.close(fig)


# ============================================================
# F02: Filtering Pipeline Flow
# ============================================================
def create_f02():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, (name, stats) in enumerate(filtering_stats.items()):
        ax = axes[idx]
        
        stages = ['Generated\n(100)', 'Sanity\nPass', 'Dedup\nPass', 'Selected\n(10)']
        values = [stats['n_gen'], stats['n_sanity'], stats['n_dedup'], stats['n_selected']]
        
        # Colors: green for flow, red for rejection
        colors = ['#3498db', '#2ecc71', '#2ecc71', '#27ae60']
        
        bars = ax.bar(stages, values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add rejection arrows
        if name == 'M1':
            ax.annotate('Align\nReject: 81', xy=(2.5, 50), fontsize=10, 
                        color='#e74c3c', ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor='#e74c3c'))
            selection_method = 'Align Top-10'
        else:
            ax.annotate('Random\nReject: ' + str(stats['reject_random']), 
                        xy=(2.5, 50), fontsize=10,
                        color='#e74c3c', ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor='#e74c3c'))
            selection_method = 'Random-10'
        
        ax.set_title(f'{name}: {selection_method}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 120)
        ax.set_ylabel('Count' if idx == 0 else '')
    
    fig.suptitle('Gate3 Filtering Pipeline: M1 (Align) vs M2/M5 (Random)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / 'F02_filtering_pipeline.png')
    fig.savefig(output_dir / 'F02_filtering_pipeline.pdf')
    print(f"✅ Saved: F02_filtering_pipeline.png + .pdf")
    plt.close(fig)


# ============================================================
# F03: Delta vs Gate1 Comparison
# ============================================================
def create_f03():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = ['M1\n(VAE+Align)', 'M2\n(gen_only)', 'M3\n(copy)', 'M4\n(noise)', 'M5\n(random)']
    deltas = [+0.0986, -0.0342, +0.1362, -0.3377, -0.0402]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]
    
    bars = ax.bar(names, deltas, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, val in zip(bars, deltas):
        height = bar.get_height()
        va = 'bottom' if val >= 0 else 'top'
        offset = 3 if val >= 0 else -12
        ax.annotate(f'{val:+.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('ΔTest R² (vs Gate1)')
    ax.set_title('Gate3 Phase1-mini: Performance Delta vs Gate1 Baseline', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(-0.45, 0.20)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Improvement'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Degradation'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_dir / 'F03_delta_comparison.png')
    fig.savefig(output_dir / 'F03_delta_comparison.pdf')
    print(f"✅ Saved: F03_delta_comparison.png + .pdf")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Gate3 Phase1-mini Figure Generation")
    print("=" * 60)
    
    create_f01()
    create_f02()
    create_f03()
    
    print("\n" + "=" * 60)
    print(f"  All figures saved to: {output_dir}")
    print("=" * 60)
