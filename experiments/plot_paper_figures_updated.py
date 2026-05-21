"""
plot_paper_figures.py
=====================
논문 Figure 생성 스크립트 (EAAI Paper 1)

데이터 출처: Gate4_Paper1_Results_Tables_v1_8.md (POST-PATCH)
출력: figures/ 디렉토리 (PNG + PDF 듀얼)

생성 Figure:
    Fig 2  — CP Gate3 Forest Plot (18 runs)
    Fig 3  — CP D-opt vs Random (10 seeds)
    Fig 4  — AEK 실험 히스토리 (noise-free)
    Fig 5  — AEK Coverage Gate 진단 (std_ratio)
    Fig 6  — Applicability Condition (κ + std(cos))
    Fig 7  — 4-System D-opt Selection Verdict (small-multiples, main text)
    Fig 8  — Silverbox Random score_aligned 분포 (D-opt point only)
    Fig A2 — Lynx-Hare score_aligned 분포 (Appendix)

사용법:
    python plot_paper_figures.py           # 전체 생성
    python plot_paper_figures.py --fig 3   # 특정 Figure만

SSOT 규칙:
    - matplotlib only (seaborn 금지)
    - PNG + PDF 듀얼 저장 via plot_style.save_figure()
    - plot_style.py 색상/스타일 체계 사용
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── SSOT import: plot_style.save_figure() (Instructions Rule #2) ─────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from src.contracts.plot_style import save_figure as _ssot_save
    _USE_SSOT_SAVE = True
except ImportError:
    try:
        from plot_style import save_figure as _ssot_save
        _USE_SSOT_SAVE = True
    except ImportError:
        _USE_SSOT_SAVE = False
        print("⚠️  plot_style.py not found; using local save fallback")

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)

# ── Style SSOT ────────────────────────────────────────────────────────────────
COLORS = {
    'dopt':      '#2166AC',   # blue — D-optimal
    'random':    '#4DAC26',   # green — Random
    'null':      '#D73027',   # red — NULL
    'soft':      '#FC8D59',   # orange — SOFT_PASS
    'strong':    '#1A9641',   # dark green — STRONG_PASS
    'ceiling':   '#762A83',   # purple — CEILING_BREAK
    'neutral':   '#636363',   # grey
    'aek':       '#E08214',   # amber — AEK
    'lorenz':    '#5AAE61',   # green — Lorenz
    'silverbox': '#3288BD',   # blue — Silverbox
    'lh':        '#D53E4F',   # red — Lynx-Hare
    'cp':        '#756BB1',   # purple — Cart-Pole
    'ceiling_line': '#762A83',
    'zero_line': '#AAAAAA',
}

FONT_SIZES = {'title': 11, 'label': 10, 'tick': 9, 'legend': 9, 'annot': 8}
CEILING = 0.058

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': FONT_SIZES['tick'],
    'axes.labelsize': FONT_SIZES['label'],
    'axes.titlesize': FONT_SIZES['title'],
    'legend.fontsize': FONT_SIZES['legend'],
    'xtick.labelsize': FONT_SIZES['tick'],
    'ytick.labelsize': FONT_SIZES['tick'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


def save_figure(fig, name: str):
    """Delegate to plot_style.save_figure() per SSOT Rule #2.
    Falls back to local implementation only if plot_style.py unavailable."""
    if _USE_SSOT_SAVE:
        _ssot_save(fig, OUT_DIR, name, close=True)
    else:
        for ext in ['png', 'pdf']:
            path = OUT_DIR / f'{name}.{ext}'
            fig.savefig(path, bbox_inches='tight', dpi=300 if ext == 'png' else None,
                        facecolor='white', edgecolor='none')
        print(f'  ✅ Saved: {OUT_DIR}/{name}.png / .pdf')
        plt.close(fig)


def pass_color(score, ci_lower=None):
    """Return color for a score_aligned value."""
    if score is None or np.isnan(score):
        return COLORS['neutral']
    if score <= 0:
        return COLORS['null']
    if ci_lower is not None and ci_lower > CEILING:
        return COLORS['ceiling']
    if ci_lower is not None and ci_lower > 0:
        return COLORS['strong']
    return COLORS['soft']


# ── DATA (Results Tables v1.8 SSOT — POST-PATCH) ────────────────────────────

# Fig 3: CP D-opt vs Random [POST-PATCH SavGol]
CP_RANDOM_SEEDS = list(range(10))
CP_RANDOM_SCORES = [-0.050, +0.138, -0.056, +0.019, +0.106,
                    +0.366, -0.049, +0.062, +0.104, +0.069]
CP_DOPT_SCORE   = +0.294
CP_DOPT_CI      = (-0.049, +0.975)

# Fig 8: Silverbox Random
SB_RANDOM_SEEDS  = list(range(10))
SB_RANDOM_SCORES = [+2.642, +2.513, +2.548, +2.537, +2.559,
                    +2.493, +2.441, +2.371, +2.397, +2.437]
SB_DOPT_SCORE    = +2.513
SB_DOPT_CI       = (-2.641, +11.450)

# Fig A2 (Appendix): Lynx-Hare Random
LH_RANDOM_SEEDS  = list(range(10))
LH_RANDOM_SCORES = [-0.165, +0.512, +0.282, +0.526, +0.403,
                    +0.443, +0.401, +0.672, +0.746, +0.543]
LH_DOPT_SCORE    = +0.410

# Lorenz Random
LZ_RANDOM_SCORES = [+1.716, +1.118, +1.492, +1.437, +1.501,
                    +1.466, +1.556, +1.668, +1.550, +1.566]
LZ_DOPT_SCORE    = +0.827

# AEK κ progression (Table 3-5)
AEK_KAPPA = {
    'Standard\n(Gate4b)':     4.7e9,
    'Reparam-1\n(Gate4c)':    4.5e4,
    'Reparam-2\n(exploratory)': 473,
}

# Coverage Gate (Table 7.4)
COVERAGE = {
    'sin(φ)':       {'baseline': 0.37, 'dither': 1.68, 'threshold': 0.70},
    'θ̇²':          {'baseline': 0.15, 'dither': 1.29, 'threshold': 0.50},
}

# Applicability Condition (Table 8)
AC_DATA = {
    'AEK':       {'std_cos': 0.00, 'delta_log_kappa': +5.02, 'pass': True},
    'CP':        {'std_cos': 0.28, 'delta_log_kappa': -0.07, 'pass': False},
}

# 5-System summary (Table 9) [POST-PATCH]
SYSTEMS_5 = ['CP', 'AEK', 'Lorenz', 'Silverbox', 'Lynx-Hare']
DOPT_SCORES   = [+0.294, None,   +0.827, +2.513, +0.410]  # None = EXPLODED
RANDOM_SCORES = [+0.066, -0.280, +1.525, +2.503, +0.478]
SYSTEM_COLORS = [COLORS['cp'], COLORS['aek'], COLORS['lorenz'],
                 COLORS['silverbox'], COLORS['lh']]
FAILURE_MODES = ['Recall\nFragility', 'Precision\nCollapse',
                 'Precision\nCollapse', 'Mixed', 'Precision\nCollapse']


# =============================================================================
# Fig 2: CP Gate3 Forest Plot
# =============================================================================
def plot_fig2_gate3_forest():
    """CP Gate3: 18-run summary with pass level breakdown."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Summary data from Table 1
    categories = ['CEILING_BREAK', 'SOFT_PASS', 'FAIL / NULL']
    counts      = [2, 14, 2]
    bar_colors  = [COLORS['ceiling'], COLORS['soft'], COLORS['null']]

    bars = ax.barh(categories, counts, color=bar_colors, edgecolor='white',
                   height=0.55)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{cnt}/18', va='center', ha='left',
                fontsize=FONT_SIZES['annot'], color='#333333')

    ax.axvline(x=18 * 0.89, color=COLORS['neutral'], lw=1,
               linestyle='--', alpha=0.6, label='89% pass rate')
    ax.set_xlabel('Number of runs (total = 18)')
    ax.set_title('Fig 2: CP Gate3 — Pass Level Distribution (18 runs)')
    ax.set_xlim(0, 20)
    ax.legend(fontsize=FONT_SIZES['legend'])

    # Annotation
    ax.text(0.98, 0.05,
            'Gate3: 89% SOFT_PASS+\nPhysics ceiling = 0.058',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT_SIZES['annot'], color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', alpha=0.8))

    fig.tight_layout()
    save_figure(fig, 'fig2_gate3_forest')


# =============================================================================
# Fig 3: CP D-opt vs Random
# =============================================================================
def plot_fig3_cp_dopt_random():
    """CP Gate4a: D-opt vs Random 10 seeds strip plot."""
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Random seeds (jittered x positions)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, len(CP_RANDOM_SCORES))
    x_rand = np.ones(len(CP_RANDOM_SCORES)) + jitter

    colors_r = [pass_color(s) for s in CP_RANDOM_SCORES]
    ax.scatter(x_rand, CP_RANDOM_SCORES, c=colors_r, s=60, zorder=3,
               edgecolors='white', linewidths=0.5)

    # Random median
    rand_med = float(np.median(CP_RANDOM_SCORES))
    ax.hlines(rand_med, 0.82, 1.18, colors=COLORS['random'], linewidths=2,
              linestyles='-', zorder=4, label=f'Random median = {rand_med:+.3f}')

    # D-opt with CI
    ax.errorbar(2.0, CP_DOPT_SCORE,
                yerr=[[CP_DOPT_SCORE - CP_DOPT_CI[0]],
                      [CP_DOPT_CI[1]  - CP_DOPT_SCORE]],
                fmt='D', color=COLORS['dopt'], markersize=10,
                capsize=6, capthick=2, elinewidth=2, zorder=5,
                label=f'D-opt = {CP_DOPT_SCORE:+.3f} (SOFT_PASS)')

    # Reference lines
    ax.axhline(0,         color=COLORS['zero_line'],    lw=1, ls='--', alpha=0.7)
    ax.axhline(CEILING,   color=COLORS['ceiling_line'], lw=1.2, ls=':', alpha=0.8,
               label=f'Ceiling = {CEILING}')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Random\n(10 seeds)', 'D-optimal'], fontsize=FONT_SIZES['label'])
    ax.set_ylabel('score_aligned (positive = improvement)')
    ax.set_title('Fig 3: Cart-Pole — D-optimal vs Random Selection')
    ax.set_xlim(0.5, 2.5)
    ax.legend(fontsize=FONT_SIZES['legend'])

    # NULL annotation (inside plot area, not overlapping axis label)
    null_count = sum(1 for s in CP_RANDOM_SCORES if s <= 0)
    ax.text(0.55, min(CP_RANDOM_SCORES) - 0.01,
            f'{null_count}/10 NULL', ha='left', va='top',
            fontsize=FONT_SIZES['annot'], color=COLORS['null'])

    fig.tight_layout()
    save_figure(fig, 'fig3_cp_dopt_random')


# =============================================================================
# Fig 4: AEK Experiment History
# =============================================================================
def plot_fig4_aek_history():
    """AEK κ progression and key score_aligned milestones."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: κ reduction
    labels = list(AEK_KAPPA.keys())
    kappas = list(AEK_KAPPA.values())
    log_kappas = [np.log10(k) for k in kappas]
    bar_colors = [COLORS['null'], COLORS['soft'], COLORS['neutral']]

    bars = ax1.bar(labels, log_kappas, color=bar_colors, edgecolor='white',
                   width=0.55)
    ax1.set_ylabel('log₁₀(κ)')
    ax1.set_title('(a) AEK Library Conditioning')
    for bar, lk, k in zip(bars, log_kappas, kappas):
        ax1.text(bar.get_x() + bar.get_width() / 2, lk + 0.1,
                 f'κ={k:.1e}', ha='center', va='bottom',
                 fontsize=FONT_SIZES['annot'])

    # Threshold line (κ < 10^6 considered "acceptable" informally)
    ax1.axhline(6, color=COLORS['neutral'], lw=1, ls='--', alpha=0.5)
    ax1.set_ylim(0, 12)

    # Right: score_aligned milestones (key results only)
    experiments = [
        ('Std\nRandom', -0.65,  COLORS['null'],   'baseline pool'),
        ('RP1\nRandom', -0.65,  COLORS['null'],   'baseline pool'),
        ('RP1\nDither\nRandom', -0.280, COLORS['soft'],  'dither pool'),
    ]
    x_pos   = np.arange(len(experiments))
    labels2 = [e[0] for e in experiments]
    scores2 = [e[1] for e in experiments]
    cols2   = [e[2] for e in experiments]

    ax2.bar(x_pos, scores2, color=cols2, edgecolor='white', width=0.55)
    ax2.axhline(0, color=COLORS['zero_line'], lw=1, ls='--', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels2, fontsize=FONT_SIZES['tick'])
    ax2.set_ylabel('score_aligned (median)')
    ax2.set_title('(b) AEK score_aligned Milestones')
    ax2.set_ylim(-1.0, 0.3)

    # Annotation for SOFT_PASS — must show median is still negative
    ax2.annotate('2/10 SOFT_PASS\n(first positive seeds;\nmedian = −0.280)',
                 xy=(2, -0.280), xytext=(1.5, 0.15),
                 arrowprops=dict(arrowstyle='->', color='#555'),
                 fontsize=FONT_SIZES['annot'], color=COLORS['soft'])

    fig.suptitle('Fig 4: AEK System — Conditioning Improvement & Score History',
                 fontsize=FONT_SIZES['title'])
    fig.tight_layout()
    save_figure(fig, 'fig4_aek_history')


# =============================================================================
# Fig 5: AEK Coverage Gate Diagnosis
# =============================================================================
def plot_fig5_coverage():
    """AEK Coverage Gate: std_ratio before/after dither."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    variables = list(COVERAGE.keys())
    x = np.arange(len(variables))
    w = 0.3

    for i, (var, data) in enumerate(COVERAGE.items()):
        ax.bar(i - w/2, data['baseline'], width=w, color=COLORS['null'],
               label='Baseline' if i == 0 else '', edgecolor='white')
        ax.bar(i + w/2, data['dither'],   width=w, color=COLORS['random'],
               label='Dither+' if i == 0 else '', edgecolor='white')
        # Threshold line per variable
        ax.hlines(data['threshold'], i - 0.4, i + 0.4,
                  colors=COLORS['ceiling_line'], linewidths=1.5,
                  linestyles=':', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=FONT_SIZES['label'])
    ax.set_ylabel('std_ratio (augmented / training)')
    ax.set_title('Fig 5: AEK Coverage Gate Diagnosis')
    ax.legend(fontsize=FONT_SIZES['legend'])

    # Threshold annotation
    ax.text(1.45, COVERAGE['θ̇²']['threshold'] + 0.03,
            'Gate threshold', fontsize=FONT_SIZES['annot'],
            color=COLORS['ceiling_line'])

    # PASS/FAIL annotations
    for i, (var, data) in enumerate(COVERAGE.items()):
        ax.text(i - w/2, data['baseline'] + 0.05, '✗',
                ha='center', color=COLORS['null'], fontsize=11)
        ax.text(i + w/2, data['dither'] + 0.05, '✓',
                ha='center', color=COLORS['random'], fontsize=11)

    ax.set_ylim(0, 2.2)
    fig.tight_layout()
    save_figure(fig, 'fig5_aek_coverage')


# =============================================================================
# Fig 6: Applicability Condition
# =============================================================================
def plot_fig6_applicability():
    """Gate4d: Applicability Condition — std(cos(θ)) vs κ improvement."""
    fig, ax = plt.subplots(figsize=(5, 4))

    systems = list(AC_DATA.keys())
    std_vals = [AC_DATA[s]['std_cos'] for s in systems]
    delta_log = [AC_DATA[s]['delta_log_kappa'] for s in systems]
    colors_ac = [COLORS['strong'] if AC_DATA[s]['pass'] else COLORS['null']
                 for s in systems]

    sc = ax.scatter(std_vals, delta_log, c=colors_ac, s=150, zorder=4,
                    edgecolors='white', linewidths=1.5)

    for s, x, y in zip(systems, std_vals, delta_log):
        ax.annotate(f' {s}', (x, y), fontsize=FONT_SIZES['label'],
                    va='center')

    ax.axhline(0, color=COLORS['zero_line'], lw=1, ls='--', alpha=0.7)
    ax.axvline(0.01, color=COLORS['ceiling_line'], lw=1.2, ls=':',
               alpha=0.8, label='Applicability threshold\n(std(cos(θ)) < 0.01)')

    ax.set_xlabel('std(cos(θ)) of training data')
    ax.set_ylabel('Δlog₁₀(κ)  [positive = improvement]')
    ax.set_title('Fig 6: Applicability Condition — Reparameterization Effect')

    # Legend patches
    pass_patch = mpatches.Patch(color=COLORS['strong'], label='AC: PASS')
    fail_patch = mpatches.Patch(color=COLORS['null'],   label='AC: FAIL')
    ax.legend(handles=[pass_patch, fail_patch,
                        plt.Line2D([0], [0], color=COLORS['ceiling_line'],
                                   lw=1.5, ls=':', label='Threshold')],
              fontsize=FONT_SIZES['legend'])

    fig.tight_layout()
    save_figure(fig, 'fig6_applicability')


# =============================================================================
# Fig 7: 4-System D-opt vs Random — Verdict Figure (main text)
# =============================================================================
def plot_fig7_verdict():
    """4 main-text systems: verdict-based comparison (no shared y-axis bars).
    Absolute score_aligned magnitudes are NOT comparable across systems.
    This figure shows the VERDICT (D-opt benefit direction) per system."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.2), sharey=False)

    # 4 main-text systems only (Lynx-Hare → Appendix)
    systems_4 = ['Cart-Pole', 'AEK', 'Lorenz-63', 'Silverbox']
    random_4  = [+0.066, -0.280, +1.525, +2.503]
    dopt_4    = [+0.294, None,   +0.827, +2.513]  # None = EXPLODED
    fm_4      = ['Recall\nFragility', 'Precision\nCollapse',
                 'Precision\nCollapse', 'Mixed\n(PC dominant)']
    verdicts  = ['D-opt > Random', 'Random > D-opt\n(EXPLODED)', 'D-opt ≈ Random', 'D-opt ≈ Random']
    verdict_colors = [COLORS['dopt'], COLORS['null'], COLORS['neutral'], COLORS['neutral']]
    sys_colors = [COLORS['cp'], COLORS['aek'], COLORS['lorenz'], COLORS['silverbox']]

    for i, ax in enumerate(axes):
        # Draw Random and D-opt as two dots
        ax.plot(0, random_4[i], 'o', color=COLORS['random'], markersize=10,
                markeredgecolor='white', markeredgewidth=1, zorder=5)
        ax.text(0, random_4[i], f' {random_4[i]:+.3f}', va='center', ha='left',
                fontsize=FONT_SIZES['annot'], color=COLORS['random'] if random_4[i] > 0 else COLORS['null'])

        if dopt_4[i] is not None:
            ax.plot(1, dopt_4[i], 'D', color=COLORS['dopt'], markersize=10,
                    markeredgecolor='white', markeredgewidth=1, zorder=5)
            ax.text(1, dopt_4[i], f' {dopt_4[i]:+.3f}', va='center', ha='left',
                    fontsize=FONT_SIZES['annot'], color=COLORS['dopt'])
        else:
            # EXPLODED: no marker at y=0 (avoid '0' oread); text-only badge
            ax.text(0.5, 0.75, 'EXPLODED\n(coeff ~1.3e8)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=FONT_SIZES['annot']+1, fontweight='bold',
                    color=COLORS['null'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE', alpha=0.9))

        # Zero line
        ax.axhline(0, color=COLORS['zero_line'], lw=1, ls='--', alpha=0.5)

        # Verdict banner at bottom
        ax.text(0.5, -0.22, verdicts[i], transform=ax.transAxes,
                ha='center', va='top', fontsize=7.5, fontweight='bold',
                color=verdict_colors[i],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5', alpha=0.9))

        # System title includes failure mode
        ax.set_title(f'{systems_4[i]}\n({fm_4[i].replace(chr(10), " ")})',
                     fontsize=FONT_SIZES['label'],
                     color=sys_colors[i], fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Rand', 'D-opt'], fontsize=FONT_SIZES['tick'])
        ax.set_xlim(-0.5, 1.8)

        # Remove y-axis for non-first panels to emphasize non-comparability
        if i > 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('score_aligned', fontsize=FONT_SIZES['label'])

    fig.suptitle('Fig 7: D-optimal Selection Verdict — 4 Main-Text Systems\n'
                 '(absolute magnitudes not comparable across systems)',
                 fontsize=FONT_SIZES['title'], y=1.08)

    # Shared legend
    rand_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['random'],
                              markersize=8, label='Random (median)')
    dopt_marker = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['dopt'],
                              markersize=8, label='D-optimal')
    fig.legend(handles=[rand_marker, dopt_marker],
               fontsize=FONT_SIZES['legend'], loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure(fig, 'fig7_5system_comparison')


# =============================================================================
# Fig 8: Silverbox Random distribution
# =============================================================================
def plot_fig8_silverbox():
    """Silverbox: Random 10-seed score_aligned distribution."""
    fig, ax = plt.subplots(figsize=(5.5, 4))

    rng = np.random.default_rng(1)
    jitter = rng.uniform(-0.08, 0.08, len(SB_RANDOM_SCORES))
    x_rand = np.ones(len(SB_RANDOM_SCORES)) + jitter

    colors_r = [pass_color(s) for s in SB_RANDOM_SCORES]
    ax.scatter(x_rand, SB_RANDOM_SCORES, c=colors_r, s=70, zorder=3,
               edgecolors='white', linewidths=0.5)

    # Median line
    sb_med = float(np.median(SB_RANDOM_SCORES))
    ax.hlines(sb_med, 0.82, 1.18, colors=COLORS['random'], linewidths=2.5,
              label=f'Random median = {sb_med:+.3f}')

    # D-opt point ONLY (CI unstable — omitted from plot per SSOT)
    ax.plot(2.0, SB_DOPT_SCORE, 'D', color=COLORS['dopt'],
            markersize=10, zorder=5,
            label=f'D-opt = {SB_DOPT_SCORE:+.3f} (point est. only;\nCI unstable and omitted)')

    # Reference lines
    ax.axhline(0, color=COLORS['zero_line'], lw=1, ls='--', alpha=0.7, label='Zero baseline')
    # NOTE: CP-derived ceiling (0.058) intentionally omitted from Silverbox figure.
    # Cross-system absolute score_aligned comparison is not meaningful due to
    # differing z-score scales. Silverbox headline = CI entirely positive, not ceiling comparison.

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Random\n(10 seeds)', 'D-optimal\n(point est. only)'],
                       fontsize=FONT_SIZES['label'])
    ax.set_ylabel('score_aligned (positive = improvement)')
    ax.set_title('Fig 8: Silverbox — GMM Augmentation Results\n'
                 '(Real Engineering Data, κ=3.814)')
    ax.legend(fontsize=FONT_SIZES['legend'])

    # CI entirely positive annotation
    ax.text(0.98, 0.05,
            '10/10 SOFT_PASS\nCI entirely positive\n[+2.419, +2.553]',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT_SIZES['annot'], color=COLORS['random'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', alpha=0.8))

    ax.set_xlim(0.5, 2.7)
    ax.set_ylim(-0.5, 4.0)
    fig.tight_layout()
    save_figure(fig, 'fig8_silverbox_random')


# =============================================================================
# Fig A2 (Appendix): Lynx-Hare
# =============================================================================
def plot_figA2_lynxhare():
    """Lynx-Hare Random 10 seeds (Appendix)."""
    fig, ax = plt.subplots(figsize=(5.5, 4))

    rng = np.random.default_rng(2)
    jitter = rng.uniform(-0.08, 0.08, len(LH_RANDOM_SCORES))
    x_rand = np.ones(len(LH_RANDOM_SCORES)) + jitter

    colors_r = [pass_color(s) for s in LH_RANDOM_SCORES]
    ax.scatter(x_rand, LH_RANDOM_SCORES, c=colors_r, s=60, zorder=3,
               edgecolors='white', linewidths=0.5)

    lh_med = float(np.median(LH_RANDOM_SCORES))
    ax.hlines(lh_med, 0.82, 1.18, colors=COLORS['random'], linewidths=2,
              label=f'Random median = {lh_med:+.3f}')

    # D-opt pass level (dynamic)
    dopt_pass = 'SOFT_PASS' if LH_DOPT_SCORE > 0 else 'NULL'
    ax.plot(2.0, LH_DOPT_SCORE, 'D', color=COLORS['dopt'],
            markersize=9, label=f'D-opt = {LH_DOPT_SCORE:+.3f} ({dopt_pass})', zorder=5)

    ax.axhline(0, color=COLORS['zero_line'], lw=1, ls='--', alpha=0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Random\n(10 seeds)', 'D-optimal'], fontsize=FONT_SIZES['label'])
    ax.set_ylabel('score_aligned (positive = improvement)')
    ax.set_title('Fig A2: Lynx-Hare (Appendix)\n'
                 'Hudson Bay fur records 1900–1920, n_train=3')
    ax.legend(fontsize=FONT_SIZES['legend'])

    null_count = sum(1 for s in LH_RANDOM_SCORES if s <= 0)
    soft_count = sum(1 for s in LH_RANDOM_SCORES if s > 0)
    ax.text(0.98, 0.95,
            f'NULL:{null_count}/SOFT:{soft_count}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_SIZES['annot'], color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', alpha=0.8))

    fig.tight_layout()
    save_figure(fig, 'figA2_lynxhare_appendix')


# =============================================================================
# Main
# =============================================================================
FIGURE_MAP = {
    '2':  plot_fig2_gate3_forest,
    '3':  plot_fig3_cp_dopt_random,
    '4':  plot_fig4_aek_history,
    '5':  plot_fig5_coverage,
    '6':  plot_fig6_applicability,
    '7':  plot_fig7_verdict,
    '8':  plot_fig8_silverbox,
    'A2': plot_figA2_lynxhare,
}


def main():
    parser = argparse.ArgumentParser(description='Generate EAAI paper figures')
    parser.add_argument('--fig', nargs='+', default=['all'],
                        help='Figure numbers to generate (2 3 4 5 6 7 8 A2 or all)')
    args = parser.parse_args()

    targets = list(FIGURE_MAP.keys()) if 'all' in args.fig else args.fig
    print(f'\n[plot_paper_figures.py] Generating {len(targets)} figure(s)...\n')

    for key in targets:
        if key not in FIGURE_MAP:
            print(f'  ⚠️  Unknown figure: {key}')
            continue
        print(f'  → Fig {key}')
        FIGURE_MAP[key]()

    print(f'\n✅ Done. Output: {OUT_DIR.resolve()}/\n')


if __name__ == '__main__':
    main()