"""
generate_fig1_matplotlib.py
============================

Phase R1 Step 10B-7 v1.4 hot-fix (P1-3 resolution candidate).

Deterministic matplotlib-based Fig 1 generator, replacing the GPT-generated
image to address:
  - P1-1 GenAI declaration burden (no AI image generation involved)
  - P1-3 Fig 1 notation quality (LaTeX mathtext renders F_or, F_sp, D, pi cleanly)

Outputs:
  figures/fig1_pipeline_overview.png  (300 DPI for paper inclusion)
  figures/fig1_pipeline_overview.pdf  (vector-equivalent)

Visual consistency with other 8 figures:
  - Same COLORS palette (orange diamonds for conditional gates, etc.)
  - Same FONT_SIZES (sans-serif title 11, label 10, tick 9)
  - Same PNG+PDF dual save convention

Usage:
  python generate_fig1_matplotlib.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── SSOT import: plot_style.save_figure() ────────────────────────────────────
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
        print("[WARN] plot_style.py not found; using local save fallback")

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)

# ── Color palette (consistent with plot_paper_figures.py) ────────────────────
COLORS = {
    'dopt':       '#2166AC',   # blue — D-optimal
    'random':     '#4DAC26',   # green — Random
    'null':       '#D73027',   # red — NULL/FAIL
    'soft':       '#FC8D59',   # orange — emphasis (Re-baseline / Re-generate pool)
    'strong':     '#1A9641',   # dark green — PASS/final
    'aek':        '#E08214',   # amber — diamond (conditional gate)
    'neutral':    '#636363',   # grey
    'box_face':   '#FFFFFF',
    'box_edge':   '#333333',
    'diamond_face':  '#FFF3E0',   # light cream
    'diamond_edge':  '#E08214',
    'sublabel':   '#888888',
}

FONT = {
    'title': 15,
    'stage_header': 10,
    'stage_body': 8.5,
    'branch': 9,
    'sublabel': 8,
    'arrow_label': 9.5,
    'math': 9.5,
}


# ── Helper functions for flowchart drawing ──────────────────────────────────

def draw_stage_box(ax, x, y, width, height, header, body=None, 
                   edge_color=None, header_weight='bold'):
    """Draw a rounded rectangle stage box with header + optional body text."""
    edge = edge_color or COLORS['box_edge']
    rect = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.4, edgecolor=edge, facecolor=COLORS['box_face']
    )
    ax.add_patch(rect)
    if body:
        # Header at top, body below
        ax.text(x, y + height*0.20, header, ha='center', va='center',
                fontsize=FONT['stage_header'], weight=header_weight)
        ax.text(x, y - height*0.22, body, ha='center', va='center',
                fontsize=FONT['stage_body'])
    else:
        ax.text(x, y, header, ha='center', va='center',
                fontsize=FONT['stage_header'], weight=header_weight)


def draw_diamond(ax, x, y, width, height, text, color=None):
    """Draw a diamond-shaped conditional gate node."""
    edge = color or COLORS['diamond_edge']
    pts = [(x, y + height/2), (x + width/2, y), (x, y - height/2), (x - width/2, y)]
    poly = mpatches.Polygon(
        pts, closed=True, linewidth=1.6, edgecolor=edge,
        facecolor=COLORS['diamond_face']
    )
    ax.add_patch(poly)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=FONT['branch'], weight='bold', color=edge)


def draw_branch_box(ax, x, y, width, height, header, body=None, edge_color=None):
    """Draw a left/right branch box (smaller than main stage)."""
    edge = edge_color or COLORS['box_edge']
    rect = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2, edgecolor=edge, facecolor=COLORS['box_face']
    )
    ax.add_patch(rect)
    if body:
        ax.text(x, y + height*0.18, header, ha='center', va='center',
                fontsize=FONT['branch'])
        ax.text(x, y - height*0.22, body, ha='center', va='center',
                fontsize=FONT['arrow_label'], weight='bold', color=COLORS['soft'])
    else:
        ax.text(x, y, header, ha='center', va='center', fontsize=FONT['branch'])


def draw_arrow(ax, x1, y1, x2, y2, color=None, lw=1.3, style='->',
               connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    arrow = mpatches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color or '#333333',
        connectionstyle=connectionstyle,
        shrinkA=4, shrinkB=4
    )
    ax.add_patch(arrow)


def label(ax, x, y, text, color=None, fontsize=None, weight='normal', style='normal'):
    """Place a text label at (x, y)."""
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize or FONT['arrow_label'],
            color=color or '#333333', weight=weight, style=style)


# ── Main figure construction ──────────────────────────────────────────────────

def build_fig1():
    """Build the Diagnose-Before-Augment Decision Protocol flowchart."""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # X positions
    x_c = 7.0       # center column
    x_left = 2.6   # left branch column
    x_right = 11.6 # right branch column

    # Y positions (top to bottom)
    y_title    = 10.4
    y_s1       = 9.5
    y_diam_app = 8.4
    y_s2       = 7.3
    y_s3       = 6.3
    y_s4       = 5.3
    y_s5       = 4.4
    y_s6       = 3.4
    y_diam_cov = 2.3
    y_s7       = 1.1
    y_s8       = 0.0  # will use height/2 above 0

    # Box dimensions
    stage_w  = 5.4
    stage_h  = 0.85
    branch_w = 3.6
    branch_h = 0.95
    diam_w   = 2.6
    diam_h   = 1.0
    diam7_w  = 2.8
    diam7_h  = 1.1

    # =========================================================================
    # Title
    # =========================================================================
    ax.text(x_c, y_title, 'Diagnose-Before-Augment Decision Protocol',
            ha='center', va='center', fontsize=FONT['title'], weight='bold')

    # =========================================================================
    # Stage 1
    # =========================================================================
    draw_stage_box(ax, x_c, y_s1, stage_w, stage_h,
                   header='Stage 1.  Structural Pre-check',
                   body='Assess structural applicability before augmentation')
    draw_arrow(ax, x_c, y_s1 - stage_h/2, x_c, y_diam_app + diam_h/2)

    # =========================================================================
    # Applicability Condition diamond
    # =========================================================================
    draw_diamond(ax, x_c, y_diam_app, diam_w, diam_h,
                 'Applicability\nCondition?')

    # YES branch (left): Reparameterize library
    # Left branch box
    draw_branch_box(ax, x_left, y_diam_app, branch_w, branch_h,
                    header=r'Reparameterize library  ($\Theta \rightarrow \Theta\,\prime$)',
                    body='Re-baseline')
    label(ax, x_left, y_diam_app - branch_h/2 - 0.22,
          'Library changed', color=COLORS['sublabel'],
          fontsize=FONT['sublabel'], style='italic')
    # Arrow: diamond -> reparameterize
    draw_arrow(ax, x_c - diam_w/2, y_diam_app,
               x_left + branch_w/2, y_diam_app,
               color=COLORS['strong'], lw=1.5)
    label(ax, (x_c - diam_w/2 + x_left + branch_w/2)/2, y_diam_app + 0.25,
          'YES', color=COLORS['strong'], weight='bold')

    # NO branch (right): Identity (text only, no box)
    label(ax, x_c + diam_w/2 + 0.4, y_diam_app, 'NO',
          color=COLORS['neutral'], weight='bold', fontsize=FONT['arrow_label'])
    label(ax, x_c + diam_w/2 + 0.9, y_diam_app - 0.22, 'Identity',
          color=COLORS['sublabel'], fontsize=FONT['sublabel'], style='italic')

    # Arrow: diamond -> Stage 2 (down)
    draw_arrow(ax, x_c, y_diam_app - diam_h/2, x_c, y_s2 + stage_h/2)

    # Arrow: Reparameterize (left branch) -> Stage 2 (joining back via curve)
    # We curve from below the branch back to top-left of stage 2
    draw_arrow(ax, x_left, y_diam_app - branch_h/2 - 0.45,
               x_c - stage_w/2 + 0.5, y_s2 + stage_h/2,
               connectionstyle='arc3,rad=-0.15',
               color='#555555')

    # =========================================================================
    # Stage 2
    # =========================================================================
    draw_stage_box(ax, x_c, y_s2, stage_w, stage_h,
                   header='Stage 2.  Baseline E-SINDy Ensemble',
                   body=r'Bootstrap-resample training data ($B$ times);  '
                        r'compute $z$-scores and inclusion probabilities')
    draw_arrow(ax, x_c, y_s2 - stage_h/2, x_c, y_s3 + stage_h/2)

    # =========================================================================
    # Stage 3
    # =========================================================================
    draw_stage_box(ax, x_c, y_s3, stage_w, stage_h,
                   header='Stage 3.  Diagnostic-Pair Extraction',
                   body=r'$\mathcal{F}_{\mathrm{or}}$: oracle terms with $z < z_{\mathrm{thr}}$  '
                        r'(missed)    '
                        r'$\mathcal{F}_{\mathrm{sp}}$: non-oracle terms with high $P_{\mathrm{inc}}$ or high $z$ (retained)' '\n'
                        r'$\mathcal{D} = \mathcal{F}_{\mathrm{or}} \cup \mathcal{F}_{\mathrm{sp}}$')
    draw_arrow(ax, x_c, y_s3 - stage_h/2, x_c, y_s4 + stage_h/2)

    # =========================================================================
    # Stage 4
    # =========================================================================
    draw_stage_box(ax, x_c, y_s4, stage_w, stage_h,
                   header='Stage 4.  Failure-Mode Classification',
                   body=r'Based on $(n_{\mathrm{or}}, n_{\mathrm{sp}})$ counts:  '
                        r'recall-fragility  |  precision-collapse  |  mixed  |  no-diagnostic-pair')
    draw_arrow(ax, x_c, y_s4 - stage_h/2, x_c, y_s5 + stage_h/2)

    # =========================================================================
    # Stage 5
    # =========================================================================
    draw_stage_box(ax, x_c, y_s5, stage_w, stage_h,
                   header=r'Stage 5.  Sign Fixation  ($s$ frozen)',
                   body=r'$s = +1$ for recall-fragility    '
                        r'$s = -1$ for precision-collapse or mixed')
    draw_arrow(ax, x_c, y_s5 - stage_h/2, x_c, y_s6 + stage_h/2)

    # =========================================================================
    # Stage 6
    # =========================================================================
    draw_stage_box(ax, x_c, y_s6, stage_w, stage_h,
                   header='Stage 6.  Pool Generation + Coverage Check',
                   body=r'Fit GMM on training data;  simulate via $f_{\mathrm{teacher}}$;  '
                        r'compute pool-to-training coverage ratios')
    draw_arrow(ax, x_c, y_s6 - stage_h/2, x_c, y_diam_cov + diam_h/2)

    # =========================================================================
    # Coverage Gate diamond
    # =========================================================================
    draw_diamond(ax, x_c, y_diam_cov, diam_w, diam_h,
                 'Coverage\nGate?')

    # FAIL branch (left): Coverage-aware generation
    draw_branch_box(ax, x_left, y_diam_cov, branch_w, branch_h,
                    header='Coverage-aware generation\n(excitation dither)',
                    body='Re-generate pool')
    label(ax, x_left, y_diam_cov - branch_h/2 - 0.22,
          'Controller or GMM changed', color=COLORS['sublabel'],
          fontsize=FONT['sublabel'], style='italic')
    draw_arrow(ax, x_c - diam_w/2, y_diam_cov,
               x_left + branch_w/2, y_diam_cov,
               color=COLORS['null'], lw=1.5)
    label(ax, (x_c - diam_w/2 + x_left + branch_w/2)/2, y_diam_cov + 0.25,
          'FAIL', color=COLORS['null'], weight='bold')

    # PASS branch (right): Default pool (text only)
    label(ax, x_c + diam_w/2 + 0.4, y_diam_cov, 'PASS',
          color=COLORS['strong'], weight='bold', fontsize=FONT['arrow_label'])
    label(ax, x_c + diam_w/2 + 0.95, y_diam_cov - 0.22, 'Default pool',
          color=COLORS['sublabel'], fontsize=FONT['sublabel'], style='italic')

    # Arrow: diamond -> Stage 7 (down)
    draw_arrow(ax, x_c, y_diam_cov - diam_h/2, x_c, y_s7 + diam7_h/2)

    # Arrow: Coverage-aware (left) -> Stage 7 (curve back)
    draw_arrow(ax, x_left, y_diam_cov - branch_h/2 - 0.45,
               x_c - diam7_w/2 + 0.3, y_s7 + 0.2,
               connectionstyle='arc3,rad=-0.15',
               color='#555555')

    # =========================================================================
    # Stage 7: Conditional Selection (diamond form)
    # =========================================================================
    draw_diamond(ax, x_c, y_s7, diam7_w, diam7_h,
                 'Stage 7.\nConditional\nSelection')

    # Branch 1 (left, blue): If recall-fragility -> D-optimal
    draw_branch_box(ax, x_left, y_s7, branch_w, branch_h,
                    header='If recall-fragility:\nD-optimal selection',
                    edge_color=COLORS['dopt'])
    draw_arrow(ax, x_c - diam7_w/2, y_s7,
               x_left + branch_w/2, y_s7,
               color=COLORS['dopt'], lw=1.5)

    # Branch 2 (right, green): Else -> Random
    draw_branch_box(ax, x_right, y_s7, branch_w, branch_h,
                    header=r'Else $\rightarrow$ Random selection',
                    edge_color=COLORS['strong'])
    draw_arrow(ax, x_c + diam7_w/2, y_s7,
               x_right - branch_w/2, y_s7,
               color=COLORS['strong'], lw=1.5)

    # Arrows from both branches back down to Stage 8
    s8_y_top = y_s8 + stage_h/2 + 0.25  # slightly above Stage 8
    draw_arrow(ax, x_left, y_s7 - branch_h/2 - 0.05,
               x_c - 1.5, s8_y_top,
               connectionstyle='arc3,rad=-0.15',
               color=COLORS['dopt'])
    draw_arrow(ax, x_right, y_s7 - branch_h/2 - 0.05,
               x_c + 1.5, s8_y_top,
               connectionstyle='arc3,rad=0.15',
               color=COLORS['strong'])

    # =========================================================================
    # Stage 8: Evaluation & Reporting (final, dark green outline)
    # =========================================================================
    # Make stage 8 slightly taller and emphasized
    s8_h = 1.1
    draw_stage_box(ax, x_c, y_s8 + s8_h/2 - 0.4, stage_w + 0.3, s8_h,
                   header='Stage 8.  Evaluation & Reporting',
                   body=r'Re-fit E-SINDy on $\mathcal{T} \cup \pi$;  '
                        r'compute $\mathrm{score\_aligned} = s \cdot \delta_{\mathrm{raw}}$' '\n'
                        r'Confidence interval + pass-level assignment;  '
                        r'report F1, coefficient error, RMSE / $R^2$',
                   edge_color=COLORS['strong'])

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig


def main():
    print('[plot_fig1_pipeline_overview] Building Fig 1 (deterministic matplotlib)...')
    fig = build_fig1()

    name = 'fig1_pipeline_overview'
    if _USE_SSOT_SAVE:
        _ssot_save(fig, OUT_DIR, name, close=True)
        print(f'  Saved via plot_style.save_figure(): {OUT_DIR}/{name}.png + .pdf')
    else:
        for ext in ['png', 'pdf']:
            path = OUT_DIR / f'{name}.{ext}'
            fig.savefig(path, bbox_inches='tight',
                        dpi=300 if ext == 'png' else None,
                        facecolor='white', edgecolor='none')
            print(f'  Saved: {path}')
        plt.close(fig)

    print()
    print('Quality evaluation checklist:')
    print('  [ ] Title: "Diagnose-Before-Augment Decision Protocol"')
    print('  [ ] Math notation crisp: F_or, F_sp, D = F_or ∪ F_sp, T ∪ π, s·delta_raw')
    print('  [ ] 3 diamonds (Applicability Condition / Coverage Gate / Stage 7)')
    print('  [ ] Re-baseline + Library changed annotation (left branch)')
    print('  [ ] Re-generate pool + Controller-or-GMM-changed annotation (left branch)')
    print('  [ ] D-optimal (blue) vs Random (green) selection branches')
    print('  [ ] Stage 8 emphasized (dark green outline)')
    print('  [ ] Output PNG file size and timestamp updated')


if __name__ == '__main__':
    main()
