"""
src/contracts/plot_style.py

SSOT #2: Figure 저장 규칙 강제
- matplotlib only (seaborn 금지)
- PNG + PDF 동시 저장 필수
- save_figure() 함수만 사용

Gate: 0-1
Version: v3.2 (Lean Mode)
"""

import matplotlib
# Agg 백엔드 설정 (GUI 불필요, headless/CI 환경 호환)
# 이 설정은 pyplot import 전에 해야 함
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import warnings

# ============================================================
# 전역 스타일 설정 (고정)
# ============================================================

STYLE_CONFIG = {
    # 폰트 (크로스플랫폼 안정성 우선)
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    
    # 선 스타일
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 0.8,
    
    # 그리드
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    
    # Figure
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    
    # 기타
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.axisbelow': True,
}

# ============================================================
# 색상 팔레트 (고정)
# ============================================================

COLORS = {
    # 방법별 색상
    'proposed': '#2E86AB',      # 파란색 (제안 방법)
    'esindy': '#E94F37',        # 빨간색 (E-SINDy)
    'sindy_ae': '#F39C12',      # 주황색 (SINDy-AE)
    'real_only': '#7D3C98',     # 보라색 (Real only)
    
    # 데이터 유형별 색상
    'real': '#1A5276',          # 진한 파랑 (Real data)
    'generated': '#27AE60',     # 초록색 (Generated)
    'prediction': '#E74C3C',    # 빨간색 (Prediction)
    'ground_truth': '#2C3E50',  # 진한 회색 (Ground truth)
    
    # 기타
    'neutral': '#7F8C8D',       # 회색
    'highlight': '#F1C40F',     # 노란색 (강조)
    
    # 조건별 색상 (condition distribution용)
    'train': '#3498DB',         # 파란색
    'val': '#9B59B6',           # 보라색
    'test': '#E74C3C',          # 빨간색
}

# ============================================================
# 마커 스타일 (고정)
# ============================================================

MARKERS = {
    'proposed': 'o',
    'esindy': 's',
    'sindy_ae': '^',
    'real_only': 'x',
    'train': 'o',
    'val': 's',
    'test': '^',
}

# ============================================================
# Figure 크기 (고정)
# ============================================================

FIG_SIZES = {
    'single': (4, 3),           # 단일 플롯
    'wide': (8, 3),             # 가로로 넓은
    'square': (4, 4),           # 정사각형
    'double': (8, 4),           # 2열 플롯
    'tall': (4, 6),             # 세로로 긴
    'large': (8, 6),            # 큰 플롯
}


# ============================================================
# 스타일 적용 함수
# ============================================================

def setup_style() -> None:
    """
    전역 matplotlib 스타일 적용
    
    모든 Figure 생성 전에 호출 권장
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    # seaborn 사용 감지 시 경고
    if 'seaborn' in plt.style.available:
        pass  # seaborn이 설치되어 있어도 사용하지 않음


def reset_style() -> None:
    """
    matplotlib 스타일 초기화
    """
    matplotlib.rcdefaults()


# ============================================================
# Figure 생성 함수
# ============================================================

def create_figure(
    size_key: str = 'single',
    nrows: int = 1,
    ncols: int = 1
) -> Tuple[plt.Figure, any]:
    """
    표준 Figure 생성
    
    Args:
        size_key: Figure 크기 키 ('single', 'wide', 'square', 'double', 'tall', 'large')
        nrows: subplot 행 수
        ncols: subplot 열 수
    
    Returns:
        (fig, ax) 또는 (fig, axes) 튜플
    
    Example:
        >>> fig, ax = create_figure('single')
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure(fig, results_dir / 'figures', 'my_plot')
    """
    setup_style()
    
    if size_key not in FIG_SIZES:
        warnings.warn(f"Unknown size_key '{size_key}', using 'single'")
        size_key = 'single'
    
    figsize = FIG_SIZES[size_key]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    return fig, axes


# ============================================================
# Figure 저장 함수 (핵심 - 이것만 사용)
# ============================================================

def save_figure(
    fig: plt.Figure,
    save_dir: Path,
    name: str,
    close: bool = True
) -> Tuple[Path, Path]:
    """
    Figure를 PNG + PDF로 동시 저장 (필수)
    
    ⚠️ 이 함수만 사용하여 Figure 저장
    ⚠️ fig.savefig() 직접 호출 금지
    
    Args:
        fig: matplotlib Figure 객체
        save_dir: 저장 디렉토리 (예: results_dir / 'figures')
        name: 파일명 (확장자 제외, 예: 'F00_condition_distribution')
        close: 저장 후 Figure 닫기 (기본값: True)
    
    Returns:
        (png_path, pdf_path) 튜플
    
    Example:
        >>> fig, ax = create_figure()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure(fig, Path('results/.../figures'), 'F00_condition_distribution')
        ✅ 저장: F00_condition_distribution.png + .pdf
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = save_dir / f"{name}.png"
    pdf_path = save_dir / f"{name}.pdf"
    
    # PNG 저장 (300 DPI)
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
    # PDF 저장 (벡터)
    fig.savefig(
        pdf_path,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
    if close:
        plt.close(fig)
    
    print(f"  ✅ 저장: {name}.png + .pdf")
    
    return png_path, pdf_path


# ============================================================
# Gate0 필수 Figure 생성 헬퍼
# ============================================================

def get_color(key: str) -> str:
    """색상 코드 반환"""
    return COLORS.get(key, COLORS['neutral'])


def get_marker(key: str) -> str:
    """마커 스타일 반환"""
    return MARKERS.get(key, 'o')


# ============================================================
# 테스트/검증
# ============================================================

if __name__ == "__main__":
    import numpy as np
    import tempfile
    import os
    
    print("=" * 60)
    print("plot_style.py 검증")
    print("=" * 60)
    
    # 1. 스타일 적용 테스트
    print("\n[스타일 적용 테스트]")
    setup_style()
    print("  ✅ setup_style() 성공")
    
    # 2. Figure 생성 테스트
    print("\n[Figure 생성 테스트]")
    fig, ax = create_figure('single')
    print(f"  Figure 크기: {fig.get_size_inches()}")
    
    # 3. 테스트 플롯 생성
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), color=get_color('proposed'), label='Proposed')
    ax.plot(x, np.cos(x), color=get_color('esindy'), label='E-SINDy')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Test Plot')
    ax.legend()
    
    # 4. 저장 테스트 (임시 디렉토리)
    print("\n[저장 테스트]")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        png_path, pdf_path = save_figure(fig, tmpdir, 'test_figure')
        
        print(f"  PNG 존재: {png_path.exists()}")
        print(f"  PDF 존재: {pdf_path.exists()}")
        print(f"  PNG 크기: {png_path.stat().st_size:,} bytes")
        print(f"  PDF 크기: {pdf_path.stat().st_size:,} bytes")
    
    # 5. 색상 팔레트 확인
    print("\n[색상 팔레트]")
    for name, color in list(COLORS.items())[:5]:
        print(f"  {name}: {color}")
    
    print("\n" + "=" * 60)
    print("✅ plot_style.py 검증 완료")
    print("=" * 60)