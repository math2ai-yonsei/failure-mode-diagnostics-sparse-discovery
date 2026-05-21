"""
src/contracts/__init__.py

SSOT + Preflight Guard 모듈
Gate0-1에서 경로/그림/데이터 규격 강제

Version: v3.2 (Lean Mode)
"""

from . import paths
from . import plot_style
from . import schema_dataset_lite

# 자주 사용하는 함수 직접 노출
from .paths import (
    ROOT,
    DATA_ROOT,
    RESULTS_ROOT,
    generate_run_id,
    get_results_dir,
    get_dataset_path,
    get_meta_path,
    get_context_packet_path,
)

from .plot_style import (
    setup_style,
    create_figure,
    save_figure,
    COLORS,
    MARKERS,
    FIG_SIZES,
)

from .schema_dataset_lite import (
    validate_dataset_lite,
    get_dataset_info,
)

__all__ = [
    # 모듈
    'paths',
    'plot_style',
    'schema_dataset_lite',
    
    # paths
    'ROOT',
    'DATA_ROOT',
    'RESULTS_ROOT',
    'generate_run_id',
    'get_results_dir',
    'get_dataset_path',
    'get_meta_path',
    'get_context_packet_path',
    
    # plot_style
    'setup_style',
    'create_figure',
    'save_figure',
    'COLORS',
    'MARKERS',
    'FIG_SIZES',
    
    # schema_dataset_lite
    'validate_dataset_lite',
    'get_dataset_info',
]