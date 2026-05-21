"""
Evaluation 모듈 패키지

포함 모듈:
- structure_eval: 구조 평가 (Phase 3.5)
"""

from .structure_eval import (
    StructureEvaluator,
    StructureEvalResult,
    DeltaFloorResult,
    compute_support_metrics
)

__all__ = [
    'StructureEvaluator',
    'StructureEvalResult',
    'DeltaFloorResult',
    'compute_support_metrics'
]