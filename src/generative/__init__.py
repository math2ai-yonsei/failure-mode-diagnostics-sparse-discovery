"""
Gate3 Generative Module
=======================
학습 기반 궤적 생성을 위한 모듈.

Components:
- vae.py: TrajectoryVAE 모델
- alignment.py: Teacher alignment 및 score 계산
- filtering.py: Sanity, dedup, align-score 필터링
"""

from .vae import TrajectoryVAE, VAEConfig
from .alignment import TeacherAlignment, compute_align_score
from .filtering import (
    SanityFilter,
    DedupFilter,
    AlignScoreFilter,
    FilteringResult,
    apply_filters
)

__all__ = [
    'TrajectoryVAE',
    'VAEConfig',
    'TeacherAlignment',
    'compute_align_score',
    'SanityFilter',
    'DedupFilter',
    'AlignScoreFilter',
    'FilteringResult',
    'apply_filters',
]