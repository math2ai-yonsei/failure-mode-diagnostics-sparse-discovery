"""
Experiment runners for Gate0-1 pipelines.

Gate0: SINDy baseline
Gate1: E-SINDy (planned)
"""

from .gate0_runner import Gate0Runner, Gate0Config

__all__ = [
    'Gate0Runner',
    'Gate0Config',
]