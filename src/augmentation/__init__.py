"""
Gate2: Data Augmentation Module

Physics-consistent augmentation for SINDy training data.
Tier-0 methods: IC jitter + re-simulation, param jitter + re-simulation

Usage:
    from src.augmentation import PhysicsAugmentor
    
    augmentor = PhysicsAugmentor(config)
    aug_x, aug_u, aug_dx, aug_params = augmentor.augment(
        original_x, original_u, original_params, ...
    )
"""

from .base import BaseAugmentor, AugmentationResult
from .physics_augmentor import PhysicsAugmentor

__all__ = [
    'BaseAugmentor',
    'AugmentationResult',
    'PhysicsAugmentor',
]