"""
Data generation and processing modules.
"""

from .data_generator import (
    CartPoleDataGenerator,
    generate_dataset,
)

from .normalization import (
    compute_stats,
    normalize,
    denormalize,
    compute_norm_stats,
    normalize_dataset,
    denormalize_dataset,
    save_norm_stats,
    load_norm_stats,
    generate_norm_stats_from_dataset,
    validate_normalization,
    validate_inverse,
)

__all__ = [
    # Data generation
    'CartPoleDataGenerator',
    'generate_dataset',
    # Normalization
    'compute_stats',
    'normalize',
    'denormalize',
    'compute_norm_stats',
    'normalize_dataset',
    'denormalize_dataset',
    'save_norm_stats',
    'load_norm_stats',
    'generate_norm_stats_from_dataset',
    'validate_normalization',
    'validate_inverse',
]