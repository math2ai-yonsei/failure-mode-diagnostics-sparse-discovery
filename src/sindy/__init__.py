"""
SINDy (Sparse Identification of Nonlinear Dynamics) modules.

Gate0-1 components:
- library: Feature library construction (S04-B)
- optimizer: STLSQ sparse regression (S05)

Phase 3.5 components:
- core_mining: Stable-core / Fragile-pool extraction
"""

from .library import (
    SINDyLibrary,
    LIBRARY_CONFIGS,
    STATE_INDICES,
    STATE_NAMES,
    build_library_matrix,
    get_derivative_key,
    get_library_manifest,
)

from .optimizer import (
    ColumnScaler,
    STLSQOptimizer,
    STLSQ_CONFIGS,
    TARGET_NAMES,
    save_coefficients_csv,
    load_coefficients_csv,
    get_optimizer_manifest,
)

from .core_mining import (
    StableCoreMiner,
    CoreMiningResult,
    validate_against_qc2,
)

__all__ = [
    # Library (S04-B)
    'SINDyLibrary',
    'LIBRARY_CONFIGS',
    'STATE_INDICES',
    'STATE_NAMES',
    'build_library_matrix',
    'get_derivative_key',
    'get_library_manifest',
    # Optimizer (S05)
    'ColumnScaler',
    'STLSQOptimizer',
    'STLSQ_CONFIGS',
    'TARGET_NAMES',
    'save_coefficients_csv',
    'load_coefficients_csv',
    'get_optimizer_manifest',
    # Core Mining (Phase 3.5)
    'StableCoreMiner',
    'CoreMiningResult',
    'validate_against_qc2',
]