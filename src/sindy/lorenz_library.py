"""
Lorenz-63 SINDy Library Builder (lorenz_library.py)

Degree-2 polynomial library for Lorenz-63 system.
10 terms — minimum sufficient to exactly represent the true EOM.

Library terms:
    Index  Feature   In Oracle?
    ──────────────────────────────────
    0      1         No
    1      x         Yes (dx/dt, dy/dt)
    2      y         Yes (dx/dt, dy/dt)
    3      z         Yes (dz/dt)
    4      x^2       No
    5      y^2       No
    6      z^2       No
    7      xy        Yes (dz/dt: +xy)
    8      xz        Yes (dy/dt: -xz)
    9      yz        No
    ──────────────────────────────────

Oracle EOM (true Lorenz at rho=28, sigma=10, beta=8/3):
    dx/dt = sigma*(y - x)   = -10*x + 10*y
        → non-zero: x(1), y(2)

    dy/dt = rho*x - y - x*z = +28*x - y - xz
        → non-zero: x(1), y(2), xz(8)
        NOTE: coefficient of x(1) is rho — this changes with OOD rho!
              SINDy must correctly identify the magnitude, not just sign.

    dz/dt = x*y - beta*z     = xy - (8/3)*z
        → non-zero: z(3), xy(7)

No reparameterization needed (polynomial library, no trig collinearity).
κ is expected to be well-conditioned (no small-angle regime).

Oracle support array (10 features × 3 targets):
    Target: [dx/dt, dy/dt, dz/dt]
    Feature 1 (x):  [True,  True,  False]
    Feature 2 (y):  [True,  True,  False]
    Feature 3 (z):  [False, False, True ]
    Feature 7 (xy): [False, False, True ]
    Feature 8 (xz): [False, True,  False]
    All others: False

Fragile pair design (consistent with CP/AEK z-metric approach):
    Fragile pairs = feature indices where non-oracle terms appear
    (terms that should be zero but might be activated spuriously).
    Primary fragile: {0,4,5,6,9} × {0,1,2} = non-oracle terms

Usage:
    from src.sindy.lorenz_library import (
        build_lorenz_library,
        get_lorenz_feature_names,
        get_lorenz_oracle_support,
        N_LORENZ_FEATURES,
        LORENZ_TARGET_NAMES,
    )
    Theta, names = build_lorenz_library(x_flat)
"""

import numpy as np
from typing import Tuple, List


# =============================================================================
# Constants (SSOT)
# =============================================================================

N_LORENZ_FEATURES: int = 10

LORENZ_TARGET_NAMES: List[str] = [
    'd(x)/dt',   # 0: sigma*(y-x)
    'd(y)/dt',   # 1: rho*x - y - xz
    'd(z)/dt',   # 2: xy - beta*z
]

LORENZ_FEATURE_NAMES: List[str] = [
    '1',    # 0: constant
    'x',    # 1: oracle (dx/dt, dy/dt)
    'y',    # 2: oracle (dx/dt, dy/dt)
    'z',    # 3: oracle (dz/dt)
    'x^2',  # 4: non-oracle
    'y^2',  # 5: non-oracle
    'z^2',  # 6: non-oracle
    'xy',   # 7: oracle (dz/dt)
    'xz',   # 8: oracle (dy/dt)
    'yz',   # 9: non-oracle
]

assert len(LORENZ_FEATURE_NAMES) == N_LORENZ_FEATURES, \
    f"Feature name count mismatch: {len(LORENZ_FEATURE_NAMES)} != {N_LORENZ_FEATURES}"

# State indices
LORENZ_STATE_INDICES = {'x': 0, 'y': 1, 'z': 2}


# =============================================================================
# Oracle Support (SSOT — physical EOM, reparameterization-invariant)
# =============================================================================

def get_lorenz_oracle_support() -> np.ndarray:
    """
    Return oracle support mask for Lorenz system.

    Oracle support is based on the physical EOM structure —
    which terms are non-zero in the true dynamics.

    NOTE: coefficient of x in dy/dt (rho) changes with OOD rho,
    but the support (non-zero = True) is constant across rho values.
    This is correct: we are checking structural identification, not
    exact coefficient recovery.

    Returns:
        oracle: (10, 3) bool array
                oracle[i, j] = True means feature i is in oracle for target j
    """
    oracle = np.zeros((N_LORENZ_FEATURES, 3), dtype=bool)

    # dx/dt = sigma*(y - x) = -sigma*x + sigma*y
    oracle[1, 0] = True   # x → dx/dt
    oracle[2, 0] = True   # y → dx/dt

    # dy/dt = rho*x - y - x*z
    oracle[1, 1] = True   # x → dy/dt
    oracle[2, 1] = True   # y → dy/dt
    oracle[8, 1] = True   # xz → dy/dt

    # dz/dt = x*y - beta*z
    oracle[3, 2] = True   # z → dz/dt
    oracle[7, 2] = True   # xy → dz/dt

    return oracle


def get_lorenz_fragile_pairs(
    z_matrix: np.ndarray,
    z_threshold: float = 1.0,
    oracle_only: bool = False,
) -> List[List[int]]:
    """
    Identify fragile pairs from a z-metric matrix.

    Fragile pairs are (feature, target) pairs where the feature is
    in the oracle but has low z-score (recall fragility),
    OR where the feature is NOT in the oracle but has high z-score
    (precision collapse / spurious detection).

    For Lorenz (expected: recall fragility, like CP):
        Primary fragile = oracle terms with low z (recall fragility)
        z < z_threshold → fragile

    Args:
        z_matrix: (10, 3) z-metric matrix from baseline run
        z_threshold: z < threshold → fragile
        oracle_only: If True, only return oracle-term fragile pairs

    Returns:
        List of [feature_idx, target_idx] pairs
    """
    oracle = get_lorenz_oracle_support()
    pairs = []

    for f in range(N_LORENZ_FEATURES):
        for t in range(3):
            if oracle_only:
                if oracle[f, t] and z_matrix[f, t] < z_threshold:
                    pairs.append([f, t])
            else:
                # Include both recall (oracle, low z) and
                # precision (non-oracle, high z) fragile pairs
                if oracle[f, t] and z_matrix[f, t] < z_threshold:
                    pairs.append([f, t])
                elif not oracle[f, t] and z_matrix[f, t] >= z_threshold:
                    pairs.append([f, t])

    return pairs


# =============================================================================
# Library Builder
# =============================================================================

def build_lorenz_library(
    x: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build degree-2 polynomial library for Lorenz system.

    Args:
        x: State matrix, shape (N, 3) — flattened (time * trajectories)
           Columns: [x_state, y_state, z_state]

    Returns:
        Theta: (N, 10) feature matrix
        names: List of 10 feature names
    """
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected x shape (N, 3), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    N = x.shape[0]
    xs = x[:, 0]   # x state
    ys = x[:, 1]   # y state
    zs = x[:, 2]   # z state

    Theta = np.column_stack([
        np.ones(N),    # 0: 1
        xs,            # 1: x
        ys,            # 2: y
        zs,            # 3: z
        xs * xs,       # 4: x^2
        ys * ys,       # 5: y^2
        zs * zs,       # 6: z^2
        xs * ys,       # 7: xy (oracle: dz/dt)
        xs * zs,       # 8: xz (oracle: dy/dt)
        ys * zs,       # 9: yz
    ]).astype(np.float64)

    assert Theta.shape == (N, N_LORENZ_FEATURES), \
        f"Theta shape mismatch: {Theta.shape}"

    return Theta, list(LORENZ_FEATURE_NAMES)


def get_lorenz_feature_names() -> List[str]:
    """Return Lorenz feature names (SSOT copy)."""
    return list(LORENZ_FEATURE_NAMES)


def assert_lorenz_feature_integrity() -> None:
    """
    AC2 equivalent: verify feature index → name mapping is correct.

    Checks that oracle-critical features are at expected positions.
    Call at start of every runner.
    """
    names = get_lorenz_feature_names()

    expected = {
        0: '1',
        1: 'x',
        2: 'y',
        3: 'z',
        7: 'xy',
        8: 'xz',
    }
    for idx, name in expected.items():
        actual = names[idx]
        assert actual == name, \
            f"AC2 FAIL: lorenz feature idx {idx} expected '{name}', got '{actual}'"

    assert len(names) == N_LORENZ_FEATURES, \
        f"AC2 FAIL: expected {N_LORENZ_FEATURES} features, got {len(names)}"

    # Verify oracle support shape
    oracle = get_lorenz_oracle_support()
    assert oracle.shape == (N_LORENZ_FEATURES, 3), \
        f"AC2 FAIL: oracle shape {oracle.shape}"

    # Verify oracle non-zero count: 7 total oracle entries
    # x→dx, y→dx, x→dy, y→dy, xz→dy, z→dz, xy→dz = 7
    assert oracle.sum() == 7, \
        f"AC2 FAIL: expected 7 oracle entries, got {oracle.sum()}"


# =============================================================================
# Condition Number Diagnostic
# =============================================================================

def compute_lorenz_kappa(
    x: np.ndarray,
    scaler=None,
) -> float:
    """
    Compute condition number of (scaled) Lorenz library.

    Args:
        x: State matrix (N, 3)
        scaler: ColumnScaler instance (fit on training data only).
                If None, raw kappa is returned.

    Returns:
        Condition number kappa_2
    """
    Theta, _ = build_lorenz_library(x)
    if scaler is not None:
        Theta = scaler.transform(Theta)
    return float(np.linalg.cond(Theta))
