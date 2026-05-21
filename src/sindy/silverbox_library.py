"""
Silverbox SINDy Library Builder (silverbox_library.py)

8-term Duffing-informed library for the Silverbox electronic benchmark.
State: [x1=y, x2=dy/dt] (output voltage and its velocity estimate)
Input: [u] (excitation voltage)

Library terms:
    Index  Feature    In Oracle? (dx1)  In Oracle? (dx2)
    ────────────────────────────────────────────────────
    0      1          No                No
    1      x1         No                Yes  (Duffing: k·x1)
    2      x2         Yes               Yes  (kinematic + damping: c·x2)
    3      x1²        No                No
    4      x1·x2      No                No
    5      x2²        No                No
    6      x1³        No                Yes  (Duffing: k3·x1³)
    7      u          No                Yes  (input: b·u)
    ────────────────────────────────────────────────────

Oracle EOM (fitted Duffing at k=−93283, k3=−44580, c=−2.95, b=3100):
    dx1/dt = x2
        → non-zero: x2 (idx 2)

    dx2/dt = k*x1 + k3*x1³ + c*x2 + b*u
        → non-zero: x1(1), x2(2), x1³(6), u(7)

NOTE: x2 (idx 2) is oracle for BOTH targets:
    - dx1/dt = x2 is the kinematic identity (always well-identified)
    - dx2/dt has c*x2 damping term

Oracle support array (8 features × 2 targets):
    Target: [dx1/dt, dx2/dt]
    Feature 2 (x2):  [True,  True ]   ← appears in both
    Feature 1 (x1):  [False, True ]
    Feature 6 (x1³): [False, True ]
    Feature 7 (u):   [False, True ]
    All others: False

Total oracle entries: 1 + 4 = 5 (1 for dx1, 4 for dx2)

Usage:
    from src.sindy.silverbox_library import (
        build_silverbox_library,
        get_silverbox_feature_names,
        get_silverbox_oracle_support,
        N_SILVERBOX_FEATURES,
        SILVERBOX_TARGET_NAMES,
    )
    Theta, names = build_silverbox_library(x_flat, u_flat)
    # x_flat: (N, 2), u_flat: (N, 1)

Reference:
    Wigren & Schoukens (2013). Three free data sets for development and
    benchmarking in nonlinear system identification. ECC 2013.
    Fasel et al. (2022). Ensemble-SINDy. Proc. R. Soc. A.
"""

import numpy as np
from typing import Tuple, List


# =============================================================================
# Constants (SSOT)
# =============================================================================

N_SILVERBOX_FEATURES: int = 8

SILVERBOX_TARGET_NAMES: List[str] = [
    'd(x1)/dt',   # 0: kinematic (= x2)
    'd(x2)/dt',   # 1: Duffing dynamics
]

SILVERBOX_FEATURE_NAMES: List[str] = [
    '1',       # 0: constant
    'x1',      # 1: oracle (dx2/dt)
    'x2',      # 2: oracle (dx1/dt and dx2/dt)
    'x1^2',    # 3: non-oracle
    'x1*x2',   # 4: non-oracle
    'x2^2',    # 5: non-oracle
    'x1^3',    # 6: oracle (dx2/dt)
    'u',       # 7: oracle (dx2/dt)
]

assert len(SILVERBOX_FEATURE_NAMES) == N_SILVERBOX_FEATURES, \
    f"Feature name count mismatch: {len(SILVERBOX_FEATURE_NAMES)} != {N_SILVERBOX_FEATURES}"

# State and input indices (SSOT)
SILVERBOX_STATE_INDICES = {'x1': 0, 'x2': 1}
SILVERBOX_INPUT_INDICES = {'u': 0}


# =============================================================================
# Oracle Support (SSOT — physical EOM, parameter-invariant structure)
# =============================================================================

def get_silverbox_oracle_support() -> np.ndarray:
    """
    Return oracle support mask for Silverbox system.

    Oracle is based on physical Duffing EOM structure.
    NOTE: We check structural identification (which terms are non-zero),
    not exact coefficient recovery. The oracle support is independent of
    the specific Duffing parameter values (k, k3, c, b).

    Returns:
        oracle: (8, 2) bool array
                oracle[i, j] = True means feature i is in oracle for target j
    """
    oracle = np.zeros((N_SILVERBOX_FEATURES, 2), dtype=bool)

    # dx1/dt = x2  (kinematic identity)
    oracle[2, 0] = True   # x2 → dx1/dt

    # dx2/dt = k*x1 + k3*x1³ + c*x2 + b*u  (Duffing EOM)
    oracle[1, 1] = True   # x1 → dx2/dt  (linear stiffness)
    oracle[2, 1] = True   # x2 → dx2/dt  (damping)
    oracle[6, 1] = True   # x1³ → dx2/dt (cubic stiffness)
    oracle[7, 1] = True   # u → dx2/dt   (input forcing)

    return oracle


def get_silverbox_fragile_pairs(
    z_matrix: np.ndarray,
    z_threshold: float = 1.0,
    oracle_only: bool = False,
) -> List[List[int]]:
    """
    Identify fragile pairs from a z-metric matrix.

    For Silverbox (expected: precision_collapse like AEK/Lorenz):
        Primary fragile = non-oracle terms with high z-score (spurious detection)
        z >= z_threshold AND non-oracle → fragile (precision collapse)

    NOTE: dx1/dt = x2 is a kinematic identity and is always well-identified.
    Fragile pairs are almost exclusively in the dx2/dt target.

    Args:
        z_matrix: (8, 2) z-metric matrix from baseline E-SINDy run
        z_threshold: z >= threshold for non-oracle → spurious fragile
                     z < threshold for oracle → recall fragile
        oracle_only: If True, only return oracle-term fragile pairs

    Returns:
        List of [feature_idx, target_idx] pairs
    """
    oracle = get_silverbox_oracle_support()
    pairs = []

    for f in range(N_SILVERBOX_FEATURES):
        for t in range(2):
            if oracle_only:
                if oracle[f, t] and z_matrix[f, t] < z_threshold:
                    pairs.append([f, t])
            else:
                if oracle[f, t] and z_matrix[f, t] < z_threshold:
                    # Recall fragility (oracle term missed)
                    pairs.append([f, t])
                elif not oracle[f, t] and z_matrix[f, t] >= z_threshold:
                    # Precision collapse (spurious term selected)
                    pairs.append([f, t])

    return pairs


# =============================================================================
# Library Builder
# =============================================================================

def build_silverbox_library(
    x: np.ndarray,
    u: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build 8-term Duffing-informed library for Silverbox system.

    Terms: {1, x1, x2, x1², x1·x2, x2², x1³, u}

    Args:
        x: State matrix, shape (N, 2) — flattened (time * trajectories)
           Column 0: x1 = y (output voltage, V)
           Column 1: x2 = dy/dt (velocity estimate, V/s)
        u: Input matrix, shape (N, 1) — excitation voltage (V)

    Returns:
        Theta: (N, 8) feature matrix
        names: List of 8 feature names
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x shape (N, 2), got {x.shape}")
    if u.ndim != 2 or u.shape[1] != 1:
        raise ValueError(f"Expected u shape (N, 1), got {u.shape}")
    if x.shape[0] != u.shape[0]:
        raise ValueError(
            f"x and u must have same number of rows: {x.shape[0]} vs {u.shape[0]}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    if not np.all(np.isfinite(u)):
        raise ValueError("u contains non-finite values")

    N = x.shape[0]
    x1s = x[:, 0].astype(np.float64)   # output voltage
    x2s = x[:, 1].astype(np.float64)   # velocity
    us  = u[:, 0].astype(np.float64)   # excitation

    Theta = np.column_stack([
        np.ones(N),         # 0: constant
        x1s,                # 1: x1
        x2s,                # 2: x2
        x1s * x1s,          # 3: x1²
        x1s * x2s,          # 4: x1·x2
        x2s * x2s,          # 5: x2²
        x1s * x1s * x1s,    # 6: x1³  (Duffing cubic term)
        us,                 # 7: u    (input forcing)
    ]).astype(np.float64)

    assert Theta.shape == (N, N_SILVERBOX_FEATURES), \
        f"Theta shape mismatch: {Theta.shape}"

    return Theta, list(SILVERBOX_FEATURE_NAMES)


def get_silverbox_feature_names() -> List[str]:
    """Return Silverbox feature names (SSOT copy)."""
    return list(SILVERBOX_FEATURE_NAMES)


def assert_silverbox_feature_integrity() -> None:
    """
    AC2 equivalent: verify feature index → name mapping is correct.

    Checks that oracle-critical features are at expected positions.
    Call at start of every runner.
    """
    names = get_silverbox_feature_names()

    expected = {
        0: '1',
        1: 'x1',
        2: 'x2',
        3: 'x1^2',
        6: 'x1^3',
        7: 'u',
    }
    for idx, name in expected.items():
        actual = names[idx]
        assert actual == name, \
            f"AC2 FAIL: silverbox feature idx {idx} expected '{name}', got '{actual}'"

    assert len(names) == N_SILVERBOX_FEATURES, \
        f"AC2 FAIL: expected {N_SILVERBOX_FEATURES} features, got {len(names)}"

    # Verify oracle support shape
    oracle = get_silverbox_oracle_support()
    assert oracle.shape == (N_SILVERBOX_FEATURES, 2), \
        f"AC2 FAIL: oracle shape {oracle.shape}"

    # Verify oracle non-zero count:
    # x2→dx1(1), x1→dx2(1), x2→dx2(1), x1³→dx2(1), u→dx2(1) = 5 total
    assert oracle.sum() == 5, \
        f"AC2 FAIL: expected 5 oracle entries, got {oracle.sum()}"

    # Verify kinematic identity: only x2 in dx1/dt oracle
    assert oracle[:, 0].sum() == 1, \
        f"AC2 FAIL: dx1/dt should have 1 oracle term, got {oracle[:, 0].sum()}"
    assert oracle[2, 0] == True, \
        "AC2 FAIL: x2 must be the sole oracle term for dx1/dt"

    # Verify dx2/dt oracle has 4 terms
    assert oracle[:, 1].sum() == 4, \
        f"AC2 FAIL: dx2/dt should have 4 oracle terms, got {oracle[:, 1].sum()}"


# =============================================================================
# Condition Number Diagnostic
# =============================================================================

def compute_silverbox_kappa(
    x: np.ndarray,
    u: np.ndarray,
    scaler=None,
) -> float:
    """
    Compute condition number of (scaled) Silverbox library.

    IMPORTANT: Without ColumnScaler, kappa is expected to be extremely large
    due to ~300x scale difference between x1 (V) and x2 (V/s).
    Always pass a trained ColumnScaler for meaningful kappa.

    Args:
        x: State matrix (N, 2)
        u: Input matrix (N, 1)
        scaler: ColumnScaler instance (fit on training data only).
                If None, raw kappa is returned (expected very large).

    Returns:
        Condition number kappa_2
    """
    Theta, _ = build_silverbox_library(x, u)
    if scaler is not None:
        Theta = scaler.transform(Theta)
    return float(np.linalg.cond(Theta))
