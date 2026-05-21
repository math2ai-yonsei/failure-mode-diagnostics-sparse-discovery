"""
Lynx-Hare (Lotka-Volterra) SINDy Library Builder.

Degree-2 polynomial library for 2-state LV system.
6 terms — minimum sufficient to represent true LV EOM.

State definition:
    index 0: H  (hare population, thousands)
    index 1: L  (lynx population, thousands)

No control input. u stored as zeros for schema compatibility.

Library terms:
    Index  Feature   In Oracle?
    ──────────────────────────────────
    0      1         No
    1      H         Yes (dH/dt)
    2      L         Yes (dL/dt)
    3      H^2       No
    4      HL        Yes (dH/dt, dL/dt)
    5      L^2       No
    ──────────────────────────────────

Oracle EOM (Lotka-Volterra):
    dH/dt = α*H  - β*H*L     → non-zero: H(1), HL(4)
    dL/dt = δ*H*L - γ*L      → non-zero: L(2), HL(4)

Oracle support (6 features × 2 targets):
    Target: [dH/dt, dL/dt]
    Feature 1 (H):  [True,  False]
    Feature 2 (L):  [False, True ]
    Feature 4 (HL): [True,  True ]
    All others: False

Oracle count: 4 total entries

Fragile pair convention (consistent with Lorenz/AEK):
    - Oracle terms with low z  → recall_fragility
    - Non-oracle terms with high z → precision_collapse
    Failure mode detected at runtime from fragile pair composition.

score_aligned convention:
    Determined at runtime from dominant failure mode.
    recall_fragility     → score_aligned = +delta_raw (like CP)
    precision_collapse   → score_aligned = −delta_raw (like AEK, Lorenz)

Data source:
    Hudson Bay lynx-hare fur trade records, 1900-1920.
    Elton & Nicholson (1942), MacLulich (1937).
    Same dataset as E-SINDy original paper (Fasel et al., 2022).

Author: Claude (Gate-LynxHare)
Date: 2026-03-09
"""

import numpy as np
from typing import List, Tuple


# =============================================================================
# Constants (SSOT)
# =============================================================================

N_LYNXHARE_FEATURES: int = 6

LYNXHARE_TARGET_NAMES: List[str] = [
    'd(H)/dt',  # 0: α*H - β*H*L
    'd(L)/dt',  # 1: δ*H*L - γ*L
]

LYNXHARE_FEATURE_NAMES: List[str] = [
    '1',    # 0: constant (non-oracle)
    'H',    # 1: oracle (dH/dt: α*H)
    'L',    # 2: oracle (dL/dt: -γ*L)
    'H^2',  # 3: non-oracle
    'HL',   # 4: oracle (dH/dt: -β*HL, dL/dt: δ*HL)
    'L^2',  # 5: non-oracle
]

assert len(LYNXHARE_FEATURE_NAMES) == N_LYNXHARE_FEATURES

# State indices
LYNXHARE_STATE_INDICES = {'H': 0, 'L': 1}


# =============================================================================
# Oracle Support (SSOT — physical LV EOM)
# =============================================================================

def get_lynxhare_oracle_support() -> np.ndarray:
    """
    Return oracle support mask for Lynx-Hare (Lotka-Volterra) system.

    Oracle based on structural EOM:
        dH/dt = α*H - β*H*L   → features: H(1), HL(4)
        dL/dt = δ*H*L - γ*L   → features: HL(4), L(2)

    Returns:
        oracle: (6, 2) bool array
                oracle[i, j] = True means feature i is oracle for target j
    """
    oracle = np.zeros((N_LYNXHARE_FEATURES, 2), dtype=bool)

    # dH/dt = α*H - β*H*L
    oracle[1, 0] = True   # H  → dH/dt (growth)
    oracle[4, 0] = True   # HL → dH/dt (predation)

    # dL/dt = δ*H*L - γ*L
    oracle[4, 1] = True   # HL → dL/dt (prey consumption)
    oracle[2, 1] = True   # L  → dL/dt (death)

    return oracle


# =============================================================================
# Fragile Pair Detection
# =============================================================================

def get_lynxhare_fragile_pairs(
    z_matrix: np.ndarray,
    z_threshold: float = 2.0,
) -> Tuple[List[List[int]], str]:
    """
    Identify fragile pairs and dominant failure mode.

    Fragile pairs:
        - Oracle terms with z < z_threshold  → recall fragility
        - Non-oracle terms with z >= z_threshold → precision collapse

    Args:
        z_matrix: (6, 2) z-metric matrix from baseline
        z_threshold: boundary for fragile/stable classification

    Returns:
        pairs: List of [feature_idx, target_idx]
        failure_mode: 'recall_fragility' or 'precision_collapse'
    """
    oracle = get_lynxhare_oracle_support()
    pairs = []
    n_oracle_fragile = 0
    n_spurious_fragile = 0

    for f in range(N_LYNXHARE_FEATURES):
        for t in range(2):
            if oracle[f, t] and z_matrix[f, t] < z_threshold:
                pairs.append([f, t])
                n_oracle_fragile += 1
            elif not oracle[f, t] and z_matrix[f, t] >= z_threshold:
                pairs.append([f, t])
                n_spurious_fragile += 1

    if n_oracle_fragile >= n_spurious_fragile:
        failure_mode = 'recall_fragility'
    else:
        failure_mode = 'precision_collapse'

    return pairs, failure_mode


# =============================================================================
# Library Builder
# =============================================================================

def build_lynxhare_library(
    x: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build degree-2 polynomial library for Lynx-Hare system.

    Args:
        x: State matrix, shape (N, 2) — flattened (time * trajectories)
           Columns: [H (hare), L (lynx)]

    Returns:
        Theta: (N, 6) feature matrix
        names: List of 6 feature names
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x shape (N, 2), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    N = x.shape[0]
    H = x[:, 0]
    L = x[:, 1]

    Theta = np.column_stack([
        np.ones(N),   # 0: 1
        H,            # 1: H  (oracle: dH/dt)
        L,            # 2: L  (oracle: dL/dt)
        H * H,        # 3: H^2
        H * L,        # 4: HL (oracle: dH/dt, dL/dt)
        L * L,        # 5: L^2
    ]).astype(np.float64)

    assert Theta.shape == (N, N_LYNXHARE_FEATURES), \
        f"Theta shape mismatch: {Theta.shape}"

    return Theta, list(LYNXHARE_FEATURE_NAMES)


def get_lynxhare_feature_names() -> List[str]:
    """Return Lynx-Hare feature names (SSOT copy)."""
    return list(LYNXHARE_FEATURE_NAMES)


def assert_lynxhare_feature_integrity() -> None:
    """
    AC2 equivalent: verify feature index → name mapping is correct.
    Call at start of every runner.
    """
    names = get_lynxhare_feature_names()

    expected = {
        0: '1',
        1: 'H',
        2: 'L',
        3: 'H^2',
        4: 'HL',
        5: 'L^2',
    }
    for idx, name in expected.items():
        actual = names[idx]
        assert actual == name, \
            f"AC2 FAIL: lynxhare feature idx {idx} expected '{name}', got '{actual}'"

    assert len(names) == N_LYNXHARE_FEATURES, \
        f"AC2 FAIL: expected {N_LYNXHARE_FEATURES} features, got {len(names)}"

    oracle = get_lynxhare_oracle_support()
    assert oracle.shape == (N_LYNXHARE_FEATURES, 2), \
        f"AC2 FAIL: oracle shape {oracle.shape}"

    # 4 oracle entries: H→dH, HL→dH, L→dL, HL→dL
    assert oracle.sum() == 4, \
        f"AC2 FAIL: expected 4 oracle entries, got {oracle.sum()}"


# =============================================================================
# Condition Number Diagnostic
# =============================================================================

def compute_lynxhare_kappa(
    x: np.ndarray,
    scaler=None,
) -> float:
    """Compute condition number of (scaled) Lynx-Hare library."""
    Theta, _ = build_lynxhare_library(x)
    if scaler is not None:
        Theta = scaler.transform(Theta)
    return float(np.linalg.cond(Theta))
