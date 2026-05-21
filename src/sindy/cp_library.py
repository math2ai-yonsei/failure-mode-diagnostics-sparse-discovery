"""
Cart-Pole SINDy Library Builder (cp_library.py)

Standalone library builder for Cart-Pole, analogous to aek_library.py.
21-term library with Standard and Reparam-1 variants.

Standard Library (21 terms):
    Index  Feature
    ─────────────────────────────────────────────
    0      1                    (constant)
    1      x                    (cart position)
    2      x_dot                (cart velocity)
    3      theta_dot            (pole angular vel)
    4      sin(theta)           (oracle term)
    5      cos(theta)           (near-constant near eq)
    6      u                    (force input)
    7      x^2
    8      x*x_dot
    9      x_dot^2
    10     theta_dot^2
    11     x*theta_dot
    12     x_dot*theta_dot
    13     x*sin(theta)
    14     x*cos(theta)
    15     x_dot*sin(theta)
    16     x_dot*cos(theta)
    17     theta_dot*sin(theta)
    18     theta_dot*cos(theta)
    19     u*sin(theta)
    20     u*cos(theta)
    ─────────────────────────────────────────────

Reparam-1 (Gate4d):
    5 features with cos(theta) → cos(theta)-1:
    Index 5:  cos(theta)           → cos(theta)-1
    Index 14: x*cos(theta)         → x*(cos(theta)-1)
    Index 16: x_dot*cos(theta)     → x_dot*(cos(theta)-1)
    Index 18: theta_dot*cos(theta) → theta_dot*(cos(theta)-1)
    Index 20: u*cos(theta)         → u*(cos(theta)-1)

    Rationale:
        Near equilibrium (theta≈0), cos(theta)≈1 → near-constant →
        multicollinearity with the constant term '1' → ill-conditioning.
        cos(theta)-1 → 0 at equilibrium → breaks collinearity.
        Analogy: AEK cos(phi)-1 gave Δlog₁₀(κ)=5.02.

Oracle EOM support (non-zero terms per target):
    d(x)/dt:           x_dot                        [kinematic]
    d(x_dot)/dt:       1, sin(theta), cos(theta), u, theta_dot^2*sin(theta)
                       (theta_dot^2*sin not in library → approximate)
    d(theta)/dt:       theta_dot                     [kinematic]
    d(theta_dot)/dt:   sin(theta), cos(theta), u

    Note: Oracle for dynamics targets uses sin(theta) and cos(theta).
    Reparam-1 replaces cos(theta) with cos(theta)-1, which changes
    the oracle coefficient for 'cos(theta)-1' terms.
    cos(theta) = (cos(theta)-1) + 1, so coefficients can be re-expressed.

Usage:
    from src.sindy.cp_library import (
        build_cp_library_by_name,
        get_cp_feature_names,
        N_CP_FEATURES,
        CP_TARGET_NAMES,
    )

    Theta, names = build_cp_library_by_name(x_flat, u_flat, reparam='standard')
    Theta_rp1, names_rp1 = build_cp_library_by_name(x_flat, u_flat, reparam='reparam1')

Location: src/sindy/cp_library.py
Date: 2026-03-04 (Gate4d)
"""
import numpy as np
from typing import Tuple, List

# =============================================================================
# Constants
# =============================================================================

N_CP_FEATURES: int = 21

CP_TARGET_NAMES: List[str] = [
    'd(x)/dt',          # 0: kinematic (= x_dot)
    'd(x_dot)/dt',      # 1: dynamics  (= x_ddot)
    'd(theta)/dt',      # 2: kinematic (= theta_dot)
    'd(theta_dot)/dt',  # 3: dynamics  (= theta_ddot)
]

# Reparam-1 cosine feature indices (0-based)
CP_REPARAM1_COS_INDICES: List[int] = [5, 14, 16, 18, 20]

# =============================================================================
# Feature Names — Standard
# =============================================================================

CP_FEATURE_NAMES_STANDARD: List[str] = [
    '1',                      # 0
    'x',                      # 1
    'x_dot',                  # 2
    'theta_dot',              # 3
    'sin(theta)',             # 4  oracle
    'cos(theta)',             # 5  ← Reparam-1 target
    'u',                      # 6  oracle
    'x^2',                    # 7
    'x*x_dot',                # 8
    'x_dot^2',                # 9
    'theta_dot^2',            # 10
    'x*theta_dot',            # 11
    'x_dot*theta_dot',        # 12
    'x*sin(theta)',           # 13
    'x*cos(theta)',           # 14 ← Reparam-1 target
    'x_dot*sin(theta)',       # 15
    'x_dot*cos(theta)',       # 16 ← Reparam-1 target
    'theta_dot*sin(theta)',   # 17
    'theta_dot*cos(theta)',   # 18 ← Reparam-1 target
    'u*sin(theta)',           # 19
    'u*cos(theta)',           # 20 ← Reparam-1 target
]

# =============================================================================
# Feature Names — Reparam-1
# =============================================================================

CP_FEATURE_NAMES_REPARAM1: List[str] = [
    '1',                         # 0  unchanged
    'x',                         # 1  unchanged
    'x_dot',                     # 2  unchanged
    'theta_dot',                 # 3  unchanged
    'sin(theta)',                # 4  unchanged (oracle)
    'cos(theta)-1',              # 5  ★ REPARAM-1
    'u',                         # 6  unchanged (oracle)
    'x^2',                       # 7  unchanged
    'x*x_dot',                   # 8  unchanged
    'x_dot^2',                   # 9  unchanged
    'theta_dot^2',               # 10 unchanged
    'x*theta_dot',               # 11 unchanged
    'x_dot*theta_dot',           # 12 unchanged
    'x*sin(theta)',              # 13 unchanged
    'x*(cos(theta)-1)',          # 14 ★ REPARAM-1
    'x_dot*sin(theta)',          # 15 unchanged
    'x_dot*(cos(theta)-1)',      # 16 ★ REPARAM-1
    'theta_dot*sin(theta)',      # 17 unchanged
    'theta_dot*(cos(theta)-1)', # 18 ★ REPARAM-1
    'u*sin(theta)',              # 19 unchanged
    'u*(cos(theta)-1)',          # 20 ★ REPARAM-1
]

assert len(CP_FEATURE_NAMES_STANDARD) == N_CP_FEATURES
assert len(CP_FEATURE_NAMES_REPARAM1) == N_CP_FEATURES

# =============================================================================
# Library Builders
# =============================================================================

def _extract_cp_states(x: np.ndarray, u: np.ndarray):
    """Extract named state components from flat (N, 4) arrays."""
    N = x.shape[0]
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"Expected x shape (N, 4), got {x.shape}")
    if u.ndim != 2 or u.shape[1] != 1:
        raise ValueError(f"Expected u shape (N, 1), got {u.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    if not np.all(np.isfinite(u)):
        raise ValueError("u contains non-finite values")
    return (
        N,
        x[:, 0],   # x (cart position)
        x[:, 1],   # x_dot
        x[:, 2],   # theta
        x[:, 3],   # theta_dot
        u[:, 0],   # F (force)
    )


def build_cp_standard_library(
    x: np.ndarray,
    u: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build CP SINDy feature matrix — Standard (21 terms).

    Args:
        x: State array, shape (N_samples, 4)
           [x, x_dot, theta, theta_dot]
        u: Input array, shape (N_samples, 1)
           [F (force)]

    Returns:
        Theta: Feature matrix, shape (N_samples, 21)
        feature_names: List of 21 feature name strings
    """
    N, xc, xd, th, thd, F = _extract_cp_states(x, u)
    sin_th = np.sin(th)
    cos_th = np.cos(th)

    Theta = np.column_stack([
        np.ones(N),           # 0: 1
        xc,                   # 1: x
        xd,                   # 2: x_dot
        thd,                  # 3: theta_dot
        sin_th,               # 4: sin(theta)
        cos_th,               # 5: cos(theta)
        F,                    # 6: u
        xc ** 2,              # 7: x^2
        xc * xd,              # 8: x*x_dot
        xd ** 2,              # 9: x_dot^2
        thd ** 2,             # 10: theta_dot^2
        xc * thd,             # 11: x*theta_dot
        xd * thd,             # 12: x_dot*theta_dot
        xc * sin_th,          # 13: x*sin(theta)
        xc * cos_th,          # 14: x*cos(theta)
        xd * sin_th,          # 15: x_dot*sin(theta)
        xd * cos_th,          # 16: x_dot*cos(theta)
        thd * sin_th,         # 17: theta_dot*sin(theta)
        thd * cos_th,         # 18: theta_dot*cos(theta)
        F * sin_th,           # 19: u*sin(theta)
        F * cos_th,           # 20: u*cos(theta)
    ])

    assert Theta.shape == (N, N_CP_FEATURES), \
        f"Shape mismatch: {Theta.shape} != ({N}, {N_CP_FEATURES})"
    return Theta, list(CP_FEATURE_NAMES_STANDARD)


def build_cp_reparam1_library(
    x: np.ndarray,
    u: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build CP SINDy feature matrix — Reparam-1 (21 terms).

    Replaces cos(theta) with cos(theta)-1 at indices 5, 14, 16, 18, 20.
    All other 16 features identical to Standard.

    Rationale:
        Near theta≈0 (equilibrium), cos(theta)≈1 is nearly identical
        to the constant term '1', causing severe collinearity.
        cos(theta)-1 → 0 at equilibrium, breaking this collinearity
        and improving condition number κ.

    Args:
        x: State array, shape (N_samples, 4)
        u: Input array, shape (N_samples, 1)

    Returns:
        Theta: Feature matrix, shape (N_samples, 21)
        feature_names: List of 21 feature name strings (with 'cos(theta)-1')
    """
    N, xc, xd, th, thd, F = _extract_cp_states(x, u)
    sin_th = np.sin(th)
    cos_th_m1 = np.cos(th) - 1.0   # ★ Reparam-1 kernel

    Theta = np.column_stack([
        np.ones(N),            # 0: 1                    (unchanged)
        xc,                    # 1: x                    (unchanged)
        xd,                    # 2: x_dot                (unchanged)
        thd,                   # 3: theta_dot             (unchanged)
        sin_th,                # 4: sin(theta)            (unchanged, oracle)
        cos_th_m1,             # 5: cos(theta)-1          ★ REPARAM-1
        F,                     # 6: u                    (unchanged, oracle)
        xc ** 2,               # 7: x^2                  (unchanged)
        xc * xd,               # 8: x*x_dot              (unchanged)
        xd ** 2,               # 9: x_dot^2              (unchanged)
        thd ** 2,              # 10: theta_dot^2          (unchanged)
        xc * thd,              # 11: x*theta_dot          (unchanged)
        xd * thd,              # 12: x_dot*theta_dot      (unchanged)
        xc * sin_th,           # 13: x*sin(theta)         (unchanged)
        xc * cos_th_m1,        # 14: x*(cos(theta)-1)     ★ REPARAM-1
        xd * sin_th,           # 15: x_dot*sin(theta)     (unchanged)
        xd * cos_th_m1,        # 16: x_dot*(cos(theta)-1) ★ REPARAM-1
        thd * sin_th,          # 17: theta_dot*sin(theta) (unchanged)
        thd * cos_th_m1,       # 18: theta_dot*(cos(theta)-1) ★ REPARAM-1
        F * sin_th,            # 19: u*sin(theta)         (unchanged)
        F * cos_th_m1,         # 20: u*(cos(theta)-1)     ★ REPARAM-1
    ])

    assert Theta.shape == (N, N_CP_FEATURES), \
        f"Shape mismatch: {Theta.shape} != ({N}, {N_CP_FEATURES})"
    return Theta, list(CP_FEATURE_NAMES_REPARAM1)


def build_cp_library_by_name(
    x: np.ndarray,
    u: np.ndarray,
    reparam: str = 'standard',
) -> Tuple[np.ndarray, List[str]]:
    """
    Dispatch CP library builder by reparam name.

    Args:
        x: State array, shape (N_samples, 4)
        u: Input array, shape (N_samples, 1)
        reparam: 'standard' or 'reparam1'

    Returns:
        Theta: Feature matrix, shape (N_samples, 21)
        feature_names: List of 21 feature name strings
    """
    if reparam == 'standard':
        return build_cp_standard_library(x, u)
    elif reparam == 'reparam1':
        return build_cp_reparam1_library(x, u)
    else:
        raise ValueError(
            f"Unknown reparam: '{reparam}'. Valid: 'standard', 'reparam1'"
        )


def get_cp_feature_names(reparam: str = 'standard') -> List[str]:
    """Return feature names for the specified parameterization."""
    if reparam == 'standard':
        return list(CP_FEATURE_NAMES_STANDARD)
    elif reparam == 'reparam1':
        return list(CP_FEATURE_NAMES_REPARAM1)
    else:
        raise ValueError(f"Unknown reparam: '{reparam}'")


# =============================================================================
# AC2: Feature Integrity Check
# =============================================================================

def assert_cp_feature_integrity(reparam: str) -> None:
    """
    AC2: Verify feature names match expected pattern.
    Fail fast if library order has been altered.

    Checks:
        - Total feature count == 21
        - Index 0 == '1'
        - Index 4 == 'sin(theta)' (oracle — must not change)
        - Index 5 changes correctly per reparam
        - Index 6 == 'u' (oracle — must not change)
    """
    names = get_cp_feature_names(reparam)

    assert len(names) == N_CP_FEATURES, \
        f"AC2 FAIL: Expected {N_CP_FEATURES} features, got {len(names)}"
    assert names[0] == '1', \
        f"AC2 FAIL: idx 0 expected '1', got '{names[0]}'"
    assert names[4] == 'sin(theta)', \
        f"AC2 FAIL: idx 4 expected 'sin(theta)', got '{names[4]}'"
    assert names[6] == 'u', \
        f"AC2 FAIL: idx 6 expected 'u', got '{names[6]}'"

    if reparam == 'standard':
        assert names[5] == 'cos(theta)', \
            f"AC2 FAIL: Standard idx 5 expected 'cos(theta)', got '{names[5]}'"
    elif reparam == 'reparam1':
        assert names[5] == 'cos(theta)-1', \
            f"AC2 FAIL: Reparam1 idx 5 expected 'cos(theta)-1', got '{names[5]}'"
        # Verify all 5 reparam positions
        rp1_expected = {
            5:  'cos(theta)-1',
            14: 'x*(cos(theta)-1)',
            16: 'x_dot*(cos(theta)-1)',
            18: 'theta_dot*(cos(theta)-1)',
            20: 'u*(cos(theta)-1)',
        }
        for idx, exp in rp1_expected.items():
            assert names[idx] == exp, \
                f"AC2 FAIL: Reparam1 idx {idx} expected '{exp}', got '{names[idx]}'"

    print(f"  [AC2] CP library integrity: PASS (reparam='{reparam}', n={len(names)})")


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == '__main__':
    print("cp_library.py self-test")
    print("=" * 50)

    N = 100
    rng = np.random.default_rng(42)
    x_test = rng.normal(0, 0.1, (N, 4))
    u_test = rng.normal(0, 1.0, (N, 1))

    # Standard
    assert_cp_feature_integrity('standard')
    Th_std, names_std = build_cp_library_by_name(x_test, u_test, 'standard')
    print(f"  Standard: {Th_std.shape}, kappa={np.linalg.cond(Th_std):.3e}")

    # Reparam-1
    assert_cp_feature_integrity('reparam1')
    Th_rp1, names_rp1 = build_cp_library_by_name(x_test, u_test, 'reparam1')
    print(f"  Reparam1: {Th_rp1.shape}, kappa={np.linalg.cond(Th_rp1):.3e}")

    # Verify only reparam indices differ
    diff_indices = [i for i in range(N_CP_FEATURES)
                    if not np.allclose(Th_std[:, i], Th_rp1[:, i])]
    print(f"  Differing indices: {diff_indices} (expected {CP_REPARAM1_COS_INDICES})")
    assert diff_indices == CP_REPARAM1_COS_INDICES, \
        f"Unexpected diffs: {diff_indices}"

    print("\nFeature name comparison (cos-related):")
    for i in CP_REPARAM1_COS_INDICES:
        print(f"  [{i:2d}] {names_std[i]:30s} → {names_rp1[i]}")

    print("\nSelf-test PASS")
