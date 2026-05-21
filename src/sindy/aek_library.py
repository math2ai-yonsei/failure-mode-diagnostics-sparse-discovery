"""
AEK SINDy Library Builder

Builds the feature matrix Theta for AEK Self-balancing Motorcycle.
14 library terms (aek.yaml post-GPT review 2026-02-06):

  Active terms:
    1, phi_dot, theta_w_dot, tau,
    sin(phi), cos(phi),
    phi_dot^2, theta_w_dot^2, tau^2,
    phi*phi_dot, phi*tau, phi_dot*tau,
    theta_w_dot*tau, sin(phi)*tau

  Removed (GPT P1 review):
    phi        — collinear with sin(phi) at ±3°
    phi^2      — not in oracle + negligible at small angles
    theta_w    — not in EOM + ±100 rad scale drift

Oracle EOM (non-zero terms per target):
    phi_ddot     = +(M*g*h/I_p)*sin(phi) - (1/I_p)*tau
    theta_w_ddot = -(M*g*h/I_p)*sin(phi) + (1/I_w_C + 1/I_p)*tau

Reparam-1 (Gate4c Phase 1):
    Index #5 only: cos(phi) → cos(phi)-1
    Breaks 1↔cos(phi) collinearity at small angles
    κ₂ improvement: 4.7e9 → 4.5e4 (Δlog₁₀=5.02)
    Oracle support/coefficients unchanged (cos(phi) not in oracle)

Reparam-2 (Gate4c Phase 2):
    Cumulative: RP1 (#5) + #13 cross-term replacement
    Index #5: cos(phi) → cos(phi)-1  (inherited from RP1)
    Index #13: sin(phi)*tau → (cos(phi)-1)*tau  (★ RP2 new)
    Breaks phi*tau ↔ sin(phi)*tau cancellation pair
    (phi*tau = O(φτ), (cos(phi)-1)*tau = O(φ²τ) → different polynomial order)
    Oracle support/coefficients unchanged (sin(phi)*tau not in oracle)
    RISK: (cos(phi)-1)*tau ≈ 9e-6 at |φ|<0.03 → near-zero column risk

Usage:
    from src.sindy.aek_library import build_aek_library_by_name
    Theta, names = build_aek_library_by_name(x_flat, u_flat, reparam='reparam2')

Location: src/sindy/aek_library.py
"""
import numpy as np
from typing import Tuple, List

# =============================================================================
# AEK Feature Names — Standard (SSOT — matches aek.yaml)
# =============================================================================

AEK_FEATURE_NAMES: List[str] = [
    '1',                # 0: constant
    'phi_dot',          # 1: lean angular velocity
    'theta_w_dot',      # 2: wheel angular velocity
    'tau',              # 3: motor torque
    'sin(phi)',         # 4: trig (oracle term!)
    'cos(phi)',         # 5: trig
    'phi_dot^2',        # 6: quadratic
    'theta_w_dot^2',    # 7: quadratic
    'tau^2',            # 8: quadratic
    'phi*phi_dot',      # 9: cross term (precision test — oracle should zero)
    'phi*tau',          # 10: cross term (precision test)
    'phi_dot*tau',      # 11: cross term
    'theta_w_dot*tau',  # 12: cross term
    'sin(phi)*tau',     # 13: cross term
]

# =============================================================================
# AEK Feature Names — Reparam-1 (Gate4c: cos(phi) → cos(phi)-1)
# =============================================================================

AEK_REPARAM1_FEATURE_NAMES: List[str] = [
    '1',                # 0: constant
    'phi_dot',          # 1
    'theta_w_dot',      # 2
    'tau',              # 3: Oracle — 변경 금지
    'sin(phi)',         # 4: Oracle — 변경 금지
    'cos(phi)-1',       # 5: ★ Reparam-1 핵심 변환
    'phi_dot^2',        # 6
    'theta_w_dot^2',    # 7
    'tau^2',            # 8
    'phi*phi_dot',      # 9
    'phi*tau',          # 10
    'phi_dot*tau',      # 11
    'theta_w_dot*tau',  # 12
    'sin(phi)*tau',     # 13
]

# =============================================================================
# AEK Feature Names — Reparam-2 (Gate4c Phase 2: RP1 + cross-term)
#   #5:  cos(phi) → cos(phi)-1       (inherited from RP1)
#   #13: sin(phi)*tau → (cos(phi)-1)*tau  (★ RP2 new)
# =============================================================================

AEK_REPARAM2_FEATURE_NAMES: List[str] = [
    '1',                # 0: constant
    'phi_dot',          # 1
    'theta_w_dot',      # 2
    'tau',              # 3: Oracle — 변경 금지
    'sin(phi)',         # 4: Oracle — 변경 금지
    'cos(phi)-1',       # 5: RP1 유지
    'phi_dot^2',        # 6
    'theta_w_dot^2',    # 7
    'tau^2',            # 8
    'phi*phi_dot',      # 9
    'phi*tau',          # 10
    'phi_dot*tau',      # 11
    'theta_w_dot*tau',  # 12
    '(cos(phi)-1)*tau', # 13: ★ Reparam-2 핵심 변환
]

N_AEK_FEATURES = 14

# AEK state indices
AEK_STATE_INDICES = {
    'phi': 0,
    'phi_dot': 1,
    'theta_w': 2,
    'theta_w_dot': 3,
}

# AEK target names (dx/dt components)
AEK_TARGET_NAMES: List[str] = [
    'd(phi)/dt',         # = phi_dot (kinematic)
    'd(phi_dot)/dt',     # = phi_ddot (dynamics)
    'd(theta_w)/dt',     # = theta_w_dot (kinematic)
    'd(theta_w_dot)/dt', # = theta_w_ddot (dynamics)
]


def build_aek_library(
    x: np.ndarray,
    u: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build AEK SINDy feature matrix — Standard (14 terms).

    Args:
        x: State array, shape (N_samples, 4)
           [phi, phi_dot, theta_w, theta_w_dot]
        u: Input array, shape (N_samples, 1)
           [tau]

    Returns:
        Theta: Feature matrix, shape (N_samples, 14)
        feature_names: List of 14 feature name strings
    """
    N = x.shape[0]
    if x.shape != (N, 4):
        raise ValueError(f"Expected x shape (N, 4), got {x.shape}")
    if u.shape != (N, 1):
        raise ValueError(f"Expected u shape (N, 1), got {u.shape}")

    phi = x[:, 0]
    phi_dot = x[:, 1]
    # theta_w = x[:, 2]  # NOT used (removed from library)
    theta_w_dot = x[:, 3]
    tau = u[:, 0]

    Theta = np.column_stack([
        np.ones(N),             # 0: 1
        phi_dot,                # 1: phi_dot
        theta_w_dot,            # 2: theta_w_dot
        tau,                    # 3: tau
        np.sin(phi),            # 4: sin(phi)
        np.cos(phi),            # 5: cos(phi)
        phi_dot ** 2,           # 6: phi_dot^2
        theta_w_dot ** 2,       # 7: theta_w_dot^2
        tau ** 2,               # 8: tau^2
        phi * phi_dot,          # 9: phi*phi_dot
        phi * tau,              # 10: phi*tau
        phi_dot * tau,          # 11: phi_dot*tau
        theta_w_dot * tau,      # 12: theta_w_dot*tau
        np.sin(phi) * tau,      # 13: sin(phi)*tau
    ])

    assert Theta.shape == (N, N_AEK_FEATURES), \
        f"Expected Theta shape (N, {N_AEK_FEATURES}), got {Theta.shape}"
    return Theta, list(AEK_FEATURE_NAMES)


def build_aek_reparam1_library(
    x: np.ndarray,
    u: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build AEK SINDy feature matrix — Reparam-1 (14 terms).

    Only index #5 changes: cos(phi) → cos(phi)-1.
    All other 13 features identical to Standard.

    Rationale (R1-0 diagnostic, 2026-02-10):
        At ±2.5° operating range, cos(phi) ≈ 1.0 causes near-perfect
        collinearity with the constant term '1'.
        cos(phi)-1 ≈ -phi²/2 breaks this collinearity.
        κ₂ improvement: 4.7e9 → 4.5e4 (Δlog₁₀ = 5.02).

    Oracle note:
        cos(phi) is NOT in the oracle EOM, so this reparameterization
        does not affect oracle support or oracle coefficients.

    Args:
        x: State array, shape (N_samples, 4)
        u: Input array, shape (N_samples, 1)

    Returns:
        Theta: Feature matrix, shape (N_samples, 14)
        feature_names: List of 14 feature name strings (with 'cos(phi)-1')
    """
    N = x.shape[0]
    if x.shape != (N, 4):
        raise ValueError(f"Expected x shape (N, 4), got {x.shape}")
    if u.shape != (N, 1):
        raise ValueError(f"Expected u shape (N, 1), got {u.shape}")

    phi = x[:, 0]
    phi_dot = x[:, 1]
    theta_w_dot = x[:, 3]
    tau = u[:, 0]

    Theta = np.column_stack([
        np.ones(N),             # 0: 1
        phi_dot,                # 1: phi_dot
        theta_w_dot,            # 2: theta_w_dot
        tau,                    # 3: tau
        np.sin(phi),            # 4: sin(phi)
        np.cos(phi) - 1.0,     # 5: cos(phi)-1  ★ REPARAM-1
        phi_dot ** 2,           # 6: phi_dot^2
        theta_w_dot ** 2,       # 7: theta_w_dot^2
        tau ** 2,               # 8: tau^2
        phi * phi_dot,          # 9: phi*phi_dot
        phi * tau,              # 10: phi*tau
        phi_dot * tau,          # 11: phi_dot*tau
        theta_w_dot * tau,      # 12: theta_w_dot*tau
        np.sin(phi) * tau,      # 13: sin(phi)*tau
    ])

    assert Theta.shape == (N, N_AEK_FEATURES), \
        f"Expected Theta shape (N, {N_AEK_FEATURES}), got {Theta.shape}"
    return Theta, list(AEK_REPARAM1_FEATURE_NAMES)


def build_aek_reparam2_library(
    x: np.ndarray,
    u: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build AEK SINDy feature matrix — Reparam-2 (14 terms).

    Cumulative reparameterization (RP1 + cross-term):
        #5:  cos(phi) → cos(phi)-1         (inherited from RP1)
        #13: sin(phi)*tau → (cos(phi)-1)*tau  (★ RP2 new)

    Rationale (Phase 2, 2026-03-03):
        Reparam-1 resolved 1↔cos(phi) collinearity (κ: 4.7e9 → 4.5e4),
        but phi*tau (#10) ↔ sin(phi)*tau (#13) cancellation pair persisted
        (corr > 0.999, teacher coeff ±1.2e8).

        Replacing sin(phi)*tau with (cos(phi)-1)*tau changes polynomial
        order: phi*tau = O(φτ), (cos(phi)-1)*tau = O(φ²τ), breaking
        the near-identity at small angles.

    Oracle note:
        sin(phi)*tau is NOT in the oracle EOM (verified: fragile_pairs.json
        shows #13 as spurious for both target 1 and target 3).
        Oracle support and coefficients are IDENTICAL across Standard,
        Reparam-1, and Reparam-2.

    Near-zero warning:
        (cos(phi)-1)*tau ≈ (-φ²/2)*tau ≈ 9e-6 at |φ|<0.03, |tau|<0.02.
        ColumnScaler may amplify ~1e5×. R2-0 fail-fast must verify
        std ≥ 1e-6 and scale_ratio ≤ 1e4 before proceeding.

    Args:
        x: State array, shape (N_samples, 4)
        u: Input array, shape (N_samples, 1)

    Returns:
        Theta: Feature matrix, shape (N_samples, 14)
        feature_names: List of 14 feature name strings
    """
    N = x.shape[0]
    if x.shape != (N, 4):
        raise ValueError(f"Expected x shape (N, 4), got {x.shape}")
    if u.shape != (N, 1):
        raise ValueError(f"Expected u shape (N, 1), got {u.shape}")

    phi = x[:, 0]
    phi_dot = x[:, 1]
    theta_w_dot = x[:, 3]
    tau = u[:, 0]

    cosm1 = np.cos(phi) - 1.0  # shared: used for both #5 and #13

    Theta = np.column_stack([
        np.ones(N),             # 0: 1
        phi_dot,                # 1: phi_dot
        theta_w_dot,            # 2: theta_w_dot
        tau,                    # 3: tau
        np.sin(phi),            # 4: sin(phi)
        cosm1,                  # 5: cos(phi)-1       ★ RP1 (inherited)
        phi_dot ** 2,           # 6: phi_dot^2
        theta_w_dot ** 2,       # 7: theta_w_dot^2
        tau ** 2,               # 8: tau^2
        phi * phi_dot,          # 9: phi*phi_dot
        phi * tau,              # 10: phi*tau
        phi_dot * tau,          # 11: phi_dot*tau
        theta_w_dot * tau,      # 12: theta_w_dot*tau
        cosm1 * tau,            # 13: (cos(phi)-1)*tau ★ RP2 (new)
    ])

    assert Theta.shape == (N, N_AEK_FEATURES), \
        f"Expected Theta shape (N, {N_AEK_FEATURES}), got {Theta.shape}"
    return Theta, list(AEK_REPARAM2_FEATURE_NAMES)


def get_aek_feature_names(reparam: str = 'standard') -> List[str]:
    """
    Return feature names for specified parameterization.

    Args:
        reparam: 'standard', 'reparam1', or 'reparam2'

    Returns:
        List of 14 feature name strings
    """
    if reparam == 'standard':
        return list(AEK_FEATURE_NAMES)
    elif reparam == 'reparam1':
        return list(AEK_REPARAM1_FEATURE_NAMES)
    elif reparam == 'reparam2':
        return list(AEK_REPARAM2_FEATURE_NAMES)
    else:
        raise ValueError(
            f"Unknown reparam: {reparam}. "
            f"Use 'standard', 'reparam1', or 'reparam2'"
        )


def build_aek_library_by_name(
    x: np.ndarray,
    u: np.ndarray,
    reparam: str = 'standard',
) -> Tuple[np.ndarray, List[str]]:
    """
    Build AEK library by parameterization name (dispatcher).

    Args:
        x: State array, shape (N_samples, 4)
        u: Input array, shape (N_samples, 1)
        reparam: 'standard', 'reparam1', or 'reparam2'

    Returns:
        Theta, feature_names
    """
    if reparam == 'standard':
        return build_aek_library(x, u)
    elif reparam == 'reparam1':
        return build_aek_reparam1_library(x, u)
    elif reparam == 'reparam2':
        return build_aek_reparam2_library(x, u)
    else:
        raise ValueError(
            f"Unknown reparam: {reparam}. "
            f"Use 'standard', 'reparam1', or 'reparam2'"
        )


def get_aek_oracle_support(n_features: int = N_AEK_FEATURES) -> np.ndarray:
    """
    Return oracle support mask for AEK EOM.

    Oracle EOM:
        phi_ddot     = (M*g*h/I_p)*sin(phi) - (1/I_p)*tau
        theta_w_ddot = -(M*g*h/I_p)*sin(phi) + (1/I_w_C + 1/I_p)*tau

    For 4 targets (dx/dt):
        target 0 (d_phi/dt = phi_dot):        phi_dot only → feature 1
        target 1 (d_phi_dot/dt = phi_ddot):   sin(phi), tau → features 4, 3
        target 2 (d_theta_w/dt = theta_w_dot): theta_w_dot only → feature 2
        target 3 (d_theta_w_dot/dt):          sin(phi), tau → features 4, 3

    Note: Oracle support is IDENTICAL for Standard, Reparam-1, and Reparam-2,
    because cos(phi), cos(phi)-1, sin(phi)*tau, and (cos(phi)-1)*tau are
    all NOT oracle terms.

    Returns:
        support: Boolean mask, shape (n_features, 4)
    """
    support = np.zeros((n_features, 4), dtype=bool)

    # Target 0: d(phi)/dt = phi_dot
    support[1, 0] = True    # phi_dot

    # Target 1: d(phi_dot)/dt = (M*g*h*sin(phi) - tau) / I_p
    support[4, 1] = True    # sin(phi)
    support[3, 1] = True    # tau

    # Target 2: d(theta_w)/dt = theta_w_dot
    support[2, 2] = True    # theta_w_dot

    # Target 3: d(theta_w_dot)/dt = tau/I_w_C - phi_ddot
    #   = -(M*g*h/I_p)*sin(phi) + (1/I_w_C + 1/I_p)*tau
    support[4, 3] = True    # sin(phi)
    support[3, 3] = True    # tau

    return support


def get_aek_oracle_coefficients(
    M_total: float,
    g: float,
    h_cm: float,
    I_p: float,
    I_w_C: float,
    n_features: int = N_AEK_FEATURES,
) -> np.ndarray:
    """
    Return oracle coefficient matrix for AEK EOM.

    Note: Oracle coefficients are IDENTICAL for Standard, Reparam-1,
    and Reparam-2, because cos(phi)/cos(phi)-1/sin(phi)*tau/(cos(phi)-1)*tau
    all have zero oracle coefficients.

    Args:
        M_total: Total mass (kg)
        g: Gravity (m/s^2)
        h_cm: COM height (m)
        I_p: Pivot inertia (kg*m^2)
        I_w_C: Wheel spin inertia (kg*m^2)

    Returns:
        coefficients: shape (n_features, 4) — oracle xi matrix
    """
    coeff = np.zeros((n_features, 4))

    Mgh = M_total * g * h_cm

    # Target 0: d(phi)/dt = 1.0 * phi_dot
    coeff[1, 0] = 1.0

    # Target 1: d(phi_dot)/dt = (Mgh/I_p)*sin(phi) - (1/I_p)*tau
    coeff[4, 1] = Mgh / I_p       # sin(phi)
    coeff[3, 1] = -1.0 / I_p      # tau

    # Target 2: d(theta_w)/dt = 1.0 * theta_w_dot
    coeff[2, 2] = 1.0

    # Target 3: d(theta_w_dot)/dt = -(Mgh/I_p)*sin(phi) + (1/I_w_C + 1/I_p)*tau
    coeff[4, 3] = -Mgh / I_p
    coeff[3, 3] = 1.0 / I_w_C + 1.0 / I_p

    return coeff