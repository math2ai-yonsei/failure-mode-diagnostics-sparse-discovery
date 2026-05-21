"""
S03: Derivative Computation Utilities

Provides Savitzky-Golay based numerical differentiation with proper
handling of angular variables (theta wrap).

Usage:
    from src.utils.derivatives import compute_derivatives_savgol
    dx = compute_derivatives_savgol(x, dt, theta_idx=2)
"""
import numpy as np
from scipy.signal import savgol_filter
from typing import Optional, Dict, Tuple

# =============================================================================
# Configuration (Fixed for Gate0-1)
# =============================================================================

SAVGOL_CONFIG = {
    'window': 11,
    'polyorder': 3
}

# =============================================================================
# Angle Utilities
# =============================================================================

def unwrap_angle(theta: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Unwrap angle to continuous values (no jumps at ±π).
    
    Args:
        theta: Angle array in radians, any shape
        axis: Axis along which to unwrap (default: last axis, typically time)
    
    Returns:
        Unwrapped angle (may exceed [-π, π])
    """
    return np.unwrap(theta, axis=axis)


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """
    Wrap angle to (-π, π].
    
    Args:
        theta: Angle array in radians
    
    Returns:
        Wrapped angle in (-π, π]
    """
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


# =============================================================================
# Derivative Computation
# =============================================================================

def savgol_derivative(
    x: np.ndarray,
    dt: float,
    window: int = SAVGOL_CONFIG['window'],
    polyorder: int = SAVGOL_CONFIG['polyorder'],
    axis: int = 1
) -> np.ndarray:
    """
    Compute derivative using Savitzky-Golay filter.
    
    Args:
        x: Input array, shape (..., T, ...) where T is time axis
        dt: Time step
        window: Filter window length (must be odd)
        polyorder: Polynomial order for fitting
        axis: Time axis (default: 1 for (N, T, D) arrays)
    
    Returns:
        dx: Derivative array, same shape as x
    """
    if window % 2 == 0:
        raise ValueError(f"Window must be odd, got {window}")
    if polyorder >= window:
        raise ValueError(f"polyorder ({polyorder}) must be < window ({window})")
    
    T = x.shape[axis]
    if T < window:
        raise ValueError(
            f"Time dimension ({T}) must be >= window ({window}). "
            f"Consider reducing window size or using central_difference."
        )
    
    return savgol_filter(x, window, polyorder, deriv=1, delta=dt, axis=axis)


def central_difference(
    x: np.ndarray,
    dt: float,
    axis: int = 1
) -> np.ndarray:
    """
    Compute derivative using central difference.
    
    Fallback method when trajectory is too short for Savgol.
    Uses forward/backward difference at boundaries.
    
    Args:
        x: Input array
        dt: Time step
        axis: Time axis
    
    Returns:
        dx: Derivative array, same shape as x
    """
    return np.gradient(x, dt, axis=axis)


def compute_derivatives_savgol(
    x: np.ndarray,
    dt: float,
    theta_idx: Optional[int] = None,
    window: int = SAVGOL_CONFIG['window'],
    polyorder: int = SAVGOL_CONFIG['polyorder']
) -> np.ndarray:
    """
    Compute state derivatives with proper angular handling.
    
    For Cart-Pole: x = [x, x_dot, theta, theta_dot]
    - Regular states: direct Savgol derivative
    - Angular states (theta): unwrap → derivative → (result is theta_dot_numerical)
    
    Args:
        x: State trajectory, shape (N, T, D) or (T, D)
        dt: Time step
        theta_idx: Index of angular variable (None = no angular handling)
        window: Savgol window length
        polyorder: Savgol polynomial order
    
    Returns:
        dx: State derivatives, shape same as x
            For Cart-Pole: [x_dot, x_ddot, theta_dot, theta_ddot]
    """
    single_traj = x.ndim == 2
    if single_traj:
        x = x[np.newaxis, ...]  # (1, T, D)
    
    N, T, D = x.shape
    dx = np.zeros_like(x)
    
    # Check if we can use Savgol
    use_savgol = T >= window
    
    for d in range(D):
        x_d = x[..., d]  # (N, T)
        
        # Handle angular variable
        if theta_idx is not None and d == theta_idx:
            x_d = unwrap_angle(x_d, axis=1)
        
        # Compute derivative
        if use_savgol:
            dx[..., d] = savgol_derivative(x_d, dt, window, polyorder, axis=1)
        else:
            dx[..., d] = central_difference(x_d, dt, axis=1)
    
    if single_traj:
        dx = dx[0]
    
    return dx


def compute_derivatives_batch(
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    theta_idx: int = 2,
    include_analytic: bool = False,
    dynamics_fn: Optional[callable] = None
) -> Dict[str, np.ndarray]:
    """
    Compute derivatives for entire dataset with multiple methods.
    
    Args:
        x: State trajectories, shape (N, T, D)
        u: Input trajectories, shape (N, T, 1)
        dt: Time step
        theta_idx: Index of angular variable
        include_analytic: If True, also compute analytic derivatives
        dynamics_fn: Function(state, input) -> dx for analytic derivatives
    
    Returns:
        Dict with:
            'dx_savgol': Savgol derivatives (N, T, D)
            'dx_central': Central difference derivatives (N, T, D)
            'dx_analytic': Analytic derivatives (only if include_analytic=True)
    """
    result = {}
    
    # Savgol derivatives
    result['dx_savgol'] = compute_derivatives_savgol(
        x, dt, theta_idx=theta_idx
    )
    
    # Central difference for comparison
    x_unwrap = x.copy()
    if theta_idx is not None:
        x_unwrap[..., theta_idx] = unwrap_angle(x[..., theta_idx], axis=1)
    result['dx_central'] = central_difference(x_unwrap, dt, axis=1)
    
    # Analytic derivatives (optional)
    if include_analytic and dynamics_fn is not None:
        N, T, D = x.shape
        dx_analytic = np.zeros_like(x)
        for i in range(N):
            for t in range(T):
                dx_analytic[i, t] = dynamics_fn(x[i, t], u[i, t, 0])
        result['dx_analytic'] = dx_analytic
    
    return result


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_derivatives(
    dx: np.ndarray,
    dx_ref: np.ndarray,
    state_names: Optional[list] = None,
    rtol: float = 0.1,
    atol: float = 0.1
) -> Dict[str, float]:
    """
    Compare computed derivatives against reference.
    
    Args:
        dx: Computed derivatives (N, T, D)
        dx_ref: Reference derivatives (N, T, D)
        state_names: Names for each state dimension
        rtol: Relative tolerance for "close" comparison
        atol: Absolute tolerance
    
    Returns:
        Dict with per-dimension metrics:
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - max_error: Maximum absolute error
            - r2: R-squared correlation
    """
    D = dx.shape[-1]
    if state_names is None:
        state_names = [f'dim_{i}' for i in range(D)]
    
    metrics = {}
    
    for d, name in enumerate(state_names):
        diff = dx[..., d] - dx_ref[..., d]
        ref = dx_ref[..., d]
        
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        max_err = np.abs(diff).max()
        
        # R-squared
        ss_res = np.sum(diff**2)
        ss_tot = np.sum((ref - ref.mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        metrics[name] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'max_error': float(max_err),
            'r2': float(r2)
        }
    
    return metrics


def check_wrap_boundary_issues(
    x: np.ndarray,
    dx: np.ndarray,
    theta_idx: int = 2,
    threshold: float = 10.0
) -> Dict[str, any]:
    """
    Check for potential wrap boundary issues in derivatives.
    
    Large spikes in theta_dot derivative near ±π indicate improper handling.
    
    Args:
        x: State trajectories (N, T, D)
        dx: Derivative trajectories (N, T, D)
        theta_idx: Index of angular variable
        threshold: Spike detection threshold (rad/s²)
    
    Returns:
        Dict with diagnostic info
    """
    theta = x[..., theta_idx]
    theta_dot_deriv = dx[..., theta_idx + 1] if theta_idx + 1 < dx.shape[-1] else None
    
    # Find points near wrap boundary
    near_boundary = np.abs(np.abs(theta) - np.pi) < 0.2  # within 0.2 rad of ±π
    
    # Check for spikes
    theta_ddot = np.diff(dx[..., theta_idx], axis=1)
    spike_mask = np.abs(theta_ddot) > threshold
    
    # Combine: spikes near boundary = potential issue
    if spike_mask.any() and near_boundary[:, 1:].any():
        problematic = spike_mask & near_boundary[:, 1:]
        n_issues = problematic.sum()
    else:
        n_issues = 0
    
    return {
        'n_near_boundary': int(near_boundary.sum()),
        'n_spikes': int(spike_mask.sum()),
        'n_boundary_spikes': int(n_issues),
        'has_issues': n_issues > 0,
        'max_theta_ddot': float(np.abs(theta_ddot).max())
    }