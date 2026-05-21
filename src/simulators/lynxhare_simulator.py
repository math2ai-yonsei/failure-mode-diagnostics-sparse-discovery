"""
Lynx-Hare (Lotka-Volterra) Data Loader and Simulator.

Data source:
    Hudson Bay Company fur trade records, 1900-1920 (21 annual observations).
    Elton & Nicholson (1942), MacLulich (1937).
    Same dataset used in E-SINDy original paper (Fasel et al., Proc. R. Soc. A, 2022).

State definition:
    index 0: H  (hare population, unit: thousands)
    index 1: L  (lynx population, unit: thousands)

No control input. u stored as zeros for schema compatibility (like Lorenz).

ODE: Lotka-Volterra predator-prey
    dH/dt = α*H  - β*H*L    (hare growth minus predation)
    dL/dt = δ*H*L - γ*L     (lynx growth minus natural death)

Dataset design (v1.1 — sliding window):
    Training: n_train pseudo-trajectories from the real 21-pt series
              via sliding window (window_size=7, stride=2 → 8 windows available).
              n_train=3 (first 3 windows: 1900-1906, 1902-1908, 1904-1910).
              Enables valid trajectory-level bootstrap (n_traj ≥ 2).
    Val/Test: synthetic LV trajectories with IC variation.
              Derivatives computed analytically from LV ODE.
    Parameters α, β, γ, δ are estimated from the full real series.

GMM pool design:
    GMM is fitted on the 21 UNIQUE raw observations (full 1900-1920 series),
    NOT on training windows (which would over-represent the rising phase 1900-1910).

Noise and derivative:
    Real data: already contains measurement noise (fur trapping uncertainty).
    Derivatives: Savitzky-Golay filter (window=5, polyorder=3) applied to
                 the full 21-pt series before windowing.
    Synthetic data: computed analytically from LV ODE.

Role in EAAI paper:
    Exploratory / public real-data lineage validation.
    Same dataset as Fasel et al. (2022) — direct lineage comparison.
    Results are marginal (NULL:5, SOFT:5 with n_train=3, T=7);
    not cited as primary evidence.

Author: Claude (Gate-LynxHare)
Date: 2026-03-09
Version: v1.1
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from typing import Dict, Optional, Tuple


# =============================================================================
# Raw Data (SSOT — hardcoded, never change)
# =============================================================================

# Hudson Bay lynx-hare fur trade records, 1900-1920
# Source: Elton & Nicholson (1942), MacLulich (1937)
# Units: thousands of animals
_YEARS = np.arange(1900, 1921, dtype=float)

_HARE_RAW = np.array([
    30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
    27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2,  7.6, 14.6, 16.2, 24.7
], dtype=float)

_LYNX_RAW = np.array([
     4.0,  6.1,  9.8, 35.2, 59.4, 41.7, 19.0, 13.0,  8.3,  9.1,
     7.4,  8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8,  9.7, 10.1,  8.6
], dtype=float)

assert len(_YEARS) == len(_HARE_RAW) == len(_LYNX_RAW) == 21, \
    "Data length mismatch"

T_REAL = 21       # Number of real data points
DT_REAL = 1.0     # Annual time step (years)
STATE_DIM = 2     # H, L
INPUT_DIM = 0     # No control input (u stored as zeros)

# Sliding window design constants
# 21 points, window=7, stride=2 → 8 pseudo-trajectories
# n_train=3 maintains low-data regime with valid bootstrap
WINDOW_SIZE = 7   # T per pseudo-trajectory
WINDOW_STRIDE = 2 # Stride between windows
N_WINDOWS = (T_REAL - WINDOW_SIZE) // WINDOW_STRIDE + 1  # = 8


# =============================================================================
# Data Access
# =============================================================================

def get_lynxhare_data() -> Dict[str, np.ndarray]:
    """
    Return raw Lynx-Hare time series.

    Returns:
        Dict with keys: 'years', 'H', 'L', 't', 'dt'
    """
    return {
        'years': _YEARS.copy(),
        'H': _HARE_RAW.copy(),
        'L': _LYNX_RAW.copy(),
        't': (_YEARS - _YEARS[0]),   # time relative to 1900, [0..20]
        'dt': DT_REAL,
    }


# =============================================================================
# Derivative Estimation (SavGol)
# =============================================================================

def compute_savgol_derivatives(
    H: np.ndarray,
    L: np.ndarray,
    dt: float = DT_REAL,
    window: int = 5,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate dH/dt and dL/dt using Savitzky-Golay filter.

    Standard derivative protocol (Brunton 2016).
    Window=5 chosen for T=21 (smaller than Lorenz window=7,
    appropriate for short annual time series).

    Args:
        H: Hare time series, shape (T,)
        L: Lynx time series, shape (T,)
        dt: Time step (1.0 year for real data)
        window: SavGol window length (must be odd, < len(H))
        polyorder: Polynomial order (< window)

    Returns:
        dH: dH/dt, shape (T,)
        dL: dL/dt, shape (T,)
    """
    if window >= len(H):
        raise ValueError(f"SavGol window {window} >= data length {len(H)}")
    if polyorder >= window:
        raise ValueError(f"polyorder {polyorder} must be < window {window}")

    dH = savgol_filter(H, window_length=window, polyorder=polyorder,
                       deriv=1, delta=dt)
    dL = savgol_filter(L, window_length=window, polyorder=polyorder,
                       deriv=1, delta=dt)
    return dH, dL


# =============================================================================
# LV Parameter Estimation
# =============================================================================

def estimate_lv_params(
    H: np.ndarray,
    L: np.ndarray,
    dH: np.ndarray,
    dL: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate Lotka-Volterra parameters from data via least squares.

    EOM rearranged for linear regression:
        dH/dt = α*H - β*H*L  → X=[H, H*L], y=dH/dt, θ=[α, -β]
        dL/dt = δ*H*L - γ*L  → X=[H*L, L], y=dL/dt, θ=[δ, -γ]

    Args:
        H, L: State time series, shape (T,)
        dH, dL: Derivative estimates, shape (T,)

    Returns:
        Dict with keys: 'alpha', 'beta', 'gamma', 'delta'
        All values clipped to [1e-6, inf] to ensure positive parameters.
    """
    # dH/dt = α*H - β*H*L
    X_H = np.column_stack([H, H * L])
    theta_H, _, _, _ = np.linalg.lstsq(X_H, dH, rcond=None)
    alpha = float(max(theta_H[0], 1e-6))
    beta  = float(max(-theta_H[1], 1e-6))

    # dL/dt = δ*H*L - γ*L
    X_L = np.column_stack([H * L, L])
    theta_L, _, _, _ = np.linalg.lstsq(X_L, dL, rcond=None)
    delta = float(max(theta_L[0], 1e-6))
    gamma = float(max(-theta_L[1], 1e-6))

    return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}


# =============================================================================
# LV ODE Simulator
# =============================================================================

def lv_ode(t: float, y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Lotka-Volterra ODE right-hand side.

    Args:
        t: Current time (unused, autonomous system)
        y: [H, L] state vector
        params: Dict with 'alpha', 'beta', 'gamma', 'delta'

    Returns:
        [dH/dt, dL/dt]
    """
    H, L = y[0], y[1]
    alpha, beta = params['alpha'], params['beta']
    gamma, delta = params['gamma'], params['delta']
    dH = alpha * H - beta * H * L
    dL = delta * H * L - gamma * L
    return np.array([dH, dL])


def simulate_lv_trajectory(
    H0: float,
    L0: float,
    params: Dict[str, float],
    T_steps: int = T_REAL,
    dt: float = DT_REAL,
    max_state: float = 500.0,
) -> Optional[np.ndarray]:
    """
    Simulate LV trajectory from initial condition (H0, L0).

    Args:
        H0, L0: Initial hare and lynx populations (thousands)
        params: LV parameters from estimate_lv_params
        T_steps: Number of time steps
        dt: Time step (years)
        max_state: QC upper bound; returns None if exceeded

    Returns:
        x: (T_steps, 2) state trajectory, or None if QC fails
    """
    if H0 <= 0 or L0 <= 0:
        return None

    t_eval = np.arange(T_steps) * dt
    t_span = (t_eval[0], t_eval[-1])

    try:
        sol = solve_ivp(
            lv_ode,
            t_span,
            [H0, L0],
            args=(params,),
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10,
            dense_output=False,
        )
        if not sol.success:
            return None

        x = sol.y.T  # (T_steps, 2)

        # QC: physical validity
        if not np.all(np.isfinite(x)):
            return None
        if np.any(x < 0):
            return None
        if np.any(x > max_state):
            return None

    except Exception:
        return None

    return x


def compute_lv_derivatives(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Compute LV derivatives analytically from state trajectory.

    Args:
        x: (T, 2) state trajectory
        params: LV parameters

    Returns:
        dx: (T, 2) derivative trajectory
    """
    H = x[:, 0]
    L = x[:, 1]
    alpha, beta = params['alpha'], params['beta']
    gamma, delta = params['gamma'], params['delta']

    dH = alpha * H - beta * H * L
    dL = delta * H * L - gamma * L

    return np.column_stack([dH, dL])


# =============================================================================
# Dataset Generation
# =============================================================================

def make_sliding_windows(
    H: np.ndarray,
    L: np.ndarray,
    dH: np.ndarray,
    dL: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> Dict[str, np.ndarray]:
    """
    Convert a single time series into multiple pseudo-trajectories
    via sliding window, enabling trajectory-level bootstrap.

    Rationale:
        With n_traj=1, trajectory-level bootstrap is degenerate
        (all 100 samples identical → std≈0 → z inflates).
        Sliding window creates multiple overlapping sub-segments
        that each capture local dynamics. This is the standard
        treatment for real time-series in SINDy literature.

    Args:
        H, L: State time series, shape (T_total,)
        dH, dL: Derivative time series, shape (T_total,)
        window_size: Length of each pseudo-trajectory (7 years)
        stride: Step between window starts (2 years)

    Returns:
        Dict with 'x' (N_windows, window_size, 2),
                  'dx' (N_windows, window_size, 2),
                  'u' (N_windows, window_size, 1) zeros
    """
    T_total = len(H)
    starts = list(range(0, T_total - window_size + 1, stride))
    xs, dxs = [], []
    for s in starts:
        e = s + window_size
        xs.append(np.column_stack([H[s:e], L[s:e]]))
        dxs.append(np.column_stack([dH[s:e], dL[s:e]]))

    x_arr  = np.stack(xs,  axis=0).astype(np.float64)   # (N, W, 2)
    dx_arr = np.stack(dxs, axis=0).astype(np.float64)   # (N, W, 2)
    u_arr  = np.zeros((*x_arr.shape[:2], 1), dtype=np.float64)

    return {'x': x_arr, 'dx': dx_arr, 'u': u_arr, 'n_windows': len(starts)}


def generate_lynxhare_dataset(
    n_train: int = 3,
    n_val: int = 5,
    n_test: int = 15,
    window_size: int = WINDOW_SIZE,
    window_stride: int = WINDOW_STRIDE,
    dt: float = DT_REAL,
    savgol_window: int = 5,
    savgol_polyorder: int = 3,
    master_seed: int = 42,
    max_state: float = 500.0,
) -> Dict[str, np.ndarray]:
    """
    Generate Lynx-Hare dataset for E-SINDy pipeline.

    Design (v1.1 — sliding window):
        train: n_train pseudo-trajectories from first n_train sliding windows
               of the real 21-point time series. Each window has length=7 years.
               This gives valid trajectory-level bootstrap (n_traj >= 2).
        val/test: synthetic LV trajectories with IC variation.
                  Derivatives computed analytically from LV ODE.

    Sliding window rationale:
        21 annual observations, window=7, stride=2 → 8 pseudo-trajectories.
        n_train=3 (first 3 windows = years 1900-1906, 1902-1908, 1904-1910)
        captures the rising phase of the population cycle.
        This maintains the low-data regime while enabling valid bootstrap.

    Dataset schema (consistent with CP/AEK/Lorenz):
        *_x:  (N, T, 2)  state  (T = window_size = 7)
        *_u:  (N, T, 1)  zeros (no control input)
        *_dx: (N, T, 2)  derivatives

    Args:
        n_train: Number of training pseudo-trajectories (from sliding windows)
        n_val: Number of synthetic validation trajectories
        n_test: Number of synthetic test trajectories
        window_size: Sliding window length (default: 7)
        window_stride: Stride between windows (default: 2)
        dt: Time step in years (default: 1.0)
        savgol_window: SavGol window for derivative estimation
        savgol_polyorder: Polynomial order for SavGol
        master_seed: Random seed for synthetic IC sampling
        max_state: QC upper bound for simulation

    Returns:
        Dict with train/val/test arrays + metadata
    """
    raw = get_lynxhare_data()
    H_real = raw['H']
    L_real = raw['L']
    t_full = raw['t']

    # ── Full-series SavGol derivatives (applied to full 21-pt series) ───
    # Full series gives better edge accuracy than per-window.
    dH_real, dL_real = compute_savgol_derivatives(
        H_real, L_real, dt=dt,
        window=savgol_window, polyorder=savgol_polyorder,
    )

    # ── Estimate LV parameters from full real data ─────────────────────
    lv_params = estimate_lv_params(H_real, L_real, dH_real, dL_real)

    # ── Training: sliding windows from real data ───────────────────────
    all_windows = make_sliding_windows(
        H_real, L_real, dH_real, dL_real,
        window_size=window_size, stride=window_stride,
    )
    n_available = all_windows['n_windows']
    if n_train > n_available:
        raise ValueError(
            f"n_train={n_train} exceeds available windows={n_available} "
            f"(T={T_REAL}, window={window_size}, stride={window_stride})"
        )

    train_x  = all_windows['x'][:n_train]    # (n_train, window_size, 2)
    train_dx = all_windows['dx'][:n_train]
    train_u  = all_windows['u'][:n_train]

    # t for train windows (use first window's time axis)
    t_train = (np.arange(window_size) * dt).astype(np.float64)

    # ── Synthetic val/test (T = window_size, IC from observed range) ───
    rng = np.random.default_rng(master_seed)

    def _generate_synthetic_split(n: int) -> Dict[str, np.ndarray]:
        xs, dxs, us = [], [], []
        attempts = 0
        while len(xs) < n and attempts < n * 200:
            attempts += 1
            idx = rng.integers(0, len(H_real))
            H0 = float(H_real[idx]) * rng.uniform(0.7, 1.3)
            L0 = float(L_real[idx]) * rng.uniform(0.7, 1.3)

            x = simulate_lv_trajectory(
                H0, L0, lv_params,
                T_steps=window_size, dt=dt, max_state=max_state,
            )
            if x is None:
                continue

            dx = compute_lv_derivatives(x, lv_params)
            xs.append(x)
            dxs.append(dx)
            us.append(np.zeros((window_size, 1)))

        if len(xs) < n:
            raise RuntimeError(
                f"Could not generate {n} valid synthetic trajectories "
                f"(got {len(xs)} after {attempts} attempts). "
                f"LV params: {lv_params}"
            )
        return {
            'x':  np.stack(xs[:n],  axis=0).astype(np.float64),
            'dx': np.stack(dxs[:n], axis=0).astype(np.float64),
            'u':  np.stack(us[:n],  axis=0).astype(np.float64),
        }

    val_data  = _generate_synthetic_split(n_val)
    test_data = _generate_synthetic_split(n_test)

    dataset = {
        'train_x':  train_x,
        'train_u':  train_u,
        'train_dx': train_dx,

        'val_x':    val_data['x'],
        'val_u':    val_data['u'],
        'val_dx':   val_data['dx'],

        'test_x':   test_data['x'],
        'test_u':   test_data['u'],
        'test_dx':  test_data['dx'],

        't':        t_train,
        'dt':       np.float64(dt),

        # LV parameters (SSOT)
        'lv_alpha': np.float64(lv_params['alpha']),
        'lv_beta':  np.float64(lv_params['beta']),
        'lv_gamma': np.float64(lv_params['gamma']),
        'lv_delta': np.float64(lv_params['delta']),

        # Provenance
        'n_windows_available': np.int64(n_available),
        'n_train_windows':     np.int64(n_train),
        'window_size':         np.int64(window_size),
        'window_stride':       np.int64(window_stride),
    }

    return dataset