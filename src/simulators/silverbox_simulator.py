"""
Silverbox System: Real Data Loader + Duffing Teacher (silverbox_simulator.py)

This module provides two components:
    1. generate_silverbox_dataset():
       Loads real Silverbox measurement data via nonlinear_benchmarks,
       computes state/derivative estimates via Savitzky-Golay filtering,
       and slices into fixed-length windows for E-SINDy training.

    2. SilverboxDuffingTeacher:
       Integrates the fitted Duffing ODE (u=0, free oscillation) to
       generate synthetic trajectories for GMM augmentation pool.

Physical system:
    State: [x1=y(t), x2=dy/dt(t)] — output voltage and velocity
    Input: u(t) — excitation voltage (multisine in real data; zero in teacher)

Duffing EOM:
    dx1/dt = x2
    dx2/dt = k*x1 + k3*x1³ + c*x2 + b*u

TEACHER PARAMETERS (SSOT — exploration-fixed defaults):
    These values are the OFFICIAL Duffing parameters used for all pool generation.
    They were derived from a least-squares fit on 5000 points of the full
    Silverbox signal (2026-03-10 exploration) and are physically validated:
        k  = -93282.91   (linear stiffness; k < 0 → restoring force)
        k3 = -44580.43   (cubic stiffness)
        c  = -2.9527     (damping coefficient; c < 0 → positive damping)
        b  =  3100.43    (input gain; irrelevant for teacher since u=0)

    IMPORTANT: These defaults must NOT be overridden by n_train-level re-fitting.
    Re-fitting on n_train=3 windows is numerically unstable and may produce
    physically invalid parameters (e.g. c > 0 = negative damping observed).
    The exploration-fixed defaults are the SSOT for teacher provenance.

NOTE on state estimation:
    x1 = y  (direct measurement)
    x2 = savgol_filter(y, deriv=1)  (velocity estimate)
    dx1/dt = x2  (kinematic identity; reuse same SavGol output)
    dx2/dt = savgol_filter(y, deriv=2)  (acceleration estimate from raw y)

NOTE on teacher (u=0):
    Real data uses multisine excitation. Teacher uses u=0 (free oscillation).
    Rationale: The Duffing oscillator is stable (c<0 damping, k<0 restoring).
    Free oscillations from GMM-sampled ICs cover the physical state space
    relevant to training data. The u coefficient (b) is identified from
    real training data (which has non-zero u), not from teacher trajectories.

Reference:
    Wigren, T. & Schoukens, J. (2013). Three free data sets for development
    and benchmarking in nonlinear system identification. ECC 2013.
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter


# =============================================================================
# Default Duffing Parameters (from exploration 2026-03-10)
# =============================================================================

DUFFING_K_DEFAULT   = -93282.9087
DUFFING_K3_DEFAULT  = -44580.4296
DUFFING_C_DEFAULT   = -2.9527
DUFFING_B_DEFAULT   =  3100.4267


# =============================================================================
# Duffing Teacher (for GMM augmentation pool)
# =============================================================================

class SilverboxDuffingTeacher:
    """
    Duffing ODE integrator for Silverbox augmentation pool generation.

    Integrates the free (u=0) Duffing system from IC [x1_0, x2_0]:
        dx1/dt = x2
        dx2/dt = k*x1 + k3*x1³ + c*x2

    The teacher is used to generate diverse trajectories from GMM-sampled
    initial conditions. The input u is set to zero for all teacher trajectories.
    """

    DEFAULT_PARAMS = {
        'k':  DUFFING_K_DEFAULT,
        'k3': DUFFING_K3_DEFAULT,
        'c':  DUFFING_C_DEFAULT,
    }

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize Duffing teacher.

        Args:
            params: Dict with keys 'k', 'k3', 'c'.
                    Missing keys use fitted defaults.
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if params is not None:
            self.params.update(params)
        self._k  = self.params['k']
        self._k3 = self.params['k3']
        self._c  = self.params['c']
        self._validate()

    def _validate(self) -> None:
        """Validate Duffing stability conditions."""
        # c < 0 → positive damping (ẍ += c*ẋ, c<0 means damping)
        if self._c >= 0:
            raise ValueError(
                f"c must be < 0 for positive damping (got c={self._c}). "
                f"Check parameter sign convention."
            )
        # k < 0 → restoring force (ẍ += k*x, k<0 means restoring)
        if self._k >= 0:
            raise ValueError(
                f"k must be < 0 for restoring force (got k={self._k}). "
                f"Check parameter sign convention."
            )

    @property
    def state_dim(self) -> int:
        return 2

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute Duffing free dynamics (u=0).

        Args:
            t: Current time (unused)
            state: [x1, x2]

        Returns:
            [dx1/dt, dx2/dt]
        """
        x1, x2 = state[0], state[1]
        dx1 = x2
        dx2 = self._k * x1 + self._k3 * (x1 ** 3) + self._c * x2
        return np.array([dx1, dx2], dtype=np.float64)

    def simulate(
        self,
        x0: np.ndarray,
        W: int,
        dt: float,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate Duffing trajectory from IC x0 for W time steps.

        Args:
            x0: Initial state [x1_0, x2_0], shape (2,)
            W: Number of time steps (including t=0)
            dt: Time step (s)
            rtol, atol: ODE solver tolerances

        Returns:
            t_arr: Time array (W,)
            x_arr: State trajectory (W, 2)
            dx_arr: Derivative trajectory (W, 2)
        """
        t_end = (W - 1) * dt
        t_eval = np.linspace(0.0, t_end, W)

        sol = solve_ivp(
            fun=self.dynamics,
            t_span=(0.0, t_end),
            y0=x0.astype(np.float64),
            method='RK45',
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Duffing ODE integration failed: {sol.message}")

        x_arr = sol.y.T   # (W, 2)
        t_arr = sol.t     # (W,)

        # Analytical derivatives at each time step
        dx_arr = np.zeros_like(x_arr)
        for i in range(W):
            dx_arr[i] = self.dynamics(t_arr[i], x_arr[i])

        return t_arr, x_arr, dx_arr

    def is_bounded(
        self,
        x_arr: np.ndarray,
        max_state_norm: float = 300.0,
    ) -> bool:
        """
        Check if trajectory stays within bounded region.

        Bound is chosen generously (≈5× max observed real data norm).

        Args:
            x_arr: State trajectory (W, 2)
            max_state_norm: Maximum allowed ||[x1, x2]||

        Returns:
            True if trajectory is bounded and finite
        """
        norms = np.linalg.norm(x_arr, axis=1)
        return bool(
            np.all(norms < max_state_norm) and np.all(np.isfinite(x_arr))
        )

    def generate_trajectory(
        self,
        rng: np.random.Generator,
        ic: np.ndarray,
        W: int,
        dt: float,
        max_state_norm: float = 300.0,
        max_attempts: int = 50,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> Optional[Dict]:
        """
        Generate a valid (bounded) Duffing trajectory from given IC.

        If trajectory diverges (unlikely given stable Duffing), returns None.
        In practice with c<0 damping and k<0 restoring, all trajectories
        are bounded.

        Args:
            rng: NumPy random generator (unused; kept for interface consistency)
            ic: Initial condition [x1_0, x2_0], shape (2,)
            W: Window length (time steps)
            dt: Time step (s)
            max_state_norm: Reject threshold
            max_attempts: Retry limit
            rtol, atol: ODE solver tolerances

        Returns:
            Dict with keys:
                x   : (W, 2) state trajectory [x1, x2]
                dx  : (W, 2) derivative trajectory [dx1/dt, dx2/dt]
                t   : (W,) time array
                u   : (W, 1) zeros (u=0 for teacher)
                x0  : (2,) initial condition
            None if integration fails after max_attempts.
        """
        x0 = ic.astype(np.float64)

        for _ in range(max_attempts):
            try:
                t_arr, x_arr, dx_arr = self.simulate(x0, W, dt, rtol, atol)
                if self.is_bounded(x_arr, max_state_norm):
                    u_arr = np.zeros((W, 1), dtype=np.float64)
                    return {
                        'x':  x_arr,    # (W, 2)
                        'dx': dx_arr,   # (W, 2)
                        't':  t_arr,    # (W,)
                        'u':  u_arr,    # (W, 1) zeros
                        'x0': x0,       # (2,)
                    }
            except RuntimeError:
                continue
        return None

    def get_params(self) -> Dict[str, float]:
        """Return current Duffing parameters for logging."""
        return {'k': self._k, 'k3': self._k3, 'c': self._c, 'u_teacher': 0.0}


# =============================================================================
# Duffing Parameter Estimation (for teacher initialization)
# =============================================================================

def fit_duffing_params(
    x1: np.ndarray,
    x2: np.ndarray,
    dx2: np.ndarray,
    u: np.ndarray,
    n_points: int = 5000,
) -> Dict[str, float]:
    """
    Estimate Duffing EOM parameters via least squares.

    Fits: dx2 = k*x1 + k3*x1³ + c*x2 + b*u

    Args:
        x1: Position (V), shape (N,)
        x2: Velocity (V/s), shape (N,)
        dx2: Acceleration (V/s²), shape (N,)
        u: Input voltage (V), shape (N,)
        n_points: Number of points to use (first n_points)

    Returns:
        Dict with keys 'k', 'k3', 'c', 'b'
    """
    N = min(n_points, len(x1) - 1)
    A = np.column_stack([x1[:N], x1[:N] ** 3, x2[:N], u[:N]])
    b_vec = dx2[:N]
    coeffs, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    return {
        'k':  float(coeffs[0]),
        'k3': float(coeffs[1]),
        'c':  float(coeffs[2]),
        'b':  float(coeffs[3]),
    }


# =============================================================================
# Real Data Loader and Dataset Generator
# =============================================================================

def generate_silverbox_dataset(
    n_train: int = 3,
    n_val: int = 3,
    n_test: int = 20,
    W: int = 500,
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
    train_fraction: float = 0.70,
    master_seed: int = 42,
) -> Dict:
    """
    Load real Silverbox data and generate windowed dataset for E-SINDy.

    Data pipeline:
    1. Load multisine train_val data from nonlinear_benchmarks
    2. Estimate states and derivatives via Savitzky-Golay:
       x1  = y (raw measurement)
       x2  = savgol(y, deriv=1)  (velocity)
       dx1 = x2  (kinematic identity)
       dx2 = savgol(y, deriv=2)  (acceleration, from raw y directly)
    3. Temporal split: first train_fraction of signal → train region
    4. Sample non-overlapping windows randomly within each region

    NOTE: Duffing teacher parameters are NOT stored in the dataset.
    Teacher always uses exploration-fixed DUFFING_*_DEFAULT constants.
    See module-level docstring for rationale (n_train re-fitting is unstable).

    Args:
        n_train: Number of training windows (default 3; n=10/5 trivially identified)
        n_val: Number of validation windows
        n_test: Number of test windows
        W: Window length (time steps)
        savgol_window: SavGol filter window length (must be odd)
        savgol_polyorder: SavGol polynomial order
        train_fraction: Fraction of signal length for train+val region
        master_seed: Random seed for window selection

    Returns:
        Dict compatible with E-SINDy pipeline:
            train_x  : (n_train, W, 2) — [x1, x2]
            train_u  : (n_train, W, 1) — [u]
            train_dx : (n_train, W, 2) — [dx1/dt, dx2/dt]
            train_params : (n_train, 1) — [k_nominal] identifier
            train_cond_id : (n_train,)
            val_*   : same structure
            test_*  : same structure
            t       : (W,) — relative time within window
            dt      : float
            sampling_time: float
            window_starts_train: (n_train,) — absolute sample indices
    """
    try:
        import nonlinear_benchmarks
    except ImportError:
        raise ImportError(
            "nonlinear_benchmarks package required. "
            "Install with: pip install nonlinear_benchmarks"
        )

    # ------------------------------------------------------------------
    # Step 1: Load real data
    # ------------------------------------------------------------------
    train_val, _test_data = nonlinear_benchmarks.Silverbox()
    u_full = train_val.u.astype(np.float64)   # (N_total,) excitation voltage
    y_full = train_val.y.astype(np.float64)   # (N_total,) output voltage
    dt = float(train_val.sampling_time)        # ≈ 0.001638 s

    N_total = len(u_full)
    assert len(y_full) == N_total, "u and y length mismatch"

    # ------------------------------------------------------------------
    # Step 2: Estimate states and derivatives via SavGol
    # ------------------------------------------------------------------
    # x1: direct measurement
    x1_full = y_full

    # x2: first derivative of y (velocity estimate)
    x2_full = savgol_filter(
        y_full, window_length=savgol_window,
        polyorder=savgol_polyorder, deriv=1, delta=dt
    )

    # dx1/dt = x2 (kinematic identity — exact, no additional filtering)
    dx1_full = x2_full

    # dx2/dt: second derivative from raw y (avoids SavGol error accumulation)
    dx2_full = savgol_filter(
        y_full, window_length=savgol_window,
        polyorder=savgol_polyorder, deriv=2, delta=dt
    )

    # ------------------------------------------------------------------
    # Step 3: Determine valid window start indices (temporal split)
    # ------------------------------------------------------------------
    edge_guard = W // 2  # Skip edge samples with high SavGol error
    train_val_end = int(N_total * train_fraction)
    test_start    = train_val_end  # No overlap gap needed (temporal split)

    # Train+val region: [edge_guard, train_val_end - W)
    # Test region: [test_start, N_total - W - edge_guard)
    stride = 200  # Candidate window stride (allows overlap for diversity)

    train_val_candidates = list(
        range(edge_guard, train_val_end - W, stride)
    )
    test_candidates = list(
        range(test_start, N_total - W - edge_guard, stride)
    )

    n_needed = n_train + n_val
    if len(train_val_candidates) < n_needed:
        raise RuntimeError(
            f"Not enough train+val windows: {len(train_val_candidates)} available, "
            f"{n_needed} needed. Reduce n_train/n_val or increase train_fraction."
        )
    if len(test_candidates) < n_test:
        raise RuntimeError(
            f"Not enough test windows: {len(test_candidates)} available, "
            f"{n_test} needed."
        )

    rng = np.random.default_rng(master_seed)

    # Select non-overlapping windows within each split
    train_val_selected = _sample_nonoverlapping(
        rng, train_val_candidates, n_needed, W
    )
    test_selected = _sample_nonoverlapping(
        rng, test_candidates, n_test, W
    )

    train_starts = train_val_selected[:n_train]
    val_starts   = train_val_selected[n_train:]

    # ------------------------------------------------------------------
    # Step 4: Extract windows
    # ------------------------------------------------------------------
    def extract_windows(starts: List[int]) -> Tuple:
        x_list, dx_list, u_list, cond_list = [], [], [], []
        for cond_id, s in enumerate(starts):
            x_seg = np.column_stack([
                x1_full[s:s+W],
                x2_full[s:s+W],
            ])   # (W, 2)
            dx_seg = np.column_stack([
                dx1_full[s:s+W],
                dx2_full[s:s+W],
            ])   # (W, 2)
            u_seg = u_full[s:s+W, np.newaxis]  # (W, 1)

            x_list.append(x_seg)
            dx_list.append(dx_seg)
            u_list.append(u_seg)
            cond_list.append(cond_id)

        return (
            np.array(x_list,    dtype=np.float32),    # (n, W, 2)
            np.array(dx_list,   dtype=np.float32),    # (n, W, 2)
            np.array(u_list,    dtype=np.float32),    # (n, W, 1)
            np.array(cond_list, dtype=np.int32),      # (n,)
        )

    train_x, train_dx, train_u, train_cond = extract_windows(train_starts)
    val_x,   val_dx,   val_u,   val_cond   = extract_windows(val_starts)
    test_x,  test_dx,  test_u,  test_cond  = extract_windows(test_selected)

    # params column: nominal k as system identifier (constant per trajectory)
    k_nominal = DUFFING_K_DEFAULT
    train_params = np.full((n_train, 1), k_nominal, dtype=np.float32)
    val_params   = np.full((n_val,   1), k_nominal, dtype=np.float32)
    test_params  = np.full((n_test,  1), k_nominal, dtype=np.float32)

    t_arr = np.linspace(0.0, (W - 1) * dt, W, dtype=np.float32)

    dataset = {
        'train_x':        train_x,
        'train_u':        train_u,
        'train_dx':       train_dx,
        'train_params':   train_params,
        'train_cond_id':  train_cond,
        'val_x':          val_x,
        'val_u':          val_u,
        'val_dx':         val_dx,
        'val_params':     val_params,
        'val_cond_id':    val_cond,
        'test_x':         test_x,
        'test_u':         test_u,
        'test_dx':        test_dx,
        'test_params':    test_params,
        'test_cond_id':   test_cond,
        't':              t_arr,
        'dt':             np.float32(dt),
        'sampling_time':  np.float32(dt),
        'window_starts_train': np.array(train_starts, dtype=np.int32),
        'window_starts_val':   np.array(val_starts,   dtype=np.int32),
        'window_starts_test':  np.array(test_selected, dtype=np.int32),
        'N_total_signal':     np.int32(N_total),
        'train_fraction':     np.float32(train_fraction),
    }

    # NOTE: Duffing parameters are NOT fitted here.
    # Teacher always uses exploration-fixed DUFFING_*_DEFAULT constants.
    # n_train-level re-fitting is disabled — numerically unstable at n=3.
    # See module-level docstring for rationale.

    return dataset


def _sample_nonoverlapping(
    rng: np.random.Generator,
    candidates: List[int],
    n: int,
    W: int,
) -> List[int]:
    """
    Sample n non-overlapping windows from candidate start indices.

    Attempts greedy random sampling with non-overlap constraint.
    Falls back to sorted selection if too few non-overlapping windows available.

    Args:
        rng: NumPy random generator
        candidates: List of candidate start indices
        n: Number of windows to select
        W: Window length (for overlap check)

    Returns:
        List of n selected start indices (sorted)
    """
    shuffled = list(rng.permutation(candidates))
    selected = []
    for s in shuffled:
        # Check non-overlap with already selected windows
        overlap = any(abs(s - sel) < W for sel in selected)
        if not overlap:
            selected.append(s)
        if len(selected) == n:
            break

    if len(selected) < n:
        # Fallback: allow overlap if necessary (warns in runner)
        remaining = [s for s in shuffled if s not in selected]
        selected.extend(remaining[:n - len(selected)])

    return sorted(selected[:n])


# =============================================================================
# Dataset Validation (Silverbox-specific preflight guard)
# =============================================================================

def validate_silverbox_dataset(data: Dict) -> None:
    """
    Preflight validation for Silverbox dataset.

    Raises AssertionError immediately on any violation (fail-fast design).
    Call at start of every runner before any computation.

    Checks:
        - Required keys present
        - state_dim = 2, input_dim = 1
        - No NaN/Inf in train/val/test arrays
        - Reasonable signal ranges (sanity check against known Silverbox ranges)
        - dt consistency
    """
    REQUIRED_KEYS = [
        'train_x', 'train_u', 'train_dx',
        'val_x', 'val_u', 'val_dx',
        'test_x', 'test_u', 'test_dx',
        't', 'dt',
    ]
    for key in REQUIRED_KEYS:
        assert key in data, f"PREFLIGHT FAIL: missing key '{key}'"

    # Shape checks
    train_x  = data['train_x']
    train_u  = data['train_u']
    train_dx = data['train_dx']

    assert train_x.ndim == 3, \
        f"PREFLIGHT FAIL: train_x must be 3D (n, W, state_dim), got {train_x.ndim}D"
    n_train, W, state_dim = train_x.shape
    assert state_dim == 2, \
        f"PREFLIGHT FAIL: state_dim must be 2 (Silverbox), got {state_dim}"

    assert train_u.shape == (n_train, W, 1), \
        f"PREFLIGHT FAIL: train_u shape {train_u.shape}, expected ({n_train}, {W}, 1)"
    assert train_dx.shape == (n_train, W, 2), \
        f"PREFLIGHT FAIL: train_dx shape {train_dx.shape}, expected ({n_train}, {W}, 2)"

    # NaN/Inf checks
    for split in ['train', 'val', 'test']:
        for arr_name in [f'{split}_x', f'{split}_u', f'{split}_dx']:
            arr = data[arr_name]
            assert np.all(np.isfinite(arr)), \
                f"PREFLIGHT FAIL: {arr_name} contains NaN/Inf"

    # Signal range sanity (generous bounds to allow augmented data)
    x1_arr = data['train_x'][:, :, 0]
    x2_arr = data['train_x'][:, :, 1]
    assert np.abs(x1_arr).max() < 5.0, \
        f"PREFLIGHT FAIL: x1 max={np.abs(x1_arr).max():.3f} exceeds 5.0V sanity bound"
    assert np.abs(x2_arr).max() < 5000.0, \
        f"PREFLIGHT FAIL: x2 max={np.abs(x2_arr).max():.1f} exceeds 5000 V/s sanity bound"

    # dt sanity
    dt = float(data['dt'])
    assert 0.001 < dt < 0.01, \
        f"PREFLIGHT FAIL: dt={dt:.6f} outside expected range (0.001, 0.01) for Silverbox"

    print(f"  ✅ PREFLIGHT: Silverbox dataset validated "
          f"(n_train={n_train}, W={W}, state_dim={state_dim}, dt={dt:.6f}s)")