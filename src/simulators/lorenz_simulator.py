"""
Lorenz-63 Chaotic System Simulator.

Standard Lorenz system with OOD parameter rho (Rayleigh number).
Reference: Lorenz (1963), J. Atmospheric Sciences 20(2), 130-141.

State definition:
    index 0: x  (dimensionless)
    index 1: y  (dimensionless)
    index 2: z  (dimensionless)

No control input. u is stored as zeros for schema compatibility.

Equations of motion:
    dx/dt = sigma * (y - x)
    dy/dt = rho * x - y - x * z
    dz/dt = x * y - beta * z

Standard parameters: sigma=10, beta=8/3, rho_nominal=28
OOD design: Initial conditions (IC) only. rho is fixed at 28.
    IMPORTANT: rho appears directly as EOM coefficient (dy/dt = rho*x - y - x*z).
    Mixing multiple rho values in training makes SINDy x-coefficient structurally
    undefined (one coefficient must represent different physical values).
    → Single rho=28 is structurally required for well-posed SINDy identification.
    → OOD generalization is tested via IC variation (different attractor trajectories).
    (Parametric SINDy with varying rho is a separate research topic.)

Key property: Lorenz is bounded (strange attractor),
so no stabilizing controller is needed for data generation.
This eliminates the Coverage-collapse issue seen in AEK.

Reference: configs/systems/lorenz.yaml (SSOT for all parameters)
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.integrate import solve_ivp


class LorenzSimulator:
    """
    Lorenz-63 Chaotic System Simulator.

    Default parameters: sigma=10, beta=8/3, rho=28 (canonical values).
    OOD knob: rho (changes convection intensity, alters dy/dt coefficient).

    No control input; u is stored as zeros for dataset schema consistency.
    """

    SIGMA = 10.0
    BETA = 8.0 / 3.0       # Exact rational value
    RHO_NOMINAL = 28.0

    REQUIRED_PARAMS = ['sigma', 'beta', 'rho']

    DEFAULT_PARAMS = {
        'sigma': 10.0,
        'beta': 8.0 / 3.0,
        'rho': 28.0,
    }

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize Lorenz simulator.

        Args:
            params: Physical parameters dict. Missing keys use defaults.
                    Key OOD parameter: rho (Rayleigh number).
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if params is not None:
            self.params.update(params)
        self._validate_params()

        self._sigma = self.params['sigma']
        self._beta = self.params['beta']
        self._rho = self.params['rho']

    def _validate_params(self) -> None:
        """Validate physical parameters."""
        for p in self.REQUIRED_PARAMS:
            if p not in self.params:
                raise ValueError(f"Missing required parameter: {p}")
        if self.params['sigma'] <= 0:
            raise ValueError(f"sigma must be positive, got {self.params['sigma']}")
        if self.params['beta'] <= 0:
            raise ValueError(f"beta must be positive, got {self.params['beta']}")
        if self.params['rho'] <= 0:
            raise ValueError(f"rho must be positive, got {self.params['rho']}")

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def input_dim(self) -> int:
        return 0  # No control input

    def dynamics(self, t: float, state: np.ndarray, u: float = 0.0) -> np.ndarray:
        """
        Compute Lorenz dynamics analytically.

        Args:
            t: Current time (unused, for ODE solver interface)
            state: [x, y, z]
            u: Unused (kept for interface consistency; Lorenz has no input)

        Returns:
            [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state[0], state[1], state[2]

        dx = self._sigma * (y - x)
        dy = self._rho * x - y - x * z
        dz = x * y - self._beta * z

        return np.array([dx, dy, dz], dtype=np.float64)

    def simulate(
        self,
        x0: np.ndarray,
        T_steps: int,
        dt: float,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate Lorenz trajectory from initial condition x0.

        Uses scipy RK45 (dense output) for high-accuracy integration.
        Derivatives computed analytically at each time step.

        Args:
            x0: Initial state [x0, y0, z0], shape (3,)
            T_steps: Number of time steps (including t=0)
            dt: Time step
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver

        Returns:
            t_arr: Time array, shape (T_steps,)
            x_arr: State trajectory, shape (T_steps, 3)
            dx_arr: Derivative trajectory, shape (T_steps, 3)
        """
        t_span = (0.0, (T_steps - 1) * dt)
        t_eval = np.linspace(0.0, (T_steps - 1) * dt, T_steps)

        sol = solve_ivp(
            fun=self.dynamics,
            t_span=t_span,
            y0=x0.astype(np.float64),
            method='RK45',
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        x_arr = sol.y.T  # (T_steps, 3)
        t_arr = sol.t    # (T_steps,)

        # Compute derivatives analytically at each time step
        dx_arr = np.zeros_like(x_arr)
        for i in range(T_steps):
            dx_arr[i] = self.dynamics(t_arr[i], x_arr[i])

        return t_arr, x_arr, dx_arr

    def is_bounded(
        self,
        x_arr: np.ndarray,
        max_norm: float = 200.0,
    ) -> bool:
        """
        Check if trajectory stays within bounded region.

        Lorenz attractor is bounded for standard parameters.
        Divergence indicates numerical issues or extreme ICs.

        Args:
            x_arr: State trajectory, shape (T_steps, 3)
            max_norm: Maximum allowed state norm

        Returns:
            True if trajectory is bounded
        """
        norms = np.linalg.norm(x_arr, axis=1)
        return bool(np.all(norms < max_norm) and np.all(np.isfinite(x_arr)))

    def sample_ic(
        self,
        rng: np.random.Generator,
        x_range: Tuple[float, float] = (-15.0, 15.0),
        y_range: Tuple[float, float] = (-20.0, 20.0),
        z_range: Tuple[float, float] = (5.0, 40.0),
    ) -> np.ndarray:
        """
        Sample random initial condition within attractor bounds.

        Ranges cover the Lorenz attractor for standard parameters.
        For OOD rho, the attractor scales roughly as sqrt(rho/rho_nominal).

        Args:
            rng: NumPy random generator
            x_range, y_range, z_range: IC sampling bounds

        Returns:
            Initial state [x0, y0, z0], shape (3,)
        """
        # Scale ranges by sqrt(rho/rho_nominal) for OOD rho
        scale = np.sqrt(self._rho / self.RHO_NOMINAL)
        x0 = rng.uniform(x_range[0] * scale, x_range[1] * scale)
        y0 = rng.uniform(y_range[0] * scale, y_range[1] * scale)
        z0 = rng.uniform(z_range[0], z_range[1] * scale)  # z is always positive
        z0 = max(z0, 1.0)  # z > 0 always on attractor
        return np.array([x0, y0, z0], dtype=np.float64)

    def generate_trajectory(
        self,
        rng: np.random.Generator,
        T_steps: int,
        dt: float,
        max_attempts: int = 20,
        max_state_norm: float = 200.0,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> Optional[Dict]:
        """
        Generate a valid (bounded) trajectory with random IC.

        Retries up to max_attempts if trajectory diverges.

        Args:
            rng: NumPy random generator
            T_steps: Number of time steps
            dt: Time step
            max_attempts: Maximum retries
            max_state_norm: Reject if ||state|| > max_state_norm
            rtol, atol: ODE solver tolerances

        Returns:
            Dict with keys: x (T,3), dx (T,3), t (T,), u (T,1) [zeros]
            None if all attempts fail.
        """
        for attempt in range(max_attempts):
            x0 = self.sample_ic(rng)
            try:
                t_arr, x_arr, dx_arr = self.simulate(
                    x0, T_steps, dt, rtol=rtol, atol=atol
                )
                if self.is_bounded(x_arr, max_norm=max_state_norm):
                    # u is zeros (no control input); stored for schema compatibility
                    u_arr = np.zeros((T_steps, 1), dtype=np.float64)
                    return {
                        'x': x_arr,      # (T, 3)
                        'dx': dx_arr,    # (T, 3)
                        't': t_arr,      # (T,)
                        'u': u_arr,      # (T, 1) — zeros, schema compatibility
                        'x0': x0,        # (3,)
                        'rho': self._rho,
                    }
            except RuntimeError:
                continue  # retry
        return None  # Failed after max_attempts

    def get_params(self) -> Dict[str, float]:
        """Return current parameters for logging."""
        return {
            'sigma': self._sigma,
            'beta': self._beta,
            'rho': self._rho,
            'beta_exact': '8/3',
        }


def generate_lorenz_dataset(
    train_rho: List[float],
    val_rho: List[float],
    test_rho: List[float],
    n_train: int,
    n_val: int,
    n_test: int,
    T_steps: int,
    dt: float,
    master_seed: int = 42,
    max_state_norm: float = 200.0,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    noise_std_fraction: float = 0.05,
    savgol_window: int = 7,
    savgol_polyorder: int = 3,
) -> Dict:
    """
    Generate Lorenz dataset with measurement noise and SavGol derivative estimation.

    Standard SINDy benchmark protocol (Brunton et al. 2016):
    1. Simulate clean trajectories
    2. Add Gaussian measurement noise: std = noise_std_fraction * per-state std
    3. Estimate derivatives via Savitzky-Golay filter

    Single rho design:
        train/val/test all use the same rho list.
        OOD is achieved through different initial conditions, not rho variation.
        Rationale: rho directly appears as a coefficient in dy/dt = rho*x - y - xz.
        Mixing rho values in training makes SINDy coefficients structurally undefined.

    Args:
        train_rho: List of rho values (use [28.0] for single-rho)
        val_rho, test_rho: Same structure
        n_train, n_val, n_test: Total trajectories per split
        T_steps: Time steps per trajectory
        dt: Time step
        master_seed: Reproducibility seed
        max_state_norm: Trajectory quality threshold
        sigma, beta: Fixed Lorenz parameters
        noise_std_fraction: Measurement noise (fraction of per-state std)
        savgol_window, savgol_polyorder: SavGol filter parameters

    Returns:
        Dict for np.savez with train_x (noisy), train_dx (SavGol), etc.
    """
    from scipy.signal import savgol_filter

    rng = np.random.default_rng(master_seed)

    def _generate_split(rho_list, n_total, split_seed):
        split_rng = np.random.default_rng(split_seed)
        n_rho = len(rho_list)
        counts = [n_total // n_rho] * n_rho
        for i in range(n_total % n_rho):
            counts[i] += 1

        x_list, u_list, dx_list, params_list, cond_list = [], [], [], [], []
        cond_id = 0

        for rho_idx, (rho, count) in enumerate(zip(rho_list, counts)):
            sim = LorenzSimulator(params={'sigma': sigma, 'beta': beta, 'rho': rho})
            for _ in range(count):
                result = sim.generate_trajectory(
                    rng=split_rng,
                    T_steps=T_steps,
                    dt=dt,
                    max_attempts=50,
                    max_state_norm=max_state_norm,
                )
                if result is None:
                    raise RuntimeError(
                        f"Failed to generate trajectory for rho={rho}"
                    )
                x_clean = result['x']   # (T, 3) — clean

                # Step 1: Add measurement noise
                if noise_std_fraction > 0:
                    state_std = x_clean.std(axis=0)
                    state_std[state_std < 1e-6] = 1.0
                    noise = split_rng.normal(
                        0, noise_std_fraction * state_std, x_clean.shape
                    )
                    x_noisy = x_clean + noise
                else:
                    x_noisy = x_clean.copy()

                # Step 2: SavGol derivative estimation from noisy x
                dx_savgol = np.zeros_like(x_noisy)
                for s in range(3):
                    dx_savgol[:, s] = savgol_filter(
                        x_noisy[:, s],
                        window_length=savgol_window,
                        polyorder=savgol_polyorder,
                        deriv=1,
                        delta=dt,
                    )

                x_list.append(x_noisy)
                u_list.append(result['u'])        # zeros
                dx_list.append(dx_savgol)
                params_list.append([rho])
                cond_list.append(cond_id)
            cond_id += 1

        return (
            np.array(x_list, dtype=np.float32),
            np.array(u_list, dtype=np.float32),
            np.array(dx_list, dtype=np.float32),
            np.array(params_list, dtype=np.float32),
            np.array(cond_list, dtype=np.int32),
        )

    seeds = rng.integers(0, 2**31, size=3)
    train_x, train_u, train_dx, train_params, train_cond = \
        _generate_split(train_rho, n_train, int(seeds[0]))
    val_x, val_u, val_dx, val_params, val_cond = \
        _generate_split(val_rho, n_val, int(seeds[1]))
    test_x, test_u, test_dx, test_params, test_cond = \
        _generate_split(test_rho, n_test, int(seeds[2]))

    t = np.linspace(0.0, (T_steps - 1) * dt, T_steps, dtype=np.float32)

    return {
        'train_x': train_x, 'train_u': train_u, 'train_dx': train_dx,
        'train_params': train_params, 'train_cond_id': train_cond,
        'val_x': val_x, 'val_u': val_u, 'val_dx': val_dx,
        'val_params': val_params, 'val_cond_id': val_cond,
        'test_x': test_x, 'test_u': test_u, 'test_dx': test_dx,
        'test_params': test_params, 'test_cond_id': test_cond,
        't': t, 'dt': np.float32(dt),
    }