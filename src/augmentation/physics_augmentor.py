"""
Gate2 Tier-0: Physics-Consistent Augmentation

Implements IC jitter and parameter jitter with re-simulation.
Guarantees dx-x physical consistency.

Methods:
    - IC jitter: Perturb initial condition, re-simulate entire trajectory
    - Param jitter: Perturb physical parameters, re-simulate
    - Combined: Both IC and param jitter

Why re-simulation?
    - Noise injection on x alone breaks dx-x consistency
    - Re-simulation ensures dx = f(x, u, params) holds exactly
    - Track-agnostic: works for both standardized and author_recommended
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from .base import BaseAugmentor, AugmentorConfig, AugmentationResult


@dataclass
class PhysicsAugmentorConfig(AugmentorConfig):
    """Configuration for physics-based augmentation."""
    
    # Method name
    method: str = 'physics_resim'
    
    # IC jitter settings
    ic_jitter_enabled: bool = True
    ic_jitter_std: Dict[str, float] = field(default_factory=lambda: {
        'x': 0.05,           # Cart position jitter (m)
        'x_dot': 0.05,       # Cart velocity jitter (m/s)
        'theta': 0.02,       # Pole angle jitter (rad)
        'theta_dot': 0.05,   # Pole angular velocity jitter (rad/s)
    })
    
    # Param jitter settings
    param_jitter_enabled: bool = True
    param_jitter_rel_std: Dict[str, float] = field(default_factory=lambda: {
        'm_cart': 0.05,      # 5% relative jitter
        'm_pole': 0.05,
        'L': 0.02,           # 2% relative jitter (more sensitive)
        'g': 0.0,            # No jitter on gravity (fixed constant)
    })
    
    # Jitter mode: 'ic_only', 'param_only', 'both', 'random'
    jitter_mode: str = 'both'
    
    # If 'random', probability of each mode
    random_mode_probs: Dict[str, float] = field(default_factory=lambda: {
        'ic_only': 0.4,
        'param_only': 0.3,
        'both': 0.3,
    })
    
    # Quality filter
    max_theta: float = 3.1  # Reject if |theta| exceeds this (near π wrap boundary)
    max_steps_with_nan: int = 0  # Reject if any NaN
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for manifest."""
        base = super().to_dict()
        base.update({
            'ic_jitter_enabled': self.ic_jitter_enabled,
            'ic_jitter_std': self.ic_jitter_std,
            'param_jitter_enabled': self.param_jitter_enabled,
            'param_jitter_rel_std': self.param_jitter_rel_std,
            'jitter_mode': self.jitter_mode,
            'max_theta': self.max_theta,
        })
        return base


class PhysicsAugmentor(BaseAugmentor):
    """
    Physics-consistent augmentor using re-simulation.
    
    Workflow:
        1. Select jitter mode (IC, param, or both)
        2. Perturb initial condition and/or parameters
        3. Re-simulate trajectory using CartPoleSimulator
        4. Compute dx analytically from dynamics
        5. Apply quality filter
    
    Attributes:
        simulator: CartPoleSimulator instance
        controller_fn: Controller function u = f(t, x) or None (use original u)
    """
    
    def __init__(
        self,
        config: PhysicsAugmentorConfig,
        simulator=None,
        controller_fn: Optional[Callable] = None,
    ):
        """
        Initialize physics augmentor.
        
        Args:
            config: PhysicsAugmentorConfig instance
            simulator: CartPoleSimulator instance (created if None)
            controller_fn: Controller function (optional)
        """
        super().__init__(config)
        self.config: PhysicsAugmentorConfig = config
        
        # Create simulator if not provided
        if simulator is None:
            from src.simulators import CartPoleSimulator
            self.simulator = CartPoleSimulator()
        else:
            self.simulator = simulator
        
        self.controller_fn = controller_fn
        
        # State indices for Cart-Pole
        self._state_keys = ['x', 'x_dot', 'theta', 'theta_dot']
        self._param_keys = ['m_cart', 'm_pole', 'L', 'g']
    
    def _get_method_config(self) -> Dict:
        """Return method-specific configuration."""
        return {
            'ic_jitter_std': self.config.ic_jitter_std,
            'param_jitter_rel_std': self.config.param_jitter_rel_std,
            'jitter_mode': self.config.jitter_mode,
        }
    
    def _select_jitter_mode(self) -> str:
        """Select jitter mode based on config and enable flags."""
        import warnings
        
        # Build available modes based on enable flags
        available_modes = []
        if self.config.ic_jitter_enabled:
            available_modes.append('ic_only')
        if self.config.param_jitter_enabled:
            available_modes.append('param_only')
        if self.config.ic_jitter_enabled and self.config.param_jitter_enabled:
            available_modes.append('both')
        
        # If no modes available, return 'none' (will use original with analytic dx)
        if not available_modes:
            warnings.warn(
                "Both ic_jitter_enabled and param_jitter_enabled are False. "
                "Augmentation will return original trajectories with analytic dx. "
                "Set aug_ratio=0 if you want to skip augmentation entirely.",
                UserWarning
            )
            return 'none'
        
        if self.config.jitter_mode == 'random':
            probs = self.config.random_mode_probs
            # Filter to only enabled modes
            modes = [m for m in available_modes if m in probs]
            if not modes:
                modes = available_modes
            p = np.array([probs.get(m, 1.0) for m in modes])
            p = p / p.sum()  # Normalize
            return self.rng.choice(modes, p=p)
        
        # If requested mode is disabled, fall back to available
        if self.config.jitter_mode not in available_modes:
            return available_modes[0] if available_modes else 'none'
        
        return self.config.jitter_mode
    
    def _jitter_ic(self, x0: np.ndarray) -> np.ndarray:
        """
        Apply IC jitter to initial state.
        
        Args:
            x0: Original initial state [x, x_dot, theta, theta_dot]
        
        Returns:
            Jittered initial state
        """
        x0_new = x0.copy()
        std = self.config.ic_jitter_std
        
        for i, key in enumerate(self._state_keys):
            if key in std and std[key] > 0:
                x0_new[i] += self.rng.normal(0, std[key])
        
        # Wrap theta to (-π, π]
        x0_new[2] = self.simulator.wrap_angle(x0_new[2])
        
        return x0_new
    
    def _jitter_params(self, params: np.ndarray) -> Dict[str, float]:
        """
        Apply relative jitter to physical parameters.
        
        Args:
            params: Original parameters [m_cart, m_pole, L, g, ...]
        
        Returns:
            Dictionary of jittered parameters for simulator
        """
        rel_std = self.config.param_jitter_rel_std
        
        # Map params array to dict (assuming standard order)
        param_dict = {}
        for i, key in enumerate(self._param_keys):
            if i < len(params):
                val = params[i]
                if key in rel_std and rel_std[key] > 0:
                    # Relative jitter: val * (1 + N(0, rel_std))
                    jitter = 1.0 + self.rng.normal(0, rel_std[key])
                    jitter = max(0.5, min(1.5, jitter))  # Clamp to [0.5, 1.5]
                    val = val * jitter
                param_dict[key] = val
        
        return param_dict
    
    def _resimulate(
        self,
        x0: np.ndarray,
        params_dict: Dict[str, float],
        original_u: np.ndarray,
        T: int,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Re-simulate trajectory with given IC and parameters.
        
        Args:
            x0: Initial state
            params_dict: Physical parameters dict
            original_u: Original input sequence, shape (T, 1)
            T: Number of time steps
            dt: Time step
        
        Returns:
            (x, u, dx) - all shape (T, dim)
        """
        from src.simulators import CartPoleSimulator
        
        # Create simulator with new parameters
        sim = CartPoleSimulator(params=params_dict)
        
        # Create controller that replays original inputs
        # (with optional smoothing/interpolation if needed)
        t_span = (0.0, (T - 1) * dt)
        
        # Use original control sequence
        def replay_controller(t_val, x_val):
            idx = min(int(t_val / dt), T - 1)
            return original_u[idx, 0]
        
        # Simulate
        t_arr, x_arr, u_arr = sim.simulate(
            x0=x0,
            t_span=t_span,
            dt=dt,
            controller=replay_controller,
        )
        
        # Compute dx analytically using dynamics
        dx_arr = np.zeros_like(x_arr)
        for i in range(len(t_arr)):
            dx_arr[i] = sim.dynamics(t_arr[i], x_arr[i], u_arr[i, 0])
        
        return x_arr, u_arr, dx_arr
    
    def _check_quality(self, x: np.ndarray, dx: np.ndarray) -> bool:
        """
        Check if augmented trajectory passes quality filter.
        
        Args:
            x: State trajectory (T, 4)
            dx: Derivative trajectory (T, 4)
        
        Returns:
            True if passes quality check
        """
        # Check for NaN/Inf
        if not np.isfinite(x).all() or not np.isfinite(dx).all():
            return False
        
        # Check theta bounds
        if np.abs(x[:, 2]).max() > self.config.max_theta:
            return False
        
        return True
    
    def _augment_single(
        self,
        x: np.ndarray,
        u: np.ndarray,
        params: np.ndarray,
        traj_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Augment a single trajectory via re-simulation.
        
        Args:
            x: Original state trajectory, shape (T, 4)
            u: Original input trajectory, shape (T, 1)
            params: Original parameters, shape (n_params,)
            traj_idx: Index of trajectory
        
        Returns:
            (aug_x, aug_u, aug_dx, aug_params, aug_type)
        """
        T = x.shape[0]
        dt = self.config.dt
        n_params_original = len(params)  # Preserve original params length
        
        # Select jitter mode
        mode = self._select_jitter_mode()
        
        # Get initial condition and parameters
        x0 = x[0].copy()
        params_dict = {k: params[i] for i, k in enumerate(self._param_keys) if i < len(params)}
        
        # Fill missing params with defaults for simulation
        from src.simulators import CartPoleSimulator
        for k in self._param_keys:
            if k not in params_dict:
                params_dict[k] = CartPoleSimulator.DEFAULT_PARAMS.get(k, 1.0)
        
        # Handle 'none' mode - return original with analytic dx
        if mode == 'none':
            aug_x = x.copy()
            aug_u = u.copy()
            sim = CartPoleSimulator(params=params_dict)
            aug_dx = np.zeros_like(x)
            for i in range(T):
                aug_dx[i] = sim.dynamics(i * dt, x[i], u[i, 0])
            aug_params = np.zeros(n_params_original, dtype=np.float64)
            for i in range(n_params_original):
                if i < len(self._param_keys):
                    aug_params[i] = params_dict.get(self._param_keys[i], params[i])
                else:
                    aug_params[i] = params[i]
            return aug_x, aug_u, aug_dx, aug_params, 'none'
        
        # Apply jitter based on mode
        if mode == 'ic_only':
            x0_new = self._jitter_ic(x0)
            params_new = params_dict.copy()
            aug_type = 'ic_jitter'
        elif mode == 'param_only':
            x0_new = x0
            params_new = self._jitter_params(params)
            aug_type = 'param_jitter'
        else:  # 'both'
            x0_new = self._jitter_ic(x0)
            params_new = self._jitter_params(params)
            aug_type = 'ic_param_jitter'
        
        # Re-simulate with retry on quality failure
        max_retries = 5
        for retry in range(max_retries):
            try:
                aug_x, aug_u, aug_dx = self._resimulate(
                    x0_new, params_new, u, T, dt
                )
                
                if self._check_quality(aug_x, aug_dx):
                    break
                
                # On quality failure, re-jitter
                if mode in ['ic_only', 'both']:
                    x0_new = self._jitter_ic(x0)
                if mode in ['param_only', 'both']:
                    params_new = self._jitter_params(params)
                    
            except Exception as e:
                # On simulation failure, re-jitter
                if mode in ['ic_only', 'both']:
                    x0_new = self._jitter_ic(x0)
                if mode in ['param_only', 'both']:
                    params_new = self._jitter_params(params)
        else:
            # If all retries fail, return original (fallback)
            aug_x = x.copy()
            aug_u = u.copy()
            # Compute dx from original with correct time values
            sim = CartPoleSimulator(params=params_dict)
            aug_dx = np.zeros_like(x)
            for i in range(T):
                aug_dx[i] = sim.dynamics(i * dt, x[i], u[i, 0])
            aug_type = 'original_fallback'
            params_new = params_dict
        
        # Convert params_new dict back to array - SAME LENGTH as original
        aug_params = np.zeros(n_params_original, dtype=np.float64)
        for i in range(n_params_original):
            if i < len(self._param_keys):
                key = self._param_keys[i]
                aug_params[i] = params_new.get(key, params[i])
            else:
                aug_params[i] = params[i]  # Keep original for extra params
        
        return aug_x, aug_u, aug_dx, aug_params, aug_type


def create_physics_augmentor(
    aug_ratio: float = 1.0,
    seed: int = 42,
    dt: float = 0.02,
    T: int = 101,
    jitter_mode: str = 'both',
    ic_std_scale: float = 1.0,
    param_rel_std_scale: float = 1.0,
) -> PhysicsAugmentor:
    """
    Factory function to create PhysicsAugmentor with common settings.
    
    Args:
        aug_ratio: Augmentation ratio
        seed: Random seed
        dt: Time step
        T: Number of time steps
        jitter_mode: 'ic_only', 'param_only', 'both', or 'random'
        ic_std_scale: Scale factor for IC jitter std
        param_rel_std_scale: Scale factor for param jitter relative std
    
    Returns:
        Configured PhysicsAugmentor
    """
    # Scale default IC jitter
    ic_jitter_std = {
        'x': 0.05 * ic_std_scale,
        'x_dot': 0.05 * ic_std_scale,
        'theta': 0.02 * ic_std_scale,
        'theta_dot': 0.05 * ic_std_scale,
    }
    
    # Scale default param jitter
    param_jitter_rel_std = {
        'm_cart': 0.05 * param_rel_std_scale,
        'm_pole': 0.05 * param_rel_std_scale,
        'L': 0.02 * param_rel_std_scale,
        'g': 0.0,  # Never jitter gravity
    }
    
    config = PhysicsAugmentorConfig(
        aug_ratio=aug_ratio,
        seed=seed,
        dt=dt,
        T=T,
        jitter_mode=jitter_mode,
        ic_jitter_std=ic_jitter_std,
        param_jitter_rel_std=param_jitter_rel_std,
    )
    
    return PhysicsAugmentor(config)