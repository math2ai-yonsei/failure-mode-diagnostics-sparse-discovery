"""
Base simulator class for dynamical systems.
Provides common interface and utilities for all simulators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Callable
import numpy as np
from scipy.integrate import solve_ivp


class BaseSimulator(ABC):
    """
    Abstract base class for dynamical system simulators.
    
    All simulators must implement:
    - dynamics(): System equations of motion
    - state_dim: Dimension of state space
    - input_dim: Dimension of control input
    """
    
    def __init__(self, params: Dict[str, float]):
        """
        Args:
            params: Dictionary of physical parameters
        """
        self.params = params
        self._validate_params()
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state space."""
        pass
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimension of control input."""
        pass
    
    @abstractmethod
    def dynamics(self, t: float, state: np.ndarray, u: float) -> np.ndarray:
        """
        Compute state derivatives.
        
        Args:
            t: Current time
            state: Current state vector
            u: Control input
            
        Returns:
            State derivatives (dx/dt)
        """
        pass
    
    @abstractmethod
    def _validate_params(self) -> None:
        """Validate physical parameters."""
        pass
    
    @abstractmethod
    def wrap_state(self, state: np.ndarray) -> np.ndarray:
        """
        Apply any state wrapping (e.g., angle to (-π, π]).
        
        Args:
            state: State vector or trajectory
            
        Returns:
            Wrapped state
        """
        pass
    
    def simulate(
        self,
        x0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float,
        controller: Optional[Callable[[float, np.ndarray], float]] = None,
        method: str = 'RK45',
        rtol: float = 1e-8,
        atol: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate system dynamics.
        
        Control input handling:
        - During integration: controller is called at each integrator step
        - Output u_history: computed POST-HOC at t_eval points using final x
        - This ensures (t, x, u) triplets are consistent at sample times
        
        Args:
            x0: Initial state
            t_span: (t_start, t_end)
            dt: Time step for output
            controller: Function (t, x) -> u, or None for zero input
            method: Integration method
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            t: Time vector (T,)
            x: State trajectory (T, state_dim)
            u: Control input trajectory (T, input_dim)
        """
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (self.state_dim,):
            raise ValueError(f"x0 must have shape ({self.state_dim},), got {x0.shape}")
        
        # Generate evaluation times
        t_eval = np.arange(t_span[0], t_span[1] + dt/2, dt)
        n_steps = len(t_eval)
        
        # Dynamics wrapper for integration
        def dynamics_wrapper(t, state):
            if controller is not None:
                u = controller(t, state)
            else:
                u = 0.0
            return self.dynamics(t, state, u)
        
        # Integrate
        sol = solve_ivp(
            dynamics_wrapper,
            t_span,
            x0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol
        )
        
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        
        # Transpose to (T, state_dim) and wrap angles
        x = sol.y.T.astype(np.float64)
        
        # Check for NaN
        if np.any(np.isnan(x)):
            raise RuntimeError("Integration produced NaN values")
        
        x = self.wrap_state(x)
        
        # Compute u_history POST-HOC at evaluation times
        # This ensures (t[i], x[i], u[i]) are consistent triplets
        u_history = np.zeros((n_steps, self.input_dim), dtype=np.float64)
        if controller is not None:
            for i in range(n_steps):
                u_history[i, 0] = controller(t_eval[i], x[i])
        
        return t_eval.astype(np.float64), x, u_history
    
    def simulate_batch(
        self,
        x0_batch: np.ndarray,
        t_span: Tuple[float, float],
        dt: float,
        controller: Optional[Callable[[float, np.ndarray], float]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate multiple trajectories.
        
        Args:
            x0_batch: Initial states (N, state_dim)
            t_span: (t_start, t_end)
            dt: Time step
            controller: Control function or None
            **kwargs: Additional arguments for simulate()
            
        Returns:
            t: Time vector (T,)
            x_batch: Trajectories (N, T, state_dim)
            u_batch: Control inputs (N, T, input_dim)
        """
        n_traj = x0_batch.shape[0]
        
        # Get time vector length
        t_test = np.arange(t_span[0], t_span[1] + dt/2, dt)
        n_steps = len(t_test)
        
        x_batch = np.zeros((n_traj, n_steps, self.state_dim), dtype=np.float64)
        u_batch = np.zeros((n_traj, n_steps, self.input_dim), dtype=np.float64)
        
        for i in range(n_traj):
            t, x, u = self.simulate(x0_batch[i], t_span, dt, controller, **kwargs)
            x_batch[i] = x
            u_batch[i] = u
        
        return t, x_batch, u_batch
    
    @staticmethod
    def wrap_angle(theta: np.ndarray) -> np.ndarray:
        """
        Wrap angle to (-π, π].
        
        Args:
            theta: Angle(s) in radians
            
        Returns:
            Wrapped angle(s)
        """
        wrapped = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        wrapped = np.where(wrapped == -np.pi, np.pi, wrapped)
        return wrapped