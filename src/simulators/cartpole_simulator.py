"""
Cart-Pole simulator with accurate nonlinear dynamics.

State definition:
    index 0: x (m) - cart position
    index 1: x_dot (m/s) - cart velocity
    index 2: theta (rad) - pole angle from vertical (up = 0), wrapped to (-π, π]
    index 3: theta_dot (rad/s) - pole angular velocity

Sign convention:
    - Positive x: rightward
    - Positive theta: counterclockwise from vertical
    - Positive u (force): rightward on cart
    
Model: Point mass at pole tip (not uniform rod)
"""

from typing import Dict, Optional, Callable, Tuple
import numpy as np
from .base_simulator import BaseSimulator


class CartPoleSimulator(BaseSimulator):
    """
    Nonlinear Cart-Pole simulator.
    
    Uses Euler-Lagrange equations for cart-pole system with point mass pole.
    Supports external force input on cart.
    
    Default parameters (standard benchmark):
        m_cart: 1.0 kg (cart mass)
        m_pole: 0.1 kg (pole mass, concentrated at tip)
        L: 0.5 m (pole length to mass)
        g: 9.81 m/s² (gravity)
        b_cart: 0.0 (cart friction coefficient)
        b_pole: 0.0 (pole friction coefficient)
    """
    
    REQUIRED_PARAMS = ['m_cart', 'm_pole', 'L', 'g']
    
    DEFAULT_PARAMS = {
        'm_cart': 1.0,
        'm_pole': 0.1,
        'L': 0.5,
        'g': 9.81,
        'b_cart': 0.0,
        'b_pole': 0.0,
    }
    
    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize Cart-Pole simulator.
        
        Args:
            params: Physical parameters dict. Missing params use defaults.
        """
        full_params = self.DEFAULT_PARAMS.copy()
        if params is not None:
            full_params.update(params)
        
        super().__init__(full_params)
        
        self._mc = self.params['m_cart']
        self._mp = self.params['m_pole']
        self._L = self.params['L']
        self._g = self.params['g']
        self._b_cart = self.params['b_cart']
        self._b_pole = self.params['b_pole']
        self._mt = self._mc + self._mp
    
    @property
    def state_dim(self) -> int:
        return 4
    
    @property
    def input_dim(self) -> int:
        return 1
    
    def _validate_params(self) -> None:
        """Check that all required parameters are positive."""
        for param in self.REQUIRED_PARAMS:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
            if self.params[param] <= 0:
                raise ValueError(f"Parameter {param} must be positive, got {self.params[param]}")
        
        for param in ['b_cart', 'b_pole']:
            if param in self.params and self.params[param] < 0:
                raise ValueError(f"Parameter {param} must be non-negative")
    
    def dynamics(self, t: float, state: np.ndarray, u: float) -> np.ndarray:
        """
        Compute Cart-Pole dynamics using Euler-Lagrange equations.
        
        Point mass model:
        - Pole mass concentrated at distance L from pivot
        - No rotational inertia term (I = m_p * L^2 for point mass)
        
        Equations:
            (m_c + m_p) x_ddot + m_p L theta_ddot cos(theta) = u + m_p L theta_dot^2 sin(theta)
            L x_ddot cos(theta) + L^2 theta_ddot = g L sin(theta)
        
        Args:
            t: Current time (unused)
            state: [x, x_dot, theta, theta_dot]
            u: Force applied to cart (N)
            
        Returns:
            [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x, x_dot, theta, theta_dot = state
        
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        mc, mp, L, g = self._mc, self._mp, self._L, self._g
        mt = self._mt
        
        # Denominator: L * (m_c + m_p * sin^2(theta))
        denom = L * (mc + mp * sin_t**2)
        
        # Add friction terms
        u_eff = u - self._b_cart * x_dot
        tau_pole = -self._b_pole * theta_dot
        
        # Centrifugal term
        centrifugal = mp * L * theta_dot**2 * sin_t
        
        # Solve 2x2 system using Cramer's rule
        # [m_c + m_p,  m_p*L*cos(θ)] [x_ddot    ]   [u_eff + centrifugal    ]
        # [cos(θ),     L            ] [theta_ddot] = [g*sin(θ) + tau_pole/mpL]
        
        # x_ddot = (L * rhs1 - m_p*L*cos(θ) * rhs2) / denom
        # theta_ddot = ((m_c + m_p) * rhs2 - cos(θ) * rhs1) / denom
        
        rhs1 = u_eff + centrifugal
        rhs2 = g * sin_t + tau_pole / (mp * L) if mp * L > 0 else g * sin_t
        
        x_ddot = (L * rhs1 - mp * L * cos_t * rhs2) / denom
        theta_ddot = (mt * rhs2 - cos_t * rhs1) / denom
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=np.float64)
    
    def wrap_state(self, state: np.ndarray) -> np.ndarray:
        """
        Wrap theta to (-π, π].
        
        Args:
            state: State array, shape (..., 4) or (4,)
            
        Returns:
            State with wrapped theta
        """
        state = state.copy()
        if state.ndim == 1:
            state[2] = self.wrap_angle(state[2])
        else:
            state[..., 2] = self.wrap_angle(state[..., 2])
        return state
    
    def energy(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute system energy (point mass model).
        
        Kinetic energy:
            T = 0.5 * m_c * x_dot^2 + 0.5 * m_p * (v_pole_x^2 + v_pole_y^2)
            
        Potential energy:
            V = m_p * g * L * cos(theta)
            (zero at theta = π/2, max at theta = 0)
        
        Args:
            state: State array, shape (..., 4)
            
        Returns:
            Dict with 'kinetic', 'potential', 'total' energies
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        x_dot = state[..., 1]
        theta = state[..., 2]
        theta_dot = state[..., 3]
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Pole mass velocity components
        v_pole_x = x_dot + self._L * theta_dot * cos_t
        v_pole_y = -self._L * theta_dot * sin_t
        
        # Kinetic energy
        KE_cart = 0.5 * self._mc * x_dot**2
        KE_pole = 0.5 * self._mp * (v_pole_x**2 + v_pole_y**2)
        KE = KE_cart + KE_pole
        
        # Potential energy (reference: theta = π/2)
        PE = self._mp * self._g * self._L * cos_t
        
        total = KE + PE
        
        if squeeze:
            return {'kinetic': KE[0], 'potential': PE[0], 'total': total[0]}
        return {'kinetic': KE, 'potential': PE, 'total': total}
    
    def get_stable_equilibrium(self) -> np.ndarray:
        """Return stable equilibrium state (pole down)."""
        return np.array([0.0, 0.0, np.pi, 0.0])
    
    def get_unstable_equilibrium(self) -> np.ndarray:
        """Return unstable equilibrium state (pole up)."""
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    def sample_initial_state(
        self,
        rng: np.random.Generator,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        x_dot_range: Tuple[float, float] = (-0.5, 0.5),
        theta_range: Tuple[float, float] = (-0.3, 0.3),
        theta_dot_range: Tuple[float, float] = (-0.5, 0.5),
        near_equilibrium: str = 'unstable'
    ) -> np.ndarray:
        """
        Sample random initial state.
        
        Args:
            rng: NumPy random generator
            x_range: Cart position range
            x_dot_range: Cart velocity range
            theta_range: Pole angle range (offset from equilibrium)
            theta_dot_range: Pole angular velocity range
            near_equilibrium: 'stable' (down) or 'unstable' (up)
            
        Returns:
            Initial state [x, x_dot, theta, theta_dot]
        """
        x0 = rng.uniform(*x_range)
        x_dot0 = rng.uniform(*x_dot_range)
        theta_offset = rng.uniform(*theta_range)
        theta_dot0 = rng.uniform(*theta_dot_range)
        
        if near_equilibrium == 'stable':
            theta0 = np.pi + theta_offset
        else:
            theta0 = theta_offset
        
        state = np.array([x0, x_dot0, theta0, theta_dot0], dtype=np.float64)
        return self.wrap_state(state)