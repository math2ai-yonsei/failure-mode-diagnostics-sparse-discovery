"""
AEK Self-balancing Motorcycle simulator.
Reaction Wheel Inverted Pendulum with 1-DOF roll dynamics.

Source: MathWorks Arduino Engineering Kit Rev2 (exercise_6_1/modelParameters.m)
Reference: configs/systems/aek.yaml (SSOT for all physical parameters)
Parameter traceability: docs/aek_parameters_source.md

State definition:
    index 0: phi (rad) - Lean/roll angle (body tilt from vertical, + = CCW from behind)
    index 1: phi_dot (rad/s) - Lean angular velocity
    index 2: theta_w (rad) - Reaction wheel angle (RELATIVE to body, + = CCW)
    index 3: theta_w_dot (rad/s) - Reaction wheel angular velocity (relative)

Sign convention:
    - Positive phi: CCW lean viewed from behind
    - Positive theta_w: CCW wheel rotation relative to body
    - Positive tau: motor torque accelerating wheel CCW; reaction on body = -tau
    - Gravity torque +M*g*h*sin(phi) destabilizes the body

Notation mapping (MathWorks Source <-> This project):
    MathWorks theta     -> Our phi        (lean/roll angle)
    MathWorks phi (abs) -> Our theta_w    (relative wheel angle)
    theta_w = phi_abs - phi  (body-frame relative angle)

Equations of motion:
    phi_ddot     = (M_total * g * h_cm * sin(phi) - tau) / I_p
    theta_w_ddot = tau / I_w_C - phi_ddot

OOD parameter: I_w_C (wheel spin inertia)
    When I_w_C changes (flywheel swap), derived quantities are recomputed:
    m_w, I_w_A, I_p, M_total, h_cm
"""

from typing import Dict, Optional, Callable, Tuple
import numpy as np
from .base_simulator import BaseSimulator


class AEKSimulator(BaseSimulator):
    """
    AEK Self-balancing Motorcycle Simulator.

    Reaction wheel inverted pendulum with 1-DOF roll dynamics.
    Based on MathWorks Arduino Engineering Kit Rev2 parameters.

    Default parameters (nominal, from modelParameters.m):
        m_r: 0.2948 kg (rod/body mass)
        R: 0.05 m (wheel radius)
        r: 0.02 m (rod cross-section radius)
        l: 0.13 m (rod length)
        I_w_C: 8.6875e-05 kg*m^2 (wheel spin inertia, OOD knob)
        g: 9.80665 m/s^2 (gravity)
        tau_max: 0.02 N*m (motor torque limit)
    """

    REQUIRED_PARAMS = ['m_r', 'R', 'r', 'l', 'I_w_C', 'g']

    DEFAULT_PARAMS = {
        'm_r': 0.2948,          # rod mass (kg) - modelParameters.m L5
        'R': 0.05,              # wheel radius (m) - L8
        'r': 0.02,              # rod cross-section radius (m) - L9
        'l': 0.13,              # rod length (m) - L10
        'I_w_C': 8.6875e-05,    # wheel spin inertia (kg*m^2) - OOD knob
        'g': 9.80665,           # gravity (m/s^2)
        'tau_max': 0.02,        # motor torque limit (N*m)
    }

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize AEK simulator.

        Args:
            params: Physical parameters dict. Missing params use defaults.
                    Key OOD parameter: I_w_C (wheel spin inertia).
                    When I_w_C differs from nominal, derived quantities
                    (m_w, I_p, M_total, h_cm) are recomputed consistently.
        """
        full_params = self.DEFAULT_PARAMS.copy()
        if params is not None:
            full_params.update(params)

        super().__init__(full_params)

        # Cache base parameters
        self._m_r = self.params['m_r']
        self._R = self.params['R']
        self._r = self.params['r']
        self._l = self.params['l']
        self._I_w_C = self.params['I_w_C']
        self._g = self.params['g']
        self._tau_max = self.params['tau_max']

        # Geometric derived quantities (fixed, independent of I_w_C)
        self._l_AB = self._l / 2.0    # rod CoM height from pivot
        self._l_AC = self._l           # wheel mount height = rod length

        # Rod inertia about own center (B): thin cylinder
        # I_r_B = (1/12) * m_r * (3*r^2 + l^2)
        self._I_r_B = (1.0 / 12.0) * self._m_r * (3.0 * self._r**2 + self._l**2)

        # Rod inertia about pivot (A): parallel axis theorem
        # I_r_A = I_r_B + m_r * l_AB^2
        self._I_r_A = self._I_r_B + self._m_r * self._l_AB**2

        # Compute I_w_C-dependent derived quantities
        self._recompute_derived()

    def _recompute_derived(self):
        """
        Recompute derived quantities from I_w_C.

        Physical basis: swapping flywheel changes wheel mass (m_w = 2*I_w_C/R^2),
        which cascades to total mass, CoM height, and pivot inertia.
        """
        # Wheel mass from I_w_C: I_w_C = 0.5 * m_w * R^2  =>  m_w = 2*I_w_C/R^2
        self._m_w = 2.0 * self._I_w_C / self._R**2

        # Wheel inertia about pivot (A): parallel axis theorem
        self._I_w_A = self._I_w_C + self._m_w * self._l_AC**2

        # Total system inertia about pivot (for phi equation)
        # I_p = I_r_A + m_w * l_AC^2
        # NOTE: This is NOT I_r_A + I_w_A (that would double-count I_w_C)
        self._I_p = self._I_r_A + self._m_w * self._l_AC**2

        # Total mass
        self._M_total = self._m_r + self._m_w

        # System CoM height from pivot
        self._h_cm = (
            (self._m_r * self._l_AB + self._m_w * self._l_AC) / self._M_total
        )

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def input_dim(self) -> int:
        return 1

    def _validate_params(self) -> None:
        """Validate physical parameters (all required params must be positive)."""
        for param in self.REQUIRED_PARAMS:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
            if self.params[param] <= 0:
                raise ValueError(
                    f"Parameter {param} must be positive, got {self.params[param]}"
                )

        if 'tau_max' in self.params and self.params['tau_max'] < 0:
            raise ValueError("tau_max must be non-negative")

    def dynamics(self, t: float, state: np.ndarray, u: float) -> np.ndarray:
        """
        Compute AEK dynamics.

        Equations of motion (phi=lean, theta_w=wheel relative):
            phi_ddot     = (M_total * g * h_cm * sin(phi) - tau) / I_p
            theta_w_ddot = tau / I_w_C - phi_ddot

        The coupling term (-phi_ddot) in theta_w_ddot arises because theta_w
        is measured in the body frame (relative angle).

        Args:
            t: Current time (unused, for ODE solver interface)
            state: [phi, phi_dot, theta_w, theta_w_dot]
            u: tau - motor torque applied to wheel (N*m)

        Returns:
            [phi_dot, phi_ddot, theta_w_dot, theta_w_ddot]
        """
        phi, phi_dot, theta_w, theta_w_dot = state

        # Clip torque to motor limits
        tau = np.clip(u, -self._tau_max, self._tau_max)

        # Body equation (torque balance about pivot A)
        # Gravity destabilizes (+), motor reaction stabilizes (-)
        phi_ddot = (
            self._M_total * self._g * self._h_cm * np.sin(phi) - tau
        ) / self._I_p

        # Wheel equation (relative to body, includes coupling term)
        theta_w_ddot = tau / self._I_w_C - phi_ddot

        return np.array(
            [phi_dot, phi_ddot, theta_w_dot, theta_w_ddot], dtype=np.float64
        )

    def wrap_state(self, state: np.ndarray) -> np.ndarray:
        """
        Wrap phi (lean angle) to (-pi, pi].
        theta_w is NOT wrapped (wheel can spin continuously).

        Args:
            state: State array, shape (..., 4) or (4,)

        Returns:
            State with wrapped phi
        """
        state = state.copy()
        if state.ndim == 1:
            state[0] = self.wrap_angle(state[0])
        else:
            state[..., 0] = self.wrap_angle(state[..., 0])
        return state

    def energy(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute system energy.

        Kinetic energy (Lagrangian derivation):
            KE = 0.5 * I_p * phi_dot^2
               + 0.5 * I_w_C * (phi_dot + theta_w_dot)^2

            First term: body + wheel orbital KE about pivot
            Second term: wheel spin KE
              (absolute wheel angular velocity = phi_dot + theta_w_dot)

        Potential energy:
            PE = M_total * g * h_cm * cos(phi)
            (max at phi=0 upright, min at phi=pi hanging)

        Total energy is conserved when tau=0 (no motor input, no friction).

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

        phi = state[..., 0]
        phi_dot = state[..., 1]
        theta_w_dot = state[..., 3]

        # Kinetic energy
        KE_orbital = 0.5 * self._I_p * phi_dot**2
        omega_wheel_abs = phi_dot + theta_w_dot
        KE_spin = 0.5 * self._I_w_C * omega_wheel_abs**2
        KE = KE_orbital + KE_spin

        # Potential energy (CoM height above pivot reference)
        PE = self._M_total * self._g * self._h_cm * np.cos(phi)

        total = KE + PE

        if squeeze:
            return {'kinetic': KE[0], 'potential': PE[0], 'total': total[0]}
        return {'kinetic': KE, 'potential': PE, 'total': total}

    def get_unstable_equilibrium(self) -> np.ndarray:
        """Return unstable equilibrium state (body upright, phi=0)."""
        return np.array([0.0, 0.0, 0.0, 0.0])

    def get_stable_equilibrium(self) -> np.ndarray:
        """Return stable equilibrium state (body hanging down, phi=pi)."""
        return np.array([np.pi, 0.0, 0.0, 0.0])

    def sample_initial_state(
        self,
        rng: np.random.Generator,
        phi_range: Tuple[float, float] = (-0.15, 0.15),
        phi_dot_range: Tuple[float, float] = (-1.0, 1.0),
        theta_w_range: Tuple[float, float] = (-1.0, 1.0),
        theta_w_dot_range: Tuple[float, float] = (-10.0, 10.0),
        near_equilibrium: str = 'unstable',
    ) -> np.ndarray:
        """
        Sample random initial state.

        Default ranges from aek.yaml simulation.initial_conditions.

        Args:
            rng: NumPy random generator
            phi_range: Lean angle range (offset from equilibrium)
            phi_dot_range: Lean angular velocity range
            theta_w_range: Wheel angle range
            theta_w_dot_range: Wheel angular velocity range
            near_equilibrium: 'stable' (hanging) or 'unstable' (upright)

        Returns:
            Initial state [phi, phi_dot, theta_w, theta_w_dot]
        """
        phi_offset = rng.uniform(*phi_range)
        phi_dot0 = rng.uniform(*phi_dot_range)
        theta_w0 = rng.uniform(*theta_w_range)
        theta_w_dot0 = rng.uniform(*theta_w_dot_range)

        if near_equilibrium == 'stable':
            phi0 = np.pi + phi_offset
        else:
            phi0 = phi_offset  # Near upright (phi=0)

        state = np.array(
            [phi0, phi_dot0, theta_w0, theta_w_dot0], dtype=np.float64
        )
        return self.wrap_state(state)

    def get_derived_params(self) -> Dict[str, float]:
        """
        Return all derived physical parameters for logging/verification.

        Useful for:
        - manifest.json recording
        - MATLAB cross-validation
        - Verifying I_w_C-dependent recalculations

        Returns:
            Dict with derived quantities and their values
        """
        omega_n = np.sqrt(
            self._M_total * self._g * self._h_cm / self._I_p
        )
        return {
            # Base params
            'm_r': self._m_r,
            'R': self._R,
            'r': self._r,
            'l': self._l,
            'I_w_C': self._I_w_C,
            'g': self._g,
            'tau_max': self._tau_max,
            # Geometric
            'l_AB': self._l_AB,
            'l_AC': self._l_AC,
            # Inertias
            'I_r_B': self._I_r_B,
            'I_r_A': self._I_r_A,
            'I_w_A': self._I_w_A,
            'I_p': self._I_p,
            # Mass / CoM
            'm_w': self._m_w,
            'M_total': self._M_total,
            'h_cm': self._h_cm,
            # Dynamics characteristics
            'I_w_C_over_I_p': self._I_w_C / self._I_p,
            'omega_n_rad_s': omega_n,
            'time_to_fall_s': 1.0 / omega_n,
        }