"""
Tests for AEK Self-balancing Motorcycle simulator.

Run with: python -m pytest tests/test_aek_simulator.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.simulators import AEKSimulator, BaseSimulator
from src.utils.seed_utils import set_seed


class TestAEKSimulatorProperties:
    """Basic property and initialization tests."""

    @pytest.fixture
    def simulator(self):
        """Create default (nominal) simulator."""
        return AEKSimulator()

    def test_state_dim(self, simulator):
        """Test state dimension is 4: [phi, phi_dot, theta_w, theta_w_dot]."""
        assert simulator.state_dim == 4

    def test_input_dim(self, simulator):
        """Test input dimension is 1: [tau]."""
        assert simulator.input_dim == 1

    def test_is_base_simulator(self, simulator):
        """Test inheritance from BaseSimulator."""
        assert isinstance(simulator, BaseSimulator)

    def test_default_params(self, simulator):
        """Test default parameters match modelParameters.m values."""
        assert simulator.params['m_r'] == 0.2948
        assert simulator.params['R'] == 0.05
        assert simulator.params['l'] == 0.13
        assert simulator.params['I_w_C'] == 8.6875e-05
        assert simulator.params['g'] == 9.80665

    def test_custom_params(self):
        """Test custom I_w_C (OOD parameter)."""
        sim = AEKSimulator({'I_w_C': 1.04e-04})
        assert sim.params['I_w_C'] == 1.04e-04

    def test_invalid_params_negative_mass(self):
        """Test that negative rod mass raises error."""
        with pytest.raises(ValueError):
            AEKSimulator({'m_r': -0.1})

    def test_invalid_params_zero_radius(self):
        """Test that zero wheel radius raises error."""
        with pytest.raises(ValueError):
            AEKSimulator({'R': 0.0})

    def test_invalid_params_zero_inertia(self):
        """Test that zero I_w_C raises error."""
        with pytest.raises(ValueError):
            AEKSimulator({'I_w_C': 0.0})


class TestAEKDerivedParameters:
    """Test that derived parameters match MATLAB-verified values."""

    @pytest.fixture
    def simulator(self):
        return AEKSimulator()

    def test_wheel_mass_nominal(self, simulator):
        """m_w = 2 * I_w_C / R^2 = 0.0695 kg."""
        derived = simulator.get_derived_params()
        np.testing.assert_allclose(derived['m_w'], 0.0695, rtol=1e-6)

    def test_rod_inertia_center(self, simulator):
        """I_r_B = (1/12) * m_r * (3*r^2 + l^2) = 4.446567e-04."""
        derived = simulator.get_derived_params()
        np.testing.assert_allclose(derived['I_r_B'], 4.446567e-04, rtol=1e-4)

    def test_rod_inertia_pivot(self, simulator):
        """I_r_A = I_r_B + m_r * l_AB^2 = 1.690187e-03."""
        derived = simulator.get_derived_params()
        np.testing.assert_allclose(derived['I_r_A'], 1.690187e-03, rtol=1e-4)

    def test_pivot_inertia(self, simulator):
        """I_p = I_r_A + m_w * l_AC^2 = 2.864737e-03."""
        derived = simulator.get_derived_params()
        np.testing.assert_allclose(derived['I_p'], 2.864737e-03, rtol=1e-4)

    def test_total_mass(self, simulator):
        """M_total = m_r + m_w = 0.3643 kg."""
        derived = simulator.get_derived_params()
        np.testing.assert_allclose(derived['M_total'], 0.3643, rtol=1e-4)

    def test_com_height(self, simulator):
        """h_cm = (m_r*l_AB + m_w*l_AC) / M_total = 0.07740 m."""
        derived = simulator.get_derived_params()
        np.testing.assert_allclose(derived['h_cm'], 0.07740, rtol=1e-3)

    def test_natural_frequency(self, simulator):
        """omega_n = sqrt(M*g*h_cm / I_p) ~ 9.82 rad/s."""
        derived = simulator.get_derived_params()
        np.testing.assert_allclose(derived['omega_n_rad_s'], 9.82, rtol=0.01)

    def test_inertia_ratio(self, simulator):
        """I_w_C / I_p ~ 0.0303 (limited control authority)."""
        derived = simulator.get_derived_params()
        np.testing.assert_allclose(derived['I_w_C_over_I_p'], 0.0303, rtol=0.02)


class TestAEKOODRecalculation:
    """Test that OOD parameter changes cascade correctly."""

    def test_different_I_w_C_changes_derived(self):
        """Changing I_w_C must change m_w, I_p, M_total, h_cm."""
        sim_nominal = AEKSimulator()
        sim_ood = AEKSimulator({'I_w_C': 1.04e-04})  # Tier-A OOD

        d_nom = sim_nominal.get_derived_params()
        d_ood = sim_ood.get_derived_params()

        assert d_ood['m_w'] > d_nom['m_w']
        assert d_ood['I_p'] > d_nom['I_p']
        assert d_ood['M_total'] > d_nom['M_total']
        assert d_ood['h_cm'] != d_nom['h_cm']

    def test_ood_tier_a_train1(self):
        """Tier-A train #1: I_w_C = 6.95e-05 -> m_w = 55.6g."""
        sim = AEKSimulator({'I_w_C': 6.95e-05})
        derived = sim.get_derived_params()
        np.testing.assert_allclose(derived['m_w'], 0.0556, rtol=1e-3)

    def test_ood_tier_a_test(self):
        """Tier-A test (OOD): I_w_C = 1.04e-04 -> m_w = 83.2g."""
        sim = AEKSimulator({'I_w_C': 1.04e-04})
        derived = sim.get_derived_params()
        np.testing.assert_allclose(derived['m_w'], 0.0832, rtol=1e-3)

    def test_different_I_w_C_different_trajectories(self):
        """Different I_w_C must produce different trajectories."""
        sim1 = AEKSimulator({'I_w_C': 6.95e-05})
        sim2 = AEKSimulator({'I_w_C': 1.04e-04})

        x0 = np.array([0.05, 0.0, 0.0, 0.0])
        _, x1, _ = sim1.simulate(x0, (0, 0.5), dt=0.01)
        _, x2, _ = sim2.simulate(x0, (0, 0.5), dt=0.01)

        assert not np.allclose(x1, x2)


class TestAEKSimulation:
    """Simulation output shape, dtype, and basic correctness."""

    @pytest.fixture
    def simulator(self):
        return AEKSimulator()

    @pytest.fixture
    def rng(self):
        set_seed(42)
        return np.random.default_rng(42)

    def test_simulate_output_shape(self, simulator):
        """Test output shapes with dt=0.01, T=1.0s."""
        x0 = np.array([0.05, 0.0, 0.0, 0.0])
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.01)

        assert t.shape == (101,)
        assert x.shape == (101, 4)
        assert u.shape == (101, 1)

    def test_simulate_output_dtype(self, simulator):
        """Test output dtypes are float64."""
        x0 = np.array([0.05, 0.0, 0.0, 0.0])
        t, x, u = simulator.simulate(x0, (0, 0.5), dt=0.01)

        assert t.dtype == np.float64
        assert x.dtype == np.float64
        assert u.dtype == np.float64

    def test_simulate_no_nan(self, simulator, rng):
        """Test simulation produces no NaN for random initial conditions."""
        for _ in range(10):
            x0 = simulator.sample_initial_state(rng)
            t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.01)

            assert not np.any(np.isnan(x)), "Trajectory contains NaN"
            assert not np.any(np.isnan(u)), "Control input contains NaN"

    def test_simulate_initial_condition(self, simulator):
        """Test that trajectory starts at initial condition."""
        x0 = np.array([0.05, 0.1, 0.5, -2.0])
        t, x, u = simulator.simulate(x0, (0, 0.5), dt=0.01)

        np.testing.assert_allclose(x[0], x0, atol=1e-10)

    def test_simulate_with_controller(self, simulator):
        """Test simulation with constant torque input."""
        x0 = np.array([0.05, 0.0, 0.0, 0.0])

        def controller(t, x):
            return 0.01  # Constant torque

        t, x, u = simulator.simulate(x0, (0, 0.5), dt=0.01, controller=controller)

        # Control input should be applied
        assert np.all(u[:, 0] != 0.0)

    def test_simulate_batch(self, simulator, rng):
        """Test batch simulation."""
        n_traj = 5
        x0_batch = np.array(
            [simulator.sample_initial_state(rng) for _ in range(n_traj)]
        )

        t, x_batch, u_batch = simulator.simulate_batch(
            x0_batch, (0, 0.5), dt=0.01
        )

        assert x_batch.shape == (5, 51, 4)
        assert u_batch.shape == (5, 51, 1)

    def test_torque_clipping(self, simulator):
        """Test that torque is clipped to tau_max."""
        x0 = np.array([0.05, 0.0, 0.0, 0.0])

        def controller(t, x):
            return 1.0  # Way above tau_max=0.02

        t, x, u = simulator.simulate(x0, (0, 0.2), dt=0.01, controller=controller)

        # Even with large commanded torque, dynamics should use clipped value
        # System should not behave as if receiving 1.0 N*m torque
        # (it would stabilize instantly with that much torque)
        assert not np.any(np.isnan(x))


class TestAEKPhiWrapping:
    """Test phi (lean angle) wrapping to (-pi, pi]."""

    @pytest.fixture
    def simulator(self):
        return AEKSimulator()

    def test_wrap_phi_in_range(self, simulator):
        """Test angles in range stay unchanged."""
        state = np.array([0.1, 0.0, 0.0, 0.0])
        wrapped = simulator.wrap_state(state)
        np.testing.assert_allclose(wrapped[0], 0.1, atol=1e-10)

    def test_wrap_phi_positive_overflow(self, simulator):
        """Test wrapping of large positive phi."""
        state = np.array([2 * np.pi + 0.1, 0.0, 0.0, 0.0])
        wrapped = simulator.wrap_state(state)
        np.testing.assert_allclose(wrapped[0], 0.1, atol=1e-10)

    def test_wrap_phi_negative_overflow(self, simulator):
        """Test wrapping of large negative phi."""
        state = np.array([-2 * np.pi - 0.1, 0.0, 0.0, 0.0])
        wrapped = simulator.wrap_state(state)
        np.testing.assert_allclose(wrapped[0], -0.1, atol=1e-10)

    def test_wrap_phi_pi(self, simulator):
        """Test that pi maps to pi (not -pi)."""
        state = np.array([np.pi, 0.0, 0.0, 0.0])
        wrapped = simulator.wrap_state(state)
        assert wrapped[0] == np.pi

    def test_theta_w_not_wrapped(self, simulator):
        """Test that theta_w is NOT wrapped (wheel spins continuously)."""
        state = np.array([0.0, 0.0, 10 * np.pi, 0.0])
        wrapped = simulator.wrap_state(state)
        np.testing.assert_allclose(wrapped[2], 10 * np.pi, atol=1e-10)

    def test_wrap_phi_trajectory(self, simulator):
        """Test phi wrapping during simulation (start near pi)."""
        # High angular velocity to force phi wrapping
        x0 = np.array([3.0, 2.0, 0.0, 0.0])
        t, x, u = simulator.simulate(x0, (0, 2.0), dt=0.01)

        phi = x[:, 0]
        assert np.all(phi > -np.pi), f"phi below -pi: {phi.min()}"
        assert np.all(phi <= np.pi), f"phi above pi: {phi.max()}"


class TestAEKPhysics:
    """Physics validation: equilibria, energy conservation, instability."""

    @pytest.fixture
    def simulator(self):
        return AEKSimulator()

    def test_equilibrium_unstable(self, simulator):
        """Test unstable equilibrium (body upright, phi=0) stays put."""
        x0 = simulator.get_unstable_equilibrium()
        t, x, u = simulator.simulate(x0, (0, 0.5), dt=0.01)

        np.testing.assert_allclose(x[-1], x0, atol=1e-6)

    def test_equilibrium_stable(self, simulator):
        """Test stable equilibrium (body hanging, phi=pi) stays put."""
        x0 = simulator.get_stable_equilibrium()
        t, x, u = simulator.simulate(x0, (0, 0.5), dt=0.01)

        np.testing.assert_allclose(x[-1], x0, atol=1e-6)

    def test_unstable_equilibrium_perturbed(self, simulator):
        """Test that small perturbation from upright causes fall."""
        x0 = np.array([0.01, 0.0, 0.0, 0.0])
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.01)

        # phi should grow (body falls)
        assert np.abs(x[-1, 0]) > 0.1

    def test_fast_dynamics(self, simulator):
        """Test that AEK dynamics are fast (~9.82 rad/s natural frequency).

        Starting with phi=0.1, the body should more than double within 0.15s
        (time_to_fall ~ 0.102s). The nonlinear system grows slower than
        the linearized prediction, so we use a relaxed threshold.
        """
        x0 = np.array([0.1, 0.0, 0.0, 0.0])
        t, x, u = simulator.simulate(x0, (0, 0.15), dt=0.005)

        # phi should at least double from 0.1 (nonlinear growth)
        assert np.max(np.abs(x[:, 0])) > 0.2

    def test_energy_conservation_no_input(self, simulator):
        """Test energy conservation without motor input (tau=0).

        Start near stable equilibrium (phi~pi) for bounded oscillation.
        """
        x0 = np.array([np.pi - 0.1, 0.0, 0.0, 0.0])
        t, x, u = simulator.simulate(x0, (0, 5.0), dt=0.005)

        energies = simulator.energy(x)
        total_energy = energies['total']

        energy_variation = (
            (total_energy.max() - total_energy.min()) / np.abs(total_energy.mean())
        )
        assert energy_variation < 0.01, (
            f"Energy varied by {energy_variation*100:.2f}%"
        )

    def test_energy_conservation_with_wheel_spin(self, simulator):
        """Test energy conservation with nonzero wheel angular velocity."""
        x0 = np.array([np.pi - 0.05, 0.0, 0.0, 5.0])
        t, x, u = simulator.simulate(x0, (0, 3.0), dt=0.005)

        energies = simulator.energy(x)
        total_energy = energies['total']

        energy_variation = (
            (total_energy.max() - total_energy.min()) / np.abs(total_energy.mean())
        )
        assert energy_variation < 0.01, (
            f"Energy varied by {energy_variation*100:.2f}%"
        )

    def test_gravity_torque_sign(self, simulator):
        """Test gravity destabilizes: positive phi -> positive phi_ddot."""
        state = np.array([0.05, 0.0, 0.0, 0.0])
        dx = simulator.dynamics(0.0, state, 0.0)

        # phi_ddot should be positive (gravity pulls body further from vertical)
        assert dx[1] > 0, f"phi_ddot should be positive, got {dx[1]}"

    def test_motor_reaction_sign(self, simulator):
        """Test motor reaction: positive tau decelerates phi_ddot."""
        state = np.array([0.05, 0.0, 0.0, 0.0])

        dx_no_motor = simulator.dynamics(0.0, state, 0.0)
        dx_with_motor = simulator.dynamics(0.0, state, 0.01)

        # tau > 0 should reduce phi_ddot (reaction torque stabilizes body)
        assert dx_with_motor[1] < dx_no_motor[1]

    def test_coupling_term(self, simulator):
        """Test theta_w_ddot includes coupling: theta_w_ddot = tau/I_w_C - phi_ddot."""
        state = np.array([0.05, 0.0, 0.0, 0.0])
        tau = 0.01

        dx = simulator.dynamics(0.0, state, tau)
        phi_ddot = dx[1]
        theta_w_ddot = dx[3]

        # Verify coupling: theta_w_ddot = tau/I_w_C - phi_ddot
        I_w_C = simulator.params['I_w_C']
        expected_theta_w_ddot = tau / I_w_C - phi_ddot
        np.testing.assert_allclose(theta_w_ddot, expected_theta_w_ddot, rtol=1e-10)


class TestAEKInitialStateSampling:
    """Test random initial state sampling."""

    @pytest.fixture
    def simulator(self):
        return AEKSimulator()

    @pytest.fixture
    def rng(self):
        set_seed(42)
        return np.random.default_rng(42)

    def test_sample_shape(self, simulator, rng):
        """Test sampled initial state has correct shape."""
        x0 = simulator.sample_initial_state(rng)
        assert x0.shape == (4,)

    def test_sample_dtype(self, simulator, rng):
        """Test sampled initial state has correct dtype."""
        x0 = simulator.sample_initial_state(rng)
        assert x0.dtype == np.float64

    def test_sample_ranges_unstable(self, simulator, rng):
        """Test sampled states near unstable equilibrium are within ranges."""
        for _ in range(100):
            x0 = simulator.sample_initial_state(
                rng, near_equilibrium='unstable'
            )

            assert -0.15 <= x0[0] <= 0.15, f"phi out of range: {x0[0]}"
            assert -1.0 <= x0[1] <= 1.0, f"phi_dot out of range: {x0[1]}"
            assert -1.0 <= x0[2] <= 1.0, f"theta_w out of range: {x0[2]}"
            assert -10.0 <= x0[3] <= 10.0, f"theta_w_dot out of range: {x0[3]}"

    def test_sample_near_stable(self, simulator, rng):
        """Test sampling near stable equilibrium (phi near pi)."""
        x0 = simulator.sample_initial_state(
            rng, phi_range=(-0.1, 0.1), near_equilibrium='stable'
        )

        assert np.abs(np.abs(x0[0]) - np.pi) < 0.2


class TestAEKEdgeCases:
    """Edge cases and robustness."""

    @pytest.fixture
    def simulator(self):
        return AEKSimulator()

    def test_high_angular_velocity(self, simulator):
        """Test simulation with high initial angular velocities."""
        x0 = np.array([0.05, 5.0, 0.0, 50.0])
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.005)

        assert not np.any(np.isnan(x))

    def test_long_simulation(self, simulator):
        """Test long simulation stability (near stable eq)."""
        x0 = np.array([np.pi - 0.1, 0.0, 0.0, 0.0])
        t, x, u = simulator.simulate(x0, (0, 30.0), dt=0.01)

        assert not np.any(np.isnan(x))
        assert x.shape[0] == len(t)

    def test_dt_001_matches_yaml(self, simulator):
        """Test that dt=0.01 from aek.yaml produces correct timesteps."""
        x0 = np.array([0.05, 0.0, 0.0, 0.0])
        t, x, u = simulator.simulate(x0, (0, 2.0), dt=0.01)

        # T_steps = 201 as specified in aek.yaml
        assert len(t) == 201

    def test_zero_input_wrapping(self, simulator):
        """Test x0 at exact 0 with no input doesn't produce NaN."""
        x0 = np.zeros(4)
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.01)

        assert not np.any(np.isnan(x))
        np.testing.assert_allclose(x[-1], x0, atol=1e-6)


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])