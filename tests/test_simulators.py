"""
Tests for Cart-Pole simulator.

Run with: python -m pytest tests/test_simulators.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.simulators import CartPoleSimulator, BaseSimulator
from src.utils.seed_utils import set_seed


class TestCartPoleSimulator:
    """Test suite for CartPoleSimulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create default simulator."""
        return CartPoleSimulator()
       
    @pytest.fixture
    def rng(self):
        """Create seeded random generator."""
        set_seed(42)  # 전역 시드 설정
        return np.random.default_rng(42)  # Generator 반환
    
    # =========================================================================
    # Basic Property Tests
    # =========================================================================
    
    def test_state_dim(self, simulator):
        """Test state dimension is 4."""
        assert simulator.state_dim == 4
    
    def test_input_dim(self, simulator):
        """Test input dimension is 1."""
        assert simulator.input_dim == 1
    
    def test_is_base_simulator(self, simulator):
        """Test inheritance from BaseSimulator."""
        assert isinstance(simulator, BaseSimulator)
    
    def test_default_params(self, simulator):
        """Test default parameters are set."""
        assert simulator.params['m_cart'] == 1.0
        assert simulator.params['m_pole'] == 0.1
        assert simulator.params['L'] == 0.5
        assert simulator.params['g'] == 9.81
    
    def test_custom_params(self):
        """Test custom parameters."""
        params = {'m_cart': 2.0, 'm_pole': 0.2, 'L': 1.0, 'g': 10.0}
        sim = CartPoleSimulator(params)
        assert sim.params['m_cart'] == 2.0
        assert sim.params['m_pole'] == 0.2
    
    def test_invalid_params_negative_mass(self):
        """Test that negative mass raises error."""
        with pytest.raises(ValueError):
            CartPoleSimulator({'m_cart': -1.0})
    
    def test_invalid_params_zero_length(self):
        """Test that zero length raises error."""
        with pytest.raises(ValueError):
            CartPoleSimulator({'L': 0.0})
    
    # =========================================================================
    # Simulation Tests
    # =========================================================================
    
    def test_simulate_output_shape(self, simulator):
        """Test simulation output shapes."""
        x0 = np.array([0.0, 0.0, 0.1, 0.0])
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.02)
        
        assert t.shape == (51,)  # 0 to 1.0 with dt=0.02
        assert x.shape == (51, 4)
        assert u.shape == (51, 1)
    
    def test_simulate_output_dtype(self, simulator):
        """Test output dtypes are float64."""
        x0 = np.array([0.0, 0.0, 0.1, 0.0])
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.02)
        
        assert t.dtype == np.float64
        assert x.dtype == np.float64
        assert u.dtype == np.float64
    
    def test_simulate_no_nan(self, simulator, rng):
        """Test simulation produces no NaN values."""
        for _ in range(10):
            x0 = simulator.sample_initial_state(rng)
            t, x, u = simulator.simulate(x0, (0, 2.0), dt=0.02)
            
            assert not np.any(np.isnan(x)), "Trajectory contains NaN"
            assert not np.any(np.isnan(u)), "Control input contains NaN"
    
    def test_simulate_initial_condition(self, simulator):
        """Test that trajectory starts at initial condition."""
        x0 = np.array([0.5, -0.1, 0.2, 0.3])
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.02)
        
        np.testing.assert_allclose(x[0], x0, atol=1e-10)
    
    def test_simulate_with_controller(self, simulator):
        """Test simulation with control input."""
        x0 = np.array([0.0, 0.0, 0.1, 0.0])
        
        def controller(t, x):
            return 1.0  # Constant force
        
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.02, controller=controller)
        
        # Check control input is applied
        assert np.all(u[:, 0] != 0.0)
    
    def test_simulate_batch(self, simulator, rng):
        """Test batch simulation."""
        n_traj = 5
        x0_batch = np.array([simulator.sample_initial_state(rng) for _ in range(n_traj)])
        
        t, x_batch, u_batch = simulator.simulate_batch(x0_batch, (0, 1.0), dt=0.02)
        
        assert x_batch.shape == (5, 51, 4)
        assert u_batch.shape == (5, 51, 1)
    
    # =========================================================================
    # Theta Wrapping Tests
    # =========================================================================
    
    def test_wrap_angle_in_range(self, simulator):
        """Test angles in range stay unchanged."""
        angles = np.array([0.0, 0.5, -0.5, np.pi - 0.1, -np.pi + 0.1])
        wrapped = simulator.wrap_angle(angles)
        np.testing.assert_allclose(wrapped, angles, atol=1e-10)
    
    def test_wrap_angle_positive_overflow(self, simulator):
        """Test wrapping of large positive angles."""
        theta = 2 * np.pi + 0.5
        wrapped = simulator.wrap_angle(theta)
        np.testing.assert_allclose(wrapped, 0.5, atol=1e-10)
    
    def test_wrap_angle_negative_overflow(self, simulator):
        """Test wrapping of large negative angles."""
        theta = -2 * np.pi - 0.5
        wrapped = simulator.wrap_angle(theta)
        np.testing.assert_allclose(wrapped, -0.5, atol=1e-10)
    
    def test_wrap_angle_pi(self, simulator):
        """Test that π maps to π (not -π)."""
        wrapped = simulator.wrap_angle(np.pi)
        assert wrapped == np.pi
    
    def test_wrap_angle_minus_pi(self, simulator):
        """Test that -π maps to π."""
        wrapped = simulator.wrap_angle(-np.pi)
        assert wrapped == np.pi
    
    def test_wrap_state_trajectory(self, simulator):
        """Test theta wrapping in trajectory."""
        # Start with theta that will cross 2π
        x0 = np.array([0.0, 0.0, 3.0, 2.0])  # High angular velocity
        t, x, u = simulator.simulate(x0, (0, 5.0), dt=0.02)
        
        # All theta values should be in (-π, π]
        theta = x[:, 2]
        assert np.all(theta > -np.pi), f"theta below -π: {theta.min()}"
        assert np.all(theta <= np.pi), f"theta above π: {theta.max()}"
    
    # =========================================================================
    # Physics Tests
    # =========================================================================
    
    def test_equilibrium_stable(self, simulator):
        """Test stable equilibrium (pole down)."""
        x0 = simulator.get_stable_equilibrium()
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.02)
        
        # Should stay at equilibrium
        np.testing.assert_allclose(x[-1], x0, atol=1e-6)
    
    def test_equilibrium_unstable(self, simulator):
        """Test unstable equilibrium (pole up)."""
        x0 = simulator.get_unstable_equilibrium()
        t, x, u = simulator.simulate(x0, (0, 1.0), dt=0.02)
        
        # Should stay at equilibrium (no perturbation)
        np.testing.assert_allclose(x[-1], x0, atol=1e-6)
    
    def test_equilibrium_unstable_perturbed(self, simulator):
        """Test unstable equilibrium with perturbation falls."""
        x0 = np.array([0.0, 0.0, 0.01, 0.0])  # Small angle from vertical
        t, x, u = simulator.simulate(x0, (0, 2.0), dt=0.02)
        
        # Pole should fall (theta increases)
        assert np.abs(x[-1, 2]) > 0.1
    
    def test_energy_conservation_no_input_no_damping(self, simulator):
        """Test energy conservation without input or damping."""
        x0 = np.array([0.0, 0.0, 0.5, 0.0])  # Start tilted
        t, x, u = simulator.simulate(x0, (0, 5.0), dt=0.01)
        
        # Compute energy at each timestep
        energies = simulator.energy(x)
        total_energy = energies['total']
        
        # Energy should be conserved (within numerical tolerance)
        energy_variation = (total_energy.max() - total_energy.min()) / total_energy.mean()
        assert energy_variation < 0.01, f"Energy varied by {energy_variation*100:.2f}%"
    
    def test_energy_with_damping(self):
        """Test energy decreases with damping."""
        sim = CartPoleSimulator({'b_cart': 0.1, 'b_pole': 0.05})
        x0 = np.array([0.0, 0.5, 0.5, 0.5])  # Some initial velocity
        t, x, u = sim.simulate(x0, (0, 5.0), dt=0.02)
        
        energies = sim.energy(x)
        
        # Energy should decrease
        assert energies['total'][-1] < energies['total'][0]
    
    # =========================================================================
    # Initial State Sampling Tests
    # =========================================================================
    
    def test_sample_initial_state_shape(self, simulator, rng):
        """Test sampled initial state has correct shape."""
        x0 = simulator.sample_initial_state(rng)
        assert x0.shape == (4,)
    
    def test_sample_initial_state_dtype(self, simulator, rng):
        """Test sampled initial state has correct dtype."""
        x0 = simulator.sample_initial_state(rng)
        assert x0.dtype == np.float64
    
    def test_sample_initial_state_ranges(self, simulator, rng):
        """Test sampled states are within specified ranges."""
        for _ in range(100):
            x0 = simulator.sample_initial_state(
                rng,
                x_range=(-1.0, 1.0),
                x_dot_range=(-0.5, 0.5),
                theta_range=(-0.3, 0.3),
                theta_dot_range=(-0.5, 0.5),
                near_equilibrium='unstable'
            )
            
            assert -1.0 <= x0[0] <= 1.0, f"x out of range: {x0[0]}"
            assert -0.5 <= x0[1] <= 0.5, f"x_dot out of range: {x0[1]}"
            assert -0.3 <= x0[2] <= 0.3, f"theta out of range: {x0[2]}"
            assert -0.5 <= x0[3] <= 0.5, f"theta_dot out of range: {x0[3]}"
    
    def test_sample_initial_state_near_stable(self, simulator, rng):
        """Test sampling near stable equilibrium."""
        x0 = simulator.sample_initial_state(
            rng,
            theta_range=(-0.1, 0.1),
            near_equilibrium='stable'
        )
        
        # Theta should be near π
        assert np.abs(np.abs(x0[2]) - np.pi) < 0.2
    
    # =========================================================================
    # Edge Case Tests
    # =========================================================================
    
    def test_large_initial_velocity(self, simulator):
        """Test simulation with large initial velocities."""
        x0 = np.array([0.0, 5.0, 0.1, 10.0])
        t, x, u = simulator.simulate(x0, (0, 2.0), dt=0.01)
        
        assert not np.any(np.isnan(x))
    
    def test_long_simulation(self, simulator):
        """Test long simulation stability."""
        x0 = np.array([0.0, 0.0, 0.3, 0.0])
        t, x, u = simulator.simulate(x0, (0, 30.0), dt=0.02)
        
        assert not np.any(np.isnan(x))
        assert x.shape[0] == len(t)
    
    def test_different_params_give_different_trajectories(self):
        """Test that different parameters produce different trajectories."""
        sim1 = CartPoleSimulator({'m_cart': 1.0})
        sim2 = CartPoleSimulator({'m_cart': 2.0})
        
        x0 = np.array([0.0, 0.0, 0.3, 0.0])
        
        t1, x1, u1 = sim1.simulate(x0, (0, 1.0), dt=0.02)
        t2, x2, u2 = sim2.simulate(x0, (0, 1.0), dt=0.02)
        
        # Trajectories should differ
        assert not np.allclose(x1, x2)


class TestBaseSimulatorWrapAngle:
    """Test angle wrapping utility."""
    
    def test_wrap_angle_array(self):
        """Test wrapping array of angles."""
        angles = np.array([0, np.pi, -np.pi, 2*np.pi, -2*np.pi, 3*np.pi])
        wrapped = BaseSimulator.wrap_angle(angles)
        
        expected = np.array([0, np.pi, np.pi, 0, 0, np.pi])
        np.testing.assert_allclose(wrapped, expected, atol=1e-10)
    
    def test_wrap_angle_scalar(self):
        """Test wrapping scalar angle."""
        assert np.isclose(BaseSimulator.wrap_angle(0.0), 0.0)
        assert np.isclose(BaseSimulator.wrap_angle(np.pi), np.pi)
        assert np.isclose(BaseSimulator.wrap_angle(2*np.pi), 0.0, atol=1e-10)


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])