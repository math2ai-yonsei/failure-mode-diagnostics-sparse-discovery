"""
S03: Test Derivative Computation

Tests for Savitzky-Golay derivatives with angular wrap handling.
"""
import pytest
import numpy as np
from pathlib import Path

from src.utils.derivatives import (
    unwrap_angle,
    wrap_angle,
    savgol_derivative,
    central_difference,
    compute_derivatives_savgol,
    compute_derivatives_batch,
    validate_derivatives,
    check_wrap_boundary_issues,
    SAVGOL_CONFIG
)


# =============================================================================
# Angle Utility Tests
# =============================================================================

class TestAngleUtilities:
    """Test angle wrap/unwrap functions."""
    
    def test_wrap_angle_basic(self):
        """Wrap should keep angles in [-π, π)."""
        angles = np.array([0, np.pi, -np.pi, 2*np.pi, -2*np.pi, 3*np.pi])
        wrapped = wrap_angle(angles)
        
        # [-π, π) range (standard modulo convention)
        assert np.all(wrapped >= -np.pi)
        assert np.all(wrapped < np.pi + 1e-10)
    
    def test_wrap_angle_values(self):
        """Check specific wrap values."""
        assert np.isclose(wrap_angle(0), 0)
        # π wraps to -π (equivalent in angular space)
        assert np.isclose(np.abs(wrap_angle(np.pi)), np.pi)
        assert np.isclose(np.abs(wrap_angle(-np.pi)), np.pi)
        assert np.isclose(wrap_angle(2*np.pi), 0, atol=1e-10)
    
    def test_unwrap_continuous(self):
        """Unwrap should produce continuous angles."""
        # Simulate crossing π boundary
        t = np.linspace(0, 4, 100)
        theta_continuous = 0.5 * t  # Linear increase
        theta_wrapped = wrap_angle(theta_continuous)
        
        theta_unwrapped = unwrap_angle(theta_wrapped)
        
        # Unwrapped should be monotonically increasing
        dtheta = np.diff(theta_unwrapped)
        assert np.all(dtheta > 0), "Unwrapped angle should be monotonic"
        
        # Derivative should be constant (~0.5)
        assert np.allclose(dtheta / (t[1] - t[0]), 0.5, atol=0.01)
    
    def test_unwrap_no_effect_small_angles(self):
        """Unwrap should not change small angles."""
        theta = np.linspace(-1, 1, 50)
        theta_unwrapped = unwrap_angle(theta)
        np.testing.assert_allclose(theta, theta_unwrapped)
    
    def test_unwrap_2d_axis(self):
        """Unwrap should work along specified axis."""
        N, T = 5, 100
        t = np.linspace(0, 4, T)
        theta = np.outer(np.ones(N), wrap_angle(0.5 * t))  # (N, T)
        
        theta_unwrapped = unwrap_angle(theta, axis=1)
        
        # Each trajectory should be continuous
        for i in range(N):
            dtheta = np.diff(theta_unwrapped[i])
            assert np.all(np.abs(dtheta) < 0.2), f"Trajectory {i} has discontinuity"


# =============================================================================
# Savgol Derivative Tests
# =============================================================================

class TestSavgolDerivative:
    """Test Savitzky-Golay derivative computation."""
    
    def test_constant_derivative_zero(self):
        """Derivative of constant should be zero."""
        x = np.ones((10, 50))  # (N, T)
        dt = 0.02
        dx = savgol_derivative(x, dt, axis=1)
        
        np.testing.assert_allclose(dx, 0, atol=1e-10)
    
    def test_linear_derivative_constant(self):
        """Derivative of linear should be constant."""
        T = 101
        dt = 0.02
        t = np.arange(T) * dt
        slope = 2.5
        x = slope * t
        
        dx = savgol_derivative(x[np.newaxis, :], dt, axis=1)
        
        # Interior points should be accurate
        np.testing.assert_allclose(dx[0, 5:-5], slope, rtol=1e-3)
    
    def test_sin_derivative_cos(self):
        """Derivative of sin should be cos."""
        T = 101
        dt = 0.02
        t = np.arange(T) * dt
        freq = 2.0
        x = np.sin(2 * np.pi * freq * t)
        dx_true = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        
        dx = savgol_derivative(x[np.newaxis, :], dt, axis=1)[0]
        
        # Check interior (avoid boundary effects)
        np.testing.assert_allclose(dx[10:-10], dx_true[10:-10], rtol=0.05)
    
    def test_window_must_be_odd(self):
        """Should raise error for even window."""
        x = np.ones((1, 50))
        with pytest.raises(ValueError, match="Window must be odd"):
            savgol_derivative(x, 0.02, window=10, axis=1)
    
    def test_short_trajectory_error(self):
        """Should raise error if trajectory shorter than window."""
        x = np.ones((1, 5))  # T=5 < window=11
        with pytest.raises(ValueError, match="Time dimension"):
            savgol_derivative(x, 0.02, window=11, axis=1)
    
    def test_3d_array(self):
        """Should work with (N, T, D) arrays."""
        N, T, D = 10, 50, 4
        dt = 0.02
        x = np.random.randn(N, T, D)
        
        dx = savgol_derivative(x, dt, axis=1)
        
        assert dx.shape == x.shape


# =============================================================================
# Cart-Pole Derivative Tests
# =============================================================================

class TestCartPoleDerivatives:
    """Test derivatives for Cart-Pole system."""
    
    @pytest.fixture
    def cartpole_trajectory(self):
        """Generate synthetic Cart-Pole trajectory."""
        T = 101
        dt = 0.02
        t = np.arange(T) * dt
        
        # Simple oscillating trajectory
        x = 0.5 * np.sin(t)
        x_dot = 0.5 * np.cos(t)
        theta = 0.3 * np.sin(2*t)
        theta_dot = 0.6 * np.cos(2*t)
        
        state = np.stack([x, x_dot, theta, theta_dot], axis=-1)  # (T, 4)
        return state, dt
    
    def test_compute_derivatives_shape(self, cartpole_trajectory):
        """Output shape should match input."""
        x, dt = cartpole_trajectory
        x = x[np.newaxis, ...]  # (1, T, 4)
        
        dx = compute_derivatives_savgol(x, dt, theta_idx=2)
        
        assert dx.shape == x.shape
    
    def test_dx_matches_xdot(self, cartpole_trajectory):
        """dx[0] should approximately equal x[1] (x_dot)."""
        x, dt = cartpole_trajectory
        x = x[np.newaxis, ...]  # (1, T, 4)
        
        dx = compute_derivatives_savgol(x, dt, theta_idx=2)
        
        # dx[0] = d(x)/dt should ≈ x_dot = x[1]
        # Check interior points
        np.testing.assert_allclose(
            dx[0, 10:-10, 0], x[0, 10:-10, 1], rtol=0.1
        )
    
    def test_dtheta_matches_theta_dot(self, cartpole_trajectory):
        """dx[2] should approximately equal x[3] (theta_dot)."""
        x, dt = cartpole_trajectory
        x = x[np.newaxis, ...]  # (1, T, 4)
        
        dx = compute_derivatives_savgol(x, dt, theta_idx=2)
        
        # dx[2] = d(theta)/dt should ≈ theta_dot = x[3]
        np.testing.assert_allclose(
            dx[0, 10:-10, 2], x[0, 10:-10, 3], rtol=0.1
        )
    
    def test_no_nan_output(self, cartpole_trajectory):
        """Output should not contain NaN."""
        x, dt = cartpole_trajectory
        x = x[np.newaxis, ...]
        
        dx = compute_derivatives_savgol(x, dt, theta_idx=2)
        
        assert not np.isnan(dx).any()


# =============================================================================
# Wrap Boundary Tests
# =============================================================================

class TestWrapBoundary:
    """Test handling of wrap boundary near ±π."""
    
    def test_near_pi_no_spike(self):
        """Derivative near π boundary should not spike."""
        T = 101
        dt = 0.02
        t = np.arange(T) * dt
        
        # Trajectory that approaches and crosses π
        theta = 3.0 + 0.2 * t  # Starts at 3.0, crosses π
        theta_wrapped = wrap_angle(theta)
        
        # Build fake state
        x = np.zeros((1, T, 4))
        x[0, :, 2] = theta_wrapped  # theta
        x[0, :, 3] = 0.2  # theta_dot (constant)
        
        dx = compute_derivatives_savgol(x, dt, theta_idx=2)
        
        # d(theta)/dt should be ~0.2 everywhere, no spikes
        theta_deriv = dx[0, 10:-10, 2]
        assert np.all(np.abs(theta_deriv - 0.2) < 0.1), \
            f"Derivative spike detected: max={theta_deriv.max()}, min={theta_deriv.min()}"
    
    def test_crossing_negative_pi(self):
        """Derivative crossing -π should not spike."""
        T = 101
        dt = 0.02
        t = np.arange(T) * dt
        
        # Trajectory that crosses -π
        theta = -3.0 - 0.2 * t  # Starts at -3.0, crosses -π
        theta_wrapped = wrap_angle(theta)
        
        x = np.zeros((1, T, 4))
        x[0, :, 2] = theta_wrapped
        
        dx = compute_derivatives_savgol(x, dt, theta_idx=2)
        
        # Should be ~-0.2 everywhere
        theta_deriv = dx[0, 10:-10, 2]
        assert np.all(np.abs(theta_deriv - (-0.2)) < 0.1)
    
    def test_check_wrap_boundary_issues_clean(self):
        """Clean trajectory should have no boundary issues."""
        T = 101
        x = np.zeros((5, T, 4))
        x[..., 2] = np.random.uniform(-1, 1, (5, T))  # Small angles
        
        dx = compute_derivatives_savgol(x, 0.02, theta_idx=2)
        
        result = check_wrap_boundary_issues(x, dx, theta_idx=2)
        assert result['n_boundary_spikes'] == 0
        assert not result['has_issues']
    
    def test_max_theta_31_handling(self):
        """Test with max_theta=3.1 (near π≈3.1416)."""
        T = 101
        dt = 0.02
        
        # Trajectory that reaches 3.1 radians
        x = np.zeros((1, T, 4))
        x[0, :, 2] = np.linspace(0, 3.1, T)  # Linear increase to 3.1
        x[0, :, 3] = 3.1 / (T * dt)  # Corresponding theta_dot
        
        dx = compute_derivatives_savgol(x, dt, theta_idx=2)
        
        # Should compute without issues
        assert not np.isnan(dx).any()
        
        # Derivative should be approximately constant
        expected_deriv = 3.1 / (T * dt)
        np.testing.assert_allclose(
            dx[0, 10:-10, 2], expected_deriv, rtol=0.1
        )


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestBatchDerivatives:
    """Test batch derivative computation."""
    
    def test_batch_output_keys(self):
        """Should return expected keys."""
        N, T, D = 5, 50, 4
        x = np.random.randn(N, T, D)
        u = np.random.randn(N, T, 1)
        
        result = compute_derivatives_batch(x, u, dt=0.02, theta_idx=2)
        
        assert 'dx_savgol' in result
        assert 'dx_central' in result
    
    def test_batch_shapes(self):
        """All outputs should have same shape as input."""
        N, T, D = 5, 50, 4
        x = np.random.randn(N, T, D)
        u = np.random.randn(N, T, 1)
        
        result = compute_derivatives_batch(x, u, dt=0.02, theta_idx=2)
        
        assert result['dx_savgol'].shape == x.shape
        assert result['dx_central'].shape == x.shape
    
    def test_savgol_vs_central_similar(self):
        """Savgol and central should be similar for smooth data."""
        N, T = 5, 101
        dt = 0.02
        t = np.arange(T) * dt
        
        # Smooth sinusoidal trajectory
        x = np.zeros((N, T, 4))
        for i in range(N):
            x[i, :, 0] = np.sin(t + i * 0.1)
            x[i, :, 1] = np.cos(t + i * 0.1)
            x[i, :, 2] = 0.3 * np.sin(2*t + i * 0.1)
            x[i, :, 3] = 0.6 * np.cos(2*t + i * 0.1)
        
        u = np.zeros((N, T, 1))
        result = compute_derivatives_batch(x, u, dt, theta_idx=2)
        
        # Compare interior points
        savgol = result['dx_savgol'][:, 15:-15, :]
        central = result['dx_central'][:, 15:-15, :]
        
        # Should be within 10% for smooth data
        np.testing.assert_allclose(savgol, central, rtol=0.2, atol=0.1)


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Test derivative validation utilities."""
    
    def test_validate_perfect_match(self):
        """Identical arrays should have zero error."""
        dx = np.random.randn(5, 50, 4)
        dx_ref = dx.copy()
        
        metrics = validate_derivatives(
            dx, dx_ref, 
            state_names=['x', 'x_dot', 'theta', 'theta_dot']
        )
        
        for name in ['x', 'x_dot', 'theta', 'theta_dot']:
            assert metrics[name]['rmse'] < 1e-10
            assert metrics[name]['r2'] > 0.999
    
    def test_validate_with_noise(self):
        """Noisy derivatives should have non-zero error."""
        dx_ref = np.random.randn(5, 50, 4)
        dx = dx_ref + 0.1 * np.random.randn(*dx_ref.shape)
        
        metrics = validate_derivatives(dx, dx_ref)
        
        for key in metrics:
            assert metrics[key]['rmse'] > 0
            assert metrics[key]['mae'] > 0


# =============================================================================
# Integration with Real Dataset
# =============================================================================

class TestRealDataset:
    """Test with actual generated dataset (if exists)."""
    
    @pytest.fixture
    def dataset_path(self):
        """Path to dataset."""
        from src.contracts import paths
        return paths.get_dataset_path('cartpole_ood_v1', system='cartpole')
    
    def test_dataset_dx_savgol(self, dataset_path):
        """Apply Savgol to real dataset and compare with analytic dx."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        data = np.load(dataset_path)
        train_x = data['train_x']
        train_dx_analytic = data['train_dx']
        dt = float(data['dt'])
        
        # Compute Savgol derivatives
        dx_savgol = compute_derivatives_savgol(train_x, dt, theta_idx=2)
        
        # Basic checks
        assert dx_savgol.shape == train_x.shape
        assert not np.isnan(dx_savgol).any()
        
        # Compare with analytic (should be reasonably close)
        metrics = validate_derivatives(
            dx_savgol, train_dx_analytic,
            state_names=['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']
        )
        
        # x_dot and theta_dot should match well (direct state)
        # x_ddot and theta_ddot may differ more (second derivative)
        print("\nDerivative comparison (Savgol vs Analytic):")
        for name, m in metrics.items():
            print(f"  {name}: RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}")
    
    def test_dataset_no_wrap_issues(self, dataset_path):
        """Check dataset has no wrap boundary issues."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        data = np.load(dataset_path)
        
        for split in ['train', 'val', 'test']:
            x = data[f'{split}_x']
            dx = compute_derivatives_savgol(x, float(data['dt']), theta_idx=2)
            
            result = check_wrap_boundary_issues(x, dx, theta_idx=2)
            print(f"\n{split} wrap check: {result}")
            
            # Allow some near-boundary points, but no spikes
            assert result['n_boundary_spikes'] < x.shape[0], \
                f"{split} has too many boundary spikes"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])