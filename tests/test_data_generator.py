"""
S02: Test Data Generator

Tests for CartPoleDataGenerator and dataset generation.
"""
import pytest
import numpy as np
import tempfile
import yaml
from pathlib import Path

from src.data.data_generator import (
    CartPoleDataGenerator,
    generate_dataset,
    make_zero_controller,
    make_random_smooth_controller,
    make_sinusoidal_controller
)
from src.contracts import paths
from src.contracts.schema_dataset_lite import validate_dataset_lite


# =============================================================================
# Controller Tests
# =============================================================================

class TestControllers:
    """Test controller factory functions."""
    
    def test_zero_controller(self):
        """Zero controller should always return 0."""
        ctrl = make_zero_controller()
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        assert ctrl(0.0, state) == 0.0
        assert ctrl(1.0, state) == 0.0
        assert ctrl(10.0, state) == 0.0
    
    def test_random_smooth_controller_shape(self):
        """Random smooth controller should return scalar."""
        ctrl = make_random_smooth_controller(
            amplitude=5.0,
            frequency=2.0,
            duration=2.0,
            dt=0.02,
            seed=42
        )
        state = np.array([0.0, 0.0, 0.0, 0.0])
        
        u = ctrl(0.0, state)
        assert isinstance(u, float)
    
    def test_random_smooth_controller_bounded(self):
        """Random smooth controller should be bounded by amplitude."""
        amplitude = 5.0
        ctrl = make_random_smooth_controller(
            amplitude=amplitude,
            frequency=2.0,
            duration=2.0,
            dt=0.02,
            seed=42
        )
        state = np.zeros(4)
        
        # Test at multiple times
        for t in np.linspace(0, 2.0, 100):
            u = ctrl(t, state)
            assert abs(u) <= amplitude + 1e-6
    
    def test_random_smooth_controller_reproducible(self):
        """Same seed should give same controller."""
        ctrl1 = make_random_smooth_controller(
            amplitude=5.0, frequency=2.0, duration=2.0, dt=0.02, seed=42
        )
        ctrl2 = make_random_smooth_controller(
            amplitude=5.0, frequency=2.0, duration=2.0, dt=0.02, seed=42
        )
        
        state = np.zeros(4)
        for t in np.linspace(0, 2.0, 50):
            assert np.isclose(ctrl1(t, state), ctrl2(t, state), rtol=1e-10)
    
    def test_sinusoidal_controller(self):
        """Sinusoidal controller should follow sine function."""
        amplitude = 3.0
        frequency = 1.0
        ctrl = make_sinusoidal_controller(amplitude, frequency, phase=0.0)
        
        state = np.zeros(4)
        
        # At t=0, sin(0) = 0
        assert abs(ctrl(0.0, state)) < 1e-10
        
        # At t=0.25/freq, sin(π/2) = 1
        assert abs(ctrl(0.25, state) - amplitude) < 1e-6


# =============================================================================
# Data Generator Tests
# =============================================================================

class TestCartPoleDataGenerator:
    """Test CartPoleDataGenerator class."""
    
    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config file for testing."""
        config = {
            'physics': {
                'm_cart': 1.0,
                'm_pole': 0.1,
                'L': 0.5,
                'g': 9.81,
                'b_cart': 0.1,
                'b_pole': 0.01
            },
            'simulation': {
                'dt': 0.02,
                'duration': 1.0,
                'T': 51
            },
            'initial_state': {
                'x': {'min': -0.5, 'max': 0.5},
                'x_dot': {'min': -1.0, 'max': 1.0},
                'theta': {'min': -0.3, 'max': 0.3},
                'theta_dot': {'min': -1.0, 'max': 1.0}
            },
            'controller': {
                'type': 'random_smooth',
                'random_smooth': {
                    'amplitude': 3.0,
                    'frequency': 2.0,
                    'seed': None
                }
            },
            'conditions': {
                'train': {
                    'm_cart': [1.0, 1.2],
                    'm_pole': [0.08, 0.10]
                },
                'val': {
                    'm_cart': [1.0, 1.2],
                    'm_pole': [0.08, 0.10]
                },
                'test': {
                    'm_cart': [1.6, 1.8],
                    'm_pole': [0.14, 0.16]
                }
            },
            'data_generation': {
                'n_traj_per_condition': {
                    'train': 2,
                    'val': 1,
                    'test': 2
                },
                'n_trajectories': {
                    'train': 5,
                    'val': 3,
                    'test': 5
                },
                'quality_filters': {
                    'max_x': 10.0,
                    'max_theta': 2.5,
                    'max_velocity': 30.0,
                    'max_attempts': 10
                },
                'master_seed': 42
            },
            'state_definition': [
                {'index': 0, 'name': 'x', 'unit': 'm'},
                {'index': 1, 'name': 'x_dot', 'unit': 'm/s'},
                {'index': 2, 'name': 'theta', 'unit': 'rad'},
                {'index': 3, 'name': 'theta_dot', 'unit': 'rad/s'}
            ],
            'param_definition': [
                {'index': 0, 'name': 'm_cart', 'unit': 'kg'},
                {'index': 1, 'name': 'm_pole', 'unit': 'kg'}
            ]
        }
        
        config_path = tmp_path / 'test_cartpole.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def test_generator_init(self, temp_config):
        """Generator should initialize from config."""
        gen = CartPoleDataGenerator(temp_config)
        
        assert gen.physics['m_cart'] == 1.0
        assert gen.sim_config['dt'] == 0.02
        assert gen.quality_filters['max_x'] == 10.0
    
    def test_generate_single_trajectory(self, temp_config):
        """Should generate valid single trajectory."""
        gen = CartPoleDataGenerator(temp_config)
        
        result = gen._generate_single_trajectory(
            m_cart=1.0,
            m_pole=0.1,
            seed=42
        )
        
        assert result is not None
        assert result['x'].shape == (51, 4)
        assert result['u'].shape == (51, 1)
        assert result['dx'].shape == (51, 4)
        assert result['params'].shape == (2,)
        assert not np.isnan(result['x']).any()
        assert not np.isnan(result['dx']).any()
    
    def test_generate_split(self, temp_config):
        """Should generate complete split."""
        gen = CartPoleDataGenerator(temp_config)
        
        x, u, dx, params, cond_id = gen.generate_split(
            split_name='test',
            conditions={'m_cart': [1.0], 'm_pole': [0.1]},
            n_traj_per_condition=3,
            base_seed=42
        )
        
        assert x.shape[0] == 3  # 1 condition × 3 trajectories
        assert x.shape[1] == 51  # T
        assert x.shape[2] == 4   # state_dim
        assert u.shape == (3, 51, 1)
        assert dx.shape == x.shape
        assert params.shape == (3, 2)
        assert cond_id.shape == (3,)
    
    def test_dx_matches_dynamics(self, temp_config):
        """Analytic dx should match dynamics computation."""
        gen = CartPoleDataGenerator(temp_config)
        
        result = gen._generate_single_trajectory(
            m_cart=1.0,
            m_pole=0.1,
            seed=42
        )
        
        x = result['x']
        u = result['u']
        dx = result['dx']
        
        # Basic check: dx[0] = x_dot, dx[2] = theta_dot
        np.testing.assert_allclose(dx[:, 0], x[:, 1], rtol=1e-6)
        np.testing.assert_allclose(dx[:, 2], x[:, 3], rtol=1e-6)
        
        # Full check: verify all 4 components against dynamics() at sample points
        # Use same physics params as generator config
        from src.simulators import CartPoleSimulator
        sim_params = {
            'm_cart': 1.0,
            'm_pole': 0.1,
            'L': gen.physics['L'],
            'g': gen.physics['g'],
            'b_cart': gen.physics['b_cart'],
            'b_pole': gen.physics['b_pole']
        }
        sim = CartPoleSimulator(params=sim_params)
        
        # Check at 5 random time indices
        test_indices = [0, 10, 25, 40, 50]
        for idx in test_indices:
            if idx < len(x):
                state = x[idx]
                input_force = u[idx, 0]
                expected_dx = sim.dynamics(0.0, state, input_force)
                np.testing.assert_allclose(
                    dx[idx], expected_dx, rtol=1e-6,
                    err_msg=f"dx mismatch at index {idx}"
                )
    
    def test_condition_separated_params(self, temp_config):
        """Train and test should have different parameter ranges."""
        gen = CartPoleDataGenerator(temp_config)
        result = gen.generate_dataset('test_v1')
        
        train_params = result['data']['train_params']
        test_params = result['data']['test_params']
        
        # Train: m_cart in [1.0, 1.2], m_pole in [0.08, 0.10]
        assert train_params[:, 0].min() >= 1.0 - 0.01
        assert train_params[:, 0].max() <= 1.2 + 0.01
        
        # Test: m_cart in [1.6, 1.8], m_pole in [0.14, 0.16]
        assert test_params[:, 0].min() >= 1.6 - 0.01
        assert test_params[:, 0].max() <= 1.8 + 0.01


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestSchemaCompliance:
    """Test that generated dataset passes schema validation."""
    
    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create minimal config for schema testing."""
        config = {
            'physics': {
                'm_cart': 1.0,
                'm_pole': 0.1,
                'L': 0.5,
                'g': 9.81,
                'b_cart': 0.1,
                'b_pole': 0.01
            },
            'simulation': {
                'dt': 0.02,
                'duration': 1.0,
                'T': 51
            },
            'initial_state': {
                'x': {'min': -0.3, 'max': 0.3},
                'x_dot': {'min': -0.5, 'max': 0.5},
                'theta': {'min': -0.2, 'max': 0.2},
                'theta_dot': {'min': -0.5, 'max': 0.5}
            },
            'controller': {
                'type': 'zero'
            },
            'conditions': {
                'train': {'m_cart': [1.0], 'm_pole': [0.1]},
                'val': {'m_cart': [1.0], 'm_pole': [0.1]},
                'test': {'m_cart': [1.5], 'm_pole': [0.15]}
            },
            'data_generation': {
                'n_traj_per_condition': {'train': 3, 'val': 2, 'test': 3},
                'n_trajectories': {'train': 3, 'val': 2, 'test': 3},
                'quality_filters': {
                    'max_x': 10.0,
                    'max_theta': 2.5,
                    'max_velocity': 30.0,
                    'max_attempts': 10
                },
                'master_seed': 42
            },
            'state_definition': [
                {'index': 0, 'name': 'x', 'unit': 'm'},
                {'index': 1, 'name': 'x_dot', 'unit': 'm/s'},
                {'index': 2, 'name': 'theta', 'unit': 'rad'},
                {'index': 3, 'name': 'theta_dot', 'unit': 'rad/s'}
            ],
            'param_definition': [
                {'index': 0, 'name': 'm_cart', 'unit': 'kg'},
                {'index': 1, 'name': 'm_pole', 'unit': 'kg'}
            ]
        }
        
        config_path = tmp_path / 'schema_test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path, tmp_path
    
    def test_schema_validation_passes(self, temp_config):
        """Generated dataset should pass schema_dataset_lite validation."""
        config_path, tmp_path = temp_config
        
        gen = CartPoleDataGenerator(config_path)
        result = gen.generate_dataset('schema_test')
        
        # Save to temp file
        npz_path = tmp_path / 'test_dataset.npz'
        np.savez_compressed(npz_path, **result['data'])
        
        # Should not raise
        validate_dataset_lite(npz_path)
    
    def test_all_required_keys_present(self, temp_config):
        """Dataset should contain all required keys."""
        config_path, _ = temp_config
        
        gen = CartPoleDataGenerator(config_path)
        result = gen.generate_dataset('key_test')
        
        required_keys = [
            'train_x', 'val_x', 'test_x',
            'train_u', 'val_u', 'test_u',
            'train_dx', 'val_dx', 'test_dx',
            'train_params', 'val_params', 'test_params',
            'train_cond_id', 'val_cond_id', 'test_cond_id',
            't', 'dt'
        ]
        
        for key in required_keys:
            assert key in result['data'], f"Missing key: {key}"
    
    def test_shapes_consistent(self, temp_config):
        """Array shapes should be internally consistent."""
        config_path, _ = temp_config
        
        gen = CartPoleDataGenerator(config_path)
        result = gen.generate_dataset('shape_test')
        data = result['data']
        
        T = len(data['t'])
        
        for split in ['train', 'val', 'test']:
            x = data[f'{split}_x']
            u = data[f'{split}_u']
            dx = data[f'{split}_dx']
            params = data[f'{split}_params']
            cond_id = data[f'{split}_cond_id']
            
            N = x.shape[0]
            
            assert x.shape == (N, T, 4), f"{split}_x shape mismatch"
            assert u.shape == (N, T, 1), f"{split}_u shape mismatch"
            assert dx.shape == (N, T, 4), f"{split}_dx shape mismatch"
            assert params.shape[0] == N, f"{split}_params N mismatch"
            assert cond_id.shape == (N,), f"{split}_cond_id shape mismatch"
    
    def test_no_nan_values(self, temp_config):
        """Dataset should not contain NaN values."""
        config_path, _ = temp_config
        
        gen = CartPoleDataGenerator(config_path)
        result = gen.generate_dataset('nan_test')
        data = result['data']
        
        for key in ['train_x', 'val_x', 'test_x',
                    'train_u', 'val_u', 'test_u',
                    'train_dx', 'val_dx', 'test_dx']:
            assert not np.isnan(data[key]).any(), f"NaN in {key}"


# =============================================================================
# Metadata Tests
# =============================================================================

class TestMetadata:
    """Test metadata generation."""
    
    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create minimal config for metadata testing."""
        config = {
            'physics': {
                'm_cart': 1.0, 'm_pole': 0.1, 'L': 0.5,
                'g': 9.81, 'b_cart': 0.1, 'b_pole': 0.01
            },
            'simulation': {'dt': 0.02, 'duration': 1.0, 'T': 51},
            'initial_state': {
                'x': {'min': -0.3, 'max': 0.3},
                'x_dot': {'min': -0.5, 'max': 0.5},
                'theta': {'min': -0.2, 'max': 0.2},
                'theta_dot': {'min': -0.5, 'max': 0.5}
            },
            'controller': {'type': 'zero'},
            'conditions': {
                'train': {'m_cart': [1.0], 'm_pole': [0.1]},
                'val': {'m_cart': [1.0], 'm_pole': [0.1]},
                'test': {'m_cart': [1.5], 'm_pole': [0.15]}
            },
            'data_generation': {
                'n_traj_per_condition': {'train': 2, 'val': 1, 'test': 2},
                'n_trajectories': {'train': 2, 'val': 1, 'test': 2},
                'quality_filters': {
                    'max_x': 10.0, 'max_theta': 3.0, 
                    'max_velocity': 30.0, 'max_attempts': 10
                },
                'master_seed': 42
            },
            'state_definition': [
                {'index': 0, 'name': 'x', 'unit': 'm'},
                {'index': 1, 'name': 'x_dot', 'unit': 'm/s'},
                {'index': 2, 'name': 'theta', 'unit': 'rad'},
                {'index': 3, 'name': 'theta_dot', 'unit': 'rad/s'}
            ],
            'param_definition': [
                {'index': 0, 'name': 'm_cart', 'unit': 'kg'},
                {'index': 1, 'name': 'm_pole', 'unit': 'kg'}
            ]
        }
        
        config_path = tmp_path / 'meta_test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def test_metadata_contains_statistics(self, temp_config):
        """Metadata should contain generation statistics."""
        gen = CartPoleDataGenerator(temp_config)
        result = gen.generate_dataset('meta_test')
        
        meta = result['meta']
        
        assert 'statistics' in meta
        assert 'n_train' in meta['statistics']
        assert 'n_val' in meta['statistics']
        assert 'n_test' in meta['statistics']
        assert 'acceptance_rate' in meta['statistics']
    
    def test_metadata_contains_conditions(self, temp_config):
        """Metadata should contain condition definitions."""
        gen = CartPoleDataGenerator(temp_config)
        result = gen.generate_dataset('cond_test')
        
        meta = result['meta']
        
        assert 'conditions' in meta
        assert 'train' in meta['conditions']
        assert 'test' in meta['conditions']


# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_generation_with_real_config(self, tmp_path):
        """Test full generation using project config (if exists)."""
        config_path = paths.ROOT / 'configs' / 'systems' / 'cartpole.yaml'
        
        if not config_path.exists():
            pytest.skip("Project config not found")
        
        gen = CartPoleDataGenerator(config_path)
        result = gen.generate_dataset('integration_test')
        
        # Save
        npz_path = tmp_path / 'integration_test.npz'
        np.savez_compressed(npz_path, **result['data'])
        
        # Validate
        validate_dataset_lite(npz_path)
        
        # Check sizes
        data = result['data']
        assert data['train_x'].shape[0] > 0
        assert data['val_x'].shape[0] > 0
        assert data['test_x'].shape[0] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])