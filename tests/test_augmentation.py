"""
Gate2: Augmentation Module Tests

Tests for physics-consistent data augmentation.

Test Categories:
    1. Configuration tests
    2. Reproducibility tests (seed-based)
    3. dx-x consistency tests (re-simulation)
    4. Shape/dtype validation
    5. Quality filter tests
    6. Train subset idx consistency (Gate1 parity)
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_trajectories():
    """Create sample trajectory data for testing."""
    np.random.seed(42)
    
    n_traj = 5
    T = 50
    state_dim = 4
    input_dim = 1
    n_params = 4
    
    # Reasonable Cart-Pole-like data
    x = np.random.randn(n_traj, T, state_dim) * 0.5
    x[:, :, 2] = np.clip(x[:, :, 2], -0.5, 0.5)  # theta near upright
    
    u = np.random.randn(n_traj, T, input_dim) * 2.0
    
    # Standard Cart-Pole parameters
    params = np.array([
        [1.0, 0.1, 0.5, 9.81],  # m_cart, m_pole, L, g
        [1.2, 0.1, 0.5, 9.81],
        [1.0, 0.12, 0.5, 9.81],
        [1.1, 0.11, 0.5, 9.81],
        [1.0, 0.1, 0.5, 9.81],
    ])
    
    return {
        'x': x,
        'u': u,
        'params': params,
        'n_traj': n_traj,
        'T': T,
        'dt': 0.02,
    }


@pytest.fixture
def physics_augmentor_config():
    """Create default PhysicsAugmentorConfig."""
    from src.augmentation.physics_augmentor import PhysicsAugmentorConfig
    
    return PhysicsAugmentorConfig(
        aug_ratio=1.0,
        seed=42,
        dt=0.02,
        T=50,
        jitter_mode='both',
    )


# =============================================================================
# BaseAugmentor Tests
# =============================================================================

class TestBaseAugmentor:
    """Tests for BaseAugmentor and related utilities."""
    
    def test_augmentation_result_validation(self):
        """Test AugmentationResult validates shapes."""
        from src.augmentation.base import AugmentationResult
        
        n_aug = 3
        T = 50
        state_dim = 4
        input_dim = 1
        n_params = 4
        
        result = AugmentationResult(
            x=np.zeros((n_aug, T, state_dim)),
            u=np.zeros((n_aug, T, input_dim)),
            dx=np.zeros((n_aug, T, state_dim)),
            params=np.zeros((n_aug, n_params)),
            n_original=5,
            n_augmented=n_aug,
            aug_method='test',
            aug_config={},
            source_idx=np.zeros(n_aug, dtype=np.int64),
            aug_type=['test'] * n_aug,
        )
        
        assert result.n_augmented == n_aug
        assert result.x.shape == (n_aug, T, state_dim)
    
    def test_augmentation_result_shape_mismatch(self):
        """Test AugmentationResult raises on shape mismatch."""
        from src.augmentation.base import AugmentationResult
        
        with pytest.raises(AssertionError):
            AugmentationResult(
                x=np.zeros((3, 50, 4)),  # n_aug=3
                u=np.zeros((3, 50, 1)),
                dx=np.zeros((3, 50, 4)),
                params=np.zeros((3, 4)),
                n_original=5,
                n_augmented=5,  # Mismatch!
                aug_method='test',
                aug_config={},
                source_idx=np.zeros(3, dtype=np.int64),
                aug_type=['test'] * 3,
            )
    
    def test_get_train_subset_idx_reproducibility(self):
        """Test train subset idx is reproducible with same seed."""
        from src.augmentation.base import get_train_subset_idx
        
        idx1 = get_train_subset_idx(n_total=50, n_train=10, seed=42)
        idx2 = get_train_subset_idx(n_total=50, n_train=10, seed=42)
        
        assert np.array_equal(idx1, idx2)
    
    def test_get_train_subset_idx_different_seeds(self):
        """Test train subset idx differs with different seeds."""
        from src.augmentation.base import get_train_subset_idx
        
        idx1 = get_train_subset_idx(n_total=50, n_train=10, seed=42)
        idx2 = get_train_subset_idx(n_total=50, n_train=10, seed=43)
        
        assert not np.array_equal(idx1, idx2)
    
    def test_get_train_subset_idx_sorted(self):
        """Test train subset idx is sorted."""
        from src.augmentation.base import get_train_subset_idx
        
        idx = get_train_subset_idx(n_total=100, n_train=20, seed=42)
        
        assert np.all(idx[:-1] <= idx[1:])  # Sorted
    
    def test_get_train_subset_idx_n_train_exceeds(self):
        """Test train subset idx when n_train >= n_total."""
        from src.augmentation.base import get_train_subset_idx
        
        idx = get_train_subset_idx(n_total=10, n_train=20, seed=42)
        
        assert len(idx) == 10
        assert np.array_equal(idx, np.arange(10))


# =============================================================================
# PhysicsAugmentor Config Tests
# =============================================================================

class TestPhysicsAugmentorConfig:
    """Tests for PhysicsAugmentorConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.augmentation.physics_augmentor import PhysicsAugmentorConfig
        
        config = PhysicsAugmentorConfig()
        
        assert config.method == 'physics_resim'
        assert config.ic_jitter_enabled is True
        assert config.param_jitter_enabled is True
        assert config.jitter_mode == 'both'
    
    def test_config_to_dict(self):
        """Test config serialization."""
        from src.augmentation.physics_augmentor import PhysicsAugmentorConfig
        
        config = PhysicsAugmentorConfig(aug_ratio=2.0, seed=123)
        d = config.to_dict()
        
        assert d['aug_ratio'] == 2.0
        assert d['seed'] == 123
        assert 'ic_jitter_std' in d
        assert 'param_jitter_rel_std' in d


# =============================================================================
# PhysicsAugmentor Core Tests
# =============================================================================

class TestPhysicsAugmentor:
    """Tests for PhysicsAugmentor."""
    
    def test_init_creates_simulator(self, physics_augmentor_config):
        """Test augmentor creates simulator if not provided."""
        from src.augmentation.physics_augmentor import PhysicsAugmentor
        
        augmentor = PhysicsAugmentor(physics_augmentor_config)
        
        assert augmentor.simulator is not None
    
    def test_augment_output_shapes(self, physics_augmentor_config, sample_trajectories):
        """Test augment output shapes are correct."""
        from src.augmentation.physics_augmentor import PhysicsAugmentor
        
        augmentor = PhysicsAugmentor(physics_augmentor_config)
        
        result = augmentor.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
        )
        
        n_aug = int(sample_trajectories['n_traj'] * physics_augmentor_config.aug_ratio)
        T = sample_trajectories['T']
        
        assert result.x.shape == (n_aug, T, 4)
        assert result.u.shape == (n_aug, T, 1)
        assert result.dx.shape == (n_aug, T, 4)
        assert result.params.shape == (n_aug, 4)
    
    def test_augment_reproducibility(self, sample_trajectories):
        """Test augmentation is reproducible with same seed."""
        from src.augmentation.physics_augmentor import (
            PhysicsAugmentor, PhysicsAugmentorConfig
        )
        
        config1 = PhysicsAugmentorConfig(seed=42, T=sample_trajectories['T'])
        config2 = PhysicsAugmentorConfig(seed=42, T=sample_trajectories['T'])
        
        aug1 = PhysicsAugmentor(config1)
        aug2 = PhysicsAugmentor(config2)
        
        result1 = aug1.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
            n_aug=2,
        )
        
        result2 = aug2.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
            n_aug=2,
        )
        
        # Source indices should match
        assert np.array_equal(result1.source_idx, result2.source_idx)
        # Aug types should match
        assert result1.aug_type == result2.aug_type
    
    def test_augment_different_seeds(self, sample_trajectories):
        """Test different seeds produce different results."""
        from src.augmentation.physics_augmentor import (
            PhysicsAugmentor, PhysicsAugmentorConfig
        )
        
        config1 = PhysicsAugmentorConfig(seed=42, T=sample_trajectories['T'])
        config2 = PhysicsAugmentorConfig(seed=43, T=sample_trajectories['T'])
        
        aug1 = PhysicsAugmentor(config1)
        aug2 = PhysicsAugmentor(config2)
        
        result1 = aug1.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
            n_aug=3,
        )
        
        result2 = aug2.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
            n_aug=3,
        )
        
        # Results should differ
        assert not np.allclose(result1.x, result2.x)
    
    def test_augment_no_nan(self, physics_augmentor_config, sample_trajectories):
        """Test augmented data contains no NaN."""
        from src.augmentation.physics_augmentor import PhysicsAugmentor
        
        augmentor = PhysicsAugmentor(physics_augmentor_config)
        
        result = augmentor.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
        )
        
        assert np.isfinite(result.x).all()
        assert np.isfinite(result.u).all()
        assert np.isfinite(result.dx).all()
        assert np.isfinite(result.params).all()
    
    def test_augment_theta_wrapped(self, physics_augmentor_config, sample_trajectories):
        """Test augmented theta is in (-π, π]."""
        from src.augmentation.physics_augmentor import PhysicsAugmentor
        
        augmentor = PhysicsAugmentor(physics_augmentor_config)
        
        result = augmentor.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
        )
        
        theta = result.x[:, :, 2]
        assert (theta > -np.pi).all()
        assert (theta <= np.pi).all()


# =============================================================================
# dx-x Consistency Tests (Critical)
# =============================================================================

class TestDxXConsistency:
    """Tests for dx-x physical consistency."""
    
    def test_dx_matches_dynamics(self, physics_augmentor_config, sample_trajectories):
        """Test dx is consistent with simulator dynamics."""
        from src.augmentation.physics_augmentor import PhysicsAugmentor
        from src.simulators import CartPoleSimulator
        
        augmentor = PhysicsAugmentor(physics_augmentor_config)
        
        result = augmentor.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
            n_aug=2,
        )
        
        # For each augmented trajectory, verify dx matches dynamics
        for i in range(result.n_augmented):
            x_traj = result.x[i]
            u_traj = result.u[i]
            dx_traj = result.dx[i]
            params = result.params[i]
            
            # Create simulator with these parameters
            sim = CartPoleSimulator(params={
                'm_cart': params[0],
                'm_pole': params[1],
                'L': params[2],
                'g': params[3],
            })
            
            # Verify dx at a few time points
            for t_idx in [0, 10, 25]:
                if t_idx >= len(x_traj):
                    continue
                    
                dx_computed = sim.dynamics(0, x_traj[t_idx], u_traj[t_idx, 0])
                
                # Should match within numerical tolerance
                assert np.allclose(dx_traj[t_idx], dx_computed, rtol=1e-5, atol=1e-8), \
                    f"dx mismatch at traj {i}, t={t_idx}"


# =============================================================================
# Jitter Mode Tests
# =============================================================================

class TestJitterModes:
    """Tests for different jitter modes."""
    
    def test_ic_only_mode(self, sample_trajectories):
        """Test IC-only jitter mode."""
        from src.augmentation.physics_augmentor import (
            PhysicsAugmentor, PhysicsAugmentorConfig
        )
        
        config = PhysicsAugmentorConfig(
            seed=42,
            T=sample_trajectories['T'],
            jitter_mode='ic_only',
        )
        
        augmentor = PhysicsAugmentor(config)
        result = augmentor.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
            n_aug=3,
        )
        
        # All should be IC jitter
        for atype in result.aug_type:
            assert atype in ['ic_jitter', 'original_fallback']
    
    def test_param_only_mode(self, sample_trajectories):
        """Test param-only jitter mode."""
        from src.augmentation.physics_augmentor import (
            PhysicsAugmentor, PhysicsAugmentorConfig
        )
        
        config = PhysicsAugmentorConfig(
            seed=42,
            T=sample_trajectories['T'],
            jitter_mode='param_only',
        )
        
        augmentor = PhysicsAugmentor(config)
        result = augmentor.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
            n_aug=3,
        )
        
        # All should be param jitter
        for atype in result.aug_type:
            assert atype in ['param_jitter', 'original_fallback']
    
    def test_both_mode(self, sample_trajectories):
        """Test combined IC + param jitter mode."""
        from src.augmentation.physics_augmentor import (
            PhysicsAugmentor, PhysicsAugmentorConfig
        )
        
        config = PhysicsAugmentorConfig(
            seed=42,
            T=sample_trajectories['T'],
            jitter_mode='both',
        )
        
        augmentor = PhysicsAugmentor(config)
        result = augmentor.augment(
            sample_trajectories['x'],
            sample_trajectories['u'],
            sample_trajectories['params'],
            n_aug=3,
        )
        
        # All should be combined jitter
        for atype in result.aug_type:
            assert atype in ['ic_param_jitter', 'original_fallback']


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunction:
    """Tests for create_physics_augmentor factory."""
    
    def test_create_default(self):
        """Test factory with default args."""
        from src.augmentation.physics_augmentor import create_physics_augmentor
        
        augmentor = create_physics_augmentor()
        
        assert augmentor is not None
        assert augmentor.config.aug_ratio == 1.0
        assert augmentor.config.seed == 42
    
    def test_create_custom(self):
        """Test factory with custom args."""
        from src.augmentation.physics_augmentor import create_physics_augmentor
        
        augmentor = create_physics_augmentor(
            aug_ratio=2.0,
            seed=123,
            jitter_mode='ic_only',
        )
        
        assert augmentor.config.aug_ratio == 2.0
        assert augmentor.config.seed == 123
        assert augmentor.config.jitter_mode == 'ic_only'
    
    def test_create_scaled_jitter(self):
        """Test factory with scaled jitter."""
        from src.augmentation.physics_augmentor import create_physics_augmentor
        
        augmentor = create_physics_augmentor(
            ic_std_scale=2.0,
            param_rel_std_scale=0.5,
        )
        
        # IC std should be doubled
        assert augmentor.config.ic_jitter_std['x'] == 0.1  # 0.05 * 2
        # Param std should be halved
        assert augmentor.config.param_jitter_rel_std['m_cart'] == 0.025  # 0.05 * 0.5


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for input validation."""
    
    def test_invalid_x_dim(self, physics_augmentor_config, sample_trajectories):
        """Test error on wrong x dimensions."""
        from src.augmentation.physics_augmentor import PhysicsAugmentor
        
        augmentor = PhysicsAugmentor(physics_augmentor_config)
        
        x_2d = sample_trajectories['x'][0]  # (T, 4) instead of (n, T, 4)
        
        with pytest.raises(ValueError, match="must be 3D"):
            augmentor.augment(
                x_2d,
                sample_trajectories['u'],
                sample_trajectories['params'],
            )
    
    def test_invalid_u_dim(self, physics_augmentor_config, sample_trajectories):
        """Test error on wrong u dimensions."""
        from src.augmentation.physics_augmentor import PhysicsAugmentor
        
        augmentor = PhysicsAugmentor(physics_augmentor_config)
        
        u_2d = sample_trajectories['u'][0]  # (T, 1) instead of (n, T, 1)
        
        with pytest.raises(ValueError, match="must be 3D"):
            augmentor.augment(
                sample_trajectories['x'],
                u_2d,
                sample_trajectories['params'],
            )
    
    def test_invalid_params_dim(self, physics_augmentor_config, sample_trajectories):
        """Test error on wrong params dimensions."""
        from src.augmentation.physics_augmentor import PhysicsAugmentor
        
        augmentor = PhysicsAugmentor(physics_augmentor_config)
        
        params_1d = sample_trajectories['params'][0]  # (4,) instead of (n, 4)
        
        with pytest.raises(ValueError, match="must be 2D"):
            augmentor.augment(
                sample_trajectories['x'],
                sample_trajectories['u'],
                params_1d,
            )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])