"""
S04-B: Test SINDy Library

Tests for SINDy feature library with focus on:
1. Feature matrix shape and values
2. Wrap-safe θ handling (no θ polynomials by default)
3. Configuration validation
4. Input validation (fail-fast for S05 pipeline)
5. Integration with real dataset
"""
import pytest
import numpy as np
from pathlib import Path

from src.sindy.library import (
    SINDyLibrary,
    LIBRARY_CONFIGS,
    STATE_INDICES,
    STATE_NAMES,
    build_library_matrix,
    get_derivative_key,
    get_library_manifest,
)


# =============================================================================
# Basic Library Tests
# =============================================================================

class TestSINDyLibraryBasic:
    """Test basic library functionality."""
    
    def test_create_default_library(self):
        """Should create library with default config."""
        lib = SINDyLibrary()
        
        assert lib.config_name == 'gate0_min'
        assert lib.n_features > 0
        assert len(lib.feature_names) == lib.n_features
    
    def test_create_with_config(self):
        """Should create library with specified config."""
        for config_name in LIBRARY_CONFIGS.keys():
            lib = SINDyLibrary(config=config_name)
            assert lib.config_name == config_name
    
    def test_invalid_config_raises_error(self):
        """Should raise error for unknown config."""
        with pytest.raises(ValueError, match="Unknown config"):
            SINDyLibrary(config='nonexistent')
    
    def test_feature_names_unique(self):
        """Feature names should be unique."""
        lib = SINDyLibrary()
        assert len(lib.feature_names) == len(set(lib.feature_names))
    
    def test_repr(self):
        """Should have informative repr."""
        lib = SINDyLibrary()
        repr_str = repr(lib)
        
        assert 'gate0_min' in repr_str
        assert 'n_features' in repr_str


class TestFeatureMatrix:
    """Test feature matrix computation."""
    
    @pytest.fixture
    def sample_data_2d(self):
        """Generate 2D sample data (N, D)."""
        np.random.seed(42)
        N = 100
        x = np.random.randn(N, 4)
        u = np.random.randn(N, 1)
        return x, u
    
    @pytest.fixture
    def sample_data_3d(self):
        """Generate 3D sample data (N, T, D)."""
        np.random.seed(42)
        N, T = 10, 50
        x = np.random.randn(N, T, 4)
        u = np.random.randn(N, T, 1)
        return x, u
    
    def test_output_shape_2d(self, sample_data_2d):
        """Output shape should be (N, n_features) for 2D input."""
        x, u = sample_data_2d
        lib = SINDyLibrary()
        
        Theta = lib.fit_transform(x, u)
        
        assert Theta.shape == (x.shape[0], lib.n_features)
    
    def test_output_shape_3d(self, sample_data_3d):
        """Output shape should be (N*T, n_features) for 3D input."""
        x, u = sample_data_3d
        N, T, D = x.shape
        lib = SINDyLibrary()
        
        Theta = lib.fit_transform(x, u)
        
        assert Theta.shape == (N * T, lib.n_features)
    
    def test_output_dtype(self, sample_data_2d):
        """Output should be float64."""
        x, u = sample_data_2d
        lib = SINDyLibrary()
        
        Theta = lib.fit_transform(x, u)
        
        assert Theta.dtype == np.float64
    
    def test_no_nan(self, sample_data_3d):
        """Output should not contain NaN."""
        x, u = sample_data_3d
        lib = SINDyLibrary()
        
        Theta = lib.fit_transform(x, u)
        
        assert not np.isnan(Theta).any()
    
    def test_constant_feature(self, sample_data_2d):
        """First feature should be constant 1."""
        x, u = sample_data_2d
        lib = SINDyLibrary()
        
        Theta = lib.fit_transform(x, u)
        
        # First feature should be '1'
        assert lib.feature_names[0] == '1'
        np.testing.assert_allclose(Theta[:, 0], 1.0)


class TestTrigFeatures:
    """Test trigonometric feature handling."""
    
    def test_sin_cos_present(self):
        """sin(θ) and cos(θ) should be in features."""
        lib = SINDyLibrary(config='gate0_min')
        
        assert 'sin(theta)' in lib.feature_names
        assert 'cos(theta)' in lib.feature_names
    
    def test_sin_cos_values(self):
        """sin(θ) and cos(θ) should be computed correctly."""
        # Known theta values
        theta_vals = np.array([0, np.pi/2, np.pi, -np.pi/2])
        N = len(theta_vals)
        
        x = np.zeros((N, 4))
        x[:, 2] = theta_vals  # theta column
        u = np.zeros((N, 1))
        
        lib = SINDyLibrary()
        Theta = lib.fit_transform(x, u)
        
        sin_idx = lib.feature_names.index('sin(theta)')
        cos_idx = lib.feature_names.index('cos(theta)')
        
        np.testing.assert_allclose(Theta[:, sin_idx], np.sin(theta_vals), atol=1e-10)
        np.testing.assert_allclose(Theta[:, cos_idx], np.cos(theta_vals), atol=1e-10)
    
    def test_no_theta_polynomial_by_default(self):
        """θ polynomial should NOT be in default features."""
        lib = SINDyLibrary(config='gate0_min')
        
        assert 'theta' not in lib.feature_names
        assert 'theta^2' not in lib.feature_names
    
    def test_theta_polynomial_with_custom_config(self):
        """θ polynomial should be present with custom config."""
        lib = SINDyLibrary(
            config='gate0_min',
            custom_config={'theta_poly': True}
        )
        
        assert 'theta' in lib.feature_names
        assert 'theta^2' in lib.feature_names


class TestInputFeatures:
    """Test input (u) feature handling."""
    
    def test_u_present(self):
        """u should be in features."""
        lib = SINDyLibrary(config='gate0_min')
        
        assert 'u' in lib.feature_names
    
    def test_u_trig_coupling(self):
        """u*sin(θ) and u*cos(θ) should be in features."""
        lib = SINDyLibrary(config='gate0_min')
        
        assert 'u*sin(theta)' in lib.feature_names
        assert 'u*cos(theta)' in lib.feature_names
    
    def test_u_values(self):
        """u feature should match input values."""
        N = 50
        x = np.zeros((N, 4))
        u = np.random.randn(N, 1)
        
        lib = SINDyLibrary()
        Theta = lib.fit_transform(x, u)
        
        u_idx = lib.feature_names.index('u')
        np.testing.assert_allclose(Theta[:, u_idx], u[:, 0])


class TestQuadraticFeatures:
    """Test quadratic feature handling."""
    
    def test_quadratic_present(self):
        """Quadratic terms should be in features."""
        lib = SINDyLibrary(config='gate0_min')
        
        assert 'x^2' in lib.feature_names
        assert 'x*x_dot' in lib.feature_names
        assert 'x_dot^2' in lib.feature_names
        assert 'theta_dot^2' in lib.feature_names
    
    def test_quadratic_values(self):
        """Quadratic features should be computed correctly."""
        N = 50
        x = np.random.randn(N, 4)
        u = np.zeros((N, 1))
        
        lib = SINDyLibrary()
        Theta = lib.fit_transform(x, u)
        
        x2_idx = lib.feature_names.index('x^2')
        np.testing.assert_allclose(Theta[:, x2_idx], x[:, 0]**2)
        
        xx_dot_idx = lib.feature_names.index('x*x_dot')
        np.testing.assert_allclose(Theta[:, xx_dot_idx], x[:, 0] * x[:, 1])


# =============================================================================
# Wrap Safety Tests
# =============================================================================

class TestWrapSafety:
    """Test that library is safe for θ wrap boundary."""
    
    def test_near_pi_boundary(self):
        """Features should be continuous near ±π."""
        # Create theta values near ±π
        theta_vals = np.linspace(-np.pi + 0.1, np.pi - 0.1, 100)
        N = len(theta_vals)
        
        x = np.zeros((N, 4))
        x[:, 2] = theta_vals
        u = np.zeros((N, 1))
        
        lib = SINDyLibrary()
        Theta = lib.fit_transform(x, u)
        
        # sin and cos should be smooth (no jumps)
        sin_idx = lib.feature_names.index('sin(theta)')
        cos_idx = lib.feature_names.index('cos(theta)')
        
        sin_diff = np.abs(np.diff(Theta[:, sin_idx]))
        cos_diff = np.abs(np.diff(Theta[:, cos_idx]))
        
        # Max jump should be small (smooth transition)
        assert sin_diff.max() < 0.1, "sin(θ) has discontinuity"
        assert cos_diff.max() < 0.1, "cos(θ) has discontinuity"
    
    def test_crossing_pi(self):
        """Features should handle θ crossing π smoothly."""
        # Simulate crossing π
        theta_vals = np.array([3.0, 3.1, -3.1, -3.0])  # Wrapped around π
        N = len(theta_vals)
        
        x = np.zeros((N, 4))
        x[:, 2] = theta_vals
        u = np.zeros((N, 1))
        
        lib = SINDyLibrary()
        Theta = lib.fit_transform(x, u)
        
        # sin and cos should be smooth even across wrap
        assert not np.isnan(Theta).any()


# =============================================================================
# Input Validation Tests (fail-fast for S05 pipeline)
# =============================================================================

class TestInputValidation:
    """Test fit_transform input validation (fail-fast for S05 pipeline)."""
    
    def test_invalid_x_ndim_1d(self):
        """Should raise ValueError for 1D x."""
        x = np.random.randn(100)  # 1D
        u = np.random.randn(100, 1)
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="2D.*3D"):
            lib.fit_transform(x, u)
    
    def test_invalid_x_ndim_4d(self):
        """Should raise ValueError for 4D x."""
        x = np.random.randn(10, 5, 4, 2)  # 4D
        u = np.random.randn(10, 5, 1, 2)
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="2D.*3D"):
            lib.fit_transform(x, u)
    
    def test_ndim_mismatch(self):
        """Should raise ValueError when x and u have different ndim."""
        x = np.random.randn(10, 50, 4)  # 3D
        u = np.random.randn(500, 1)      # 2D (flattened by mistake)
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="ndim.*must match"):
            lib.fit_transform(x, u)
    
    def test_invalid_state_dim(self):
        """Should raise ValueError when x last dim != 4."""
        x = np.random.randn(100, 3)  # Wrong: 3 instead of 4
        u = np.random.randn(100, 1)
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="last dimension must be 4"):
            lib.fit_transform(x, u)
    
    def test_invalid_input_dim(self):
        """Should raise ValueError when u last dim != 1."""
        x = np.random.randn(100, 4)
        u = np.random.randn(100, 2)  # Wrong: 2 instead of 1
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="last dimension must be 1"):
            lib.fit_transform(x, u)
    
    def test_sample_count_mismatch_2d(self):
        """Should raise ValueError when N differs in 2D."""
        x = np.random.randn(100, 4)
        u = np.random.randn(50, 1)  # Wrong: 50 instead of 100
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="samples"):
            lib.fit_transform(x, u)
    
    def test_shape_mismatch_3d(self):
        """Should raise ValueError when (N, T) differs in 3D."""
        x = np.random.randn(10, 50, 4)
        u = np.random.randn(10, 40, 1)  # Wrong: T=40 instead of 50
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="must match.*dimensions"):
            lib.fit_transform(x, u)
    
    def test_nan_in_x(self):
        """Should raise ValueError when x contains NaN."""
        x = np.random.randn(100, 4)
        x[50, 2] = np.nan  # Inject NaN
        u = np.random.randn(100, 1)
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="NaN"):
            lib.fit_transform(x, u)
    
    def test_inf_in_x(self):
        """Should raise ValueError when x contains Inf."""
        x = np.random.randn(100, 4)
        x[25, 0] = np.inf  # Inject Inf
        u = np.random.randn(100, 1)
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="Inf"):
            lib.fit_transform(x, u)
    
    def test_nan_in_u(self):
        """Should raise ValueError when u contains NaN."""
        x = np.random.randn(100, 4)
        u = np.random.randn(100, 1)
        u[75, 0] = np.nan  # Inject NaN
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="NaN"):
            lib.fit_transform(x, u)
            
    def test_inf_in_u(self):
        """Should raise ValueError when u contains Inf."""
        x = np.random.randn(100, 4)
        u = np.random.randn(100, 1)
        u[30, 0] = np.inf  # Inject Inf
        
        lib = SINDyLibrary()
        with pytest.raises(ValueError, match="Inf"):
            lib.fit_transform(x, u)
    
    def test_valid_input_passes(self):
        """Valid input should pass without error."""
        x = np.random.randn(100, 4)
        u = np.random.randn(100, 1)
        
        lib = SINDyLibrary()
        Theta = lib.fit_transform(x, u)  # Should not raise
        
        assert Theta.shape == (100, lib.n_features)
    
    def test_valid_3d_input_passes(self):
        """Valid 3D input should pass without error."""
        x = np.random.randn(10, 50, 4)
        u = np.random.randn(10, 50, 1)
        
        lib = SINDyLibrary()
        Theta = lib.fit_transform(x, u)  # Should not raise
        
        assert Theta.shape == (500, lib.n_features)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_build_library_matrix(self):
        """build_library_matrix should return Theta and names."""
        x = np.random.randn(50, 4)
        u = np.random.randn(50, 1)
        
        Theta, names = build_library_matrix(x, u)
        
        assert Theta.shape[0] == 50
        assert len(names) == Theta.shape[1]
    
    def test_get_derivative_key_standardized(self):
        """standardized → derivative_dx_savgol."""
        key = get_derivative_key('standardized')
        assert key == 'derivative_dx_savgol'
    
    def test_get_derivative_key_oracle(self):
        """author_recommended → derivative_dx."""
        key = get_derivative_key('author_recommended')
        assert key == 'derivative_dx'
    
    def test_get_derivative_key_invalid(self):
        """Unknown track should raise error."""
        with pytest.raises(ValueError, match="Unknown track"):
            get_derivative_key('invalid_track')
    
    def test_get_library_manifest(self):
        """Manifest should contain required fields."""
        lib = SINDyLibrary()
        manifest = get_library_manifest(lib, 'standardized')
        
        required_keys = [
            'library_id', 'n_features', 'feature_names',
            'poly_degree', 'use_trig', 'include_u', 'theta_poly',
            'track', 'derivative_key'
        ]
        
        for key in required_keys:
            assert key in manifest, f"Missing key: {key}"
        
        assert manifest['derivative_key'] == 'derivative_dx_savgol'


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfigurations:
    """Test library configurations."""
    
    def test_all_configs_valid(self):
        """All configs in LIBRARY_CONFIGS should work."""
        x = np.random.randn(50, 4)
        u = np.random.randn(50, 1)
        
        for config_name in LIBRARY_CONFIGS.keys():
            lib = SINDyLibrary(config=config_name)
            Theta = lib.fit_transform(x, u)
            
            assert not np.isnan(Theta).any(), f"NaN in config: {config_name}"
    
    def test_config_descriptions(self):
        """All configs should have descriptions."""
        for name, cfg in LIBRARY_CONFIGS.items():
            assert 'description' in cfg, f"Missing description: {name}"
    
    def test_gate0_min_features(self):
        """gate0_min should have specific features."""
        lib = SINDyLibrary(config='gate0_min')
        
        # Must have
        assert '1' in lib.feature_names
        assert 'x' in lib.feature_names
        assert 'x_dot' in lib.feature_names
        assert 'theta_dot' in lib.feature_names
        assert 'sin(theta)' in lib.feature_names
        assert 'cos(theta)' in lib.feature_names
        assert 'u' in lib.feature_names
        
        # Must NOT have (wrap safety)
        assert 'theta' not in lib.feature_names


# =============================================================================
# Integration with Real Dataset
# =============================================================================

class TestRealDataset:
    """Test with actual generated dataset."""
    
    @pytest.fixture
    def dataset_path(self):
        """Get path to dataset."""
        from src.contracts import paths
        return paths.get_dataset_path('cartpole_ood_v1')
    
    def test_with_real_data(self, dataset_path):
        """Should work with real dataset."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        data = np.load(dataset_path)
        train_x = data['train_x']
        train_u = data['train_u']
        
        lib = SINDyLibrary(config='gate0_min')
        Theta = lib.fit_transform(train_x, train_u)
        
        N, T, D = train_x.shape
        assert Theta.shape == (N * T, lib.n_features)
        assert not np.isnan(Theta).any()
    
    def test_feature_statistics(self, dataset_path):
        """Feature statistics should be reasonable."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        data = np.load(dataset_path)
        train_x = data['train_x']
        train_u = data['train_u']
        
        lib = SINDyLibrary()
        Theta = lib.fit_transform(train_x, train_u)
        
        # Check feature statistics
        means = Theta.mean(axis=0)
        stds = Theta.std(axis=0)
        
        # Constant feature should be exactly 1
        assert np.isclose(means[0], 1.0)
        assert np.isclose(stds[0], 0.0)
        
        # sin/cos should be in [-1, 1]
        sin_idx = lib.feature_names.index('sin(theta)')
        cos_idx = lib.feature_names.index('cos(theta)')
        
        assert Theta[:, sin_idx].min() >= -1.0
        assert Theta[:, sin_idx].max() <= 1.0
        assert Theta[:, cos_idx].min() >= -1.0
        assert Theta[:, cos_idx].max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])