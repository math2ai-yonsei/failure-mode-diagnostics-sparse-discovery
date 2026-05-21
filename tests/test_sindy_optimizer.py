"""
S05: Test STLSQ Optimizer

Tests for:
1. ColumnScaler - scale-only normalization, constant column preservation
2. STLSQOptimizer - sparsity, convergence, input validation
3. Coefficient inverse transform (unscaling)
4. CSV I/O round-trip
5. Integration with real dataset
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.sindy.optimizer import (
    ColumnScaler,
    STLSQOptimizer,
    STLSQ_CONFIGS,
    TARGET_NAMES,
    save_coefficients_csv,
    load_coefficients_csv,
    get_optimizer_manifest,
)


# =============================================================================
# ColumnScaler Tests
# =============================================================================

class TestColumnScalerBasic:
    """Test basic ColumnScaler functionality."""
    
    def test_fit_transform(self):
        """Should fit and transform correctly."""
        Theta = np.random.randn(100, 5)
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        assert scaler._is_fitted
        assert Theta_scaled.shape == Theta.shape
    
    def test_scale_only_no_mean(self):
        """Should use scale-only (no mean subtraction)."""
        np.random.seed(42)
        Theta = np.random.randn(100, 5) * 3 + 10  # mean≠0, std≈3
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        # Scale-only: Θ/scale, NOT (Θ-mean)/scale
        # So scaled mean should be original_mean / scale ≈ 10/3 ≈ 3.3
        # NOT ≈ 0 (which would indicate z-score)
        scaled_means = Theta_scaled.mean(axis=0)
        assert np.all(np.abs(scaled_means) > 1), "Mean should NOT be ~0 (scale-only)"
    
    def test_transform_not_fitted(self):
        """Should raise error if transform called before fit."""
        scaler = ColumnScaler()
        
        with pytest.raises(ValueError, match="not fitted"):
            scaler.transform(np.random.randn(10, 5))


class TestColumnScalerConstantColumn:
    """Test constant column handling (CRITICAL for SINDy)."""
    
    def test_constant_column_preserved(self):
        """Constant column '1' should remain 1 after transform."""
        Theta = np.random.randn(100, 5)
        Theta[:, 0] = 1.0  # Constant column
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        # Critical: constant column must stay 1
        np.testing.assert_allclose(
            Theta_scaled[:, 0], 1.0,
            err_msg="Constant column '1' was NOT preserved!"
        )
    
    def test_constant_mask_detected(self):
        """Should detect constant columns via mask."""
        Theta = np.random.randn(100, 5)
        Theta[:, 0] = 1.0  # Constant
        Theta[:, 2] = 5.0  # Another constant
        
        scaler = ColumnScaler()
        scaler.fit(Theta)
        
        assert scaler.constant_mask_[0] == True
        assert scaler.constant_mask_[2] == True
        assert scaler.constant_mask_[1] == False
    
    def test_constant_scale_is_one(self):
        """Constant columns should have scale=1.0."""
        Theta = np.random.randn(100, 5)
        Theta[:, 0] = 1.0
        
        scaler = ColumnScaler()
        scaler.fit(Theta)
        
        assert scaler.scale_[0] == 1.0


class TestColumnScalerInverse:
    """Test inverse transform."""
    
    def test_inverse_transform(self):
        """inverse_transform should recover original."""
        np.random.seed(42)
        Theta = np.random.randn(100, 5) * 2 + 3
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        Theta_recovered = scaler.inverse_transform(Theta_scaled)
        
        np.testing.assert_allclose(Theta, Theta_recovered, rtol=1e-10)
    
    def test_inverse_with_constant(self):
        """Inverse should work with constant columns."""
        Theta = np.random.randn(100, 5)
        Theta[:, 0] = 1.0
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        Theta_recovered = scaler.inverse_transform(Theta_scaled)
        
        np.testing.assert_allclose(Theta, Theta_recovered, rtol=1e-10)


class TestColumnScalerValidation:
    """Test input validation."""
    
    def test_invalid_ndim(self):
        """Should reject non-2D input."""
        scaler = ColumnScaler()
        
        with pytest.raises(ValueError, match="2D"):
            scaler.fit(np.random.randn(100))
    
    def test_nan_input(self):
        """Should reject NaN input."""
        Theta = np.random.randn(100, 5)
        Theta[50, 2] = np.nan
        
        scaler = ColumnScaler()
        with pytest.raises(ValueError, match="NaN"):
            scaler.fit(Theta)
    
    def test_column_mismatch(self):
        """Should reject mismatched column count."""
        scaler = ColumnScaler()
        scaler.fit(np.random.randn(100, 5))
        
        with pytest.raises(ValueError, match="columns"):
            scaler.transform(np.random.randn(50, 3))


# =============================================================================
# STLSQOptimizer Tests
# =============================================================================

class TestSTLSQBasic:
    """Test basic STLSQ functionality."""
    
    def test_fit_basic(self):
        """Should fit without error."""
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100, 4)
        
        optimizer = STLSQOptimizer(threshold=0.1)
        optimizer.fit(Theta, dx)
        
        assert optimizer._is_fitted
        assert optimizer.coefficients_.shape == (10, 4)
    
    def test_predict(self):
        """Should predict correctly."""
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100, 4)
        
        optimizer = STLSQOptimizer(threshold=0.1)
        optimizer.fit(Theta, dx)
        
        dx_pred = optimizer.predict(Theta)
        assert dx_pred.shape == dx.shape
    
    def test_1d_dx_handling(self):
        """Should handle 1D dx input."""
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100)  # 1D
        
        optimizer = STLSQOptimizer(threshold=0.1)
        optimizer.fit(Theta, dx)
        
        assert optimizer.coefficients_.shape == (10, 1)


class TestSTLSQSparsity:
    """Test sparsity inducing behavior."""
    
    @pytest.fixture
    def sparse_problem(self):
        """Create problem with known sparse solution."""
        np.random.seed(42)
        n_samples, n_features, n_targets = 500, 10, 4
        
        # True sparse coefficients
        true_coef = np.zeros((n_features, n_targets))
        true_coef[0, :] = [0.5, -0.3, 0.2, 0.1]   # constant
        true_coef[1, :] = [1.0, -0.5, 0.0, 0.3]   # feature 1
        true_coef[3, :] = [0.0, 0.8, -1.0, 0.0]   # feature 3
        
        Theta = np.random.randn(n_samples, n_features)
        Theta[:, 0] = 1.0
        
        # Small noise
        dx = Theta @ true_coef + 0.001 * np.random.randn(n_samples, n_targets)
        
        return Theta, dx, true_coef
    
    def test_induces_sparsity(self, sparse_problem):
        """STLSQ should produce sparse coefficients."""
        Theta, dx, _ = sparse_problem
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        optimizer = STLSQOptimizer(threshold=0.05)
        optimizer.fit(Theta_scaled, dx)
        
        sparsity = optimizer.get_sparsity_info()
        assert sparsity['n_zero'] > 0, "Should have some zero coefficients"
    
    def test_support_recovery(self, sparse_problem):
        """Should recover correct support structure."""
        Theta, dx, true_coef = sparse_problem
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        optimizer = STLSQOptimizer(threshold=0.05)
        optimizer.fit(Theta_scaled, dx)
        
        true_support = true_coef != 0
        pred_support = optimizer.support_mask_
        
        # Should have high support accuracy
        accuracy = np.mean(true_support == pred_support)
        assert accuracy > 0.8, f"Support accuracy too low: {accuracy}"
    
    def test_coefficient_recovery(self, sparse_problem):
        """Should recover coefficient values approximately."""
        Theta, dx, true_coef = sparse_problem
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        optimizer = STLSQOptimizer(threshold=0.05)
        optimizer.fit(Theta_scaled, dx)
        
        # Unscale coefficients
        recovered = optimizer.get_unscaled_coefficients(scaler)
        
        # Compare on active terms
        error = np.abs(true_coef - recovered)
        max_error = error[true_coef != 0].max()
        
        assert max_error < 0.1, f"Max coefficient error too high: {max_error}"


class TestSTLSQThreshold:
    """Test threshold handling."""
    
    def test_threshold_zero_is_ols(self):
        """threshold=0 should give pure OLS (no sparsity)."""
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100, 4)
        
        optimizer = STLSQOptimizer(threshold=0)
        optimizer.fit(Theta, dx)
        
        # All terms should be active
        assert np.all(optimizer.support_mask_)
        assert optimizer.n_iter_ == 1
    
    def test_higher_threshold_more_sparse(self):
        """Higher threshold should give more sparsity."""
        np.random.seed(42)
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100, 4)
        
        opt_low = STLSQOptimizer(threshold=0.01)
        opt_low.fit(Theta, dx)
        
        opt_high = STLSQOptimizer(threshold=0.5)
        opt_high.fit(Theta, dx)
        
        low_nonzero = opt_low.get_sparsity_info()['n_nonzero']
        high_nonzero = opt_high.get_sparsity_info()['n_nonzero']
        
        assert high_nonzero <= low_nonzero
    
    def test_negative_threshold_error(self):
        """Negative threshold should raise error."""
        with pytest.raises(ValueError, match="threshold"):
            STLSQOptimizer(threshold=-0.1)


class TestSTLSQConvergence:
    """Test convergence behavior."""
    
    def test_converges_within_max_iter(self):
        """Should converge within max_iter."""
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100, 4)
        
        optimizer = STLSQOptimizer(threshold=0.1, max_iter=10)
        optimizer.fit(Theta, dx)
        
        assert optimizer.n_iter_ <= 10
    
    def test_reports_iteration_count(self):
        """Should report correct iteration count."""
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100, 4)
        
        optimizer = STLSQOptimizer(threshold=0.1)
        optimizer.fit(Theta, dx)
        
        assert optimizer.n_iter_ >= 1


class TestSTLSQValidation:
    """Test input validation."""
    
    def test_invalid_theta_ndim(self):
        """Should reject non-2D Theta."""
        optimizer = STLSQOptimizer()
        
        with pytest.raises(ValueError, match="2D"):
            optimizer.fit(np.random.randn(100), np.random.randn(100))
    
    def test_sample_count_mismatch(self):
        """Should reject mismatched sample counts."""
        optimizer = STLSQOptimizer()
        
        with pytest.raises(ValueError, match="mismatch"):
            optimizer.fit(np.random.randn(100, 10), np.random.randn(50, 4))
    
    def test_nan_in_theta(self):
        """Should reject NaN in Theta."""
        Theta = np.random.randn(100, 10)
        Theta[50, 5] = np.nan
        
        optimizer = STLSQOptimizer()
        with pytest.raises(ValueError, match="NaN"):
            optimizer.fit(Theta, np.random.randn(100, 4))
    
    def test_nan_in_dx(self):
        """Should reject NaN in dx."""
        dx = np.random.randn(100, 4)
        dx[25, 2] = np.nan
        
        optimizer = STLSQOptimizer()
        with pytest.raises(ValueError, match="NaN"):
            optimizer.fit(np.random.randn(100, 10), dx)
    
    def test_inf_in_theta(self):
        """Should reject Inf in Theta."""
        Theta = np.random.randn(100, 10)
        Theta[10, 3] = np.inf
        
        optimizer = STLSQOptimizer()
        with pytest.raises(ValueError, match="Inf"):
            optimizer.fit(Theta, np.random.randn(100, 4))
    
    def test_predict_not_fitted(self):
        """Should raise error if predict before fit."""
        optimizer = STLSQOptimizer()
        
        with pytest.raises(ValueError, match="not fitted"):
            optimizer.predict(np.random.randn(10, 5))


class TestSTLSQRidge:
    """Test ridge regularization."""
    
    def test_ridge_runs(self):
        """Ridge regularization should run without error."""
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100, 4)
        
        optimizer = STLSQOptimizer(threshold=0.1, ridge_alpha=0.01)
        optimizer.fit(Theta, dx)
        
        assert optimizer._is_fitted
    
    def test_ridge_changes_coefficients(self):
        """Ridge should change coefficients compared to no ridge."""
        np.random.seed(42)
        Theta = np.random.randn(100, 10)
        dx = np.random.randn(100, 4)
        
        opt_no_ridge = STLSQOptimizer(threshold=0, ridge_alpha=0)
        opt_no_ridge.fit(Theta, dx)
        
        opt_ridge = STLSQOptimizer(threshold=0, ridge_alpha=1.0)
        opt_ridge.fit(Theta, dx)
        
        # Coefficients should be different
        assert not np.allclose(opt_no_ridge.coefficients_, opt_ridge.coefficients_)


# =============================================================================
# Coefficient Unscaling Tests
# =============================================================================

class TestCoefficientUnscaling:
    """Test coefficient inverse transform."""
    
    def test_unscale_without_target_scale(self):
        """Unscale when dx is not normalized."""
        np.random.seed(42)
        
        # Setup
        Theta = np.random.randn(100, 5) * 2
        Theta[:, 0] = 1.0
        true_coef = np.array([[0.5], [1.0], [0.0], [-0.5], [0.2]])
        dx = Theta @ true_coef + 0.001 * np.random.randn(100, 1)
        
        # Scale and fit
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        opt = STLSQOptimizer(threshold=0.05)
        opt.fit(Theta_scaled, dx)
        
        # Unscale
        coef_unscaled = opt.get_unscaled_coefficients(scaler)
        
        # Compare with true (approximately)
        error = np.abs(true_coef.flatten() - coef_unscaled.flatten())
        assert error.max() < 0.1
    
    def test_unscale_with_target_scale(self):
        """Unscale when dx is normalized."""
        np.random.seed(42)
        
        # Setup with scaled dx
        Theta = np.random.randn(100, 5) * 2
        Theta[:, 0] = 1.0
        true_coef = np.array([[0.5], [1.0], [0.0], [-0.5], [0.2]])
        dx_orig = Theta @ true_coef
        
        # Scale dx
        dx_scale = np.std(dx_orig, axis=0)
        dx_scaled = dx_orig / dx_scale
        
        # Scale Theta and fit
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        opt = STLSQOptimizer(threshold=0)  # Pure OLS for exact recovery
        opt.fit(Theta_scaled, dx_scaled)
        
        # Unscale with target scale
        coef_unscaled = opt.get_unscaled_coefficients(scaler, target_scale=dx_scale)
        
        # Should match true coefficients (atol for zero comparison)
        np.testing.assert_allclose(
            coef_unscaled.flatten(), true_coef.flatten(), rtol=1e-6, atol=1e-12
        )


# =============================================================================
# CSV I/O Tests
# =============================================================================

class TestCoefficientIO:
    """Test coefficient save/load."""
    
    def test_save_load_roundtrip(self):
        """Save and load should preserve coefficients."""
        coefficients = np.random.randn(10, 4)
        feature_names = ['1'] + [f'f{i}' for i in range(1, 10)]
        target_names = ['dx0', 'dx1', 'dx2', 'dx3']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'coef.csv'
            save_coefficients_csv(coefficients, feature_names, target_names, path)
            
            # Verify header uses term_name
            with open(path, 'r') as f:
                header = f.readline().strip()
            assert header.startswith('term_name,'), f"Header should start with 'term_name,', got: {header}"
            
            loaded, feat_loaded, tgt_loaded = load_coefficients_csv(path)
            
            # atol for CSV 8-digit precision loss
            np.testing.assert_allclose(coefficients, loaded, atol=1e-7)
            assert feature_names == feat_loaded
            assert target_names == tgt_loaded
    
    def test_load_legacy_feature_header(self):
        """Should load CSV with legacy 'feature' header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'legacy.csv'
            
            # Write legacy format
            with open(path, 'w') as f:
                f.write('feature,dx0,dx1\n')
                f.write('1,0.5,0.3\n')
                f.write('x,1.0,-0.5\n')
            
            coef, feat, tgt = load_coefficients_csv(path)
            
            assert feat == ['1', 'x']
            assert tgt == ['dx0', 'dx1']
            np.testing.assert_allclose(coef, [[0.5, 0.3], [1.0, -0.5]])
    
    def test_save_creates_directory(self):
        """Should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'nested' / 'dir' / 'coef.csv'
            
            save_coefficients_csv(
                np.random.randn(5, 2),
                ['f0', 'f1', 'f2', 'f3', 'f4'],
                ['dx0', 'dx1'],
                path
            )
            
            assert path.exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestRealDatasetIntegration:
    """Test with actual generated dataset."""
    
    @pytest.fixture
    def dataset_path(self):
        from src.contracts import paths
        return paths.get_dataset_path('cartpole_ood_v1')
    
    @pytest.fixture
    def norm_stats(self):
        from src.data.normalization import load_norm_stats
        try:
            return load_norm_stats('cartpole_ood_v1')
        except FileNotFoundError:
            pytest.skip("norm_stats.json not found")
    
    def test_full_pipeline(self, dataset_path, norm_stats):
        """Test full STLSQ pipeline with real data."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        from src.sindy.library import SINDyLibrary, get_derivative_key
        from src.data.normalization import normalize_dataset
        
        # Load data
        data = np.load(dataset_path)
        train_x = data['train_x']
        train_u = data['train_u']
        train_dx = data['train_dx']
        
        # Normalize state/input/derivative
        x_norm, u_norm, dx_norm = normalize_dataset(
            train_x, train_u, train_dx,
            norm_stats, 'derivative_dx'
        )
        
        # Build feature matrix
        lib = SINDyLibrary(config='gate0_min')
        Theta = lib.fit_transform(x_norm, u_norm)
        
        # Flatten dx
        dx_flat = dx_norm.reshape(-1, 4)
        
        # Scale Theta
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        # Verify constant column preserved
        assert np.allclose(Theta_scaled[:, 0], 1.0), "Constant column not preserved"
        
        # Fit STLSQ
        optimizer = STLSQOptimizer(threshold=0.01)
        optimizer.fit(Theta_scaled, dx_flat)
        
        # Check results
        assert optimizer._is_fitted
        sparsity = optimizer.get_sparsity_info()
        assert sparsity['n_nonzero'] > 0, "No active terms"
        
        # Prediction quality
        dx_pred = optimizer.predict(Theta_scaled)
        r2 = 1 - np.var(dx_flat - dx_pred, axis=0) / np.var(dx_flat, axis=0)
        
        print(f"\n  Pipeline R² per target: {r2}")
        print(f"  Sparsity: {sparsity['sparsity']:.1%}")
        print(f"  Nonzero: {sparsity['n_nonzero']} / {sparsity['n_total']}")
        
        # Should have reasonable R²
        assert np.mean(r2) > 0.5, "R² too low"


class TestManifest:
    """Test manifest generation."""
    
    def test_get_optimizer_manifest(self):
        """Manifest should contain required fields."""
        Theta = np.random.randn(100, 10)
        Theta[:, 0] = 1.0
        dx = np.random.randn(100, 4)
        
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        optimizer = STLSQOptimizer(threshold=0.1)
        optimizer.fit(Theta_scaled, dx)
        
        feature_names = ['1'] + [f'f{i}' for i in range(1, 10)]
        manifest = get_optimizer_manifest(optimizer, scaler, feature_names)
        
        required_keys = [
            'optimizer', 'threshold', 'sparsity', 'n_nonzero',
            'scaler_type', 'active_terms_per_target'
        ]
        
        for key in required_keys:
            assert key in manifest, f"Missing key: {key}"
        
        assert manifest['scaler_type'] == 'scale_only'


class TestConfigs:
    """Test STLSQ configurations."""
    
    def test_gate0_config_exists(self):
        """Gate0 config should exist."""
        assert 'gate0' in STLSQ_CONFIGS
        assert 'threshold' in STLSQ_CONFIGS['gate0']
    
    def test_gate1_grid_exists(self):
        """Gate1 grid should exist."""
        assert 'gate1_grid' in STLSQ_CONFIGS
        assert 'thresholds' in STLSQ_CONFIGS['gate1_grid']
        
        thresholds = STLSQ_CONFIGS['gate1_grid']['thresholds']
        assert 0 in thresholds
        assert len(thresholds) >= 5
    
    def test_target_names(self):
        """TARGET_NAMES should be correct."""
        assert len(TARGET_NAMES) == 4
        assert 'x_dot' in TARGET_NAMES
        assert 'theta_ddot' in TARGET_NAMES


if __name__ == '__main__':
    pytest.main([__file__, '-v'])