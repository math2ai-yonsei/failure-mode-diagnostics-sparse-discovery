"""
S07: Test E-SINDy Module

Tests for E-SINDy ensemble learning:
1. ESINDyEnsemble class
2. Trajectory-level bootstrap
3. Threshold sweep
4. Best threshold selection
5. I/O helpers
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.sindy.esindy import (
    ESINDyEnsemble, ESINDyResult,
    threshold_sweep, select_best_threshold,
    save_coefficients_std_csv, save_inclusion_prob_csv, save_threshold_sweep_csv,
)
from src.sindy.optimizer import ColumnScaler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_trajectory_data():
    """Create synthetic trajectory data for testing."""
    np.random.seed(42)
    n_traj, T, n_features, n_targets = 10, 50, 8, 4
    
    # True sparse coefficients
    true_coef = np.zeros((n_features, n_targets))
    true_coef[0, :] = [0.5, -0.3, 0.2, 0.1]
    true_coef[1, :] = [1.0, -0.5, 0.0, 0.3]
    true_coef[3, :] = [0.0, 0.8, -1.0, 0.0]
    
    Theta_raw = np.random.randn(n_traj * T, n_features)
    Theta_raw[:, 0] = 1.0
    dx = Theta_raw @ true_coef + 0.05 * np.random.randn(n_traj * T, n_targets)
    
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta_raw)
    
    return {
        'Theta_scaled': Theta_scaled,
        'dx': dx,
        'n_traj': n_traj,
        'T': T,
        'n_features': n_features,
        'n_targets': n_targets,
        'true_coef': true_coef,
        'scaler': scaler,
    }


@pytest.fixture
def train_val_split_data(synthetic_trajectory_data):
    """Split synthetic data into train/val."""
    d = synthetic_trajectory_data
    n_train, n_val = 8, 2
    T = d['T']
    
    train_end = n_train * T
    
    return {
        'Theta_train': d['Theta_scaled'][:train_end],
        'dx_train': d['dx'][:train_end],
        'Theta_val': d['Theta_scaled'][train_end:],
        'dx_val': d['dx'][train_end:],
        'n_train': n_train,
        'n_val': n_val,
        'T': T,
        'scaler': d['scaler'],
        'true_coef': d['true_coef'],
    }


# =============================================================================
# ESINDyEnsemble Tests
# =============================================================================

class TestESINDyEnsembleInit:
    """Test ESINDyEnsemble initialization."""
    
    def test_default_values(self):
        ensemble = ESINDyEnsemble()
        assert ensemble.n_bootstrap == 20
        assert ensemble.threshold == 0.01
        assert ensemble.inclusion_eps == 0.0
    
    def test_custom_values(self):
        ensemble = ESINDyEnsemble(n_bootstrap=30, threshold=0.05)
        assert ensemble.n_bootstrap == 30
        assert ensemble.threshold == 0.05
    
    def test_invalid_n_bootstrap(self):
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            ESINDyEnsemble(n_bootstrap=1)
    
    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold must be >= 0"):
            ESINDyEnsemble(threshold=-0.1)


class TestESINDyEnsembleFit:
    """Test ESINDyEnsemble.fit()."""
    
    def test_fit_basic(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=10, threshold=0.05, random_state=42)
        
        ensemble.fit(
            d['Theta_scaled'], d['dx'],
            n_trajectories=d['n_traj'], T=d['T'],
            scaler=d['scaler'], target_scale=None
        )
        
        assert ensemble._is_fitted
        assert ensemble.coefficients_mean_.shape == (d['n_features'], d['n_targets'])
        assert ensemble.coefficients_std_.shape == (d['n_features'], d['n_targets'])
        assert ensemble.inclusion_probability_.shape == (d['n_features'], d['n_targets'])
    
    def test_fit_trajectory_bootstrap_correct_count(self, synthetic_trajectory_data):
        """Verify trajectory-level bootstrap samples correct number."""
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=5, threshold=0.05, random_state=42)
        
        ensemble.fit(
            d['Theta_scaled'], d['dx'],
            n_trajectories=d['n_traj'], T=d['T'],
            scaler=d['scaler'], target_scale=None
        )
        
        assert len(ensemble._individual_coefficients) == 5
    
    def test_fit_reproducibility(self, synthetic_trajectory_data):
        """Same random_state should give same results."""
        d = synthetic_trajectory_data
        
        e1 = ESINDyEnsemble(n_bootstrap=10, threshold=0.05, random_state=123)
        e1.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        e2 = ESINDyEnsemble(n_bootstrap=10, threshold=0.05, random_state=123)
        e2.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        np.testing.assert_array_equal(e1.coefficients_mean_, e2.coefficients_mean_)
    
    def test_fit_different_seeds(self, synthetic_trajectory_data):
        """Different seeds should give different results."""
        d = synthetic_trajectory_data
        
        e1 = ESINDyEnsemble(n_bootstrap=10, threshold=0.05, random_state=1)
        e1.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        e2 = ESINDyEnsemble(n_bootstrap=10, threshold=0.05, random_state=2)
        e2.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        assert not np.allclose(e1.coefficients_mean_, e2.coefficients_mean_)
    
    def test_fit_sample_mismatch_error(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble()
        
        with pytest.raises(ValueError, match="Sample count mismatch"):
            ensemble.fit(
                d['Theta_scaled'], d['dx'][:100],  # Wrong size
                d['n_traj'], d['T'], d['scaler'], None
            )
    
    def test_fit_trajectory_count_error(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble()
        
        with pytest.raises(ValueError, match="expected"):
            ensemble.fit(
                d['Theta_scaled'], d['dx'],
                n_trajectories=5,  # Wrong count
                T=d['T'],
                scaler=d['scaler'],
                target_scale=None
            )


class TestESINDyEnsemblePredict:
    """Test ESINDyEnsemble.predict()."""
    
    def test_predict_not_fitted(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble()
        
        with pytest.raises(ValueError, match="Not fitted"):
            ensemble.predict(d['Theta_scaled'], d['scaler'], None)
    
    def test_predict_shape(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=5, threshold=0.05, random_state=42)
        ensemble.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        pred = ensemble.predict(d['Theta_scaled'], d['scaler'], None)
        assert pred.shape == d['dx'].shape


class TestESINDyEnsembleSparsity:
    """Test sparsity and active terms."""
    
    def test_get_sparsity_info(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=10, threshold=0.1, random_state=42)
        ensemble.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        info = ensemble.get_sparsity_info()
        
        assert 'n_active' in info
        assert 'sparsity' in info
        assert 'mean_coef_std' in info
        assert 0 <= info['sparsity'] <= 1
    
    def test_get_result(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=5, threshold=0.05, random_state=42)
        ensemble.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        result = ensemble.get_result()
        
        assert isinstance(result, ESINDyResult)
        assert result.n_bootstrap == 5
        assert result.bootstrap_unit == 'trajectory'
        assert len(result.individual_coefficients) == 5


class TestESINDyCoeffRecovery:
    """Test coefficient recovery quality."""
    
    def test_sparse_recovery(self, synthetic_trajectory_data):
        """E-SINDy should recover sparse structure."""
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=20, threshold=0.05, random_state=42)
        ensemble.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        true_support = d['true_coef'] != 0
        pred_support = ensemble.inclusion_probability_ > 0.5
        
        support_accuracy = np.mean(true_support == pred_support)
        assert support_accuracy > 0.7  # Reasonable recovery
    
    def test_coefficient_values(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=20, threshold=0.05, random_state=42)
        ensemble.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        # Check that recovered coefficients are close to true
        error = np.abs(d['true_coef'] - ensemble.coefficients_mean_)
        assert error.mean() < 0.2


# =============================================================================
# Threshold Sweep Tests
# =============================================================================

class TestThresholdSweep:
    """Test threshold_sweep function."""
    
    def test_sweep_basic(self, train_val_split_data):
        d = train_val_split_data
        thresholds = [0, 0.01, 0.05]
        
        results = threshold_sweep(
            d['Theta_train'], d['dx_train'],
            d['Theta_val'], d['dx_val'],
            thresholds,
            d['n_train'], d['n_val'], d['T'],
            d['scaler'], None,
            n_bootstrap=5, random_state=42
        )
        
        assert len(results) == len(thresholds)
        for r in results:
            assert 'threshold' in r
            assert 'train_r2_mean' in r
            assert 'val_r2_mean' in r
            assert 'sparsity' in r
    
    def test_sweep_monotonic_sparsity(self, train_val_split_data):
        """Higher threshold should give higher sparsity."""
        d = train_val_split_data
        thresholds = [0, 0.01, 0.05, 0.1]
        
        results = threshold_sweep(
            d['Theta_train'], d['dx_train'],
            d['Theta_val'], d['dx_val'],
            thresholds,
            d['n_train'], d['n_val'], d['T'],
            d['scaler'], None,
            n_bootstrap=5, random_state=42
        )
        
        sparsities = [r['sparsity'] for r in results]
        # Sparsity should generally increase (not strictly due to randomness)
        assert sparsities[-1] >= sparsities[0]


class TestSelectBestThreshold:
    """Test select_best_threshold function."""
    
    def test_select_by_val_r2(self):
        results = [
            {'threshold': 0.01, 'val_r2_mean': 0.8, 'sparsity': 0.3, 'mean_coef_std': 0.1},
            {'threshold': 0.05, 'val_r2_mean': 0.9, 'sparsity': 0.5, 'mean_coef_std': 0.1},
            {'threshold': 0.1, 'val_r2_mean': 0.7, 'sparsity': 0.7, 'mean_coef_std': 0.1},
        ]
        
        best = select_best_threshold(results)
        assert best['threshold'] == 0.05
    
    def test_tie_break_sparsity(self):
        results = [
            {'threshold': 0.01, 'val_r2_mean': 0.9, 'sparsity': 0.3, 'mean_coef_std': 0.1},
            {'threshold': 0.05, 'val_r2_mean': 0.9, 'sparsity': 0.5, 'mean_coef_std': 0.1},
        ]
        
        best = select_best_threshold(results, tie_tolerance=0.01)
        assert best['threshold'] == 0.05  # Higher sparsity wins
    
    def test_tie_break_uncertainty(self):
        results = [
            {'threshold': 0.01, 'val_r2_mean': 0.9, 'sparsity': 0.5, 'mean_coef_std': 0.2},
            {'threshold': 0.05, 'val_r2_mean': 0.9, 'sparsity': 0.5, 'mean_coef_std': 0.1},
        ]
        
        best = select_best_threshold(results, tie_tolerance=0.01)
        assert best['threshold'] == 0.05  # Lower uncertainty wins
    
    def test_empty_results_error(self):
        with pytest.raises(ValueError, match="empty"):
            select_best_threshold([])


# =============================================================================
# I/O Tests
# =============================================================================

class TestIO:
    """Test I/O helper functions."""
    
    def test_save_coefficients_std_csv(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=5, threshold=0.05, random_state=42)
        ensemble.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'coef_std.csv'
            save_coefficients_std_csv(
                ensemble.coefficients_std_,
                [f'f{i}' for i in range(d['n_features'])],
                [f't{i}' for i in range(d['n_targets'])],
                path
            )
            assert path.exists()
            content = path.read_text()
            assert 'term_name' in content
    
    def test_save_inclusion_prob_csv(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble(n_bootstrap=5, threshold=0.05, random_state=42)
        ensemble.fit(d['Theta_scaled'], d['dx'], d['n_traj'], d['T'], d['scaler'], None)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'incl_prob.csv'
            save_inclusion_prob_csv(
                ensemble.inclusion_probability_,
                [f'f{i}' for i in range(d['n_features'])],
                [f't{i}' for i in range(d['n_targets'])],
                path
            )
            assert path.exists()
    
    def test_save_threshold_sweep_csv(self):
        results = [
            {'threshold': 0.01, 'train_r2_mean': 0.9, 'val_r2_mean': 0.85,
             'sparsity': 0.3, 'n_active': 20, 'mean_coef_std': 0.1},
            {'threshold': 0.05, 'train_r2_mean': 0.85, 'val_r2_mean': 0.8,
             'sparsity': 0.5, 'n_active': 15, 'mean_coef_std': 0.08},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'sweep.csv'
            save_threshold_sweep_csv(results, path)
            assert path.exists()
            content = path.read_text()
            assert 'threshold' in content
            assert '0.010000' in content


# =============================================================================
# Validation Tests
# =============================================================================

class TestInputValidation:
    """Test input validation."""
    
    def test_theta_1d_error(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        ensemble = ESINDyEnsemble()
        
        with pytest.raises(ValueError, match="Theta must be 2D"):
            ensemble.fit(
                d['Theta_scaled'].flatten(), d['dx'],
                d['n_traj'], d['T'], d['scaler'], None
            )
    
    def test_nan_in_theta(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        Theta_bad = d['Theta_scaled'].copy()
        Theta_bad[0, 0] = np.nan
        
        ensemble = ESINDyEnsemble()
        with pytest.raises(ValueError, match="NaN or Inf"):
            ensemble.fit(Theta_bad, d['dx'], d['n_traj'], d['T'], d['scaler'], None)
    
    def test_inf_in_dx(self, synthetic_trajectory_data):
        d = synthetic_trajectory_data
        dx_bad = d['dx'].copy()
        dx_bad[0, 0] = np.inf
        
        ensemble = ESINDyEnsemble()
        with pytest.raises(ValueError, match="NaN or Inf"):
            ensemble.fit(d['Theta_scaled'], dx_bad, d['n_traj'], d['T'], d['scaler'], None)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])