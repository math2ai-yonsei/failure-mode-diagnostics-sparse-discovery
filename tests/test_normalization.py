"""
S04-A: Test Normalization Utilities

Tests for normalization functions with focus on:
1. mean≈0, std≈1 after normalization
2. Inverse accuracy (denormalize ∘ normalize = identity)
3. Integration with real dataset
4. Deterministic leakage prevention verification
"""
import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from src.data.normalization import (
    compute_stats,
    normalize,
    denormalize,
    compute_norm_stats,
    normalize_dataset,
    denormalize_dataset,
    stats_to_json_serializable,
    stats_from_json,
    save_norm_stats,
    load_norm_stats,
    generate_norm_stats_from_dataset,
    validate_normalization,
    validate_inverse,
)


# =============================================================================
# Basic Function Tests
# =============================================================================

class TestComputeStats:
    """Test compute_stats function."""
    
    def test_3d_array(self):
        """Should work with (N, T, D) arrays."""
        data = np.random.randn(10, 50, 4)
        stats = compute_stats(data)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert stats['mean'].shape == (4,)
        assert stats['std'].shape == (4,)
    
    def test_2d_array(self):
        """Should work with (N, D) arrays."""
        data = np.random.randn(100, 4)
        stats = compute_stats(data)
        
        assert stats['mean'].shape == (4,)
        assert stats['std'].shape == (4,)
    
    def test_known_statistics(self):
        """Should compute correct statistics for known data."""
        np.random.seed(42)
        N, T, D = 1000, 100, 2
        
        # Known mean and std
        true_mean = np.array([5.0, -3.0])
        true_std = np.array([2.0, 0.5])
        
        data = np.random.randn(N, T, D) * true_std + true_mean
        stats = compute_stats(data)
        
        np.testing.assert_allclose(stats['mean'], true_mean, rtol=0.05)
        np.testing.assert_allclose(stats['std'], true_std, rtol=0.05)
    
    def test_constant_feature_handling(self):
        """Should handle constant features (std=0) by setting std=1."""
        data = np.ones((10, 50, 4))  # All constant
        stats = compute_stats(data)
        
        # std should be 1.0 (not 0) to prevent division by zero
        np.testing.assert_array_equal(stats['std'], np.ones(4))
    
    def test_dtype_float64(self):
        """Output should be float64."""
        data = np.random.randn(10, 50, 4).astype(np.float32)
        stats = compute_stats(data)
        
        assert stats['mean'].dtype == np.float64
        assert stats['std'].dtype == np.float64


class TestNormalize:
    """Test normalize function."""
    
    def test_output_shape(self):
        """Output shape should match input."""
        data = np.random.randn(10, 50, 4)
        stats = compute_stats(data)
        normalized = normalize(data, stats)
        
        assert normalized.shape == data.shape
    
    def test_mean_zero_std_one(self):
        """Normalized data should have mean≈0, std≈1."""
        np.random.seed(42)
        data = np.random.randn(100, 50, 4) * 5 + 10
        stats = compute_stats(data)
        normalized = normalize(data, stats)
        
        flat = normalized.reshape(-1, 4)
        np.testing.assert_allclose(flat.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(flat.std(axis=0), 1, atol=1e-10)
    
    def test_broadcasting_1d(self):
        """Should work with 1D input (single sample)."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        stats = {'mean': np.array([0.5, 1.0, 1.5, 2.0]),
                 'std': np.array([0.5, 0.5, 0.5, 0.5])}
        
        normalized = normalize(data, stats)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(normalized, expected)
    
    def test_broadcasting_2d(self):
        """Should work with 2D input (T, D)."""
        data = np.random.randn(50, 4)
        stats = compute_stats(data)
        normalized = normalize(data, stats)
        
        assert normalized.shape == data.shape


class TestDenormalize:
    """Test denormalize function."""
    
    def test_inverse_property(self):
        """denormalize(normalize(x)) should equal x."""
        np.random.seed(42)
        data = np.random.randn(10, 50, 4) * 5 + 10
        stats = compute_stats(data)
        
        normalized = normalize(data, stats)
        recovered = denormalize(normalized, stats)
        
        np.testing.assert_allclose(recovered, data, rtol=1e-10)
    
    def test_known_transform(self):
        """Should correctly invert known transformation."""
        normalized = np.array([0.0, 1.0, -1.0, 2.0])
        stats = {'mean': np.array([5.0, 5.0, 5.0, 5.0]),
                 'std': np.array([2.0, 2.0, 2.0, 2.0])}
        
        original = denormalize(normalized, stats)
        expected = np.array([5.0, 7.0, 3.0, 9.0])
        np.testing.assert_allclose(original, expected)


# =============================================================================
# Dataset-Level Tests
# =============================================================================

class TestComputeNormStats:
    """Test compute_norm_stats function."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample train data."""
        np.random.seed(42)
        N, T = 50, 101
        return {
            'train_x': np.random.randn(N, T, 4) * 2 + 1,
            'train_u': np.random.randn(N, T, 1) * 5,
            'train_dx': np.random.randn(N, T, 4) * 3,
            'train_dx_savgol': np.random.randn(N, T, 4) * 3.1,
        }
    
    def test_all_keys_present(self, sample_data):
        """Should return all expected keys."""
        stats = compute_norm_stats(**sample_data)
        
        assert 'state' in stats
        assert 'input' in stats
        assert 'derivative_dx' in stats
        assert 'derivative_dx_savgol' in stats
    
    def test_optional_derivatives(self, sample_data):
        """Should work without derivative inputs."""
        stats = compute_norm_stats(
            sample_data['train_x'],
            sample_data['train_u']
        )
        
        assert 'state' in stats
        assert 'input' in stats
        assert 'derivative_dx' not in stats
        assert 'derivative_dx_savgol' not in stats
    
    def test_stats_shapes(self, sample_data):
        """Stats should have correct shapes."""
        stats = compute_norm_stats(**sample_data)
        
        assert stats['state']['mean'].shape == (4,)
        assert stats['state']['std'].shape == (4,)
        assert stats['input']['mean'].shape == (1,)
        assert stats['input']['std'].shape == (1,)
        assert stats['derivative_dx']['mean'].shape == (4,)
        assert stats['derivative_dx_savgol']['mean'].shape == (4,)


class TestNormalizeDataset:
    """Test normalize_dataset function."""
    
    @pytest.fixture
    def dataset_and_stats(self):
        """Generate sample dataset and compute stats."""
        np.random.seed(42)
        N, T = 50, 101
        train_x = np.random.randn(N, T, 4) * 2 + 1
        train_u = np.random.randn(N, T, 1) * 5
        train_dx = np.random.randn(N, T, 4) * 3
        
        stats = compute_norm_stats(train_x, train_u, train_dx)
        return train_x, train_u, train_dx, stats
    
    def test_normalize_all(self, dataset_and_stats):
        """Should normalize all components."""
        x, u, dx, stats = dataset_and_stats
        x_n, u_n, dx_n = normalize_dataset(x, u, dx, stats, 'derivative_dx')
        
        # Check mean≈0, std≈1 for each
        x_flat = x_n.reshape(-1, 4)
        np.testing.assert_allclose(x_flat.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(x_flat.std(axis=0), 1, atol=1e-10)
        
        u_flat = u_n.reshape(-1, 1)
        np.testing.assert_allclose(u_flat.mean(axis=0), 0, atol=1e-10)
    
    def test_inverse_dataset(self, dataset_and_stats):
        """denormalize_dataset should invert normalize_dataset."""
        x, u, dx, stats = dataset_and_stats
        
        x_n, u_n, dx_n = normalize_dataset(x, u, dx, stats, 'derivative_dx')
        x_r, u_r, dx_r = denormalize_dataset(x_n, u_n, dx_n, stats, 'derivative_dx')
        
        np.testing.assert_allclose(x_r, x, rtol=1e-10)
        np.testing.assert_allclose(u_r, u, rtol=1e-10)
        np.testing.assert_allclose(dx_r, dx, rtol=1e-10)
    
    def test_missing_derivative_key_raises_error(self, dataset_and_stats):
        """Should raise KeyError if derivative_key not in stats (fail-fast)."""
        x, u, dx, stats = dataset_and_stats
        
        with pytest.raises(KeyError, match="derivative_key"):
            normalize_dataset(x, u, dx, stats, 'nonexistent_key')
    
    def test_denormalize_missing_key_raises_error(self, dataset_and_stats):
        """Should raise KeyError on denormalize if derivative_key not in stats."""
        x, u, dx, stats = dataset_and_stats
        
        # First normalize correctly
        x_n, u_n, dx_n = normalize_dataset(x, u, dx, stats, 'derivative_dx')
        
        # Then try to denormalize with wrong key
        with pytest.raises(KeyError, match="derivative_key"):
            denormalize_dataset(x_n, u_n, dx_n, stats, 'nonexistent_key')


# =============================================================================
# I/O Tests
# =============================================================================

class TestJsonSerialization:
    """Test JSON serialization functions."""
    
    def test_round_trip(self):
        """stats_from_json(stats_to_json_serializable(x)) ≈ x."""
        original = {
            'state': {
                'mean': np.array([1.0, 2.0, 3.0, 4.0]),
                'std': np.array([0.5, 0.5, 0.5, 0.5])
            },
            'input': {
                'mean': np.array([0.0]),
                'std': np.array([5.0])
            }
        }
        
        json_format = stats_to_json_serializable(original)
        recovered = stats_from_json(json_format)
        
        np.testing.assert_allclose(
            recovered['state']['mean'], original['state']['mean']
        )
        np.testing.assert_allclose(
            recovered['input']['std'], original['input']['std']
        )
    
    def test_json_compatible(self):
        """Serialized stats should be JSON-compatible."""
        stats = {
            'state': {'mean': np.array([1.0, 2.0]), 'std': np.array([0.5, 0.5])}
        }
        json_format = stats_to_json_serializable(stats)
        
        # Should not raise
        json_str = json.dumps(json_format)
        parsed = json.loads(json_str)
        
        assert parsed['state']['mean'] == [1.0, 2.0]


class TestSaveLoad:
    """Test save/load functions."""
    
    def test_save_and_load(self, tmp_path, monkeypatch):
        """Should save and load correctly."""
        # Mock paths.get_norm_stats_path
        norm_stats_path = tmp_path / "norm_stats.json"
        
        def mock_get_norm_stats_path(version, system):
            return norm_stats_path
        
        monkeypatch.setattr(
            'src.data.normalization.paths.get_norm_stats_path',
            mock_get_norm_stats_path
        )
        
        # Create stats
        original_stats = {
            'state': {'mean': np.array([1.0, 2.0, 3.0, 4.0]),
                      'std': np.array([0.5, 0.6, 0.7, 0.8])},
            'input': {'mean': np.array([0.0]), 'std': np.array([5.0])},
            'derivative_dx': {'mean': np.array([0.1, 0.2, 0.3, 0.4]),
                              'std': np.array([1.0, 1.0, 1.0, 1.0])}
        }
        
        # Save
        save_norm_stats(original_stats, 'test_v1')
        assert norm_stats_path.exists()
        
        # Load
        loaded_stats = load_norm_stats('test_v1')
        
        # Verify
        np.testing.assert_allclose(
            loaded_stats['state']['mean'], original_stats['state']['mean']
        )
        np.testing.assert_allclose(
            loaded_stats['derivative_dx']['std'], 
            original_stats['derivative_dx']['std']
        )
        
        # Check metadata
        with open(norm_stats_path) as f:
            raw = json.load(f)
        assert 'created_at' in raw
        assert raw['computed_from'] == 'train'


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Test validation utilities."""
    
    def test_validate_normalization_pass(self):
        """Should pass for correctly normalized data."""
        np.random.seed(42)
        data = np.random.randn(100, 50, 4) * 5 + 10
        stats = compute_stats(data)
        
        result = validate_normalization(data, stats)
        
        assert result['mean_ok'] is True
        assert result['std_ok'] is True
    
    def test_validate_normalization_fail_mean(self):
        """Should detect wrong mean."""
        data = np.random.randn(100, 50, 4)
        # Wrong stats
        stats = {'mean': np.array([10.0, 10.0, 10.0, 10.0]),
                 'std': np.array([1.0, 1.0, 1.0, 1.0])}
        
        result = validate_normalization(data, stats)
        
        # Mean will be off (≈-10 instead of 0)
        assert result['mean_ok'] is False
    
    def test_validate_inverse_pass(self):
        """Should pass for correct inverse."""
        data = np.random.randn(10, 50, 4) * 5 + 10
        stats = compute_stats(data)
        
        assert validate_inverse(data, stats) is True
    
    def test_validate_inverse_numerical_precision(self):
        """Should handle numerical precision."""
        # Use extreme values to test precision
        data = np.random.randn(10, 50, 4) * 1e6 + 1e8
        stats = compute_stats(data)
        
        # Should still pass with appropriate tolerance
        assert validate_inverse(data, stats, rtol=1e-8, atol=1e-8) is True


# =============================================================================
# Integration with Real Dataset
# =============================================================================

class TestRealDataset:
    """Integration tests with actual dataset."""
    
    @pytest.fixture
    def dataset_path(self):
        """Get path to dataset."""
        from src.contracts import paths
        return paths.get_dataset_path('cartpole_ood_v1')
    
    def test_generate_from_real_dataset(self, dataset_path):
        """Test generating stats from real dataset."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        stats = generate_norm_stats_from_dataset('cartpole_ood_v1', save=False)
        
        # Check all keys
        assert 'state' in stats
        assert 'input' in stats
        assert 'derivative_dx' in stats
        # dx_savgol might not exist yet
    
    def test_normalize_real_train(self, dataset_path):
        """Normalized train data should have mean≈0, std≈1."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        data = np.load(dataset_path)
        train_x = data['train_x']
        
        stats = compute_stats(train_x)
        result = validate_normalization(train_x, stats)
        
        assert result['mean_ok'], f"Mean not zero: {result['actual_mean']}"
        assert result['std_ok'], f"Std not one: {result['actual_std']}"
    
    def test_inverse_on_real_data(self, dataset_path):
        """Inverse should be exact on real data."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        data = np.load(dataset_path)
        train_x = data['train_x']
        
        stats = compute_stats(train_x)
        assert validate_inverse(train_x, stats)
    
    def test_no_leakage_deterministic(self, dataset_path):
        """
        Deterministic leakage prevention verification.
        
        Verifies that:
        1. Metadata says computed_from='train'
        2. Recomputed stats from train splits match saved stats exactly
        3. Train normalization produces exact mean=0, std=1
        
        Note: Uses rtol=0 for strict deterministic comparison.
        """
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        # Load saved norm_stats
        stats = load_norm_stats('cartpole_ood_v1')
        
        # ========================================
        # 결정적 증거 1: computed_from 메타데이터 확인
        # ========================================
        assert stats.get('computed_from') == 'train', \
            f"Expected computed_from='train', got '{stats.get('computed_from')}'"
        
        # ========================================
        # 결정적 증거 2: train split으로 재계산 → 저장값과 일치 확인
        # (rtol=0으로 엄격한 deterministic 비교)
        # ========================================
        data = np.load(dataset_path)
        
        # 2-1. state (train_x)
        train_x = data['train_x']
        recomputed_state = compute_stats(train_x)
        np.testing.assert_allclose(
            recomputed_state['mean'], stats['state']['mean'],
            rtol=0, atol=1e-10,
            err_msg="state.mean mismatch: possible leakage"
        )
        np.testing.assert_allclose(
            recomputed_state['std'], stats['state']['std'],
            rtol=0, atol=1e-10,
            err_msg="state.std mismatch: possible leakage"
        )
        
        # 2-2. input (train_u)
        train_u = data['train_u']
        recomputed_input = compute_stats(train_u)
        np.testing.assert_allclose(
            recomputed_input['mean'], stats['input']['mean'],
            rtol=0, atol=1e-10,
            err_msg="input.mean mismatch: possible leakage"
        )
        np.testing.assert_allclose(
            recomputed_input['std'], stats['input']['std'],
            rtol=0, atol=1e-10,
            err_msg="input.std mismatch: possible leakage"
        )
        
        # 2-3. derivative_dx_savgol (if exists)
        if 'train_dx_savgol' in data and 'derivative_dx_savgol' in stats:
            train_dx_savgol = data['train_dx_savgol']
            recomputed_dx_savgol = compute_stats(train_dx_savgol)
            np.testing.assert_allclose(
                recomputed_dx_savgol['mean'], stats['derivative_dx_savgol']['mean'],
                rtol=0, atol=1e-10,
                err_msg="derivative_dx_savgol.mean mismatch: possible leakage"
            )
            np.testing.assert_allclose(
                recomputed_dx_savgol['std'], stats['derivative_dx_savgol']['std'],
                rtol=0, atol=1e-10,
                err_msg="derivative_dx_savgol.std mismatch: possible leakage"
            )
        
        # 2-4. derivative_dx (if exists) - SSOT 산출물 완전성 검증
        if 'train_dx' in data and 'derivative_dx' in stats:
            train_dx = data['train_dx']
            recomputed_dx = compute_stats(train_dx)
            np.testing.assert_allclose(
                recomputed_dx['mean'], stats['derivative_dx']['mean'],
                rtol=0, atol=1e-10,
                err_msg="derivative_dx.mean mismatch: possible leakage"
            )
            np.testing.assert_allclose(
                recomputed_dx['std'], stats['derivative_dx']['std'],
                rtol=0, atol=1e-10,
                err_msg="derivative_dx.std mismatch: possible leakage"
            )
        
        # ========================================
        # 검증 3: train 정규화가 정확히 mean=0, std=1
        # ========================================
        train_norm = normalize(train_x, stats['state'])
        train_flat = train_norm.reshape(-1, 4)
        np.testing.assert_allclose(train_flat.mean(axis=0), 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(train_flat.std(axis=0), 1, rtol=0, atol=1e-10)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])