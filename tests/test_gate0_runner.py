"""
S06: Test Gate0 Runner

Tests for Gate0 SINDy baseline pipeline:
1. Configuration handling
2. Pipeline execution
3. Artifact generation
4. Metrics computation
5. Context Packet generation
"""
import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.experiments.gate0_runner import Gate0Runner, Gate0Config
from src.contracts import paths


# =============================================================================
# Configuration Tests
# =============================================================================

class TestGate0Config:
    """Test Gate0Config dataclass."""
    
    def test_default_values(self):
        """Should have correct default values."""
        config = Gate0Config()
        
        assert config.dataset_version == 'cartpole_ood_v1'
        assert config.system == 'cartpole'
        assert config.n_train == 10
        assert config.seed == 0
        assert config.track == 'standardized'
        assert config.threshold == 0.01
        assert config.gate == 'gate0'
        assert config.method == 'sindy'
    
    def test_from_dict(self):
        """Should create config from dict."""
        d = {
            'dataset_version': 'test_v1',
            'n_train': 20,
            'seed': 42,
            'track': 'author_recommended',
        }
        config = Gate0Config.from_dict(d)
        
        assert config.dataset_version == 'test_v1'
        assert config.n_train == 20
        assert config.seed == 42
        assert config.track == 'author_recommended'
        # Defaults preserved
        assert config.threshold == 0.01
    
    def test_from_dict_ignores_unknown(self):
        """Should ignore unknown keys."""
        d = {
            'n_train': 5,
            'unknown_key': 'value',
        }
        config = Gate0Config.from_dict(d)
        
        assert config.n_train == 5
        assert not hasattr(config, 'unknown_key')
    
    def test_from_yaml(self):
        """Should load config from YAML file."""
        yaml_content = """
dataset_version: yaml_test
n_train: 15
seed: 123
track: standardized
threshold: 0.02
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)
        
        try:
            config = Gate0Config.from_yaml(yaml_path)
            
            assert config.dataset_version == 'yaml_test'
            assert config.n_train == 15
            assert config.seed == 123
            assert config.threshold == 0.02
        finally:
            yaml_path.unlink()


# =============================================================================
# Runner Initialization Tests
# =============================================================================

class TestGate0RunnerInit:
    """Test Gate0Runner initialization."""
    
    def test_init_creates_run_id(self):
        """Should generate unique run_id."""
        config = Gate0Config(note='test')
        runner = Gate0Runner(config)
        
        assert runner.run_id is not None
        assert 'test' in runner.run_id
        assert len(runner.run_id) > 10
    
    def test_init_sets_results_dir(self):
        """Should set correct results directory."""
        config = Gate0Config(
            dataset_version='test_v1',
            n_train=10,
            seed=0,
            track='standardized',
            note='test'
        )
        runner = Gate0Runner(config)
        
        # Check path structure
        assert 'test_v1' in str(runner.results_dir)
        assert 'gate0' in str(runner.results_dir)
        assert 'standardized' in str(runner.results_dir)
        assert 'sindy' in str(runner.results_dir)
        assert 'n10' in str(runner.results_dir)
        assert 'seed0' in str(runner.results_dir)
    
    def test_init_sets_derivative_key(self):
        """Should set correct derivative key for track."""
        config_std = Gate0Config(track='standardized')
        runner_std = Gate0Runner(config_std)
        assert runner_std.derivative_key == 'derivative_dx_savgol'
        
        config_auth = Gate0Config(track='author_recommended')
        runner_auth = Gate0Runner(config_auth)
        assert runner_auth.derivative_key == 'derivative_dx'


# =============================================================================
# Integration Tests (require real dataset)
# =============================================================================

class TestGate0RunnerIntegration:
    """Integration tests with real dataset."""
    
    @pytest.fixture
    def dataset_exists(self):
        """Check if real dataset exists."""
        dataset_path = paths.get_dataset_path('cartpole_ood_v1')
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        return True
    
    @pytest.fixture
    def norm_stats_exists(self):
        """Check if norm_stats exists."""
        norm_path = paths.get_norm_stats_path('cartpole_ood_v1')
        if not norm_path.exists():
            pytest.skip("norm_stats.json not found")
        return True
    
    def test_full_pipeline(self, dataset_exists, norm_stats_exists):
        """Test complete pipeline execution."""
        config = Gate0Config(
            n_train=10,
            seed=0,
            note='pytest'
        )
        runner = Gate0Runner(config)
        
        result = runner.run()
        
        # Check success
        assert result['success'] == True
        assert result['run_id'] == runner.run_id
        
        # Check artifacts exist
        results_dir = Path(result['results_dir'])
        assert (results_dir / 'manifest.json').exists()
        assert (results_dir / 'metrics.json').exists()
        assert (results_dir / 'sindy_coefficients.csv').exists()
        
        # Check figures
        figures_dir = results_dir / 'figures'
        assert (figures_dir / 'F00_condition_distribution.png').exists()
        assert (figures_dir / 'F00_condition_distribution.pdf').exists()
        assert (figures_dir / 'F01_rollout_example.png').exists()
        assert (figures_dir / 'F02_coeff_heatmap.png').exists()
        
        # Check Context Packet
        cp_path = paths.get_context_packet_path(runner.run_id)
        assert cp_path.exists()
    
    def test_metrics_structure(self, dataset_exists, norm_stats_exists):
        """Test metrics.json structure."""
        config = Gate0Config(
            n_train=10,
            seed=0,
            note='pytest_metrics'
        )
        runner = Gate0Runner(config)
        result = runner.run()
        
        # Load metrics
        metrics_path = Path(result['results_dir']) / 'metrics.json'
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Check structure
        assert 'run_id' in metrics
        assert 'config' in metrics
        assert 'sparsity' in metrics
        assert 'splits' in metrics
        
        # Check splits
        for split in ['train', 'val', 'test']:
            assert split in metrics['splits']
            assert 'r2_per_target' in metrics['splits'][split]
            assert 'r2_mean' in metrics['splits'][split]
            assert 'rmse_per_target' in metrics['splits'][split]
            assert len(metrics['splits'][split]['r2_per_target']) == 4
    
    def test_manifest_structure(self, dataset_exists, norm_stats_exists):
        """Test manifest.json structure."""
        config = Gate0Config(
            n_train=10,
            seed=0,
            note='pytest_manifest'
        )
        runner = Gate0Runner(config)
        result = runner.run()
        
        # Load manifest
        manifest_path = Path(result['results_dir']) / 'manifest.json'
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Check required fields
        assert manifest['run_id'] == runner.run_id
        assert manifest['gate'] == 'gate0'
        assert manifest['track'] == 'standardized'
        assert manifest['method'] == 'sindy'
        
        # Check library info
        assert 'library' in manifest
        assert 'n_features' in manifest['library']
        assert 'feature_names' in manifest['library']
        
        # Check optimizer info
        assert 'optimizer' in manifest
        assert manifest['optimizer']['optimizer'] == 'STLSQ'
        assert 'sparsity' in manifest['optimizer']
    
    def test_coefficients_csv(self, dataset_exists, norm_stats_exists):
        """Test sindy_coefficients.csv format."""
        from src.sindy.optimizer import load_coefficients_csv
        
        config = Gate0Config(
            n_train=10,
            seed=0,
            note='pytest_csv'
        )
        runner = Gate0Runner(config)
        result = runner.run()
        
        # Load CSV
        csv_path = Path(result['results_dir']) / 'sindy_coefficients.csv'
        coeffs, feature_names, target_names = load_coefficients_csv(csv_path)
        
        # Check shape
        assert coeffs.shape[1] == 4  # 4 targets
        assert len(feature_names) == coeffs.shape[0]
        assert len(target_names) == 4
        
        # Check target names
        assert 'x_dot' in target_names
        assert 'theta_ddot' in target_names
    
    def test_different_seeds(self, dataset_exists, norm_stats_exists):
        """Test that different seeds give different results."""
        config0 = Gate0Config(n_train=10, seed=0, note='seed0')
        config1 = Gate0Config(n_train=10, seed=1, note='seed1')
        
        runner0 = Gate0Runner(config0)
        runner1 = Gate0Runner(config1)
        
        result0 = runner0.run()
        result1 = runner1.run()
        
        # Both should succeed
        assert result0['success']
        assert result1['success']
        
        # Run IDs should differ
        assert result0['run_id'] != result1['run_id']
    
    def test_author_recommended_track(self, dataset_exists, norm_stats_exists):
        """Test author_recommended track."""
        config = Gate0Config(
            n_train=10,
            seed=0,
            track='author_recommended',
            note='pytest_author'
        )
        runner = Gate0Runner(config)
        
        result = runner.run()
        
        assert result['success']
        assert runner.derivative_key == 'derivative_dx'


# =============================================================================
# Preflight Tests
# =============================================================================

class TestGate0Preflight:
    """Test preflight validation."""
    
    def test_preflight_fails_missing_dataset(self):
        """Should fail if dataset doesn't exist."""
        config = Gate0Config(
            dataset_version='nonexistent_v999',
            note='pytest_fail'
        )
        runner = Gate0Runner(config)
        
        with pytest.raises(FileNotFoundError):
            runner.run()


# =============================================================================
# Context Packet Tests
# =============================================================================

class TestContextPacket:
    """Test Context Packet generation."""
    
    @pytest.fixture
    def dataset_exists(self):
        dataset_path = paths.get_dataset_path('cartpole_ood_v1')
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        return True
    
    @pytest.fixture
    def norm_stats_exists(self):
        norm_path = paths.get_norm_stats_path('cartpole_ood_v1')
        if not norm_path.exists():
            pytest.skip("norm_stats.json not found")
        return True
    
    def test_context_packet_content(self, dataset_exists, norm_stats_exists):
        """Test Context Packet contains required info."""
        config = Gate0Config(
            n_train=10,
            seed=0,
            note='pytest_cp'
        )
        runner = Gate0Runner(config)
        runner.run()
        
        cp_path = paths.get_context_packet_path(runner.run_id)
        content = cp_path.read_text(encoding='utf-8')
        
        # Check required sections
        assert runner.run_id in content
        assert 'gate0' in content
        assert 'cartpole_ood_v1' in content
        assert 'R²' in content or 'R2' in content
        assert 'manifest.json' in content
        assert 'metrics.json' in content
        assert '다음 작업' in content or 'Next' in content


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def dataset_exists(self):
        dataset_path = paths.get_dataset_path('cartpole_ood_v1')
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        return True
    
    @pytest.fixture
    def norm_stats_exists(self):
        norm_path = paths.get_norm_stats_path('cartpole_ood_v1')
        if not norm_path.exists():
            pytest.skip("norm_stats.json not found")
        return True
    
    def test_n_train_exceeds_available(self, dataset_exists, norm_stats_exists):
        """Should handle n_train > available trajectories."""
        config = Gate0Config(
            n_train=1000,  # More than available
            seed=0,
            note='pytest_large_n'
        )
        runner = Gate0Runner(config)
        
        # Should not raise, just use all available
        result = runner.run()
        assert result['success']
    
    def test_zero_threshold(self, dataset_exists, norm_stats_exists):
        """Should handle threshold=0 (pure OLS)."""
        config = Gate0Config(
            n_train=10,
            seed=0,
            threshold=0.0,
            note='pytest_ols'
        )
        runner = Gate0Runner(config)
        result = runner.run()
        
        # Should succeed with no sparsity
        assert result['success']
        sparsity = result['metrics']['sparsity']
        assert sparsity['sparsity'] == 0.0  # No sparsity with threshold=0
    
    def test_standardized_track_requires_savgol(self):
        """Should fail-fast if standardized track but dx_savgol missing."""
        # Create a mock dataset without dx_savgol
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create minimal dataset WITHOUT dx_savgol
            N, T = 10, 50
            mock_data = {
                'train_x': np.random.randn(N, T, 4),
                'train_u': np.random.randn(N, T, 1),
                'train_dx': np.random.randn(N, T, 4),  # Only analytic, no savgol
                'train_params': np.random.randn(N, 2),
                'train_cond_id': np.arange(N),
                'val_x': np.random.randn(5, T, 4),
                'val_u': np.random.randn(5, T, 1),
                'val_dx': np.random.randn(5, T, 4),
                'val_params': np.random.randn(5, 2),
                'val_cond_id': np.arange(5),
                'test_x': np.random.randn(5, T, 4),
                'test_u': np.random.randn(5, T, 1),
                'test_dx': np.random.randn(5, T, 4),
                'test_params': np.random.randn(5, 2),
                'test_cond_id': np.arange(5),
                't': np.linspace(0, 1, T),
                'dt': 1.0 / (T - 1),
            }
            
            # Save mock dataset (paths.py uses ROOT/data/system/version)
            data_dir = tmpdir / 'data' / 'cartpole' / 'mock_v1'
            data_dir.mkdir(parents=True)
            np.savez(data_dir / 'dataset.npz', **mock_data)
            
            # Create mock norm_stats
            norm_stats = {
                'state': {'mean': [0]*4, 'std': [1]*4},
                'input': {'mean': [0], 'std': [1]},
                'derivative_dx': {'mean': [0]*4, 'std': [1]*4},
                # NOTE: No derivative_dx_savgol key
            }
            import json
            with open(data_dir / 'norm_stats.json', 'w') as f:
                json.dump(norm_stats, f)
            
            # Patch paths to use temp directory
            import os
            old_root = os.environ.get('PHD_PROJECT_ROOT')
            os.environ['PHD_PROJECT_ROOT'] = str(tmpdir)
            
            try:
                # Reload paths module to pick up new root
                import importlib
                from src.contracts import paths as paths_module
                importlib.reload(paths_module)
                
                config = Gate0Config(
                    dataset_version='mock_v1',
                    track='standardized',  # Requires dx_savgol
                    note='pytest_no_savgol'
                )
                runner = Gate0Runner(config)
                
                # Should raise ValueError about missing dx_savgol
                with pytest.raises(ValueError, match="dx_savgol"):
                    runner.run()
            finally:
                # Restore original root
                if old_root:
                    os.environ['PHD_PROJECT_ROOT'] = old_root
                elif 'PHD_PROJECT_ROOT' in os.environ:
                    del os.environ['PHD_PROJECT_ROOT']
                importlib.reload(paths_module)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])