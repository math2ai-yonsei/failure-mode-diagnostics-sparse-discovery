"""
Phase 3.5 테스트: StableCoreMiner 단위 테스트

실행 방법:
    pytest tests/test_core_mining.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# 프로젝트 루트를 PATH에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from sindy.core_mining import StableCoreMiner  # type: ignore  # noqa: E402  # pylint: disable=import-error


class TestStableCoreMiner:
    """StableCoreMiner 기본 기능 테스트"""
    
    def test_init_valid_params(self):
        """유효한 파라미터로 초기화"""
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0, eps=1e-12)
        assert miner.tau_hi == 0.5
        assert miner.z0 == 2.0
        assert miner.eps == 1e-12
    
    def test_init_invalid_tau_hi(self):
        """tau_hi 범위 벗어남"""
        with pytest.raises(ValueError, match="tau_hi"):
            StableCoreMiner(tau_hi=1.5)
        with pytest.raises(ValueError, match="tau_hi"):
            StableCoreMiner(tau_hi=-0.1)
    
    def test_init_invalid_z0(self):
        """z0 음수"""
        with pytest.raises(ValueError, match="z0"):
            StableCoreMiner(z0=-1.0)
    
    def test_init_invalid_eps(self):
        """eps 0 이하"""
        with pytest.raises(ValueError, match="eps"):
            StableCoreMiner(eps=0)
        with pytest.raises(ValueError, match="eps"):
            StableCoreMiner(eps=-1e-12)
    
    def test_compute_z_scores(self):
        """z-score 계산 테스트"""
        miner = StableCoreMiner()
        
        coef_mean = np.array([[10.0, -5.0], [0.0, 2.0]])
        coef_std = np.array([[1.0, 2.5], [0.0, 0.5]])  # std=0도 테스트
        
        z = miner.compute_z_scores(coef_mean, coef_std)
        
        # z = |mean| / (std + eps)
        assert z[0, 0] == pytest.approx(10.0, rel=1e-6)
        assert z[0, 1] == pytest.approx(2.0, rel=1e-6)  # 5.0 / 2.5
        assert z[1, 0] == pytest.approx(0.0, rel=1e-6)  # 0 / eps
        assert z[1, 1] == pytest.approx(4.0, rel=1e-6)  # 2.0 / 0.5


class TestCoreMining:
    """Core mining 로직 테스트"""
    
    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        coef_mean = np.array([
            [10.0, 0.5],   # feat_a: high z, low z
            [2.0, 0.1],    # feat_b: high z, inactive
            [0.8, 3.0]     # feat_c: inactive, high z
        ])
        coef_std = np.array([
            [1.0, 1.0],
            [0.5, 0.1],
            [1.0, 0.5]
        ])
        inc_prob = np.array([
            [1.0, 0.6],    # 둘 다 active
            [0.8, 0.3],    # 첫째만 active
            [0.4, 0.9]     # 둘째만 active
        ])
        feature_names = ['feat_a', 'feat_b', 'feat_c']
        target_names = ['target_1', 'target_2']
        
        return coef_mean, coef_std, inc_prob, feature_names, target_names
    
    def test_mine_counts(self, sample_data):
        """Stable-core / Fragile-pool 개수 테스트"""
        coef_mean, coef_std, inc_prob, feature_names, target_names = sample_data
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, feature_names, target_names)
        
        # 전체 6개 항
        assert result.n_total_terms == 6
        
        # active: (0,0), (0,1), (1,0), (2,1) = 4개
        assert result.n_active_terms == 4
        
        # z >= 2.0 AND active: (0,0)=10.0, (1,0)=4.0, (2,1)=6.0 = 3개
        assert result.n_stable_core == 3
        
        # z < 2.0 AND active: (0,1)=0.5 = 1개
        assert result.n_fragile_pool == 1
    
    def test_mine_masks(self, sample_data):
        """마스크 정확성 테스트"""
        coef_mean, coef_std, inc_prob, feature_names, target_names = sample_data
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, feature_names, target_names)
        
        # Stable-core mask
        expected_stable = np.array([
            [True, False],
            [True, False],
            [False, True]
        ])
        np.testing.assert_array_equal(result.stable_core_mask, expected_stable)
        
        # Fragile-pool mask
        expected_fragile = np.array([
            [False, True],
            [False, False],
            [False, False]
        ])
        np.testing.assert_array_equal(result.fragile_pool_mask, expected_fragile)
    
    def test_mine_term_details(self, sample_data):
        """상세 결과 내용 테스트"""
        coef_mean, coef_std, inc_prob, feature_names, target_names = sample_data
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, feature_names, target_names)
        
        # Stable-core는 z-score 내림차순 정렬
        assert len(result.stable_core_terms) == 3
        assert result.stable_core_terms[0]['feature'] == 'feat_a'
        assert result.stable_core_terms[0]['z_score'] == pytest.approx(10.0, rel=1e-6)
        
        # Fragile-pool
        assert len(result.fragile_pool_terms) == 1
        assert result.fragile_pool_terms[0]['feature'] == 'feat_a'
        assert result.fragile_pool_terms[0]['target'] == 'target_2'
    
    def test_shape_mismatch_errors(self, sample_data):
        """Shape 불일치 에러 테스트"""
        coef_mean, coef_std, inc_prob, feature_names, target_names = sample_data
        
        miner = StableCoreMiner()
        
        # coef_std shape 불일치
        bad_std = np.zeros((2, 2))
        with pytest.raises(ValueError, match="coef_std shape"):
            miner.mine(coef_mean, bad_std, inc_prob, feature_names, target_names)
        
        # inc_prob shape 불일치
        bad_prob = np.zeros((2, 3))
        with pytest.raises(ValueError, match="inc_prob shape"):
            miner.mine(coef_mean, coef_std, bad_prob, feature_names, target_names)
        
        # feature_names 길이 불일치
        with pytest.raises(ValueError, match="feature_names"):
            miner.mine(coef_mean, coef_std, inc_prob, ['a', 'b'], target_names)


class TestCoreMiningResult:
    """CoreMiningResult 기능 테스트"""
    
    @pytest.fixture
    def sample_result(self):
        """테스트용 CoreMiningResult"""
        coef_mean = np.array([[10.0, 0.5], [2.0, 3.0]])
        coef_std = np.array([[1.0, 1.0], [0.5, 0.5]])
        inc_prob = np.array([[1.0, 0.6], [0.8, 0.9]])
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        return miner.mine(coef_mean, coef_std, inc_prob, ['f1', 'f2'], ['t1', 't2'])
    
    def test_to_dict(self, sample_result):
        """to_dict() 테스트"""
        d = sample_result.to_dict()
        
        assert 'params' in d
        assert d['params']['tau_hi'] == 0.5
        assert d['params']['z0'] == 2.0
        
        assert 'summary' in d
        assert d['summary']['n_stable_core'] == sample_result.n_stable_core
        
        assert 'stable_core_terms' in d
        assert 'fragile_pool_terms' in d
    
    def test_save_json(self, sample_result, tmp_path):
        """save_json() 테스트"""
        json_path = tmp_path / "test_result.json"
        sample_result.save_json(json_path)
        
        assert json_path.exists()
        
        with open(json_path, encoding='utf-8') as f:
            loaded = json.load(f)
        
        assert loaded['summary']['n_stable_core'] == sample_result.n_stable_core
    
    def test_get_stable_core_coef_mean(self, sample_result):
        """get_stable_core_coef_mean() 테스트"""
        masked = sample_result.get_stable_core_coef_mean()
        
        # Stable-core 위치만 값 유지
        assert masked.shape == sample_result.coef_mean.shape
        
        # Stable-core가 아닌 위치는 0
        for i in range(masked.shape[0]):
            for j in range(masked.shape[1]):
                if not sample_result.stable_core_mask[i, j]:
                    assert masked[i, j] == 0.0


class TestEdgeCases:
    """경계 조건 테스트"""
    
    def test_all_stable(self):
        """모든 항이 Stable-core인 경우"""
        coef_mean = np.array([[10.0, 20.0]])
        coef_std = np.array([[1.0, 1.0]])
        inc_prob = np.array([[1.0, 1.0]])
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, ['f'], ['t1', 't2'])
        
        assert result.n_stable_core == 2
        assert result.n_fragile_pool == 0
    
    def test_all_fragile(self):
        """모든 active 항이 Fragile-pool인 경우"""
        coef_mean = np.array([[0.1, 0.2]])
        coef_std = np.array([[1.0, 1.0]])
        inc_prob = np.array([[0.6, 0.7]])
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, ['f'], ['t1', 't2'])
        
        assert result.n_stable_core == 0
        assert result.n_fragile_pool == 2
    
    def test_no_active(self):
        """active 항이 없는 경우"""
        coef_mean = np.array([[10.0, 20.0]])
        coef_std = np.array([[1.0, 1.0]])
        inc_prob = np.array([[0.3, 0.4]])  # 모두 tau_hi 미만
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, ['f'], ['t1', 't2'])
        
        assert result.n_active_terms == 0
        assert result.n_stable_core == 0
        assert result.n_fragile_pool == 0
    
    def test_boundary_z_score(self):
        """z-score가 z0보다 약간 큰 경우 (eps 고려)"""
        # eps=1e-12이므로 z = |mean| / (std + eps)
        # z >= 2.0을 만족하려면 |mean| / (std + 1e-12) >= 2.0
        # mean = 2.0 + 1e-11으로 설정하여 확실히 z >= 2.0
        coef_mean = np.array([[2.0 + 1e-11]])
        coef_std = np.array([[1.0]])
        inc_prob = np.array([[1.0]])
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, ['f'], ['t'])
        
        # z >= z0 이므로 Stable-core
        assert result.n_stable_core == 1
        assert result.n_fragile_pool == 0
    
    def test_boundary_z_score_below(self):
        """z-score가 z0보다 약간 작은 경우"""
        # z = 2.0 / (1.0 + 1e-12) ≈ 1.999999... < 2.0
        coef_mean = np.array([[2.0]])
        coef_std = np.array([[1.0]])  
        inc_prob = np.array([[1.0]])
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, ['f'], ['t'])
        
        # eps로 인해 z < 2.0이므로 Fragile-pool
        assert result.n_stable_core == 0
        assert result.n_fragile_pool == 1
    
    def test_boundary_inc_prob(self):
        """inc_prob이 정확히 tau_hi인 경우"""
        coef_mean = np.array([[10.0]])
        coef_std = np.array([[1.0]])
        inc_prob = np.array([[0.5]])  # 정확히 tau_hi
        
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, ['f'], ['t'])
        
        # inc_prob >= tau_hi 이므로 active
        assert result.n_active_terms == 1
        assert result.n_stable_core == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])