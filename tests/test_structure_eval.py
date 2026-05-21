"""
Phase 3.5 테스트: StructureEvaluator 단위 테스트

실행 방법:
    pytest tests/test_structure_eval.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# 프로젝트 루트를 PATH에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from evaluation.structure_eval import (  # type: ignore  # noqa: E402  # pylint: disable=import-error
    StructureEvaluator,
    StructureEvalResult,
    compute_support_metrics
)


class TestStructureEvaluator:
    """StructureEvaluator 기본 기능 테스트"""
    
    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        # Oracle: 4개 항
        oracle_support = np.array([
            [True,  False, True ],
            [True,  False, False],
            [False, True,  False]
        ])
        
        # Teacher: 6개 항 (4 both + 2 teacher-only)
        teacher_support = np.array([
            [True,  True,  True ],  # [0,1] teacher-only
            [True,  False, True ],  # [1,2] teacher-only
            [False, True,  False]
        ])
        
        # Final: 5개 항
        final_support = np.array([
            [True,  False, True ],  # [0,1] 제거됨
            [True,  False, True ],  # [1,2] 재유입
            [False, True,  False]
        ])
        
        feature_names = ['f0', 'f1', 'f2']
        target_names = ['t0', 't1', 't2']
        
        return {
            'oracle': oracle_support,
            'teacher': teacher_support,
            'final': final_support,
            'features': feature_names,
            'targets': target_names
        }
    
    def test_init(self, sample_data):
        """초기화 테스트"""
        evaluator = StructureEvaluator(
            sample_data['oracle'],
            sample_data['features'],
            sample_data['targets']
        )
        assert evaluator.n_features == 3
        assert evaluator.n_targets == 3
    
    def test_teacher_only_mask(self, sample_data):
        """Teacher-only 마스크 테스트"""
        evaluator = StructureEvaluator(
            sample_data['oracle'],
            sample_data['features'],
            sample_data['targets']
        )
        
        teacher_only = evaluator.compute_teacher_only_mask(sample_data['teacher'])
        
        # Teacher-only: [0,1], [1,2]
        expected = np.array([
            [False, True,  False],
            [False, False, True ],
            [False, False, False]
        ])
        np.testing.assert_array_equal(teacher_only, expected)
        assert teacher_only.sum() == 2


class TestSpuriousReduction:
    """Spurious Reduction 테스트"""
    
    @pytest.fixture
    def evaluator(self):
        oracle = np.array([
            [True, False],
            [False, True]
        ])
        return StructureEvaluator(oracle, ['f0', 'f1'], ['t0', 't1'])
    
    def test_full_reduction(self, evaluator):
        """모든 Teacher-only 제거"""
        teacher = np.array([
            [True, True],   # [0,1] teacher-only
            [True, True]    # [1,0] teacher-only
        ])
        final = np.array([
            [True, False],  # teacher-only 제거
            [False, True]   # teacher-only 제거
        ])
        
        rate, n_removed, n_total, _ = evaluator.compute_spurious_reduction(
            teacher, final
        )
        
        assert n_total == 2
        assert n_removed == 2
        assert rate == 1.0
    
    def test_partial_reduction(self, evaluator):
        """일부 Teacher-only 제거"""
        teacher = np.array([
            [True, True],
            [True, True]
        ])
        final = np.array([
            [True, True],   # [0,1] 유지 (재유입)
            [False, True]   # [1,0] 제거
        ])
        
        rate, n_removed, n_total, _ = evaluator.compute_spurious_reduction(
            teacher, final
        )
        
        assert n_total == 2
        assert n_removed == 1
        assert rate == 0.5
    
    def test_no_teacher_only(self, evaluator):
        """Teacher-only가 없는 경우"""
        teacher = np.array([
            [True, False],
            [False, True]
        ])
        final = teacher.copy()
        
        rate, _, n_total, _ = evaluator.compute_spurious_reduction(
            teacher, final
        )
        
        assert n_total == 0
        assert rate == 1.0  # 0개 중 0개 제거 = 100%


class TestSpuriousRetention:
    """Spurious Retention (잔존율) 테스트"""
    
    @pytest.fixture
    def evaluator(self):
        oracle = np.array([[True, False]])
        return StructureEvaluator(oracle, ['f'], ['t0', 't1'])
    
    def test_no_retention(self, evaluator):
        """잔존 없음 (모두 제거)"""
        teacher = np.array([[True, True]])  # [0,1] teacher-only
        final = np.array([[True, False]])   # 제거됨
        
        rate, n_retained, _, _ = evaluator.compute_spurious_retention(
            teacher, final
        )
        
        assert n_retained == 0
        assert rate == 0.0
    
    def test_full_retention(self, evaluator):
        """모든 spurious 잔존"""
        teacher = np.array([[True, True]])
        final = np.array([[True, True]])  # 유지 (잔존)
        
        rate, n_retained, _, _ = evaluator.compute_spurious_retention(
            teacher, final
        )
        
        assert n_retained == 1
        assert rate == 1.0


class TestPromotionRate:
    """Promotion Rate 테스트"""
    
    @pytest.fixture
    def evaluator(self):
        oracle = np.array([
            [True, True],
            [True, False]
        ])
        return StructureEvaluator(oracle, ['f0', 'f1'], ['t0', 't1'])
    
    def test_promotion_success(self, evaluator):
        """승급 성공"""
        # Fragile-pool: oracle-true이면서 z < 2.0
        fragile = np.array([
            [True, True],   # [0,0], [0,1] fragile
            [False, False]
        ])
        
        z_before = np.array([
            [1.5, 1.0],  # < 2.0
            [3.0, 0.5]
        ])
        
        z_after = np.array([
            [2.5, 1.5],  # [0,0] 승급, [0,1] 미승급
            [3.5, 0.6]
        ])
        
        rate, n_promoted, n_fragile_oracle, _ = evaluator.compute_promotion_rate(
            fragile, z_before, z_after, z0=2.0
        )
        
        # Fragile oracle-true: [0,0], [0,1]
        assert n_fragile_oracle == 2
        # 승급: [0,0]만
        assert n_promoted == 1
        assert rate == 0.5
    
    def test_no_fragile_oracle_true(self, evaluator):
        """Fragile oracle-true가 없는 경우"""
        fragile = np.array([
            [False, False],
            [False, True]  # oracle-false이므로 제외
        ])
        
        z_before = np.zeros((2, 2))
        z_after = np.ones((2, 2)) * 3.0
        
        rate, _, n_fragile_oracle, _ = evaluator.compute_promotion_rate(
            fragile, z_before, z_after
        )
        
        assert n_fragile_oracle == 0
        assert rate == 0.0


class TestDeltaZ:
    """Δz 계산 테스트"""
    
    def test_delta_z_oracle_true(self):
        oracle = np.array([
            [True, False],
            [True, True]
        ])
        evaluator = StructureEvaluator(oracle, ['f0', 'f1'], ['t0', 't1'])
        
        z_before = np.array([
            [1.0, 0.5],
            [2.0, 3.0]
        ])
        z_after = np.array([
            [2.0, 0.6],  # Δ=+1.0, Δ=+0.1
            [2.5, 4.0]   # Δ=+0.5, Δ=+1.0
        ])
        
        median, mean, details = evaluator.compute_delta_z(z_before, z_after)
        
        # Oracle-true: [0,0], [1,0], [1,1]
        # Δz: +1.0, +0.5, +1.0
        assert len(details) == 3
        assert median == pytest.approx(1.0)
        assert mean == pytest.approx((1.0 + 0.5 + 1.0) / 3)


class TestDeltaFloor:
    """Δfloor 계산 테스트"""
    
    def test_positive_delta_floor(self):
        proposed = [0.95, 0.93, 0.94]
        random = [0.85, 0.80, 0.82]
        
        result = StructureEvaluator.compute_delta_floor(proposed, random)
        
        assert result.proposed_mean > result.random_mean
        assert result.delta_floor > 0
        assert result.significant
    
    def test_negative_delta_floor(self):
        proposed = [0.80, 0.82, 0.81]
        random = [0.90, 0.92, 0.91]
        
        result = StructureEvaluator.compute_delta_floor(proposed, random)
        
        assert result.delta_floor < 0
        assert not result.significant
    
    def test_floor_calculation(self):
        # 정확한 계산 검증
        proposed = [0.9, 0.9, 0.9]  # mean=0.9, std=0
        random = [0.8, 0.8, 0.8]    # mean=0.8, std=0
        
        result = StructureEvaluator.compute_delta_floor(proposed, random)
        
        # floor = mean - 1*std = mean (std=0)
        assert result.proposed_floor == pytest.approx(0.9)
        assert result.random_floor == pytest.approx(0.8)
        assert result.delta_floor == pytest.approx(0.1)


class TestSupportMetrics:
    """Support metrics 테스트"""
    
    def test_perfect_match(self):
        oracle = np.array([[True, False], [False, True]])
        pred = oracle.copy()
        
        metrics = compute_support_metrics(oracle, pred)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['jaccard'] == 1.0
    
    def test_no_overlap(self):
        oracle = np.array([[True, False]])
        pred = np.array([[False, True]])
        
        metrics = compute_support_metrics(oracle, pred)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['jaccard'] == 0.0
    
    def test_partial_overlap(self):
        oracle = np.array([[True, True, False]])
        pred = np.array([[True, False, True]])
        
        # TP=1, FP=1, FN=1
        metrics = compute_support_metrics(oracle, pred)
        
        assert metrics['n_true_positive'] == 1
        assert metrics['n_false_positive'] == 1
        assert metrics['n_false_negative'] == 1
        assert metrics['precision'] == pytest.approx(0.5)
        assert metrics['recall'] == pytest.approx(0.5)


class TestStructureEvalResult:
    """StructureEvalResult 테스트"""
    
    def test_to_dict(self):
        result = StructureEvalResult(
            spurious_reduction=0.8,
            spurious_retention=0.1,
            promotion_rate=0.5,
            delta_z_median=1.5,
            delta_z_mean=1.2,
            n_teacher_only_total=10,
            n_teacher_only_removed=8,
            n_spurious_retained=1,
            n_fragile_oracle_true=2,
            n_promoted=1
        )
        
        d = result.to_dict()
        
        assert d['primary_metrics']['spurious_reduction'] == 0.8
        assert d['counts']['n_teacher_only_total'] == 10
    
    def test_summary(self):
        result = StructureEvalResult(
            spurious_reduction=0.8,
            spurious_retention=0.1,
            promotion_rate=0.5,
            delta_z_median=1.5,
            delta_z_mean=1.2,
            n_teacher_only_total=10,
            n_teacher_only_removed=8,
            n_spurious_retained=1,
            n_fragile_oracle_true=2,
            n_promoted=1
        )
        
        summary = result.summary()
        
        assert "80.0%" in summary
        assert "10.0%" in summary
        assert "50.0%" in summary
    
    def test_save_json(self, tmp_path):
        result = StructureEvalResult(
            spurious_reduction=0.8,
            spurious_retention=0.1,
            promotion_rate=0.5,
            delta_z_median=1.5,
            delta_z_mean=1.2,
            n_teacher_only_total=10,
            n_teacher_only_removed=8,
            n_spurious_retained=1,
            n_fragile_oracle_true=2,
            n_promoted=1
        )
        
        json_path = tmp_path / "test.json"
        result.save_json(json_path)
        
        assert json_path.exists()
        
        with open(json_path, encoding='utf-8') as f:
            loaded = json.load(f)
        
        assert loaded['primary_metrics']['spurious_reduction'] == 0.8


class TestFullEvaluation:
    """전체 평가 통합 테스트"""
    
    def test_evaluate_integration(self):
        """evaluate() 통합 테스트"""
        oracle = np.array([
            [True, False, True],
            [True, True, False]
        ])
        
        teacher = np.array([
            [True, True, True],   # [0,1] teacher-only
            [True, True, True]    # [1,2] teacher-only
        ])
        
        final = np.array([
            [True, False, True],  # [0,1] 제거
            [True, True, True]    # [1,2] 잔존
        ])
        
        z_before = np.array([
            [10.0, 1.0, 5.0],
            [3.0, 8.0, 1.5]
        ])
        
        z_after = np.array([
            [12.0, 0.5, 6.0],
            [4.0, 9.0, 1.8]
        ])
        
        fragile = teacher & (z_before < 2.0)
        
        evaluator = StructureEvaluator(
            oracle, ['f0', 'f1'], ['t0', 't1', 't2']
        )
        
        result = evaluator.evaluate(
            teacher_support=teacher,
            final_support=final,
            z_before=z_before,
            z_after=z_after,
            fragile_pool_mask=fragile,
            z0=2.0
        )
        
        # Teacher-only 2개 중 1개 제거
        assert result.n_teacher_only_total == 2
        assert result.n_teacher_only_removed == 1
        assert result.spurious_reduction == 0.5
        
        # 1개 잔존
        assert result.n_spurious_retained == 1
        assert result.spurious_retention == 0.5
        
        # Re-entry는 선택적 (selection mask 없이 호출)
        assert result.spurious_reentry is None
    
    def test_evaluate_with_selection(self):
        """evaluate() with selection support 테스트"""
        oracle = np.array([
            [True, False, True],
            [True, True, False]
        ])
        
        teacher = np.array([
            [True, True, True],   # [0,1] teacher-only
            [True, True, True]    # [1,2] teacher-only
        ])
        
        # Selection: teacher-only 중 [0,1]만 제거, [1,2]는 유지
        selected = np.array([
            [True, False, True],  # [0,1] 제거
            [True, True, True]    # [1,2] 유지
        ])
        
        # Final: [0,1] 재유입, [1,2] 유지
        final = np.array([
            [True, True, True],   # [0,1] 재유입!
            [True, True, True]    # [1,2] 유지
        ])
        
        z_before = np.array([
            [10.0, 1.0, 5.0],
            [3.0, 8.0, 1.5]
        ])
        
        z_after = np.array([
            [12.0, 2.5, 6.0],
            [4.0, 9.0, 1.8]
        ])
        
        fragile = teacher & (z_before < 2.0)
        
        evaluator = StructureEvaluator(
            oracle, ['f0', 'f1'], ['t0', 't1', 't2']
        )
        
        result = evaluator.evaluate(
            teacher_support=teacher,
            final_support=final,
            z_before=z_before,
            z_after=z_after,
            fragile_pool_mask=fragile,
            z0=2.0,
            selected_support_pre_aug=selected  # Selection mask 제공
        )
        
        # Re-entry 계산됨
        assert result.spurious_reentry is not None
        
        # teacher-only 2개, selection에서 1개 제거 ([0,1])
        # 그 중 [0,1]이 final에 재유입
        assert result.n_removed_at_selection == 1
        assert result.n_spurious_reentered == 1
        assert result.spurious_reentry == 1.0  # 100% 재유입


class TestSpuriousReentry:
    """Spurious Re-entry 테스트"""
    
    @pytest.fixture
    def evaluator(self):
        oracle = np.array([[True, False, True]])
        return StructureEvaluator(oracle, ['f'], ['t0', 't1', 't2'])
    
    def test_no_reentry(self, evaluator):
        """재유입 없음"""
        teacher = np.array([[True, True, True]])   # [0,1] teacher-only
        selected = np.array([[True, False, True]]) # [0,1] 제거
        final = np.array([[True, False, True]])    # [0,1] 재유입 안 됨
        
        rate, n_reentered, n_removed, _ = evaluator.compute_spurious_reentry(
            teacher, selected, final
        )
        
        assert n_removed == 1  # selection에서 1개 제거
        assert n_reentered == 0
        assert rate == 0.0
    
    def test_full_reentry(self, evaluator):
        """100% 재유입"""
        teacher = np.array([[True, True, True]])   # [0,1] teacher-only
        selected = np.array([[True, False, True]]) # [0,1] 제거
        final = np.array([[True, True, True]])     # [0,1] 재유입!
        
        rate, n_reentered, n_removed, _ = evaluator.compute_spurious_reentry(
            teacher, selected, final
        )
        
        assert n_removed == 1
        assert n_reentered == 1
        assert rate == 1.0
    
    def test_no_removal_at_selection(self, evaluator):
        """Selection에서 제거된 것이 없음"""
        teacher = np.array([[True, True, True]])   # [0,1] teacher-only
        selected = np.array([[True, True, True]])  # 전부 유지
        final = np.array([[True, True, True]])
        
        rate, n_reentered, n_removed, _ = evaluator.compute_spurious_reentry(
            teacher, selected, final
        )
        
        assert n_removed == 0
        assert n_reentered == 0
        assert rate == 0.0  # 분모가 0이면 0% 반환


class TestSelectionSubsetValidation:
    """Selection ⊆ Teacher 검증 테스트"""
    
    def test_invalid_selection_raises_error(self):
        """selected가 teacher의 부분집합이 아니면 에러"""
        oracle = np.array([
            [True, False, True],
            [True, True, False]
        ])
        
        teacher = np.array([
            [True, False, True],   # teacher에 [0,1] 없음
            [True, True, False]
        ])
        
        # selected에 teacher에 없는 항 [0,1] 포함
        selected = np.array([
            [True, True, True],    # [0,1]은 teacher에 없음!
            [True, True, False]
        ])
        
        final = np.array([
            [True, True, True],
            [True, True, False]
        ])
        
        z_before = np.ones((2, 3)) * 5.0
        z_after = np.ones((2, 3)) * 6.0
        fragile = np.zeros((2, 3), dtype=bool)
        
        evaluator = StructureEvaluator(
            oracle, ['f0', 'f1'], ['t0', 't1', 't2']
        )
        
        with pytest.raises(ValueError, match="not in teacher_support"):
            evaluator.evaluate(
                teacher_support=teacher,
                final_support=final,
                z_before=z_before,
                z_after=z_after,
                fragile_pool_mask=fragile,
                selected_support_pre_aug=selected
            )
    
    def test_valid_selection_passes(self):
        """selected가 teacher의 부분집합이면 통과"""
        oracle = np.array([[True, False, True]])
        teacher = np.array([[True, True, True]])
        selected = np.array([[True, False, True]])  # teacher의 부분집합
        final = np.array([[True, False, True]])
        
        z_before = np.ones((1, 3)) * 5.0
        z_after = np.ones((1, 3)) * 6.0
        fragile = np.zeros((1, 3), dtype=bool)
        
        evaluator = StructureEvaluator(oracle, ['f'], ['t0', 't1', 't2'])
        
        # 에러 없이 통과해야 함
        result = evaluator.evaluate(
            teacher_support=teacher,
            final_support=final,
            z_before=z_before,
            z_after=z_after,
            fragile_pool_mask=fragile,
            selected_support_pre_aug=selected
        )
        
        assert result.spurious_reentry is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])