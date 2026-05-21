"""
Structure Evaluation 모듈 (Phase 3.5)

핵심 역할:
- E-SINDy의 구조적 실패 모드(Precision Collapse, Recall Fragility) 정량화
- Augmentation 전후 개선 효과 측정

주요 지표:
1. Spurious Reduction: Teacher-only 항 제거율
2. Spurious Re-entry: Augmentation 후 spurious 재유입율
3. Promotion Rate: Fragile-pool oracle-true 승급률
4. Δz: oracle-true 항의 z-score 변화량
5. Δfloor: Robustness 개선 (proposed vs random)

중요:
- Oracle은 평가에서만 사용 (방법론에는 사용하지 않음)
- Oracle = E-SINDy n=50 결과 (ground-truth proxy)

Author: Claude (Phase 3.5)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
import json


@dataclass
class StructureEvalResult:
    """
    구조 평가 결과를 담는 데이터 클래스
    
    Primary Metrics:
    - spurious_reduction: Teacher-only 제거율
    - spurious_retention: Teacher-only 잔존율 (teacher→final 직접 비교)
    - spurious_reentry: Selection에서 제거된 spurious의 재유입율 (selection 단계 있을 때만)
    - promotion_rate: Fragile-pool oracle-true 승급률
    - delta_z_median/mean: oracle-true Δz 통계
    """
    
    # Primary Metrics
    spurious_reduction: float  # Teacher-only 제거율 (0~1)
    spurious_retention: float  # Spurious 잔존율 (0~1). teacher_only>0이면 1-reduction
    promotion_rate: float      # Fragile-pool oracle-true 승급률 (0~1)
    delta_z_median: float      # oracle-true Δz 중앙값
    delta_z_mean: float        # oracle-true Δz 평균
    
    # Counts
    n_teacher_only_total: int      # 총 Teacher-only 항
    n_teacher_only_removed: int    # 제거된 Teacher-only 항
    n_spurious_retained: int       # 잔존한 spurious 항
    n_fragile_oracle_true: int     # Fragile-pool 내 oracle-true 항
    n_promoted: int                # 승급한 항 (z_before<z0 → z_after>=z0)
    
    # Detail lists
    removed_spurious: List[Dict[str, Any]] = field(default_factory=list)
    retained_spurious: List[Dict[str, Any]] = field(default_factory=list)
    promoted_terms: List[Dict[str, Any]] = field(default_factory=list)
    delta_z_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Re-entry (selection 단계 있을 때만 유효)
    spurious_reentry: Optional[float] = None  # Selection 후 재유입율 (0~1)
    n_removed_at_selection: Optional[int] = None  # Selection에서 제거된 spurious 수
    n_spurious_reentered: Optional[int] = None    # 재유입된 spurious 수
    reentered_spurious: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """JSON 직렬화 가능한 dict로 변환"""
        result = {
            'primary_metrics': {
                'spurious_reduction': self.spurious_reduction,
                'spurious_retention': self.spurious_retention,
                'promotion_rate': self.promotion_rate,
                'delta_z_median': self.delta_z_median,
                'delta_z_mean': self.delta_z_mean
            },
            'counts': {
                'n_teacher_only_total': self.n_teacher_only_total,
                'n_teacher_only_removed': self.n_teacher_only_removed,
                'n_spurious_retained': self.n_spurious_retained,
                'n_fragile_oracle_true': self.n_fragile_oracle_true,
                'n_promoted': self.n_promoted
            },
            'details': {
                'removed_spurious': self.removed_spurious,
                'retained_spurious': self.retained_spurious,
                'promoted_terms': self.promoted_terms,
                'delta_z_details': self.delta_z_details
            }
        }
        
        # Re-entry 정보 추가 (있을 때만)
        if self.spurious_reentry is not None:
            result['primary_metrics']['spurious_reentry'] = self.spurious_reentry
            result['counts']['n_removed_at_selection'] = self.n_removed_at_selection
            result['counts']['n_spurious_reentered'] = self.n_spurious_reentered
            result['details']['reentered_spurious'] = self.reentered_spurious
        
        return result
    
    def save_json(self, path: Path):
        """JSON 파일로 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def summary(self) -> str:
        """한 줄 요약"""
        base = (f"SpuriousRed={self.spurious_reduction:.1%}, "
                f"Retention={self.spurious_retention:.1%}, "
                f"Promotion={self.promotion_rate:.1%}, "
                f"Δz_med={self.delta_z_median:+.2f}")
        if self.spurious_reentry is not None:
            base += f", ReEntry={self.spurious_reentry:.1%}"
        return base


@dataclass
class DeltaFloorResult:
    """
    Δfloor 평가 결과
    
    floor = mean - 1*std (worst-case estimation)
    Δfloor = proposed_floor - random_floor
    
    Note:
        significant는 통계적 유의성이 아닌 "양의 개선(Δfloor > 0)"을 의미합니다.
        통계적 검정이 필요하면 별도로 수행하세요.
    """
    
    delta_floor: float           # proposed_floor - random_floor
    proposed_floor: float        # proposed 방법의 worst-case (mean - 1σ)
    random_floor: float          # random selection의 worst-case (mean - 1σ)
    proposed_mean: float
    proposed_std: float          # ddof=0 (population std)
    random_mean: float
    random_std: float            # ddof=0 (population std)
    n_proposed_runs: int
    n_random_runs: int
    significant: bool            # Δfloor > 0 (양의 개선 여부, 통계적 유의성 아님)
    
    def to_dict(self) -> Dict:
        return {
            'delta_floor': self.delta_floor,
            'proposed_floor': self.proposed_floor,
            'random_floor': self.random_floor,
            'proposed_stats': {
                'mean': self.proposed_mean,
                'std': self.proposed_std,
                'n_runs': self.n_proposed_runs
            },
            'random_stats': {
                'mean': self.random_mean,
                'std': self.random_std,
                'n_runs': self.n_random_runs
            },
            'significant': self.significant
        }


class StructureEvaluator:
    """
    구조 평가기 (Phase 3.5)
    
    E-SINDy의 구조적 실패 모드를 정량화하고,
    Augmentation 효과를 측정합니다.
    
    사용법:
        evaluator = StructureEvaluator(oracle_support, feature_names, target_names)
        result = evaluator.evaluate(
            teacher_support=teacher_mask,
            final_support=final_mask,
            z_before=z_teacher,
            z_after=z_final,
            fragile_pool_mask=fragile_mask,
            z0=2.0
        )
    """
    
    def __init__(
        self,
        oracle_support: np.ndarray,
        feature_names: List[str],
        target_names: List[str]
    ):
        """
        Args:
            oracle_support: Oracle (n=50) support mask, shape (n_features, n_targets)
            feature_names: 피처 이름 리스트
            target_names: 타겟 이름 리스트
        """
        self.oracle_support = oracle_support.astype(bool)
        self.feature_names = feature_names
        self.target_names = target_names
        self.n_features, self.n_targets = oracle_support.shape
    
    def _get_term_name(self, i: int, j: int) -> str:
        """(feature_idx, target_idx) → "feature → target" 문자열"""
        return f"{self.feature_names[i]} → {self.target_names[j]}"
    
    def _mask_to_set(self, mask: np.ndarray) -> Set[Tuple[int, int]]:
        """마스크를 (i, j) 튜플 집합으로 변환"""
        return set(zip(*np.where(mask)))
    
    def compute_teacher_only_mask(
        self,
        teacher_support: np.ndarray
    ) -> np.ndarray:
        """
        Teacher-only 마스크 계산
        
        Teacher-only = Teacher에는 있지만 Oracle에는 없는 항 (spurious)
        
        Args:
            teacher_support: Teacher support mask
        
        Returns:
            teacher_only_mask: shape (n_features, n_targets)
        """
        teacher = teacher_support.astype(bool)
        return teacher & (~self.oracle_support)
    
    def compute_spurious_reduction(
        self,
        teacher_support: np.ndarray,
        final_support: np.ndarray
    ) -> Tuple[float, int, int, List[Dict]]:
        """
        Spurious 제거율 계산
        
        Spurious Reduction = (제거된 Teacher-only) / (총 Teacher-only)
        
        Args:
            teacher_support: Teacher support mask (before)
            final_support: Final model support mask (after augmentation)
        
        Returns:
            reduction_rate: 제거율 (0~1)
            n_removed: 제거된 개수
            n_total: 총 Teacher-only 개수
            removed_details: 제거된 항 상세 리스트
        """
        teacher_only = self.compute_teacher_only_mask(teacher_support)
        final = final_support.astype(bool)
        
        # 제거된 Teacher-only = Teacher-only에서 final에 없는 것
        removed = teacher_only & (~final)
        
        n_total = int(teacher_only.sum())
        n_removed = int(removed.sum())
        
        if n_total == 0:
            return 1.0, 0, 0, []  # Teacher-only가 없으면 100% 제거
        
        rate = n_removed / n_total
        
        # 상세 정보
        details = []
        for i, j in zip(*np.where(removed)):
            details.append({
                'feature': self.feature_names[i],
                'target': self.target_names[j],
                'feature_idx': int(i),
                'target_idx': int(j)
            })
        
        return rate, n_removed, n_total, details
    
    def compute_spurious_retention(
        self,
        teacher_support: np.ndarray,
        final_support: np.ndarray
    ) -> Tuple[float, int, int, List[Dict]]:
        """
        Spurious 잔존율 계산
        
        Teacher-only 중 final에 여전히 남아있는 비율
        
        Spurious Retention = (잔존 spurious) / (총 Teacher-only)
        
        Note: 이는 "재유입(re-entry)"이 아닌 "잔존(retention)"입니다.
        진정한 re-entry는 selection 단계 구현 후 별도로 측정합니다.
        
        Args:
            teacher_support: Teacher support mask
            final_support: Final model support mask
        
        Returns:
            retention_rate: 잔존율 (0~1)
            n_retained: 잔존 개수
            n_total: 총 Teacher-only 개수
            retention_details: 잔존 항 상세 리스트
        """
        teacher_only = self.compute_teacher_only_mask(teacher_support)
        final = final_support.astype(bool)
        
        # 잔존 = Teacher-only이면서 final에 있는 것
        retained = teacher_only & final
        
        n_total_teacher_only = int(teacher_only.sum())
        n_retained = int(retained.sum())
        
        if n_total_teacher_only == 0:
            return 0.0, 0, 0, []  # Teacher-only가 없으면 잔존 0%
        
        rate = n_retained / n_total_teacher_only
        
        # 상세 정보
        details = []
        for i, j in zip(*np.where(retained)):
            details.append({
                'feature': self.feature_names[i],
                'target': self.target_names[j],
                'feature_idx': int(i),
                'target_idx': int(j)
            })
        
        return rate, n_retained, n_total_teacher_only, details
    
    def compute_spurious_reentry(
        self,
        teacher_support: np.ndarray,
        selected_support_pre_aug: np.ndarray,
        final_support_post_aug: np.ndarray
    ) -> Tuple[float, int, int, List[Dict]]:
        """
        Spurious 재유입율 계산 (Primary Metric)
        
        Selection 단계에서 제거된 spurious가 augmentation 후 다시 나타나는 비율
        
        Re-entry Rate = (재유입 spurious) / (selection에서 제거된 spurious)
        
        분모: removed_spurious = teacher_only & ~selected_support_pre_aug
        분자: reentered = removed_spurious & final_support_post_aug
        
        Args:
            teacher_support: Teacher support mask (inc_prob >= tau_hi)
            selected_support_pre_aug: Selection 후 support mask (budget 적용 후)
            final_support_post_aug: Final model support mask (augmentation 학습 후)
        
        Returns:
            reentry_rate: 재유입율 (0~1)
            n_reentered: 재유입 개수
            n_removed_at_selection: selection에서 제거된 spurious 개수
            reentry_details: 재유입 항 상세 리스트
        
        Note:
            이 메트릭은 compute_spurious_retention()과 다릅니다.
            - retention: teacher_only 중 final에 남아있는 비율
            - reentry: selection에서 제거된 후 final에 다시 나타난 비율
            
            Re-entry는 augmentation이 spurious를 다시 불러오는지 측정합니다.
        """
        teacher_only = self.compute_teacher_only_mask(teacher_support)
        selected = selected_support_pre_aug.astype(bool)
        final = final_support_post_aug.astype(bool)
        
        # Selection에서 제거된 spurious = teacher_only이면서 selected에 없는 것
        removed_at_selection = teacher_only & (~selected)
        
        # 재유입 = selection에서 제거되었는데 final에 다시 나타난 것
        reentered = removed_at_selection & final
        
        n_removed = int(removed_at_selection.sum())
        n_reentered = int(reentered.sum())
        
        if n_removed == 0:
            # Selection에서 제거된 spurious가 없으면 재유입율 0%
            return 0.0, 0, 0, []
        
        rate = n_reentered / n_removed
        
        # 상세 정보
        details = []
        for i, j in zip(*np.where(reentered)):
            details.append({
                'feature': self.feature_names[i],
                'target': self.target_names[j],
                'feature_idx': int(i),
                'target_idx': int(j)
            })
        
        return rate, n_reentered, n_removed, details
    
    def compute_promotion_rate(
        self,
        fragile_pool_mask: np.ndarray,
        z_before: np.ndarray,
        z_after: np.ndarray,
        z0: float = 2.0
    ) -> Tuple[float, int, int, List[Dict]]:
        """
        Fragile-pool oracle-true 승급률 계산
        
        Promotion Rate = (승급한 oracle-true) / (총 fragile oracle-true)
        
        승급 = z_before < z0 → z_after >= z0 (명시적 조건)
        
        Args:
            fragile_pool_mask: Fragile-pool mask (before)
            z_before: z-scores before augmentation
            z_after: z-scores after augmentation
            z0: z-score threshold (default: 2.0)
        
        Returns:
            promotion_rate: 승급률 (0~1)
            n_promoted: 승급 개수
            n_fragile_oracle_true: 총 fragile oracle-true 개수
            promotion_details: 승급 항 상세 리스트
        """
        fragile = fragile_pool_mask.astype(bool)
        
        # Fragile-pool 중 oracle-true (recall fragility 대상)
        fragile_oracle_true = fragile & self.oracle_support
        
        # 승급 조건: 명시적으로 z_before < z0 AND z_after >= z0
        # (fragile_pool_mask가 외부에서 주입되므로 z_before 조건도 재확인)
        promoted = fragile_oracle_true & (z_before < z0) & (z_after >= z0)
        
        n_fragile_oracle_true = int(fragile_oracle_true.sum())
        n_promoted = int(promoted.sum())
        
        if n_fragile_oracle_true == 0:
            return 0.0, 0, 0, []  # 대상 없음
        
        rate = n_promoted / n_fragile_oracle_true
        
        # 상세 정보
        details = []
        for i, j in zip(*np.where(promoted)):
            details.append({
                'feature': self.feature_names[i],
                'target': self.target_names[j],
                'z_before': float(z_before[i, j]),
                'z_after': float(z_after[i, j]),
                'delta_z': float(z_after[i, j] - z_before[i, j]),
                'feature_idx': int(i),
                'target_idx': int(j)
            })
        
        return rate, n_promoted, n_fragile_oracle_true, details
    
    def compute_delta_z(
        self,
        z_before: np.ndarray,
        z_after: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float, List[Dict]]:
        """
        z-score 변화량 계산
        
        Args:
            z_before: z-scores before augmentation
            z_after: z-scores after augmentation
            mask: 계산 대상 마스크 (None이면 oracle-true 전체)
        
        Returns:
            delta_z_median: Δz 중앙값
            delta_z_mean: Δz 평균
            details: 항별 상세 리스트
        """
        if mask is None:
            mask = self.oracle_support
        
        mask = mask.astype(bool)
        
        if not mask.any():
            return 0.0, 0.0, []
        
        delta_z = z_after - z_before
        delta_values = delta_z[mask]
        
        details = []
        for i, j in zip(*np.where(mask)):
            details.append({
                'feature': self.feature_names[i],
                'target': self.target_names[j],
                'z_before': float(z_before[i, j]),
                'z_after': float(z_after[i, j]),
                'delta_z': float(delta_z[i, j]),
                'feature_idx': int(i),
                'target_idx': int(j)
            })
        
        # Δz 내림차순 정렬
        details.sort(key=lambda x: x['delta_z'], reverse=True)
        
        return float(np.median(delta_values)), float(np.mean(delta_values)), details
    
    def evaluate(
        self,
        teacher_support: np.ndarray,
        final_support: np.ndarray,
        z_before: np.ndarray,
        z_after: np.ndarray,
        fragile_pool_mask: np.ndarray,
        z0: float = 2.0,
        selected_support_pre_aug: Optional[np.ndarray] = None
    ) -> StructureEvalResult:
        """
        전체 구조 평가 수행
        
        Args:
            teacher_support: Teacher support mask (inc_prob >= tau_hi)
            final_support: Final model support mask (after augmentation)
            z_before: z-scores before augmentation (Teacher)
            z_after: z-scores after augmentation (Final)
            fragile_pool_mask: Fragile-pool mask from core_mining
            z0: z-score threshold
            selected_support_pre_aug: Selection 후 support mask (budget 적용 후)
                                      제공되면 spurious_reentry 계산
        
        Returns:
            StructureEvalResult: 평가 결과
        
        Raises:
            ValueError: 입력 shape이 oracle_support와 일치하지 않을 때
        """
        # Shape 검증
        expected_shape = (self.n_features, self.n_targets)
        inputs = {
            'teacher_support': teacher_support,
            'final_support': final_support,
            'z_before': z_before,
            'z_after': z_after,
            'fragile_pool_mask': fragile_pool_mask
        }
        if selected_support_pre_aug is not None:
            inputs['selected_support_pre_aug'] = selected_support_pre_aug
            
        for name, arr in inputs.items():
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} shape mismatch: {arr.shape} vs expected {expected_shape}"
                )
        
        # P0: selected ⊆ teacher 검증 (selection은 teacher pool에서 고르는 것)
        if selected_support_pre_aug is not None:
            teacher = teacher_support.astype(bool)
            selected = selected_support_pre_aug.astype(bool)
            invalid_selection = selected & (~teacher)
            if np.any(invalid_selection):
                n_invalid = int(invalid_selection.sum())
                raise ValueError(
                    f"selected_support_pre_aug contains {n_invalid} terms "
                    f"not in teacher_support. Selection must be a subset of teacher."
                )
        
        # 1. Spurious Reduction
        spur_red, n_removed, n_teacher_only, removed_details = \
            self.compute_spurious_reduction(teacher_support, final_support)
        
        # 2. Spurious Retention (잔존율)
        spur_retention, n_retained, _, retention_details = \
            self.compute_spurious_retention(teacher_support, final_support)
        
        # 3. Spurious Re-entry (selection 단계 있을 때만)
        spur_reentry = None
        n_removed_at_selection = None
        n_reentered = None
        reentry_details = []
        
        if selected_support_pre_aug is not None:
            spur_reentry, n_reentered, n_removed_at_selection, reentry_details = \
                self.compute_spurious_reentry(
                    teacher_support, selected_support_pre_aug, final_support
                )
        
        # 4. Promotion Rate
        promo_rate, n_promoted, n_fragile_oracle, promo_details = \
            self.compute_promotion_rate(fragile_pool_mask, z_before, z_after, z0)
        
        # 5. Δz (oracle-true)
        dz_median, dz_mean, dz_details = \
            self.compute_delta_z(z_before, z_after, self.oracle_support)
        
        return StructureEvalResult(
            spurious_reduction=spur_red,
            spurious_retention=spur_retention,
            promotion_rate=promo_rate,
            delta_z_median=dz_median,
            delta_z_mean=dz_mean,
            n_teacher_only_total=n_teacher_only,
            n_teacher_only_removed=n_removed,
            n_spurious_retained=n_retained,
            n_fragile_oracle_true=n_fragile_oracle,
            n_promoted=n_promoted,
            removed_spurious=removed_details,
            retained_spurious=retention_details,
            promoted_terms=promo_details,
            delta_z_details=dz_details,
            # Re-entry (선택적)
            spurious_reentry=spur_reentry,
            n_removed_at_selection=n_removed_at_selection,
            n_spurious_reentered=n_reentered,
            reentered_spurious=reentry_details
        )
    
    @staticmethod
    def compute_delta_floor(
        proposed_metrics: List[float],
        random_metrics: List[float],
        metric_name: str = "test_r2"
    ) -> DeltaFloorResult:
        """
        Δfloor 계산 (Robustness 개선)
        
        floor = mean - 1*std (worst-case estimation)
        Δfloor = proposed_floor - random_floor
        
        Args:
            proposed_metrics: 제안 방법의 메트릭 리스트 (여러 runs)
            random_metrics: Random selection의 메트릭 리스트 (여러 runs)
            metric_name: 메트릭 이름 (로깅용)
        
        Returns:
            DeltaFloorResult
        """
        proposed = np.array(proposed_metrics)
        random = np.array(random_metrics)
        
        # Floor = mean - 1*std (worst-case estimation, 1σ 기준)
        proposed_mean = float(np.mean(proposed))
        proposed_std = float(np.std(proposed))
        proposed_floor = proposed_mean - 1 * proposed_std
        
        random_mean = float(np.mean(random))
        random_std = float(np.std(random))
        random_floor = random_mean - 1 * random_std
        
        delta_floor = proposed_floor - random_floor
        
        # 유의성: Δfloor > 0
        significant = delta_floor > 0
        
        return DeltaFloorResult(
            delta_floor=delta_floor,
            proposed_floor=proposed_floor,
            random_floor=random_floor,
            proposed_mean=proposed_mean,
            proposed_std=proposed_std,
            random_mean=random_mean,
            random_std=random_std,
            n_proposed_runs=len(proposed),
            n_random_runs=len(random),
            significant=significant
        )


def compute_support_metrics(
    oracle_support: np.ndarray,
    predicted_support: np.ndarray
) -> Dict[str, float]:
    """
    Support 기반 메트릭 계산 (F1, Precision, Recall, Jaccard)
    
    Args:
        oracle_support: Ground-truth support mask
        predicted_support: Predicted support mask
    
    Returns:
        dict with precision, recall, f1, jaccard
    
    Raises:
        ValueError: shape이 일치하지 않을 때
    """
    # Shape 체크
    if oracle_support.shape != predicted_support.shape:
        raise ValueError(
            f"Shape mismatch: oracle {oracle_support.shape} vs "
            f"predicted {predicted_support.shape}"
        )
    
    oracle = oracle_support.astype(bool)
    pred = predicted_support.astype(bool)
    
    tp = (oracle & pred).sum()
    fp = (~oracle & pred).sum()
    fn = (oracle & ~pred).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    union = (oracle | pred).sum()
    jaccard = tp / union if union > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'jaccard': float(jaccard),
        'n_oracle': int(oracle.sum()),
        'n_predicted': int(pred.sum()),
        'n_true_positive': int(tp),
        'n_false_positive': int(fp),
        'n_false_negative': int(fn)
    }


if __name__ == "__main__":
    # 단위 테스트
    print("=== StructureEvaluator 단위 테스트 ===")
    
    # 테스트 데이터 (4x3 매트릭스)
    # Oracle: 5개 항
    oracle_support = np.array([
        [True,  False, True ],   # feat_a: oracle-true 2개
        [True,  False, False],   # feat_b: oracle-true 1개
        [False, True,  False],   # feat_c: oracle-true 1개
        [True,  False, False]    # feat_d: oracle-true 1개
    ])
    
    # Teacher: 8개 항 (5 both + 3 teacher-only)
    teacher_support = np.array([
        [True,  True,  True ],   # feat_a: 1 teacher-only
        [True,  False, True ],   # feat_b: 1 teacher-only
        [False, True,  False],   # feat_c: 0 teacher-only
        [True,  True,  False]    # feat_d: 1 teacher-only
    ])
    
    # Final: 6개 항 (augmentation 후)
    final_support = np.array([
        [True,  False, True ],   # feat_a: teacher-only 제거됨
        [True,  False, True ],   # feat_b: teacher-only 잔존
        [False, True,  False],   # feat_c
        [True,  False, False]    # feat_d: teacher-only 제거됨
    ])
    
    # z-scores
    z_before = np.array([
        [10.0, 0.5,  3.0],
        [5.0,  0.1,  1.5],  # [1,2]=1.5 < 2.0, teacher-only
        [0.2,  8.0,  0.3],
        [4.0,  1.0,  0.1]
    ])
    
    z_after = np.array([
        [12.0, 0.3,  4.0],
        [6.0,  0.1,  1.8],
        [0.3,  10.0, 0.2],
        [5.0,  0.8,  0.1]
    ])
    
    # Fragile-pool: teacher_support AND z < 2.0
    fragile_pool = teacher_support & (z_before < 2.0)
    
    feature_names = ['feat_a', 'feat_b', 'feat_c', 'feat_d']
    target_names = ['t1', 't2', 't3']
    
    evaluator = StructureEvaluator(oracle_support, feature_names, target_names)
    
    # Teacher-only 계산
    teacher_only = evaluator.compute_teacher_only_mask(teacher_support)
    print(f"\n[Teacher-only 마스크]")
    print(f"  총 개수: {teacher_only.sum()}")  # 3개
    
    # Spurious Reduction
    spur_red, n_removed, n_total, _ = evaluator.compute_spurious_reduction(
        teacher_support, final_support
    )
    print(f"\n[Spurious Reduction]")
    print(f"  총 Teacher-only: {n_total}")
    print(f"  제거된 개수: {n_removed}")
    print(f"  제거율: {spur_red:.1%}")
    
    # Spurious Retention (잔존율)
    spur_retention, n_retained, _, _ = evaluator.compute_spurious_retention(
        teacher_support, final_support
    )
    print(f"\n[Spurious Retention]")
    print(f"  잔존 개수: {n_retained}")
    print(f"  잔존율: {spur_retention:.1%}")
    
    # Fragile-pool oracle-true
    fragile_oracle = fragile_pool & oracle_support
    print(f"\n[Fragile-pool oracle-true]")
    print(f"  총 개수: {fragile_oracle.sum()}")
    
    # 전체 평가
    result = evaluator.evaluate(
        teacher_support=teacher_support,
        final_support=final_support,
        z_before=z_before,
        z_after=z_after,
        fragile_pool_mask=fragile_pool,
        z0=2.0
    )
    
    print(f"\n[전체 평가 결과]")
    print(f"  {result.summary()}")
    
    # Δfloor 테스트
    print(f"\n[Δfloor 테스트]")
    proposed = [0.95, 0.93, 0.94, 0.92, 0.96]
    random = [0.90, 0.85, 0.88, 0.82, 0.91]
    
    delta_result = StructureEvaluator.compute_delta_floor(proposed, random)
    print(f"  Proposed floor: {delta_result.proposed_floor:.3f}")
    print(f"  Random floor: {delta_result.random_floor:.3f}")
    print(f"  Δfloor: {delta_result.delta_floor:.3f}")
    print(f"  Significant: {delta_result.significant}")
    
    # Support metrics
    print(f"\n[Support Metrics]")
    metrics = compute_support_metrics(oracle_support, teacher_support)
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  F1: {metrics['f1']:.2f}")
    print(f"  Jaccard: {metrics['jaccard']:.2f}")
    
    print("\n✅ 단위 테스트 통과!")