"""
Stable-core Mining 모듈 (Phase 3.5)

핵심 역할:
- E-SINDy Teacher의 coefficient 통계로부터 Stable-core / Fragile-pool 추출
- 방법론은 Stable-core만 사용, Fragile-pool은 진단/평가 전용

정의:
- Stable-core: (inc_prob >= tau_hi) AND (z >= z0)
- Fragile-pool: (inc_prob >= tau_hi) AND (z < z0)
- z = |mean| / (std + eps)

파라미터 (사전 고정):
- tau_hi = 0.5
- z0 = 2.0
- eps = 1e-12

Author: Claude (Phase 3.5)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


@dataclass
class CoreMiningResult:
    """Core mining 결과를 담는 데이터 클래스"""
    
    # 파라미터
    tau_hi: float
    z0: float
    eps: float
    
    # 전체 통계
    n_total_terms: int
    n_active_terms: int  # inc_prob >= tau_hi
    n_stable_core: int
    n_fragile_pool: int
    
    # 마스크 (feature_name -> target_name -> bool)
    stable_core_mask: np.ndarray  # shape: (n_features, n_targets)
    fragile_pool_mask: np.ndarray
    active_mask: np.ndarray  # inc_prob >= tau_hi
    
    # z-score 배열
    z_scores: np.ndarray  # shape: (n_features, n_targets)
    
    # coefficient 원본 (selection score 계산용)
    coef_mean: np.ndarray
    coef_std: np.ndarray
    inc_prob: np.ndarray
    
    # 피처/타겟 이름
    feature_names: List[str]
    target_names: List[str]
    
    # 상세 결과 (리스트 of dict)
    stable_core_terms: List[Dict[str, Any]] = field(default_factory=list)
    fragile_pool_terms: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """JSON 직렬화 가능한 dict로 변환"""
        return {
            'params': {
                'tau_hi': self.tau_hi,
                'z0': self.z0,
                'eps': self.eps
            },
            'summary': {
                'n_total_terms': self.n_total_terms,
                'n_active_terms': self.n_active_terms,
                'n_stable_core': self.n_stable_core,
                'n_fragile_pool': self.n_fragile_pool
            },
            'stable_core_terms': self.stable_core_terms,
            'fragile_pool_terms': self.fragile_pool_terms,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
    
    def save_json(self, path: Path):
        """JSON 파일로 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_stable_core_coef_mean(self) -> np.ndarray:
        """Stable-core만 포함한 coefficient mean (나머지 0)"""
        masked = self.coef_mean.copy()
        masked[~self.stable_core_mask] = 0.0
        return masked
    
    def get_stable_core_features(self) -> List[Tuple[str, str]]:
        """Stable-core 항의 (feature, target) 리스트"""
        return [(t['feature'], t['target']) for t in self.stable_core_terms]
    
    def get_fragile_pool_features(self) -> List[Tuple[str, str]]:
        """Fragile-pool 항의 (feature, target) 리스트"""
        return [(t['feature'], t['target']) for t in self.fragile_pool_terms]


class StableCoreMiner:
    """
    Stable-core / Fragile-pool 추출기
    
    Phase 3.5 핵심 컴포넌트:
    - E-SINDy Teacher의 coefficient 통계로부터 신뢰성 높은 항(Stable-core) 추출
    - 불안정한 항(Fragile-pool)은 진단/평가용으로만 사용
    
    사용법:
        miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
        result = miner.mine(coef_mean, coef_std, inc_prob, feature_names, target_names)
        
        # 또는 CSV에서 직접 로드
        result = StableCoreMiner.from_esindy_csv(csv_path)
    """
    
    def __init__(self, tau_hi: float = 0.5, z0: float = 2.0, eps: float = 1e-12):
        """
        Args:
            tau_hi: inclusion probability threshold (default: 0.5)
            z0: z-score threshold (default: 2.0)
            eps: small value for numerical stability (default: 1e-12)
        """
        if tau_hi < 0 or tau_hi > 1:
            raise ValueError(f"tau_hi must be in [0, 1], got {tau_hi}")
        if z0 < 0:
            raise ValueError(f"z0 must be non-negative, got {z0}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        
        self.tau_hi = tau_hi
        self.z0 = z0
        self.eps = eps
    
    def compute_z_scores(self, coef_mean: np.ndarray, coef_std: np.ndarray) -> np.ndarray:
        """
        z-score 계산: z = |mean| / (std + eps)
        
        Args:
            coef_mean: coefficient mean, shape (n_features, n_targets)
            coef_std: coefficient std, shape (n_features, n_targets)
        
        Returns:
            z_scores: shape (n_features, n_targets)
        """
        return np.abs(coef_mean) / (coef_std + self.eps)
    
    def mine(
        self,
        coef_mean: np.ndarray,
        coef_std: np.ndarray,
        inc_prob: np.ndarray,
        feature_names: List[str],
        target_names: List[str]
    ) -> CoreMiningResult:
        """
        Stable-core / Fragile-pool 추출
        
        Args:
            coef_mean: coefficient mean, shape (n_features, n_targets)
            coef_std: coefficient std, shape (n_features, n_targets)
            inc_prob: inclusion probability, shape (n_features, n_targets)
            feature_names: 피처 이름 리스트
            target_names: 타겟 이름 리스트
        
        Returns:
            CoreMiningResult: 추출 결과
        """
        # Shape 검증
        n_features, n_targets = coef_mean.shape
        if coef_std.shape != (n_features, n_targets):
            raise ValueError(f"coef_std shape mismatch: {coef_std.shape} vs {coef_mean.shape}")
        if inc_prob.shape != (n_features, n_targets):
            raise ValueError(f"inc_prob shape mismatch: {inc_prob.shape} vs {coef_mean.shape}")
        if len(feature_names) != n_features:
            raise ValueError(f"feature_names length mismatch: {len(feature_names)} vs {n_features}")
        if len(target_names) != n_targets:
            raise ValueError(f"target_names length mismatch: {len(target_names)} vs {n_targets}")
        
        # z-score 계산
        z_scores = self.compute_z_scores(coef_mean, coef_std)
        
        # === P1 FIX: dtype 검증 후 std >= 0, z_scores >= 0 검증 ===
        # dtype 검증은 입력 단계에서 수행됨 (from_esindy_csv)
        # std >= 0 검증 (물리적으로 음수 불가)
        if np.any(coef_std < 0):
            n_negative = (coef_std < 0).sum()
            raise ValueError(f"coef_std contains {n_negative} negative values (physically impossible)")
        # z_scores >= 0 검증 (fail-fast 강제)
        if not np.all(z_scores >= 0):
            raise ValueError(f"z_scores must be non-negative, min={z_scores.min()}")
        
        # 마스크 생성
        active_mask = inc_prob >= self.tau_hi
        high_z_mask = z_scores >= self.z0
        low_z_mask = z_scores < self.z0
        
        stable_core_mask = active_mask & high_z_mask
        fragile_pool_mask = active_mask & low_z_mask
        
        # 상세 결과 생성
        stable_core_terms = []
        fragile_pool_terms = []
        
        for i, feat in enumerate(feature_names):
            for j, tgt in enumerate(target_names):
                # global_candidate_index for deterministic tie-break
                global_idx = i * n_targets + j
                
                if stable_core_mask[i, j]:
                    stable_core_terms.append({
                        'feature': feat,
                        'target': tgt,
                        'coef_mean': float(coef_mean[i, j]),
                        'coef_std': float(coef_std[i, j]),
                        'inc_prob': float(inc_prob[i, j]),
                        'z_score': float(z_scores[i, j]),
                        'feature_idx': i,
                        'target_idx': j,
                        'global_idx': global_idx
                    })
                elif fragile_pool_mask[i, j]:
                    fragile_pool_terms.append({
                        'feature': feat,
                        'target': tgt,
                        'coef_mean': float(coef_mean[i, j]),
                        'coef_std': float(coef_std[i, j]),
                        'inc_prob': float(inc_prob[i, j]),
                        'z_score': float(z_scores[i, j]),
                        'feature_idx': i,
                        'target_idx': j,
                        'global_idx': global_idx
                    })
        
        # z-score 내림차순, 동점시 global_idx 오름차순 (deterministic tie-break)
        stable_core_terms.sort(key=lambda x: (-x['z_score'], x['global_idx']))
        fragile_pool_terms.sort(key=lambda x: (-x['z_score'], x['global_idx']))
        
        return CoreMiningResult(
            tau_hi=self.tau_hi,
            z0=self.z0,
            eps=self.eps,
            n_total_terms=n_features * n_targets,
            n_active_terms=int(active_mask.sum()),
            n_stable_core=int(stable_core_mask.sum()),
            n_fragile_pool=int(fragile_pool_mask.sum()),
            stable_core_mask=stable_core_mask,
            fragile_pool_mask=fragile_pool_mask,
            active_mask=active_mask,
            z_scores=z_scores,
            coef_mean=coef_mean,
            coef_std=coef_std,
            inc_prob=inc_prob,
            feature_names=feature_names,
            target_names=target_names,
            stable_core_terms=stable_core_terms,
            fragile_pool_terms=fragile_pool_terms
        )
    
    @classmethod
    def from_esindy_csv(
        cls,
        csv_path: Path,
        tau_hi: float = 0.5,
        z0: float = 2.0,
        eps: float = 1e-12,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        strict: bool = True
    ) -> CoreMiningResult:
        """
        E-SINDy sindy_coefficients.csv 파일에서 직접 core mining 수행
        
        CSV 포맷:
        - columns: feature, target, mean, std, inc_prob
        
        Args:
            csv_path: sindy_coefficients.csv 경로
            tau_hi, z0, eps: StableCoreMiner 파라미터
            feature_names: 피처 이름 순서 (SSOT). Gate1 manifest에서 로드 필수
            target_names: 타겟 이름 순서 (SSOT). Gate1 manifest에서 로드 필수
            strict: True이면 feature_names/target_names가 None일 때 ValueError
                    False이면 CSV에서 추론 (개발/디버깅용만, 비권장)
        
        Returns:
            CoreMiningResult
        
        Raises:
            FileNotFoundError: CSV 파일이 없을 때
            ValueError: 필수 컬럼 누락, row 누락/중복, strict=True인데 names가 None일 때
        
        Note:
            논문 수준 재현성을 위해 strict=True (기본값)를 사용하고,
            feature_names/target_names를 Gate1 manifest에서 로드하여 전달하세요.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # 필수 컬럼 확인
        required_cols = {'feature', 'target', 'mean', 'std', 'inc_prob'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        
        # SSOT 검증 (strict 모드)
        if strict:
            if feature_names is None:
                raise ValueError(
                    "feature_names is required in strict mode. "
                    "Load from Gate1 manifest for reproducibility. "
                    "Set strict=False only for development/debugging."
                )
            if target_names is None:
                raise ValueError(
                    "target_names is required in strict mode. "
                    "Load from Gate1 manifest for reproducibility. "
                    "Set strict=False only for development/debugging."
                )
        else:
            # CSV에서 추론 (비권장 - 순서가 df row order에 의존)
            if feature_names is None:
                import warnings
                warnings.warn(
                    "feature_names inferred from CSV order. "
                    "This may cause reproducibility issues.",
                    UserWarning
                )
                feature_names = list(dict.fromkeys(df['feature']))
            if target_names is None:
                import warnings
                warnings.warn(
                    "target_names inferred from CSV order. "
                    "This may cause reproducibility issues.",
                    UserWarning
                )
                target_names = list(dict.fromkeys(df['target']))
        
        # CSV에 있는 피처/타겟이 제공된 리스트와 일치하는지 검증
        csv_features = set(df['feature'].unique())
        csv_targets = set(df['target'].unique())
        if csv_features != set(feature_names):
            raise ValueError(
                f"Feature mismatch: CSV has {csv_features}, "
                f"but feature_names is {set(feature_names)}"
            )
        if csv_targets != set(target_names):
            raise ValueError(
                f"Target mismatch: CSV has {csv_targets}, "
                f"but target_names is {set(target_names)}"
            )
        
        n_features = len(feature_names)
        n_targets = len(target_names)
        expected_rows = n_features * n_targets
        
        # Row 개수 검증 (누락/중복 방지)
        if len(df) != expected_rows:
            raise ValueError(
                f"CSV row count mismatch: {len(df)} rows, "
                f"expected {expected_rows} ({n_features} features × {n_targets} targets)"
            )
        
        # 중복 key 검증
        key_counts = df.groupby(['feature', 'target']).size()
        duplicates = key_counts[key_counts > 1]
        if len(duplicates) > 0:
            raise ValueError(
                f"Duplicate (feature, target) pairs in CSV: {duplicates.to_dict()}"
            )
        
        # 인덱스 맵 (SSOT 순서 기준)
        feat_idx = {f: i for i, f in enumerate(feature_names)}
        tgt_idx = {t: i for i, t in enumerate(target_names)}
        
        # 배열 초기화
        coef_mean = np.zeros((n_features, n_targets))
        coef_std = np.zeros((n_features, n_targets))
        inc_prob = np.zeros((n_features, n_targets))
        
        # 데이터 채우기
        for _, row in df.iterrows():
            i = feat_idx[row['feature']]
            j = tgt_idx[row['target']]
            coef_mean[i, j] = row['mean']
            coef_std[i, j] = row['std']
            inc_prob[i, j] = row['inc_prob']
        
        # Mining 수행
        miner = cls(tau_hi=tau_hi, z0=z0, eps=eps)
        return miner.mine(coef_mean, coef_std, inc_prob, feature_names, target_names)
    
    @classmethod
    def from_gate1_artifacts(
        cls,
        run_dir: Path,
        feature_names: List[str],
        target_names: List[str],
        tau_hi: float = 0.5,
        z0: float = 2.0,
        eps: float = 1e-12
    ) -> CoreMiningResult:
        """
        Gate1 artifacts (3개 Wide-format CSV)에서 Core Mining 수행
        
        Gate1 저장 포맷:
        - sindy_coefficients.csv: final mean coefficients (Wide: term_name, x_dot, ...)
        - coefficient_std.csv: coefficient std (Wide: term_name, x_dot, ...)
        - inclusion_probability.csv: inclusion probability (Wide: term_name, x_dot, ...)
        
        Args:
            run_dir: Gate1 run 디렉토리 (manifest.json이 있는 폴더)
            feature_names: 피처 이름 순서 (SSOT, manifest에서 로드)
            target_names: 타겟 이름 순서 (SSOT)
            tau_hi, z0, eps: StableCoreMiner 파라미터
        
        Returns:
            CoreMiningResult
        
        Raises:
            FileNotFoundError: 필수 CSV 파일이 없을 때
            ValueError: CSV 포맷 불일치
        
        Note:
            Gate1 manifest.json에서 feature_names, target_names를 로드하여 전달하세요.
            optimizer.target_names가 없으면 기본값 ["x_dot", "x_ddot", "theta_dot", "theta_ddot"] 사용
        """
        run_dir = Path(run_dir)
        term_col = 'term_name'
        
        # 파일 경로
        coef_path = run_dir / 'sindy_coefficients.csv'
        std_path = run_dir / 'coefficient_std.csv'
        inc_prob_path = run_dir / 'inclusion_probability.csv'
        
        # 파일 존재 확인
        for p in [coef_path, std_path, inc_prob_path]:
            if not p.exists():
                raise FileNotFoundError(f"Gate1 artifact not found: {p}")
        
        # Wide-format CSV 로드
        df_coef = pd.read_csv(coef_path)
        df_std = pd.read_csv(std_path)
        df_inc_prob = pd.read_csv(inc_prob_path)
        
        n_features = len(feature_names)
        n_targets = len(target_names)
        
        # === P0 FIX: 3개 CSV 모두에 대해 동일한 검증 수행 ===
        def validate_wide_csv(df: pd.DataFrame, csv_name: str, check_range: bool = False):
            """Wide-format CSV 검증 (fail-fast)
            
            Args:
                df: DataFrame to validate
                csv_name: CSV file name for error messages
                check_range: True면 [0, 1] 범위 체크 (inc_prob용)
            """
            # 1. term_name 컬럼 존재 확인
            if term_col not in df.columns:
                raise ValueError(f"'{term_col}' column not found in {csv_name}")
            
            # 2. target 컬럼 확인
            csv_targets = [c for c in df.columns if c != term_col]
            if set(csv_targets) != set(target_names):
                raise ValueError(
                    f"Target mismatch in {csv_name}: CSV has {csv_targets}, "
                    f"but target_names is {target_names}"
                )
            
            # 3. Row count 확인
            if len(df) != n_features:
                raise ValueError(
                    f"Row count mismatch in {csv_name}: {len(df)} rows, "
                    f"expected {n_features} features"
                )
            
            # 4. Feature set 확인
            csv_features = df[term_col].tolist()
            if set(csv_features) != set(feature_names):
                raise ValueError(
                    f"Feature mismatch in {csv_name}: CSV has {set(csv_features)}, "
                    f"but feature_names is {set(feature_names)}"
                )
            
            # 5. 중복 term_name 확인 (pandas duplicated 사용)
            if df[term_col].duplicated().any():
                duplicates = df[term_col][df[term_col].duplicated()].tolist()
                raise ValueError(
                    f"Duplicate term_name in {csv_name}: {set(duplicates)}"
                )
            
            # 6. NaN/inf 체크 (값 오염 방지)
            numeric_cols = [c for c in df.columns if c != term_col]
            for col in numeric_cols:
                if df[col].isna().any():
                    nan_count = df[col].isna().sum()
                    raise ValueError(
                        f"NaN values found in {csv_name}[{col}]: {nan_count} NaN(s)"
                    )
                if np.isinf(df[col]).any():
                    inf_count = np.isinf(df[col]).sum()
                    raise ValueError(
                        f"Inf values found in {csv_name}[{col}]: {inf_count} Inf(s)"
                    )
            
            # 7. 범위 체크 (inc_prob 전용)
            if check_range:
                for col in numeric_cols:
                    min_val, max_val = df[col].min(), df[col].max()
                    if min_val < 0 or max_val > 1:
                        raise ValueError(
                            f"Out of range [0,1] in {csv_name}[{col}]: "
                            f"min={min_val:.4f}, max={max_val:.4f}"
                        )
        
        # 3개 CSV 모두 검증 (inc_prob만 범위 체크)
        validate_wide_csv(df_coef, 'sindy_coefficients.csv', check_range=False)
        validate_wide_csv(df_std, 'coefficient_std.csv', check_range=False)
        validate_wide_csv(df_inc_prob, 'inclusion_probability.csv', check_range=True)
        
        # 인덱스 맵 (SSOT 순서 기준)
        feat_idx = {f: i for i, f in enumerate(feature_names)}
        tgt_idx = {t: j for j, t in enumerate(target_names)}
        
        # 배열 초기화
        coef_mean = np.zeros((n_features, n_targets))
        coef_std = np.zeros((n_features, n_targets))
        inc_prob = np.zeros((n_features, n_targets))
        
        # Wide → 2D array 변환 (SSOT 순서 기준)
        for _, row in df_coef.iterrows():
            feat = row[term_col]
            i = feat_idx[feat]
            for tgt in target_names:
                j = tgt_idx[tgt]
                coef_mean[i, j] = row[tgt]
        
        for _, row in df_std.iterrows():
            feat = row[term_col]
            i = feat_idx[feat]
            for tgt in target_names:
                j = tgt_idx[tgt]
                coef_std[i, j] = row[tgt]
        
        for _, row in df_inc_prob.iterrows():
            feat = row[term_col]
            i = feat_idx[feat]
            for tgt in target_names:
                j = tgt_idx[tgt]
                inc_prob[i, j] = row[tgt]
        
        # Mining 수행
        miner = cls(tau_hi=tau_hi, z0=z0, eps=eps)
        return miner.mine(coef_mean, coef_std, inc_prob, feature_names, target_names)


def validate_against_qc2(result: CoreMiningResult, qc2_path: Path) -> Dict[str, Any]:
    """
    QC-2 분석 결과와 core mining 결과 비교 검증
    
    Args:
        result: CoreMiningResult
        qc2_path: qc2_teacher_only_analysis.json 경로
    
    Returns:
        검증 결과 dict
    """
    qc2_path = Path(qc2_path)
    with open(qc2_path, 'r', encoding='utf-8') as f:
        qc2 = json.load(f)
    
    # QC-2 기대값
    n_both_qc2 = qc2['summary']['n_both']
    n_teacher_only_qc2 = qc2['summary']['n_teacher_only']
    
    # Both 중 z>=2.0 개수 계산
    both_terms = qc2['details']['both']
    n_both_stable = sum(1 for t in both_terms if t['teacher_z'] >= result.z0)
    n_both_fragile = n_both_qc2 - n_both_stable
    
    # Teacher-only는 모두 z<2.0이므로 전부 fragile
    n_teacher_only_fragile = n_teacher_only_qc2
    
    # 기대값
    expected_stable = n_both_stable
    expected_fragile = n_both_fragile + n_teacher_only_fragile
    
    # 검증
    validation = {
        'qc2_summary': qc2['summary'],
        'expected': {
            'n_stable_core': expected_stable,
            'n_fragile_pool': expected_fragile,
            'breakdown': {
                'both_stable': n_both_stable,
                'both_fragile': n_both_fragile,
                'teacher_only_fragile': n_teacher_only_fragile
            }
        },
        'actual': {
            'n_stable_core': result.n_stable_core,
            'n_fragile_pool': result.n_fragile_pool
        },
        'match': {
            'stable_core': result.n_stable_core == expected_stable,
            'fragile_pool': result.n_fragile_pool == expected_fragile
        },
        'passed': (result.n_stable_core == expected_stable and 
                   result.n_fragile_pool == expected_fragile)
    }
    
    return validation


if __name__ == "__main__":
    # 간단한 테스트
    print("=== StableCoreMiner 단위 테스트 ===")
    
    # 테스트 데이터 (3x2 매트릭스)
    coef_mean = np.array([
        [10.0, 0.5],   # 첫 번째 피처
        [2.0, 0.1],    # 두 번째 피처
        [0.8, 3.0]     # 세 번째 피처
    ])
    coef_std = np.array([
        [1.0, 1.0],    # z = [10.0, 0.5]
        [0.5, 0.1],    # z = [4.0, 1.0]
        [1.0, 0.5]     # z = [0.8, 6.0]
    ])
    inc_prob = np.array([
        [1.0, 0.6],    # 둘 다 active
        [0.8, 0.3],    # 첫째만 active
        [0.4, 0.9]     # 둘째만 active
    ])
    
    feature_names = ['feat_a', 'feat_b', 'feat_c']
    target_names = ['target_1', 'target_2']
    
    miner = StableCoreMiner(tau_hi=0.5, z0=2.0)
    result = miner.mine(coef_mean, coef_std, inc_prob, feature_names, target_names)
    
    print(f"\n파라미터: tau_hi={result.tau_hi}, z0={result.z0}")
    print(f"전체 항: {result.n_total_terms}")
    print(f"활성 항 (inc_prob >= 0.5): {result.n_active_terms}")
    print(f"Stable-core: {result.n_stable_core}")
    print(f"Fragile-pool: {result.n_fragile_pool}")
    
    print("\n=== Stable-core 항 ===")
    for term in result.stable_core_terms:
        print(f"  {term['feature']} → {term['target']}: "
              f"z={term['z_score']:.2f}, inc_prob={term['inc_prob']:.2f}")
    
    print("\n=== Fragile-pool 항 ===")
    for term in result.fragile_pool_terms:
        print(f"  {term['feature']} → {term['target']}: "
              f"z={term['z_score']:.2f}, inc_prob={term['inc_prob']:.2f}")
    
    # 예상 결과 검증
    assert result.n_stable_core == 3, f"Expected 3 stable-core, got {result.n_stable_core}"
    assert result.n_fragile_pool == 1, f"Expected 1 fragile-pool, got {result.n_fragile_pool}"
    
    print("\n✅ 단위 테스트 통과!")