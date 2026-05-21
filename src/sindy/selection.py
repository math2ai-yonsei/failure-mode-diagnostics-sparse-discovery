"""
Term Selection Module for Phase 3.5

SSOT:
- Ranking: Lexicographic (-inc_prob, -z_score, +global_idx)
- P0 Lock: selected ⊆ teacher_support

Author: Claude (Phase 3.5 Day3)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import json
import numpy as np


@dataclass
class SelectionResult:
    """Selection 결과"""
    method: str  # 'stable_core_only' or 'budget_plus_fragile'
    selected_mask: np.ndarray  # (n_features, n_targets)
    n_selected: int
    n_stable_core_selected: int
    n_fragile_selected: int
    budget: Optional[int]
    selected_terms: List[Dict]
    
    def to_dict(self) -> Dict:
        return {
            'method': self.method,
            'summary': {
                'n_selected': self.n_selected,
                'n_stable_core_selected': self.n_stable_core_selected,
                'n_fragile_selected': self.n_fragile_selected,
                'budget': self.budget
            },
            'ranking': {
                'type': 'lexicographic',
                'keys': ['-inc_prob', '-z_score', '+global_idx'],
                'description': 'inc_prob descending, z_score descending, global_idx ascending (tie-break)'
            },
            'selected_terms': self.selected_terms
        }
    
    def save_json(self, path: Path):
        """JSON으로 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _get_lexicographic_key(
    i: int, 
    j: int, 
    inc_prob: np.ndarray, 
    z_scores: np.ndarray, 
    n_targets: int
):
    """
    Lexicographic sort key 반환
    
    SSOT 정렬 기준:
    1. inc_prob 내림차순 (높을수록 우선)
    2. z_score 내림차순 (높을수록 우선)  
    3. global_idx 오름차순 (tie-break)
    
    Args:
        i: feature index
        j: target index
        inc_prob: inclusion probability matrix
        z_scores: z-score matrix
        n_targets: target 개수
    
    Returns:
        tuple: (-inc_prob, -z_score, global_idx) for sorting
    """
    global_idx = i * n_targets + j
    return (-inc_prob[i, j], -z_scores[i, j], global_idx)


def select_terms(
    stable_core_mask: np.ndarray,
    fragile_pool_mask: np.ndarray,
    teacher_support: np.ndarray,
    inc_prob: np.ndarray,
    z_scores: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    method: str = 'stable_core_only',
    budget: Optional[int] = None
) -> SelectionResult:
    """
    Term Selection 수행
    
    SSOT:
    - method='stable_core_only': Arm A, stable-core만 선택
    - method='budget_plus_fragile': Arm B, stable-core + fragile top-k
    - Ranking: Lexicographic (-inc_prob, -z_score, +global_idx)
    - P0 Lock: selected ⊆ teacher_support
    
    Args:
        stable_core_mask: stable-core 마스크 (n_features, n_targets)
        fragile_pool_mask: fragile-pool 마스크 (n_features, n_targets)
        teacher_support: teacher support 마스크 (n_features, n_targets)
        inc_prob: inclusion probability (n_features, n_targets)
        z_scores: z-score (n_features, n_targets)
        feature_names: feature 이름 리스트
        target_names: target 이름 리스트
        method: 선택 방법 ('stable_core_only' or 'budget_plus_fragile')
        budget: Arm B에서 총 선택할 term 개수
    
    Returns:
        SelectionResult: 선택 결과
    
    Raises:
        ValueError: budget이 Arm B에서 필수인데 없는 경우
        AssertionError: P0 Lock 위반 시
    """
    n_features, n_targets = inc_prob.shape
    stable_mask = stable_core_mask.astype(bool)
    fragile_mask = fragile_pool_mask.astype(bool)
    
    if method == 'stable_core_only':
        # Arm A: stable_core만 선택
        selected_mask = stable_mask.copy()
        n_stable_selected = int(stable_mask.sum())
        n_fragile_selected = 0
        
    elif method == 'budget_plus_fragile':
        # Arm B: stable_core 우선 + fragile top-k
        if budget is None:
            raise ValueError("budget is required for 'budget_plus_fragile' method")
        
        n_stable = int(stable_mask.sum())
        
        if n_stable >= budget:
            # Stable-core가 budget보다 많으면 ranking 기준 상위 선택
            stable_candidates = [
                (i, j) for i in range(n_features) for j in range(n_targets)
                if stable_mask[i, j]
            ]
            stable_candidates.sort(
                key=lambda x: _get_lexicographic_key(x[0], x[1], inc_prob, z_scores, n_targets)
            )
            
            selected_mask = np.zeros((n_features, n_targets), dtype=bool)
            for i, j in stable_candidates[:budget]:
                selected_mask[i, j] = True
            n_stable_selected = budget
            n_fragile_selected = 0
        else:
            # Stable-core 전부 + fragile top-k
            remaining = budget - n_stable
            
            fragile_candidates = [
                (i, j) for i in range(n_features) for j in range(n_targets)
                if fragile_mask[i, j]
            ]
            fragile_candidates.sort(
                key=lambda x: _get_lexicographic_key(x[0], x[1], inc_prob, z_scores, n_targets)
            )
            
            selected_mask = stable_mask.copy()
            n_fragile_selected = 0
            for i, j in fragile_candidates[:remaining]:
                selected_mask[i, j] = True
                n_fragile_selected += 1
            n_stable_selected = n_stable
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # ============================================================
    # P0 LOCK: selected ⊆ teacher_support (fail-fast 강제)
    # NOTE: assert 대신 raise 사용 (python -O 에서도 동작 보장)
    # ============================================================
    if not np.all(selected_mask <= teacher_support):
        raise ValueError("P0 VIOLATION: selected contains terms not in teacher_support")
    
    # selected_terms 생성
    n_selected = int(selected_mask.sum())
    selected_terms = []
    
    for i, feat in enumerate(feature_names):
        for j, tgt in enumerate(target_names):
            if selected_mask[i, j]:
                source = 'stable_core' if stable_core_mask[i, j] else 'fragile_pool'
                selected_terms.append({
                    'feature': feat,
                    'target': tgt,
                    'feature_idx': i,
                    'target_idx': j,
                    'global_idx': i * n_targets + j,
                    'inc_prob': float(inc_prob[i, j]),
                    'z_score': float(z_scores[i, j]),
                    'source': source
                })
    
    # Lexicographic 정렬
    selected_terms.sort(key=lambda x: (-x['inc_prob'], -x['z_score'], x['global_idx']))
    
    return SelectionResult(
        method=method,
        selected_mask=selected_mask,
        n_selected=n_selected,
        n_stable_core_selected=n_stable_selected,
        n_fragile_selected=n_fragile_selected,
        budget=budget if method == 'budget_plus_fragile' else None,
        selected_terms=selected_terms
    )