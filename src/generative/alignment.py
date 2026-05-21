#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Teacher Alignment Module
==============================
E-SINDy Teacher를 활용한 alignment score 계산.

Lock-3 준수:
- align_score 정의: mean_t || dx_aug - Theta(x_aug, u_aug) @ Xi_bar ||^2
- 입력 dx는 dx_policy로 계산된 값 사용 (teacher로 dx 생성 금지)
- Teacher는 Gate1에서 고정된 계수만 사용
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json


@dataclass
class TeacherInfo:
    """Teacher 정보 (Gate1 E-SINDy에서 로드)"""
    source_run_id: str
    coefficients_mean: np.ndarray     # Xi_bar: (n_features, n_states)
    coefficients_std: np.ndarray      # sigma_Xi: (n_features, n_states)
    inclusion_probability: np.ndarray # (n_features, n_states)
    n_bootstrap: int
    library_config: str
    feature_names: List[str]
    
    # Lock-2: 매칭 검증용
    track: str
    n_train: int
    data_seed: int


class TeacherAlignment:
    """
    Teacher Alignment 계산기
    
    Lock-3 준수:
    - align_score_definition: "mean_t || dx_aug - Theta(x_aug, u_aug) @ Xi_bar ||^2"
    - dx는 외부에서 계산된 값 사용
    """
    
    ALIGN_SCORE_DEFINITION = "mean_t || dx_aug - Theta(x_aug, u_aug) @ Xi_bar ||^2"
    
    def __init__(self, teacher_info: TeacherInfo, library_builder=None):
        """
        Args:
            teacher_info: Gate1에서 로드한 Teacher 정보
            library_builder: SINDy library 빌더 (None이면 기본 사용)
        """
        self.teacher = teacher_info
        self.library_builder = library_builder
        
        # Validate teacher
        if self.teacher.coefficients_mean is None:
            raise ValueError("Teacher coefficients_mean is None")
    
    @classmethod
    def from_gate1_run(cls, gate1_results_dir: Path, run_id: str,
                       expected_track: Optional[str] = None,
                       expected_n_train: Optional[int] = None,
                       expected_data_seed: Optional[int] = None,
                       mismatch_action: str = 'fail') -> 'TeacherAlignment':
        """
        Gate1 결과에서 Teacher 로드
        
        Lock-2 준수: 매칭 검증 수행
        
        Args:
            gate1_results_dir: Gate1 results 루트 디렉토리
            run_id: Gate1 run_id
            expected_*: 매칭 검증용 기대값
            mismatch_action: 'fail' or 'warn'
        """
        # Find run directory
        matches = list(gate1_results_dir.rglob(f'*{run_id}*/metrics.json'))
        
        if not matches:
            raise FileNotFoundError(f"Gate1 run not found: {run_id}")
        
        run_dir = matches[0].parent
        
        # Load manifest
        manifest_path = run_dir / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {run_dir}")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Lock-2: 매칭 검증
        actual_track = manifest.get('track', 'unknown')
        # n_train과 seed는 config 안에 있음
        actual_n_train = manifest.get('config', {}).get('n_train', -1)
        actual_data_seed = manifest.get('config', {}).get('seed', -1)
        
        mismatches = []
        if expected_track and actual_track != expected_track:
            mismatches.append(f"track: expected {expected_track}, got {actual_track}")
        if expected_n_train and actual_n_train != expected_n_train:
            mismatches.append(f"n_train: expected {expected_n_train}, got {actual_n_train}")
        if expected_data_seed is not None and actual_data_seed != expected_data_seed:
            mismatches.append(f"data_seed: expected {expected_data_seed}, got {actual_data_seed}")
        
        if mismatches:
            mismatch_msg = f"Gate1 baseline mismatch: {'; '.join(mismatches)}"
            if mismatch_action == 'fail':
                raise ValueError(mismatch_msg)
            else:
                import warnings
                warnings.warn(mismatch_msg)
        
        # Load coefficients
        coef_path = run_dir / 'sindy_coefficients.csv'
        coef_std_path = run_dir / 'coefficient_std.csv'
        inclusion_path = run_dir / 'inclusion_probability.csv'
        
        # Parse CSV files
        coefficients_mean = _load_coefficient_csv(coef_path)
        coefficients_std = _load_coefficient_csv(coef_std_path) if coef_std_path.exists() else None
        inclusion_prob = _load_coefficient_csv(inclusion_path) if inclusion_path.exists() else None
        
        # Get feature names from coefficient file header
        feature_names = _get_feature_names(coef_path)
        
        # Create TeacherInfo
        teacher_info = TeacherInfo(
            source_run_id=run_id,
            coefficients_mean=coefficients_mean,
            coefficients_std=coefficients_std if coefficients_std is not None else np.zeros_like(coefficients_mean),
            inclusion_probability=inclusion_prob if inclusion_prob is not None else np.ones_like(coefficients_mean),
            n_bootstrap=manifest.get('config', {}).get('n_bootstrap', 20),
            library_config=manifest.get('config', {}).get('library_config', 'gate0_min'),
            feature_names=feature_names,
            track=actual_track,
            n_train=actual_n_train,
            data_seed=actual_data_seed,
        )
        
        return cls(teacher_info)
    
    def build_library(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        SINDy 라이브러리 구성
        
        Args:
            x: (N, T, 4) 상태 [x, x_dot, theta, theta_dot]
            u: (N, T, 1) 입력
        
        Returns:
            Theta: (N*T, n_features)
        """
        if self.library_builder is not None:
            return self.library_builder(x, u)
        
        # Default: Gate0/Gate1 라이브러리 (21 features for Cart-Pole)
        return _build_cartpole_library(x, u)
    
    def compute_align_score(self, x_aug: np.ndarray, dx_aug: np.ndarray, 
                            u_aug: np.ndarray) -> np.ndarray:
        """
        Alignment score 계산
        
        Lock-3 정의: align_score = mean_t || dx_aug - Theta(x_aug, u_aug) @ Xi_bar ||^2
        
        Args:
            x_aug: (N, T, 4) 생성된 상태
            dx_aug: (N, T, 4) 외부에서 계산된 미분 (Lock-3: teacher로 생성 금지)
            u_aug: (N, T, 1) 입력
        
        Returns:
            scores: (N,) 각 궤적의 alignment score (낮을수록 좋음)
        """
        N, T, state_dim = x_aug.shape
        
        # Build library: (N*T, n_features)
        Theta = self.build_library(x_aug, u_aug)
        
        # Reshape dx: (N*T, state_dim)
        dx_flat = dx_aug.reshape(N * T, state_dim)
        
        # Predict: dx_pred = Theta @ Xi_bar
        # Xi_bar: (n_features, state_dim)
        dx_pred = Theta @ self.teacher.coefficients_mean
        
        # Residual: (N*T, state_dim)
        residual = dx_flat - dx_pred
        
        # Per-timestep MSE: (N*T,)
        mse_per_timestep = np.mean(residual**2, axis=1)
        
        # Per-trajectory mean: (N,)
        mse_per_traj = mse_per_timestep.reshape(N, T).mean(axis=1)
        
        return mse_per_traj
    
    def compute_align_score_stats(self, scores: np.ndarray) -> Dict:
        """Alignment score 통계 (aug_manifest용)"""
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
        }
    
    def get_teacher_info_dict(self) -> Dict:
        """Teacher 정보 반환 (aug_manifest용)"""
        return {
            'source_run_id': self.teacher.source_run_id,
            'frozen': True,  # Lock: Teacher는 항상 고정
            'n_bootstrap': self.teacher.n_bootstrap,
            'library_config': self.teacher.library_config,
            'track': self.teacher.track,
            'n_train': self.teacher.n_train,
            'data_seed': self.teacher.data_seed,
            'n_features': len(self.teacher.feature_names),
            'mean_coef_norm': float(np.linalg.norm(self.teacher.coefficients_mean)),
        }


def compute_align_score(x_aug: np.ndarray, dx_aug: np.ndarray, u_aug: np.ndarray,
                        teacher_coefficients: np.ndarray,
                        library_builder=None) -> np.ndarray:
    """
    Standalone align score 계산 함수
    
    Args:
        x_aug: (N, T, 4) 생성된 상태
        dx_aug: (N, T, 4) 외부에서 계산된 미분
        u_aug: (N, T, 1) 입력
        teacher_coefficients: (n_features, state_dim) Teacher 계수 평균
        library_builder: SINDy library 빌더
    
    Returns:
        scores: (N,) alignment scores
    """
    N, T, state_dim = x_aug.shape
    
    # Build library
    if library_builder is not None:
        Theta = library_builder(x_aug, u_aug)
    else:
        Theta = _build_cartpole_library(x_aug, u_aug)
    
    # Flatten
    dx_flat = dx_aug.reshape(N * T, state_dim)
    
    # Predict and compute residual
    dx_pred = Theta @ teacher_coefficients
    residual = dx_flat - dx_pred
    
    # Per-trajectory mean MSE
    mse_per_timestep = np.mean(residual**2, axis=1)
    mse_per_traj = mse_per_timestep.reshape(N, T).mean(axis=1)
    
    return mse_per_traj


# =============================================================================
# Helper Functions
# =============================================================================

def _load_coefficient_csv(path: Path) -> np.ndarray:
    """coefficient CSV 파일 로드"""
    import csv
    
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Find state columns (dx_0, dx_1, dx_2, dx_3 or similar)
        data_cols = [i for i, h in enumerate(header) if h.startswith('dx_') or h.startswith('d')]
        if not data_cols:
            data_cols = list(range(1, len(header)))  # Skip first column (term_name)
        
        rows = []
        for row in reader:
            rows.append([float(row[i]) for i in data_cols])
    
    return np.array(rows)


def _get_feature_names(path: Path) -> List[str]:
    """coefficient CSV에서 feature names 추출"""
    import csv
    
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        
        names = []
        for row in reader:
            if row:
                names.append(row[0])  # First column is term_name
    
    return names


def _build_cartpole_library(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Cart-Pole SINDy 라이브러리 구성
    
    Gate0/Gate1과 동일한 21개 feature:
    1, x, x_dot, sin(theta), cos(theta), theta_dot, u,
    x^2, x*x_dot, x*sin, x*cos, x*theta_dot, x*u,
    x_dot^2, x_dot*sin, x_dot*cos, x_dot*theta_dot, x_dot*u,
    sin*cos, theta_dot^2, theta_dot*u
    """
    N, T, _ = x.shape
    
    # Flatten: (N*T, 4)
    x_flat = x.reshape(N * T, 4)
    u_flat = u.reshape(N * T, 1) if u.ndim == 3 else u.reshape(N * T, 1)
    
    # Extract states
    pos = x_flat[:, 0]        # x
    vel = x_flat[:, 1]        # x_dot
    theta = x_flat[:, 2]      # theta
    omega = x_flat[:, 3]      # theta_dot
    ctrl = u_flat[:, 0]       # u
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Build library (21 features)
    features = [
        np.ones_like(pos),           # 1
        pos,                          # x
        vel,                          # x_dot
        sin_theta,                    # sin(theta)
        cos_theta,                    # cos(theta)
        omega,                        # theta_dot
        ctrl,                         # u
        pos**2,                       # x^2
        pos * vel,                    # x * x_dot
        pos * sin_theta,              # x * sin
        pos * cos_theta,              # x * cos
        pos * omega,                  # x * theta_dot
        pos * ctrl,                   # x * u
        vel**2,                       # x_dot^2
        vel * sin_theta,              # x_dot * sin
        vel * cos_theta,              # x_dot * cos
        vel * omega,                  # x_dot * theta_dot
        vel * ctrl,                   # x_dot * u
        sin_theta * cos_theta,        # sin * cos
        omega**2,                     # theta_dot^2
        omega * ctrl,                 # theta_dot * u
    ]
    
    Theta = np.column_stack(features)
    
    return Theta