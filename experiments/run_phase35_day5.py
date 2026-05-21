"""
Phase 3.5 Day5 Runner: Control Experiments for Confound Separation

핵심 목표:
- 4-way 분해로 augmentation 효과 분리
  1. Baseline: z_before (Day3)
  2. Control-50: 원본 50으로 재학습 (retrain only)
  3. Control-250: 원본 복제 250 (resample only, perturb 없음)
  4. Treatment: aug + retrain (Day4)

효과 분해:
- Retrain Effect = z_ctrl50 - z_before
- Size Effect = z_ctrl250 - z_ctrl50
- Aug Pure Effect = z_treat - z_ctrl250
- Total Effect = z_treat - z_before

산출물:
- manifest.json: 실험 메타데이터 + control_equivalence
- metrics.json: Training effect metrics
- structure_eval.json: delta_z_details
- z_after.npy: z_after_ctrl50 또는 z_after_ctrl250
- coef_mean_after.npy, coef_std_after.npy, inc_prob_after.npy
- (compare mode) comparison_4way.json: 4-way 비교표

Author: Claude (Phase 3.5 Day5)
Updated: Phase 3.5 Option B - Modern Schema with dx_key_used 실측
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# 기존 프로젝트 모듈 활용
from src.sindy.library import SINDyLibrary, get_derivative_key
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble

# === MODERNIZE IMPORT ===
from phase35_manifest_modernize import (
    get_control_equivalence,
    compute_file_sha256
)


# ============================================================
# SSOT Constants (Day3/4와 동일)
# ============================================================
DEFAULT_TARGET_NAMES = ["x_dot", "x_ddot", "theta_dot", "theta_ddot"]
DEFAULT_TAU_SUPPORT = 0.5
DEFAULT_Z0 = 2.0
DEFAULT_EPS = 1e-12
BOOTSTRAP_B = 20


# ============================================================
# Configuration
# ============================================================

@dataclass
class Day5Config:
    """Day5 실험 설정"""
    mode: str = "control50"  # 'control50', 'control250', 'compare'
    day3_run_id: str = ""
    dataset_version: str = "cartpole_ood_v1"
    dataset_path: str = ""
    track: str = "standardized"
    tau_support: float = DEFAULT_TAU_SUPPORT
    z0: float = DEFAULT_Z0
    eps: float = DEFAULT_EPS
    bootstrap_B: int = BOOTSTRAP_B
    threshold: float = 0.05
    seed: int = 0
    note: str = "day5"
    # Compare mode 전용
    ctrl50_run_id: str = ""
    ctrl250_run_id: str = ""
    treat_run_id: str = ""


# ============================================================
# Helper Functions
# ============================================================

def generate_run_id(note: str = "") -> str:
    """run_id 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_part = f"_{note}" if note else ""
    return f"{timestamp}_nogit{note_part}"


def compute_file_hash(filepath: Path) -> str:
    """파일의 SHA256 해시 계산"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_code_snapshot(results_dir: Path, source_files: List[Path]) -> Dict[str, str]:
    """코드 스냅샷 생성"""
    snapshot_dir = results_dir / 'code_snapshot'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    code_hash = {}
    for src_file in source_files:
        if src_file.exists():
            dst_file = snapshot_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            code_hash[src_file.name] = compute_file_hash(dst_file)
    
    hash_path = results_dir / 'code_hash.json'
    with open(hash_path, 'w', encoding='utf-8') as f:
        json.dump(code_hash, f, indent=2)
    
    return code_hash


def safe_float(val) -> Optional[float]:
    """numpy scalar를 Python float로 안전하게 변환"""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    return val


# ============================================================
# Day5 Control Runner
# ============================================================

class Day5ControlRunner:
    """Phase 3.5 Day5 Control Runner (Control-50 또는 Control-250)"""
    
    def __init__(self, config: Day5Config):
        self.config = config
        self.run_id = generate_run_id(config.note)
        
        self.project_root = _PROJECT_ROOT
        self.day3_results_dir: Optional[Path] = None
        self.results_dir: Optional[Path] = None
        
        # Loaded data
        self.day3_manifest: Optional[Dict] = None
        self.day3_selection: Optional[Dict] = None
        self.day3_metrics: Optional[Dict] = None
        self.feature_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        self.dx_key_used: Optional[str] = None  # 실측 dx key (manifest용)
        
        # Arrays
        self.z_before: Optional[np.ndarray] = None
        self.selected_mask: Optional[np.ndarray] = None
        self.teacher_support: Optional[np.ndarray] = None
        self.oracle_support: Optional[np.ndarray] = None
        
        # Day5 results
        self.z_after: Optional[np.ndarray] = None
        self.coef_mean_after: Optional[np.ndarray] = None
        self.coef_std_after: Optional[np.ndarray] = None
        self.inc_prob_after: Optional[np.ndarray] = None
        
        # Oracle trace
        self.oracle_trace = {
            'oracle_inputs_loaded': False,
            'oracle_used_in_selection': False,
            'oracle_used_in_augmentation': False,
            'oracle_used_in_training': False,
            'oracle_used_in_metrics': False
        }
    
    def _banner(self, msg: str):
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"{'='*60}")
    
    def _section(self, step: str, title: str):
        print(f"\n[{step}] {title}")
        print("-" * 50)
    
    def run(self) -> Dict[str, Any]:
        """Day5 Control 파이프라인 실행"""
        mode = self.config.mode
        self._banner(f"Phase 3.5 Day5 Control Runner ({mode}): {self.run_id}")
        
        # Step 1: Day3 결과 로드
        self._section("1/6", "Loading Day3 Results")
        self._load_day3_results()
        
        # Step 2: Training 데이터 로드
        self._section("2/6", "Loading Training Data")
        x_train, dx_train, u_train, T = self._load_training_data()
        
        # Step 3: 데이터 준비 (mode에 따라 다름)
        self._section("3/6", f"Preparing Data ({mode})")
        x_data, dx_data, u_data = self._prepare_data(x_train, dx_train, u_train)
        
        # Step 4: ESINDy로 z_after 계산
        self._section("4/6", "Computing z_after (ESINDy Bootstrap)")
        self._compute_z_after(x_data, dx_data, u_data, T)
        
        # Step 5: Training Effect 지표 계산
        self._section("5/6", "Computing Training Effect Metrics")
        metrics = self._compute_training_effect()
        
        # Step 6: 산출물 저장
        self._section("6/6", "Saving Artifacts")
        code_hash = self._create_code_snapshot()
        self._save_artifacts(metrics, code_hash)
        
        self._banner(f"✅ Day5 {mode} Complete: {self.run_id}")
        print(f"  Results: {self.results_dir}")
        
        return {'run_id': self.run_id, 'metrics': metrics}
    
    def _load_day3_results(self):
        """Day3 결과 로드"""
        cfg = self.config
        
        self.day3_results_dir = (
            self.project_root / 'results' / cfg.dataset_version / 'phase35' /
            cfg.track / 'stable_core' / 'n10' / f'seed{cfg.seed}' / cfg.day3_run_id
        )
        
        if not self.day3_results_dir.exists():
            raise FileNotFoundError(f"Day3 results not found: {self.day3_results_dir}")
        
        # JSON 파일 로드
        with open(self.day3_results_dir / 'manifest.json', 'r') as f:
            self.day3_manifest = json.load(f)
        print(f"  ✅ Loaded manifest.json")
        
        with open(self.day3_results_dir / 'selection.json', 'r') as f:
            self.day3_selection = json.load(f)
        print(f"  ✅ Loaded selection.json")
        
        with open(self.day3_results_dir / 'metrics.json', 'r') as f:
            self.day3_metrics = json.load(f)
        print(f"  ✅ Loaded metrics.json")
        
        # Feature/Target names
        self.feature_names = self.day3_manifest['ssot']['feature_names']
        self.target_names = self.day3_manifest['ssot']['target_names']
        print(f"  Features: {len(self.feature_names)}, Targets: {len(self.target_names)}")
        
        # Numpy arrays 로드
        self.z_before = np.load(self.day3_results_dir / 'z_before.npy')
        self.selected_mask = np.load(self.day3_results_dir / 'selected_support_pre_aug.npy')
        self.teacher_support = np.load(self.day3_results_dir / 'teacher_support.npy')
        print(f"  ✅ Loaded numpy arrays (z_before, selected_mask, teacher_support)")
        
        # Oracle support 로드
        self._load_oracle_support()
        
        # Results 디렉토리 생성
        self.results_dir = (
            self.project_root / 'results' / cfg.dataset_version / 'phase35' /
            cfg.track / 'stable_core' / 'n10' / f'seed{cfg.seed}' / self.run_id
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created results dir: {self.results_dir}")
    
    def _load_oracle_support(self):
        """Oracle support 로드"""
        cfg = self.config
        oracle_run_id = self.day3_manifest['gate1_artifacts']['oracle_run_id']
        oracle_dir = (
            self.project_root / 'results' / cfg.dataset_version / 'gate1' /
            cfg.track / 'esindy' / 'n50' / f'seed{cfg.seed}' / oracle_run_id
        )
        
        oracle_inc_prob_path = oracle_dir / 'inclusion_probability.csv'
        if oracle_inc_prob_path.exists():
            df_oracle = pd.read_csv(oracle_inc_prob_path)
            n_features = len(self.feature_names)
            n_targets = len(self.target_names)
            
            feat_idx = {f: i for i, f in enumerate(self.feature_names)}
            tgt_idx = {t: j for j, t in enumerate(self.target_names)}
            
            oracle_inc_prob = np.zeros((n_features, n_targets))
            for _, row in df_oracle.iterrows():
                feat = row['term_name']
                i = feat_idx.get(feat, -1)
                if i >= 0:
                    for tgt in self.target_names:
                        j = tgt_idx.get(tgt, -1)
                        if j >= 0 and tgt in df_oracle.columns:
                            oracle_inc_prob[i, j] = row[tgt]
            
            self.oracle_support = oracle_inc_prob >= cfg.tau_support
            self.oracle_trace['oracle_inputs_loaded'] = True
            print(f"  ✅ Loaded Oracle support (n_active={self.oracle_support.sum()})")
        else:
            print(f"  ⚠️ Oracle file not found, using teacher_support")
            self.oracle_support = self.teacher_support.copy()
    
    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Training 데이터 로드"""
        cfg = self.config
        
        if cfg.dataset_path:
            dataset_path = Path(cfg.dataset_path)
        else:
            dataset_path = (
                self.project_root / 'data' / 'cartpole' / cfg.dataset_version /
                'dataset.npz'
            )
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        data = np.load(dataset_path)
        x_train = data['train_x']
        
        # dx key 결정 (track에 따라) - SSOT, fail-fast (fallback 금지)
        if cfg.track == 'standardized':
            dx_key = 'train_dx_savgol'
            if dx_key not in data.files:
                raise RuntimeError(
                    f"standardized track requires '{dx_key}' in dataset. "
                    f"Available keys: {list(data.files)}"
                )
        else:
            dx_key = 'train_dx'
            if dx_key not in data.files:
                raise RuntimeError(
                    f"Required dx key '{dx_key}' not found. "
                    f"Available keys: {list(data.files)}"
                )
        
        dx_train = data[dx_key]
        self.dx_key_used = dx_key  # 실측값 저장 (manifest용)
        
        u_train = data['train_u']
        
        # T (timesteps per trajectory)
        T = x_train.shape[1]
        
        print(f"  x_train: {x_train.shape}")
        print(f"  dx_train: {dx_train.shape} (key={dx_key})")
        print(f"  u_train: {u_train.shape}")
        print(f"  T (timesteps): {T}")
        
        return x_train, dx_train, u_train, T
    
    def _prepare_data(
        self, 
        x_train: np.ndarray, 
        dx_train: np.ndarray, 
        u_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        mode에 따른 데이터 준비
        
        - control50: 원본 50 그대로 사용
        - control250: 원본 50을 복제하여 250개 (perturb 없음)
        """
        mode = self.config.mode
        n_original = x_train.shape[0]
        
        if mode == 'control50':
            # 원본 그대로
            print(f"  Control-50: Using original {n_original} trajectories (no augmentation)")
            return x_train, dx_train, u_train
        
        elif mode == 'control250':
            # 원본 복제로 250개 (Day4 Treatment와 동일 크기)
            target_n = 250
            n_repeats = target_n // n_original
            remainder = target_n % n_original
            
            x_list = [x_train] * n_repeats
            dx_list = [dx_train] * n_repeats
            u_list = [u_train] * n_repeats
            
            if remainder > 0:
                x_list.append(x_train[:remainder])
                dx_list.append(dx_train[:remainder])
                u_list.append(u_train[:remainder])
            
            x_resampled = np.concatenate(x_list, axis=0)
            dx_resampled = np.concatenate(dx_list, axis=0)
            u_resampled = np.concatenate(u_list, axis=0)
            
            print(f"  Control-250: Resampled {n_original} → {x_resampled.shape[0]} trajectories")
            print(f"    (replicate {n_repeats}x + {remainder} remainder, NO perturbation)")
            
            # 리샘플 결과 저장
            resample_dir = self.results_dir / 'resampled_data'
            resample_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                resample_dir / 'resampled_trajectories.npz',
                x_resampled=x_resampled,
                dx_resampled=dx_resampled,
                u_resampled=u_resampled,
                n_original=n_original,
                n_resampled=x_resampled.shape[0]
            )
            print(f"  ✅ Saved: resampled_data/resampled_trajectories.npz")
            
            return x_resampled, dx_resampled, u_resampled
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_z_after(
        self, 
        x_data: np.ndarray, 
        dx_data: np.ndarray, 
        u_data: np.ndarray,
        T: int
    ):
        """ESINDy Bootstrap으로 z_after 계산"""
        cfg = self.config
        n_traj = x_data.shape[0]
        
        print(f"  Building library (gate0_min)...")
        
        # SINDy Library 구성
        library = SINDyLibrary(config='gate0_min')
        
        # Feature matrix 생성
        Theta = library.fit_transform(x_data, u_data)
        
        # dx를 2D로 flatten
        dx_flat = dx_data.reshape(-1, dx_data.shape[-1])
        
        print(f"  Theta: {Theta.shape}")
        print(f"  dx_flat: {dx_flat.shape}")
        
        # Column scaling
        scaler = ColumnScaler()
        Theta_scaled = scaler.fit_transform(Theta)
        
        # ESINDy Ensemble
        print(f"  ESINDy Bootstrap B={cfg.bootstrap_B}, seed={cfg.seed}")
        ensemble = ESINDyEnsemble(
            n_bootstrap=cfg.bootstrap_B,
            threshold=cfg.threshold,
            random_state=cfg.seed
        )
        
        ensemble.fit(
            Theta_scaled, dx_flat,
            n_trajectories=n_traj,
            T=T,
            scaler=scaler,
            target_scale=None
        )
        
        # 결과 추출
        self.coef_mean_after = ensemble.coefficients_mean_
        self.coef_std_after = ensemble.coefficients_std_
        self.inc_prob_after = ensemble.inclusion_probability_
        
        # z_after 계산 (SSOT: abs(mean)/(std+eps))
        self.z_after = np.abs(self.coef_mean_after) / (self.coef_std_after + cfg.eps)
        
        print(f"  coef_mean_after: {self.coef_mean_after.shape}")
        print(f"  coef_std_after: {self.coef_std_after.shape}")
        print(f"  inc_prob_after: {self.inc_prob_after.shape}")
        print(f"  z_after range: [{self.z_after.min():.2f}, {self.z_after.max():.2f}]")
        
        self.oracle_trace['oracle_used_in_training'] = False
    
    def _compute_training_effect(self) -> Dict[str, Any]:
        """Training Effect 지표 계산 (Day4와 동일 로직)"""
        cfg = self.config
        
        if self.z_after is None or self.z_before is None:
            raise RuntimeError("z_before or z_after not computed")
        
        self.oracle_trace['oracle_used_in_metrics'] = True
        
        teacher_active = self.teacher_support.astype(bool)
        n_teacher_active = teacher_active.sum()
        oracle_active = self.oracle_support.astype(bool) if self.oracle_support is not None else teacher_active
        
        print(f"  Teacher active pairs: {n_teacher_active}")
        
        # Delta-z 계산
        delta_z = self.z_after - self.z_before
        
        # delta_z_details 생성
        n_targets = len(self.target_names)
        delta_z_details = []
        
        for i in range(len(self.feature_names)):
            for j in range(len(self.target_names)):
                if teacher_active[i, j]:
                    delta_z_details.append({
                        'feature': self.feature_names[i],
                        'target': self.target_names[j],
                        'feature_idx': i,
                        'target_idx': j,
                        'global_idx': i * n_targets + j,
                        'z_before': float(self.z_before[i, j]),
                        'z_after': float(self.z_after[i, j]),
                        'delta_z': float(delta_z[i, j])
                    })
        
        print(f"  delta_z_details: {len(delta_z_details)} pairs")
        
        # Fragile oracle-true
        fragile_mask = teacher_active & (self.z_before < cfg.z0)
        fragile_oracle_true = fragile_mask & oracle_active
        n_fragile_oracle_true = fragile_oracle_true.sum()
        
        print(f"  Fragile oracle-true: {n_fragile_oracle_true} pairs")
        
        # 지표 1: promotion_rate_delta
        if n_fragile_oracle_true > 0:
            improved = fragile_oracle_true & (delta_z > 0)
            promotion_rate_delta = float(improved.sum()) / n_fragile_oracle_true
            promotion_rate_delta_reason = None
        else:
            promotion_rate_delta = None
            promotion_rate_delta_reason = "denominator_zero"
        
        # 지표 2: promotion_to_stable_rate
        if n_fragile_oracle_true > 0:
            promoted = fragile_oracle_true & (self.z_before < cfg.z0) & (self.z_after >= cfg.z0)
            promotion_to_stable_rate = float(promoted.sum()) / n_fragile_oracle_true
            promotion_to_stable_rate_reason = None
        else:
            promotion_to_stable_rate = None
            promotion_to_stable_rate_reason = "denominator_zero"
        
        # 지표 3: spurious_reentry_rate
        spurious_set = teacher_active & (~oracle_active)
        removed_at_selection = spurious_set & (~self.selected_mask)
        n_removed_at_selection = removed_at_selection.sum()
        
        # support_after와 stable_after 정의 (inc_prob_after 활용)
        support_after = self.inc_prob_after >= cfg.tau_support
        stable_after = support_after & (self.z_after >= cfg.z0)
        
        # z 기반 (기존, 참고용)
        final_raw_z = self.z_after >= cfg.z0
        
        if n_removed_at_selection > 0:
            # reentry_to_support: support_after로 재유입
            reentered_support = removed_at_selection & support_after
            reentry_to_support_rate = float(reentered_support.sum()) / n_removed_at_selection
            n_reentered_support = int(reentered_support.sum())
            
            # reentry_to_stable: stable_after로 재유입 (더 엄격)
            reentered_stable = removed_at_selection & stable_after
            reentry_to_stable_rate = float(reentered_stable.sum()) / n_removed_at_selection
            n_reentered_stable = int(reentered_stable.sum())
            
            # 기존 z 기반 (비교용)
            reentered_z = removed_at_selection & final_raw_z
            spurious_reentry_rate = float(reentered_z.sum()) / n_removed_at_selection
            spurious_reentry_rate_reason = None
            n_reentered = int(reentered_z.sum())
            
            reentry_details = []
            for i in range(len(self.feature_names)):
                for j in range(len(self.target_names)):
                    if removed_at_selection[i, j] and (support_after[i, j] or final_raw_z[i, j]):
                        reentry_details.append({
                            'feature': self.feature_names[i],
                            'target': self.target_names[j],
                            'feature_idx': i,
                            'target_idx': j,
                            'z_after': float(self.z_after[i, j]),
                            'inc_prob_after': float(self.inc_prob_after[i, j]),
                            'in_support_after': bool(support_after[i, j]),
                            'in_stable_after': bool(stable_after[i, j])
                        })
        else:
            spurious_reentry_rate = None
            spurious_reentry_rate_reason = "denominator_zero"
            reentry_to_support_rate = None
            reentry_to_stable_rate = None
            n_reentered = 0
            n_reentered_support = 0
            n_reentered_stable = 0
            reentry_details = []
        
        # 지표 4: delta_z 통계
        teacher_delta_z = delta_z[teacher_active]
        delta_z_median = float(np.median(teacher_delta_z))
        delta_z_mean = float(np.mean(teacher_delta_z))
        
        # Trimmed mean (상위/하위 10% 제외)
        sorted_dz = np.sort(teacher_delta_z)
        trim_n = max(1, len(sorted_dz) // 10)
        if len(sorted_dz) > 2 * trim_n:
            delta_z_trimmed_mean = float(np.mean(sorted_dz[trim_n:-trim_n]))
        else:
            delta_z_trimmed_mean = delta_z_median
        
        # 분위수
        delta_z_q10 = float(np.percentile(teacher_delta_z, 10))
        delta_z_q25 = float(np.percentile(teacher_delta_z, 25))
        delta_z_q75 = float(np.percentile(teacher_delta_z, 75))
        delta_z_q90 = float(np.percentile(teacher_delta_z, 90))
        
        # fragile-pool 분리 통계
        stable_core_mask = teacher_active & (self.z_before >= cfg.z0)
        fragile_pool_mask = teacher_active & (self.z_before < cfg.z0)
        
        n_stable_core = stable_core_mask.sum()
        n_fragile_pool = fragile_pool_mask.sum()
        
        if n_fragile_pool > 0:
            fragile_delta_z = delta_z[fragile_pool_mask]
            fragile_delta_z_median = float(np.median(fragile_delta_z))
            fragile_improved_count = int((fragile_delta_z > 0).sum())
        else:
            fragile_delta_z_median = None
            fragile_improved_count = 0
        
        if n_stable_core > 0:
            stable_delta_z = delta_z[stable_core_mask]
            stable_delta_z_median = float(np.median(stable_delta_z))
        else:
            stable_delta_z_median = None
        
        # Dynamics targets 분리 통계
        dynamics_target_indices = [1, 3]  # x_ddot, theta_ddot
        
        # Dynamics targets mask
        dynamics_mask = np.zeros_like(teacher_active, dtype=bool)
        for ti in dynamics_target_indices:
            if ti < delta_z.shape[1]:
                dynamics_mask[:, ti] = teacher_active[:, ti]
        
        n_dynamics = dynamics_mask.sum()
        if n_dynamics > 0:
            dynamics_delta_z = delta_z[dynamics_mask]
            dynamics_delta_z_median = float(np.median(dynamics_delta_z))
            dynamics_improved_count = int((dynamics_delta_z > 0).sum())
            dynamics_improved_rate = dynamics_improved_count / n_dynamics
        else:
            dynamics_delta_z_median = None
            dynamics_improved_count = 0
            dynamics_improved_rate = None
        
        # fragile∩dynamics
        fragile_dynamics_mask = fragile_pool_mask & dynamics_mask
        n_fragile_dynamics = fragile_dynamics_mask.sum()
        
        if n_fragile_dynamics > 0:
            fragile_dynamics_delta_z = delta_z[fragile_dynamics_mask]
            fragile_dynamics_median = float(np.median(fragile_dynamics_delta_z))
            fragile_dynamics_improved = int((fragile_dynamics_delta_z > 0).sum())
        else:
            fragile_dynamics_median = None
            fragile_dynamics_improved = 0
        
        # stable∩dynamics
        stable_dynamics_mask = stable_core_mask & dynamics_mask
        n_stable_dynamics = stable_dynamics_mask.sum()
        
        if n_stable_dynamics > 0:
            stable_dynamics_delta_z = delta_z[stable_dynamics_mask]
            stable_dynamics_median = float(np.median(stable_dynamics_delta_z))
        else:
            stable_dynamics_median = None
        
        print(f"  promotion_rate_delta: {promotion_rate_delta}")
        print(f"  promotion_to_stable_rate: {promotion_to_stable_rate}")
        print(f"  spurious_reentry_rate (z-based): {spurious_reentry_rate}")
        print(f"  reentry_to_support_rate: {reentry_to_support_rate}")
        print(f"  reentry_to_stable_rate: {reentry_to_stable_rate}")
        print(f"  delta_z_median: {delta_z_median:.4f}")
        print(f"  delta_z_trimmed_mean: {delta_z_trimmed_mean:.4f}")
        print(f"  [fragile-pool] n={n_fragile_pool}, median={fragile_delta_z_median}")
        print(f"  [stable-core] n={n_stable_core}, median={stable_delta_z_median}")
        print(f"  [fragile∩dynamics] n={n_fragile_dynamics}, median={fragile_dynamics_median}")
        print(f"  [stable∩dynamics] n={n_stable_dynamics}, median={stable_dynamics_median}")
        
        return {
            'stage': f'{self.config.mode}_applied',
            'mode': self.config.mode,
            'augmentation_applied': False,
            'training_effect_available': True,
            
            'primary_metrics': {
                # Dynamics 중심
                'dynamics_delta_z_median': dynamics_delta_z_median,
                'dynamics_improved_rate': dynamics_improved_rate,
                'fragile_dynamics_median': fragile_dynamics_median,
                
                # Reentry rates
                'spurious_reentry_rate': spurious_reentry_rate,
                'reentry_to_support_rate': reentry_to_support_rate,
                'reentry_to_stable_rate': reentry_to_stable_rate,
                'spurious_reentry_rate_reason': spurious_reentry_rate_reason,
                
                # Promotion rates
                'promotion_rate_delta': promotion_rate_delta,
                'promotion_rate_delta_reason': promotion_rate_delta_reason,
                'promotion_to_stable_rate': promotion_to_stable_rate,
                'promotion_to_stable_rate_reason': promotion_to_stable_rate_reason,
                
                # Overall statistics
                'delta_z_median': delta_z_median,
                'delta_z_trimmed_mean': delta_z_trimmed_mean,
                'delta_z_mean': delta_z_mean,
                'delta_z_quantiles': {
                    'q10': delta_z_q10,
                    'q25': delta_z_q25,
                    'q75': delta_z_q75,
                    'q90': delta_z_q90
                }
            },
            
            'counts': {
                'n_teacher_active': int(n_teacher_active),
                'n_fragile_oracle_true': int(n_fragile_oracle_true),
                'n_removed_at_selection': int(n_removed_at_selection),
                'n_spurious_reentered_z': n_reentered,
                'n_spurious_reentered_support': n_reentered_support,
                'n_spurious_reentered_stable': n_reentered_stable,
                'n_stable_core': int(n_stable_core),
                'n_fragile_pool': int(n_fragile_pool),
                'n_dynamics': int(n_dynamics),
                'n_fragile_dynamics': int(n_fragile_dynamics),
                'n_stable_dynamics': int(n_stable_dynamics)
            },
            
            'fragile_pool_stats': {
                'n': int(n_fragile_pool),
                'delta_z_median': fragile_delta_z_median,
                'n_improved': fragile_improved_count
            },
            
            'stable_core_stats': {
                'n': int(n_stable_core),
                'delta_z_median': stable_delta_z_median
            },
            
            'fragile_dynamics_stats': {
                'n': int(n_fragile_dynamics),
                'delta_z_median': fragile_dynamics_median,
                'n_improved': fragile_dynamics_improved
            },
            
            'stable_dynamics_stats': {
                'n': int(n_stable_dynamics),
                'delta_z_median': stable_dynamics_median
            },
            
            '_structure_eval': {
                'delta_z_details': delta_z_details,
                'reentry_details': reentry_details
            }
        }
    
    def _create_code_snapshot(self) -> Dict[str, str]:
        """Code Snapshot 생성"""
        source_files = [
            Path(__file__),
            self.project_root / 'src' / 'sindy' / 'selection.py',
            self.project_root / 'src' / 'sindy' / 'core_mining.py',
            self.project_root / 'src' / 'evaluation' / 'structure_eval.py',
            self.project_root / 'src' / 'sindy' / 'esindy.py',
            self.project_root / 'src' / 'sindy' / 'library.py',
        ]
        
        code_hash = create_code_snapshot(self.results_dir, source_files)
        
        print(f"  ✅ Created code_snapshot/ with {len(code_hash)} files")
        return code_hash
    
    def _save_artifacts(self, metrics: Dict, code_hash: Dict[str, str]):
        """산출물 저장 (Modern Schema 적용)"""
        cfg = self.config
        
        # teacher_support hash 계산
        teacher_support_path = self.day3_results_dir / 'teacher_support.npy'
        if teacher_support_path.exists():
            teacher_support_sha256 = compute_file_sha256(teacher_support_path)
        else:
            raise RuntimeError(f"teacher_support.npy not found at {teacher_support_path}")
        
        # 1. manifest.json (Modern Schema)
        manifest = {
            'phase': 'phase35',
            'day': 5,
            'run_id': self.run_id,
            'day3_run_id': cfg.day3_run_id,
            'mode': cfg.mode,
            'created_at': datetime.now().isoformat(),
            'dataset_version': cfg.dataset_version,
            'track': cfg.track,
            'method': 'stable_core',
            'n_train': 10,
            'seed': cfg.seed,
            
            'gate1_artifacts': self.day3_manifest.get('gate1_artifacts', {}),
            
            'hyperparameters': {
                'tau_support': cfg.tau_support,
                'z0': cfg.z0,
                'eps': cfg.eps,
                'bootstrap_B': cfg.bootstrap_B,
                'threshold': cfg.threshold
            },
            
            # === MODERNIZE FIELDS ===
            'control_equivalence': get_control_equivalence(bootstrap_B=cfg.bootstrap_B),
            'teacher_support_sha256': teacher_support_sha256,
            'teacher_support_source': f"{cfg.day3_run_id}/teacher_support.npy",
            'preflight_qc': {
                'dx_equivalence': {
                    'dx_key_used': self.dx_key_used,  # 실측값
                    'note': f'Loaded from dataset key: {self.dx_key_used}'
                }
            },
            'code_hash': code_hash,
            
            'data_config': {
                'n_trajectories': 50 if cfg.mode == 'control50' else 250,
                'augmentation': 'none' if cfg.mode == 'control50' else 'replicate_only',
                'perturbation': False
            },
            
            'z_after_source': 'bootstrap',
            'z_formula': 'abs(mean)/(std+eps)',
            'oracle_usage': 'evaluation_only',
            'seed_rule': 'seed_b = base_seed + b',
            'resample_unit': 'trajectory',
            
            'ssot': self.day3_manifest.get('ssot', {}),
            
            'definitions': {
                'z_score': '|mean| / (std + eps)',
                'support': f'inc_prob >= {cfg.tau_support}',
                'stable_core': f'support AND z >= {cfg.z0}',
                'fragile_pool': f'support AND z < {cfg.z0}'
            },
            
            'oracle_trace': self.oracle_trace
        }
        
        with open(self.results_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: manifest.json (with modern schema)")
        
        # 2. metrics.json
        metrics_clean = {k: v for k, v in metrics.items() if not k.startswith('_')}
        with open(self.results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_clean, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: metrics.json")
        
        # 3. structure_eval.json
        structure_eval = {
            'mode': cfg.mode,
            'primary_metrics': metrics['primary_metrics'],
            'counts': metrics['counts'],
            'fragile_pool_stats': metrics['fragile_pool_stats'],
            'stable_core_stats': metrics['stable_core_stats'],
            'fragile_dynamics_stats': metrics['fragile_dynamics_stats'],
            'stable_dynamics_stats': metrics['stable_dynamics_stats'],
            'details': {
                'delta_z_details': metrics['_structure_eval']['delta_z_details'],
                'reentry_details': metrics['_structure_eval']['reentry_details']
            }
        }
        
        with open(self.results_dir / 'structure_eval.json', 'w', encoding='utf-8') as f:
            json.dump(structure_eval, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: structure_eval.json")
        
        # 4. selection.json (Day3에서 복사)
        with open(self.results_dir / 'selection.json', 'w', encoding='utf-8') as f:
            json.dump(self.day3_selection, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: selection.json")
        
        # 5. numpy arrays
        np.save(self.results_dir / 'z_before.npy', self.z_before)
        np.save(self.results_dir / 'z_after.npy', self.z_after)
        np.save(self.results_dir / 'coef_mean_after.npy', self.coef_mean_after)
        np.save(self.results_dir / 'coef_std_after.npy', self.coef_std_after)
        np.save(self.results_dir / 'inc_prob_after.npy', self.inc_prob_after)
        print(f"  ✅ Saved: z_before, z_after, coef_mean_after, coef_std_after, inc_prob_after (.npy)")


# ============================================================
# Day5 Compare Runner
# ============================================================

class Day5CompareRunner:
    """Phase 3.5 Day5 4-way Comparison Runner"""
    
    def __init__(self, config: Day5Config):
        self.config = config
        self.run_id = generate_run_id(config.note)
        
        self.project_root = _PROJECT_ROOT
        
        # Loaded data
        self.day3_manifest: Optional[Dict] = None
        self.feature_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        
        # z arrays
        self.z_before: Optional[np.ndarray] = None
        self.z_ctrl50: Optional[np.ndarray] = None
        self.z_ctrl250: Optional[np.ndarray] = None
        self.z_treat: Optional[np.ndarray] = None
        
        # metrics for reentry comparison
        self.metrics_ctrl50: Optional[Dict] = None
        self.metrics_ctrl250: Optional[Dict] = None
        self.metrics_treat: Optional[Dict] = None
        
        self.teacher_support: Optional[np.ndarray] = None
        self.oracle_support: Optional[np.ndarray] = None
        
        self.results_dir: Optional[Path] = None
    
    def _banner(self, msg: str):
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"{'='*60}")
    
    def _section(self, step: str, title: str):
        print(f"\n[{step}] {title}")
        print("-" * 50)
    
    def run(self) -> Dict[str, Any]:
        """4-way 비교 실행"""
        self._banner(f"Phase 3.5 Day5 4-way Compare: {self.run_id}")
        
        # Step 1: 모든 결과 로드
        self._section("1/3", "Loading All Results")
        self._load_all_results()
        
        # Step 2: Control Equivalence 검증
        self._section("2/3", "Validating Control Equivalence")
        self._validate_control_equivalence()
        
        # Step 3: 4-way 비교표 생성
        self._section("3/3", "Computing 4-way Comparison")
        comparison = self._compute_comparison()
        
        # 저장
        self._save_comparison(comparison)
        
        self._banner(f"✅ Day5 4-way Compare Complete: {self.run_id}")
        print(f"  Results: {self.results_dir}")
        
        return {'run_id': self.run_id, 'comparison': comparison}
    
    def _get_results_dir(self, run_id: str) -> Path:
        """run_id로 결과 디렉토리 경로 생성"""
        cfg = self.config
        return (
            self.project_root / 'results' / cfg.dataset_version / 'phase35' /
            cfg.track / 'stable_core' / 'n10' / f'seed{cfg.seed}' / run_id
        )
    
    def _load_all_results(self):
        """모든 결과 로드"""
        cfg = self.config
        
        # Day3 (baseline)
        day3_dir = self._get_results_dir(cfg.day3_run_id)
        if not day3_dir.exists():
            raise FileNotFoundError(f"Day3 results not found: {day3_dir}")
        
        with open(day3_dir / 'manifest.json', 'r') as f:
            self.day3_manifest = json.load(f)
        
        self.feature_names = self.day3_manifest['ssot']['feature_names']
        self.target_names = self.day3_manifest['ssot']['target_names']
        
        self.z_before = np.load(day3_dir / 'z_before.npy')
        self.teacher_support = np.load(day3_dir / 'teacher_support.npy')
        print(f"  ✅ Day3 (baseline): {cfg.day3_run_id}")
        
        # Oracle support
        oracle_run_id = self.day3_manifest['gate1_artifacts']['oracle_run_id']
        oracle_dir = (
            self.project_root / 'results' / cfg.dataset_version / 'gate1' /
            cfg.track / 'esindy' / 'n50' / f'seed{cfg.seed}' / oracle_run_id
        )
        oracle_inc_prob_path = oracle_dir / 'inclusion_probability.csv'
        if oracle_inc_prob_path.exists():
            df_oracle = pd.read_csv(oracle_inc_prob_path)
            n_features = len(self.feature_names)
            n_targets = len(self.target_names)
            
            feat_idx = {f: i for i, f in enumerate(self.feature_names)}
            tgt_idx = {t: j for j, t in enumerate(self.target_names)}
            
            oracle_inc_prob = np.zeros((n_features, n_targets))
            for _, row in df_oracle.iterrows():
                feat = row['term_name']
                i = feat_idx.get(feat, -1)
                if i >= 0:
                    for tgt in self.target_names:
                        j = tgt_idx.get(tgt, -1)
                        if j >= 0 and tgt in df_oracle.columns:
                            oracle_inc_prob[i, j] = row[tgt]
            
            self.oracle_support = oracle_inc_prob >= cfg.tau_support
            print(f"  ✅ Oracle support loaded (n_active={self.oracle_support.sum()})")
        else:
            self.oracle_support = self.teacher_support.copy()
            print(f"  ⚠️ Oracle not found, using teacher_support")
        
        # Control-50
        ctrl50_dir = self._get_results_dir(cfg.ctrl50_run_id)
        if not ctrl50_dir.exists():
            raise FileNotFoundError(f"Control-50 results not found: {ctrl50_dir}")
        self.z_ctrl50 = np.load(ctrl50_dir / 'z_after.npy')
        with open(ctrl50_dir / 'metrics.json', 'r') as f:
            self.metrics_ctrl50 = json.load(f)
        print(f"  ✅ Control-50: {cfg.ctrl50_run_id}")
        
        # Control-250
        ctrl250_dir = self._get_results_dir(cfg.ctrl250_run_id)
        if not ctrl250_dir.exists():
            raise FileNotFoundError(f"Control-250 results not found: {ctrl250_dir}")
        self.z_ctrl250 = np.load(ctrl250_dir / 'z_after.npy')
        with open(ctrl250_dir / 'metrics.json', 'r') as f:
            self.metrics_ctrl250 = json.load(f)
        print(f"  ✅ Control-250: {cfg.ctrl250_run_id}")
        
        # Treatment (Day4)
        treat_dir = self._get_results_dir(cfg.treat_run_id)
        if not treat_dir.exists():
            raise FileNotFoundError(f"Treatment results not found: {treat_dir}")
        self.z_treat = np.load(treat_dir / 'z_after.npy')
        with open(treat_dir / 'metrics.json', 'r') as f:
            self.metrics_treat = json.load(f)
        print(f"  ✅ Treatment: {cfg.treat_run_id}")
        
        # Results 디렉토리 생성
        self.results_dir = self._get_results_dir(self.run_id)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created results dir: {self.results_dir}")
    
    def _validate_control_equivalence(self):
        """Control Equivalence 검증"""
        cfg = self.config
        
        # 각 run의 manifest에서 control_equivalence 확인
        runs_to_check = [
            ('ctrl50', cfg.ctrl50_run_id),
            ('ctrl250', cfg.ctrl250_run_id),
            ('treat', cfg.treat_run_id)
        ]
        
        reference = get_control_equivalence(bootstrap_B=cfg.bootstrap_B)
        all_match = True
        
        for label, run_id in runs_to_check:
            run_dir = self._get_results_dir(run_id)
            manifest_path = run_dir / 'manifest.json'
            
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                run_ce = manifest.get('control_equivalence', {})
                
                # 핵심 필드 비교
                keys_to_check = ['library', 'threshold', 'bootstrap_B', 'tau_support', 'z0']
                mismatches = []
                for key in keys_to_check:
                    if run_ce.get(key) != reference.get(key):
                        mismatches.append(f"{key}: {run_ce.get(key)} != {reference.get(key)}")
                
                if mismatches:
                    print(f"  ⚠️ {label} ({run_id}): control_equivalence mismatch")
                    for m in mismatches:
                        print(f"      {m}")
                    all_match = False
                else:
                    print(f"  ✅ {label}: control_equivalence OK")
            else:
                print(f"  ⚠️ {label}: manifest.json not found")
                all_match = False
        
        if all_match:
            print(f"\n  Control Equivalence: VERIFIED")
        else:
            print(f"\n  ⚠️ Control Equivalence: PARTIAL (check warnings)")
    
    def _compute_effect_stats(self, effect_array: np.ndarray, mask: np.ndarray) -> Dict:
        """효과 배열의 통계 계산"""
        if mask.sum() == 0:
            return {'n': 0, 'median': None, 'mean': None, 'std': None}
        
        values = effect_array[mask]
        n_positive = int((values > 0).sum())
        n_negative = int((values < 0).sum())
        n_zero = int((values == 0).sum())
        
        return {
            'n': int(len(values)),
            'median': safe_float(np.median(values)),
            'mean': safe_float(np.mean(values)),
            'std': safe_float(np.std(values)),
            'min': safe_float(np.min(values)),
            'max': safe_float(np.max(values)),
            'q25': safe_float(np.percentile(values, 25)),
            'q75': safe_float(np.percentile(values, 75)),
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_zero': n_zero,
            'zero_crossing': n_positive > 0 and n_negative > 0
        }
    
    def _compute_comparison(self) -> Dict:
        """4-way 비교 계산"""
        cfg = self.config
        
        teacher_active = self.teacher_support.astype(bool)
        oracle_active = self.oracle_support.astype(bool)
        
        # 효과 분해
        retrain_effect = self.z_ctrl50 - self.z_before
        volume_effect = self.z_ctrl250 - self.z_ctrl50
        aug_pure_effect = self.z_treat - self.z_ctrl250
        total_effect = self.z_treat - self.z_before
        
        # 마스크 정의
        fragile_mask = teacher_active & (self.z_before < cfg.z0)
        stable_mask = teacher_active & (self.z_before >= cfg.z0)
        
        # Dynamics targets
        dynamics_mask = np.zeros_like(teacher_active, dtype=bool)
        for ti in [1, 3]:  # x_ddot, theta_ddot
            if ti < teacher_active.shape[1]:
                dynamics_mask[:, ti] = teacher_active[:, ti]
        
        fragile_dynamics_mask = fragile_mask & dynamics_mask
        stable_dynamics_mask = stable_mask & dynamics_mask
        
        # 통계 계산
        # All teacher_active
        all_retrain = self._compute_effect_stats(retrain_effect, teacher_active)
        all_volume = self._compute_effect_stats(volume_effect, teacher_active)
        all_aug = self._compute_effect_stats(aug_pure_effect, teacher_active)
        all_total = self._compute_effect_stats(total_effect, teacher_active)
        
        # fragile∩dynamics (핵심)
        fragile_dyn_retrain = self._compute_effect_stats(retrain_effect, fragile_dynamics_mask)
        fragile_dyn_volume = self._compute_effect_stats(volume_effect, fragile_dynamics_mask)
        fragile_dyn_aug = self._compute_effect_stats(aug_pure_effect, fragile_dynamics_mask)
        fragile_dyn_total = self._compute_effect_stats(total_effect, fragile_dynamics_mask)
        
        # stable∩dynamics
        stable_dyn_retrain = self._compute_effect_stats(retrain_effect, stable_dynamics_mask)
        stable_dyn_volume = self._compute_effect_stats(volume_effect, stable_dynamics_mask)
        stable_dyn_aug = self._compute_effect_stats(aug_pure_effect, stable_dynamics_mask)
        stable_dyn_total = self._compute_effect_stats(total_effect, stable_dynamics_mask)
        
        # Identity check: total = retrain + volume + aug (element-wise)
        identity_error = total_effect - (retrain_effect + volume_effect + aug_pure_effect)
        identity_max_error = float(np.max(np.abs(identity_error[teacher_active])))
        identity_check = {
            'max_error': identity_max_error,
            'identity_holds': identity_max_error < 1e-10,
            'note': 'total_effect == retrain_effect + volume_effect + aug_pure_effect (element-wise)'
        }
        
        # z-level comparison
        z_comparison = {}
        for label, z_arr in [('before', self.z_before), ('ctrl50', self.z_ctrl50), 
                              ('ctrl250', self.z_ctrl250), ('treat', self.z_treat)]:
            fd_vals = z_arr[fragile_dynamics_mask]
            sd_vals = z_arr[stable_dynamics_mask]
            z_comparison[label] = {
                'fragile_dynamics': {
                    'median': safe_float(np.median(fd_vals)) if len(fd_vals) > 0 else None,
                    'mean': safe_float(np.mean(fd_vals)) if len(fd_vals) > 0 else None
                },
                'stable_dynamics': {
                    'median': safe_float(np.median(sd_vals)) if len(sd_vals) > 0 else None,
                    'mean': safe_float(np.mean(sd_vals)) if len(sd_vals) > 0 else None
                }
            }
        
        comparison = {
            'source_runs': {
                'day3_baseline': cfg.day3_run_id,
                'ctrl50': cfg.ctrl50_run_id,
                'ctrl250': cfg.ctrl250_run_id,
                'treatment': cfg.treat_run_id
            },
            
            'effect_definitions': {
                'retrain_effect': 'z_ctrl50 - z_before',
                'volume_effect': 'z_ctrl250 - z_ctrl50',
                'aug_pure_effect': 'z_treat - z_ctrl250',
                'total_effect': 'z_treat - z_before'
            },
            
            'identity_verification': identity_check,
            
            'fragile_dynamics_effects': {
                'description': 'fragile-pool ∩ dynamics targets (핵심 개선 대상)',
                'retrain_effect': fragile_dyn_retrain,
                'volume_effect': fragile_dyn_volume,
                'aug_pure_effect': fragile_dyn_aug,
                'total_effect': fragile_dyn_total
            },
            
            'stable_dynamics_effects': {
                'description': 'stable-core ∩ dynamics targets',
                'retrain_effect': stable_dyn_retrain,
                'volume_effect': stable_dyn_volume,
                'aug_pure_effect': stable_dyn_aug,
                'total_effect': stable_dyn_total
            },
            
            'all_teacher_active_effects': {
                'description': 'all teacher_active pairs',
                'n': int(teacher_active.sum()),
                'summary': {
                    'retrain_effect_median': all_retrain['median'],
                    'volume_effect_median': all_volume['median'],
                    'aug_pure_effect_median': all_aug['median'],
                    'total_effect_median': all_total['median']
                }
            },
            
            'z_level_comparison': z_comparison,
            
            'counts': {
                'n_teacher_active': int(teacher_active.sum()),
                'n_fragile_dynamics': int(fragile_dynamics_mask.sum()),
                'n_stable_dynamics': int(stable_dynamics_mask.sum())
            },
            
            'interpretation_guide': {
                'q1_fragile_source': 'fragile∩dynamics median이 negative인 효과 확인',
                'q2_reentry_source': 'reentry는 Control별 metrics.json에서 확인',
                'case_a': 'aug_pure > 0 → augmentation 기여 확인',
                'case_b': 'aug_pure ≈ 0, volume > 0 → resample과 동등',
                'case_c': 'aug_pure < 0 → augmentation 부작용',
                'case_d': 'retrain < 0 → 재학습 자체 문제',
                'median_non_additivity': 'median(A)+median(B) ≠ median(A+B); use identity_verification for element-wise check'
            }
        }
        
        # 해석 추가
        fragile_retrain = fragile_dyn_retrain['median']
        fragile_volume = fragile_dyn_volume['median']
        fragile_aug = fragile_dyn_aug['median']
        
        interpretation = []
        if fragile_retrain is not None:
            if fragile_retrain < 0:
                interpretation.append("CASE D: retrain_effect < 0 → 재학습 자체가 fragile에 부정적")
            else:
                interpretation.append(f"retrain_effect >= 0 ({fragile_retrain:.3f})")
        
        if fragile_aug is not None:
            if fragile_aug > 0:
                interpretation.append("CASE A: aug_pure_effect > 0 → augmentation 순수 기여 있음")
            elif abs(fragile_aug) < 0.1 and fragile_volume is not None and fragile_volume > 0:
                interpretation.append("CASE B: aug_pure ≈ 0, volume > 0 → resample과 동등한 효과")
            elif fragile_aug < 0:
                interpretation.append("CASE C: aug_pure_effect < 0 → augmentation 부작용")
        
        # aug_pure 분포 해석 추가
        if fragile_dyn_aug.get('zero_crossing'):
            interpretation.append(f"Note: aug_pure has zero-crossing distribution (n_pos={fragile_dyn_aug['n_positive']}, n_neg={fragile_dyn_aug['n_negative']})")
        
        comparison['interpretation'] = interpretation
        
        # Reentry 비교표
        def get_reentry_rates(metrics: Dict) -> Dict:
            pm = metrics.get('primary_metrics', {})
            return {
                'reentry_to_support_rate': pm.get('reentry_to_support_rate'),
                'reentry_to_stable_rate': pm.get('reentry_to_stable_rate'),
                'spurious_reentry_rate': pm.get('spurious_reentry_rate')
            }
        
        reentry_comparison = {
            'ctrl50': get_reentry_rates(self.metrics_ctrl50),
            'ctrl250': get_reentry_rates(self.metrics_ctrl250),
            'treatment': get_reentry_rates(self.metrics_treat),
            'note': 'If all three have similar reentry rates, reentry is structural (not augmentation-caused)'
        }
        comparison['reentry_comparison'] = reentry_comparison
        
        # Reentry 분석 콘솔 출력
        print(f"\n  Reentry Comparison (support rate):")
        print(f"    ctrl50:    {reentry_comparison['ctrl50']['reentry_to_support_rate']}")
        print(f"    ctrl250:   {reentry_comparison['ctrl250']['reentry_to_support_rate']}")
        print(f"    treatment: {reentry_comparison['treatment']['reentry_to_support_rate']}")
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("  4-WAY COMPARISON RESULTS (fragile∩dynamics)")
        print("="*60)
        print(f"  n = {fragile_dyn_retrain['n']}")
        print(f"  Retrain Effect median: {fragile_retrain}")
        print(f"  Volume Effect median: {fragile_volume}")
        print(f"  Aug Pure Effect median: {fragile_aug}")
        print(f"  Total Effect median: {fragile_dyn_total['median']}")
        print(f"  Identity check: {identity_check['identity_holds']}")
        if fragile_dyn_aug.get('zero_crossing'):
            print(f"  Aug Pure: zero-crossing (n+={fragile_dyn_aug['n_positive']}, n-={fragile_dyn_aug['n_negative']})")
        print("-"*60)
        for interp in interpretation:
            print(f"  → {interp}")
        print("="*60)
        
        return comparison
    
    def _save_comparison(self, comparison: Dict):
        """비교 결과 저장"""
        with open(self.results_dir / 'comparison_4way.json', 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: comparison_4way.json")
        
        # code_hash 생성
        source_files = [
            Path(__file__),
            self.project_root / 'src' / 'sindy' / 'esindy.py',
            self.project_root / 'src' / 'sindy' / 'library.py',
        ]
        code_hash = {}
        for src_file in source_files:
            if src_file.exists():
                code_hash[src_file.name] = compute_file_hash(src_file)
        
        # teacher_support hash 계산
        cfg = self.config
        day3_dir = self._get_results_dir(cfg.day3_run_id)
        teacher_support_path = day3_dir / 'teacher_support.npy'
        teacher_support_sha256 = compute_file_hash(teacher_support_path) if teacher_support_path.exists() else None
        
        # manifest.json (Modern Schema)
        manifest = {
            'phase': 'phase35',
            'day': 5,
            'mode': 'compare',
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'source_runs': comparison['source_runs'],
            'ssot': self.day3_manifest.get('ssot', {}),
            
            # === MODERNIZE FIELDS ===
            'control_equivalence': get_control_equivalence(bootstrap_B=cfg.bootstrap_B),
            'teacher_support_sha256': teacher_support_sha256,
            'teacher_support_source': f"{cfg.day3_run_id}/teacher_support.npy",
            'preflight_qc': {
                'dx_equivalence': {
                    'dx_key_used': 'compare_mode_aggregation',
                    'note': 'Compare mode aggregates results from other runs'
                }
            },
            'code_hash': code_hash
        }
        
        with open(self.results_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: manifest.json (with modern schema)")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 3.5 Day5: Control Experiments for Confound Separation'
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['control50', 'control250', 'compare'],
                        help='Experiment mode')
    parser.add_argument('--day3_run_id', type=str, required=True,
                        help='Day3 run_id (baseline)')
    parser.add_argument('--dataset_version', type=str, default='cartpole_ood_v1')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--track', type=str, default='standardized')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--note', type=str, default='day5')
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--bootstrap_B', type=int, default=20,
                        help='ESINDy bootstrap iterations (default: 20, use 100 for sensitivity)')
    
    # Compare mode 전용
    parser.add_argument('--ctrl50_run_id', type=str, default='',
                        help='Control-50 run_id (for compare mode)')
    parser.add_argument('--ctrl250_run_id', type=str, default='',
                        help='Control-250 run_id (for compare mode)')
    parser.add_argument('--treat_run_id', type=str, default='',
                        help='Treatment (Day4) run_id (for compare mode)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = Day5Config(
        mode=args.mode,
        day3_run_id=args.day3_run_id,
        dataset_version=args.dataset_version,
        dataset_path=args.dataset_path,
        track=args.track,
        seed=args.seed,
        note=args.note,
        threshold=args.threshold,
        bootstrap_B=args.bootstrap_B,
        ctrl50_run_id=args.ctrl50_run_id,
        ctrl250_run_id=args.ctrl250_run_id,
        treat_run_id=args.treat_run_id
    )
    
    if config.mode in ['control50', 'control250']:
        runner = Day5ControlRunner(config)
    elif config.mode == 'compare':
        if not config.ctrl50_run_id or not config.ctrl250_run_id or not config.treat_run_id:
            print("❌ Error: compare mode requires --ctrl50_run_id, --ctrl250_run_id, --treat_run_id")
            return 1
        runner = Day5CompareRunner(config)
    else:
        print(f"❌ Unknown mode: {config.mode}")
        return 1
    
    try:
        result = runner.run()
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())