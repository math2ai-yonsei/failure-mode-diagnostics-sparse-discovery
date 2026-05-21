"""
Phase 3.5 Day4 Runner: Augmentation & Training Effect

핵심 목표:
- Physics-based augmentation 적용
- z_after 계산 (bootstrap B=20, 기존 ESINDyEnsemble 활용)
- Training effect 지표 측정

산출물:
- manifest.json: 실험 메타데이터 + oracle_trace
- metrics.json: Primary metrics + null_reason
- structure_eval.json: delta_z_details (37 pairs)
- selection.json: Day3에서 복사
- z_before.npy, z_after.npy
- coef_mean_after.npy, coef_std_after.npy
- augmented_data/aug_trajectories.npz
- code_snapshot/, code_hash.json

Author: Claude (Phase 3.5 Day4)
Updated: Phase 3.5 Option B - Modern Schema
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

# === MODERNIZE IMPORT START ===
from phase35_manifest_modernize import (
    get_control_equivalence,
    compute_file_sha256
)
# === MODERNIZE IMPORT END ===


# ============================================================
# SSOT Constants (Day3과 동일)
# ============================================================
DEFAULT_TARGET_NAMES = ["x_dot", "x_ddot", "theta_dot", "theta_ddot"]
DEFAULT_TAU_SUPPORT = 0.5  # P1-1: tau_hi → tau_support 통일
DEFAULT_Z0 = 2.0
DEFAULT_EPS = 1e-12
BOOTSTRAP_B = 20  # Day4 SSOT


# ============================================================
# Physics Augmentor (최소 침습)
# ============================================================

@dataclass
class AugmentationConfig:
    """Augmentation 설정"""
    noise_std_ratio: float = 0.01
    ic_perturb_ratio: float = 0.02
    aug_factor: int = 5
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return {
            'noise_std_ratio': self.noise_std_ratio,
            'ic_perturb_ratio': self.ic_perturb_ratio,
            'aug_factor': self.aug_factor,
            'seed': self.seed
        }


class PhysicsAugmentor:
    """Physics-based Augmentor (최소 침습 원칙)"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(self.config.seed)
    
    def _compute_state_scales(self, x: np.ndarray) -> np.ndarray:
        """상태별 스케일 계산"""
        x_flat = x.reshape(-1, x.shape[-1])
        scales = np.std(x_flat, axis=0)
        scales = np.maximum(scales, 1e-6)
        return scales
    
    def _add_noise(self, x: np.ndarray, dx: np.ndarray, scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gaussian noise 추가 (position/angle만, kinematic consistency 보장)
        
        Cart-Pole state: [x, x_dot, theta, theta_dot]
        dx kinematic: dx[0]=x_dot=state[1], dx[2]=theta_dot=state[3]
        
        velocity (state[1], state[3])에 noise를 추가하면 dx[0], dx[2]와 불일치 발생
        dx에 독립 noise를 추가해도 kinematic 관계 위반
        → position/angle (state[0], state[2])만 noise, dx는 그대로 유지
        """
        noise_std = scales * self.config.noise_std_ratio
        
        # Position/angle만 noise (index 0, 2)
        # Velocity (index 1, 3)는 그대로 유지 → dx[0]=state[1], dx[2]=state[3] 관계 보존
        noise_mask = np.array([1.0, 0.0, 1.0, 0.0])  # [x, x_dot, theta, theta_dot]
        
        noise_x = self.rng.normal(0, noise_std, size=x.shape)
        noise_x = noise_x * noise_mask  # velocity 성분 제거
        x_noisy = x + noise_x
        
        # dx는 그대로 유지 (독립 noise 금지 → kinematic consistency 보장)
        dx_noisy = dx.copy()
        
        return x_noisy, dx_noisy
    
    def _perturb_ic(self, x: np.ndarray, dx: np.ndarray, scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """초기 조건 변형 (position/angle만, kinematic consistency 보장)
        
        Cart-Pole state: [x, x_dot, theta, theta_dot]
        dx kinematic: dx[0]=x_dot=state[1], dx[2]=theta_dot=state[3]
        
        velocity (state[1], state[3])를 perturb하면 dx[0], dx[2]와 불일치 발생
        → position/angle (state[0], state[2])만 perturb하여 consistency 유지
        """
        N, T, state_dim = x.shape
        perturb_std = scales * self.config.ic_perturb_ratio
        
        # Position/angle만 perturb (index 0, 2)
        # Velocity (index 1, 3)는 그대로 유지 → dx[0]=state[1], dx[2]=state[3] 관계 보존
        perturb_mask = np.array([1.0, 0.0, 1.0, 0.0])  # [x, x_dot, theta, theta_dot]
        
        ic_perturb = self.rng.normal(0, perturb_std, size=(N, 1, state_dim))
        ic_perturb = ic_perturb * perturb_mask  # velocity 성분 제거
        
        x_perturbed = x + ic_perturb
        return x_perturbed, dx.copy()
    
    def augment(self, x: np.ndarray, dx: np.ndarray, u: np.ndarray) -> Dict[str, Any]:
        """
        Physics-based augmentation 수행
        
        Args:
            x: (N, T, state_dim)
            dx: (N, T, state_dim)
            u: (N, T, input_dim)
        
        Returns:
            dict with x_aug, dx_aug, u_aug and metadata
        """
        N, T, state_dim = x.shape
        scales = self._compute_state_scales(x)
        
        x_list = [x]
        dx_list = [dx]
        u_list = [u]
        
        methods = ['noise', 'ic_perturb']
        n_aug_per_method = max(1, self.config.aug_factor // len(methods))
        
        for method in methods:
            for _ in range(n_aug_per_method):
                if method == 'noise':
                    x_aug, dx_aug = self._add_noise(x, dx, scales)
                else:
                    x_aug, dx_aug = self._perturb_ic(x, dx, scales)
                x_list.append(x_aug)
                dx_list.append(dx_aug)
                u_list.append(u.copy())
        
        return {
            'x_aug': np.concatenate(x_list, axis=0),
            'dx_aug': np.concatenate(dx_list, axis=0),
            'u_aug': np.concatenate(u_list, axis=0),
            'n_original': N,
            'n_augmented': len(x_list) * N - N,
            'n_total': len(x_list) * N,
            'config': self.config.to_dict()
        }


# ============================================================
# Configuration
# ============================================================

@dataclass
class Day4Config:
    """Day4 실험 설정"""
    day3_run_id: str = ""
    dataset_version: str = "cartpole_ood_v1"
    dataset_path: str = ""
    track: str = "standardized"
    tau_support: float = DEFAULT_TAU_SUPPORT
    z0: float = DEFAULT_Z0
    eps: float = DEFAULT_EPS
    bootstrap_B: int = BOOTSTRAP_B
    aug_factor: int = 5
    aug_noise_std_ratio: float = 0.01
    aug_ic_perturb_ratio: float = 0.02
    threshold: float = 0.05  # SINDy threshold
    seed: int = 0
    note: str = "day4"


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


# ============================================================
# Day4 Runner
# ============================================================

class Day4Runner:
    """Phase 3.5 Day4 Runner"""
    
    def __init__(self, config: Day4Config):
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
        
        # Day4 results
        self.aug_result: Optional[Dict] = None
        self.z_after: Optional[np.ndarray] = None
        self.coef_mean_after: Optional[np.ndarray] = None
        self.coef_std_after: Optional[np.ndarray] = None
        self.inc_prob_after: Optional[np.ndarray] = None  # P1-2: inclusion probability
        
        # Oracle trace (P1-3)
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
        """Day4 파이프라인 실행"""
        self._banner(f"Phase 3.5 Day4 Runner: {self.run_id}")
        
        # Step 1: Day3 결과 로드
        self._section("1/7", "Loading Day3 Results")
        self._load_day3_results()
        
        # Step 2: Training 데이터 로드
        self._section("2/7", "Loading Training Data")
        x_train, dx_train, u_train, T = self._load_training_data()
        
        # Step 3: Augmentation 적용
        self._section("3/7", "Applying Physics-based Augmentation")
        self._apply_augmentation(x_train, dx_train, u_train)
        
        # Step 4: ESINDy로 z_after 계산
        self._section("4/7", "Computing z_after (ESINDy Bootstrap)")
        self._compute_z_after(T)
        
        # Step 5: Training Effect 지표 계산
        self._section("5/7", "Computing Training Effect Metrics")
        metrics = self._compute_training_effect()
        
        # Step 6: Code Snapshot 생성
        self._section("6/7", "Creating Code Snapshot")
        code_hash = self._create_code_snapshot()
        
        # Step 7: 산출물 저장
        self._section("7/7", "Saving Artifacts")
        self._save_artifacts(metrics, code_hash)
        
        self._banner(f"✅ Day4 Complete: {self.run_id}")
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
        print(f"  ✅ Loaded numpy arrays")
        
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
    
    def _apply_augmentation(self, x: np.ndarray, dx: np.ndarray, u: np.ndarray):
        """Physics-based augmentation 적용"""
        cfg = self.config
        
        aug_config = AugmentationConfig(
            noise_std_ratio=cfg.aug_noise_std_ratio,
            ic_perturb_ratio=cfg.aug_ic_perturb_ratio,
            aug_factor=cfg.aug_factor,
            seed=cfg.seed
        )
        
        augmentor = PhysicsAugmentor(aug_config)
        self.aug_result = augmentor.augment(x, dx, u)
        
        print(f"  원본: {self.aug_result['n_original']} trajectories")
        print(f"  증강: {self.aug_result['n_augmented']} trajectories")
        print(f"  총합: {self.aug_result['n_total']} trajectories")
        
        # 증강 데이터 저장
        aug_dir = self.results_dir / 'augmented_data'
        aug_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            aug_dir / 'aug_trajectories.npz',
            x_aug=self.aug_result['x_aug'],
            dx_aug=self.aug_result['dx_aug'],
            u_aug=self.aug_result['u_aug']
        )
        print(f"  ✅ Saved: augmented_data/aug_trajectories.npz")
        
        self.oracle_trace['oracle_used_in_augmentation'] = False
    
    def _compute_z_after(self, T: int):
        """ESINDy Bootstrap으로 z_after 계산"""
        cfg = self.config
        
        if self.aug_result is None:
            raise RuntimeError("Augmentation not applied")
        
        x_aug = self.aug_result['x_aug']
        dx_aug = self.aug_result['dx_aug']
        u_aug = self.aug_result['u_aug']
        n_traj_aug = x_aug.shape[0]
        
        print(f"  Building library (gate0_min)...")
        
        # SINDy Library 구성
        library = SINDyLibrary(config='gate0_min')
        
        # Feature matrix 생성
        Theta = library.fit_transform(x_aug, u_aug)
        
        # dx를 2D로 flatten
        dx_flat = dx_aug.reshape(-1, dx_aug.shape[-1])
        
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
            n_trajectories=n_traj_aug,
            T=T,
            scaler=scaler,
            target_scale=None  # dx 정규화 안함
        )
        
        # 결과 추출
        self.coef_mean_after = ensemble.coefficients_mean_
        self.coef_std_after = ensemble.coefficients_std_
        self.inc_prob_after = ensemble.inclusion_probability_  # P1-2
        
        # z_after 계산 (SSOT: abs(mean)/(std+eps))
        self.z_after = np.abs(self.coef_mean_after) / (self.coef_std_after + cfg.eps)
        
        print(f"  coef_mean_after: {self.coef_mean_after.shape}")
        print(f"  coef_std_after: {self.coef_std_after.shape}")
        print(f"  inc_prob_after: {self.inc_prob_after.shape}")
        print(f"  z_after range: [{self.z_after.min():.2f}, {self.z_after.max():.2f}]")
        
        self.oracle_trace['oracle_used_in_training'] = False
    
    def _compute_training_effect(self) -> Dict[str, Any]:
        """Training Effect 지표 계산"""
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
        
        # delta_z_details 생성 (37 pairs)
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
        
        # 지표 3: spurious_reentry_rate (P0-재오픈 #2: support_after/stable_after 분리)
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
        
        # 지표 4: delta_z 통계 (P1-1: robust statistics 추가)
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
        
        # P1: fragile-pool 분리 통계 (stable-core vs fragile-pool)
        stable_core_mask = teacher_active & (self.z_before >= cfg.z0)
        fragile_pool_mask = teacher_active & (self.z_before < cfg.z0)
        
        n_stable_core = stable_core_mask.sum()
        n_fragile_pool = fragile_pool_mask.sum()
        
        if n_fragile_pool > 0:
            fragile_delta_z = delta_z[fragile_pool_mask]
            fragile_delta_z_median = float(np.median(fragile_delta_z))
            fragile_delta_z_mean = float(np.mean(fragile_delta_z))
            fragile_improved_count = int((fragile_delta_z > 0).sum())
        else:
            fragile_delta_z_median = None
            fragile_delta_z_mean = None
            fragile_improved_count = 0
        
        if n_stable_core > 0:
            stable_delta_z = delta_z[stable_core_mask]
            stable_delta_z_median = float(np.median(stable_delta_z))
            stable_delta_z_mean = float(np.mean(stable_delta_z))
        else:
            stable_delta_z_median = None
            stable_delta_z_mean = None
        
        # R2: Dynamics targets 분리 통계 (GPT 교차검토)
        # CartPole targets: ['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']
        # Dynamics: x_ddot (idx=1), theta_ddot (idx=3) - 실제 물리 학습 대상
        # Kinematic: x_dot (idx=0), theta_dot (idx=2) - identity 성격
        dynamics_target_indices = [1, 3]  # x_ddot, theta_ddot
        kinematic_target_indices = [0, 2]  # x_dot, theta_dot
        
        # Dynamics targets mask
        dynamics_mask = np.zeros_like(teacher_active, dtype=bool)
        for ti in dynamics_target_indices:
            if ti < delta_z.shape[1]:
                dynamics_mask[:, ti] = teacher_active[:, ti]
        
        n_dynamics = dynamics_mask.sum()
        if n_dynamics > 0:
            dynamics_delta_z = delta_z[dynamics_mask]
            dynamics_delta_z_median = float(np.median(dynamics_delta_z))
            dynamics_delta_z_mean = float(np.mean(dynamics_delta_z))
            dynamics_improved_count = int((dynamics_delta_z > 0).sum())
            dynamics_improved_rate = dynamics_improved_count / n_dynamics
        else:
            dynamics_delta_z_median = None
            dynamics_delta_z_mean = None
            dynamics_improved_count = 0
            dynamics_improved_rate = None
        
        # Kinematic targets mask
        kinematic_mask = np.zeros_like(teacher_active, dtype=bool)
        for ti in kinematic_target_indices:
            if ti < delta_z.shape[1]:
                kinematic_mask[:, ti] = teacher_active[:, ti]
        
        n_kinematic = kinematic_mask.sum()
        if n_kinematic > 0:
            kinematic_delta_z = delta_z[kinematic_mask]
            kinematic_delta_z_median = float(np.median(kinematic_delta_z))
            kinematic_delta_z_mean = float(np.mean(kinematic_delta_z))
        else:
            kinematic_delta_z_median = None
            kinematic_delta_z_mean = None
        
        # GPT 권고 2: fragile∩dynamics / fragile∩kinematic 분할
        fragile_dynamics_mask = fragile_pool_mask & dynamics_mask
        fragile_kinematic_mask = fragile_pool_mask & kinematic_mask
        
        n_fragile_dynamics = fragile_dynamics_mask.sum()
        n_fragile_kinematic = fragile_kinematic_mask.sum()
        
        if n_fragile_dynamics > 0:
            fragile_dynamics_delta_z = delta_z[fragile_dynamics_mask]
            fragile_dynamics_median = float(np.median(fragile_dynamics_delta_z))
            fragile_dynamics_improved = int((fragile_dynamics_delta_z > 0).sum())
            fragile_dynamics_improved_rate = fragile_dynamics_improved / n_fragile_dynamics
        else:
            fragile_dynamics_median = None
            fragile_dynamics_improved = 0
            fragile_dynamics_improved_rate = None
        
        if n_fragile_kinematic > 0:
            fragile_kinematic_delta_z = delta_z[fragile_kinematic_mask]
            fragile_kinematic_median = float(np.median(fragile_kinematic_delta_z))
        else:
            fragile_kinematic_median = None
        
        print(f"  promotion_rate_delta: {promotion_rate_delta}")
        print(f"  promotion_to_stable_rate: {promotion_to_stable_rate}")
        print(f"  spurious_reentry_rate (z-based): {spurious_reentry_rate}")
        print(f"  reentry_to_support_rate: {reentry_to_support_rate}")
        print(f"  reentry_to_stable_rate: {reentry_to_stable_rate}")
        print(f"  delta_z_median: {delta_z_median:.4f}")
        print(f"  delta_z_trimmed_mean: {delta_z_trimmed_mean:.4f}")
        print(f"  delta_z_mean: {delta_z_mean:.4f} (outlier-sensitive)")
        print(f"  [fragile-pool] n={n_fragile_pool}, median={fragile_delta_z_median}, improved={fragile_improved_count}")
        print(f"  [stable-core] n={n_stable_core}, median={stable_delta_z_median}")
        print(f"  [dynamics] n={n_dynamics}, median={dynamics_delta_z_median}, improved={dynamics_improved_count} ({dynamics_improved_rate:.1%})" if dynamics_improved_rate else f"  [dynamics] n={n_dynamics}, median={dynamics_delta_z_median}")
        print(f"  [kinematic] n={n_kinematic}, median={kinematic_delta_z_median}")
        print(f"  [fragile∩dynamics] n={n_fragile_dynamics}, median={fragile_dynamics_median}, improved={fragile_dynamics_improved}")
        print(f"  [fragile∩kinematic] n={n_fragile_kinematic}, median={fragile_kinematic_median}")
        
        return {
            'stage': 'augmentation_applied',
            'augmentation_applied': True,
            'training_effect_available': True,
            
            # GPT 권고 1: Primary metrics - dynamics 중심
            'primary_metrics': {
                # 핵심 지표 (dynamics 중심)
                'dynamics_delta_z_median': dynamics_delta_z_median,
                'dynamics_improved_rate': dynamics_improved_rate,
                'fragile_dynamics_median': fragile_dynamics_median,
                'fragile_dynamics_improved_rate': fragile_dynamics_improved_rate,
                
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
                
                # Overall statistics (보조)
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
                'n_dynamics_targets': int(n_dynamics),
                'n_kinematic_targets': int(n_kinematic),
                'n_fragile_dynamics': int(n_fragile_dynamics),
                'n_fragile_kinematic': int(n_fragile_kinematic)
            },
            
            # P1: fragile-pool 분리 통계
            'fragile_pool_stats': {
                'n': int(n_fragile_pool),
                'delta_z_median': fragile_delta_z_median,
                'delta_z_mean': fragile_delta_z_mean,
                'n_improved': fragile_improved_count
            },
            
            'stable_core_stats': {
                'n': int(n_stable_core),
                'delta_z_median': stable_delta_z_median,
                'delta_z_mean': stable_delta_z_mean
            },
            
            # R2: Dynamics vs Kinematic targets 분리 통계 (GPT 교차검토)
            'dynamics_targets_stats': {
                'description': 'x_ddot, theta_ddot - actual physics learning targets',
                'n': int(n_dynamics),
                'delta_z_median': dynamics_delta_z_median,
                'delta_z_mean': dynamics_delta_z_mean,
                'n_improved': dynamics_improved_count,
                'improved_rate': dynamics_improved_rate
            },
            
            'kinematic_targets_stats': {
                'description': 'x_dot, theta_dot - kinematic identity relationships',
                'n': int(n_kinematic),
                'delta_z_median': kinematic_delta_z_median,
                'delta_z_mean': kinematic_delta_z_mean
            },
            
            # GPT 권고 2: fragile∩dynamics / fragile∩kinematic 분할
            'fragile_dynamics_stats': {
                'description': 'fragile-pool ∩ dynamics targets (핵심 개선 대상)',
                'n': int(n_fragile_dynamics),
                'delta_z_median': fragile_dynamics_median,
                'n_improved': fragile_dynamics_improved,
                'improved_rate': fragile_dynamics_improved_rate
            },
            
            'fragile_kinematic_stats': {
                'description': 'fragile-pool ∩ kinematic targets',
                'n': int(n_fragile_kinematic),
                'delta_z_median': fragile_kinematic_median
            },
            
            '_structure_eval': {
                'delta_z_details': delta_z_details,
                'reentry_details': reentry_details
            },
            
            'selection_metrics': self.day3_metrics.get('selection_metrics', {})
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
        for fname in list(code_hash.keys())[:3]:
            print(f"      {fname}: {code_hash[fname][:16]}...")
        
        return code_hash
    
    def _save_artifacts(self, metrics: Dict, code_hash: Dict[str, str]):
        """산출물 저장 (Modern Schema 적용)"""
        cfg = self.config
        
        # === MODERNIZE: Day3 teacher_support hash 계산 ===
        day3_teacher_support = self.day3_results_dir / 'teacher_support.npy'
        if day3_teacher_support.exists():
            teacher_support_sha256 = compute_file_sha256(day3_teacher_support)
            print(f"  ✅ Computed teacher_support_sha256: {teacher_support_sha256[:16]}...")
        else:
            raise RuntimeError(
                f"Day3 teacher_support.npy not found at {day3_teacher_support}. "
                "Ensure Day3 is run with modern schema before Day4."
            )
        
        # 1. manifest.json
        manifest = {
            'phase': 'phase35',
            'day': 4,
            'run_id': self.run_id,
            'day3_run_id': cfg.day3_run_id,
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
            
            'augmentation': {
                # R1: 용어 정확화 (GPT 교차검토 v4→v5)
                'method': 'channel_consistent_perturbation',
                'description': 'Position/angle only perturbation, velocity and dx preserved for kinematic channel consistency',
                'perturbed_channels': ['x (position)', 'theta (angle)'],
                'preserved_channels': ['x_dot', 'theta_dot', 'dx'],
                'kinematic_channel_consistency': 'dx[0]=state[1], dx[2]=state[3] preserved',
                'trajectory_consistency': False,  # ODE trajectory NOT re-simulated
                'aug_factor': cfg.aug_factor,
                'noise_std_ratio': cfg.aug_noise_std_ratio,
                'ic_perturb_ratio': cfg.aug_ic_perturb_ratio,
                'n_original': self.aug_result['n_original'],
                'n_augmented': self.aug_result['n_augmented'],
                'n_total': self.aug_result['n_total']
            },
            
            'z_after_source': 'bootstrap',
            'z_formula': 'abs(mean)/(std+eps)',
            
            # SSOT 명시 (GPT 교차검토 P0-2)
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
            
            'stage': {
                'current': 'augmentation_applied',
                'selection_applied': True,
                'augmentation_applied': True
            },
            
            'oracle_trace': self.oracle_trace,
            'code_hash': code_hash,
            
            # === MODERNIZE FIELDS START ===
            'control_equivalence': get_control_equivalence(bootstrap_B=cfg.bootstrap_B),
            'teacher_support_sha256': teacher_support_sha256,
            'teacher_support_source': f"{cfg.day3_run_id}/teacher_support.npy",
            'preflight_qc': {
                'dx_equivalence': {
                    'dx_key_used': self.dx_key_used,  # 실측값
                    'note': f'Loaded from dataset key: {self.dx_key_used}'
                }
            }
            # === MODERNIZE FIELDS END ===
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
            'primary_metrics': metrics['primary_metrics'],
            'counts': metrics['counts'],
            'fragile_pool_stats': metrics['fragile_pool_stats'],
            'stable_core_stats': metrics['stable_core_stats'],
            'dynamics_targets_stats': metrics['dynamics_targets_stats'],
            'kinematic_targets_stats': metrics['kinematic_targets_stats'],
            'fragile_dynamics_stats': metrics['fragile_dynamics_stats'],
            'fragile_kinematic_stats': metrics['fragile_kinematic_stats'],
            'details': {
                'delta_z_details': metrics['_structure_eval']['delta_z_details'],
                'reentry_details': metrics['_structure_eval']['reentry_details']
            },
            'training_effect_available': True
        }
        
        with open(self.results_dir / 'structure_eval.json', 'w', encoding='utf-8') as f:
            json.dump(structure_eval, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: structure_eval.json ({len(structure_eval['details']['delta_z_details'])} delta_z pairs)")
        
        # 4. selection.json (Day3에서 복사)
        with open(self.results_dir / 'selection.json', 'w', encoding='utf-8') as f:
            json.dump(self.day3_selection, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: selection.json")
        
        # 5. numpy arrays
        np.save(self.results_dir / 'z_before.npy', self.z_before)
        np.save(self.results_dir / 'z_after.npy', self.z_after)
        np.save(self.results_dir / 'coef_mean_after.npy', self.coef_mean_after)
        np.save(self.results_dir / 'coef_std_after.npy', self.coef_std_after)
        np.save(self.results_dir / 'inc_prob_after.npy', self.inc_prob_after)  # P1-2
        print(f"  ✅ Saved: z_before, z_after, coef_mean_after, coef_std_after, inc_prob_after (.npy)")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 3.5 Day4: Augmentation & Training Effect'
    )
    
    parser.add_argument('--day3_run_id', type=str, required=True,
                        help='Day3 run_id to continue from')
    parser.add_argument('--dataset_version', type=str, default='cartpole_ood_v1')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--track', type=str, default='standardized')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--note', type=str, default='day4')
    parser.add_argument('--aug_factor', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--bootstrap_B', type=int, default=20,
                        help='ESINDy bootstrap iterations (default: 20, use 100 for sensitivity)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = Day4Config(
        day3_run_id=args.day3_run_id,
        dataset_version=args.dataset_version,
        dataset_path=args.dataset_path,
        track=args.track,
        seed=args.seed,
        note=args.note,
        aug_factor=args.aug_factor,
        threshold=args.threshold,
        bootstrap_B=args.bootstrap_B
    )
    
    runner = Day4Runner(config)
    
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