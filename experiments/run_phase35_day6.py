"""
Phase 3.5 Day6 Runner: R3 Trajectory-Consistent Augmentation

핵심 목표:
- R3 (Trajectory-Consistent Augmentation)으로 fragile∩dynamics 개선 가능성 검증
- IC만 perturb → ODE 재시뮬레이션 → savgol로 dx 재계산

Channel-Consistent (Day4) vs Trajectory-Consistent (R3):
- Day4: x + noise → 기존 dx 유지
- Day6: IC만 perturb → ODE 재시뮬레이션 → savgol 재계산

산출물:
- manifest.json: 실험 메타데이터 + control_equivalence + theta_policy
- metrics.json: Primary metrics + fidelity_qc + ic_distribution_shift
- structure_eval.json: delta_z_details
- z_before.npy, z_after.npy
- augmented_data/r3_trajectories.npz
- (compare mode) comparison_r3.json

Author: Claude (Phase 3.5 Day6)
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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable

import numpy as np
import pandas as pd
import yaml

# 기존 프로젝트 모듈 활용
from src.sindy.library import SINDyLibrary, get_derivative_key
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble
from src.simulators.cartpole_simulator import CartPoleSimulator
from src.utils.derivatives import compute_derivatives_savgol, SAVGOL_CONFIG

# Phase 3.5 Modernize Helper
from phase35_manifest_modernize import get_control_equivalence, compute_file_sha256


# ============================================================
# SSOT Constants (Day3/4/5와 동일)
# ============================================================
DEFAULT_TARGET_NAMES = ["x_dot", "x_ddot", "theta_dot", "theta_ddot"]
DEFAULT_TAU_SUPPORT = 0.5
DEFAULT_Z0 = 2.0
DEFAULT_EPS = 1e-12
BOOTSTRAP_B = 20

# Control Equivalence SSOT (Day5와 동일)
CONTROL_EQUIVALENCE = {
    'library': 'gate0_min',
    'threshold': 0.05,
    'bootstrap_B': 20,
    'resample_unit': 'trajectory',
    'seed_rule': 'seed_b = base_seed + b',
    'dx_source_key': 'train_dx_savgol',
    'tau_support': 0.5,
    'z0': 2.0,
    'eps': 1e-12
}

# R3 Config (최종 확정)
R3_CONFIG = {
    # IC Perturbation
    'ic_perturb_channels': [0, 2],      # x, theta만 (velocity 제외)
    'ic_perturb_ratio': 0.02,
    'theta_wrap_after_perturb': True,
    
    # Augmentation
    'aug_factor': 4,                    # 50 * 4 = 200
    'target_total': 250,                # 50 + 200
    
    # Controller
    'controller_source': 'original',
    'controller_type': 'exogenous_random_smooth',
    
    # Simulator - train_params에서 per-trajectory 로드
    'sim_params_source': 'train_params',
    'fixed_params': {
        'L': 0.5,
        'g': 9.81,
        'b_cart': 0.1,
        'b_pole': 0.01
    },
    
    # Solver (dataset 생성과 동일해야 함)
    'solver_config': {
        'method': 'RK45',
        'rtol': 1e-8,
        'atol': 1e-10
    },
    
    # dx Computation
    'savgol_window': 11,
    'savgol_polyorder': 3,
    'theta_idx': 2,
    
    # Quality Filter
    'quality_filter': {
        'max_x': 10.0,
        'max_theta': 3.1,
        'max_velocity': 30.0,
        'max_attempts': 10,
        'policy': 'resample_on_reject'
    },
    
    # Seed Rule (aug_idx 포함)
    'seed_rule': {
        'base_seed': 42,
        'formula': 'ic_seed = base_seed + aug_idx * N + traj_idx + attempt * retry_offset',
        'retry_offset': 1000
    },
    
    # Fidelity QC (scale-aware)
    'fidelity_qc': {
        'enabled': True,
        'threshold_type': 'scale_normalized',
        'rmse_norm_threshold': 0.01,  # 1% of data scale
        'fail_fast': True
    }
}

# Theta 처리 정책
THETA_POLICY = {
    'simulation': 'continuous',     # 시뮬레이션 중 wrap 없음
    'storage': 'wrap_to_pi',        # 저장 시 (-π, π]로 wrap
    'derivative': 'unwrap_first',   # dx 계산 전 unwrap
    'ic_perturb': 'wrap_after'      # perturb 후 wrap
}


# ============================================================
# Configuration
# ============================================================

@dataclass
class Day6Config:
    """Day6 실험 설정"""
    mode: str = "r3_treat"  # 'r3_treat', 'compare_r3'
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
    note: str = "day6"
    # Compare mode 전용
    ctrl250_run_id: str = ""
    treat_cc_run_id: str = ""
    treat_r3_run_id: str = ""


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
        json.dump(code_hash, f, indent=2, default=_json_default)
    
    return code_hash


def safe_float(val) -> Optional[float]:
    """numpy scalar를 Python float로 안전하게 변환"""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    return val


def _json_default(o):
    """JSON 직렬화를 위한 default encoder (numpy/pandas 스칼라 처리)"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap angle to (-π, π]"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


# ============================================================
# Preflight QC Class
# ============================================================

class PreflightQC:
    """Day6 Preflight Validation"""
    
    def __init__(self, dataset_path: Path, config_path: Path):
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.dataset = None
        self.yaml_config = None
        self.results = {}
    
    def run_all(self) -> Dict[str, Any]:
        """모든 preflight 검증 수행"""
        print("\n[Preflight QC] Starting...")
        
        # 1. 데이터셋 로드
        self._load_dataset()
        self._load_yaml_config()
        
        # 2. train_params 검증
        self.results['train_params'] = self._validate_train_params()
        
        # 3. Simulator Fidelity QC
        self.results['fidelity_qc'] = self._run_fidelity_qc()
        
        # 4. dx-Pipeline Equivalence QC
        self.results['dx_equivalence'] = self._run_dx_equivalence_qc()
        
        # 5. Fixed params 검증
        self.results['param_source_trace'] = self._validate_fixed_params()
        
        print("[Preflight QC] Complete!\n")
        return self.results
    
    def _load_dataset(self):
        """데이터셋 로드"""
        self.dataset = np.load(self.dataset_path)
        print(f"  ✅ Loaded dataset: {self.dataset_path}")
    
    def _load_yaml_config(self):
        """YAML 설정 로드"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.yaml_config = yaml.safe_load(f)
        print(f"  ✅ Loaded config: {self.config_path}")
    
    def _validate_train_params(self) -> Dict[str, Any]:
        """train_params에서 고정/변동 파라미터 확인"""
        train_params = self.dataset['train_params']
        n_traj, n_params = train_params.shape
        
        # 변동성 확인
        param_std = np.std(train_params, axis=0)
        param_mean = np.mean(train_params, axis=0)
        param_min = np.min(train_params, axis=0)
        param_max = np.max(train_params, axis=0)
        
        result = {
            'n_trajectories': n_traj,
            'n_params': n_params,
            'param_names': ['m_cart', 'm_pole'],
            'statistics': {
                'm_cart': {
                    'mean': float(param_mean[0]),
                    'std': float(param_std[0]),
                    'min': float(param_min[0]),
                    'max': float(param_max[0]),
                    'is_varying': bool(param_std[0] > 1e-6)
                },
                'm_pole': {
                    'mean': float(param_mean[1]),
                    'std': float(param_std[1]),
                    'min': float(param_min[1]),
                    'max': float(param_max[1]),
                    'is_varying': bool(param_std[1] > 1e-6)
                }
            }
        }
        
        print(f"  ✅ train_params validated: {n_traj} trajectories, {n_params} params")
        print(f"      m_cart: [{param_min[0]:.2f}, {param_max[0]:.2f}], varying={result['statistics']['m_cart']['is_varying']}")
        print(f"      m_pole: [{param_min[1]:.3f}, {param_max[1]:.3f}], varying={result['statistics']['m_pole']['is_varying']}")
        
        return result
    
    def _run_fidelity_qc(self) -> Dict[str, Any]:
        """Simulator Fidelity QC: IC perturb=0으로 재시뮬레이션 후 비교"""
        print("  Running Simulator Fidelity QC...")
        
        x_orig = self.dataset['train_x']
        u_orig = self.dataset['train_u']
        train_params = self.dataset['train_params']
        dt = float(self.dataset['dt'])
        T = x_orig.shape[1]
        duration = dt * (T - 1)
        
        # 첫 5개 trajectory만 테스트 (속도 위해)
        n_test = min(5, x_orig.shape[0])
        
        x_errors = []
        dx_errors = []
        
        for i in range(n_test):
            # Per-trajectory params
            m_cart = train_params[i, 0]
            m_pole = train_params[i, 1]
            
            params = {
                'm_cart': m_cart,
                'm_pole': m_pole,
                **R3_CONFIG['fixed_params']
            }
            
            sim = CartPoleSimulator(params)
            
            # IC
            ic = x_orig[i, 0, :]
            
            # Controller: 원본 u 시퀀스를 interpolate
            u_seq = u_orig[i, :, 0].copy()
            
            def controller(t, state):
                idx = int(round(t / dt))
                idx = np.clip(idx, 0, len(u_seq) - 1)
                return float(u_seq[idx])
            
            # 재시뮬레이션
            try:
                t_sim, x_sim, u_sim = sim.simulate(
                    ic, 
                    (0.0, duration), 
                    dt,
                    controller=controller,
                    method=R3_CONFIG['solver_config']['method'],
                    rtol=R3_CONFIG['solver_config']['rtol'],
                    atol=R3_CONFIG['solver_config']['atol']
                )
                
                # x 비교
                x_diff = x_sim[:T] - x_orig[i, :T, :]
                x_rmse = np.sqrt(np.mean(x_diff**2, axis=0))
                x_errors.append(x_rmse)
                
                # dx 비교
                dx_sim = compute_derivatives_savgol(
                    x_sim[:T][np.newaxis, ...], dt, theta_idx=2
                )[0]
                
                dx_orig_key = 'train_dx_savgol' if 'train_dx_savgol' in self.dataset.files else 'train_dx'
                dx_orig = self.dataset[dx_orig_key][i, :T, :]
                
                dx_diff = dx_sim - dx_orig
                dx_rmse = np.sqrt(np.mean(dx_diff**2, axis=0))
                dx_errors.append(dx_rmse)
                
            except Exception as e:
                print(f"      ⚠️ Trajectory {i} simulation failed: {e}")
                continue
        
        if len(x_errors) == 0:
            raise RuntimeError("Fidelity QC failed: no trajectories could be simulated")
        
        x_errors = np.array(x_errors)
        dx_errors = np.array(dx_errors)
        
        # Scale-normalized RMSE 계산
        x_scale = np.std(x_orig.reshape(-1, 4), axis=0)
        x_rmse_norm = np.mean(x_errors, axis=0) / (x_scale + 1e-10)
        
        dx_scale = np.std(self.dataset['train_dx_savgol'].reshape(-1, 4), axis=0)
        dx_rmse_norm = np.mean(dx_errors, axis=0) / (dx_scale + 1e-10)
        
        # 판정
        x_passed = np.all(x_rmse_norm < R3_CONFIG['fidelity_qc']['rmse_norm_threshold'])
        
        result = {
            'n_tested': n_test,
            'x_rmse_mean': x_errors.mean(axis=0).tolist(),
            'x_rmse_norm': x_rmse_norm.tolist(),
            'x_scale': x_scale.tolist(),
            'dx_rmse_mean': dx_errors.mean(axis=0).tolist(),
            'dx_rmse_norm': dx_rmse_norm.tolist(),
            'dx_scale': dx_scale.tolist(),
            'threshold': R3_CONFIG['fidelity_qc']['rmse_norm_threshold'],
            'passed': bool(x_passed),
            'solver_config': R3_CONFIG['solver_config']
        }
        
        status = "✅ PASSED" if x_passed else "❌ FAILED"
        print(f"  {status} Fidelity QC: max x_rmse_norm = {max(x_rmse_norm):.4f}")
        
        if R3_CONFIG['fidelity_qc']['fail_fast'] and not x_passed:
            raise RuntimeError(f"Fidelity QC failed: x_rmse_norm = {x_rmse_norm}")
        
        return result
    
    def _run_dx_equivalence_qc(self) -> Dict[str, Any]:
        """dx-Pipeline Equivalence QC: x_orig에서 savgol 재계산 후 비교"""
        print("  Running dx-Pipeline Equivalence QC...")
        
        x_orig = self.dataset['train_x']
        dt = float(self.dataset['dt'])
        
        # Savgol 재계산
        dx_recomputed = compute_derivatives_savgol(
            x_orig, dt, theta_idx=2,
            window=SAVGOL_CONFIG['window'],
            polyorder=SAVGOL_CONFIG['polyorder']
        )
        
        # 원본 dx
        dx_orig_key = 'train_dx_savgol' if 'train_dx_savgol' in self.dataset.files else 'train_dx'
        dx_orig = self.dataset[dx_orig_key]
        
        # 차이 계산
        diff = dx_recomputed - dx_orig
        max_abs_diff = float(np.abs(diff).max())
        mean_abs_diff = float(np.abs(diff).mean())
        
        # 임계값: 1e-6 (완화됨)
        threshold = 1e-6
        passed = max_abs_diff < threshold
        
        result = {
            'dx_key_used': dx_orig_key,
            'max_abs_diff': max_abs_diff,
            'mean_abs_diff': mean_abs_diff,
            'threshold': threshold,
            'passed': passed,
            'savgol_config': SAVGOL_CONFIG
        }
        
        status = "✅ PASSED" if passed else "⚠️ WARNING"
        print(f"  {status} dx-Pipeline Equivalence: max_abs_diff = {max_abs_diff:.2e}")
        
        return result
    
    def _validate_fixed_params(self) -> Dict[str, Any]:
        """Fixed params: yaml vs R3_CONFIG vs dataset 비교"""
        yaml_physics = self.yaml_config.get('physics', {})
        
        result = {
            'sources': {
                'yaml': {
                    'L': yaml_physics.get('L'),
                    'g': yaml_physics.get('g'),
                    'b_cart': yaml_physics.get('b_cart'),
                    'b_pole': yaml_physics.get('b_pole')
                },
                'r3_config': R3_CONFIG['fixed_params'],
                'simulator_default': CartPoleSimulator.DEFAULT_PARAMS
            },
            'consistency_check': {}
        }
        
        # 일관성 체크
        for key in ['L', 'g', 'b_cart', 'b_pole']:
            yaml_val = yaml_physics.get(key)
            r3_val = R3_CONFIG['fixed_params'].get(key)
            result['consistency_check'][key] = {
                'yaml': yaml_val,
                'r3_config': r3_val,
                'match': yaml_val == r3_val if yaml_val is not None else None
            }
        
        print(f"  ✅ Fixed params validated")
        
        return result


# ============================================================
# Effect Summarizer (Day5에서 재사용)
# ============================================================

def compute_stats(delta_z: np.ndarray, mask: np.ndarray, name: str = '') -> Dict[str, Any]:
    """마스크 영역에 대한 통계 계산"""
    if mask.sum() == 0:
        return {'n': 0, 'median': None, 'mean': None, 'n_improved': 0}
    
    values = delta_z[mask]
    n_positive = int((values > 0).sum())
    n_negative = int((values < 0).sum())
    n_zero = int((values == 0).sum())
    
    return {
        'n': int(mask.sum()),
        'median': safe_float(np.median(values)),
        'mean': safe_float(np.mean(values)),
        'std': safe_float(np.std(values)),
        'n_improved': n_positive,
        'improved_rate': safe_float(n_positive / mask.sum()),
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_zero': n_zero,
        'zero_crossing': n_positive > 0 and n_negative > 0,
        'quantiles': {
            'q10': safe_float(np.percentile(values, 10)),
            'q25': safe_float(np.percentile(values, 25)),
            'q50': safe_float(np.percentile(values, 50)),
            'q75': safe_float(np.percentile(values, 75)),
            'q90': safe_float(np.percentile(values, 90))
        }
    }


def compute_bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Dict[str, float]:
    """Bootstrap으로 median의 95% CI 계산"""
    if len(values) == 0:
        return {'lower': None, 'upper': None, 'median': None}
    
    rng = np.random.default_rng(42)
    medians = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        medians.append(np.median(sample))
    
    medians = np.array(medians)
    alpha = (1 - ci) / 2
    
    return {
        'lower': safe_float(np.percentile(medians, alpha * 100)),
        'upper': safe_float(np.percentile(medians, (1 - alpha) * 100)),
        'median': safe_float(np.median(values)),
        'crosses_zero': bool(np.percentile(medians, alpha * 100) <= 0 <= np.percentile(medians, (1 - alpha) * 100))
    }


# ============================================================
# R3 Augmentor (Trajectory-Consistent)
# ============================================================

class R3Augmentor:
    """
    Trajectory-Consistent Augmentor (R3)
    
    IC만 perturb → ODE 재시뮬레이션 → savgol로 dx 재계산
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or R3_CONFIG
        self.base_seed = self.config['seed_rule']['base_seed']
        self.retry_offset = self.config['seed_rule']['retry_offset']
        
        # 통계 추적
        self.stats = {
            'total_attempted': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'rejection_reasons': {},
            'attempts_histogram': {},
            'ic_distribution_before': {},
            'ic_distribution_after': {}
        }
    
    def _compute_ic_seed(self, aug_idx: int, traj_idx: int, attempt: int, n_traj: int) -> int:
        """IC seed 계산 (GPT 조언 #4)"""
        return self.base_seed + aug_idx * n_traj + traj_idx + attempt * self.retry_offset
    
    def _perturb_ic(self, ic: np.ndarray, scales: np.ndarray, seed: int) -> np.ndarray:
        """IC perturbation (position/angle만, theta wrap 포함)"""
        rng = np.random.default_rng(seed)
        
        ic_perturbed = ic.copy()
        
        # Position/angle만 perturb (channels 0, 2)
        for ch in self.config['ic_perturb_channels']:
            perturb = rng.normal(0, scales[ch] * self.config['ic_perturb_ratio'])
            ic_perturbed[ch] += perturb
        
        # Theta wrap after perturb
        if self.config['theta_wrap_after_perturb']:
            ic_perturbed[2] = wrap_angle(ic_perturbed[2])
        
        return ic_perturbed
    
    def _check_quality(self, x: np.ndarray) -> Tuple[bool, str]:
        """Trajectory quality check"""
        qf = self.config['quality_filter']
        
        # Cart position
        if np.abs(x[:, 0]).max() > qf['max_x']:
            return False, 'max_x_exceeded'
        
        # Pole angle
        if np.abs(x[:, 2]).max() > qf['max_theta']:
            return False, 'max_theta_exceeded'
        
        # Velocities
        if np.abs(x[:, 1]).max() > qf['max_velocity']:
            return False, 'max_x_dot_exceeded'
        
        if np.abs(x[:, 3]).max() > qf['max_velocity']:
            return False, 'max_theta_dot_exceeded'
        
        # NaN check
        if np.isnan(x).any():
            return False, 'nan_detected'
        
        return True, 'passed'
    
    def _create_controller_from_u(self, u_seq: np.ndarray, dt: float) -> Callable:
        """원본 u 시퀀스로부터 controller 함수 생성"""
        def controller(t: float, state: np.ndarray) -> float:
            idx = int(round(t / dt))
            idx = np.clip(idx, 0, len(u_seq) - 1)
            return float(u_seq[idx])
        return controller
    
    def augment(
        self,
        x_orig: np.ndarray,
        u_orig: np.ndarray,
        train_params: np.ndarray,
        dt: float,
        duration: float
    ) -> Dict[str, Any]:
        """
        R3 Trajectory-Consistent Augmentation 수행
        
        Args:
            x_orig: (N, T, 4) 원본 state trajectories
            u_orig: (N, T, 1) 원본 control inputs
            train_params: (N, 2) per-trajectory parameters [m_cart, m_pole]
            dt: time step
            duration: trajectory duration
        
        Returns:
            Dict with x_aug, u_aug, dx_aug and metadata
        """
        N, T, state_dim = x_orig.shape
        
        # State scales for perturbation
        x_flat = x_orig.reshape(-1, state_dim)
        scales = np.std(x_flat, axis=0)
        scales = np.maximum(scales, 1e-6)
        
        # IC distribution before
        ics_before = x_orig[:, 0, :]
        self.stats['ic_distribution_before'] = {
            'mean': ics_before.mean(axis=0).tolist(),
            'std': ics_before.std(axis=0).tolist(),
            'min': ics_before.min(axis=0).tolist(),
            'max': ics_before.max(axis=0).tolist()
        }
        
        # Augmented data lists (원본 포함)
        x_list = [x_orig.copy()]
        u_list = [u_orig.copy()]
        dx_list = []
        
        # 원본 dx 계산
        dx_orig = compute_derivatives_savgol(
            x_orig, dt, theta_idx=self.config['theta_idx'],
            window=self.config['savgol_window'],
            polyorder=self.config['savgol_polyorder']
        )
        dx_list.append(dx_orig)
        
        # Augmentation metadata - full-length (P1-2)
        # 원본 N개에 대해 sentinel 값 (-999) 사용
        ic_seeds_full = [-999] * N  # 원본은 -999
        source_traj_indices_full = list(range(N))  # 원본은 자기 자신
        is_fallback_full = [False] * N  # 원본은 fallback 아님
        
        # Fallback 추적 (P0-3)
        fallback_count = 0
        fallback_indices = []
        successful_aug_count = 0
        
        # Aug factor 만큼 augmentation
        aug_factor = self.config['aug_factor']
        qf = self.config['quality_filter']
        target_aug = N * aug_factor
        
        print(f"  R3 Augmentation: {N} orig x {aug_factor} aug = {target_aug} new")
        
        for aug_idx in range(aug_factor):
            x_aug_batch = np.zeros((N, T, state_dim), dtype=np.float64)
            u_aug_batch = np.zeros((N, T, 1), dtype=np.float64)
            
            for traj_idx in range(N):
                # Per-trajectory params
                m_cart = train_params[traj_idx, 0]
                m_pole = train_params[traj_idx, 1]
                
                params = {
                    'm_cart': float(m_cart),
                    'm_pole': float(m_pole),
                    **self.config['fixed_params']
                }
                
                sim = CartPoleSimulator(params)
                
                # Original IC and u sequence
                ic_orig = x_orig[traj_idx, 0, :]
                u_seq = u_orig[traj_idx, :, 0].copy()
                
                # Quality filter loop
                accepted = False
                final_ic_seed = -1
                for attempt in range(qf['max_attempts']):
                    self.stats['total_attempted'] += 1
                    
                    # Compute IC seed
                    ic_seed = self._compute_ic_seed(aug_idx, traj_idx, attempt, N)
                    
                    # Perturb IC
                    ic_perturbed = self._perturb_ic(ic_orig, scales, ic_seed)
                    
                    # Create controller from original u
                    controller = self._create_controller_from_u(u_seq, dt)
                    
                    # Simulate
                    try:
                        t_sim, x_sim, u_sim = sim.simulate(
                            ic_perturbed,
                            (0.0, duration),
                            dt,
                            controller=controller,
                            method=self.config['solver_config']['method'],
                            rtol=self.config['solver_config']['rtol'],
                            atol=self.config['solver_config']['atol']
                        )
                        
                        # Quality check
                        passed, reason = self._check_quality(x_sim[:T])
                        
                        if passed:
                            x_aug_batch[traj_idx] = x_sim[:T]
                            u_aug_batch[traj_idx] = u_sim[:T]
                            final_ic_seed = ic_seed
                            self.stats['total_accepted'] += 1
                            successful_aug_count += 1
                            
                            # Attempts histogram
                            self.stats['attempts_histogram'][attempt] = \
                                self.stats['attempts_histogram'].get(attempt, 0) + 1
                            
                            accepted = True
                            break
                        else:
                            self.stats['total_rejected'] += 1
                            self.stats['rejection_reasons'][reason] = \
                                self.stats['rejection_reasons'].get(reason, 0) + 1
                            
                    except Exception as e:
                        self.stats['total_rejected'] += 1
                        self.stats['rejection_reasons']['simulation_error'] = \
                            self.stats['rejection_reasons'].get('simulation_error', 0) + 1
                
                # Full-length metadata 추가
                ic_seeds_full.append(final_ic_seed)
                source_traj_indices_full.append(traj_idx)
                
                if not accepted:
                    # Fallback: use original (P0-3: 명시적 추적)
                    global_idx = N + aug_idx * N + traj_idx
                    print(f"    ⚠️ Traj {traj_idx}, aug {aug_idx}: max attempts reached, using original")
                    x_aug_batch[traj_idx] = x_orig[traj_idx]
                    u_aug_batch[traj_idx] = u_orig[traj_idx]
                    fallback_count += 1
                    fallback_indices.append(global_idx)
                    is_fallback_full.append(True)
                else:
                    is_fallback_full.append(False)
            
            x_list.append(x_aug_batch)
            u_list.append(u_aug_batch)
            
            # Compute dx for this batch
            dx_aug_batch = compute_derivatives_savgol(
                x_aug_batch, dt, theta_idx=self.config['theta_idx'],
                window=self.config['savgol_window'],
                polyorder=self.config['savgol_polyorder']
            )
            dx_list.append(dx_aug_batch)
            
            print(f"    Aug batch {aug_idx + 1}/{aug_factor} complete")
        
        # Concatenate
        x_aug = np.concatenate(x_list, axis=0)
        u_aug = np.concatenate(u_list, axis=0)
        dx_aug = np.concatenate(dx_list, axis=0)
        
        # IC distribution after
        ics_after = x_aug[:, 0, :]
        self.stats['ic_distribution_after'] = {
            'mean': ics_after.mean(axis=0).tolist(),
            'std': ics_after.std(axis=0).tolist(),
            'min': ics_after.min(axis=0).tolist(),
            'max': ics_after.max(axis=0).tolist()
        }
        
        # Compute IC distribution shift
        ic_shift = {
            'mean_diff': (np.array(self.stats['ic_distribution_after']['mean']) - 
                         np.array(self.stats['ic_distribution_before']['mean'])).tolist(),
            'std_diff': (np.array(self.stats['ic_distribution_after']['std']) - 
                        np.array(self.stats['ic_distribution_before']['std'])).tolist()
        }
        
        # Accept rate 정의 명확화 (P1-1)
        attempt_accept_rate = self.stats['total_accepted'] / max(1, self.stats['total_attempted'])
        sample_success_rate = successful_aug_count / max(1, target_aug)
        fallback_rate = fallback_count / max(1, target_aug)
        
        print(f"  R3 Augmentation complete:")
        print(f"    Total: {x_aug.shape[0]} trajectories ({N} orig + {target_aug} aug)")
        print(f"    Attempt accept rate: {attempt_accept_rate:.1%}")
        print(f"    Sample success rate: {sample_success_rate:.1%} ({successful_aug_count}/{target_aug})")
        print(f"    Fallback count: {fallback_count} ({fallback_rate:.1%})")
        print(f"    Rejections: {self.stats['rejection_reasons']}")
        
        return {
            'x_aug': x_aug,
            'u_aug': u_aug,
            'dx_aug': dx_aug,
            'n_original': N,
            'n_augmented': target_aug,
            'n_total': x_aug.shape[0],
            # Full-length metadata (P1-2)
            'ic_seeds_full': ic_seeds_full,
            'source_traj_indices_full': source_traj_indices_full,
            'is_fallback_full': is_fallback_full,
            # Fallback 정보 (P0-3)
            'fallback_count': fallback_count,
            'fallback_indices': fallback_indices,
            'fallback_rate': fallback_rate,
            'successful_aug_count': successful_aug_count,
            # Accept rate 상세 (P1-1)
            'quality_filter_stats': {
                'total_attempted': self.stats['total_attempted'],
                'total_accepted': self.stats['total_accepted'],
                'total_rejected': self.stats['total_rejected'],
                'attempt_accept_rate': attempt_accept_rate,
                'sample_success_rate': sample_success_rate,
                'target_aug_samples': target_aug,
                'successful_aug_samples': successful_aug_count,
                'fallback_samples': fallback_count,
                'rejection_reasons': self.stats['rejection_reasons'],
                'attempts_histogram': self.stats['attempts_histogram']
            },
            'ic_distribution_shift': ic_shift,
            'ic_distribution_before': self.stats['ic_distribution_before'],
            'ic_distribution_after': self.stats['ic_distribution_after'],
            'config': self.config
        }


# ============================================================
# Day6 Treatment Runner (r3_treat mode)
# ============================================================

class Day6TreatRunner:
    """Phase 3.5 Day6 R3 Treatment Runner"""
    
    def __init__(self, config: Day6Config):
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
        
        # Arrays
        self.z_before: Optional[np.ndarray] = None
        self.selected_mask: Optional[np.ndarray] = None
        self.teacher_support: Optional[np.ndarray] = None
        self.oracle_support: Optional[np.ndarray] = None
        
        # Day6 results
        self.z_after: Optional[np.ndarray] = None
        self.coef_mean_after: Optional[np.ndarray] = None
        self.coef_std_after: Optional[np.ndarray] = None
        self.inc_prob_after: Optional[np.ndarray] = None
        self.aug_result: Optional[Dict] = None
        self.preflight_results: Optional[Dict] = None
        
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
        """Day6 R3 Treatment 파이프라인 실행"""
        self._banner(f"Phase 3.5 Day6 R3 Treatment Runner: {self.run_id}")
        
        # Step 0: Preflight QC
        self._section("0/7", "Preflight Validation")
        self._run_preflight()
        
        # Step 1: Day3 결과 로드
        self._section("1/7", "Loading Day3 Results")
        self._load_day3_results()
        
        # Step 2: Training 데이터 로드
        self._section("2/7", "Loading Training Data")
        x_train, u_train, train_params, dt, duration, T = self._load_training_data()
        
        # Step 3: R3 Augmentation
        self._section("3/7", "R3 Trajectory-Consistent Augmentation")
        x_aug, dx_aug, u_aug = self._run_r3_augmentation(x_train, u_train, train_params, dt, duration)
        
        # Step 4: ESINDy로 z_after 계산
        self._section("4/7", "Computing z_after (ESINDy Bootstrap)")
        self._compute_z_after(x_aug, dx_aug, u_aug, T)
        
        # Step 5: Training Effect 지표 계산
        self._section("5/7", "Computing Training Effect Metrics")
        metrics = self._compute_training_effect()
        
        # Step 6: 산출물 저장
        self._section("6/7", "Saving Artifacts")
        code_hash = self._create_code_snapshot()
        self._save_artifacts(metrics, code_hash)
        
        # Step 7: Augmented data 저장
        self._section("7/7", "Saving Augmented Data")
        self._save_augmented_data(x_aug, u_aug, dx_aug, dt, duration, train_params)
        
        self._banner(f"✅ Day6 R3 Treatment Complete: {self.run_id}")
        print(f"  Results: {self.results_dir}")
        
        return {'run_id': self.run_id, 'metrics': metrics}
    
    def _run_preflight(self):
        """Preflight QC 수행"""
        cfg = self.config
        
        # Dataset path
        if cfg.dataset_path:
            dataset_path = Path(cfg.dataset_path)
        else:
            dataset_path = self.project_root / 'data' / 'cartpole' / cfg.dataset_version / 'dataset.npz'
        
        # Config path
        config_path = self.project_root / 'configs' / 'systems' / 'cartpole.yaml'
        
        preflight = PreflightQC(dataset_path, config_path)
        self.preflight_results = preflight.run_all()
    
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
        with open(self.day3_results_dir / 'manifest.json', 'r', encoding='utf-8') as f:
            self.day3_manifest = json.load(f)
        print(f"  ✅ Loaded manifest.json")
        
        with open(self.day3_results_dir / 'selection.json', 'r', encoding='utf-8') as f:
            self.day3_selection = json.load(f)
        print(f"  ✅ Loaded selection.json")
        
        with open(self.day3_results_dir / 'metrics.json', 'r', encoding='utf-8') as f:
            self.day3_metrics = json.load(f)
        print(f"  ✅ Loaded metrics.json")
        
        # Feature/Target names
        self.feature_names = self.day3_manifest['ssot']['feature_names']
        self.target_names = self.day3_manifest['ssot']['target_names']
        print(f"  Features: {len(self.feature_names)}, Targets: {len(self.target_names)}")
        
        # Numpy arrays 로드
        self.z_before = np.load(self.day3_results_dir / 'z_before.npy')
        self.teacher_support = np.load(self.day3_results_dir / 'teacher_support.npy')
        print(f"  ✅ Loaded numpy arrays (z_before, teacher_support)")
        
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
    
    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
        """Training 데이터 로드"""
        cfg = self.config
        
        if cfg.dataset_path:
            dataset_path = Path(cfg.dataset_path)
        else:
            dataset_path = self.project_root / 'data' / 'cartpole' / cfg.dataset_version / 'dataset.npz'
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        data = np.load(dataset_path)
        x_train = data['train_x']
        u_train = data['train_u']
        train_params = data['train_params']
        dt = float(data['dt'])
        
        T = x_train.shape[1]
        duration = dt * (T - 1)
        
        print(f"  x_train: {x_train.shape}")
        print(f"  u_train: {u_train.shape}")
        print(f"  train_params: {train_params.shape}")
        print(f"  dt: {dt}, T: {T}, duration: {duration}s")
        
        return x_train, u_train, train_params, dt, duration, T
    
    def _run_r3_augmentation(
        self,
        x_train: np.ndarray,
        u_train: np.ndarray,
        train_params: np.ndarray,
        dt: float,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """R3 Trajectory-Consistent Augmentation 수행"""
        augmentor = R3Augmentor(R3_CONFIG)
        self.aug_result = augmentor.augment(x_train, u_train, train_params, dt, duration)
        
        return self.aug_result['x_aug'], self.aug_result['dx_aug'], self.aug_result['u_aug']
    
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
        """Training Effect 지표 계산"""
        cfg = self.config
        
        if self.z_after is None or self.z_before is None:
            raise RuntimeError("z_before or z_after not computed")
        
        # P0-2: Oracle은 evaluation_only로 사용, training/metrics에 사용 안 함
        self.oracle_trace['oracle_used_in_metrics'] = False
        
        teacher_active = self.teacher_support.astype(bool)
        n_teacher_active = teacher_active.sum()
        oracle_active = self.oracle_support.astype(bool) if self.oracle_support is not None else teacher_active
        
        print(f"  Teacher active pairs: {n_teacher_active}")
        
        # Delta-z 계산
        delta_z = self.z_after - self.z_before
        
        # Masks
        stable_core_mask = teacher_active & (self.z_before >= cfg.z0)
        fragile_pool_mask = teacher_active & (self.z_before < cfg.z0)
        
        dynamics_target_indices = [1, 3]  # x_ddot, theta_ddot
        kinematic_target_indices = [0, 2]  # x_dot, theta_dot
        
        dynamics_mask = np.zeros_like(teacher_active, dtype=bool)
        kinematic_mask = np.zeros_like(teacher_active, dtype=bool)
        
        for ti in dynamics_target_indices:
            if ti < self.z_before.shape[1]:
                dynamics_mask[:, ti] = teacher_active[:, ti]
        
        for ti in kinematic_target_indices:
            if ti < self.z_before.shape[1]:
                kinematic_mask[:, ti] = teacher_active[:, ti]
        
        fragile_dynamics_mask = fragile_pool_mask & dynamics_mask
        stable_dynamics_mask = stable_core_mask & dynamics_mask
        
        # Training Effect 통계
        total_stats = compute_stats(delta_z, teacher_active, 'total')
        fragile_stats = compute_stats(delta_z, fragile_pool_mask, 'fragile')
        stable_stats = compute_stats(delta_z, stable_core_mask, 'stable')
        fragile_dynamics_stats = compute_stats(delta_z, fragile_dynamics_mask, 'fragile_dyn')
        stable_dynamics_stats = compute_stats(delta_z, stable_dynamics_mask, 'stable_dyn')
        
        # Primary metrics
        delta_z_fragile_dynamics = delta_z[fragile_dynamics_mask]
        delta_z_median = safe_float(np.median(delta_z_fragile_dynamics)) if len(delta_z_fragile_dynamics) > 0 else None
        
        metrics = {
            'run_id': self.run_id,
            'day3_run_id': cfg.day3_run_id,
            
            'primary_metrics': {
                'delta_z_median_fragile_dynamics': delta_z_median,
                'n_fragile_dynamics': int(fragile_dynamics_mask.sum()),
                'n_improved_fragile_dynamics': fragile_dynamics_stats.get('n_improved', 0),
                'improved_rate_fragile_dynamics': fragile_dynamics_stats.get('improved_rate')
            },
            
            'counts': {
                'n_teacher_active': int(n_teacher_active),
                'n_stable_core': int(stable_core_mask.sum()),
                'n_fragile_pool': int(fragile_pool_mask.sum()),
                'n_dynamics_targets': int(dynamics_mask.sum()),
                'n_fragile_dynamics': int(fragile_dynamics_mask.sum()),
                'n_stable_dynamics': int(stable_dynamics_mask.sum())
            },
            
            'total_stats': total_stats,
            'fragile_pool_stats': fragile_stats,
            'stable_core_stats': stable_stats,
            'fragile_dynamics_stats': fragile_dynamics_stats,
            'stable_dynamics_stats': stable_dynamics_stats,
            
            'r3_augmentation': {
                'method': 'trajectory_consistent',
                'n_original': self.aug_result['n_original'],
                'n_augmented': self.aug_result['n_augmented'],
                'n_total': self.aug_result['n_total'],
                'quality_filter_stats': self.aug_result['quality_filter_stats'],
                'ic_distribution_shift': self.aug_result['ic_distribution_shift']
            },
            
            'preflight_qc': {
                'fidelity_qc_passed': self.preflight_results['fidelity_qc']['passed'],
                'dx_equivalence_passed': self.preflight_results['dx_equivalence']['passed']
            }
        }
        
        # Console output
        print(f"\n  Primary Results (fragile∩dynamics):")
        print(f"    n = {fragile_dynamics_mask.sum()}")
        print(f"    delta_z median = {delta_z_median}")
        print(f"    improved = {fragile_dynamics_stats.get('n_improved', 0)} ({fragile_dynamics_stats.get('improved_rate', 0):.1%})")
        
        return metrics
    
    def _create_code_snapshot(self) -> Dict[str, str]:
        """Code Snapshot 생성"""
        source_files = [
            Path(__file__),
            self.project_root / 'src' / 'sindy' / 'esindy.py',
            self.project_root / 'src' / 'sindy' / 'library.py',
            self.project_root / 'src' / 'simulators' / 'cartpole_simulator.py',
            self.project_root / 'src' / 'utils' / 'derivatives.py',
        ]
        
        code_hash = create_code_snapshot(self.results_dir, source_files)
        
        print(f"  ✅ Created code_snapshot/ with {len(code_hash)} files")
        
        return code_hash
    
    def _save_artifacts(self, metrics: Dict, code_hash: Dict[str, str]):
        """산출물 저장"""
        cfg = self.config
        
        # teacher_support hash 계산
        teacher_support_path = self.day3_results_dir / 'teacher_support.npy'
        teacher_support_sha256 = compute_file_hash(teacher_support_path) if teacher_support_path.exists() else None
        
        # 1. manifest.json
        manifest = {
            'phase': 'phase35',
            'day': 6,
            'mode': 'r3_treat',
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
                'method': 'trajectory_consistent',
                'description': 'IC perturb -> ODE resimulation -> savgol dx recompute',
                'perturbed_channels': ['x (position)', 'theta (angle)'],
                'preserved_channels': ['x_dot', 'theta_dot'],
                'trajectory_consistency': True,
                'controller_source': R3_CONFIG['controller_source'],
                'config': R3_CONFIG
            },
            
            'control_equivalence': get_control_equivalence(bootstrap_B=cfg.bootstrap_B),
            'theta_policy': THETA_POLICY,
            
            'ssot': self.day3_manifest.get('ssot', {}),
            
            'definitions': {
                'z_score': '|mean| / (std + eps)',
                'support': f'inc_prob >= {cfg.tau_support}',
                'stable_core': f'support AND z >= {cfg.z0}',
                'fragile_pool': f'support AND z < {cfg.z0}'
            },
            
            'stage': {
                'current': 'r3_augmentation_applied',
                'selection_applied': True,
                'augmentation_applied': True,
                'augmentation_type': 'trajectory_consistent'
            },
            
            'preflight_qc': self.preflight_results,
            'oracle_trace': self.oracle_trace,
            'code_hash': code_hash,
            'teacher_support_sha256': teacher_support_sha256,
            'sensitivity_check': {
                'is_sensitivity': cfg.bootstrap_B != 20,
                'bootstrap_B_used': cfg.bootstrap_B,
                'bootstrap_B_baseline': 20,
                'note': 'Sensitivity check for bootstrap power analysis' if cfg.bootstrap_B != 20 else None
            }
        }
        
        with open(self.results_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"  ✅ Saved: manifest.json")
        
        # 2. metrics.json
        with open(self.results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"  ✅ Saved: metrics.json")
        
        # 3. selection.json (Day3에서 복사)
        with open(self.results_dir / 'selection.json', 'w', encoding='utf-8') as f:
            json.dump(self.day3_selection, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"  ✅ Saved: selection.json")
        
        # 4. numpy arrays
        np.save(self.results_dir / 'z_before.npy', self.z_before)
        np.save(self.results_dir / 'z_after.npy', self.z_after)
        np.save(self.results_dir / 'coef_mean_after.npy', self.coef_mean_after)
        np.save(self.results_dir / 'coef_std_after.npy', self.coef_std_after)
        np.save(self.results_dir / 'inc_prob_after.npy', self.inc_prob_after)
        print(f"  ✅ Saved: z_before, z_after, coef_mean_after, coef_std_after, inc_prob_after (.npy)")
    
    def _save_augmented_data(
        self, 
        x_aug: np.ndarray, 
        u_aug: np.ndarray, 
        dx_aug: np.ndarray,
        dt: float,
        duration: float,
        train_params: np.ndarray
    ):
        """Augmented data 저장"""
        aug_dir = self.results_dir / 'augmented_data'
        aug_dir.mkdir(parents=True, exist_ok=True)
        
        # train_params_full 생성 (P1-3: 재현성을 위해)
        # source_traj_indices_full로 원본 params를 매핑
        source_indices = np.array(self.aug_result['source_traj_indices_full'])
        train_params_full = train_params[source_indices]  # (250, 2)
        
        np.savez(
            aug_dir / 'r3_trajectories.npz',
            x_aug=x_aug,
            u_aug=u_aug,
            dx_aug=dx_aug,
            # Per-trajectory params for reproducibility (P1-3)
            train_params_full=train_params_full,
            # Full-length metadata (P1-2)
            ic_seeds_full=np.array(self.aug_result['ic_seeds_full']),
            source_traj_indices_full=np.array(self.aug_result['source_traj_indices_full']),
            is_fallback_full=np.array(self.aug_result['is_fallback_full']),
            # Fallback 정보 (P0-3)
            fallback_count=self.aug_result['fallback_count'],
            fallback_indices=np.array(self.aug_result['fallback_indices']),
            # Config
            dt=dt,
            duration=duration,
            n_original=self.aug_result['n_original'],
            n_augmented=self.aug_result['n_augmented'],
            successful_aug_count=self.aug_result['successful_aug_count'],
            **{f'fixed_{k}': v for k, v in R3_CONFIG['fixed_params'].items()},
            savgol_window=R3_CONFIG['savgol_window'],
            savgol_polyorder=R3_CONFIG['savgol_polyorder'],
            solver_method=R3_CONFIG['solver_config']['method'],
            solver_rtol=R3_CONFIG['solver_config']['rtol'],
            solver_atol=R3_CONFIG['solver_config']['atol']
        )
        print(f"  ✅ Saved: augmented_data/r3_trajectories.npz")


# ============================================================
# Day6 Compare Runner (compare_r3 mode)
# ============================================================

class Day6CompareRunner:
    """Phase 3.5 Day6 Compare Runner"""
    
    def __init__(self, config: Day6Config):
        self.config = config
        self.run_id = generate_run_id(config.note)
        
        self.project_root = _PROJECT_ROOT
        self.results_dir: Optional[Path] = None
        
        # Loaded data - manifests for all 4 runs
        self.day3_manifest: Optional[Dict] = None
        self.ctrl250_manifest: Optional[Dict] = None
        self.treat_cc_manifest: Optional[Dict] = None
        self.treat_r3_manifest: Optional[Dict] = None
        
        self.feature_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        
        # Arrays
        self.z_before: Optional[np.ndarray] = None
        self.z_ctrl250: Optional[np.ndarray] = None
        self.z_treat_cc: Optional[np.ndarray] = None
        self.z_treat_r3: Optional[np.ndarray] = None
        self.teacher_support: Optional[np.ndarray] = None
        
        # Metrics
        self.metrics_ctrl250: Optional[Dict] = None
        self.metrics_treat_cc: Optional[Dict] = None
        self.metrics_treat_r3: Optional[Dict] = None
        
        # Equality validation results
        self.equality_validation: Optional[Dict] = None
    
    def _banner(self, msg: str):
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"{'='*60}")
    
    def _section(self, step: str, title: str):
        print(f"\n[{step}] {title}")
        print("-" * 50)
    
    def _get_results_dir(self, run_id: str) -> Path:
        """run_id로 results 디렉토리 경로 생성"""
        cfg = self.config
        return (
            self.project_root / 'results' / cfg.dataset_version / 'phase35' /
            cfg.track / 'stable_core' / 'n10' / f'seed{cfg.seed}' / run_id
        )
    
    def run(self) -> Dict[str, Any]:
        """Day6 Compare 파이프라인 실행"""
        self._banner(f"Phase 3.5 Day6 Compare Runner: {self.run_id}")
        
        # Step 1: 모든 run 결과 로드
        self._section("1/4", "Loading All Run Results")
        self._load_all_results()
        
        # Step 2: Control Equivalence 검증 (P0 필수)
        self._section("2/4", "Validating Control Equivalence")
        self._validate_control_equivalence()
        
        # Step 3: R3 비교 계산
        self._section("3/4", "Computing R3 Comparison")
        comparison = self._compute_comparison()
        
        # Step 4: 비교 결과 저장
        self._section("4/4", "Saving Comparison Results")
        self._save_comparison(comparison)
        
        self._banner(f"✅ Day6 Compare Complete: {self.run_id}")
        print(f"  Results: {self.results_dir}")
        
        return {'run_id': self.run_id, 'comparison': comparison}
    
    def _load_all_results(self):
        """모든 run 결과 로드 (manifests 포함)"""
        cfg = self.config
        
        # Day3 (baseline)
        day3_dir = self._get_results_dir(cfg.day3_run_id)
        with open(day3_dir / 'manifest.json', 'r', encoding='utf-8') as f:
            self.day3_manifest = json.load(f)
        
        self.feature_names = self.day3_manifest['ssot']['feature_names']
        self.target_names = self.day3_manifest['ssot']['target_names']
        
        self.z_before = np.load(day3_dir / 'z_before.npy')
        self.teacher_support = np.load(day3_dir / 'teacher_support.npy')
        print(f"  ✅ Loaded Day3 baseline: {cfg.day3_run_id}")
        
        # Control-250 (manifest 포함)
        ctrl250_dir = self._get_results_dir(cfg.ctrl250_run_id)
        self.z_ctrl250 = np.load(ctrl250_dir / 'z_after.npy')
        with open(ctrl250_dir / 'manifest.json', 'r', encoding='utf-8') as f:
            self.ctrl250_manifest = json.load(f)
        with open(ctrl250_dir / 'metrics.json', 'r', encoding='utf-8') as f:
            self.metrics_ctrl250 = json.load(f)
        print(f"  ✅ Loaded Control-250: {cfg.ctrl250_run_id}")
        
        # Treatment CC - Day4 (manifest 포함)
        treat_cc_dir = self._get_results_dir(cfg.treat_cc_run_id)
        self.z_treat_cc = np.load(treat_cc_dir / 'z_after.npy')
        with open(treat_cc_dir / 'manifest.json', 'r', encoding='utf-8') as f:
            self.treat_cc_manifest = json.load(f)
        with open(treat_cc_dir / 'metrics.json', 'r', encoding='utf-8') as f:
            self.metrics_treat_cc = json.load(f)
        print(f"  ✅ Loaded Treatment CC: {cfg.treat_cc_run_id}")
        
        # Treatment R3 - Day6 (manifest 포함)
        treat_r3_dir = self._get_results_dir(cfg.treat_r3_run_id)
        self.z_treat_r3 = np.load(treat_r3_dir / 'z_after.npy')
        with open(treat_r3_dir / 'manifest.json', 'r', encoding='utf-8') as f:
            self.treat_r3_manifest = json.load(f)
        with open(treat_r3_dir / 'metrics.json', 'r', encoding='utf-8') as f:
            self.metrics_treat_r3 = json.load(f)
        print(f"  ✅ Loaded Treatment R3: {cfg.treat_r3_run_id}")
        
        # Results directory
        self.results_dir = self._get_results_dir(self.run_id)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Shape check
        shapes = [
            self.z_before.shape,
            self.z_ctrl250.shape,
            self.z_treat_cc.shape,
            self.z_treat_r3.shape
        ]
        if len(set(shapes)) != 1:
            raise ValueError(f"Shape mismatch: {shapes}")
        print(f"  ✅ Shape check passed: {shapes[0]}")
    
    def _validate_control_equivalence(self):
        """
        Control Equivalence 검증 (P0 필수)
        
        4개 run의 manifest에서 다음 항목이 동일한지 검증:
        1. control_equivalence 설정 (legacy runs는 SSOT assumed)
        2. teacher_support_sha256
        3. ssot feature_names/target_names
        4. dx_key_used (preflight_qc에서)
        
        Legacy runs (control_equivalence 없음)은 경고만 출력하고 계속 진행
        """
        cfg = self.config
        
        manifests = {
            'day3': self.day3_manifest,
            'ctrl250': self.ctrl250_manifest,
            'treat_cc': self.treat_cc_manifest,
            'treat_r3': self.treat_r3_manifest
        }
        
        validation_results = {
            'control_equivalence': {},
            'legacy_runs': [],
            'teacher_support_sha256': {},
            'feature_names': {},
            'target_names': {},
            'dx_key_used': {},
            'all_passed': True,
            'warnings': [],
            'failures': []
        }
        
        # 1. Control Equivalence 검증 (Legacy Compatibility)
        print("  [1/4] Checking control_equivalence...")
        ctrl_eq_values = {}
        legacy_runs = []
        modern_runs = []
        
        for name, manifest in manifests.items():
            ctrl_eq = manifest.get('control_equivalence', {})
            # 핵심 키 추출
            key_subset = {
                'library': ctrl_eq.get('library'),
                'threshold': ctrl_eq.get('threshold'),
                'bootstrap_B': ctrl_eq.get('bootstrap_B'),
                'tau_support': ctrl_eq.get('tau_support'),
                'z0': ctrl_eq.get('z0'),
                'eps': ctrl_eq.get('eps')
            }
            
            # legacy run 판단: 모든 값이 None인 경우
            all_none = all(v is None for v in key_subset.values())
            if all_none:
                legacy_runs.append(name)
                # Legacy run은 현재 SSOT assumed
                key_subset = {
                    'library': CONTROL_EQUIVALENCE['library'],
                    'threshold': CONTROL_EQUIVALENCE['threshold'],
                    'bootstrap_B': CONTROL_EQUIVALENCE['bootstrap_B'],
                    'tau_support': CONTROL_EQUIVALENCE['tau_support'],
                    'z0': CONTROL_EQUIVALENCE['z0'],
                    'eps': CONTROL_EQUIVALENCE['eps']
                }
            else:
                modern_runs.append(name)
            
            ctrl_eq_values[name] = key_subset
            validation_results['control_equivalence'][name] = key_subset
        
        validation_results['legacy_runs'] = legacy_runs
        
        if legacy_runs:
            print(f"    ⚠️ Legacy runs detected (control_equivalence assumed from SSOT): {legacy_runs}")
            validation_results['warnings'].append(f"Legacy runs: {legacy_runs}")
        
        # Modern runs끼리만 비교 (값이 있는 것들)
        if len(modern_runs) >= 2:
            modern_values = [ctrl_eq_values[name] for name in modern_runs]
            unique_modern = list(set(json.dumps(v, sort_keys=True) for v in modern_values))
            
            if len(unique_modern) != 1:
                # Sensitivity mode 체크: Day3만 bootstrap_B가 다른 경우 허용
                # Day3는 Gate1 결과(B=20)를 사용하므로 B=100 민감도 실험에서 차이 발생
                day3_B = ctrl_eq_values.get('day3', {}).get('bootstrap_B')
                other_Bs = [ctrl_eq_values[name].get('bootstrap_B') for name in modern_runs if name != 'day3']
                
                is_sensitivity_mode = (
                    day3_B == 20 and 
                    len(set(other_Bs)) == 1 and 
                    other_Bs[0] != 20
                )
                
                if is_sensitivity_mode:
                    # bootstrap_B 외의 다른 필드들이 모두 일치하는지 확인
                    def without_bootstrap_B(d):
                        return {k: v for k, v in d.items() if k != 'bootstrap_B'}
                    
                    modern_values_no_B = [without_bootstrap_B(ctrl_eq_values[name]) for name in modern_runs]
                    unique_no_B = list(set(json.dumps(v, sort_keys=True) for v in modern_values_no_B))
                    
                    if len(unique_no_B) == 1:
                        print(f"    ✅ PASSED: Sensitivity mode detected (Day3 B={day3_B}, others B={other_Bs[0]})")
                        print(f"       All other control_equivalence fields match")
                        validation_results['sensitivity_mode'] = {
                            'enabled': True,
                            'day3_bootstrap_B': day3_B,
                            'treatment_bootstrap_B': other_Bs[0],
                            'note': 'Day3 uses Gate1 results (B=20), treatments use sensitivity B'
                        }
                    else:
                        validation_results['all_passed'] = False
                        validation_results['failures'].append('control_equivalence mismatch (non-bootstrap_B fields)')
                        print(f"    ❌ FAILED: control_equivalence mismatch (fields other than bootstrap_B differ)")
                        for name in modern_runs:
                            print(f"       {name}: {ctrl_eq_values[name]}")
                        raise ValueError(f"Control Equivalence mismatch among modern runs: {modern_runs}")
                else:
                    validation_results['all_passed'] = False
                    validation_results['failures'].append('control_equivalence mismatch among modern runs')
                    print(f"    ❌ FAILED: control_equivalence mismatch among modern runs")
                    for name in modern_runs:
                        print(f"       {name}: {ctrl_eq_values[name]}")
                    raise ValueError(f"Control Equivalence mismatch among modern runs: {modern_runs}")
            else:
                print(f"    ✅ PASSED: control_equivalence consistent (modern: {modern_runs}, legacy: {legacy_runs})")
        
        # 2. Teacher Support SHA256 동일성 검증
        print("  [2/4] Checking teacher_support_sha256...")
        sha_values = {}
        for name, manifest in manifests.items():
            sha = manifest.get('teacher_support_sha256')
            sha_values[name] = sha
            validation_results['teacher_support_sha256'][name] = sha
        
        # None이 아닌 값들만 비교
        non_none_sha = {k: v for k, v in sha_values.items() if v is not None}
        unique_sha = list(set(non_none_sha.values()))
        
        if len(unique_sha) > 1:
            validation_results['all_passed'] = False
            validation_results['failures'].append('teacher_support_sha256 mismatch')
            print(f"    ❌ FAILED: teacher_support_sha256 mismatch")
            for name, val in sha_values.items():
                print(f"       {name}: {val}")
            raise ValueError(f"Teacher support SHA256 mismatch!\nValues: {sha_values}")
        
        if non_none_sha:
            print(f"    ✅ PASSED: teacher_support_sha256 identical ({len(non_none_sha)} runs have SHA)")
        else:
            print(f"    ⚠️ WARNING: No runs have teacher_support_sha256")
            validation_results['warnings'].append("No teacher_support_sha256 found")
        
        # 3. Feature/Target names 동일성 검증
        print("  [3/4] Checking feature_names/target_names...")
        for key in ['feature_names', 'target_names']:
            values = {}
            for name, manifest in manifests.items():
                ssot = manifest.get('ssot', {})
                val = ssot.get(key, [])
                values[name] = val
                validation_results[key][name] = val
            
            # 리스트를 문자열로 변환해서 비교
            unique_vals = list(set(str(v) for v in values.values()))
            if len(unique_vals) != 1:
                validation_results['all_passed'] = False
                validation_results['failures'].append(f'{key} mismatch')
                print(f"    ❌ FAILED: {key} mismatch")
                raise ValueError(f"{key} mismatch across runs!")
        print(f"    ✅ PASSED: feature_names/target_names identical")
        
        # 4. dx_key_used 검증 (R3와 CC만 해당 - preflight_qc가 있는 경우)
        print("  [4/4] Checking dx_key_used...")
        for name in ['treat_cc', 'treat_r3']:
            manifest = manifests[name]
            preflight = manifest.get('preflight_qc', {})
            dx_equiv = preflight.get('dx_equivalence', {})
            dx_key = dx_equiv.get('dx_key_used', 'unknown')
            validation_results['dx_key_used'][name] = dx_key
            
            if dx_key != 'train_dx_savgol' and dx_key != 'unknown':
                validation_results['all_passed'] = False
                validation_results['failures'].append(f'{name} dx_key_used != train_dx_savgol')
                print(f"    ❌ FAILED: {name} used dx_key={dx_key}, expected train_dx_savgol")
                raise ValueError(f"{name} did not use train_dx_savgol!")
        print(f"    ✅ PASSED: dx_key_used = train_dx_savgol (or not applicable)")
        
        # 검증 결과 저장
        self.equality_validation = validation_results
        
        print(f"\n  ✅ All Control Equivalence validations PASSED")
        print(f"     - 4 runs use identical SSOT settings")
        print(f"     - Safe to compare aug_pure effects")
    
    
    def _compute_comparison(self) -> Dict[str, Any]:
        """R3 비교 계산"""
        cfg = self.config
        
        teacher_active = self.teacher_support.astype(bool)
        
        # Masks
        stable_core_mask = teacher_active & (self.z_before >= cfg.z0)
        fragile_pool_mask = teacher_active & (self.z_before < cfg.z0)
        
        dynamics_target_indices = [1, 3]
        dynamics_mask = np.zeros_like(teacher_active, dtype=bool)
        for ti in dynamics_target_indices:
            if ti < self.z_before.shape[1]:
                dynamics_mask[:, ti] = teacher_active[:, ti]
        
        fragile_dynamics_mask = fragile_pool_mask & dynamics_mask
        stable_dynamics_mask = stable_core_mask & dynamics_mask
        
        # Effect 계산
        r3_aug_pure = self.z_treat_r3 - self.z_ctrl250
        r3_vs_cc = self.z_treat_r3 - self.z_treat_cc
        r3_total = self.z_treat_r3 - self.z_before
        cc_aug_pure = self.z_treat_cc - self.z_ctrl250
        
        # fragile∩dynamics 통계
        fragile_dyn_r3_pure = compute_stats(r3_aug_pure, fragile_dynamics_mask, 'fragile_dyn')
        fragile_dyn_r3_vs_cc = compute_stats(r3_vs_cc, fragile_dynamics_mask, 'fragile_dyn')
        fragile_dyn_r3_total = compute_stats(r3_total, fragile_dynamics_mask, 'fragile_dyn')
        fragile_dyn_cc_pure = compute_stats(cc_aug_pure, fragile_dynamics_mask, 'fragile_dyn')
        
        # Bootstrap CI 95%
        r3_pure_values = r3_aug_pure[fragile_dynamics_mask]
        bootstrap_ci = compute_bootstrap_ci(r3_pure_values)
        
        comparison = {
            'run_id': self.run_id,
            'source_runs': {
                'baseline_day3': cfg.day3_run_id,
                'control250': cfg.ctrl250_run_id,
                'treatment_cc': cfg.treat_cc_run_id,
                'treatment_r3': cfg.treat_r3_run_id
            },
            
            'effect_decomposition': {
                'r3_aug_pure': {
                    'definition': 'z_r3 - z_ctrl250',
                    'meaning': 'R3 trajectory-consistent augmentation 순수 효과'
                },
                'r3_vs_cc': {
                    'definition': 'z_r3 - z_treat_cc',
                    'meaning': 'R3 vs Channel-Consistent 직접 비교'
                },
                'r3_total': {
                    'definition': 'z_r3 - z_before',
                    'meaning': 'R3 전체 효과'
                },
                'cc_aug_pure': {
                    'definition': 'z_cc - z_ctrl250',
                    'meaning': 'CC augmentation 순수 효과 (Day5 결과)'
                }
            },
            
            'detailed_stats': {
                'fragile_dynamics': {
                    'r3_aug_pure': fragile_dyn_r3_pure,
                    'r3_vs_cc': fragile_dyn_r3_vs_cc,
                    'r3_total': fragile_dyn_r3_total,
                    'cc_aug_pure': fragile_dyn_cc_pure
                },
                'stable_dynamics': {
                    'r3_aug_pure': compute_stats(r3_aug_pure, stable_dynamics_mask, 'stable_dyn'),
                    'r3_vs_cc': compute_stats(r3_vs_cc, stable_dynamics_mask, 'stable_dyn'),
                    'r3_total': compute_stats(r3_total, stable_dynamics_mask, 'stable_dyn')
                }
            },
            
            'primary_results': {
                'fragile_dynamics': {
                    'n': fragile_dyn_r3_pure['n'],
                    'r3_aug_pure_median': fragile_dyn_r3_pure['median'],
                    'r3_vs_cc_median': fragile_dyn_r3_vs_cc['median'],
                    'r3_total_median': fragile_dyn_r3_total['median'],
                    'cc_aug_pure_median': fragile_dyn_cc_pure['median'],
                    'r3_improved_rate': fragile_dyn_r3_pure['improved_rate'],
                    'r3_n_positive': fragile_dyn_r3_pure['n_positive'],
                    'r3_n_negative': fragile_dyn_r3_pure['n_negative'],
                    'r3_zero_crossing': fragile_dyn_r3_pure['zero_crossing']
                }
            },
            
            'bootstrap_ci_95': bootstrap_ci,
            
            'counts': {
                'n_teacher_active': int(teacher_active.sum()),
                'n_fragile_dynamics': int(fragile_dynamics_mask.sum()),
                'n_stable_dynamics': int(stable_dynamics_mask.sum())
            },
            
            'success_criteria': {
                'minimum': {
                    'condition': 'median(r3_aug_pure) > 0 AND n_positive > n_negative',
                    'met': bool(
                        fragile_dyn_r3_pure['median'] is not None and 
                        fragile_dyn_r3_pure['median'] > 0 and
                        fragile_dyn_r3_pure['n_positive'] > fragile_dyn_r3_pure['n_negative']
                    )
                },
                'strong': {
                    'condition': 'minimum + median(r3_vs_cc) > 0',
                    'met': bool(
                        fragile_dyn_r3_pure['median'] is not None and 
                        fragile_dyn_r3_pure['median'] > 0 and
                        fragile_dyn_r3_vs_cc['median'] is not None and
                        fragile_dyn_r3_vs_cc['median'] > 0
                    )
                },
                'ci_check': {
                    'condition': 'bootstrap CI 95% lower bound > 0',
                    'met': bool(bootstrap_ci['lower'] is not None and bootstrap_ci['lower'] > 0)
                }
            }
        }
        
        # Interpretation
        interpretation = []
        
        if fragile_dyn_r3_pure['median'] is not None:
            if fragile_dyn_r3_pure['median'] > 0:
                interpretation.append(f"R3 aug_pure > 0 ({fragile_dyn_r3_pure['median']:.4f}) → R3 순수 효과 있음")
            elif fragile_dyn_r3_pure['median'] == 0 or fragile_dyn_r3_pure['zero_crossing']:
                interpretation.append(f"R3 aug_pure ≈ 0 (zero-crossing) → R3 효과 불확실")
            else:
                interpretation.append(f"R3 aug_pure < 0 ({fragile_dyn_r3_pure['median']:.4f}) → R3 부작용")
        
        if fragile_dyn_r3_vs_cc['median'] is not None:
            if fragile_dyn_r3_vs_cc['median'] > 0:
                interpretation.append(f"R3 > CC ({fragile_dyn_r3_vs_cc['median']:.4f}) → R3가 CC보다 우수")
            else:
                interpretation.append(f"R3 ≤ CC ({fragile_dyn_r3_vs_cc['median']:.4f}) → CC와 동등 또는 열등")
        
        if bootstrap_ci['crosses_zero']:
            interpretation.append("Bootstrap CI 95%가 0을 포함 → 통계적 유의성 부족")
        else:
            interpretation.append(f"Bootstrap CI 95%: [{bootstrap_ci['lower']:.4f}, {bootstrap_ci['upper']:.4f}]")
        
        comparison['interpretation'] = interpretation
        
        # Console output
        print("\n" + "="*60)
        print("  R3 COMPARISON RESULTS (fragile∩dynamics)")
        print("="*60)
        print(f"  n = {fragile_dyn_r3_pure['n']}")
        print(f"  R3 aug_pure median: {fragile_dyn_r3_pure['median']}")
        print(f"  R3 vs CC median: {fragile_dyn_r3_vs_cc['median']}")
        print(f"  CC aug_pure median: {fragile_dyn_cc_pure['median']}")
        print(f"  R3 improved rate: {fragile_dyn_r3_pure['improved_rate']}")
        print(f"  Bootstrap CI 95%: [{bootstrap_ci['lower']}, {bootstrap_ci['upper']}]")
        print("-"*60)
        for interp in interpretation:
            print(f"  → {interp}")
        print("="*60)
        
        return comparison
    
    def _save_comparison(self, comparison: Dict):
        """비교 결과 저장"""
        cfg = self.config
        
        # equality_validation 결과를 comparison에 추가
        comparison['equality_validation'] = self.equality_validation
        
        # comparison_r3.json
        with open(self.results_dir / 'comparison_r3.json', 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"  ✅ Saved: comparison_r3.json")
        
        # Code hash
        source_files = [
            Path(__file__),
            self.project_root / 'src' / 'sindy' / 'esindy.py',
            self.project_root / 'src' / 'sindy' / 'library.py',
        ]
        code_hash = {}
        for src_file in source_files:
            if src_file.exists():
                code_hash[src_file.name] = compute_file_hash(src_file)
        
        # teacher_support hash
        day3_dir = self._get_results_dir(cfg.day3_run_id)
        teacher_support_path = day3_dir / 'teacher_support.npy'
        teacher_support_sha256 = compute_file_hash(teacher_support_path) if teacher_support_path.exists() else None
        
        # manifest.json (equality_validation 포함)
        manifest = {
            'phase': 'phase35',
            'day': 6,
            'mode': 'compare_r3',
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'source_runs': comparison['source_runs'],
            'ssot': self.day3_manifest.get('ssot', {}),
            'control_equivalence': get_control_equivalence(bootstrap_B=cfg.bootstrap_B),
            'theta_policy': THETA_POLICY,
            'equality_validation': self.equality_validation,
            'code_hash': code_hash,
            'teacher_support_sha256': teacher_support_sha256
        }
        
        with open(self.results_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"  ✅ Saved: manifest.json")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 3.5 Day6: R3 Trajectory-Consistent Augmentation'
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['r3_treat', 'compare_r3'],
                        help='Experiment mode')
    parser.add_argument('--day3_run_id', type=str, required=True,
                        help='Day3 run_id (baseline)')
    parser.add_argument('--dataset_version', type=str, default='cartpole_ood_v1')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--track', type=str, default='standardized')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--note', type=str, default='day6')
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--bootstrap_B', type=int, default=20,
                        help='Bootstrap iterations (default=20, use 100 for sensitivity check)')
    
    # Compare mode 전용
    parser.add_argument('--ctrl250_run_id', type=str, default='',
                        help='Control-250 run_id (for compare mode)')
    parser.add_argument('--treat_cc_run_id', type=str, default='',
                        help='Treatment CC (Day4) run_id (for compare mode)')
    parser.add_argument('--treat_r3_run_id', type=str, default='',
                        help='Treatment R3 run_id (for compare mode)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = Day6Config(
        mode=args.mode,
        day3_run_id=args.day3_run_id,
        dataset_version=args.dataset_version,
        dataset_path=args.dataset_path,
        track=args.track,
        seed=args.seed,
        note=args.note,
        threshold=args.threshold,
        bootstrap_B=args.bootstrap_B,
        ctrl250_run_id=args.ctrl250_run_id,
        treat_cc_run_id=args.treat_cc_run_id,
        treat_r3_run_id=args.treat_r3_run_id
    )
    
    if config.mode == 'r3_treat':
        runner = Day6TreatRunner(config)
    elif config.mode == 'compare_r3':
        if not config.ctrl250_run_id or not config.treat_cc_run_id or not config.treat_r3_run_id:
            print("❌ Error: compare_r3 mode requires --ctrl250_run_id, --treat_cc_run_id, --treat_r3_run_id")
            return 1
        runner = Day6CompareRunner(config)
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