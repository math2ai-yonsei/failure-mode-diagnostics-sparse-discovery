"""
Gate3 Generative Augmentation Runner

목표: Learned generative augmentation이 Gate2 ceiling을 돌파하는지 검증

설계서: Gate3_Design_v1.3.md
- D1: Pipeline bring-up (GMM, Track A only, B=20)
- D2: Primary result (Track A+B, B=100, 2-seed)
- D3: Reproducibility & ablation

핵심:
- GMM sampler로 IC+params 6D 생성
- ODE 시뮬레이션으로 trajectory 생성
- Track A: 상위 10% error 제거 (OOD 방지)
- Track B: Information-seeking ranking (D2에서 활성화)

산출물:
- manifest.json, metrics.json, comparison_gen.json
- generated_pool.npz, selected_trajectories.npz
- z_after.npy, inc_prob_after.npy

Author: Claude (Gate3 Implementation)
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
from typing import Dict, List, Optional, Any, Tuple
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.mixture import GaussianMixture

# 프로젝트 모듈
from src.sindy.library import SINDyLibrary
from src.sindy.optimizer import ColumnScaler
from src.sindy.esindy import ESINDyEnsemble
from src.utils.derivatives import compute_derivatives_savgol, SAVGOL_CONFIG


# ============================================================
# SSOT Constants (Gate2와 동일)
# ============================================================
DEFAULT_TARGET_NAMES = ["x_dot", "x_ddot", "theta_dot", "theta_ddot"]
DEFAULT_TAU_SUPPORT = 0.5
DEFAULT_Z0 = 2.0
DEFAULT_EPS = 1e-12
DEFAULT_BOOTSTRAP_B = 20

# Control Equivalence SSOT (Day3/Gate2와 동일)
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

# Gate3 Generation Config
GATE3_CONFIG = {
    # GMM Sampler
    'gmm_n_components': 3,
    'gmm_covariance_type': 'full',
    'gmm_random_state': 42,
    
    # Pool Generation
    'target_n_accept': 200,           # D1=200, D2: override via CLI
    'max_pool_attempts': 15000,       # P0-2 FIX: 5000→15000 (acceptance ~26%)
    'pool_batch_size': 100,
    
    # Quality Control (dataset 생성과 동일)
    'qc': {
        'max_x': 10.0,
        'max_theta': 3.1,
        'max_velocity': 30.0,
    },
    
    # Simulation (dataset meta와 동일)
    'simulation': {
        'dt': 0.02,
        'duration': 2.0,
        'T': 101,
        'method': 'RK45',
        'rtol': 1e-8,
        'atol': 1e-10,
    },
    
    # dx Computation (SAVGOL_CONFIG와 동일)
    'savgol': {
        'window': 11,
        'polyorder': 3,
    },
    
    # Track A Selection (v1.3: 상위 10%만 제거)
    'track_a': {
        'reject_ratio': 0.10,
        'min_after_a': 800,  # Track B용 최소 후보
        'error_type': 'normalized_rmse',  # dynamics target, per-target std
    },
    
    # Track B Selection (D2에서 활성화)
    'track_b': {
        'enabled': False,  # D1에서는 비활성화
        'n_select': 200,
        'weights': {
            'residual': 0.4,
            'coverage': 0.3,
            'identifiability': 0.3,
        },
    },
    
    # Physics params (fixed, train_params에서 m_cart, m_pole만 변동)
    'fixed_physics': {
        'L': 0.5,
        'g': 9.81,
        'b_cart': 0.1,
        'b_pole': 0.01,
    },
}


# ============================================================
# Configuration Dataclass
# ============================================================

@dataclass
class Gate3Config:
    """Gate3 실험 설정"""
    mode: str = "gen_treat"  # 'gen_treat', 'compare_gen'
    day3_run_id: str = ""
    dataset_version: str = "cartpole_ood_v1"
    dataset_path: str = ""
    track: str = "standardized"
    method: str = "stable_core"
    variant: str = "IC"  # 'IC' (u 재사용) or 'ICU' (u 새로 생성)
    tau_support: float = DEFAULT_TAU_SUPPORT
    z0: float = DEFAULT_Z0
    eps: float = DEFAULT_EPS
    bootstrap_B: int = DEFAULT_BOOTSTRAP_B
    threshold: float = 0.05
    seed: int = 0
    note: str = "gate3"
    n_train: int = 10
    
    # D2 options
    target_n_accept: int = 200  # D1=200, D2=2000+
    n_select: int = 200         # Final selection (CTRL250=240 for n_total=250)
    
    # D1 specific
    track_b_enabled: bool = False  # D1: False, D2: True
    
    # D1 Rebaseline: Pool reuse & selection mode
    pool_source: str = ""  # Path to existing pool (empty = generate new)
    selection_mode: str = "random"  # 'random', 'track_a_filtered_random', 'track_b', 'd_optimal'
    reject_ratio: float = 0.10  # Track A: reject top 10% error
    
    # Track B parameters v0.2 (GPT P0 fix)
    track_b_alpha: float = 0.3  # x_penalty weight
    track_b_diversity_mode: str = 'top_m_diversity'  # 'score_only' or 'top_m_diversity'
    track_b_top_m_ratio: float = 5.0  # M = ratio × n_select (default 5.0)
    track_b_score_floor: Optional[float] = None  # Minimum score threshold (None = no floor)
    
    # Track B v0.3 D-optimal parameters
    track_b_dopt_lambda: float = 1e-6  # Regularization for logdet
    track_b_dopt_use_teacher_intersection: bool = True  # F = fragile ∩ teacher_active
    track_b_dopt_pre_gate_mode: str = 'score'  # 'score' or 'none'
    track_b_dopt_gram_energy_mode: str = 'raw'  # v0.3.2: 'raw' or 'unit_trace'
    track_b_dopt_trace_power: float = 1.0  # v0.3.3: G_i / (trace(G_i)**p + eps), p=0=raw, p=1=unit_trace
    
    # Compare mode
    ctrl250_run_id: str = ""
    gen_run_ids: List[str] = field(default_factory=list)
    
    # Compare mode: fragile_pairs fail-fast
    fragile_pairs_source: str = ""  # Path to existing fragile_pairs.json (empty = fail-fast)
    allow_fragile_compute: bool = False  # Override: allow compute if not exists


# ============================================================
# Helper Functions
# ============================================================

def create_rng_streams(base_seed: int) -> Dict[str, np.random.Generator]:
    """
    Create 3 independent RNG streams using SeedSequence.
    
    P0-1 SSOT: RNG 분리로 재현성 보장
    - rng_pool: Pool generation (GMM sampling, u selection)
    - rng_select: Final selection (random sampling from Track A passed)
    - rng_bootstrap: E-SINDy bootstrap resampling
    
    Args:
        base_seed: Base seed for SeedSequence
        
    Returns:
        Dict with 'pool', 'select', 'bootstrap' RNG generators
    """
    from numpy.random import SeedSequence, default_rng
    
    ss = SeedSequence(base_seed)
    child_seeds = ss.spawn(3)
    
    return {
        'pool': default_rng(child_seeds[0]),
        'select': default_rng(child_seeds[1]),
        'bootstrap': default_rng(child_seeds[2]),
        '_seed_sequence_entropy': ss.entropy,  # For manifest recording
    }


def compute_array_hash(arr: np.ndarray) -> str:
    """Compute SHA256 hash of numpy array (for SSOT verification)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


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


def safe_float(val) -> Optional[float]:
    """numpy scalar를 Python float로 안전하게 변환"""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    return val


def _json_default(obj):
    """JSON 직렬화 헬퍼"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap angle to (-π, π]"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def get_control_equivalence(bootstrap_B: int = 20) -> Dict:
    """Control equivalence 설정 반환 (bootstrap_B 오버라이드 가능)"""
    ce = CONTROL_EQUIVALENCE.copy()
    ce['bootstrap_B'] = bootstrap_B
    return ce


# ============================================================
# v2.1 FIX: Normalization Stats Loading
# ============================================================

def load_norm_stats(
    dataset_version: str,
    system: str = 'cartpole',
    project_root: Path = None
) -> Dict[str, Any]:
    """
    Load normalization statistics from norm_stats.json.
    
    v2.1 FIX: Gate1 teacher coefficients were trained on normalized inputs.
    Gate3 alignment error calculation must use the same normalization.
    
    Gate1 relationship:
        x_norm = (x - state_mean) / state_std
        u_norm = (u - input_mean) / input_std
        Theta = library(x_norm, u_norm)
        dx_pred = Theta @ coef + dx_mean
    
    Args:
        dataset_version: Dataset version (e.g., 'cartpole_ood_v1')
        system: System name (default: 'cartpole')
        project_root: Project root path
        
    Returns:
        Dict with normalization stats
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    
    norm_stats_path = project_root / 'data' / system / dataset_version / 'norm_stats.json'
    
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm_stats.json not found: {norm_stats_path}")
    
    with open(norm_stats_path, 'r', encoding='utf-8') as f:
        norm_stats = json.load(f)
    
    # Convert lists to numpy arrays for easy use
    for key in ['state', 'input', 'derivative_dx', 'derivative_dx_savgol']:
        if key in norm_stats:
            norm_stats[key]['mean'] = np.array(norm_stats[key]['mean'])
            norm_stats[key]['std'] = np.array(norm_stats[key]['std'])
    
    return norm_stats


def normalize_array(data: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Normalize data using precomputed statistics.
    
    Formula: (data - mean) / std
    
    Args:
        data: Array of shape (..., D)
        stats: Dict with 'mean' and 'std' of shape (D,)
        
    Returns:
        Normalized array, same shape as input
    """
    mean = np.asarray(stats['mean'])
    std = np.asarray(stats['std'])
    return (data - mean) / std


# ============================================================
# P0-2: dx Equivalence Test
# ============================================================

def test_dx_equivalence(
    dataset_dx: np.ndarray,
    computed_dx: np.ndarray,
    sample_indices: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Test equivalence between dataset dx and computed dx.
    
    P0-2: Prove that generated dx (savgol 11/3) matches dataset's train_dx_savgol.
    
    Args:
        dataset_dx: dx from dataset (train_dx_savgol), shape (N, T, D)
        computed_dx: dx computed from trajectories, shape (N, T, D)
        sample_indices: Indices to compare (if subset)
        
    Returns:
        Dict with equivalence metrics
    """
    if sample_indices is not None:
        dataset_dx = dataset_dx[sample_indices]
    
    diff = computed_dx - dataset_dx
    
    max_abs_diff = float(np.abs(diff).max())
    mean_abs_diff = float(np.abs(diff).mean())
    
    # Relative difference
    denom = np.abs(dataset_dx) + 1e-10
    rel_diff = np.abs(diff) / denom
    max_rel_diff = float(rel_diff.max())
    mean_rel_diff = float(rel_diff.mean())
    
    # Per-dimension stats
    dim_stats = {}
    dim_names = ['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']
    for d, name in enumerate(dim_names):
        dim_diff = diff[..., d]
        dim_stats[name] = {
            'max_abs': float(np.abs(dim_diff).max()),
            'mean_abs': float(np.abs(dim_diff).mean()),
            'std': float(dim_diff.std()),
        }
    
    # Pass criterion: max_abs_diff < 1e-10 (should be identical if same pipeline)
    is_equivalent = max_abs_diff < 1e-6
    
    result = {
        'is_equivalent': is_equivalent,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'per_dim': dim_stats,
        'n_samples': dataset_dx.shape[0],
        'note': 'Dataset dx vs computed dx equivalence test',
    }
    
    return result


# ============================================================
# P0-4: CTRL250 Fair Comparison Assert (Strengthened)
# ============================================================

CTRL250_COMPARISON_KEYS_CRITICAL = [
    # P0-4: Critical keys (must match exactly)
    'threshold',
    'tau_support',
    'z0',
    'eps',
    'teacher_support_sha256',
    'resample_unit',
    'seed_rule',
]

CTRL250_COMPARISON_KEYS_STRONG = [
    # P0-2 FIX: These should match for fair comparison
    'bootstrap_B',
    'n_original',
    'n_augmented', 
    'n_total',
]

CTRL250_COMPARISON_KEYS_INFORMATIONAL = [
    # Informational (log mismatch but don't fail)
    'selection_mode',
]

def load_ctrl250_manifest(ctrl250_run_id: str, project_root: Path, config: 'Gate3Config') -> Dict:
    """Load CTRL250 manifest for fair comparison."""
    ctrl250_dir = (
        project_root / 'results' / config.dataset_version / 'phase35' /
        config.track / config.method / f"n{config.n_train}" / f"seed{config.seed}" / ctrl250_run_id
    )
    
    manifest_path = ctrl250_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"CTRL250 manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def assert_ctrl250_equivalence(
    gate3_config: Dict,
    ctrl250_manifest: Dict,
    gen_n_total: int = None,
    gen_n_original: int = None,
    gen_n_augmented: int = None,
    gen_bootstrap_B: int = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Assert fair comparison conditions between Gate3 and CTRL250.
    
    P0-2 FIX: Strengthened equivalence check including bootstrap_B, n_orig, n_aug
    P0-4: CTRL250 manifest equivalence assert
    
    Args:
        gate3_config: Gate3 config dict
        ctrl250_manifest: CTRL250 manifest dict
        gen_n_total: GEN run's n_total
        gen_n_original: GEN run's n_original (n_train)
        gen_n_augmented: GEN run's n_augmented (n_select)
        gen_bootstrap_B: GEN run's bootstrap_B
        strict: If True, raise error on critical mismatch
        
    Returns:
        Dict with comparison results
    """
    ctrl_ce = ctrl250_manifest.get('control_equivalence', {})
    ctrl_hyper = ctrl250_manifest.get('hyperparameters', {})
    ctrl_data = ctrl250_manifest.get('data_config', {})
    
    critical_mismatches = []
    strong_mismatches = []
    info_mismatches = []
    matches = []
    
    # Check critical keys
    critical_checks = {
        'threshold': (gate3_config.get('threshold'), ctrl_hyper.get('threshold')),
        'tau_support': (gate3_config.get('tau_support'), ctrl_ce.get('tau_support')),
        'z0': (gate3_config.get('z0'), ctrl_ce.get('z0')),
        'eps': (gate3_config.get('eps'), ctrl_ce.get('eps')),
        'teacher_support_sha256': (
            gate3_config.get('teacher_support_sha256'),
            ctrl250_manifest.get('teacher_support_sha256')
        ),
        'resample_unit': ('trajectory', ctrl_ce.get('resample_unit')),
        'seed_rule': ('seed_b = base_seed + b', ctrl_ce.get('seed_rule')),
    }
    
    for key, (gate3_val, ctrl_val) in critical_checks.items():
        if gate3_val == ctrl_val:
            matches.append(key)
        else:
            critical_mismatches.append({
                'key': key,
                'gate3': gate3_val,
                'ctrl250': ctrl_val,
            })
    
    # P0-2 FIX: Check strong keys (bootstrap_B, n_orig, n_aug, n_total)
    ctrl_n_total = ctrl_data.get('n_trajectories', 250)
    ctrl_bootstrap_B = ctrl_hyper.get('bootstrap_B', 100)
    ctrl_n_original = ctrl250_manifest.get('n_train', 10)
    ctrl_n_augmented = ctrl_n_total - ctrl_n_original
    
    strong_checks = {
        'bootstrap_B': (gen_bootstrap_B, ctrl_bootstrap_B),
        'n_original': (gen_n_original, ctrl_n_original),
        'n_augmented': (gen_n_augmented, ctrl_n_augmented),
        'n_total': (gen_n_total, ctrl_n_total),
    }
    
    for key, (gate3_val, ctrl_val) in strong_checks.items():
        if gate3_val is not None:
            if gate3_val == ctrl_val:
                matches.append(key)
            else:
                strong_mismatches.append({
                    'key': key,
                    'gate3': gate3_val,
                    'ctrl250': ctrl_val,
                    'note': f'P0-2: {key} mismatch may affect comparison validity',
                })
    
    result = {
        'is_equivalent': len(critical_mismatches) == 0,
        'is_strongly_equivalent': len(critical_mismatches) == 0 and len(strong_mismatches) == 0,
        'n_critical_matches': len([m for m in matches if m in [k for k in critical_checks.keys()]]),
        'n_critical_mismatches': len(critical_mismatches),
        'n_strong_mismatches': len(strong_mismatches),
        'n_info_mismatches': len(info_mismatches),
        'critical_matches': matches,
        'critical_mismatches': critical_mismatches,
        'strong_mismatches': strong_mismatches,
        'info_mismatches': info_mismatches,
        'ctrl_n_total': ctrl_n_total,
        'ctrl_bootstrap_B': ctrl_bootstrap_B,
    }
    
    if strict and len(critical_mismatches) > 0:
        mismatch_str = '\n'.join([
            f"  {m['key']}: Gate3={m['gate3']} vs CTRL250={m['ctrl250']}"
            for m in critical_mismatches
        ])
        raise ValueError(f"CTRL250 fair comparison FAILED (critical):\n{mismatch_str}")
    
    return result


def load_teacher_coefficients(teacher_run_id: str, project_root: Path, config: 'Gate3Config') -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load teacher coefficients from Gate1 sindy_coefficients.csv.
    
    Args:
        teacher_run_id: Gate1 teacher run_id
        project_root: Project root path
        config: Gate3 config
        
    Returns:
        coefficients: (n_features, n_targets) array
        feature_names: List of feature names
        target_names: List of target names
    """
    # Build path to Gate1 results
    teacher_dir = (
        project_root / 'results' / config.dataset_version / 'gate1' /
        config.track / 'esindy' / f"n{config.n_train}" / f"seed{config.seed}" / teacher_run_id
    )
    
    coef_path = teacher_dir / 'sindy_coefficients.csv'
    
    if not coef_path.exists():
        raise FileNotFoundError(f"Teacher coefficients not found: {coef_path}")
    
    # Read CSV
    import csv
    with open(coef_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # term_name, x_dot, x_ddot, theta_dot, theta_ddot
        
        target_names = header[1:]  # ['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']
        feature_names = []
        coefficients = []
        
        for row in reader:
            feature_names.append(row[0])
            coefficients.append([float(v) for v in row[1:]])
    
    coefficients = np.array(coefficients)  # (n_features, n_targets)
    
    print(f"  Loaded teacher coefficients: {coef_path.name}")
    print(f"    Shape: {coefficients.shape}")
    print(f"    Features: {len(feature_names)}, Targets: {len(target_names)}")
    
    return coefficients, feature_names, target_names


# ============================================================
# CartPole Dynamics (Inline - simulator import 대체)
# ============================================================

def cartpole_dynamics(t: float, state: np.ndarray, u: float, params: Dict) -> np.ndarray:
    """
    Cart-Pole dynamics using Euler-Lagrange equations.
    
    State: [x, x_dot, theta, theta_dot]
    Params: m_cart, m_pole, L, g, b_cart, b_pole
    """
    x, x_dot, theta, theta_dot = state
    
    mc = params['m_cart']
    mp = params['m_pole']
    L = params['L']
    g = params['g']
    b_cart = params.get('b_cart', 0.0)
    b_pole = params.get('b_pole', 0.0)
    
    mt = mc + mp
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    # Denominator
    denom = L * (mc + mp * sin_t**2)
    
    # Effective force with friction
    u_eff = u - b_cart * x_dot
    tau_pole = -b_pole * theta_dot
    
    # Centrifugal term
    centrifugal = mp * L * theta_dot**2 * sin_t
    
    # Solve 2x2 system
    rhs1 = u_eff + centrifugal
    rhs2 = g * sin_t + tau_pole / (mp * L) if mp * L > 0 else g * sin_t
    
    x_ddot = (L * rhs1 - mp * L * cos_t * rhs2) / denom
    theta_ddot = (mt * rhs2 - cos_t * rhs1) / denom
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=np.float64)


def simulate_trajectory(
    ic: np.ndarray,
    u_sequence: np.ndarray,
    params: Dict,
    dt: float,
    T: int,
    method: str = 'RK45',
    rtol: float = 1e-8,
    atol: float = 1e-10
) -> Tuple[np.ndarray, bool]:
    """
    Simulate Cart-Pole trajectory.
    
    Args:
        ic: Initial condition [x0, x_dot0, theta0, theta_dot0]
        u_sequence: Control input sequence, shape (T,) or (T, 1)
        params: Physics parameters dict
        dt: Time step
        T: Number of time steps
        
    Returns:
        trajectory: (T, 4) state trajectory
        success: True if simulation completed without issues
    """
    u_seq = u_sequence.flatten()
    if len(u_seq) != T:
        raise ValueError(f"u_sequence length {len(u_seq)} != T {T}")
    
    t_span = (0, (T - 1) * dt)
    t_eval = np.linspace(0, (T - 1) * dt, T)
    
    # Create interpolated control function
    def get_u(t):
        idx = int(t / dt)
        idx = min(idx, T - 1)
        return u_seq[idx]
    
    # ODE function
    def ode_fn(t, state):
        u = get_u(t)
        return cartpole_dynamics(t, state, u, params)
    
    try:
        sol = solve_ivp(
            ode_fn,
            t_span,
            ic,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
        
        if not sol.success:
            return np.zeros((T, 4)), False
        
        trajectory = sol.y.T  # (T, 4)
        
        # Wrap theta to (-π, π]
        trajectory[:, 2] = wrap_angle(trajectory[:, 2])
        
        return trajectory, True
        
    except Exception as e:
        return np.zeros((T, 4)), False


# ============================================================
# Quality Control
# ============================================================

def check_trajectory_quality(
    trajectory: np.ndarray,
    qc_config: Dict
) -> Tuple[bool, str]:
    """
    Check if trajectory passes quality control.
    
    Args:
        trajectory: (T, 4) state trajectory
        qc_config: QC thresholds dict
        
    Returns:
        passed: True if all checks pass
        reason: Rejection reason (empty if passed)
    """
    max_x = qc_config['max_x']
    max_theta = qc_config['max_theta']
    max_velocity = qc_config['max_velocity']
    
    # Check for NaN/Inf
    if not np.isfinite(trajectory).all():
        return False, "nan_or_inf"
    
    # Check x bounds
    if np.abs(trajectory[:, 0]).max() > max_x:
        return False, "max_x_exceeded"
    
    # Check theta bounds
    if np.abs(trajectory[:, 2]).max() > max_theta:
        return False, "max_theta_exceeded"
    
    # Check velocity bounds
    if np.abs(trajectory[:, 1]).max() > max_velocity:
        return False, "max_x_dot_exceeded"
    
    if np.abs(trajectory[:, 3]).max() > max_velocity:
        return False, "max_theta_dot_exceeded"
    
    return True, ""


# ============================================================
# GMM Sampler for IC + Params
# ============================================================

class GMMProposalSampler:
    """
    GMM-based proposal distribution for IC + params (6D).
    
    Learns distribution from training data:
    - IC: [x0, x_dot0, theta0, theta_dot0] (4D)
    - params: [m_cart, m_pole] (2D)
    """
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        random_state: int = 42
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None
        self._fitted = False
        self._train_mean = None
        self._train_std = None
    
    def fit(
        self,
        train_x: np.ndarray,
        train_params: np.ndarray
    ) -> 'GMMProposalSampler':
        """
        Fit GMM to training data.
        
        Args:
            train_x: Training trajectories, shape (N, T, 4)
            train_params: Training parameters, shape (N, 2)
        """
        N = train_x.shape[0]
        
        # Extract initial conditions
        ic = train_x[:, 0, :]  # (N, 4)
        
        # Combine IC + params
        data = np.hstack([ic, train_params])  # (N, 6)
        
        # Store statistics for bounded sampling
        self._train_mean = data.mean(axis=0)
        self._train_std = data.std(axis=0)
        self._train_min = data.min(axis=0)
        self._train_max = data.max(axis=0)
        
        # P1-1: Automatic component adjustment for small sample sizes
        # Rule: n_samples >= 5 * K for stable full covariance
        effective_n_components = self.n_components
        effective_covariance = self.covariance_type
        
        if N < 5 * self.n_components:
            # Reduce components or switch to diagonal covariance
            if N >= 5:
                effective_n_components = max(1, N // 5)
                print(f"  ⚠️ GMM: Reduced components {self.n_components} → {effective_n_components} (N={N} < 5*K)")
            else:
                effective_n_components = 1
                effective_covariance = 'diag'
                print(f"  ⚠️ GMM: Switched to K=1, diag covariance (N={N} too small)")
        
        # Fit GMM with regularization for stability
        self.gmm = GaussianMixture(
            n_components=effective_n_components,
            covariance_type=effective_covariance,
            random_state=self.random_state,
            max_iter=200,
            n_init=3,
            reg_covar=1e-6,  # Regularization for numerical stability
        )
        self.gmm.fit(data)
        self._fitted = True
        self._effective_n_components = effective_n_components
        self._effective_covariance = effective_covariance
        
        print(f"  GMM fitted: {N} samples → {effective_n_components} components ({effective_covariance})")
        print(f"  IC range: x=[{ic[:, 0].min():.2f}, {ic[:, 0].max():.2f}], "
              f"theta=[{ic[:, 2].min():.2f}, {ic[:, 2].max():.2f}]")
        print(f"  Params range: m_cart=[{train_params[:, 0].min():.2f}, {train_params[:, 0].max():.2f}], "
              f"m_pole=[{train_params[:, 1].min():.3f}, {train_params[:, 1].max():.3f}]")
        
        return self
    
    def sample(
        self,
        n_samples: int,
        bounds_sigma: float = 3.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample IC + params from GMM.
        
        Args:
            n_samples: Number of samples
            bounds_sigma: Clip to mean ± bounds_sigma * std
            seed: Random seed (None = use GMM's random_state)
            
        Returns:
            ic: Initial conditions, shape (n_samples, 4)
            params: Physics parameters, shape (n_samples, 2)
        """
        if not self._fitted:
            raise RuntimeError("GMM not fitted. Call fit() first.")
        
        if seed is not None:
            self.gmm.random_state = seed
        
        # Sample from GMM
        samples, _ = self.gmm.sample(n_samples)  # (n_samples, 6)
        
        # Apply soft bounds (clip to training range with margin)
        margin = bounds_sigma * self._train_std
        lower = self._train_mean - margin
        upper = self._train_mean + margin
        samples = np.clip(samples, lower, upper)
        
        # Additional physical constraints
        # theta should be in reasonable range
        samples[:, 2] = np.clip(samples[:, 2], -np.pi * 0.8, np.pi * 0.8)
        
        # m_cart, m_pole should be positive
        samples[:, 4] = np.clip(samples[:, 4], 0.5, 3.0)  # m_cart
        samples[:, 5] = np.clip(samples[:, 5], 0.05, 0.25)  # m_pole
        
        ic = samples[:, :4]
        params = samples[:, 4:]
        
        return ic, params


# ============================================================
# Teacher Alignment Score Computation (v2.1 FIX)
# ============================================================

def compute_teacher_alignment_error(
    trajectories: np.ndarray,
    dx_trajectories: np.ndarray,
    u_trajectories: np.ndarray,
    teacher_coefficients: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    dx_std: np.ndarray,
    norm_stats: Dict[str, Any],
    dynamics_target_indices: List[int] = None,
    derivative_key: str = 'derivative_dx_savgol',
) -> np.ndarray:
    """
    Compute teacher alignment error for each trajectory.
    
    v2.1 FIX: Apply Gate1-equivalent normalization!
    
    Gate1 relationship:
        Theta = library(x_norm, u_norm)  # normalized inputs
        dx_pred = Theta @ coef + dx_mean  # predict raw dx
    
    Track A uses normalized RMSE per **dynamics target only** (v1.3 P0-1):
    - dynamics_target_indices = [1, 3] → x_ddot, theta_ddot
    - Kinematics (x_dot, theta_dot) are excluded from error calculation
    
    error = mean over dynamics targets of: sqrt(mean((dx_pred - dx_true)^2)) / std(dx_true)
    
    Args:
        trajectories: Generated trajectories (raw), shape (N, T, 4)
        dx_trajectories: dx for trajectories (raw), shape (N, T, 4)
        u_trajectories: u for trajectories (raw), shape (N, T, 1)
        teacher_coefficients: Teacher model coefficients, shape (n_features, n_targets)
        feature_names: Feature names
        target_names: Target names
        dx_std: Training dx std per target (raw), shape (n_targets,)
        norm_stats: Normalization statistics from norm_stats.json
        dynamics_target_indices: Indices of dynamics targets [1, 3] for x_ddot, theta_ddot
        derivative_key: Key for derivative stats in norm_stats
        
    Returns:
        errors: Per-trajectory normalized RMSE, shape (N,)
    """
    N, T, D = trajectories.shape
    
    # SSOT: dynamics targets only (v1.3 설계서)
    if dynamics_target_indices is None:
        dynamics_target_indices = [1, 3]  # x_ddot, theta_ddot
    
    # v2.1 FIX: Get normalization stats
    state_stats = norm_stats['state']
    input_stats = norm_stats['input']
    dx_stats = norm_stats[derivative_key]
    dx_mean = dx_stats['mean']  # (4,)
    
    errors = np.zeros(N)
    
    for i in range(N):
        traj_raw = trajectories[i]  # (T, 4) raw
        dx_true = dx_trajectories[i]  # (T, 4) raw
        u_raw = u_trajectories[i]  # (T, 1) raw
        
        # v2.1 FIX: Normalize inputs (same as Gate1)
        traj_norm = normalize_array(traj_raw, state_stats)  # (T, 4) normalized
        u_norm = normalize_array(u_raw, input_stats)  # (T, 1) normalized
        
        # Build feature matrix with NORMALIZED inputs
        Theta = _compute_features_manual(traj_norm, u_norm, feature_names)  # (T, n_features)
        
        # v2.1 FIX: Predict raw dx using Gate1 formula
        # dx_raw = Theta(x_norm, u_norm) @ coef + dx_mean
        dx_pred = Theta @ teacher_coefficients + dx_mean  # (T, n_targets)
        
        # Compute normalized RMSE for DYNAMICS TARGETS ONLY (P0-1 fix)
        target_errors = []
        for t_idx in dynamics_target_indices:
            mse = np.mean((dx_pred[:, t_idx] - dx_true[:, t_idx])**2)
            rmse = np.sqrt(mse)
            # Normalize by target std (raw dx_std, NOT normalized)
            norm_rmse = rmse / (dx_std[t_idx] + 1e-10)
            target_errors.append(norm_rmse)
        
        errors[i] = np.mean(target_errors)
    
    return errors


def _compute_features_manual(traj: np.ndarray, u: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """
    Manually compute features matching gate0_min library.
    
    P0-4: Feature order MUST match Gate0_min canonical order.
    
    Features (21) in canonical order:
    1, x, x_dot, theta_dot, sin(theta), cos(theta), u,
    x^2, x*x_dot, x_dot^2, theta_dot^2, x*theta_dot, x_dot*theta_dot,
    x*sin(theta), x*cos(theta), x_dot*sin(theta), x_dot*cos(theta),
    theta_dot*sin(theta), theta_dot*cos(theta), u*sin(theta), u*cos(theta)
    """
    # P0-4: Assert feature order matches expected Gate0_min order
    GATE0_MIN_CANONICAL_ORDER = [
        '1', 'x', 'x_dot', 'theta_dot', 'sin(theta)', 'cos(theta)', 'u',
        'x^2', 'x*x_dot', 'x_dot^2', 'theta_dot^2', 'x*theta_dot', 'x_dot*theta_dot',
        'x*sin(theta)', 'x*cos(theta)', 'x_dot*sin(theta)', 'x_dot*cos(theta)',
        'theta_dot*sin(theta)', 'theta_dot*cos(theta)', 'u*sin(theta)', 'u*cos(theta)'
    ]
    
    if feature_names != GATE0_MIN_CANONICAL_ORDER:
        raise ValueError(
            f"Feature order mismatch!\n"
            f"Expected: {GATE0_MIN_CANONICAL_ORDER}\n"
            f"Got: {feature_names}"
        )
    
    T = traj.shape[0]
    x = traj[:, 0]
    x_dot = traj[:, 1]
    theta = traj[:, 2]
    theta_dot = traj[:, 3]
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    # u from input
    u_flat = u.flatten() if u.ndim > 1 else u
    
    features = np.column_stack([
        np.ones(T),           # 1
        x,                    # x
        x_dot,                # x_dot
        theta_dot,            # theta_dot
        sin_t,                # sin(theta)
        cos_t,                # cos(theta)
        u_flat,               # u
        x**2,                 # x^2
        x * x_dot,            # x*x_dot
        x_dot**2,             # x_dot^2
        theta_dot**2,         # theta_dot^2
        x * theta_dot,        # x*theta_dot
        x_dot * theta_dot,    # x_dot*theta_dot
        x * sin_t,            # x*sin(theta)
        x * cos_t,            # x*cos(theta)
        x_dot * sin_t,        # x_dot*sin(theta)
        x_dot * cos_t,        # x_dot*cos(theta)
        theta_dot * sin_t,    # theta_dot*sin(theta)
        theta_dot * cos_t,    # theta_dot*cos(theta)
        u_flat * sin_t,       # u*sin(theta)
        u_flat * cos_t,       # u*cos(theta)
    ])
    
    return features


# ============================================================
# Pool Generator
# ============================================================

class PoolGenerator:
    """
    Generate candidate trajectory pool using GMM proposal.
    
    Pipeline:
    1. Sample IC + params from GMM
    2. Select u from original training data (GEN-IC) or generate new (GEN-ICU)
    3. Simulate trajectory using CartPole dynamics
    4. Compute dx using savgol
    5. Apply QC filter
    6. Repeat until n_accept >= target
    
    D1 Rebaseline: Supports external RNG injection for reproducibility.
    """
    
    def __init__(
        self,
        gmm_sampler: GMMProposalSampler,
        train_u: np.ndarray,
        config: Dict,
        fixed_physics: Dict,
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,  # D1: External RNG injection
    ):
        """
        Args:
            gmm_sampler: Fitted GMM sampler
            train_u: Training control sequences, shape (N_train, T, 1)
            config: GATE3_CONFIG
            fixed_physics: Fixed physics parameters (L, g, b_cart, b_pole)
            seed: Random seed (used if rng is None)
            rng: External RNG generator (D1 Rebaseline: for stream separation)
        """
        self.gmm_sampler = gmm_sampler
        self.train_u = train_u.copy()
        self.config = config
        self.fixed_physics = fixed_physics
        self.seed = seed
        # D1: Use external RNG if provided, else create from seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        
        self.N_train = train_u.shape[0]
        self.T = train_u.shape[1]
        self.dt = config['simulation']['dt']
    
    def generate_pool(
        self,
        target_n_accept: int,
        max_attempts: int,
        variant: str = "IC",
    ) -> Dict[str, Any]:
        """
        Generate trajectory pool.
        
        Args:
            target_n_accept: Target number of accepted trajectories
            max_attempts: Maximum generation attempts
            variant: 'IC' (reuse training u) or 'ICU' (new random u)
            
        Returns:
            Dict with:
                'trajectories': (n_accept, T, 4)
                'dx': (n_accept, T, 4)
                'params': (n_accept, 2)
                'ic': (n_accept, 4)
                'u': (n_accept, T, 1)
                'u_indices': (n_accept,) - which training u was used (IC only)
                'stats': generation statistics
        """
        print(f"\n[Pool Generation] target={target_n_accept}, max_attempts={max_attempts}")
        print(f"  Variant: {variant}")
        
        sim_config = self.config['simulation']
        qc_config = self.config['qc']
        savgol_config = self.config['savgol']
        
        accepted_trajectories = []
        accepted_dx = []
        accepted_params = []
        accepted_ic = []
        accepted_u = []
        accepted_u_indices = []
        
        rejection_counts = {}
        n_attempts = 0
        batch_size = self.config['pool_batch_size']
        
        while len(accepted_trajectories) < target_n_accept and n_attempts < max_attempts:
            # Sample batch
            batch_n = min(batch_size, max_attempts - n_attempts)
            # D1: Use rng to generate seed for GMM (stream-separated reproducibility)
            gmm_seed = int(self.rng.integers(0, 2**31))
            ic_batch, params_batch = self.gmm_sampler.sample(
                batch_n,
                seed=gmm_seed
            )
            
            for i in range(batch_n):
                n_attempts += 1
                
                ic = ic_batch[i]
                # P0-3 FIX: params_batch order is [m_cart, m_pole] per dataset meta.json
                # param_definition: index 0 = m_cart, index 1 = m_pole
                mc, mp = params_batch[i]  # m_cart, m_pole (CORRECT ORDER)
                
                # Full physics params
                physics_params = {
                    'm_cart': mc,
                    'm_pole': mp,
                    **self.fixed_physics
                }
                
                # Select control sequence
                if variant == "IC":
                    # Reuse training u (round-robin or random)
                    u_idx = self.rng.integers(0, self.N_train)
                    u_seq = self.train_u[u_idx, :, 0]  # (T,)
                else:
                    # Generate new random smooth u (ICU variant)
                    u_seq = self._generate_random_smooth_u()
                    u_idx = -1
                
                # Simulate trajectory
                traj, success = simulate_trajectory(
                    ic=ic,
                    u_sequence=u_seq,
                    params=physics_params,
                    dt=sim_config['dt'],
                    T=sim_config['T'],
                    method=sim_config['method'],
                    rtol=sim_config['rtol'],
                    atol=sim_config['atol'],
                )
                
                if not success:
                    rejection_counts['simulation_failed'] = rejection_counts.get('simulation_failed', 0) + 1
                    continue
                
                # Quality check
                passed, reason = check_trajectory_quality(traj, qc_config)
                
                if not passed:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue
                
                # Compute dx using savgol
                dx = compute_derivatives_savgol(
                    traj[np.newaxis, ...],  # (1, T, 4)
                    dt=sim_config['dt'],
                    theta_idx=2,
                    window=savgol_config['window'],
                    polyorder=savgol_config['polyorder'],
                )[0]  # (T, 4)
                
                # Accept
                accepted_trajectories.append(traj)
                accepted_dx.append(dx)
                accepted_params.append([mc, mp])
                accepted_ic.append(ic)
                accepted_u.append(u_seq.reshape(-1, 1))
                accepted_u_indices.append(u_idx)
                
                if len(accepted_trajectories) >= target_n_accept:
                    break
            
            # Progress
            if n_attempts % 500 == 0:
                print(f"    Attempts: {n_attempts}, Accepted: {len(accepted_trajectories)}")
        
        n_accepted = len(accepted_trajectories)
        acceptance_rate = n_accepted / n_attempts if n_attempts > 0 else 0
        
        print(f"  ✅ Pool complete: {n_accepted}/{n_attempts} accepted ({acceptance_rate:.1%})")
        print(f"  Rejections: {rejection_counts}")
        
        result = {
            'trajectories': np.array(accepted_trajectories),  # (N, T, 4)
            'dx': np.array(accepted_dx),                      # (N, T, 4)
            'params': np.array(accepted_params),              # (N, 2)
            'ic': np.array(accepted_ic),                      # (N, 4)
            'u': np.array(accepted_u),                        # (N, T, 1)
            'u_indices': np.array(accepted_u_indices),        # (N,)
            'stats': {
                'n_accepted': n_accepted,
                'n_attempts': n_attempts,
                'acceptance_rate': acceptance_rate,
                'rejection_counts': rejection_counts,
                'variant': variant,
            }
        }
        
        return result
    
    def _generate_random_smooth_u(self) -> np.ndarray:
        """Generate random smooth control sequence (for ICU variant)."""
        T = self.T
        dt = self.dt
        
        # Parameters for smooth random force
        amplitude = self.rng.uniform(1.0, 5.0)
        freq = self.rng.uniform(0.5, 2.0)
        phase = self.rng.uniform(0, 2 * np.pi)
        offset = self.rng.uniform(-1.0, 1.0)
        
        t = np.arange(T) * dt
        u = amplitude * np.sin(2 * np.pi * freq * t + phase) + offset
        
        # Add some noise
        u += self.rng.normal(0, 0.2, T)
        
        return u


# ============================================================
# Track A Selection (OOD Filter) - v2.1 FIX
# ============================================================

def track_a_selection(
    pool: Dict[str, Any],
    teacher_coefficients: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    dx_std: np.ndarray,
    norm_stats: Dict[str, Any],
    reject_ratio: float = 0.10,
    n_select: int = 200,
    dynamics_target_indices: List[int] = None,
    derivative_key: str = 'derivative_dx_savgol',
) -> Dict[str, Any]:
    """
    Track A: Filter out high-error candidates (OOD prevention).
    
    v1.3: 상위 10%만 제거 (reject_ratio=0.10)
    P0-1: dynamics targets only [1, 3]
    P0-3: Detailed relaxation recording
    v2.1 FIX: Use normalized inputs for alignment error calculation
    
    Args:
        pool: Pool generator output
        teacher_coefficients: Teacher model coefficients
        feature_names: Feature names
        target_names: Target names
        dx_std: Training dx std per target (raw)
        norm_stats: Normalization statistics from norm_stats.json
        reject_ratio: Fraction to reject (top error)
        n_select: Target number to select (for min_after_a calculation)
        dynamics_target_indices: [1, 3] for x_ddot, theta_ddot
        derivative_key: Key for derivative stats in norm_stats
        
    Returns:
        Dict with:
            'selected_indices': Indices of selected candidates
            'errors': Alignment errors for all candidates
            'threshold': Error threshold used
            'stats': Selection statistics with relaxation details
    """
    print(f"\n[Track A Selection] reject_ratio={reject_ratio}")
    
    trajectories = pool['trajectories']
    dx = pool['dx']
    u = pool['u']
    N = trajectories.shape[0]
    
    # SSOT: dynamics targets only (P0-1)
    if dynamics_target_indices is None:
        dynamics_target_indices = [1, 3]
    
    # v2.1 FIX: Compute alignment errors with normalized inputs
    errors = compute_teacher_alignment_error(
        trajectories=trajectories,
        dx_trajectories=dx,
        u_trajectories=u,
        teacher_coefficients=teacher_coefficients,
        feature_names=feature_names,
        target_names=target_names,
        dx_std=dx_std,
        norm_stats=norm_stats,
        dynamics_target_indices=dynamics_target_indices,
        derivative_key=derivative_key,
    )
    
    # Design rule: min_after_a = max(5*k, 800) where k = n_select
    min_after_a = max(5 * n_select, 800)
    
    # Determine threshold (reject top reject_ratio%)
    n_reject_target = max(1, int(N * reject_ratio))
    error_threshold_strict = np.percentile(errors, (1 - reject_ratio) * 100)
    
    # Pre-relaxation pass rate
    pre_relax_mask = errors <= error_threshold_strict
    pre_relax_n_pass = pre_relax_mask.sum()
    pre_relax_pass_rate = pre_relax_n_pass / N if N > 0 else 0
    
    # Check if relaxation needed
    relaxed = False
    effective_reject_ratio = reject_ratio
    
    if pre_relax_n_pass < min_after_a and N < min_after_a:
        # Pool too small for min_after_a requirement → relax to accept all
        relaxed = True
        selected_indices = np.arange(N)
        error_threshold = errors.max() + 1e-6  # Accept all
        effective_reject_ratio = 0.0
        print(f"  ⚠️ RELAXED: pool({N}) < min_after_a({min_after_a})")
        print(f"     Pre-relax pass rate: {pre_relax_pass_rate:.1%} ({pre_relax_n_pass}/{N})")
    else:
        # Normal selection
        selected_mask = errors <= error_threshold_strict
        selected_indices = np.where(selected_mask)[0]
        error_threshold = error_threshold_strict
        effective_reject_ratio = 1.0 - (len(selected_indices) / N) if N > 0 else 0
    
    n_selected = len(selected_indices)
    
    print(f"  Dynamics targets used: {dynamics_target_indices} ({[target_names[i] for i in dynamics_target_indices]})")
    print(f"  Error stats: mean={errors.mean():.4f}, std={errors.std():.4f}")
    print(f"  Error threshold (strict): {error_threshold_strict:.4f}")
    print(f"  Selected: {n_selected}/{N} ({n_selected/N:.1%})")
    if relaxed:
        print(f"  ⚠️ relaxed_flag=True, effective_reject_ratio={effective_reject_ratio:.1%}")
    
    return {
        'selected_indices': selected_indices,
        'errors': errors,
        'threshold': error_threshold,
        'threshold_strict': error_threshold_strict,
        'stats': {
            'n_total': N,
            'n_selected': n_selected,
            'n_rejected': N - n_selected,
            'error_mean': float(errors.mean()),
            'error_std': float(errors.std()),
            'error_threshold': float(error_threshold),
            'error_threshold_strict': float(error_threshold_strict),
            # P0-3: Detailed relaxation recording
            'relaxed_flag': relaxed,
            'pre_relax_n_pass': int(pre_relax_n_pass),
            'pre_relax_pass_rate': float(pre_relax_pass_rate),
            'effective_reject_ratio': float(effective_reject_ratio),
            'min_after_a_required': min_after_a,
            'dynamics_target_indices': dynamics_target_indices,
        }
    }


# ============================================================
# Final Selection (Top-k from Track A passed)
# ============================================================

def final_selection(
    pool: Dict[str, Any],
    track_a_result: Dict[str, Any],
    n_select: int = 200,
    seed: int = 42,
    selection_mode: str = 'random',  # 'random', 'track_a_filtered_random', 'track_b'
    rng: Optional[np.random.Generator] = None,  # D1: External RNG injection
) -> Dict[str, Any]:
    """
    Final selection from pool candidates.
    
    D1 Rebaseline SSOT:
    - 'random': Select from ENTIRE pool (no Track A filtering)
    - 'track_a_filtered_random': Select from Track A passed candidates (filter + random)
    - 'track_b': Track B scoring (D2)
    
    P0-2 CRITICAL: 'track_a_filtered_random' is NOT top-k by error!
    It's: reject top reject_ratio error → random from remaining.
    
    Args:
        pool: Pool generator output
        track_a_result: Track A selection result (used only for track_a_filtered_random)
        n_select: Number to select
        seed: Random seed (used if rng is None)
        selection_mode: Selection mode
        rng: External RNG generator (D1: for stream separation)
        
    Returns:
        Dict with selected trajectories and metadata
    """
    print(f"\n[Final Selection] n_select={n_select}, mode={selection_mode}")
    
    # D1: Use external RNG if provided
    if rng is None:
        rng = np.random.default_rng(seed)
    
    # Get pool size
    n_pool = len(pool['trajectories'])
    
    # Determine candidate pool based on selection mode
    if selection_mode == 'random':
        # D1 SSOT: Select from ENTIRE pool (baseline, no filtering)
        candidate_indices = np.arange(n_pool)
        all_errors = track_a_result.get('errors', np.zeros(n_pool))  # May not have errors
        print(f"  Random from entire pool: {n_pool} candidates")
        
    elif selection_mode == 'track_a_filtered_random':
        # D1 SSOT: Select from Track A passed candidates
        # Track A = reject top 10% error, then random from remaining
        candidate_indices = track_a_result['selected_indices']
        all_errors = track_a_result['errors']
        print(f"  Track A filtered random: {len(candidate_indices)} candidates (from {n_pool} pool)")
        
    elif selection_mode == 'track_b':
        # Track B: Delegate to track_b_selection function
        # This branch should not be reached directly - use track_b_selection() instead
        raise ValueError(
            "selection_mode='track_b' should use track_b_selection() function directly, "
            "not final_selection(). This ensures proper artifact saving."
        )
        
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    
    n_available = len(candidate_indices)
    
    # Select from candidates
    if n_available < n_select:
        print(f"  ⚠️ Only {n_available} available, selecting all")
        final_indices = candidate_indices.copy()
    else:
        # Random selection using provided RNG
        chosen = rng.choice(n_available, size=n_select, replace=False)
        final_indices = candidate_indices[chosen]
    
    # Extract selected data
    result = {
        'trajectories': pool['trajectories'][final_indices],  # (n_select, T, 4)
        'dx': pool['dx'][final_indices],                      # (n_select, T, 4)
        'params': pool['params'][final_indices],              # (n_select, 2)
        'ic': pool['ic'][final_indices],                      # (n_select, 4)
        'u': pool['u'][final_indices],                        # (n_select, T, 1)
        'u_indices': pool['u_indices'][final_indices],        # (n_select,)
        'errors': all_errors[final_indices] if len(all_errors) > 0 else np.zeros(len(final_indices)),
        'original_indices': final_indices.copy(),             # (n_select,)
        'stats': {
            'n_pool': n_pool,
            'n_available': n_available,
            'n_selected': len(final_indices),
            'selection_mode': selection_mode,
        }
    }
    
    # Add error stats if available
    if len(all_errors) > 0 and len(final_indices) > 0:
        selected_errors = all_errors[final_indices]
        result['stats'].update({
            'error_mean_selected': float(selected_errors.mean()),
            'error_std_selected': float(selected_errors.std()),
            'error_min_selected': float(selected_errors.min()),
            'error_max_selected': float(selected_errors.max()),
        })
        print(f"  Selected {len(final_indices)} trajectories")
        print(f"  Error range: [{selected_errors.min():.4f}, {selected_errors.max():.4f}]")
    else:
        print(f"  Selected {len(final_indices)} trajectories")
    
    return result


# ============================================================
# Track B Selection (Information-Seeking + Diversity)
# ============================================================
# GPT P0 반영: v0.1
# - P0-1: Diversity-aware (greedy farthest-point)
# - P0-2: θ 주기 변수 처리 (sin/cos)
# - P0-3: Median/IQR 기반 정규화 + eps
# - P0-5: 아티팩트 봉인
# - P0-6: Canonicalize (global_idx 정렬)
# ============================================================

TRACK_B_FEATURE_SPEC = {
    'version': 'v0.1',
    'features': [
        'std_sin_theta',      # std(sin(theta)) over trajectory
        'std_cos_theta',      # std(cos(theta)) over trajectory
        'std_theta_dot',      # std(theta_dot) over trajectory
        'std_x',              # std(x) over trajectory
        'std_x_dot',          # std(x_dot) over trajectory
    ],
    'score_formula': 'excitation - alpha * x_penalty',
    'excitation_def': '(norm_std_sin_theta + norm_std_cos_theta + norm_std_theta_dot) / 3',
    'x_penalty_def': '(norm_std_x + norm_std_x_dot) / 2',
    'normalization': 'median_iqr',
    'eps': 1e-12,
    'default_alpha': 0.3,
    'diversity_metric': 'euclidean',
    'diversity_space': 'feature_vector',  # 5D feature space
    'tie_break': 'global_idx_asc',
}


def compute_track_b_features(
    trajectories: np.ndarray,
    feature_spec: Dict[str, Any] = None,
) -> np.ndarray:
    """
    Compute Track B features for each trajectory.
    
    GPT P0-2: θ는 주기 변수 → std(sin(θ)), std(cos(θ)) 사용
    
    Args:
        trajectories: (N, T, 4) state trajectories
                     indices: [x, x_dot, theta, theta_dot]
        feature_spec: Feature specification dict
        
    Returns:
        features: (N, 5) feature matrix
                 [std_sin_theta, std_cos_theta, std_theta_dot, std_x, std_x_dot]
    """
    if feature_spec is None:
        feature_spec = TRACK_B_FEATURE_SPEC
    
    N, T, D = trajectories.shape
    
    # Extract state components
    x = trajectories[:, :, 0]           # (N, T)
    x_dot = trajectories[:, :, 1]       # (N, T)
    theta = trajectories[:, :, 2]       # (N, T)
    theta_dot = trajectories[:, :, 3]   # (N, T)
    
    # Compute features (GPT P0-2: use sin/cos for theta)
    std_sin_theta = np.std(np.sin(theta), axis=1)   # (N,)
    std_cos_theta = np.std(np.cos(theta), axis=1)   # (N,)
    std_theta_dot = np.std(theta_dot, axis=1)       # (N,)
    std_x = np.std(x, axis=1)                       # (N,)
    std_x_dot = np.std(x_dot, axis=1)               # (N,)
    
    # Stack features: (N, 5)
    features = np.column_stack([
        std_sin_theta,
        std_cos_theta,
        std_theta_dot,
        std_x,
        std_x_dot,
    ])
    
    return features


def compute_track_b_normalization_stats(
    features: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Compute normalization statistics using median/IQR (robust scaling).
    
    GPT P0-3: median/IQR 기반 정규화
    
    Args:
        features: (N, D) feature matrix
        eps: Small constant to prevent division by zero
        
    Returns:
        Dict with 'median', 'iqr', 'eps'
    """
    median = np.median(features, axis=0)  # (D,)
    q75 = np.percentile(features, 75, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    iqr = q75 - q25  # (D,)
    
    # Ensure IQR is not zero
    iqr = np.maximum(iqr, eps)
    
    return {
        'median': median,
        'iqr': iqr,
        'q25': q25,
        'q75': q75,
        'eps': eps,
    }


def normalize_track_b_features(
    features: np.ndarray,
    norm_stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Normalize features using median/IQR scaling.
    
    Args:
        features: (N, D) feature matrix
        norm_stats: Dict with 'median', 'iqr'
        
    Returns:
        normalized: (N, D) normalized features
    """
    median = norm_stats['median']
    iqr = norm_stats['iqr']
    
    normalized = (features - median) / iqr
    return normalized


def compute_track_b_scores(
    features_normalized: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Compute Track B scores from normalized features.
    
    Score = excitation - alpha * x_penalty
    
    excitation = (std_sin_theta + std_cos_theta + std_theta_dot) / 3
    x_penalty = (std_x + std_x_dot) / 2
    
    Args:
        features_normalized: (N, 5) normalized features
                            [std_sin_theta, std_cos_theta, std_theta_dot, std_x, std_x_dot]
        alpha: Penalty weight for x-related features
        
    Returns:
        scores: (N,) Track B scores (higher is better)
    """
    # Feature indices
    # 0: std_sin_theta, 1: std_cos_theta, 2: std_theta_dot
    # 3: std_x, 4: std_x_dot
    
    excitation = (
        features_normalized[:, 0] +  # std_sin_theta
        features_normalized[:, 1] +  # std_cos_theta
        features_normalized[:, 2]    # std_theta_dot
    ) / 3.0
    
    x_penalty = (
        features_normalized[:, 3] +  # std_x
        features_normalized[:, 4]    # std_x_dot
    ) / 2.0
    
    scores = excitation - alpha * x_penalty
    
    return scores


def top_m_diversity_selection(
    ic_params: np.ndarray,
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    n_select: int,
    top_m_ratio: float = 5.0,
    score_floor: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Top-M gate + Diversity selection in IC/params space.
    
    GPT P0-2 FIX: Hard threshold → Top-M gate로 "저품질 강제 수집" 방지
    GPT P0-3 FIX: feature_vector → IC/params 공간에서 diversity
    
    Algorithm:
    1. (Optional) Score floor: score < score_floor인 후보 제외
    2. Top-M gate: score 상위 M개만 diversity 후보로 (M = top_m_ratio × n_select)
    3. Greedy farthest-point: IC/params 공간에서 diversity selection
    4. Tie-break: global_idx ascending
    
    Args:
        ic_params: (N_pool, 6) IC + params for diversity
                   [x0, x_dot0, theta0, theta_dot0, m_cart, m_pole]
        scores: (N_pool,) scores for all pool candidates
        candidate_indices: (N_cand,) indices of Track A passed candidates
        n_select: Number to select
        top_m_ratio: M = top_m_ratio × n_select (default 5.0)
        score_floor: Minimum score threshold (None = no floor)
        rng: Random generator (not used, deterministic)
        
    Returns:
        selected_indices: (n_select,) selected pool indices (NOT canonicalized)
        selection_trace: List of dicts with selection details
    """
    from scipy.spatial.distance import cdist
    
    N_cand = len(candidate_indices)
    
    # Get IC/params and scores for candidates only
    cand_ic_params = ic_params[candidate_indices]  # (N_cand, 6)
    cand_scores = scores[candidate_indices]        # (N_cand,)
    
    # Stage 0: Score floor filtering (optional)
    if score_floor is not None:
        floor_mask = cand_scores >= score_floor
        n_above_floor = floor_mask.sum()
        if n_above_floor < n_select:
            print(f"  ⚠️ Score floor: only {n_above_floor} above {score_floor}, using all")
            floor_mask = np.ones(N_cand, dtype=bool)  # Fallback: use all
    else:
        floor_mask = np.ones(N_cand, dtype=bool)
    
    # Apply floor mask
    floor_indices = np.where(floor_mask)[0]  # Local indices within candidates
    filtered_scores = cand_scores[floor_indices]
    filtered_ic_params = cand_ic_params[floor_indices]
    N_filtered = len(floor_indices)
    
    # Stage 1: Top-M gate
    M = int(top_m_ratio * n_select)
    M = min(M, N_filtered)  # Can't exceed available
    
    if N_filtered <= n_select:
        # Not enough candidates, return all filtered
        selected_local = floor_indices
        trace = [{
            'note': 'all_filtered_selected',
            'n_filtered': N_filtered,
            'score_floor': score_floor,
        }]
        selected_pool_indices = candidate_indices[selected_local]
        return selected_pool_indices, trace
    
    # Get top-M by score
    sorted_by_score = np.argsort(-filtered_scores)  # Descending
    top_m_local = sorted_by_score[:M]  # Local indices within filtered
    top_m_in_cand = floor_indices[top_m_local]  # Local indices within candidates
    
    top_m_scores = cand_scores[top_m_in_cand]
    top_m_ic_params = cand_ic_params[top_m_in_cand]
    
    print(f"  Top-M gate: M={M} from {N_filtered} filtered (ratio={top_m_ratio})")
    print(f"  Top-M score range: [{top_m_scores.min():.4f}, {top_m_scores.max():.4f}]")
    
    # Stage 2: Greedy farthest-point in IC/params space
    selection_trace = []
    selected_local = []  # Local indices within top_m_in_cand
    remaining_mask = np.ones(M, dtype=bool)
    
    # Normalize IC/params for distance computation (avoid scale issues)
    ic_params_mean = top_m_ic_params.mean(axis=0)
    ic_params_std = top_m_ic_params.std(axis=0)
    ic_params_std = np.maximum(ic_params_std, 1e-12)  # Avoid division by zero
    top_m_ic_params_norm = (top_m_ic_params - ic_params_mean) / ic_params_std
    
    # First selection: highest score among top-M, tie-break by global_idx
    first_local = 0  # Already sorted by score
    # Check for ties at top score
    top_score = top_m_scores[first_local]
    tie_mask = np.abs(top_m_scores - top_score) < 1e-12
    if tie_mask.sum() > 1:
        tie_indices = np.where(tie_mask)[0]
        tie_global = candidate_indices[top_m_in_cand[tie_indices]]
        first_local = tie_indices[np.argmin(tie_global)]
    
    selected_local.append(first_local)
    remaining_mask[first_local] = False
    selection_trace.append({
        'step': 0,
        'local_idx': int(top_m_in_cand[first_local]),
        'global_idx': int(candidate_indices[top_m_in_cand[first_local]]),
        'score': float(top_m_scores[first_local]),
        'min_dist_to_selected': None,
        'reason': 'highest_score_first',
    })
    
    # Greedy selection: maximize min distance to selected set
    for step in range(1, n_select):
        if not remaining_mask.any():
            break
        
        remaining_indices = np.where(remaining_mask)[0]
        
        # Compute min distance to selected set (in normalized IC/params space)
        selected_ic = top_m_ic_params_norm[selected_local]  # (n_selected, 6)
        remaining_ic = top_m_ic_params_norm[remaining_indices]  # (n_remaining, 6)
        
        dist_to_selected = cdist(remaining_ic, selected_ic, metric='euclidean')
        min_dists = dist_to_selected.min(axis=1)  # (n_remaining,)
        
        # Select candidate with maximum min-distance (farthest point)
        best_remaining_idx = np.argmax(min_dists)
        best_local_idx = remaining_indices[best_remaining_idx]
        best_min_dist = min_dists[best_remaining_idx]
        
        # Tie-break: if multiple have same min_dist, pick highest score
        tie_mask_dist = np.abs(min_dists - best_min_dist) < 1e-12
        if tie_mask_dist.sum() > 1:
            tie_remaining = remaining_indices[tie_mask_dist]
            tie_scores = top_m_scores[tie_remaining]
            best_score_idx = np.argmax(tie_scores)
            best_local_idx = tie_remaining[best_score_idx]
            
            # If still tied, use global_idx
            best_score = tie_scores[best_score_idx]
            score_tie_mask = np.abs(tie_scores - best_score) < 1e-12
            if score_tie_mask.sum() > 1:
                score_tie_remaining = tie_remaining[score_tie_mask]
                tie_global = candidate_indices[top_m_in_cand[score_tie_remaining]]
                best_local_idx = score_tie_remaining[np.argmin(tie_global)]
        
        selected_local.append(best_local_idx)
        remaining_mask[best_local_idx] = False
        
        selection_trace.append({
            'step': step,
            'local_idx': int(top_m_in_cand[best_local_idx]),
            'global_idx': int(candidate_indices[top_m_in_cand[best_local_idx]]),
            'score': float(top_m_scores[best_local_idx]),
            'min_dist_to_selected': float(min_dists[remaining_indices == best_local_idx][0]),
            'reason': 'max_min_distance',
        })
    
    # Convert to pool indices
    selected_pool_indices = candidate_indices[top_m_in_cand[np.array(selected_local)]]
    
    # Log final score range
    final_scores = top_m_scores[selected_local]
    print(f"  Selected score: mean={final_scores.mean():.4f}, range=[{final_scores.min():.4f}, {final_scores.max():.4f}]")
    
    return selected_pool_indices, selection_trace


# Legacy function for backward compatibility
def greedy_farthest_point_selection(
    features: np.ndarray,
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    n_select: int,
    min_distance_ratio: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Greedy farthest-point selection with score-weighted tie-breaking.
    
    GPT P0-1: Diversity-aware selection
    GPT P0-6: Tie-break by global_idx
    
    Algorithm:
    1. Sort candidates by score (descending)
    2. Select first candidate (highest score)
    3. For remaining selections:
       - Compute min distance to already selected
       - Among candidates with sufficient distance, pick highest score
       - Tie-break: global_idx ascending
    
    Args:
        features: (N_pool, D) feature matrix for diversity computation
        scores: (N_pool,) scores for all pool candidates
        candidate_indices: (N_cand,) indices of Track A passed candidates
        n_select: Number to select
        min_distance_ratio: Minimum distance as ratio of max pairwise distance
        rng: Random generator (not used, deterministic)
        
    Returns:
        selected_indices: (n_select,) selected pool indices (NOT canonicalized)
        selection_trace: List of dicts with selection details
    """
    from scipy.spatial.distance import cdist
    
    N_cand = len(candidate_indices)
    
    if N_cand <= n_select:
        # Not enough candidates, return all
        return candidate_indices.copy(), [{'note': 'all_candidates_selected', 'n_cand': N_cand}]
    
    # Get features and scores for candidates only
    cand_features = features[candidate_indices]  # (N_cand, D)
    cand_scores = scores[candidate_indices]      # (N_cand,)
    
    # Compute pairwise distances for diversity
    pairwise_dist = cdist(cand_features, cand_features, metric='euclidean')
    max_dist = pairwise_dist.max()
    min_dist_threshold = min_distance_ratio * max_dist
    
    # Selection trace
    selection_trace = []
    selected_local = []  # Local indices within candidates
    remaining_mask = np.ones(N_cand, dtype=bool)
    
    # First selection: highest score, tie-break by global_idx
    sorted_by_score = np.argsort(-cand_scores)  # Descending score
    first_idx = sorted_by_score[0]
    # Check for ties at top score
    top_score = cand_scores[first_idx]
    tie_mask = np.abs(cand_scores - top_score) < 1e-12
    if tie_mask.sum() > 1:
        # Tie-break by global_idx
        tie_indices = np.where(tie_mask)[0]
        tie_global = candidate_indices[tie_indices]
        first_idx = tie_indices[np.argmin(tie_global)]
    
    selected_local.append(first_idx)
    remaining_mask[first_idx] = False
    selection_trace.append({
        'step': 0,
        'local_idx': int(first_idx),
        'global_idx': int(candidate_indices[first_idx]),
        'score': float(cand_scores[first_idx]),
        'min_dist_to_selected': None,
        'reason': 'highest_score_first',
    })
    
    # Greedy selection
    for step in range(1, n_select):
        if not remaining_mask.any():
            break
        
        remaining_indices = np.where(remaining_mask)[0]
        
        # Compute min distance to selected set for each remaining candidate
        selected_features = cand_features[selected_local]  # (n_selected, D)
        remaining_features = cand_features[remaining_indices]  # (n_remaining, D)
        
        dist_to_selected = cdist(remaining_features, selected_features, metric='euclidean')
        min_dists = dist_to_selected.min(axis=1)  # (n_remaining,)
        
        # Filter: candidates with sufficient distance
        sufficient_dist_mask = min_dists >= min_dist_threshold
        
        if sufficient_dist_mask.any():
            # Among sufficient distance, pick highest score
            filtered_local = remaining_indices[sufficient_dist_mask]
            filtered_scores = cand_scores[filtered_local]
            best_filtered = np.argmax(filtered_scores)
            best_local_idx = filtered_local[best_filtered]
            
            # Tie-break check
            best_score = filtered_scores[best_filtered]
            tie_mask_filtered = np.abs(filtered_scores - best_score) < 1e-12
            if tie_mask_filtered.sum() > 1:
                tie_local = filtered_local[tie_mask_filtered]
                tie_global = candidate_indices[tie_local]
                best_local_idx = tie_local[np.argmin(tie_global)]
            
            reason = 'diversity_score'
            min_dist_val = min_dists[np.where(remaining_indices == best_local_idx)[0][0]]
        else:
            # No candidate with sufficient distance, just pick highest score
            remaining_scores = cand_scores[remaining_indices]
            best_remaining = np.argmax(remaining_scores)
            best_local_idx = remaining_indices[best_remaining]
            
            # Tie-break check
            best_score = remaining_scores[best_remaining]
            tie_mask_rem = np.abs(remaining_scores - best_score) < 1e-12
            if tie_mask_rem.sum() > 1:
                tie_local = remaining_indices[tie_mask_rem]
                tie_global = candidate_indices[tie_local]
                best_local_idx = tie_local[np.argmin(tie_global)]
            
            reason = 'score_only_no_diverse'
            min_dist_val = min_dists[np.where(remaining_indices == best_local_idx)[0][0]]
        
        selected_local.append(best_local_idx)
        remaining_mask[best_local_idx] = False
        selection_trace.append({
            'step': step,
            'local_idx': int(best_local_idx),
            'global_idx': int(candidate_indices[best_local_idx]),
            'score': float(cand_scores[best_local_idx]),
            'min_dist_to_selected': float(min_dist_val),
            'reason': reason,
        })
    
    # Convert to pool indices
    selected_pool_indices = candidate_indices[np.array(selected_local)]
    
    return selected_pool_indices, selection_trace


def track_b_selection(
    pool: Dict[str, Any],
    track_a_result: Dict[str, Any],
    n_select: int = 200,
    alpha: float = 0.3,
    diversity_mode: str = 'top_m_diversity',
    top_m_ratio: float = 5.0,
    score_floor: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Track B Selection v0.2: Information-seeking + Diversity.
    
    GPT P0 피드백 반영 (v0.2):
    - P0-2 FIX: Top-M gate로 "저품질 강제 수집" 방지
    - P0-3 FIX: IC/params 공간에서 diversity (feature_vector 대신)
    - Score floor 옵션 추가
    - diversity_mode로 ablation 지원 (score_only vs top_m_diversity)
    
    Args:
        pool: Pool generator output
        track_a_result: Track A selection result
        n_select: Number to select
        alpha: x_penalty weight (default 0.3)
        diversity_mode: 'score_only' or 'top_m_diversity'
        top_m_ratio: M = top_m_ratio × n_select (default 5.0)
        score_floor: Minimum score threshold (None = no floor)
        rng: Random generator (for reproducibility)
        results_dir: Directory to save artifacts
        
    Returns:
        Dict with selected trajectories, artifacts, and metadata
    """
    print(f"\n[Track B Selection v0.2] n_select={n_select}, alpha={alpha}")
    print(f"  diversity_mode={diversity_mode}, top_m_ratio={top_m_ratio}, score_floor={score_floor}")
    
    trajectories = pool['trajectories']
    N_pool = len(trajectories)
    
    # 1. Compute score features for ALL pool candidates
    features = compute_track_b_features(trajectories)  # (N_pool, 5)
    print(f"  Score features computed: shape={features.shape}")
    
    # 2. Build IC/params array for diversity (GPT P0-3: IC/params space)
    # ic: (N_pool, 4) = [x0, x_dot0, theta0, theta_dot0]
    # params: (N_pool, 2) = [m_cart, m_pole]
    ic = pool['ic']  # (N_pool, 4)
    params = pool['params']  # (N_pool, 2)
    ic_params = np.hstack([ic, params])  # (N_pool, 6)
    print(f"  IC/params for diversity: shape={ic_params.shape}")
    
    # 3. Get Track A passed candidates
    track_a_indices = track_a_result['selected_indices']
    track_a_stats = track_a_result['stats']
    relaxed_flag = track_a_stats.get('relaxed_flag', False)
    
    print(f"  Track A passed: {len(track_a_indices)}/{N_pool}")
    if relaxed_flag:
        print(f"  ⚠️ Track A was RELAXED (all candidates passed)")
    
    # 4. Compute normalization stats from Track A passed
    track_a_features = features[track_a_indices]
    norm_stats = compute_track_b_normalization_stats(track_a_features)
    print(f"  Norm stats computed (median/IQR from Track A passed)")
    
    # 5. Normalize ALL features using Track A stats
    features_normalized = normalize_track_b_features(features, norm_stats)
    
    # 6. Compute scores for ALL pool candidates
    scores = compute_track_b_scores(features_normalized, alpha=alpha)
    print(f"  Scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"  Scores (Track A passed): mean={scores[track_a_indices].mean():.4f}")
    
    # 7. Selection based on diversity_mode
    if diversity_mode == 'score_only':
        # GPT 권장 ablation: score만으로 top-k 선택 (diversity 없음)
        selected_indices, selection_trace = _score_only_selection(
            scores=scores,
            candidate_indices=track_a_indices,
            n_select=n_select,
            score_floor=score_floor,
        )
    elif diversity_mode == 'top_m_diversity':
        # GPT P0-2 FIX: Top-M gate + IC/params diversity
        selected_indices, selection_trace = top_m_diversity_selection(
            ic_params=ic_params,
            scores=scores,
            candidate_indices=track_a_indices,
            n_select=n_select,
            top_m_ratio=top_m_ratio,
            score_floor=score_floor,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown diversity_mode: {diversity_mode}")
    
    n_selected = len(selected_indices)
    print(f"  Selected: {n_selected} trajectories")
    
    # 8. Canonicalize: sort by global_idx (GPT P0-6)
    selected_indices_canonical = np.sort(selected_indices)
    
    # 9. Build feature spec with hash (v0.2 updated)
    feature_spec = {
        'version': 'v0.2',
        'score_features': TRACK_B_FEATURE_SPEC['features'],
        'score_formula': TRACK_B_FEATURE_SPEC['score_formula'],
        'normalization': TRACK_B_FEATURE_SPEC['normalization'],
        'eps': TRACK_B_FEATURE_SPEC['eps'],
        'alpha': alpha,
        'diversity_mode': diversity_mode,
        'diversity_space': 'ic_params',  # GPT P0-3 FIX
        'diversity_features': ['x0', 'x_dot0', 'theta0', 'theta_dot0', 'm_cart', 'm_pole'],
        'top_m_ratio': top_m_ratio,
        'score_floor': score_floor,
        'tie_break': 'global_idx_asc',
    }
    spec_str = json.dumps(feature_spec, sort_keys=True)
    feature_spec['spec_hash'] = hashlib.sha256(spec_str.encode()).hexdigest()[:16]
    
    # 10. Save artifacts if results_dir provided
    artifacts_saved = {}
    if results_dir is not None:
        results_dir = Path(results_dir)
        
        # track_b_score_vector.npy
        score_path = results_dir / 'track_b_score_vector.npy'
        np.save(score_path, scores)
        artifacts_saved['score_vector'] = str(score_path)
        
        # track_b_candidate_features.npy (score features)
        features_path = results_dir / 'track_b_candidate_features.npy'
        np.save(features_path, features)
        artifacts_saved['candidate_features'] = str(features_path)
        
        # track_b_ic_params.npy (diversity features) - v0.2 NEW
        ic_params_path = results_dir / 'track_b_ic_params.npy'
        np.save(ic_params_path, ic_params)
        artifacts_saved['ic_params'] = str(ic_params_path)
        
        # track_b_feature_spec.json
        spec_path = results_dir / 'track_b_feature_spec.json'
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(feature_spec, f, indent=2)
        artifacts_saved['feature_spec'] = str(spec_path)
        
        # track_b_selection_trace.json
        trace_data = {
            'version': 'v0.2',
            'n_pool': N_pool,
            'n_track_a_passed': len(track_a_indices),
            'track_a_relaxed': relaxed_flag,
            'n_selected': n_selected,
            'alpha': alpha,
            'diversity_mode': diversity_mode,
            'top_m_ratio': top_m_ratio,
            'score_floor': score_floor,
            'selection_trace': selection_trace,
            'norm_stats': {
                'median': norm_stats['median'].tolist(),
                'iqr': norm_stats['iqr'].tolist(),
                'eps': norm_stats['eps'],
            },
        }
        trace_path = results_dir / 'track_b_selection_trace.json'
        with open(trace_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2)
        artifacts_saved['selection_trace'] = str(trace_path)
        
        # track_b_stage1_pass_mask.npy
        pass_mask = np.zeros(N_pool, dtype=bool)
        pass_mask[track_a_indices] = True
        mask_path = results_dir / 'track_b_stage1_pass_mask.npy'
        np.save(mask_path, pass_mask)
        artifacts_saved['stage1_pass_mask'] = str(mask_path)
        
        # selected_pool_indices.npy (canonical order)
        indices_path = results_dir / 'selected_pool_indices.npy'
        np.save(indices_path, selected_indices_canonical)
        artifacts_saved['selected_indices'] = str(indices_path)
        
        print(f"  ✅ Artifacts saved: {len(artifacts_saved)} files")
    
    # 11. Extract selected data
    all_errors = track_a_result.get('errors', np.zeros(N_pool))
    
    result = {
        'trajectories': pool['trajectories'][selected_indices_canonical],
        'dx': pool['dx'][selected_indices_canonical],
        'params': pool['params'][selected_indices_canonical],
        'ic': pool['ic'][selected_indices_canonical],
        'u': pool['u'][selected_indices_canonical],
        'u_indices': pool['u_indices'][selected_indices_canonical],
        'errors': all_errors[selected_indices_canonical],
        'original_indices': selected_indices_canonical.copy(),
        'scores': scores[selected_indices_canonical],
        'stats': {
            'n_pool': N_pool,
            'n_track_a_passed': len(track_a_indices),
            'track_a_relaxed': relaxed_flag,
            'n_selected': n_selected,
            'selection_mode': 'track_b',
            'diversity_mode': diversity_mode,
            'alpha': alpha,
            'top_m_ratio': top_m_ratio,
            'score_floor': score_floor,
            'feature_spec_version': feature_spec['version'],
            'feature_spec_hash': feature_spec['spec_hash'],
            'score_mean_selected': float(scores[selected_indices_canonical].mean()),
            'score_std_selected': float(scores[selected_indices_canonical].std()),
            'score_min_selected': float(scores[selected_indices_canonical].min()),
            'score_max_selected': float(scores[selected_indices_canonical].max()),
            'canonicalized': True,
        },
        'artifacts_saved': artifacts_saved,
        'feature_spec': feature_spec,
        'norm_stats': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in norm_stats.items()},
    }
    
    return result


def _score_only_selection(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    n_select: int,
    score_floor: Optional[float] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Score-only selection (no diversity) - for ablation.
    
    Args:
        scores: (N_pool,) scores for all pool candidates
        candidate_indices: (N_cand,) indices of Track A passed candidates
        n_select: Number to select
        score_floor: Minimum score threshold
        
    Returns:
        selected_indices: (n_select,) selected pool indices
        selection_trace: List with selection summary
    """
    cand_scores = scores[candidate_indices]
    N_cand = len(candidate_indices)
    
    # Apply score floor
    if score_floor is not None:
        floor_mask = cand_scores >= score_floor
        n_above_floor = floor_mask.sum()
        if n_above_floor < n_select:
            print(f"  ⚠️ Score floor: only {n_above_floor} above {score_floor}, using all")
            floor_mask = np.ones(N_cand, dtype=bool)
    else:
        floor_mask = np.ones(N_cand, dtype=bool)
    
    floor_indices = np.where(floor_mask)[0]
    filtered_scores = cand_scores[floor_indices]
    
    # Select top-k by score
    k = min(n_select, len(floor_indices))
    sorted_idx = np.argsort(-filtered_scores)[:k]
    selected_local = floor_indices[sorted_idx]
    selected_pool_indices = candidate_indices[selected_local]
    
    # Trace
    trace = [{
        'method': 'score_only',
        'n_candidates': N_cand,
        'n_after_floor': len(floor_indices),
        'n_selected': len(selected_pool_indices),
        'score_floor': score_floor,
        'score_range_selected': [float(filtered_scores[sorted_idx].min()), 
                                  float(filtered_scores[sorted_idx].max())],
    }]
    
    print(f"  Score-only selection: top {k} from {len(floor_indices)} candidates")
    print(f"  Selected score range: [{filtered_scores[sorted_idx].min():.4f}, {filtered_scores[sorted_idx].max():.4f}]")
    
    return selected_pool_indices, trace


# ============================================================
# Track B v0.3: D-optimal Selection on Θ_F
# ============================================================
# GPT P0 반영:
# - P0-1: Θ 계산 시 정규화 입력 사용 (norm_stats)
# - P0-2: target별 F_t 사용 (dynamics targets)
# - P0-3: F_t = fragile_t ∩ teacher_active_t (spurious 억제)
# ============================================================

def compute_fragile_feature_sets(
    fragile_pairs: List[Tuple[int, int]],
    teacher_support: np.ndarray,
    dynamics_target_indices: List[int] = [1, 3],
    use_teacher_intersection: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Compute fragile feature sets per target (F_t).
    
    GPT P0-2: target별 F_t
    GPT P0-3: F_t = fragile_t ∩ teacher_active_t
    
    Args:
        fragile_pairs: List of (feature_idx, target_idx) tuples
        teacher_support: (n_features, n_targets) boolean mask
        dynamics_target_indices: [1, 3] for x_ddot, theta_ddot
        use_teacher_intersection: If True, F_t = fragile ∩ teacher_active
        
    Returns:
        F_by_target: {target_idx: np.array of feature indices}
    """
    F_by_target = {}
    
    for t in dynamics_target_indices:
        # fragile features for this target
        fragile_t = set([f for (f, target) in fragile_pairs if target == t])
        
        if use_teacher_intersection:
            # teacher active features for this target
            teacher_active_t = set(np.where(teacher_support[:, t])[0])
            # intersection
            F_t = fragile_t & teacher_active_t
        else:
            F_t = fragile_t
        
        F_by_target[t] = np.array(sorted(F_t), dtype=int)
    
    return F_by_target


def compute_library_features_normalized(
    trajectories: np.ndarray,
    u: np.ndarray,
    norm_stats: Dict[str, Any],
    library: 'SINDyLibrary',
) -> np.ndarray:
    """
    Compute SINDy library features with normalized inputs.
    
    GPT P0-1: Teacher error와 동일한 정규화 적용
    
    Args:
        trajectories: (N, T, 4) raw state trajectories
        u: (N, T, 1) control inputs
        norm_stats: Normalization statistics from norm_stats.json
        library: SINDy library instance
        
    Returns:
        Theta: (N, T, n_features) library features
    """
    N, T, D = trajectories.shape
    
    # Get normalization stats (same as Gate1/Teacher)
    state_stats = norm_stats.get('state_x', {})
    x_mean = np.array(state_stats.get('mean', [0, 0, 0, 0]))
    x_std = np.array(state_stats.get('std', [1, 1, 1, 1]))
    
    control_stats = norm_stats.get('control_u', {})
    u_mean = np.array(control_stats.get('mean', [0]))
    u_std = np.array(control_stats.get('std', [1]))
    
    # Normalize
    x_norm = (trajectories - x_mean) / x_std  # (N, T, 4)
    u_norm = (u - u_mean) / u_std  # (N, T, 1)
    
    # Compute library features for each trajectory
    # Flatten to (N*T, D) for library
    x_flat = x_norm.reshape(-1, D)  # (N*T, 4)
    u_flat = u_norm.reshape(-1, 1)  # (N*T, 1)
    
    Theta_flat = library.fit_transform(x_flat, u_flat)  # (N*T, n_features)
    n_features = Theta_flat.shape[1]
    
    # Reshape back to (N, T, n_features)
    Theta = Theta_flat.reshape(N, T, n_features)
    
    return Theta


def compute_gram_contributions_by_target(
    Theta: np.ndarray,
    F_by_target: Dict[int, np.ndarray],
    gram_energy_mode: str = 'raw',
    trace_power: float = 1.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute Gram matrix contributions G_i for each candidate, per target.
    
    G_i = Θ_F^T @ Θ_F  (summed over time)
    
    v0.3.2: gram_energy_mode 추가 (GPT 피드백)
    - 'raw': 기존 방식 (에너지 그대로)
    - 'unit_trace': G_i ← G_i / (trace(G_i) + eps) (에너지 중립)
    
    v0.3.3: trace_power 추가 (GPT P1 recommendation)
    - 'trace_power': G_i ← G_i / (trace(G_i)**p + eps)
    - p=0 equivalent to raw, p=1 equivalent to unit_trace
    - p=0.7 or 0.85: partial energy normalization (trade-off median vs CI)
    
    Args:
        Theta: (N, T, n_features) library features
        F_by_target: {target_idx: feature indices}
        gram_energy_mode: 'raw', 'unit_trace', or 'trace_power'
        trace_power: power p for trace_power mode (default 1.0)
        
    Returns:
        G_by_target: {target_idx: (N, |F_t|, |F_t|) Gram contributions}
        trace_by_target: {target_idx: (N,) trace values before normalization}
    """
    N, T, n_features = Theta.shape
    G_by_target = {}
    trace_by_target = {}
    eps = 1e-8  # For numerical stability
    
    for t, F_t in F_by_target.items():
        if len(F_t) == 0:
            # No features for this target, use identity contribution
            G_by_target[t] = np.zeros((N, 1, 1))
            trace_by_target[t] = np.zeros(N)
            continue
        
        # Extract fragile features: (N, T, |F_t|)
        Theta_F = Theta[:, :, F_t]
        
        # Compute Gram: G_i = Σ_t Θ_F[i,t]^T @ Θ_F[i,t]
        # For each trajectory i: (|F_t|, T) @ (T, |F_t|) = (|F_t|, |F_t|)
        n_F = len(F_t)
        G = np.zeros((N, n_F, n_F))
        traces = np.zeros(N)
        
        for i in range(N):
            # Theta_F[i] is (T, n_F)
            G[i] = Theta_F[i].T @ Theta_F[i]  # (n_F, n_F)
            traces[i] = np.trace(G[i])
            
            # v0.3.2/v0.3.3: Energy normalization (GPT P1 recommendation)
            if gram_energy_mode == 'unit_trace':
                # p=1: full energy-neutral
                G[i] = G[i] / (traces[i] + eps)
            elif gram_energy_mode == 'trace_power':
                # v0.3.3: partial energy normalization with power p
                # p=0: raw, p=1: unit_trace, p=0.7: partial
                G[i] = G[i] / (traces[i] ** trace_power + eps)
        
        G_by_target[t] = G
        trace_by_target[t] = traces
    
    return G_by_target, trace_by_target


def greedy_dopt_selection(
    G_by_target: Dict[int, np.ndarray],
    candidate_pool_indices: np.ndarray,
    n_select: int,
    lambda_reg: float = 1e-6,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Greedy D-optimal selection maximizing sum of logdet over targets.
    
    Δ_i = Σ_{t} [logdet(G_t + G_{i,t} + λI) - logdet(G_t + λI)]
    
    GPT P0-1 FIX: candidate_pool_indices are actual pool indices, not local 0..M-1
    GPT P0-2: target별 합산
    
    Args:
        G_by_target: {target_idx: (N_cand, |F_t|, |F_t|)} Gram contributions
        candidate_pool_indices: (N_cand,) **pool indices** of candidates (not local)
        n_select: Number to select
        lambda_reg: Regularization for numerical stability
        
    Returns:
        selected_pool_indices: (n_select,) selected pool indices
        selection_trace: List of dicts with local_idx and pool_idx clearly separated
    """
    N_cand = len(candidate_pool_indices)
    
    if N_cand <= n_select:
        return candidate_pool_indices.copy(), [{'note': 'all_candidates_selected', 'n_cand': N_cand}]
    
    # Initialize cumulative Gram matrices (per target)
    G_cumulative = {}
    for t, G_t in G_by_target.items():
        n_F = G_t.shape[1]
        G_cumulative[t] = lambda_reg * np.eye(n_F)  # λI
    
    # Helper: compute logdet safely
    def safe_logdet(M):
        try:
            sign, logdet = np.linalg.slogdet(M)
            if sign <= 0:
                return -np.inf
            return logdet
        except:
            return -np.inf
    
    # Current logdet (sum over targets)
    def current_total_logdet():
        total = 0.0
        for t, G_t in G_cumulative.items():
            total += safe_logdet(G_t)
        return total
    
    selection_trace = []
    selected_local = []  # Local indices (0..N_cand-1) for G_by_target indexing
    remaining_mask = np.ones(N_cand, dtype=bool)
    
    # Greedy selection
    for step in range(n_select):
        if not remaining_mask.any():
            break
        
        remaining_local_indices = np.where(remaining_mask)[0]
        current_logdet = current_total_logdet()
        
        # Compute delta for each remaining candidate
        deltas = np.full(len(remaining_local_indices), -np.inf)
        
        for idx, local_i in enumerate(remaining_local_indices):
            # Compute new logdet if we add this candidate
            new_logdet = 0.0
            for t, G_t in G_cumulative.items():
                G_i_t = G_by_target[t][local_i]  # (|F_t|, |F_t|)
                G_new = G_t + G_i_t
                new_logdet += safe_logdet(G_new)
            
            deltas[idx] = new_logdet - current_logdet
        
        # Select candidate with maximum delta
        best_idx = np.argmax(deltas)
        best_local_idx = remaining_local_indices[best_idx]
        best_delta = deltas[best_idx]
        
        # GPT P0-1 FIX: Tie-break by pool_idx (not local)
        tie_mask = np.abs(deltas - best_delta) < 1e-12
        if tie_mask.sum() > 1:
            tie_local = remaining_local_indices[tie_mask]
            tie_pool_idx = candidate_pool_indices[tie_local]
            best_tie_idx = np.argmin(tie_pool_idx)  # Smallest pool_idx wins
            best_local_idx = tie_local[best_tie_idx]
            best_delta = deltas[tie_mask][best_tie_idx]
        
        # Update cumulative Gram matrices
        for t, G_t in G_cumulative.items():
            G_cumulative[t] = G_t + G_by_target[t][best_local_idx]
        
        selected_local.append(best_local_idx)
        remaining_mask[best_local_idx] = False
        
        new_total_logdet = current_total_logdet()
        
        # GPT P0-1 FIX: trace에 local_idx와 pool_idx 명확히 구분
        selection_trace.append({
            'step': step,
            'local_idx': int(best_local_idx),  # Index within pre-gate candidates (0..M-1)
            'pool_idx': int(candidate_pool_indices[best_local_idx]),  # Actual pool index
            'delta_logdet': float(best_delta),
            'cumulative_logdet': float(new_total_logdet),
            'n_remaining': int(remaining_mask.sum()),
        })
        
        # Progress logging (every 50 steps)
        if step % 50 == 0 or step == n_select - 1:
            print(f"    D-opt step {step}: Δlogdet={best_delta:.4f}, cumulative={new_total_logdet:.4f}")
    
    # Return pool indices directly
    selected_pool_indices = candidate_pool_indices[np.array(selected_local)]
    
    return selected_pool_indices, selection_trace


def track_b_dopt_selection(
    pool: Dict[str, Any],
    track_a_result: Dict[str, Any],
    fragile_pairs: List[Tuple[int, int]],
    teacher_support: np.ndarray,
    norm_stats: Dict[str, Any],
    library: 'SINDyLibrary',
    n_select: int = 240,
    top_m_ratio: float = 5.0,
    lambda_reg: float = 1e-6,
    use_teacher_intersection: bool = True,
    dynamics_target_indices: List[int] = [1, 3],
    pre_gate_mode: str = 'score',
    alpha: float = 0.3,
    gram_energy_mode: str = 'raw',  # v0.3.2: 'raw' or 'unit_trace'
    trace_power: float = 1.0,  # v0.3.3: power p for trace_power mode
    rng: Optional[np.random.Generator] = None,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Track B Selection v0.3.3: D-optimal on fragile features (Θ_F).
    
    GPT P0 피드백 반영:
    - P0-1: Θ 계산 시 정규화 입력
    - P0-2: target별 F_t, 블록-합 logdet
    - P0-3: F_t = fragile ∩ teacher_active
    - Pre-gate: score top_M 후 D-opt
    
    v0.3.1: SSOT 강화 (pool_idx 추적, tie-break 고정)
    v0.3.2: 에너지-중립 D-opt (gram_energy_mode='unit_trace')
    v0.3.3: trace_power 일반화 (p=0=raw, p=1=unit_trace, p=0.7=partial)
    
    Args:
        pool: Pool generator output
        track_a_result: Track A selection result
        fragile_pairs: List of (feature_idx, target_idx)
        teacher_support: (n_features, n_targets) boolean mask
        norm_stats: Normalization statistics
        library: SINDy library instance
        n_select: Number to select
        top_m_ratio: Pre-gate M = ratio × n_select
        lambda_reg: Regularization for logdet
        use_teacher_intersection: If True, F = fragile ∩ teacher_active
        dynamics_target_indices: [1, 3] for x_ddot, theta_ddot
        pre_gate_mode: 'score' or 'none'
        alpha: Score alpha (for pre-gate)
        gram_energy_mode: 'raw', 'unit_trace', or 'trace_power'
        trace_power: power p for trace_power mode (default 1.0)
        rng: Random generator
        results_dir: Directory to save artifacts
        
    Returns:
        Dict with selected trajectories, artifacts, and metadata
    """
    print(f"\n[Track B Selection v0.3.3 D-opt] n_select={n_select}")
    print(f"  top_m_ratio={top_m_ratio}, lambda={lambda_reg}")
    print(f"  use_teacher_intersection={use_teacher_intersection}")
    print(f"  pre_gate_mode={pre_gate_mode}, alpha={alpha}")
    print(f"  gram_energy_mode={gram_energy_mode}, trace_power={trace_power}")  # v0.3.3
    
    trajectories = pool['trajectories']
    u = pool['u']
    N_pool = len(trajectories)
    
    # 1. Get Track A passed candidates
    track_a_indices = track_a_result['selected_indices']
    track_a_stats = track_a_result['stats']
    relaxed_flag = track_a_stats.get('relaxed_flag', False)
    
    print(f"  Track A passed: {len(track_a_indices)}/{N_pool}")
    if relaxed_flag:
        print(f"  ⚠️ Track A was RELAXED")
    
    # 2. Compute fragile feature sets (P0-2, P0-3)
    F_by_target = compute_fragile_feature_sets(
        fragile_pairs=fragile_pairs,
        teacher_support=teacher_support,
        dynamics_target_indices=dynamics_target_indices,
        use_teacher_intersection=use_teacher_intersection,
    )
    
    for t, F_t in F_by_target.items():
        target_name = ['x', 'x_ddot', 'theta', 'theta_ddot'][t]
        print(f"  F_{target_name}: {len(F_t)} features")
    
    # 3. Pre-gate: score top_M (P1-2)
    if pre_gate_mode == 'score':
        # Compute scores for pre-gate
        features = compute_track_b_features(trajectories)
        track_a_features = features[track_a_indices]
        norm_stats_score = compute_track_b_normalization_stats(track_a_features)
        features_normalized = normalize_track_b_features(features, norm_stats_score)
        scores = compute_track_b_scores(features_normalized, alpha=alpha)
        
        # Get scores for Track A passed
        track_a_scores = scores[track_a_indices]
        
        # Top-M gate
        M = int(top_m_ratio * n_select)
        M = min(M, len(track_a_indices))
        
        # GPT P0-2 FIX: lexsort for deterministic tie-break (score desc, pool_idx asc)
        # lexsort sorts by last key first, so: (track_a_indices, -track_a_scores)
        # = primary: -score (desc), secondary: pool_idx (asc)
        sort_order = np.lexsort((track_a_indices, -track_a_scores))[:M]
        pre_gate_indices = track_a_indices[sort_order]
        pre_gate_scores = track_a_scores[sort_order]
        
        print(f"  Pre-gate (score): M={M} from {len(track_a_indices)}")
        print(f"  Pre-gate score range: [{pre_gate_scores.min():.4f}, {pre_gate_scores.max():.4f}]")
    else:
        # No pre-gate, use all Track A passed
        pre_gate_indices = track_a_indices
        pre_gate_scores = None
        print(f"  Pre-gate: none (using all Track A passed)")
    
    # 4. Compute Θ for pre-gate candidates (P0-1: normalized)
    print(f"  Computing library features (normalized)...")
    pre_gate_trajectories = trajectories[pre_gate_indices]
    pre_gate_u = u[pre_gate_indices]
    
    Theta = compute_library_features_normalized(
        trajectories=pre_gate_trajectories,
        u=pre_gate_u,
        norm_stats=norm_stats,
        library=library,
    )
    print(f"  Θ shape: {Theta.shape}")
    
    # 5. Compute Gram contributions per target (v0.3.3: energy mode + trace_power)
    print(f"  Computing Gram contributions (mode={gram_energy_mode}, p={trace_power})...")
    G_by_target, trace_by_target = compute_gram_contributions_by_target(
        Theta, F_by_target, gram_energy_mode=gram_energy_mode, trace_power=trace_power
    )
    
    for t, G_t in G_by_target.items():
        target_name = ['x', 'x_ddot', 'theta', 'theta_ddot'][t]
        trace_stats = trace_by_target[t]
        print(f"  G_{target_name}: shape={G_t.shape}, trace range=[{trace_stats.min():.2f}, {trace_stats.max():.2f}]")
    
    # 6. Greedy D-optimal selection
    print(f"  Running greedy D-opt selection...")
    # GPT P0-1 FIX: Pass pool indices directly (not local 0..M-1)
    # greedy_dopt_selection will use these for tie-break and trace
    
    selected_pool_indices, selection_trace = greedy_dopt_selection(
        G_by_target=G_by_target,
        candidate_pool_indices=pre_gate_indices,  # Pool indices, not local
        n_select=n_select,
        lambda_reg=lambda_reg,
    )
    
    # v0.3.2: Keep both order (greedy selection order) and canonical (sorted)
    selected_pool_indices_order = selected_pool_indices.copy()  # Greedy selection order
    selected_indices_canonical = np.sort(selected_pool_indices)  # Canonical (sorted)
    
    n_selected = len(selected_pool_indices)
    print(f"  Selected: {n_selected} trajectories")
    
    # 8. Build feature spec (v0.3.3 with trace_power + SSOT enhancements)
    feature_spec = {
        'version': 'v0.3.3',  # Version bump for trace_power
        'method': 'd_optimal',
        'objective': 'sum_logdet_by_target',
        'dynamics_target_indices': dynamics_target_indices,
        'use_teacher_intersection': use_teacher_intersection,
        'lambda_reg': lambda_reg,
        'pre_gate_mode': pre_gate_mode,
        'top_m_ratio': top_m_ratio,
        'alpha': alpha,
        'gram_energy_mode': gram_energy_mode,  # v0.3.2: 'raw' or 'unit_trace'
        'trace_power': trace_power,  # v0.3.3: power p for trace_power mode
        'tie_break': 'pool_idx_asc',  # GPT P0-1 FIX: clarify it's pool index
        'pre_gate_tie_break': 'score_desc_pool_idx_asc',  # GPT P0-2 FIX
        'ci_quantiles': [0.025, 0.975],  # v0.3.2: CI definition SSOT (GPT feedback)
        'F_by_target': {str(t): F_t.tolist() for t, F_t in F_by_target.items()},
    }
    spec_str = json.dumps(feature_spec, sort_keys=True)
    feature_spec['spec_hash'] = hashlib.sha256(spec_str.encode()).hexdigest()[:16]
    
    # 9. Save artifacts
    artifacts_saved = {}
    if results_dir is not None:
        results_dir = Path(results_dir)
        
        # track_b_dopt_spec.json
        spec_path = results_dir / 'track_b_dopt_spec.json'
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(feature_spec, f, indent=2)
        artifacts_saved['dopt_spec'] = str(spec_path)
        
        # track_b_F_by_target.json
        F_path = results_dir / 'track_b_F_by_target.json'
        F_data = {str(t): F_t.tolist() for t, F_t in F_by_target.items()}
        with open(F_path, 'w', encoding='utf-8') as f:
            json.dump(F_data, f, indent=2)
        artifacts_saved['F_by_target'] = str(F_path)
        
        # track_b_dopt_trace.json (v0.3.2)
        trace_data = {
            'version': 'v0.3.3',
            'n_pool': N_pool,
            'n_track_a_passed': len(track_a_indices),
            'n_pre_gate': len(pre_gate_indices),
            'n_selected': n_selected,
            'lambda_reg': lambda_reg,
            'gram_energy_mode': gram_energy_mode,  # v0.3.2
            'trace_power': trace_power,  # v0.3.3
            'use_teacher_intersection': use_teacher_intersection,
            'pre_gate_mode': pre_gate_mode,
            'pre_gate_tie_break': 'score_desc_pool_idx_asc',
            'selection_tie_break': 'pool_idx_asc',
            'ci_quantiles': [0.025, 0.975],  # v0.3.2: CI definition SSOT
            'selection_trace': selection_trace,
        }
        trace_path = results_dir / 'track_b_dopt_trace.json'
        with open(trace_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2)
        artifacts_saved['dopt_trace'] = str(trace_path)
        
        # GPT 권장: pre_gate_indices.npy (audit 가능성)
        pre_gate_path = results_dir / 'pre_gate_indices.npy'
        np.save(pre_gate_path, pre_gate_indices)
        artifacts_saved['pre_gate_indices'] = str(pre_gate_path)
        
        # GPT 권장: pre_gate_scores.npy (if available)
        if pre_gate_scores is not None:
            pre_gate_scores_path = results_dir / 'pre_gate_scores.npy'
            np.save(pre_gate_scores_path, pre_gate_scores)
            artifacts_saved['pre_gate_scores'] = str(pre_gate_scores_path)
        
        # v0.3.2 GPT 권장: selected indices order/canon 분리
        # selected_pool_indices_order.npy (greedy selection order)
        order_path = results_dir / 'selected_pool_indices_order.npy'
        np.save(order_path, selected_pool_indices_order)
        artifacts_saved['selected_indices_order'] = str(order_path)
        
        # selected_pool_indices_canon.npy (canonical = sorted)
        canon_path = results_dir / 'selected_pool_indices_canon.npy'
        np.save(canon_path, selected_indices_canonical)
        artifacts_saved['selected_indices_canon'] = str(canon_path)
        
        # v0.3.3 GPT 권장: trace_by_target (에너지 분석용 + quantiles + log_stats)
        trace_stats_path = results_dir / 'trace_by_target.json'
        trace_stats_data = {}
        for t, traces in trace_by_target.items():
            # Basic stats
            stats = {
                'min': float(traces.min()),
                'max': float(traces.max()),
                'mean': float(traces.mean()),
                'std': float(traces.std()),
                # P0-3 v0.3.3: Quantiles (GPT P1 recommendation)
                'p01': float(np.percentile(traces, 1)),
                'p05': float(np.percentile(traces, 5)),
                'p50': float(np.percentile(traces, 50)),
                'p95': float(np.percentile(traces, 95)),
                'p99': float(np.percentile(traces, 99)),
            }
            # P0-3 v0.3.3: Log-scale stats (for large-scale distributions)
            log_traces = np.log(traces + 1e-10)  # Avoid log(0)
            stats['log_mean'] = float(log_traces.mean())
            stats['log_std'] = float(log_traces.std())
            stats['log_p50'] = float(np.percentile(log_traces, 50))
            # Audit-friendly: include energy mode setting
            stats['gram_energy_mode'] = gram_energy_mode
            trace_stats_data[str(t)] = stats
        
        with open(trace_stats_path, 'w', encoding='utf-8') as f:
            json.dump(trace_stats_data, f, indent=2)
        artifacts_saved['trace_by_target'] = str(trace_stats_path)
        
        # Legacy: selected_pool_indices.npy (= canon for backward compatibility)
        indices_path = results_dir / 'selected_pool_indices.npy'
        np.save(indices_path, selected_indices_canonical)
        artifacts_saved['selected_indices'] = str(indices_path)
        
        # track_b_stage1_pass_mask.npy
        pass_mask = np.zeros(N_pool, dtype=bool)
        pass_mask[track_a_indices] = True
        mask_path = results_dir / 'track_b_stage1_pass_mask.npy'
        np.save(mask_path, pass_mask)
        artifacts_saved['stage1_pass_mask'] = str(mask_path)
        
        print(f"  ✅ Artifacts saved: {len(artifacts_saved)} files")
    
    # 10. Compute final Gram logdet
    # Map selected_pool_indices back to local indices for G_by_target
    pool_to_local = {pool_idx: local_i for local_i, pool_idx in enumerate(pre_gate_indices)}
    selected_local = [pool_to_local[pool_idx] for pool_idx in selected_pool_indices]
    
    final_logdet = 0.0
    G_final_by_target = {}
    for t, G_t in G_by_target.items():
        G_sum = lambda_reg * np.eye(G_t.shape[1])
        for local_i in selected_local:
            G_sum += G_t[local_i]
        G_final_by_target[t] = G_sum
        sign, ld = np.linalg.slogdet(G_sum)
        if sign > 0:
            final_logdet += ld
    
    print(f"  Final total logdet: {final_logdet:.4f}")
    
    # 11. Extract selected data
    all_errors = track_a_result.get('errors', np.zeros(N_pool))
    
    result = {
        'trajectories': pool['trajectories'][selected_indices_canonical],
        'dx': pool['dx'][selected_indices_canonical],
        'params': pool['params'][selected_indices_canonical],
        'ic': pool['ic'][selected_indices_canonical],
        'u': pool['u'][selected_indices_canonical],
        'u_indices': pool['u_indices'][selected_indices_canonical],
        'errors': all_errors[selected_indices_canonical],
        'original_indices': selected_indices_canonical.copy(),
        'stats': {
            'n_pool': N_pool,
            'n_track_a_passed': len(track_a_indices),
            'track_a_relaxed': relaxed_flag,
            'n_pre_gate': len(pre_gate_indices),
            'n_selected': n_selected,
            'selection_mode': 'track_b_dopt',
            'version': 'v0.3.3',  # trace_power + SSOT
            'lambda_reg': lambda_reg,
            'gram_energy_mode': gram_energy_mode,  # v0.3.2
            'trace_power': trace_power,  # v0.3.3
            'use_teacher_intersection': use_teacher_intersection,
            'pre_gate_mode': pre_gate_mode,
            'top_m_ratio': top_m_ratio,
            'alpha': alpha,
            'feature_spec_hash': feature_spec['spec_hash'],
            'final_logdet': float(final_logdet),
            'canonicalized': True,
        },
        'artifacts_saved': artifacts_saved,
        'feature_spec': feature_spec,
        'F_by_target': F_by_target,
        'G_final_by_target': G_final_by_target,
        'trace_by_target': trace_by_target,  # v0.3.2
    }
    
    return result


# ============================================================
# E-SINDy Evaluation
# ============================================================

def evaluate_with_esindy(
    train_x: np.ndarray,
    train_dx: np.ndarray,
    train_u: np.ndarray,
    aug_x: np.ndarray,
    aug_dx: np.ndarray,
    aug_u: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    bootstrap_B: int,
    threshold: float,
    seed: int,
    tau_support: float,
    z0: float,
    eps: float,
) -> Dict[str, Any]:
    """
    Evaluate augmented data using E-SINDy.
    
    Args:
        train_x: Original training trajectories (N_train, T, 4)
        train_dx: Original training dx (N_train, T, 4)
        train_u: Original training u (N_train, T, 1)
        aug_x: Augmented trajectories (N_aug, T, 4)
        aug_dx: Augmented dx (N_aug, T, 4)
        aug_u: Augmented u (N_aug, T, 1)
        feature_names: Feature names
        target_names: Target names
        bootstrap_B: Number of bootstrap samples
        threshold: STLSQ threshold
        seed: Random seed
        tau_support: Support threshold
        z0: z-score threshold for stable core
        eps: Small constant for z-score computation
        
    Returns:
        Dict with z_after, inc_prob_after, coefficients, etc.
    """
    print(f"\n[E-SINDy Evaluation] B={bootstrap_B}, threshold={threshold}")
    
    # Combine original + augmented
    combined_x = np.concatenate([train_x, aug_x], axis=0)
    combined_dx = np.concatenate([train_dx, aug_dx], axis=0)
    combined_u = np.concatenate([train_u, aug_u], axis=0)
    
    N_total, T, D = combined_x.shape
    n_targets = len(target_names)
    
    print(f"  Combined: {train_x.shape[0]} orig + {aug_x.shape[0]} aug = {N_total}")
    
    # Build feature matrix using gate0_min config
    library = SINDyLibrary(config='gate0_min')
    
    # Flatten for library (N*T, D)
    x_flat = combined_x.reshape(-1, D)
    u_flat = combined_u.reshape(-1, 1)
    
    Theta = library.fit_transform(x_flat, u_flat)
    dx_flat = combined_dx.reshape(-1, n_targets)
    
    print(f"  Theta shape: {Theta.shape}")
    print(f"  dx shape: {dx_flat.shape}")
    
    # Scale features
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)
    
    # Fit E-SINDy
    ensemble = ESINDyEnsemble(
        n_bootstrap=bootstrap_B,
        threshold=threshold,
        random_state=seed,
    )
    
    ensemble.fit(
        Theta_scaled,
        dx_flat,
        n_trajectories=N_total,
        T=T,
        scaler=scaler,
        target_scale=None,
    )
    
    # Extract results
    coef_mean = ensemble.coefficients_mean_
    coef_std = ensemble.coefficients_std_
    inc_prob = ensemble.inclusion_probability_
    
    # Compute z-scores
    z_scores = np.abs(coef_mean) / (coef_std + eps)
    
    # Support mask
    support_mask = inc_prob >= tau_support
    
    # Stable core mask
    stable_core_mask = support_mask & (z_scores >= z0)
    
    # Fragile pool mask
    fragile_pool_mask = support_mask & (z_scores < z0)
    
    result = {
        'coefficients_mean': coef_mean,
        'coefficients_std': coef_std,
        'inclusion_probability': inc_prob,
        'z_scores': z_scores,
        'support_mask': support_mask,
        'stable_core_mask': stable_core_mask,
        'fragile_pool_mask': fragile_pool_mask,
        'n_total': N_total,
        'n_original': train_x.shape[0],
        'n_augmented': aug_x.shape[0],
        'feature_names': feature_names,
        'target_names': target_names,
        'bootstrap_B': bootstrap_B,
        'threshold': threshold,
    }
    
    # Summary
    n_support = support_mask.sum()
    n_stable = stable_core_mask.sum()
    n_fragile = fragile_pool_mask.sum()
    
    print(f"  Support: {n_support}, Stable: {n_stable}, Fragile: {n_fragile}")
    
    return result


# ============================================================
# Gate3 Treatment Runner
# ============================================================

class Gate3TreatRunner:
    """
    Gate3 Treatment Runner: Generate and evaluate augmented data.
    
    Pipeline:
    1. Load Day3 baseline (teacher_support, z_before)
    2. Load Gate1 teacher coefficients
    3. Fit GMM to training IC + params
    4. Generate pool OR load existing pool (D1 Rebaseline)
    5. Track A selection (filter top 10% error)
    6. Final selection (random or track_a_filtered_random)
    7. Combine: orig + selected = n_total
    8. E-SINDy evaluation
    9. Save artifacts (including selected_pool_indices.npy)
    
    D1 Rebaseline: Supports RNG stream separation and pool reuse.
    """
    
    def __init__(self, config: Gate3Config):
        self.config = config
        self.run_id = generate_run_id(config.note)
        self.project_root = Path(__file__).resolve().parent.parent
        
        # D1 Rebaseline: Create RNG streams (SSOT P0-1)
        self.rng_streams = create_rng_streams(config.seed)
        
        # Setup results directory
        self.results_dir = self._setup_results_dir()
        
        # Will be loaded/computed
        self.day3_manifest = None
        self.teacher_support = None
        self.teacher_coefficients = None
        self.teacher_run_id = None
        self.z_before = None
        self.dataset = None
        self.gmm_sampler = None
        self.feature_names = None
        self.target_names = None
        self.norm_stats = None  # D1: SSOT for dx_std
        
        print("=" * 60)
        print(f"  Gate3 Treatment Runner (D1 Rebaseline)")
        print("=" * 60)
        print(f"  Run ID: {self.run_id}")
        print(f"  Day3 baseline: {config.day3_run_id}")
        print(f"  Variant: {config.variant}")
        print(f"  Bootstrap B: {config.bootstrap_B}")
        print(f"  n_select: {config.n_select} (n_total={config.n_train}+{config.n_select}={config.n_train + config.n_select})")
        print(f"  Selection mode: {config.selection_mode}")
        if config.pool_source:
            print(f"  Pool source: {config.pool_source}")
        else:
            print(f"  Pool source: NEW (target={config.target_n_accept})")
        print(f"  Results: {self.results_dir}")
        print("=" * 60)
    
    def _setup_results_dir(self) -> Path:
        """Setup results directory."""
        cfg = self.config
        results_dir = (
            self.project_root / 'results' / cfg.dataset_version / 'gate3' /
            cfg.track / cfg.method / f"n{cfg.n_train}" / f"seed{cfg.seed}" / self.run_id
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / 'figures').mkdir(exist_ok=True)
        return results_dir
    
    def _get_day3_dir(self) -> Path:
        """Get Day3 baseline directory."""
        cfg = self.config
        return (
            self.project_root / 'results' / cfg.dataset_version / 'phase35' /
            cfg.track / cfg.method / f"n{cfg.n_train}" / f"seed{cfg.seed}" / cfg.day3_run_id
        )
    
    def _load_day3_artifacts(self):
        """Load Day3 baseline artifacts and Gate1 teacher coefficients."""
        print("\n[Loading Day3 Artifacts]")
        
        day3_dir = self._get_day3_dir()
        
        if not day3_dir.exists():
            raise FileNotFoundError(f"Day3 directory not found: {day3_dir}")
        
        # Load manifest
        manifest_path = day3_dir / 'manifest.json'
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.day3_manifest = json.load(f)
        
        # Load teacher_support
        teacher_support_path = day3_dir / 'teacher_support.npy'
        self.teacher_support = np.load(teacher_support_path)
        
        # Load z_before
        z_before_path = day3_dir / 'z_before.npy'
        self.z_before = np.load(z_before_path)
        
        # Verify SHA256
        computed_sha = compute_file_hash(teacher_support_path)
        expected_sha = self.day3_manifest.get('teacher_support_sha256', '')
        
        if computed_sha != expected_sha:
            print(f"  ⚠️ teacher_support SHA256 mismatch!")
            print(f"    Expected: {expected_sha[:16]}...")
            print(f"    Computed: {computed_sha[:16]}...")
        else:
            print(f"  ✅ teacher_support SHA256 verified: {computed_sha[:16]}...")
        
        print(f"  teacher_support shape: {self.teacher_support.shape}")
        print(f"  z_before shape: {self.z_before.shape}")
        
        # Extract teacher_run_id from Day3 manifest
        gate1_artifacts = self.day3_manifest.get('gate1_artifacts', {})
        self.teacher_run_id = gate1_artifacts.get('teacher_run_id', '')
        
        if not self.teacher_run_id:
            raise ValueError("teacher_run_id not found in Day3 manifest")
        
        print(f"\n[Loading Gate1 Teacher Coefficients]")
        print(f"  Teacher run_id: {self.teacher_run_id}")
        
        # Load teacher coefficients from Gate1
        self.teacher_coefficients, self.feature_names, self.target_names = load_teacher_coefficients(
            teacher_run_id=self.teacher_run_id,
            project_root=self.project_root,
            config=self.config,
        )
        
        return self.day3_manifest
    
    def _load_dataset(self):
        """Load dataset."""
        print("\n[Loading Dataset]")
        
        cfg = self.config
        
        if cfg.dataset_path:
            dataset_path = Path(cfg.dataset_path)
        else:
            dataset_path = self.project_root / 'data' / 'cartpole' / cfg.dataset_version / 'dataset.npz'
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        self.dataset = np.load(dataset_path, allow_pickle=True)
        
        # v2.1 FIX: Load norm_stats for alignment error calculation
        self.norm_stats = load_norm_stats(
            dataset_version=cfg.dataset_version,
            system='cartpole',
            project_root=self.project_root,
        )
        self.norm_stats_path = self.project_root / 'data' / 'cartpole' / cfg.dataset_version / 'norm_stats.json'
        print(f"  ✅ norm_stats loaded (for v2.1 alignment fix)")
        
        # Verify shape
        train_x = self.dataset['train_x']
        N_train, T, D = train_x.shape
        
        print(f"  Dataset: {dataset_path.name}")
        print(f"  train_x shape: {train_x.shape}")
        print(f"  train_params shape: {self.dataset['train_params'].shape}")
        
        return self.dataset
    
    def _fit_gmm_sampler(self):
        """Fit GMM to training data."""
        print("\n[Fitting GMM Sampler]")
        
        train_x = self.dataset['train_x'][:self.config.n_train]  # Use n_train
        train_params = self.dataset['train_params'][:self.config.n_train]
        
        self.gmm_sampler = GMMProposalSampler(
            n_components=GATE3_CONFIG['gmm_n_components'],
            covariance_type=GATE3_CONFIG['gmm_covariance_type'],
            random_state=self.config.seed,
        )
        self.gmm_sampler.fit(train_x, train_params)
        
        return self.gmm_sampler
    
    def run(self) -> Dict[str, Any]:
        """
        Run Gate3 treatment pipeline.
        
        D1 Rebaseline changes:
        - Pool reuse: load existing pool if pool_source provided
        - RNG stream separation: pool/select/bootstrap
        - dx_std from norm_stats.json (SSOT, no recompute)
        - Save selected_pool_indices.npy, track_a_pass_mask.npy
        """
        cfg = self.config
        
        # 1. Load Day3 artifacts
        self._load_day3_artifacts()
        
        # 2. Load dataset
        self._load_dataset()
        
        # 3. Get dx_std from norm_stats (SSOT - GPT P0: no recompute)
        dynamics_target_indices = [1, 3]  # SSOT: x_ddot, theta_ddot
        dx_std = self._get_dx_std_from_norm_stats()
        self._dx_std = dx_std  # Save for manifest
        dx_std_dynamics = dx_std[dynamics_target_indices]
        print(f"\n[dx_std from norm_stats.json (SSOT)]")
        print(f"  All targets: {dx_std}")
        print(f"  Dynamics only [1,3]: {dx_std_dynamics}")
        
        # 4. Pool: Load existing OR generate new
        if cfg.pool_source:
            pool, pool_meta = self._load_existing_pool(cfg.pool_source)
            print(f"\n[Pool Loaded] from {cfg.pool_source}")
            print(f"  n_trajectories: {len(pool['trajectories'])}")
        else:
            # Fit GMM sampler
            self._fit_gmm_sampler()
            
            # Generate pool with rng_pool
            train_u = self.dataset['train_u'][:cfg.n_train]
            
            pool_generator = PoolGenerator(
                gmm_sampler=self.gmm_sampler,
                train_u=train_u,
                config=GATE3_CONFIG,
                fixed_physics=GATE3_CONFIG['fixed_physics'],
                seed=cfg.seed,
                rng=self.rng_streams['pool'],  # D1: Use rng_pool
            )
            
            pool = pool_generator.generate_pool(
                target_n_accept=cfg.target_n_accept,
                max_attempts=GATE3_CONFIG['max_pool_attempts'],
                variant=cfg.variant,
            )
            pool_meta = None  # Will be created during save
        
        # 5. Track A selection (compute alignment errors)
        track_a_result = track_a_selection(
            pool=pool,
            teacher_coefficients=self.teacher_coefficients,
            feature_names=self.feature_names,
            target_names=self.target_names,
            dx_std=dx_std,
            norm_stats=self.norm_stats,
            reject_ratio=cfg.reject_ratio,  # D1: Use config value
            n_select=cfg.n_select,
            dynamics_target_indices=dynamics_target_indices,
        )
        
        # 6. Final selection with rng_select and selection_mode
        if cfg.selection_mode == 'track_b':
            # Track B v0.2: Use dedicated track_b_selection function
            selected = track_b_selection(
                pool=pool,
                track_a_result=track_a_result,
                n_select=cfg.n_select,
                alpha=cfg.track_b_alpha,
                diversity_mode=cfg.track_b_diversity_mode,  # v0.2: 'score_only' or 'top_m_diversity'
                top_m_ratio=cfg.track_b_top_m_ratio,        # v0.2: M = ratio × n_select
                score_floor=cfg.track_b_score_floor,        # v0.2: minimum score threshold
                rng=self.rng_streams['select'],
                results_dir=self.results_dir,
            )
        elif cfg.selection_mode == 'd_optimal':
            # Track B v0.3: D-optimal on fragile features
            # Load fragile_pairs from existing source (required for d_optimal)
            if not cfg.fragile_pairs_source:
                raise ValueError("d_optimal selection requires --fragile_pairs_source")
            
            fragile_pairs_path = Path(cfg.fragile_pairs_source)
            if not fragile_pairs_path.exists():
                # Try relative to project root
                fragile_pairs_path = self.project_root / cfg.fragile_pairs_source
            
            if not fragile_pairs_path.exists():
                raise FileNotFoundError(f"fragile_pairs not found: {cfg.fragile_pairs_source}")
            
            with open(fragile_pairs_path, 'r') as f:
                fragile_data = json.load(f)
            # Support both 'pairs' and 'fragile_pairs' keys
            if 'pairs' in fragile_data:
                fragile_pairs = [tuple(pair) for pair in fragile_data['pairs']]
            elif 'fragile_pairs' in fragile_data:
                fragile_pairs = [tuple(pair) for pair in fragile_data['fragile_pairs']]
            else:
                raise KeyError(f"Expected 'pairs' or 'fragile_pairs' in {fragile_pairs_path}")
            print(f"  Loaded fragile_pairs: n={len(fragile_pairs)} from {fragile_pairs_path}")
            
            # Create SINDy library (same as Gate0/Gate1)
            library = SINDyLibrary(config='gate0_min')
            
            selected = track_b_dopt_selection(
                pool=pool,
                track_a_result=track_a_result,
                fragile_pairs=fragile_pairs,
                teacher_support=self.teacher_support,
                norm_stats=self.norm_stats,
                library=library,
                n_select=cfg.n_select,
                top_m_ratio=cfg.track_b_top_m_ratio,
                lambda_reg=cfg.track_b_dopt_lambda,
                use_teacher_intersection=cfg.track_b_dopt_use_teacher_intersection,
                dynamics_target_indices=[1, 3],  # x_ddot, theta_ddot
                pre_gate_mode=cfg.track_b_dopt_pre_gate_mode,
                alpha=cfg.track_b_alpha,
                gram_energy_mode=cfg.track_b_dopt_gram_energy_mode,  # v0.3.2
                trace_power=cfg.track_b_dopt_trace_power,  # v0.3.3
                rng=self.rng_streams['select'],
                results_dir=self.results_dir,
            )
        else:
            # Random or Track A filtered random
            selected = final_selection(
                pool=pool,
                track_a_result=track_a_result,
                n_select=cfg.n_select,
                seed=cfg.seed,
                selection_mode=cfg.selection_mode,  # D1: Use config value
                rng=self.rng_streams['select'],     # D1: Use rng_select
            )
        
        # 7. E-SINDy evaluation
        # D1: Use bootstrap seed derived from SeedSequence for reproducibility
        bootstrap_seed = int(self.rng_streams['bootstrap'].integers(0, 2**31))
        
        train_x = self.dataset['train_x'][:cfg.n_train]
        train_dx = self.dataset['train_dx'][:cfg.n_train]
        train_u_data = self.dataset['train_u'][:cfg.n_train]
        
        esindy_result = evaluate_with_esindy(
            train_x=train_x,
            train_dx=train_dx,
            train_u=train_u_data,
            aug_x=selected['trajectories'],
            aug_dx=selected['dx'],
            aug_u=selected['u'],
            feature_names=self.feature_names,
            target_names=self.target_names,
            bootstrap_B=cfg.bootstrap_B,
            threshold=cfg.threshold,
            seed=bootstrap_seed,  # D1: Use derived bootstrap seed
            tau_support=cfg.tau_support,
            z0=cfg.z0,
            eps=cfg.eps,
        )
        
        # 8. Save artifacts (including D1 SSOT artifacts)
        self._save_artifacts(
            pool, track_a_result, selected, esindy_result,
            pool_meta=pool_meta,
            bootstrap_seed=bootstrap_seed,
        )
        
        return {
            'run_id': self.run_id,
            'results_dir': str(self.results_dir),
            'esindy_result': esindy_result,
            'status': 'success',
        }
    
    def _get_dx_std_from_norm_stats(self) -> np.ndarray:
        """
        Get dx_std from norm_stats.json (SSOT).
        
        GPT P0: dx_std must come from norm_stats.json, NOT recomputed from training data.
        """
        if self.norm_stats is None:
            raise RuntimeError("norm_stats not loaded. Call _load_day3_artifacts first.")
        
        # norm_stats structure: derivative_dx_savgol.std = [4 values]
        deriv_stats = self.norm_stats.get('derivative_dx_savgol', {})
        dx_std = np.array(deriv_stats.get('std', []))
        
        if len(dx_std) != 4:
            raise ValueError(f"dx_std from norm_stats has wrong shape: {dx_std.shape}")
        
        return dx_std
    
    def _load_existing_pool(self, pool_source: str) -> Tuple[Dict, Dict]:
        """
        Load existing pool from file with SHA256 verification.
        
        D1 SSOT: Pool reuse for ablation fairness.
        """
        pool_path = Path(pool_source)
        if not pool_path.exists():
            raise FileNotFoundError(f"Pool source not found: {pool_source}")
        
        print(f"\n[Loading Existing Pool]")
        print(f"  Source: {pool_source}")
        
        # Load pool data
        pool_data = np.load(pool_path, allow_pickle=True)
        
        n_trajectories = len(pool_data['trajectories'])
        
        pool = {
            'trajectories': pool_data['trajectories'],
            'dx': pool_data['dx'],
            'params': pool_data['params'],
            'ic': pool_data['ic'],
            'u': pool_data['u'],
            'u_indices': pool_data['u_indices'],
            # D1 FIX: Add default stats for loaded pool
            'stats': {
                'n_accepted': n_trajectories,
                'n_attempts': -1,  # Unknown for loaded pool
                'acceptance_rate': -1.0,  # Unknown
                'loaded_from': str(pool_path),
            },
        }
        
        # Load metadata if available
        pool_meta = {}
        if 'alignment_cfg' in pool_data:
            pool_meta['alignment_cfg'] = pool_data['alignment_cfg'].item()
        if 'alignment_errors' in pool_data:
            pool['alignment_errors'] = pool_data['alignment_errors']
        
        # Try to load original generation stats if available
        if 'generation_seed' in pool_data:
            pool['stats']['generation_seed'] = int(pool_data['generation_seed'])
        if 'generation_variant' in pool_data:
            pool['stats']['generation_variant'] = str(pool_data['generation_variant'])
        if 'target_n_accept' in pool_data:
            pool['stats']['target_n_accept'] = int(pool_data['target_n_accept'])
        
        # Compute SHA256 for verification
        pool_sha256 = compute_file_hash(pool_path)
        pool_meta['pool_sha256'] = pool_sha256
        pool_meta['pool_source'] = str(pool_path)
        
        print(f"  Pool SHA256: {pool_sha256[:16]}...")
        print(f"  n_trajectories: {n_trajectories}")
        
        return pool, pool_meta
    
    def _build_generation_config(
        self, 
        cfg: Gate3Config, 
        pool: Dict, 
        pool_meta: Optional[Dict]
    ) -> Dict:
        """
        Build generation_config for manifest.
        
        P0-B FIX: If pool was loaded, record original pool's generation params,
        not current CLI defaults.
        """
        is_pool_loaded = bool(pool_meta and pool_meta.get('pool_source'))
        
        if is_pool_loaded:
            # Pool was loaded - record that fact, use original pool's stats
            return {
                'status': 'pool_loaded',
                'pool_source': pool_meta.get('pool_source'),
                'pool_sha256': pool_meta.get('pool_sha256'),
                'original_target_n_accept': pool['stats'].get('target_n_accept', -1),
                'original_generation_seed': pool['stats'].get('generation_seed', -1),
                'original_generation_variant': pool['stats'].get('generation_variant', 'unknown'),
                # Current run's selection params
                'n_select': cfg.n_select,
                'track_a_reject_ratio': cfg.reject_ratio,
                'final_selection_mode': cfg.selection_mode,
                'dynamics_target_indices': [1, 3],
                'dynamics_target_names': ['x_ddot', 'theta_ddot'],
            }
        else:
            # Pool was newly generated - record actual generation params
            return {
                'status': 'pool_generated',
                'gmm_n_components': GATE3_CONFIG['gmm_n_components'],
                'target_n_accept': cfg.target_n_accept,
                'n_select': cfg.n_select,
                'max_pool_attempts': GATE3_CONFIG['max_pool_attempts'],
                'track_a_reject_ratio': cfg.reject_ratio,
                'final_selection_mode': cfg.selection_mode,
                'dynamics_target_indices': [1, 3],
                'dynamics_target_names': ['x_ddot', 'theta_ddot'],
            }
    
    def _build_gmm_config(self, pool_meta: Optional[Dict]) -> Dict:
        """
        Build gmm_config for manifest.
        
        P0-B FIX: If pool was loaded, GMM fitting was skipped.
        """
        is_pool_loaded = bool(pool_meta and pool_meta.get('pool_source'))
        
        if is_pool_loaded:
            return {
                'status': 'skipped_pool_loaded',
                'note': 'GMM fitting not performed when pool is loaded from source',
            }
        else:
            return {
                'status': 'fitted',
                'requested_n_components': GATE3_CONFIG['gmm_n_components'],
                'effective_n_components': getattr(self.gmm_sampler, '_effective_n_components', GATE3_CONFIG['gmm_n_components']),
                'effective_covariance': getattr(self.gmm_sampler, '_effective_covariance', GATE3_CONFIG['gmm_covariance_type']),
                'reg_covar': 1e-6,
            }
    
    def _save_artifacts(
        self,
        pool: Dict,
        track_a_result: Dict,
        selected: Dict,
        esindy_result: Dict,
        pool_meta: Optional[Dict] = None,  # D1: Pool metadata (if loaded from source)
        bootstrap_seed: Optional[int] = None,  # D1: Derived bootstrap seed
    ):
        """
        Save all artifacts.
        
        D1 Rebaseline additions:
        - selected_pool_indices.npy
        - track_a_pass_mask.npy
        - generated_pool.npz with alignment_cfg
        - manifest with SSOT information
        """
        print("\n[Saving Artifacts]")
        
        cfg = self.config
        
        # 1. z_after.npy
        z_after = esindy_result['z_scores']
        np.save(self.results_dir / 'z_after.npy', z_after)
        print(f"  ✅ Saved: z_after.npy {z_after.shape}")
        
        # 2. inc_prob_after.npy
        inc_prob_after = esindy_result['inclusion_probability']
        np.save(self.results_dir / 'inc_prob_after.npy', inc_prob_after)
        print(f"  ✅ Saved: inc_prob_after.npy {inc_prob_after.shape}")
        
        # 3. selected_trajectories.npz
        np.savez(
            self.results_dir / 'selected_trajectories.npz',
            trajectories=selected['trajectories'],
            dx=selected['dx'],
            params=selected['params'],
            ic=selected['ic'],
            u=selected['u'],
            u_indices=selected['u_indices'],
            errors=selected['errors'],
        )
        print(f"  ✅ Saved: selected_trajectories.npz")
        
        # 3.1 D1 SSOT: selected_pool_indices.npy
        np.save(
            self.results_dir / 'selected_pool_indices.npy',
            selected['original_indices']
        )
        print(f"  ✅ Saved: selected_pool_indices.npy (n={len(selected['original_indices'])})")
        
        # 3.2 D1 SSOT: track_a_pass_mask.npy (indices that passed Track A filter)
        n_pool = len(pool['trajectories'])
        track_a_pass_mask = np.zeros(n_pool, dtype=bool)
        track_a_pass_mask[track_a_result['selected_indices']] = True
        np.save(self.results_dir / 'track_a_pass_mask.npy', track_a_pass_mask)
        np.save(self.results_dir / 'track_a_pass_indices.npy', track_a_result['selected_indices'])
        print(f"  ✅ Saved: track_a_pass_mask.npy, track_a_pass_indices.npy (n_pass={track_a_pass_mask.sum()})")
        
        # 3.5 SSOT: generated_pool.npz (Pool 재사용을 위한 전체 저장)
        # GPT/Claude 합의: 공정 비교를 위해 같은 pool에서 selection만 바꿔야 함
        # D1 Rebaseline: Include alignment_cfg for full SSOT
        
        # Build alignment_cfg for SSOT
        alignment_cfg = {
            'formula_id': 'v2.1_gate1_equiv',
            'formula': 'dx_pred = Theta(x_norm, u_norm) @ coef + dx_mean',
            'dynamics_target_indices': [1, 3],
            'dynamics_target_names': ['x_ddot', 'theta_ddot'],
            'dx_std': self._dx_std.tolist(),
            'dx_std_source': 'norm_stats.json',
            'norm_stats_sha256': compute_file_hash(self.norm_stats_path),
            'teacher_coef_sha256': compute_array_hash(self.teacher_coefficients),
            'teacher_support_sha256': compute_array_hash(self.teacher_support),
            'day3_run_id': cfg.day3_run_id,
            'reject_ratio': cfg.reject_ratio,
        }
        
        # P0-B FIX: Determine if pool was loaded vs newly generated
        is_pool_loaded = bool(pool_meta and pool_meta.get('pool_source'))
        
        if is_pool_loaded:
            # Use original pool's generation metadata (from pool['stats'])
            orig_target = pool['stats'].get('target_n_accept', -1)  # -1 = unknown
            orig_seed = pool['stats'].get('generation_seed', -1)
            orig_variant = pool['stats'].get('generation_variant', 'unknown')
            pool_save_note = f"(loaded from {pool_meta['pool_source']})"
        else:
            # Use current CLI values (this run generated the pool)
            orig_target = cfg.target_n_accept
            orig_seed = cfg.seed
            orig_variant = cfg.variant
            pool_save_note = "(newly generated)"
        
        pool_path_out = self.results_dir / 'generated_pool.npz'
        np.savez(
            pool_path_out,
            trajectories=pool['trajectories'],
            dx=pool['dx'],
            params=pool['params'],
            ic=pool['ic'],
            u=pool['u'],
            u_indices=pool['u_indices'],
            alignment_errors=track_a_result['errors'],
            # D1 SSOT: alignment_cfg as dict
            alignment_cfg=alignment_cfg,
            # P0-B FIX: Use original pool's generation config, not CLI defaults
            generation_seed=orig_seed,
            generation_variant=orig_variant,
            target_n_accept=orig_target,
            # D1: Pool source info (empty if newly generated)
            pool_source=pool_meta.get('pool_source', '') if pool_meta else '',
        )
        
        # P0-A FIX: Compute SHA256 of saved artifact
        pool_sha_out = compute_file_hash(pool_path_out)
        
        # P0 CRITICAL: Preserve original pool SHA256 for loaded pools
        # - For newly generated pool: pool_sha256 = SHA of this file
        # - For loaded pool: pool_sha256 = SHA of ORIGINAL pool (for GEN↔GEN equivalence)
        if pool_meta is None:
            pool_meta = {}
        
        if is_pool_loaded:
            # Pool was loaded - keep original SHA256, record artifact SHA separately
            pool_meta['pool_artifact_sha256'] = pool_sha_out  # SHA of re-saved file
            # pool_meta['pool_sha256'] already contains original pool's SHA (from _load_existing_pool)
            print(f"  ✅ Saved: generated_pool.npz (n={pool['trajectories'].shape[0]}) {pool_save_note}")
            print(f"     original_pool_sha256: {pool_meta.get('pool_sha256', 'unknown')[:16]}...")
            print(f"     artifact_sha256: {pool_sha_out[:16]}...")
        else:
            # Pool was newly generated - this IS the original pool
            pool_meta['pool_sha256'] = pool_sha_out
            pool_meta['pool_artifact_sha256'] = pool_sha_out  # Same for new pools
            print(f"  ✅ Saved: generated_pool.npz (n={pool['trajectories'].shape[0]}) {pool_save_note}")
            print(f"     pool_sha256: {pool_sha_out[:16]}...")
        
        pool_meta['pool_artifact_path'] = str(pool_path_out)
        
        # 4. pool_stats.json (P0-3: detailed relaxation recording)
        pool_stats = {
            'generation': pool['stats'],
            'track_a': track_a_result['stats'],  # Now includes relaxed_flag, pre_relax_pass_rate, etc.
            'selection': selected['stats'],
        }
        with open(self.results_dir / 'pool_stats.json', 'w', encoding='utf-8') as f:
            json.dump(pool_stats, f, indent=2, default=_json_default)
        print(f"  ✅ Saved: pool_stats.json")
        
        # 5. metrics.json
        metrics = {
            'n_original': cfg.n_train,
            'n_augmented': selected['stats']['n_selected'],
            'n_total': cfg.n_train + selected['stats']['n_selected'],
            'pool_acceptance_rate': pool['stats']['acceptance_rate'],
            'track_a': {
                'pass_rate': track_a_result['stats']['n_selected'] / track_a_result['stats']['n_total'],
                'relaxed_flag': track_a_result['stats']['relaxed_flag'],
                'pre_relax_pass_rate': track_a_result['stats']['pre_relax_pass_rate'],
                'effective_reject_ratio': track_a_result['stats']['effective_reject_ratio'],
                'dynamics_target_indices': track_a_result['stats']['dynamics_target_indices'],
            },
            'esindy': {
                'n_support': int(esindy_result['support_mask'].sum()),
                'n_stable_core': int(esindy_result['stable_core_mask'].sum()),
                'n_fragile_pool': int(esindy_result['fragile_pool_mask'].sum()),
            }
        }
        with open(self.results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=_json_default)
        print(f"  ✅ Saved: metrics.json")
        
        # 6. manifest.json (P0-1, P0-2, P0-4 additions)
        day3_dir = self._get_day3_dir()
        teacher_support_sha = compute_file_hash(day3_dir / 'teacher_support.npy')
        
        # P0-4: teacher_coeff_sha256
        teacher_coeff_path = (
            self.project_root / 'results' / cfg.dataset_version / 'gate1' /
            cfg.track / 'esindy' / f"n{cfg.n_train}" / f"seed{cfg.seed}" /
            self.teacher_run_id / 'sindy_coefficients.csv'
        )
        teacher_coeff_sha = compute_file_hash(teacher_coeff_path)
        
        # v2.1 SSOT: code_sha256 for reproducibility
        code_path = Path(__file__).resolve()
        code_sha = compute_file_hash(code_path)
        
        manifest = {
            'phase': 'gate3',
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'dataset_version': cfg.dataset_version,
            'track': cfg.track,
            'method': cfg.method,
            'n_train': cfg.n_train,
            'seed': cfg.seed,
            'variant': cfg.variant,
            'day3_run_id': cfg.day3_run_id,
            'teacher_run_id': self.teacher_run_id,  # P0-4: explicit
            'ssot': self.day3_manifest.get('ssot', {}),
            'hyperparameters': {
                'tau_support': cfg.tau_support,
                'z0': cfg.z0,
                'eps': cfg.eps,
                'bootstrap_B': cfg.bootstrap_B,
                'threshold': cfg.threshold,
            },
            'control_equivalence': get_control_equivalence(cfg.bootstrap_B),
            # P0-B FIX: generation_config depends on whether pool was loaded
            'generation_config': self._build_generation_config(cfg, pool, pool_meta),
            # D1 Rebaseline SSOT
            'd1_rebaseline': {
                'selection_mode': cfg.selection_mode,
                'reject_ratio': cfg.reject_ratio,
                'pool_source': cfg.pool_source if cfg.pool_source else None,
                'pool_sha256': pool_meta.get('pool_sha256', None) if pool_meta else None,  # ORIGINAL pool SHA
                'pool_artifact_sha256': pool_meta.get('pool_artifact_sha256', None) if pool_meta else None,  # Saved artifact SHA
                'pool_loaded': bool(cfg.pool_source),
                'bootstrap_seed': bootstrap_seed,
                'rng_seed_sequence_entropy': self.rng_streams.get('_seed_sequence_entropy'),
            },
            # Track B SSOT v0.2 (GPT P0 fix)
            'track_b_config': {
                'enabled': cfg.selection_mode == 'track_b',
                'version': 'v0.2',
                'alpha': cfg.track_b_alpha,
                'diversity_mode': cfg.track_b_diversity_mode,
                'top_m_ratio': cfg.track_b_top_m_ratio,
                'score_floor': cfg.track_b_score_floor,
                'diversity_space': 'ic_params',  # GPT P0-3 FIX
                'tie_break': 'global_idx_asc',
                'canonicalized': True,
            } if cfg.selection_mode == 'track_b' else None,
            # P0-4: Both SHA256
            'teacher_support_sha256': teacher_support_sha,
            'teacher_coeff_sha256': teacher_coeff_sha,
            # v2.1 SSOT: code hash for full reproducibility
            'code_sha256': code_sha,
            'code_source': str(code_path),
            'preflight_qc': {
                'dx_pipeline': {
                    'method': 'savgol',
                    'window': GATE3_CONFIG['savgol']['window'],
                    'polyorder': GATE3_CONFIG['savgol']['polyorder'],
                    'delta': GATE3_CONFIG['simulation']['dt'],
                    'note': 'Generated trajectories use this pipeline for dx',
                },
                # P0-2: dx_equivalence will be added in future when comparing with dataset
                'dx_source_key': 'train_dx_savgol',
            },
            'data_config': {
                'n_trajectories': cfg.n_train + selected['stats']['n_selected'],
                'n_original': cfg.n_train,
                'n_augmented': selected['stats']['n_selected'],
                'augmentation': 'gmm_generative',
            },
            'track_a_summary': {
                'relaxed_flag': track_a_result['stats']['relaxed_flag'],
                'error_mean': track_a_result['stats']['error_mean'],
                'error_std': track_a_result['stats']['error_std'],
            },
            # P1: dx_std for scale verification
            # v2.1 FIX: Add alignment formula SSOT
            'alignment_error_config': {
                'dynamics_target_indices': [1, 3],
                'dynamics_target_names': ['x_ddot', 'theta_ddot'],
                'dx_std_all': self._dx_std.tolist(),
                'dx_std_dynamics': [float(self._dx_std[1]), float(self._dx_std[3])],
                # v2.1 FIX: Alignment formula SSOT
                'alignment_formula': 'dx_pred = Theta(x_norm, u_norm) @ coef + dx_mean',
                'norm_stats_source': str(self.norm_stats_path),
                'norm_stats_sha256': compute_file_hash(self.norm_stats_path),
                'derivative_key': 'derivative_dx_savgol',
            },
            # P1: Solver config for reproducibility
            'solver_config': {
                'method': GATE3_CONFIG['simulation']['method'],
                'rtol': GATE3_CONFIG['simulation']['rtol'],
                'atol': GATE3_CONFIG['simulation']['atol'],
                'dt': GATE3_CONFIG['simulation']['dt'],
                'T': GATE3_CONFIG['simulation']['T'],
                'duration': GATE3_CONFIG['simulation']['duration'],
            },
            # P1: GMM details (P0-B FIX: skip if pool was loaded)
            'gmm_config': self._build_gmm_config(pool_meta),
        }
        
        with open(self.results_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"  ✅ Saved: manifest.json")
        
        # 7. sindy_coefficients.csv
        coef_mean = esindy_result['coefficients_mean']
        feature_names = esindy_result['feature_names']
        target_names = esindy_result['target_names']
        
        with open(self.results_dir / 'sindy_coefficients.csv', 'w', encoding='utf-8') as f:
            header = 'term_name,' + ','.join(target_names)
            f.write(header + '\n')
            for i, fname in enumerate(feature_names):
                row = ','.join(f'{v:.8f}' for v in coef_mean[i, :])
                f.write(f'{fname},{row}\n')
        print(f"  ✅ Saved: sindy_coefficients.csv")


# ============================================================
# Gate3 Compare Runner (P0-2: Primary endpoint calculation)
# ============================================================

class Gate3CompareRunner:
    """
    Gate3 Compare Runner: Calculate aug_pure = z_after(GEN) - z_after(CTRL250)
    
    P0-1 FIX: Use fragile∩dynamics (n=30) subset with paired bootstrap B=100
    P0-4: CTRL250 manifest equivalence assert
    
    Primary Endpoint (Gate3 v1.3):
    - aug_pure = z_gen - z_ctrl on fragile∩dynamics pairs only
    - CI via paired bootstrap (B=100, resample the n=30 indices)
    - PASS criteria: CI_lower > 0.058 (Gate2 ceiling)
    """
    
    # SSOT: fragile∩dynamics definition
    # dynamics targets: [1, 3] = x_ddot, theta_ddot
    # fragile: support=True AND z < 2.0
    DYNAMICS_TARGET_INDICES = [1, 3]
    GATE2_CEILING = 0.058
    
    def __init__(self, config: Gate3Config):
        self.config = config
        self.run_id = generate_run_id(config.note)
        self.project_root = Path(__file__).resolve().parent.parent
        
        print("=" * 60)
        print(f"  Gate3 Compare Runner")
        print("=" * 60)
        print(f"  Run ID: {self.run_id}")
        print(f"  CTRL250: {config.ctrl250_run_id}")
        print(f"  GEN runs: {config.gen_run_ids}")
        print("=" * 60)
    
    def _load_fragile_pairs(self, day3_dir: Path, z_before: np.ndarray) -> List[Tuple[int, int]]:
        """
        Load or compute fragile∩dynamics pairs.
        
        P0-1 FIX: fragile definition MUST use teacher_support (inc_prob >= tau_support)
        
        Gate2 SSOT Definition:
        - support: inc_prob >= tau_support (0.5)
        - fragile: support AND z < z0 (2.0)
        - fragile∩dynamics: fragile AND target in [1, 3]
        
        Returns:
            List of (feature_idx, target_idx) pairs
        """
        # Try to load pre-computed fragile_pairs from Day3 artifacts (preferred)
        fragile_pairs_path = day3_dir / 'fragile_pairs.json'
        if fragile_pairs_path.exists():
            with open(fragile_pairs_path, 'r') as f:
                data = json.load(f)
                pairs = [tuple(p) for p in data['pairs']]
                # Filter for dynamics targets only
                dynamics_pairs = [(f, t) for f, t in pairs if t in self.DYNAMICS_TARGET_INDICES]
                print(f"  Loaded fragile_pairs from Day3: n={len(pairs)}, dynamics only: n={len(dynamics_pairs)}")
                return dynamics_pairs
        
        # Fallback: compute from Day3 artifacts using SSOT definition
        # MUST use teacher_support (inc_prob >= tau_support)
        print("  ⚠️ fragile_pairs.json not found, computing from Day3 artifacts...")
        
        # Load teacher_support or inc_prob
        teacher_support_path = day3_dir / 'teacher_support.npy'
        inc_prob_path = day3_dir / 'inc_prob_before.npy'
        
        if teacher_support_path.exists():
            teacher_support = np.load(teacher_support_path)
            print(f"  Loaded teacher_support: shape={teacher_support.shape}")
        elif inc_prob_path.exists():
            inc_prob = np.load(inc_prob_path)
            teacher_support = inc_prob >= 0.5  # tau_support = 0.5
            print(f"  Computed teacher_support from inc_prob: shape={teacher_support.shape}")
        else:
            # Last resort: try to load from CTRL250
            print("  ⚠️ No teacher_support found in Day3, checking CTRL250...")
            ctrl_dir = (
                self.project_root / 'results' / self.config.dataset_version / 'phase35' /
                self.config.track / self.config.method / f"n{self.config.n_train}" / 
                f"seed{self.config.seed}" / self.config.ctrl250_run_id
            )
            ctrl_inc_prob_path = ctrl_dir / 'inc_prob_before.npy'
            if ctrl_inc_prob_path.exists():
                inc_prob = np.load(ctrl_inc_prob_path)
                teacher_support = inc_prob >= 0.5
                print(f"  Loaded teacher_support from CTRL250: shape={teacher_support.shape}")
            else:
                raise FileNotFoundError(
                    f"Cannot compute fragile_pairs: no teacher_support or inc_prob found.\n"
                    f"Checked: {teacher_support_path}, {inc_prob_path}, {ctrl_inc_prob_path}"
                )
        
        # Compute fragile∩dynamics pairs using SSOT definition
        # fragile = support (inc_prob >= 0.5) AND z < z0 (2.0)
        Z0 = 2.0  # SSOT
        TAU_SUPPORT = 0.5  # SSOT
        
        fragile_pairs = []
        n_features, n_targets = z_before.shape
        
        n_support = 0
        n_fragile = 0
        
        for t_idx in self.DYNAMICS_TARGET_INDICES:
            for f_idx in range(n_features):
                # Check support: teacher_support[f_idx, t_idx] = True
                is_support = teacher_support[f_idx, t_idx] if teacher_support.ndim == 2 else teacher_support[f_idx]
                
                if is_support:
                    n_support += 1
                    z_val = z_before[f_idx, t_idx]
                    # fragile: support AND z < z0
                    if z_val < Z0:
                        fragile_pairs.append((f_idx, t_idx))
                        n_fragile += 1
        
        print(f"  Computed fragile∩dynamics pairs (SSOT definition):")
        print(f"    support∩dynamics: n={n_support}")
        print(f"    fragile∩dynamics (support AND z<{Z0}): n={n_fragile}")
        
        return fragile_pairs
    
    def _check_gen_gen_equivalence(self, gen_manifests: Dict[str, Dict]) -> Dict:
        """
        Check GEN↔GEN equivalence for D1 Rebaseline fairness.
        
        P0-C: All GEN runs must have matching:
        - code_sha256
        - d1_rebaseline.pool_sha256
        - alignment_error_config.norm_stats_sha256
        - teacher_support_sha256
        - teacher_coeff_sha256
        - hyperparameters (bootstrap_B, threshold, tau_support, z0, eps)
        
        Returns:
            Dict with 'all_equivalent', 'checked_keys', 'mismatches'
        """
        # Keys that MUST match for fair ablation
        required_keys = [
            ('code_sha256', lambda m: m.get('code_sha256')),
            ('pool_sha256', lambda m: m.get('d1_rebaseline', {}).get('pool_sha256')),
            ('norm_stats_sha256', lambda m: m.get('alignment_error_config', {}).get('norm_stats_sha256')),
            ('teacher_support_sha256', lambda m: m.get('teacher_support_sha256')),
            ('teacher_coeff_sha256', lambda m: m.get('teacher_coeff_sha256')),
            ('bootstrap_B', lambda m: m.get('hyperparameters', {}).get('bootstrap_B')),
            ('threshold', lambda m: m.get('hyperparameters', {}).get('threshold')),
            ('tau_support', lambda m: m.get('hyperparameters', {}).get('tau_support')),
            ('z0', lambda m: m.get('hyperparameters', {}).get('z0')),
            ('eps', lambda m: m.get('hyperparameters', {}).get('eps')),
        ]
        
        mismatches = []
        checked_keys = []
        
        for key_name, extractor in required_keys:
            values = {}
            for run_id, manifest in gen_manifests.items():
                values[run_id] = extractor(manifest)
            
            unique_values = set(str(v) for v in values.values())
            checked_keys.append(key_name)
            
            if len(unique_values) > 1:
                mismatches.append({
                    'key': key_name,
                    'values': {k: str(v)[:16] + '...' if v and len(str(v)) > 16 else v 
                              for k, v in values.items()},
                })
        
        return {
            'all_equivalent': len(mismatches) == 0,
            'checked_keys': checked_keys,
            'mismatches': mismatches,
            'n_runs': len(gen_manifests),
        }
    
    def _paired_bootstrap_ci(
        self,
        aug_pure_subset: np.ndarray,
        n_bootstrap: int = 100,
        seed: int = 0,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Compute paired bootstrap CI for aug_pure on fragile∩dynamics subset.
        
        P0-1: Paired bootstrap means we resample the n=30 pairs together.
        P1: Return bootstrap_medians for outlier diagnosis.
        """
        rng = np.random.default_rng(seed)
        n = len(aug_pure_subset)
        
        bootstrap_medians = []
        for _ in range(n_bootstrap):
            # Resample indices (paired)
            indices = rng.choice(n, size=n, replace=True)
            sample = aug_pure_subset[indices]
            bootstrap_medians.append(float(np.median(sample)))
        
        ci_lower = float(np.percentile(bootstrap_medians, (alpha / 2) * 100))
        ci_upper = float(np.percentile(bootstrap_medians, (1 - alpha / 2) * 100))
        
        return {
            'median': float(np.median(aug_pure_subset)),
            'mean': float(np.mean(aug_pure_subset)),
            'std': float(np.std(aug_pure_subset)),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_bootstrap': n_bootstrap,
            'n_pairs': n,
            # P1: Bootstrap distribution for outlier diagnosis
            'bootstrap_medians': bootstrap_medians,
            'bootstrap_p5': float(np.percentile(bootstrap_medians, 5)),
            'bootstrap_p95': float(np.percentile(bootstrap_medians, 95)),
        }
    
    def run(self) -> Dict[str, Any]:
        """Run comparison and calculate aug_pure on fragile∩dynamics subset."""
        cfg = self.config
        
        if not cfg.ctrl250_run_id:
            raise ValueError("ctrl250_run_id required for compare mode")
        if not cfg.gen_run_ids:
            raise ValueError("gen_run_ids required for compare mode")
        
        # 1. Load CTRL250 manifest and z_after
        print("\n[Loading CTRL250 Baseline]")
        ctrl250_manifest = load_ctrl250_manifest(
            cfg.ctrl250_run_id, self.project_root, cfg
        )
        
        ctrl250_dir = (
            self.project_root / 'results' / cfg.dataset_version / 'phase35' /
            cfg.track / cfg.method / f"n{cfg.n_train}" / f"seed{cfg.seed}" / cfg.ctrl250_run_id
        )
        
        z_ctrl = np.load(ctrl250_dir / 'z_after.npy')
        print(f"  z_ctrl shape: {z_ctrl.shape}")
        
        # 2. Load Day3 for fragile_pairs computation
        day3_run_id = ctrl250_manifest.get('day3_run_id', cfg.day3_run_id)
        day3_dir = (
            self.project_root / 'results' / cfg.dataset_version / 'phase35' /
            cfg.track / cfg.method / f"n{cfg.n_train}" / f"seed{cfg.seed}" / day3_run_id
        )
        
        z_before = np.load(day3_dir / 'z_before.npy')
        
        # D1 SSOT: fragile_pairs fail-fast (GPT P0-5)
        fragile_pairs_loaded = False
        fragile_pairs_source_sha = None
        
        # Option 1: Load from specified source
        if cfg.fragile_pairs_source:
            fp_path = Path(cfg.fragile_pairs_source)
            if fp_path.exists():
                with open(fp_path, 'r') as f:
                    fp_data = json.load(f)
                fragile_pairs = [(p[0], p[1]) for p in fp_data.get('pairs', [])]
                fragile_pairs_source_sha = compute_file_hash(fp_path)
                fragile_pairs_loaded = True
                print(f"  ✅ Loaded fragile_pairs from: {cfg.fragile_pairs_source}")
                print(f"     SHA256: {fragile_pairs_source_sha[:16]}...")
            else:
                raise FileNotFoundError(f"fragile_pairs_source not found: {cfg.fragile_pairs_source}")
        
        # Option 2: Check if exists in day3_dir
        elif (day3_dir / 'fragile_pairs.json').exists():
            fp_path = day3_dir / 'fragile_pairs.json'
            with open(fp_path, 'r') as f:
                fp_data = json.load(f)
            fragile_pairs = [(p[0], p[1]) for p in fp_data.get('pairs', [])]
            fragile_pairs_source_sha = compute_file_hash(fp_path)
            fragile_pairs_loaded = True
            print(f"  ✅ Loaded fragile_pairs from: {fp_path}")
            print(f"     SHA256: {fragile_pairs_source_sha[:16]}...")
        
        # Option 3: Compute if allowed (override)
        elif cfg.allow_fragile_compute:
            print("  ⚠️ fragile_pairs.json not found, computing (allow_fragile_compute=True)")
            fragile_pairs = self._load_fragile_pairs(day3_dir, z_before)
        
        # Option 4: Fail-fast (default)
        else:
            raise FileNotFoundError(
                f"D1 SSOT FAIL-FAST: fragile_pairs.json not found and allow_fragile_compute=False.\n"
                f"Checked: {day3_dir / 'fragile_pairs.json'}\n"
                f"Options:\n"
                f"  1. Specify --fragile_pairs_source <path>\n"
                f"  2. Use --allow_fragile_compute to compute from scratch"
            )
        
        if len(fragile_pairs) == 0:
            raise ValueError("No fragile∩dynamics pairs found. Cannot proceed with comparison.")
        
        print(f"  fragile∩dynamics pairs: n={len(fragile_pairs)}")
        
        # P0-3 FIX: Save fragile_pairs to compare results directory for SSOT
        fragile_pairs_data = {
            'pairs': fragile_pairs,
            'n_pairs': len(fragile_pairs),
            'definition': 'support (inc_prob >= 0.5) AND z < 2.0',
            'dynamics_targets': self.DYNAMICS_TARGET_INDICES,
            'source': 'computed from Day3 artifacts',
            'day3_run_id': day3_run_id,
        }
        
        # P0-C FIX: GEN↔GEN equivalence check (D1 Rebaseline critical)
        print("\n[GEN↔GEN Equivalence Check]")
        gen_manifests = {}
        for gen_run_id in cfg.gen_run_ids:
            gen_dir = (
                self.project_root / 'results' / cfg.dataset_version / 'gate3' /
                cfg.track / cfg.method / f"n{cfg.n_train}" / f"seed{cfg.seed}" / gen_run_id
            )
            if gen_dir.exists():
                with open(gen_dir / 'manifest.json', 'r', encoding='utf-8') as f:
                    gen_manifests[gen_run_id] = json.load(f)
        
        # P0-2 FIX v0.3.3: gen_gen_equivalence always returns object (never null)
        # GPT feedback: null breaks downstream parsers/reports
        EQUIVALENCE_CHECK_KEYS = [
            'code_sha256', 'pool_sha256', 'norm_stats_sha256',
            'teacher_support_sha256', 'teacher_coeff_sha256',
            'bootstrap_B', 'threshold', 'tau_support', 'z0', 'eps'
        ]
        
        if len(gen_manifests) == 0:
            # No manifests loaded
            gen_equiv_result = {
                'n_runs': 0,
                'all_equivalent': None,
                'checked_keys': [],
                'mismatched_keys': [],
                'note': 'no_manifests_loaded'
            }
        elif len(gen_manifests) == 1:
            # Single run: trivially equivalent (SSOT: never null)
            gen_equiv_result = {
                'n_runs': 1,
                'all_equivalent': True,
                'checked_keys': EQUIVALENCE_CHECK_KEYS,
                'mismatched_keys': [],
                'note': 'single_run_trivial'
            }
            print(f"  ℹ️ Single GEN run (trivially equivalent)")
        else:
            # Multiple runs: full equivalence check
            gen_equiv_result = self._check_gen_gen_equivalence(gen_manifests)
            if gen_equiv_result['all_equivalent']:
                print(f"  ✅ GEN↔GEN equivalence: all {len(gen_equiv_result['checked_keys'])} keys match")
            else:
                print(f"  ❌ GEN↔GEN equivalence FAILED:")
                for mismatch in gen_equiv_result['mismatches']:
                    print(f"     {mismatch['key']}: {mismatch['values']}")
                if not cfg.allow_fragile_compute:  # Use as a proxy for "strict mode"
                    raise ValueError("D1 Rebaseline requires GEN↔GEN equivalence. Fix mismatches or use --allow_fragile_compute to bypass.")
        
        # 3. Load each GEN run and compare
        results = []
        
        for gen_run_id in cfg.gen_run_ids:
            print(f"\n[Comparing: {gen_run_id}]")
            
            gen_dir = (
                self.project_root / 'results' / cfg.dataset_version / 'gate3' /
                cfg.track / cfg.method / f"n{cfg.n_train}" / f"seed{cfg.seed}" / gen_run_id
            )
            
            if not gen_dir.exists():
                print(f"  ⚠️ GEN run not found: {gen_dir}")
                continue
            
            # Load GEN manifest
            with open(gen_dir / 'manifest.json', 'r', encoding='utf-8') as f:
                gen_manifest = json.load(f)
            
            # P0-4: Assert equivalence
            print("  [Equivalence Check]")
            gen_config_for_assert = {
                'threshold': gen_manifest.get('hyperparameters', {}).get('threshold'),
                'tau_support': gen_manifest.get('hyperparameters', {}).get('tau_support'),
                'z0': gen_manifest.get('hyperparameters', {}).get('z0'),
                'eps': gen_manifest.get('hyperparameters', {}).get('eps'),
                'teacher_support_sha256': gen_manifest.get('teacher_support_sha256'),
            }
            
            # P0-2 FIX: Get all comparison values
            gen_data_config = gen_manifest.get('data_config', {})
            gen_n_total = gen_data_config.get('n_trajectories', 0)
            gen_n_original = gen_data_config.get('n_original', gen_manifest.get('n_train', 0))
            gen_n_augmented = gen_data_config.get('n_augmented', 0)
            gen_bootstrap_B = gen_manifest.get('hyperparameters', {}).get('bootstrap_B', 0)
            
            try:
                equiv_result = assert_ctrl250_equivalence(
                    gen_config_for_assert, ctrl250_manifest, 
                    gen_n_total=gen_n_total,
                    gen_n_original=gen_n_original,
                    gen_n_augmented=gen_n_augmented,
                    gen_bootstrap_B=gen_bootstrap_B,
                    strict=False
                )
                # P0-2 FIX: Store for artifact
                ctrl250_equiv_for_result = {
                    'is_strongly_equivalent': equiv_result['is_strongly_equivalent'],
                    'is_equivalent': equiv_result['is_equivalent'],
                    'n_critical_matches': equiv_result['n_critical_matches'],
                    'n_strong_mismatches': equiv_result['n_strong_mismatches'],
                    'critical_mismatches': equiv_result['critical_mismatches'],
                    'strong_mismatches': equiv_result['strong_mismatches'],
                }
                
                if equiv_result['is_strongly_equivalent']:
                    print(f"    ✅ Strong equivalence: all keys match")
                elif equiv_result['is_equivalent']:
                    print(f"    ✅ Critical keys match ({equiv_result['n_critical_matches']})")
                    if equiv_result['n_strong_mismatches'] > 0:
                        print(f"    ⚠️ Strong key mismatches ({equiv_result['n_strong_mismatches']}):")
                        for m in equiv_result['strong_mismatches']:
                            print(f"       {m['key']}: GEN={m['gate3']} vs CTRL={m['ctrl250']}")
                else:
                    print(f"    ⚠️ {equiv_result['n_critical_mismatches']} critical mismatches:")
                    for m in equiv_result['critical_mismatches']:
                        print(f"       {m['key']}: GEN={m['gate3']} vs CTRL={m['ctrl250']}")
                
                # Always log n_total and bootstrap_B comparison
                print(f"    n_total: GEN={gen_n_total}, CTRL={equiv_result['ctrl_n_total']}")
                print(f"    bootstrap_B: GEN={gen_bootstrap_B}, CTRL={equiv_result['ctrl_bootstrap_B']}")
                
                # P0-1 CRITICAL: bootstrap_B mismatch warning
                if gen_bootstrap_B != equiv_result['ctrl_bootstrap_B']:
                    print(f"    ⚠️⚠️ CRITICAL: bootstrap_B mismatch!")
                    print(f"       z-scores are computed with different B, comparison may be invalid")
                    print(f"       For valid comparison, use --bootstrap_B {equiv_result['ctrl_bootstrap_B']}")
                
            except Exception as e:
                print(f"    ❌ Equivalence check failed: {e}")
                ctrl250_equiv_for_result = {'error': str(e)}
            
            # Load z_after
            z_gen = np.load(gen_dir / 'z_after.npy')
            print(f"  z_gen shape: {z_gen.shape}")
            
            # P0-1 FIX: Calculate aug_pure on fragile∩dynamics subset ONLY
            aug_pure_full = z_gen - z_ctrl
            
            # Extract subset values
            aug_pure_subset = np.array([aug_pure_full[f_idx, t_idx] for f_idx, t_idx in fragile_pairs])
            
            print(f"  aug_pure subset: n={len(aug_pure_subset)}")
            
            # P0-1 FIX: Paired bootstrap CI with B=100
            bootstrap_result = self._paired_bootstrap_ci(
                aug_pure_subset,
                n_bootstrap=100,  # Gate2 standard
                seed=cfg.seed,
            )
            
            # PASS criteria
            ci_lower = bootstrap_result['ci_lower']
            median_aug_pure = bootstrap_result['median']
            
            if ci_lower > self.GATE2_CEILING:
                pass_level = "CEILING_BREAK"
            elif ci_lower > 0:
                pass_level = "STRONG_PASS"
            elif median_aug_pure > 0:
                pass_level = "SOFT_PASS"
            else:
                pass_level = "NULL"
            
            # P1: Compute per-pair and outliers for diagnosis
            per_pair_aug_pure = [(fragile_pairs[i][0], fragile_pairs[i][1], float(aug_pure_subset[i])) 
                                 for i in range(len(fragile_pairs))]
            
            # P1: Top outliers (by absolute value)
            abs_values = np.abs(aug_pure_subset)
            top_k = min(5, len(aug_pure_subset))
            outlier_indices = np.argsort(abs_values)[-top_k:][::-1]
            top_outliers = [(fragile_pairs[i][0], fragile_pairs[i][1], float(aug_pure_subset[i])) 
                           for i in outlier_indices]
            
            result = {
                'gen_run_id': gen_run_id,
                'median_aug_pure': median_aug_pure,
                'mean_aug_pure': bootstrap_result['mean'],
                'std_aug_pure': bootstrap_result['std'],
                'ci_lower': ci_lower,
                'ci_upper': bootstrap_result['ci_upper'],
                'pass_level': pass_level,
                'gate2_ceiling': self.GATE2_CEILING,
                'n_bootstrap': bootstrap_result['n_bootstrap'],
                'n_fragile_dynamics_pairs': len(fragile_pairs),
                'fragile_pairs': fragile_pairs,  # Store for reproducibility
                'subset_spec': {
                    'type': 'fragile_dynamics',
                    'dynamics_targets': self.DYNAMICS_TARGET_INDICES,
                    'n_pairs': len(fragile_pairs),
                },
                # P1: Detailed data for outlier diagnosis
                'per_pair_aug_pure': per_pair_aug_pure,  # [(f_idx, t_idx, value), ...]
                'top_outliers': top_outliers,  # Top 5 by |value|
                'bootstrap_medians': bootstrap_result['bootstrap_medians'],  # 100 values
                'bootstrap_p5': bootstrap_result['bootstrap_p5'],
                'bootstrap_p95': bootstrap_result['bootstrap_p95'],
                # P0-2 FIX: Store equivalence check result
                'ctrl250_equivalence': ctrl250_equiv_for_result,
            }
            results.append(result)
            
            print(f"  aug_pure median: {median_aug_pure:.4f}")
            print(f"  aug_pure CI95: [{ci_lower:.4f}, {bootstrap_result['ci_upper']:.4f}]")
            print(f"  Gate2 ceiling: {self.GATE2_CEILING}")
            print(f"  PASS level: {pass_level}")
            
            # P1: Print top outliers
            if top_outliers:
                print(f"  Top outliers (by |value|):")
                for f_idx, t_idx, val in top_outliers[:3]:
                    print(f"    ({f_idx},{t_idx}): {val:+.4f}")
        
        # 4. Save comparison results
        output_dir = (
            self.project_root / 'results' / cfg.dataset_version / 'gate3' /
            cfg.track / cfg.method / f"n{cfg.n_train}" / f"seed{cfg.seed}" / self.run_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_result = {
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'ctrl250_run_id': cfg.ctrl250_run_id,
            'day3_run_id': day3_run_id,
            'gen_run_ids': cfg.gen_run_ids,
            'endpoint_spec': {
                'type': 'fragile_dynamics_subset',
                'dynamics_target_indices': self.DYNAMICS_TARGET_INDICES,
                'n_fragile_dynamics_pairs': len(fragile_pairs),
                'bootstrap_B': 100,
                'bootstrap_method': 'paired_resample',
                'gate2_ceiling': self.GATE2_CEILING,
            },
            'fragile_pairs': fragile_pairs_data,  # P0-3: SSOT fragile_pairs
            # P0-2 FIX: Store GEN↔GEN equivalence result
            'gen_gen_equivalence': gen_equiv_result,
            'results': results,
        }
        
        with open(output_dir / 'comparison_gen.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2, default=_json_default)
        
        # P0-3 FIX: Save fragile_pairs separately for SSOT
        with open(output_dir / 'fragile_pairs.json', 'w', encoding='utf-8') as f:
            json.dump(fragile_pairs_data, f, indent=2)
        print(f"\n  ✅ Saved: fragile_pairs.json (n={len(fragile_pairs)})")
        
        print("\n" + "=" * 60)
        print("  Comparison Summary (fragile∩dynamics subset)")
        print("=" * 60)
        for r in results:
            print(f"  {r['gen_run_id']}: {r['pass_level']}")
            print(f"    median={r['median_aug_pure']:.4f}, CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
        
        return {
            'run_id': self.run_id,
            'results_dir': str(output_dir),
            'status': 'success',
            'results': results,
        }


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Gate3 Generative Augmentation Runner'
    )
    
    parser.add_argument('--mode', type=str, default='gen_treat',
                        choices=['gen_treat', 'compare_gen'],
                        help='Experiment mode')
    parser.add_argument('--day3_run_id', type=str, required=True,
                        help='Day3 run_id (baseline)')
    parser.add_argument('--dataset_version', type=str, default='cartpole_ood_v1')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--track', type=str, default='standardized')
    parser.add_argument('--method', type=str, default='stable_core')
    parser.add_argument('--variant', type=str, default='IC',
                        choices=['IC', 'ICU'],
                        help='IC: reuse training u, ICU: generate new u')
    parser.add_argument('--n_train', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--note', type=str, default='gate3_d1')
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--bootstrap_B', type=int, default=20,
                        help='Bootstrap iterations (D1=20, D2=100)')
    
    # D2 options
    parser.add_argument('--target_n_accept', type=int, default=200,
                        help='Pool generation target (D1=200, D2=2000+)')
    parser.add_argument('--n_select', type=int, default=200,
                        help='Final selection count (CTRL250=240 for n_total=250)')
    
    # D1 Rebaseline options
    parser.add_argument('--pool_source', type=str, default='',
                        help='Path to existing pool (empty = generate new)')
    parser.add_argument('--selection_mode', type=str, default='random',
                        choices=['random', 'track_a_filtered_random', 'track_b', 'd_optimal'],
                        help='Selection mode: random, track_a_filtered_random, track_b (v0.2), d_optimal (v0.3)')
    parser.add_argument('--reject_ratio', type=float, default=0.10,
                        help='Track A reject ratio (top X%% error rejected)')
    
    # Track B options v0.2 (GPT P0 fix)
    parser.add_argument('--track_b_alpha', type=float, default=0.3,
                        help='Track B: x_penalty weight (0.0=no penalty, 1.0=full penalty)')
    parser.add_argument('--track_b_diversity_mode', type=str, default='top_m_diversity',
                        choices=['score_only', 'top_m_diversity'],
                        help='Track B v0.2: selection mode (score_only=no diversity, top_m_diversity=GPT P0 fix)')
    parser.add_argument('--track_b_top_m_ratio', type=float, default=5.0,
                        help='Track B: M = ratio × n_select for Top-M gate (default 5.0)')
    parser.add_argument('--track_b_score_floor', type=float, default=None,
                        help='Track B: minimum score threshold (None=no floor)')
    
    # Track B v0.3 D-optimal options
    parser.add_argument('--track_b_dopt_lambda', type=float, default=1e-6,
                        help='D-optimal: regularization for logdet (default 1e-6)')
    parser.add_argument('--track_b_dopt_use_teacher_intersection', action='store_true', default=True,
                        help='D-optimal: F = fragile ∩ teacher_active (default True)')
    parser.add_argument('--track_b_dopt_no_teacher_intersection', action='store_false', dest='track_b_dopt_use_teacher_intersection',
                        help='D-optimal: F = fragile only (no teacher intersection)')
    parser.add_argument('--track_b_dopt_pre_gate_mode', type=str, default='score',
                        choices=['score', 'none'],
                        help='D-optimal: pre-gate mode (score=top-M by score, none=all Track A passed)')
    parser.add_argument('--track_b_dopt_gram_energy_mode', type=str, default='raw',
                        choices=['raw', 'unit_trace', 'trace_power'],
                        help='D-optimal v0.3.2: Gram energy mode (raw=original, unit_trace=energy-neutral, trace_power=partial)')
    parser.add_argument('--track_b_dopt_trace_power', type=float, default=1.0,
                        help='D-optimal v0.3.3: trace power p for G_i/(trace(G_i)**p+eps). p=0=raw, p=1=unit_trace, p=0.7=partial')
    
    # Compare mode
    parser.add_argument('--ctrl250_run_id', type=str, default='',
                        help='Control-250 run_id (for compare mode)')
    parser.add_argument('--gen_run_ids', type=str, default='',
                        help='Comma-separated GEN run_ids (for compare mode)')
    parser.add_argument('--fragile_pairs_source', type=str, default='',
                        help='Path to existing fragile_pairs.json (empty = fail-fast)')
    parser.add_argument('--allow_fragile_compute', action='store_true',
                        help='Allow computing fragile_pairs if not exists (override fail-fast)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = Gate3Config(
        mode=args.mode,
        day3_run_id=args.day3_run_id,
        dataset_version=args.dataset_version,
        dataset_path=args.dataset_path,
        track=args.track,
        method=args.method,
        variant=args.variant,
        n_train=args.n_train,
        seed=args.seed,
        note=args.note,
        threshold=args.threshold,
        bootstrap_B=args.bootstrap_B,
        target_n_accept=args.target_n_accept,
        n_select=args.n_select,
        # D1 Rebaseline
        pool_source=args.pool_source,
        selection_mode=args.selection_mode,
        reject_ratio=args.reject_ratio,
        # Track B parameters v0.2
        track_b_alpha=args.track_b_alpha,
        track_b_diversity_mode=args.track_b_diversity_mode,
        track_b_top_m_ratio=args.track_b_top_m_ratio,
        track_b_score_floor=args.track_b_score_floor,
        # Track B v0.3 D-optimal parameters
        track_b_dopt_lambda=args.track_b_dopt_lambda,
        track_b_dopt_use_teacher_intersection=args.track_b_dopt_use_teacher_intersection,
        track_b_dopt_pre_gate_mode=args.track_b_dopt_pre_gate_mode,
        track_b_dopt_gram_energy_mode=args.track_b_dopt_gram_energy_mode,  # v0.3.2
        track_b_dopt_trace_power=args.track_b_dopt_trace_power,  # v0.3.3
        # Compare mode
        ctrl250_run_id=args.ctrl250_run_id,
        gen_run_ids=args.gen_run_ids.split(',') if args.gen_run_ids else [],
        fragile_pairs_source=args.fragile_pairs_source,
        allow_fragile_compute=args.allow_fragile_compute,
    )
    
    if config.mode == 'gen_treat':
        runner = Gate3TreatRunner(config)
    elif config.mode == 'compare_gen':
        runner = Gate3CompareRunner(config)
    else:
        print(f"❌ Unknown mode: {config.mode}")
        return 1
    
    try:
        result = runner.run()
        print("\n" + "=" * 60)
        print("  ✅ Gate3 Treatment Complete")
        print("=" * 60)
        print(f"  Run ID: {result['run_id']}")
        print(f"  Results: {result['results_dir']}")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())