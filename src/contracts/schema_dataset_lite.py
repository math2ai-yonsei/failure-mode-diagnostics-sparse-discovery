"""
src/contracts/schema_dataset_lite.py

Preflight Guard: dataset.npz 최소 무결성 검증
- 실험 시작 전 필수 실행
- 실패 시 즉시 중단 (Gate0-1 시간 낭비 방지)

Gate: 0-1
Version: v3.2 (Lean Mode)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any


# ============================================================
# 필수 키 정의 (Cart-Pole 4D 기준)
# ============================================================

REQUIRED_STATE_KEYS = ['train_x', 'val_x', 'test_x']
REQUIRED_INPUT_KEYS = ['train_u', 'val_u', 'test_u']
REQUIRED_DERIV_KEYS = ['train_dx', 'val_dx', 'test_dx']
REQUIRED_PARAM_KEYS = ['train_params', 'val_params', 'test_params']
REQUIRED_COND_KEYS = ['train_cond_id', 'val_cond_id', 'test_cond_id']
REQUIRED_TIME_KEYS = ['t', 'dt']

ALL_REQUIRED_KEYS = (
    REQUIRED_STATE_KEYS + 
    REQUIRED_INPUT_KEYS + 
    REQUIRED_DERIV_KEYS + 
    REQUIRED_PARAM_KEYS + 
    REQUIRED_COND_KEYS + 
    REQUIRED_TIME_KEYS
)

# Cart-Pole 상태 정의
STATE_DIM = 4  # [x, x_dot, theta, theta_dot]
INPUT_DIM = 1  # [force]


# ============================================================
# Lite 검증 함수
# ============================================================

def validate_dataset_lite(npz_path: Path) -> bool:
    """
    dataset.npz Lite 검증 (Preflight Guard)
    
    검증 항목:
    1. 파일 존재
    2. 필수 키 존재
    3. *_x, *_dx, *_u shape가 (N, T, D)인지
    4. train/val/test 간 T, D 일치
    5. dt > 0, t.shape == (T,)
    6. train_cond_id 길이 == N
    7. NaN/Inf 체크
    
    실패 시 ValueError 발생 (실험 즉시 중단)
    
    Args:
        npz_path: dataset.npz 경로
    
    Returns:
        bool: True (성공 시)
    
    Raises:
        FileNotFoundError: 파일이 없을 때
        ValueError: 검증 실패 시
    
    Example:
        >>> from src.contracts.schema_dataset_lite import validate_dataset_lite
        >>> validate_dataset_lite(Path("data/cartpole/cartpole_ood_v1/dataset.npz"))
        ✅ Dataset Preflight 통과: dataset.npz
        True
    """
    npz_path = Path(npz_path)
    errors: List[str] = []
    
    # --------------------------------------------------------
    # 1. 파일 존재 확인
    # --------------------------------------------------------
    if not npz_path.exists():
        raise FileNotFoundError(f"❌ Dataset 파일 없음: {npz_path}")
    
    # --------------------------------------------------------
    # 2. NPZ 로드
    # --------------------------------------------------------
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"❌ NPZ 로드 실패: {e}")
    
    # --------------------------------------------------------
    # 3. 필수 키 존재 확인
    # --------------------------------------------------------
    for key in ALL_REQUIRED_KEYS:
        if key not in data:
            errors.append(f"필수 키 누락: {key}")
    
    if errors:
        raise ValueError(f"❌ Dataset 검증 실패:\n  " + "\n  ".join(errors))
    
    # --------------------------------------------------------
    # 4. Shape 검증 (3D 배열)
    # --------------------------------------------------------
    # train_x에서 기준 shape 추출
    train_x = data['train_x']
    if train_x.ndim != 3:
        errors.append(f"train_x: ndim={train_x.ndim}, 기대값=3")
    else:
        N_train, T, D_state = train_x.shape
        
        if D_state != STATE_DIM:
            errors.append(f"train_x: state_dim={D_state}, 기대값={STATE_DIM}")
    
    # 모든 state 키 검증
    for key in REQUIRED_STATE_KEYS:
        arr = data[key]
        if arr.ndim != 3:
            errors.append(f"{key}: ndim={arr.ndim}, 기대값=3")
        elif arr.shape[2] != STATE_DIM:
            errors.append(f"{key}: state_dim={arr.shape[2]}, 기대값={STATE_DIM}")
    
    # 모든 derivative 키 검증
    for key in REQUIRED_DERIV_KEYS:
        arr = data[key]
        if arr.ndim != 3:
            errors.append(f"{key}: ndim={arr.ndim}, 기대값=3")
        elif arr.shape[2] != STATE_DIM:
            errors.append(f"{key}: state_dim={arr.shape[2]}, 기대값={STATE_DIM}")
    
    # 모든 input 키 검증
    for key in REQUIRED_INPUT_KEYS:
        arr = data[key]
        if arr.ndim != 3:
            errors.append(f"{key}: ndim={arr.ndim}, 기대값=3")
        elif arr.shape[2] != INPUT_DIM:
            errors.append(f"{key}: input_dim={arr.shape[2]}, 기대값={INPUT_DIM}")
    
    if errors:
        raise ValueError(f"❌ Dataset 검증 실패:\n  " + "\n  ".join(errors))
    
    # --------------------------------------------------------
    # 5. T 일관성 검증 (모든 split에서 T 동일)
    # --------------------------------------------------------
    T = data['train_x'].shape[1]
    
    for split in ['train', 'val', 'test']:
        for suffix in ['_x', '_u', '_dx']:
            key = f"{split}{suffix}"
            if data[key].shape[1] != T:
                errors.append(f"{key}: T={data[key].shape[1]}, 기대값={T}")
    
    # --------------------------------------------------------
    # 6. 시간 축 검증
    # --------------------------------------------------------
    t = data['t']
    dt = float(data['dt'])
    
    if t.ndim != 1:
        errors.append(f"t: ndim={t.ndim}, 기대값=1")
    elif t.shape[0] != T:
        errors.append(f"t: length={t.shape[0]}, 기대값={T}")
    
    if dt <= 0:
        errors.append(f"dt: {dt} <= 0")
    
    # t와 dt 일관성 (허용 오차 내)
    if t.ndim == 1 and len(t) > 1:
        t_diff = np.diff(t)
        if not np.allclose(t_diff, dt, atol=1e-6):
            errors.append(f"t 간격과 dt 불일치: mean(diff)={t_diff.mean():.6f}, dt={dt:.6f}")
    
    # --------------------------------------------------------
    # 7. cond_id 길이 검증
    # --------------------------------------------------------
    for split in ['train', 'val', 'test']:
        x_key = f"{split}_x"
        cond_key = f"{split}_cond_id"
        
        N = data[x_key].shape[0]
        cond_len = data[cond_key].shape[0]
        
        if cond_len != N:
            errors.append(f"{cond_key}: length={cond_len}, 기대값={N}")
    
    # --------------------------------------------------------
    # 8. NaN/Inf 체크
    # --------------------------------------------------------
    nan_inf_keys = REQUIRED_STATE_KEYS + REQUIRED_DERIV_KEYS + REQUIRED_INPUT_KEYS
    
    for key in nan_inf_keys:
        arr = data[key]
        if np.isnan(arr).any():
            errors.append(f"{key}: NaN 포함")
        if np.isinf(arr).any():
            errors.append(f"{key}: Inf 포함")
    
    # --------------------------------------------------------
    # 9. 최종 결과
    # --------------------------------------------------------
    if errors:
        raise ValueError(f"❌ Dataset 검증 실패:\n  " + "\n  ".join(errors))
    
    print(f"✅ Dataset Preflight 통과: {npz_path.name}")
    return True


def get_dataset_info(npz_path: Path) -> Dict[str, Any]:
    """
    Dataset 정보 요약 반환
    
    Args:
        npz_path: dataset.npz 경로
    
    Returns:
        dict: 데이터셋 정보
    """
    data = np.load(npz_path, allow_pickle=True)
    
    info = {
        'path': str(npz_path),
        'n_train': data['train_x'].shape[0],
        'n_val': data['val_x'].shape[0],
        'n_test': data['test_x'].shape[0],
        'T': data['train_x'].shape[1],
        'state_dim': data['train_x'].shape[2],
        'input_dim': data['train_u'].shape[2],
        'dt': float(data['dt']),
        'duration': float(data['t'][-1] - data['t'][0]),
    }
    
    return info


# ============================================================
# 테스트/검증
# ============================================================

if __name__ == "__main__":
    import tempfile
    import os
    
    print("=" * 60)
    print("schema_dataset_lite.py 검증")
    print("=" * 60)
    
    # 테스트용 유효한 dataset 생성
    print("\n[유효한 dataset 생성]")
    
    N_train, N_val, N_test = 50, 15, 25
    T = 101
    dt = 0.02
    
    valid_data = {
        # State
        'train_x': np.random.randn(N_train, T, 4).astype(np.float64),
        'val_x': np.random.randn(N_val, T, 4).astype(np.float64),
        'test_x': np.random.randn(N_test, T, 4).astype(np.float64),
        
        # Input
        'train_u': np.random.randn(N_train, T, 1).astype(np.float64),
        'val_u': np.random.randn(N_val, T, 1).astype(np.float64),
        'test_u': np.random.randn(N_test, T, 1).astype(np.float64),
        
        # Derivative
        'train_dx': np.random.randn(N_train, T, 4).astype(np.float64),
        'val_dx': np.random.randn(N_val, T, 4).astype(np.float64),
        'test_dx': np.random.randn(N_test, T, 4).astype(np.float64),
        
        # Params
        'train_params': np.random.randn(N_train, 4).astype(np.float64),
        'val_params': np.random.randn(N_val, 4).astype(np.float64),
        'test_params': np.random.randn(N_test, 4).astype(np.float64),
        
        # Cond ID
        'train_cond_id': np.arange(N_train, dtype=np.int64),
        'val_cond_id': np.arange(N_val, dtype=np.int64),
        'test_cond_id': np.arange(N_test, dtype=np.int64),
        
        # Time
        't': np.linspace(0, (T-1)*dt, T).astype(np.float64),
        'dt': dt,
    }
    
    # 임시 파일에 저장 및 테스트
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_path = Path(tmpdir) / "valid_dataset.npz"
        np.savez(valid_path, **valid_data)
        
        # 1. 유효한 dataset 검증
        print("\n[유효한 dataset 검증]")
        try:
            result = validate_dataset_lite(valid_path)
            print(f"  결과: {result}")
        except Exception as e:
            print(f"  ❌ 예상치 못한 오류: {e}")
        
        # 2. Dataset 정보 출력
        print("\n[Dataset 정보]")
        info = get_dataset_info(valid_path)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 3. 잘못된 dataset 테스트 (키 누락)
        print("\n[키 누락 dataset 검증]")
        invalid_data = {k: v for k, v in valid_data.items() if k != 'train_dx'}
        invalid_path = Path(tmpdir) / "invalid_dataset.npz"
        np.savez(invalid_path, **invalid_data)
        
        try:
            validate_dataset_lite(invalid_path)
            print("  ❌ 오류가 발생해야 하는데 통과함")
        except ValueError as e:
            print(f"  ✅ 예상대로 오류 발생: {str(e)[:50]}...")
        
        # 4. NaN 포함 dataset 테스트
        print("\n[NaN 포함 dataset 검증]")
        nan_data = valid_data.copy()
        nan_data['train_x'] = nan_data['train_x'].copy()
        nan_data['train_x'][0, 0, 0] = np.nan
        nan_path = Path(tmpdir) / "nan_dataset.npz"
        np.savez(nan_path, **nan_data)
        
        try:
            validate_dataset_lite(nan_path)
            print("  ❌ 오류가 발생해야 하는데 통과함")
        except ValueError as e:
            print(f"  ✅ 예상대로 오류 발생: {str(e)[:50]}...")
    
    print("\n" + "=" * 60)
    print("✅ schema_dataset_lite.py 검증 완료")
    print("=" * 60)