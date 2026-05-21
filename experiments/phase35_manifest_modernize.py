"""
Phase 3.5 Manifest Modernize Helper

목적: Day3/Day4 runner의 manifest를 modern schema로 업그레이드
- compare에서 "legacy assumed" 제거
- STRICT_PASS 달성을 위한 필수 필드 추가

필수 필드 4개:
1. control_equivalence: SSOT dict 전체
2. teacher_support_sha256: teacher_support.npy 파일 해시
3. dx_key_used: 사용된 dx 키 명시
4. code_hash + code_snapshot/: 재현성을 위한 코드 스냅샷

Author: Claude (Phase 3.5 Option B)
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union

import numpy as np


# ============================================================
# CONTROL_EQUIVALENCE SSOT (단일 소스)
# ============================================================

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


# ============================================================
# Helper Functions
# ============================================================

def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """파일의 SHA256 해시 계산"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def create_code_snapshot(
    results_dir: Path, 
    source_files: List[Path]
) -> Dict[str, str]:
    """
    코드 스냅샷 생성 및 해시 계산
    
    Args:
        results_dir: 결과 저장 디렉토리
        source_files: 스냅샷할 소스 파일 리스트
    
    Returns:
        {filename: sha256_hash} 딕셔너리
    """
    snapshot_dir = results_dir / 'code_snapshot'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    code_hash = {}
    for src_file in source_files:
        if src_file.exists():
            dst_file = snapshot_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            code_hash[src_file.name] = compute_file_sha256(dst_file)
    
    # code_hash.json 저장
    hash_path = results_dir / 'code_hash.json'
    with open(hash_path, 'w', encoding='utf-8') as f:
        json.dump(code_hash, f, indent=2)
    
    return code_hash


def get_control_equivalence(
    bootstrap_B: int = None,
    threshold: float = None,
    tau_support: float = None,
    z0: float = None
) -> Dict[str, Any]:
    """
    CONTROL_EQUIVALENCE SSOT dict 반환 (복사본)
    
    Args:
        bootstrap_B: Override bootstrap_B (default: 20)
        threshold: Override threshold (default: 0.05)
        tau_support: Override tau_support (default: 0.5)
        z0: Override z0 (default: 2.0)
    
    Returns:
        control_equivalence dict with applied overrides
    """
    ce = CONTROL_EQUIVALENCE.copy()
    
    # Override if provided
    if bootstrap_B is not None:
        ce['bootstrap_B'] = bootstrap_B
    if threshold is not None:
        ce['threshold'] = threshold
    if tau_support is not None:
        ce['tau_support'] = tau_support
    if z0 is not None:
        ce['z0'] = z0
    
    return ce


if __name__ == '__main__':
    print("Phase 3.5 Manifest Modernize Helper")
    print("=" * 50)
    print("\nCONTROL_EQUIVALENCE SSOT:")
    for k, v in CONTROL_EQUIVALENCE.items():
        print(f"  {k}: {v}")