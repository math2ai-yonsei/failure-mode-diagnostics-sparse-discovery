#!/usr/bin/env python
"""
Gate2 설치 검증 스크립트

로컬 환경에서 Gate2 모듈이 올바르게 설치되었는지 확인합니다.

Usage:
    python scripts/verify_gate2_installation.py
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (기존 Gate0/Gate1과 동일한 방식)
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("  Gate2 설치 검증")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. 필수 모듈 import 테스트
    print("\n[1/5] 필수 모듈 Import 테스트")
    
    modules_to_test = [
        ("src.contracts.paths", "Paths SSOT"),
        ("src.contracts.plot_style", "Plot Style SSOT"),
        ("src.contracts.schema_dataset_lite", "Schema Dataset Lite"),
        ("src.augmentation", "Augmentation 패키지"),
        ("src.augmentation.base", "BaseAugmentor"),
        ("src.augmentation.physics_augmentor", "PhysicsAugmentor"),
        ("src.sindy.library", "SINDy Library"),
        ("src.sindy.optimizer", "STLSQ Optimizer"),
        ("src.sindy.esindy", "E-SINDy"),
        ("src.data.normalization", "Normalization"),
        ("src.experiments.gate1_esindy_runner", "Gate1 Runner"),
        ("src.experiments.gate2_aug_runner", "Gate2 Runner"),
    ]
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {description}")
        except ImportError as e:
            errors.append(f"{description}: {e}")
            print(f"  ❌ {description}: {e}")
    
    # 2. 핵심 클래스 존재 확인
    print("\n[2/5] 핵심 클래스 존재 확인")
    
    try:
        from src.augmentation import PhysicsAugmentor, BaseAugmentor, AugmentationResult
        from src.augmentation.physics_augmentor import PhysicsAugmentorConfig
        from src.augmentation.base import get_train_subset_idx
        print("  ✅ Augmentation 클래스들")
    except ImportError as e:
        errors.append(f"Augmentation 클래스: {e}")
        print(f"  ❌ Augmentation 클래스: {e}")
    
    try:
        from src.experiments.gate2_aug_runner import Gate2AugRunner, Gate2Config
        print("  ✅ Gate2 Runner 클래스들")
    except ImportError as e:
        errors.append(f"Gate2 Runner 클래스: {e}")
        print(f"  ❌ Gate2 Runner 클래스: {e}")
    
    # 3. 설정 파일 존재 확인
    print("\n[3/5] 설정 파일 존재 확인")
    
    config_files = [
        "configs/experiments/gate2_cartpole.yaml",
    ]
    
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"  ✅ {config_file}")
        else:
            warnings.append(f"Config 없음: {config_file}")
            print(f"  ⚠️ {config_file} (없음)")
    
    # 4. 데이터셋 존재 확인
    print("\n[4/5] 데이터셋 존재 확인")
    
    try:
        from src.contracts import paths
        dataset_path = paths.get_dataset_path('cartpole_ood_v1', 'cartpole')
        norm_stats_path = paths.get_norm_stats_path('cartpole_ood_v1', 'cartpole')
        
        if dataset_path.exists():
            print(f"  ✅ dataset.npz")
        else:
            warnings.append(f"Dataset 없음: {dataset_path}")
            print(f"  ⚠️ dataset.npz (없음)")
        
        if norm_stats_path.exists():
            print(f"  ✅ norm_stats.json")
        else:
            warnings.append(f"Norm stats 없음: {norm_stats_path}")
            print(f"  ⚠️ norm_stats.json (없음)")
    except Exception as e:
        errors.append(f"데이터셋 확인 실패: {e}")
        print(f"  ❌ 데이터셋 확인 실패: {e}")
    
    # 5. 간단한 기능 테스트
    print("\n[5/5] 기능 테스트")
    
    try:
        import numpy as np
        from src.augmentation.physics_augmentor import PhysicsAugmentorConfig, PhysicsAugmentor
        from src.augmentation.base import get_train_subset_idx
        
        # Train subset idx 재현성 테스트
        idx1 = get_train_subset_idx(50, 10, seed=42)
        idx2 = get_train_subset_idx(50, 10, seed=42)
        assert np.array_equal(idx1, idx2), "Reproducibility failed!"
        print("  ✅ Train subset idx 재현성")
        
        # PhysicsAugmentor 생성 테스트
        config = PhysicsAugmentorConfig(seed=42, dt=0.02, T=101)
        augmentor = PhysicsAugmentor(config)
        print("  ✅ PhysicsAugmentor 생성")
        
    except Exception as e:
        errors.append(f"기능 테스트 실패: {e}")
        print(f"  ❌ 기능 테스트 실패: {e}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    
    if errors:
        print("  ❌ 검증 실패")
        print("\n  [오류 목록]")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)
    elif warnings:
        print("  ⚠️ 검증 완료 (경고 있음)")
        print("\n  [경고 목록]")
        for warn in warnings:
            print(f"    - {warn}")
        print("\n  Gate2 모듈은 설치되었으나, 일부 파일이 누락되었습니다.")
        print("  실험 실행 전에 누락된 파일을 확인하세요.")
    else:
        print("  ✅ 검증 완료")
        print("\n  Gate2 모듈이 올바르게 설치되었습니다.")
        print("  실험을 시작할 수 있습니다.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()