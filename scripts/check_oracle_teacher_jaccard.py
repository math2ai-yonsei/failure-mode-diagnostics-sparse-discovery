"""
Phase 3.5 Check-1: Oracle vs Strong-teacher Jaccard 검증

목적: "Teacher-aligned → Oracle" 논리 전제 검증
- Oracle (n=50)과 Strong-teacher (n=10)의 support 일치도 측정
- Jaccard > 0.8이면 논리 성립

사용법:
    python check_oracle_teacher_jaccard.py --oracle_dir <path> --teacher_dir <path>

입력:
    - Oracle: Gate1 n=50 결과의 inclusion_probability.csv
    - Teacher: Gate1 n=10 결과의 inclusion_probability.csv

출력:
    - Jaccard similarity
    - F1 score
    - Support 비교 시각화
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json


def load_support_from_inclusion_prob(csv_path: Path, threshold: float = 0.5) -> np.ndarray:
    """
    inclusion_probability.csv에서 support 추출
    
    Args:
        csv_path: inclusion_probability.csv 경로
        threshold: support 판정 임계값 (default: 0.5)
    
    Returns:
        support: (n_features, n_targets) boolean array
    """
    df = pd.read_csv(csv_path, index_col=0)
    inclusion_prob = df.values  # (n_features, n_targets)
    support = inclusion_prob > threshold
    return support, df.index.tolist(), df.columns.tolist()


def compute_jaccard(support_a: np.ndarray, support_b: np.ndarray) -> float:
    """
    Jaccard similarity 계산
    
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    intersection = np.sum(support_a & support_b)
    union = np.sum(support_a | support_b)
    
    if union == 0:
        return 1.0  # 둘 다 empty면 identical
    
    return intersection / union


def compute_f1(support_true: np.ndarray, support_pred: np.ndarray) -> dict:
    """
    F1 score 계산 (Oracle을 ground-truth로 가정)
    """
    tp = np.sum(support_true & support_pred)
    fp = np.sum(~support_true & support_pred)
    fn = np.sum(support_true & ~support_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def compare_supports(oracle_support: np.ndarray, teacher_support: np.ndarray,
                     feature_names: list, target_names: list) -> dict:
    """
    Support 상세 비교
    """
    # 전체 Jaccard
    overall_jaccard = compute_jaccard(oracle_support.flatten(), teacher_support.flatten())
    
    # 전체 F1
    overall_f1 = compute_f1(oracle_support.flatten(), teacher_support.flatten())
    
    # Target별 Jaccard
    target_jaccards = {}
    for i, target in enumerate(target_names):
        j = compute_jaccard(oracle_support[:, i], teacher_support[:, i])
        target_jaccards[target] = j
    
    # 차이 분석
    only_oracle = oracle_support & ~teacher_support  # Oracle에만 있음
    only_teacher = ~oracle_support & teacher_support  # Teacher에만 있음
    both = oracle_support & teacher_support  # 둘 다 있음
    
    diff_details = {
        'only_oracle': [],
        'only_teacher': [],
        'both': []
    }
    
    for i, feat in enumerate(feature_names):
        for j, target in enumerate(target_names):
            if only_oracle[i, j]:
                diff_details['only_oracle'].append(f"{feat} → {target}")
            if only_teacher[i, j]:
                diff_details['only_teacher'].append(f"{feat} → {target}")
            if both[i, j]:
                diff_details['both'].append(f"{feat} → {target}")
    
    return {
        'overall_jaccard': overall_jaccard,
        'overall_f1': overall_f1,
        'target_jaccards': target_jaccards,
        'oracle_active': int(np.sum(oracle_support)),
        'teacher_active': int(np.sum(teacher_support)),
        'intersection': int(np.sum(both)),
        'union': int(np.sum(oracle_support | teacher_support)),
        'diff_details': diff_details
    }


def print_report(result: dict, oracle_n: int, teacher_n: int):
    """결과 리포트 출력"""
    print("\n" + "=" * 70)
    print("  Phase 3.5 Check-1: Oracle vs Strong-teacher Support Comparison")
    print("=" * 70)
    
    print(f"\n[Configuration]")
    print(f"  Oracle: n_train = {oracle_n}")
    print(f"  Teacher: n_train = {teacher_n}")
    
    print(f"\n[Overall Metrics]")
    print(f"  Jaccard Similarity: {result['overall_jaccard']:.4f}")
    print(f"  F1 Score: {result['overall_f1']['f1']:.4f}")
    print(f"  Precision: {result['overall_f1']['precision']:.4f}")
    print(f"  Recall: {result['overall_f1']['recall']:.4f}")
    
    print(f"\n[Support Statistics]")
    print(f"  Oracle active terms: {result['oracle_active']}")
    print(f"  Teacher active terms: {result['teacher_active']}")
    print(f"  Intersection: {result['intersection']}")
    print(f"  Union: {result['union']}")
    
    print(f"\n[Per-Target Jaccard]")
    for target, j in result['target_jaccards'].items():
        status = "✅" if j >= 0.8 else "⚠️"
        print(f"  {target}: {j:.4f} {status}")
    
    print(f"\n[Difference Analysis]")
    print(f"  Only in Oracle ({len(result['diff_details']['only_oracle'])} terms):")
    for term in result['diff_details']['only_oracle'][:5]:
        print(f"    - {term}")
    if len(result['diff_details']['only_oracle']) > 5:
        print(f"    ... and {len(result['diff_details']['only_oracle']) - 5} more")
    
    print(f"  Only in Teacher ({len(result['diff_details']['only_teacher'])} terms):")
    for term in result['diff_details']['only_teacher'][:5]:
        print(f"    - {term}")
    if len(result['diff_details']['only_teacher']) > 5:
        print(f"    ... and {len(result['diff_details']['only_teacher']) - 5} more")
    
    print(f"\n[Check-1 Result]")
    if result['overall_jaccard'] >= 0.8:
        print(f"  ✅ PASS: Jaccard = {result['overall_jaccard']:.4f} >= 0.8")
        print(f"  → 'Teacher-aligned → Oracle' 논리 성립")
        strategy = "maintain"
    elif result['overall_jaccard'] >= 0.6:
        print(f"  ⚠️ MARGINAL: Jaccard = {result['overall_jaccard']:.4f} (0.6-0.8)")
        print(f"  → 논리 약화, 포지셔닝 조정 권장")
        strategy = "adjust"
    else:
        print(f"  ❌ FAIL: Jaccard = {result['overall_jaccard']:.4f} < 0.6")
        print(f"  → 'Teacher = consistency prior' 포지셔닝으로 변경 필요")
        strategy = "reposition"
    
    print("\n" + "=" * 70)
    
    return strategy


def main():
    parser = argparse.ArgumentParser(description='Check-1: Oracle vs Teacher Jaccard')
    parser.add_argument('--oracle_dir', type=Path, required=True,
                        help='Gate1 n=50 결과 디렉토리')
    parser.add_argument('--teacher_dir', type=Path, required=True,
                        help='Gate1 n=10 결과 디렉토리')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Inclusion probability threshold (default: 0.5)')
    parser.add_argument('--output', type=Path, default=None,
                        help='결과 저장 경로 (JSON)')
    
    args = parser.parse_args()
    
    # Load supports
    oracle_csv = args.oracle_dir / 'inclusion_probability.csv'
    teacher_csv = args.teacher_dir / 'inclusion_probability.csv'
    
    if not oracle_csv.exists():
        print(f"❌ Error: Oracle file not found: {oracle_csv}")
        return
    if not teacher_csv.exists():
        print(f"❌ Error: Teacher file not found: {teacher_csv}")
        return
    
    oracle_support, feature_names, target_names = load_support_from_inclusion_prob(
        oracle_csv, args.threshold
    )
    teacher_support, _, _ = load_support_from_inclusion_prob(
        teacher_csv, args.threshold
    )
    
    # Load n_train from manifest
    oracle_manifest = args.oracle_dir / 'manifest.json'
    teacher_manifest = args.teacher_dir / 'manifest.json'
    
    oracle_n = 50
    teacher_n = 10
    
    if oracle_manifest.exists():
        with open(oracle_manifest) as f:
            oracle_n = json.load(f).get('config', {}).get('n_train', 50)
    if teacher_manifest.exists():
        with open(teacher_manifest) as f:
            teacher_n = json.load(f).get('config', {}).get('n_train', 10)
    
    # Compare
    result = compare_supports(oracle_support, teacher_support, feature_names, target_names)
    
    # Print report
    strategy = print_report(result, oracle_n, teacher_n)
    
    # Save result
    if args.output:
        output_data = {
            'oracle_n': oracle_n,
            'teacher_n': teacher_n,
            'threshold': args.threshold,
            'result': result,
            'strategy': strategy
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  ✅ Saved: {args.output}")


if __name__ == '__main__':
    main()