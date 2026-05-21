"""
QC-2: Teacher-only 항들의 z-score/안정성 분석

목적: Teacher에만 있는 항들이 정말 "작고 흔들리는" 노이즈인지 검증

입력:
    - Oracle: sindy_coefficients.csv, coefficient_std.csv, inclusion_probability.csv
    - Teacher: sindy_coefficients.csv, coefficient_std.csv, inclusion_probability.csv

출력:
    - Teacher-only 항들의 z-score 분포
    - Oracle 항들 vs Teacher-only 항들 비교
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json


def load_esindy_results(result_dir: Path) -> dict:
    """E-SINDy 결과 로드"""
    result_dir = Path(result_dir)
    
    coef_mean = pd.read_csv(result_dir / 'sindy_coefficients.csv', index_col=0)
    coef_std = pd.read_csv(result_dir / 'coefficient_std.csv', index_col=0)
    incl_prob = pd.read_csv(result_dir / 'inclusion_probability.csv', index_col=0)
    
    return {
        'coef_mean': coef_mean,
        'coef_std': coef_std,
        'incl_prob': incl_prob,
        'feature_names': coef_mean.index.tolist(),
        'target_names': coef_mean.columns.tolist()
    }


def compute_z_scores(coef_mean: pd.DataFrame, coef_std: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """z-score 계산: |mean| / (std + eps)"""
    return np.abs(coef_mean) / (coef_std + eps)


def compute_sign_consistency(coef_mean: pd.DataFrame, coef_std: pd.DataFrame) -> pd.DataFrame:
    """부호 일관성 추정 (mean이 std보다 크면 부호가 일관적)"""
    # 간단한 휴리스틱: |mean| > 2*std 이면 부호 일관
    return (np.abs(coef_mean) > 2 * coef_std).astype(float)


def get_support(incl_prob: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """inclusion probability 기반 support 추출"""
    return incl_prob.values > threshold


def analyze_term_groups(oracle_data: dict, teacher_data: dict, incl_threshold: float = 0.5):
    """Oracle vs Teacher 항 그룹 분석"""
    
    # Support 추출
    oracle_support = get_support(oracle_data['incl_prob'], incl_threshold)
    teacher_support = get_support(teacher_data['incl_prob'], incl_threshold)
    
    # z-score 계산
    teacher_z = compute_z_scores(teacher_data['coef_mean'], teacher_data['coef_std'])
    oracle_z = compute_z_scores(oracle_data['coef_mean'], oracle_data['coef_std'])
    
    # 부호 일관성
    teacher_sign = compute_sign_consistency(teacher_data['coef_mean'], teacher_data['coef_std'])
    
    feature_names = teacher_data['feature_names']
    target_names = teacher_data['target_names']
    
    # 항 분류
    both_mask = oracle_support & teacher_support  # 둘 다 있음
    teacher_only_mask = ~oracle_support & teacher_support  # Teacher에만 있음
    oracle_only_mask = oracle_support & ~teacher_support  # Oracle에만 있음
    
    results = {
        'both': [],
        'teacher_only': [],
        'oracle_only': []
    }
    
    for i, feat in enumerate(feature_names):
        for j, target in enumerate(target_names):
            term_info = {
                'feature': feat,
                'target': target,
                'teacher_mean': float(teacher_data['coef_mean'].iloc[i, j]),
                'teacher_std': float(teacher_data['coef_std'].iloc[i, j]),
                'teacher_incl_prob': float(teacher_data['incl_prob'].iloc[i, j]),
                'teacher_z': float(teacher_z.iloc[i, j]),
                'teacher_sign_consistent': bool(teacher_sign.iloc[i, j] > 0),
            }
            
            if both_mask[i, j]:
                term_info['oracle_mean'] = float(oracle_data['coef_mean'].iloc[i, j])
                term_info['oracle_z'] = float(oracle_z.iloc[i, j])
                results['both'].append(term_info)
            elif teacher_only_mask[i, j]:
                results['teacher_only'].append(term_info)
            elif oracle_only_mask[i, j]:
                term_info['oracle_mean'] = float(oracle_data['coef_mean'].iloc[i, j])
                term_info['oracle_z'] = float(oracle_z.iloc[i, j])
                results['oracle_only'].append(term_info)
    
    return results


def print_analysis_report(results: dict, incl_threshold: float):
    """분석 리포트 출력"""
    print("\n" + "=" * 80)
    print("  QC-2: Teacher-only 항 z-score 분석")
    print("=" * 80)
    
    print(f"\n[Configuration]")
    print(f"  Inclusion probability threshold: {incl_threshold}")
    
    # 통계 요약
    both = results['both']
    teacher_only = results['teacher_only']
    oracle_only = results['oracle_only']
    
    print(f"\n[Support Statistics]")
    print(f"  Both (Oracle ∩ Teacher): {len(both)} terms")
    print(f"  Teacher-only: {len(teacher_only)} terms")
    print(f"  Oracle-only: {len(oracle_only)} terms")
    
    # Teacher-only 항들의 z-score 분포
    if teacher_only:
        z_scores = [t['teacher_z'] for t in teacher_only]
        sign_consistent = [t['teacher_sign_consistent'] for t in teacher_only]
        abs_means = [abs(t['teacher_mean']) for t in teacher_only]
        
        print(f"\n[Teacher-only Terms z-score Distribution]")
        print(f"  Count: {len(z_scores)}")
        print(f"  Mean z-score: {np.mean(z_scores):.4f}")
        print(f"  Median z-score: {np.median(z_scores):.4f}")
        print(f"  Max z-score: {np.max(z_scores):.4f}")
        print(f"  Min z-score: {np.min(z_scores):.4f}")
        print(f"  Std z-score: {np.std(z_scores):.4f}")
        print(f"  Sign consistent: {sum(sign_consistent)} / {len(sign_consistent)} ({100*sum(sign_consistent)/len(sign_consistent):.1f}%)")
        
        # z-score 분포 (히스토그램 텍스트)
        print(f"\n  z-score distribution:")
        bins = [0, 1, 2, 3, 5, 10, float('inf')]
        bin_labels = ['0-1', '1-2', '2-3', '3-5', '5-10', '10+']
        for k in range(len(bins)-1):
            count = sum(1 for z in z_scores if bins[k] <= z < bins[k+1])
            bar = '█' * count
            print(f"    {bin_labels[k]:>5}: {count:>2} {bar}")
        
        # 개별 항 상세 (z-score 순으로 정렬)
        print(f"\n[Teacher-only Terms Detail (sorted by z-score)]")
        teacher_only_sorted = sorted(teacher_only, key=lambda x: x['teacher_z'], reverse=True)
        
        print(f"  {'Feature':<20} {'Target':<12} {'Mean':>12} {'Std':>12} {'z-score':>10} {'Sign':>6}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
        
        for t in teacher_only_sorted:
            sign = '✓' if t['teacher_sign_consistent'] else '✗'
            print(f"  {t['feature']:<20} {t['target']:<12} {t['teacher_mean']:>12.6f} {t['teacher_std']:>12.6f} {t['teacher_z']:>10.2f} {sign:>6}")
    
    # Both 항들의 z-score (비교용)
    if both:
        z_scores_both = [t['teacher_z'] for t in both]
        print(f"\n[Both Terms (Oracle ∩ Teacher) z-score for comparison]")
        print(f"  Mean z-score: {np.mean(z_scores_both):.4f}")
        print(f"  Median z-score: {np.median(z_scores_both):.4f}")
        print(f"  Min z-score: {np.min(z_scores_both):.4f}")
    
    # Teacher-core 추천
    if teacher_only:
        z_threshold_candidates = [2.0, 3.0, 5.0]
        print(f"\n[Teacher-core Recommendation]")
        print(f"  If we apply z-score threshold:")
        for z_th in z_threshold_candidates:
            remaining = sum(1 for t in teacher_only if t['teacher_z'] >= z_th)
            filtered = len(teacher_only) - remaining
            print(f"    z >= {z_th}: Teacher-only {remaining} remain, {filtered} filtered out")
    
    # 결론
    print(f"\n[Conclusion]")
    if teacher_only:
        median_z = np.median([t['teacher_z'] for t in teacher_only])
        if median_z < 2.0:
            print(f"  ✅ Teacher-only 항들의 median z-score = {median_z:.2f} < 2.0")
            print(f"  → '작고 흔들리는 노이즈 항' 가설 지지")
            print(f"  → Teacher-core (z >= 2.0) 필터링으로 과잉선택 완화 가능")
        else:
            print(f"  ⚠️ Teacher-only 항들의 median z-score = {median_z:.2f} >= 2.0")
            print(f"  → 일부 항은 유의미할 수 있음, 추가 분석 필요")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='QC-2: Teacher-only terms analysis')
    parser.add_argument('--oracle_dir', type=Path, required=True,
                        help='Oracle (n=50) 결과 디렉토리')
    parser.add_argument('--teacher_dir', type=Path, required=True,
                        help='Teacher (n=10) 결과 디렉토리')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Inclusion probability threshold (default: 0.5)')
    parser.add_argument('--output', type=Path, default=None,
                        help='결과 저장 경로 (JSON)')
    
    args = parser.parse_args()
    
    # 데이터 로드
    print(f"Loading Oracle from: {args.oracle_dir}")
    oracle_data = load_esindy_results(args.oracle_dir)
    
    print(f"Loading Teacher from: {args.teacher_dir}")
    teacher_data = load_esindy_results(args.teacher_dir)
    
    # 분석
    results = analyze_term_groups(oracle_data, teacher_data, args.threshold)
    
    # 리포트 출력
    print_analysis_report(results, args.threshold)
    
    # 저장
    if args.output:
        # numpy/bool 타입 변환
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        output_data = convert_types({
            'threshold': args.threshold,
            'summary': {
                'n_both': len(results['both']),
                'n_teacher_only': len(results['teacher_only']),
                'n_oracle_only': len(results['oracle_only']),
            },
            'teacher_only_z_stats': {
                'mean': float(np.mean([t['teacher_z'] for t in results['teacher_only']])) if results['teacher_only'] else None,
                'median': float(np.median([t['teacher_z'] for t in results['teacher_only']])) if results['teacher_only'] else None,
                'max': float(np.max([t['teacher_z'] for t in results['teacher_only']])) if results['teacher_only'] else None,
                'min': float(np.min([t['teacher_z'] for t in results['teacher_only']])) if results['teacher_only'] else None,
            },
            'details': results
        })
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  ✅ Saved: {args.output}")


if __name__ == '__main__':
    main()