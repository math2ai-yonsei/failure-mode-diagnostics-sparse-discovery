#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structure Metrics Smoke Test
============================
compute_structure_metrics()가 올바르게 작동하는지 검증

테스트:
1. teacher vs teacher → Jaccard=1, F1=1, Corr=1, RMSE=0
2. pred vs pred → 동일
3. 실제 teacher vs pred 비교

Usage:
    python src/experiments/test_structure_metrics.py --teacher_run_dir <path>
"""

import argparse
import sys
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_coefficients(run_dir):
    """Load coefficients from sindy_coefficients.csv"""
    coef_path = Path(run_dir) / 'sindy_coefficients.csv'
    if not coef_path.exists():
        raise FileNotFoundError(f"Coefficients not found: {coef_path}")
    
    with open(coef_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        data_cols = [i for i, h in enumerate(header) if h.startswith('dx_')]
        if not data_cols:
            data_cols = list(range(1, len(header)))
        rows = [[float(row[i]) for i in data_cols] for row in reader]
    return np.array(rows)


def compute_structure_metrics(pred_coef, teacher_coef, term_freq=None):
    """
    Compute teacher-based structure metrics (복사본 for 테스트)
    """
    # Support definition: abs(coef) > 1e-12
    teacher_support = np.abs(teacher_coef) > 1e-12
    pred_support = np.abs(pred_coef) > 1e-12
    
    # Jaccard similarity
    intersection = np.sum(teacher_support & pred_support)
    union = np.sum(teacher_support | pred_support)
    jaccard = intersection / (union + 1e-10)
    
    # Precision/Recall on support
    tp = np.sum(teacher_support & pred_support)
    fp = np.sum(~teacher_support & pred_support)
    fn = np.sum(teacher_support & ~pred_support)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Coefficient correlation on teacher support
    teacher_flat = teacher_coef[teacher_support]
    pred_flat = pred_coef[teacher_support]
    
    if len(teacher_flat) > 1:
        coef_corr = np.corrcoef(teacher_flat, pred_flat)[0, 1]
        if np.isnan(coef_corr):
            coef_corr = 0.0
    else:
        coef_corr = 0.0
    
    # Coefficient RMSE on teacher support
    if len(teacher_flat) > 0:
        coef_rmse = np.sqrt(np.mean((teacher_flat - pred_flat)**2))
    else:
        coef_rmse = 0.0
    
    return {
        'jaccard': float(jaccard),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'coef_correlation': float(coef_corr),
        'coef_rmse': float(coef_rmse),
        'n_teacher_terms': int(np.sum(teacher_support)),
        'n_pred_terms': int(np.sum(pred_support)),
        'n_intersection': int(intersection),
    }


def main():
    parser = argparse.ArgumentParser(description='Structure metrics smoke test')
    parser.add_argument('--teacher_run_dir', type=str, required=True)
    parser.add_argument('--pred_run_dir', type=str, default=None, 
                        help='Optional: compare with another run')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Structure Metrics Smoke Test")
    print("=" * 60)
    
    # Load teacher coefficients
    teacher_coef = load_coefficients(args.teacher_run_dir)
    print(f"\nTeacher coef shape: {teacher_coef.shape}")
    print(f"Teacher coef range: [{teacher_coef.min():.4f}, {teacher_coef.max():.4f}]")
    print(f"Teacher non-zero terms: {np.sum(np.abs(teacher_coef) > 1e-12)}")
    
    # Test 1: teacher vs teacher (should be perfect)
    print("\n" + "-" * 60)
    print("Test 1: teacher vs teacher (expected: Jaccard=1, Corr=1, RMSE=0)")
    print("-" * 60)
    
    metrics_self = compute_structure_metrics(teacher_coef, teacher_coef)
    print(f"  Jaccard:     {metrics_self['jaccard']:.4f} (expected: 1.0)")
    print(f"  F1:          {metrics_self['f1']:.4f} (expected: 1.0)")
    print(f"  Coef Corr:   {metrics_self['coef_correlation']:.4f} (expected: 1.0)")
    print(f"  Coef RMSE:   {metrics_self['coef_rmse']:.6f} (expected: 0.0)")
    
    # Verify
    assert abs(metrics_self['jaccard'] - 1.0) < 1e-6, "Jaccard should be 1.0"
    assert abs(metrics_self['f1'] - 1.0) < 1e-6, "F1 should be 1.0"
    assert abs(metrics_self['coef_correlation'] - 1.0) < 1e-6, "Corr should be 1.0"
    assert abs(metrics_self['coef_rmse']) < 1e-10, "RMSE should be 0.0"
    print("  ✅ PASSED")
    
    # Test 2: teacher vs scaled teacher (should have Corr=1, but RMSE > 0)
    print("\n" + "-" * 60)
    print("Test 2: teacher vs 2*teacher (expected: Corr=1, RMSE>0)")
    print("-" * 60)
    
    scaled_coef = teacher_coef * 2.0
    metrics_scaled = compute_structure_metrics(scaled_coef, teacher_coef)
    print(f"  Jaccard:     {metrics_scaled['jaccard']:.4f} (expected: 1.0)")
    print(f"  Coef Corr:   {metrics_scaled['coef_correlation']:.4f} (expected: 1.0)")
    print(f"  Coef RMSE:   {metrics_scaled['coef_rmse']:.6f} (expected: >0)")
    
    assert abs(metrics_scaled['jaccard'] - 1.0) < 1e-6, "Jaccard should be 1.0"
    assert abs(metrics_scaled['coef_correlation'] - 1.0) < 1e-6, "Corr should be 1.0"
    assert metrics_scaled['coef_rmse'] > 0, "RMSE should be > 0"
    print("  ✅ PASSED")
    
    # Test 3: teacher vs -teacher (should have Corr=-1)
    print("\n" + "-" * 60)
    print("Test 3: teacher vs -teacher (expected: Corr=-1)")
    print("-" * 60)
    
    neg_coef = -teacher_coef
    metrics_neg = compute_structure_metrics(neg_coef, teacher_coef)
    print(f"  Jaccard:     {metrics_neg['jaccard']:.4f} (expected: 1.0)")
    print(f"  Coef Corr:   {metrics_neg['coef_correlation']:.4f} (expected: -1.0)")
    
    assert abs(metrics_neg['jaccard'] - 1.0) < 1e-6, "Jaccard should be 1.0"
    assert abs(metrics_neg['coef_correlation'] - (-1.0)) < 1e-6, "Corr should be -1.0"
    print("  ✅ PASSED")
    
    # Test 4: Actual pred vs teacher (if provided)
    if args.pred_run_dir:
        print("\n" + "-" * 60)
        print(f"Test 4: Actual comparison with {Path(args.pred_run_dir).name}")
        print("-" * 60)
        
        pred_coef = load_coefficients(args.pred_run_dir)
        print(f"  Pred coef shape: {pred_coef.shape}")
        print(f"  Pred coef range: [{pred_coef.min():.4f}, {pred_coef.max():.4f}]")
        
        metrics_actual = compute_structure_metrics(pred_coef, teacher_coef)
        print(f"\n  Jaccard:     {metrics_actual['jaccard']:.4f}")
        print(f"  F1:          {metrics_actual['f1']:.4f}")
        print(f"  Coef Corr:   {metrics_actual['coef_correlation']:.4f}")
        print(f"  Coef RMSE:   {metrics_actual['coef_rmse']:.4f}")
        
        # Show coefficient comparison for teacher support
        teacher_support = np.abs(teacher_coef) > 1e-12
        print(f"\n  Teacher support ({np.sum(teacher_support)} terms):")
        
        indices = np.argwhere(teacher_support)
        for idx in indices[:10]:  # Show first 10
            i, j = idx
            t_val = teacher_coef[i, j]
            p_val = pred_coef[i, j]
            print(f"    [{i},{j}]: teacher={t_val:+.4f}, pred={p_val:+.4f}, diff={p_val-t_val:+.4f}")
        
        if len(indices) > 10:
            print(f"    ... ({len(indices)-10} more terms)")
    
    print("\n" + "=" * 60)
    print("✅ All smoke tests passed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())