"""
scripts/check_gate_artifacts.py

Gate 산출물 Lite 검증
- Gate0/1 통과 여부 판정
- 필수 산출물 3종 + Figure 3쌍 확인

Gate: 0-1
Version: v3.2 (Lean Mode)

Usage:
    python scripts/check_gate_artifacts.py --gate 0
    python scripts/check_gate_artifacts.py --gate 1 --dataset cartpole_ood_v1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import paths


# ============================================================
# 필수 산출물 정의
# ============================================================

REQUIRED_FILES = [
    'manifest.json',
    'metrics.json',
    'sindy_coefficients.csv'
]

REQUIRED_FIGURES = [
    'F00_condition_distribution',
    'F01_rollout_example',
    'F02_coeff_heatmap'
]


# ============================================================
# 검증 함수
# ============================================================

def check_run_artifacts(run_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    단일 run 폴더의 산출물 검증
    
    Args:
        run_dir: run 폴더 경로
    
    Returns:
        (통과 여부, 상세 결과)
    """
    result = {
        'run_dir': str(run_dir),
        'run_id': run_dir.name,
        'files': {},
        'figures': {},
        'errors': [],
        'passed': False
    }
    
    # 1. 필수 파일 검사
    for f in REQUIRED_FILES:
        file_path = run_dir / f
        exists = file_path.exists()
        result['files'][f] = exists
        
        if not exists:
            result['errors'].append(f"파일 누락: {f}")
        else:
            # JSON 파일 유효성 검사
            if f.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as fp:
                        json.load(fp)
                except Exception as e:
                    result['errors'].append(f"{f} 파싱 오류: {e}")
    
    # 2. 필수 Figure 검사 (PNG + PDF 쌍)
    figures_dir = run_dir / 'figures'
    
    if not figures_dir.exists():
        result['errors'].append("figures/ 폴더 없음")
    else:
        for fig_name in REQUIRED_FIGURES:
            png_path = figures_dir / f"{fig_name}.png"
            pdf_path = figures_dir / f"{fig_name}.pdf"
            
            png_exists = png_path.exists()
            pdf_exists = pdf_path.exists()
            
            result['figures'][fig_name] = {
                'png': png_exists,
                'pdf': pdf_exists,
                'both': png_exists and pdf_exists
            }
            
            if not png_exists:
                result['errors'].append(f"Figure 누락: {fig_name}.png")
            if not pdf_exists:
                result['errors'].append(f"Figure 누락: {fig_name}.pdf")
    
    # 3. 통과 여부 판정
    result['passed'] = len(result['errors']) == 0
    
    return result['passed'], result


def find_gate_runs(
    dataset_version: str,
    gate: str,
    track: str = None,
    method: str = None
) -> List[Path]:
    """
    Gate에 해당하는 모든 run 폴더 찾기
    
    Args:
        dataset_version: 데이터셋 버전
        gate: Gate 번호 (예: "gate0")
        track: 트랙 필터 (선택)
        method: 방법 필터 (선택)
    
    Returns:
        run 폴더 경로 리스트
    """
    gate_dir = paths.RESULTS_ROOT / dataset_version / gate
    
    if not gate_dir.exists():
        return []
    
    runs = []
    
    # 경로 구조: gate/track/method/n{N}/seed{S}/run_id/
    for track_dir in gate_dir.iterdir():
        if not track_dir.is_dir():
            continue
        if track and track_dir.name != track:
            continue
        
        for method_dir in track_dir.iterdir():
            if not method_dir.is_dir():
                continue
            if method and method_dir.name != method:
                continue
            
            for n_dir in method_dir.iterdir():
                if not n_dir.is_dir() or not n_dir.name.startswith('n'):
                    continue
                
                for seed_dir in n_dir.iterdir():
                    if not seed_dir.is_dir() or not seed_dir.name.startswith('seed'):
                        continue
                    
                    for run_dir in seed_dir.iterdir():
                        if run_dir.is_dir():
                            runs.append(run_dir)
    
    return runs


def check_gate(
    dataset_version: str,
    gate: str,
    track: str = None,
    method: str = None,
    verbose: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Gate 전체 검증
    
    Args:
        dataset_version: 데이터셋 버전
        gate: Gate 번호
        track: 트랙 필터 (선택)
        method: 방법 필터 (선택)
        verbose: 상세 출력 여부
    
    Returns:
        (Gate 통과 여부, 상세 결과)
    """
    # Gate 통과 기준
    MIN_SEEDS = 2  # 최소 2 seeds 필요
    
    result = {
        'dataset_version': dataset_version,
        'gate': gate,
        'track': track,
        'method': method,
        'checked_at': datetime.now().isoformat(),
        'runs': [],
        'summary': {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'unique_seeds': set()
        },
        'gate_passed': False
    }
    
    # run 폴더 찾기
    runs = find_gate_runs(dataset_version, gate, track, method)
    result['summary']['total'] = len(runs)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Gate 산출물 검증: {gate}")
        print(f"{'=' * 60}")
        print(f"Dataset: {dataset_version}")
        print(f"Track: {track or '(all)'}")
        print(f"Method: {method or '(all)'}")
        print(f"발견된 run 수: {len(runs)}")
        print(f"{'=' * 60}")
    
    if len(runs) == 0:
        if verbose:
            print("\n⚠️ 검사할 run이 없습니다.")
        return False, result
    
    # 각 run 검사
    for run_dir in runs:
        passed, run_result = check_run_artifacts(run_dir)
        result['runs'].append(run_result)
        
        if passed:
            result['summary']['passed'] += 1
            # seed 추출
            seed_dir = run_dir.parent
            if seed_dir.name.startswith('seed'):
                seed = int(seed_dir.name.replace('seed', ''))
                result['summary']['unique_seeds'].add(seed)
        else:
            result['summary']['failed'] += 1
        
        if verbose:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status}: {run_dir.name}")
            
            if not passed:
                for error in run_result['errors'][:3]:  # 최대 3개 오류만 표시
                    print(f"    └─ {error}")
                if len(run_result['errors']) > 3:
                    print(f"    └─ ... 외 {len(run_result['errors']) - 3}개 오류")
    
    # Gate 통과 판정
    unique_seeds = len(result['summary']['unique_seeds'])
    result['summary']['unique_seeds'] = list(result['summary']['unique_seeds'])
    
    gate_passed = (
        result['summary']['passed'] >= MIN_SEEDS and
        unique_seeds >= MIN_SEEDS
    )
    result['gate_passed'] = gate_passed
    
    # 요약 출력
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"검증 요약")
        print(f"{'=' * 60}")
        print(f"총 run 수: {result['summary']['total']}")
        print(f"통과: {result['summary']['passed']}")
        print(f"실패: {result['summary']['failed']}")
        print(f"고유 seed 수: {unique_seeds} (최소 {MIN_SEEDS} 필요)")
        print(f"{'=' * 60}")
        
        if gate_passed:
            print(f"\n🎉 {gate.upper()} 통과!")
        else:
            print(f"\n❌ {gate.upper()} 미통과")
            if unique_seeds < MIN_SEEDS:
                print(f"   └─ 최소 {MIN_SEEDS}개 seed 필요 (현재: {unique_seeds})")
    
    return gate_passed, result


def save_check_report(result: Dict[str, Any], output_path: Path = None) -> Path:
    """
    검증 결과를 JSON으로 저장
    """
    if output_path is None:
        gate = result['gate']
        dataset = result['dataset_version']
        output_path = paths.RESULTS_ROOT / dataset / gate / f"{gate}_check_report.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 검증 리포트 저장: {output_path}")
    
    return output_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gate 산출물 검증 (Lean Mode v3.2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_gate_artifacts.py --gate 0
  python scripts/check_gate_artifacts.py --gate 1 --dataset cartpole_ood_v1
  python scripts/check_gate_artifacts.py --gate 0 --track standardized --method latent_sindy
        """
    )
    
    parser.add_argument(
        '--gate', '-g',
        type=int,
        required=True,
        choices=[0, 1, 2],
        help='Gate 번호 (0, 1, 2)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='cartpole_ood_v1',
        help='데이터셋 버전 (기본값: cartpole_ood_v1)'
    )
    
    parser.add_argument(
        '--track', '-t',
        type=str,
        default=None,
        choices=['standardized', 'author_recommended'],
        help='트랙 필터 (선택)'
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        default=None,
        help='방법 필터 (선택)'
    )
    
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='검증 리포트 저장'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='간략 출력'
    )
    
    args = parser.parse_args()
    
    gate = f"gate{args.gate}"
    
    passed, result = check_gate(
        dataset_version=args.dataset,
        gate=gate,
        track=args.track,
        method=args.method,
        verbose=not args.quiet
    )
    
    if args.save:
        save_check_report(result)
    
    # 종료 코드 반환
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()