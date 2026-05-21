r"""
Phase 3.5 검증 스크립트: n=10 Teacher 결과로 Core Mining 검증

실행 방법:
    cd C:\python_work\PhD_project
    python scripts/verify_core_mining.py

이 스크립트는:
1. n=10 Teacher sindy_coefficients.csv 로드
2. StableCoreMiner로 core mining 수행
3. QC-2 분석 결과와 비교 검증
4. 결과를 results/phase35/core_mining/ 에 저장
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.contracts import paths  # type: ignore  # noqa: E402  # pylint: disable=import-error
from sindy.core_mining import StableCoreMiner, validate_against_qc2  # type: ignore  # noqa: E402  # pylint: disable=import-error


def main():
    print("=" * 70)
    print("Phase 3.5 검증: n=10 Teacher → Stable-core Mining")
    print("=" * 70)
    
    # 경로 설정
    teacher_csv = (paths.RESULTS_ROOT / 
                   "cartpole_ood_v1/gate1/standardized/esindy/"
                   "n10/seed0/20251229_213749_nogit_base/sindy_coefficients.csv")
    
    qc2_json = paths.RESULTS_ROOT / "qc2_teacher_only_analysis.json"
    
    print("\n[입력 파일]")
    print(f"  Teacher CSV: {teacher_csv}")
    print(f"  QC-2 JSON: {qc2_json}")
    
    # 파일 존재 확인
    if not teacher_csv.exists():
        print(f"\n❌ Teacher CSV 없음: {teacher_csv}")
        print("   먼저 n=10 E-SINDy 실험을 실행하세요.")
        return 1
    
    # Core Mining 수행
    print("\n[Core Mining 수행]")
    result = StableCoreMiner.from_esindy_csv(
        teacher_csv, 
        tau_hi=0.5, 
        z0=2.0, 
        eps=1e-12
    )
    
    print(f"  전체 항: {result.n_total_terms}")
    print(f"  활성 항 (inc_prob >= 0.5): {result.n_active_terms}")
    print(f"  Stable-core: {result.n_stable_core}")
    print(f"  Fragile-pool: {result.n_fragile_pool}")
    
    # QC-2 검증 (파일이 있으면)
    if qc2_json.exists():
        print("\n[QC-2 검증]")
        validation = validate_against_qc2(result, qc2_json)
        
        print(f"  기대 Stable-core: {validation['expected']['n_stable_core']}")
        print(f"  기대 Fragile-pool: {validation['expected']['n_fragile_pool']}")
        print(f"  실제 Stable-core: {validation['actual']['n_stable_core']}")
        print(f"  실제 Fragile-pool: {validation['actual']['n_fragile_pool']}")
        
        if validation['passed']:
            print("\n  ✅ QC-2 검증 통과!")
        else:
            print("\n  ❌ QC-2 검증 실패!")
            return 1
    else:
        print("\n[참고] QC-2 JSON 없음, 검증 건너뜀")
    
    # Stable-core 상세 출력
    print(f"\n[Stable-core 항 상세] ({result.n_stable_core}개)")
    for i, term in enumerate(result.stable_core_terms, 1):
        print(f"  {i:2d}. {term['feature']:20s} → {term['target']:12s}: "
              f"z={term['z_score']:10.2f}, inc_prob={term['inc_prob']:.2f}")
    
    # Fragile-pool oracle-true 항 (z < 2.0인 Both 항)
    print(f"\n[Fragile-pool 상위 항] (총 {result.n_fragile_pool}개)")
    for i, term in enumerate(result.fragile_pool_terms[:10], 1):
        print(f"  {i:2d}. {term['feature']:20s} → {term['target']:12s}: "
              f"z={term['z_score']:.3f}, inc_prob={term['inc_prob']:.2f}")
    
    # 결과 저장
    output_dir = paths.RESULTS_ROOT / "cartpole_ood_v1/phase35/core_mining"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_json = output_dir / "n10_teacher_core_mining.json"
    result.save_json(output_json)
    print(f"\n[결과 저장] {output_json}")
    
    print("\n" + "=" * 70)
    print("✅ Phase 3.5 Day 1 완료: core_mining.py 검증 성공!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())