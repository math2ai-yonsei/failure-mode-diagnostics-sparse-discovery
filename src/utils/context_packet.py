"""
src/utils/context_packet.py

Persistence Guard: Context Packet 자동 생성
- Runner 종료 시 자동 생성
- 새 대화창/AI에서 맥락 유지용

Gate: 0-1
Version: v3.2 (Lean Mode)
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import os

# paths.py import (상대 경로로)
try:
    from src.contracts import paths
except ImportError:
    # 직접 실행 시
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.contracts import paths


# ============================================================
# Context Packet 생성
# ============================================================

def generate_context_packet(
    results_dir: Path,
    gate: str,
    dataset_version: str,
    method: str,
    track: str,
    n_train: int,
    seed: int,
    run_id: str,
    metrics: Dict[str, Any],
    config_path: Optional[str] = None,
    artifacts: Optional[List[str]] = None,
    next_steps: Optional[List[str]] = None,
    notes: Optional[str] = None
) -> Path:
    """
    Context Packet 자동 생성
    
    Runner 종료 시 호출하여 CP_{run_id}.md 생성
    
    Args:
        results_dir: 결과 디렉토리 경로
        gate: Gate 번호 (예: "gate0")
        dataset_version: 데이터셋 버전
        method: 방법명
        track: 비교 트랙
        n_train: 학습 궤적 수
        seed: 랜덤 시드
        run_id: 실행 ID
        metrics: 메트릭 딕셔너리
        config_path: 설정 파일 경로 (선택)
        artifacts: 생성된 산출물 목록 (선택)
        next_steps: 다음 작업 목록 (선택)
        notes: 추가 메모 (선택)
    
    Returns:
        Path: 생성된 Context Packet 경로
    
    Example:
        >>> from src.utils.context_packet import generate_context_packet
        >>> cp_path = generate_context_packet(
        ...     results_dir=results_dir,
        ...     gate="gate0",
        ...     dataset_version="cartpole_ood_v1",
        ...     method="latent_sindy",
        ...     track="standardized",
        ...     n_train=10,
        ...     seed=0,
        ...     run_id=run_id,
        ...     metrics={'r2_train': 0.98, 'r2_test': 0.95}
        ... )
    """
    results_dir = Path(results_dir)
    
    # 산출물 체크 (자동 감지)
    if artifacts is None:
        artifacts = _detect_artifacts(results_dir)
    
    # 기본 다음 단계
    if next_steps is None:
        next_steps = _suggest_next_steps(gate, seed)
    
    # 실행 명령어 복원
    command = _build_command(
        gate=gate,
        config_path=config_path,
        dataset_version=dataset_version,
        n_train=n_train,
        seed=seed,
        track=track,
        note=run_id.split('_')[-1] if '_' in run_id else 'base'
    )
    
    # Metrics 포맷팅
    metrics_str = _format_metrics(metrics)
    
    # Artifacts 체크 문자열
    artifacts_str = _format_artifacts(artifacts)
    
    # Next steps 문자열
    next_steps_str = "\n".join([f"- [ ] {step}" for step in next_steps])
    
    # Context Packet 내용 생성
    content = f"""# Context Packet: {run_id}

## 프로젝트 정보
- **경로**: `{paths.ROOT}`
- **생성 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 실행 정보
| 항목 | 값 |
|------|-----|
| Gate | {gate} |
| Dataset | {dataset_version} |
| Method | {method} |
| Track | {track} |
| N_train | {n_train} |
| Seed | {seed} |
| Run ID | {run_id} |

## 실행 명령어 (복사용)
```bash
{command}
```

## Results 경로
```
{results_dir}
```

## 산출물 체크
{artifacts_str}

## Metrics 요약
{metrics_str}

## 다음 작업
{next_steps_str}

"""
    
    if notes:
        content += f"""## 메모
{notes}

"""
    
    content += """---
*이 파일은 자동 생성되었습니다. 새 대화창에서 이 내용을 붙여넣어 맥락을 유지하세요.*
"""
    
    # 파일 저장
    cp_path = paths.get_context_packet_path(run_id)
    cp_path.write_text(content, encoding='utf-8')
    
    print(f"✅ Context Packet 생성: {cp_path}")
    
    return cp_path


# ============================================================
# 헬퍼 함수
# ============================================================

def _detect_artifacts(results_dir: Path) -> List[str]:
    """
    결과 디렉토리에서 산출물 자동 감지
    """
    artifacts = []
    
    # 필수 산출물 3종
    required_files = ['manifest.json', 'metrics.json', 'sindy_coefficients.csv']
    for f in required_files:
        if (results_dir / f).exists():
            artifacts.append(f"✅ {f}")
        else:
            artifacts.append(f"❌ {f}")
    
    # 필수 Figure 3종
    figures_dir = results_dir / 'figures'
    required_figs = [
        'F00_condition_distribution',
        'F01_rollout_example',
        'F02_coeff_heatmap'
    ]
    
    for fig in required_figs:
        png_exists = (figures_dir / f"{fig}.png").exists()
        pdf_exists = (figures_dir / f"{fig}.pdf").exists()
        
        if png_exists and pdf_exists:
            artifacts.append(f"✅ figures/{fig}.png + .pdf")
        elif png_exists:
            artifacts.append(f"⚠️ figures/{fig}.png (PDF 누락)")
        elif pdf_exists:
            artifacts.append(f"⚠️ figures/{fig}.pdf (PNG 누락)")
        else:
            artifacts.append(f"❌ figures/{fig}")
    
    return artifacts


def _format_artifacts(artifacts: List[str]) -> str:
    """산출물 목록을 문자열로 포맷"""
    return "\n".join(artifacts)


def _format_metrics(metrics: Dict[str, Any]) -> str:
    """메트릭을 테이블 형식으로 포맷"""
    if not metrics:
        return "*(메트릭 없음)*"
    
    lines = ["| 메트릭 | 값 |", "|--------|-----|"]
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    
    return "\n".join(lines)


def _suggest_next_steps(gate: str, seed: int) -> List[str]:
    """다음 단계 제안"""
    steps = []
    
    if gate == "gate0":
        if seed == 0:
            steps.append(f"seed {seed + 1} 실행")
        steps.append("Gate0 통과 확인: python scripts/check_gate_artifacts.py --gate 0")
        steps.append("Gate1 진행 (E-SINDy baseline)")
    elif gate == "gate1":
        steps.append("다른 method/track 실행")
        steps.append("gate1_summary.csv 생성")
        steps.append("Gate2 진행 (Proposed method)")
    
    return steps


def _build_command(
    gate: str,
    config_path: Optional[str],
    dataset_version: str,
    n_train: int,
    seed: int,
    track: str,
    note: str
) -> str:
    """실행 명령어 복원"""
    if config_path is None:
        config_path = f"configs/experiments/{gate}_cartpole.yaml"
    
    return f"""python experiments/run_{gate}.py ^
  --config {config_path} ^
  --dataset_version {dataset_version} ^
  --n_train {n_train} ^
  --seed {seed} ^
  --track {track} ^
  --note {note}"""


# ============================================================
# 간편 함수 (Runner에서 호출용)
# ============================================================

def generate_context_packet_simple(
    results_dir: Path,
    args,  # argparse.Namespace
    metrics: Dict[str, Any],
    run_id: str
) -> Path:
    """
    Runner에서 간편하게 호출할 수 있는 버전
    
    Args:
        results_dir: 결과 디렉토리
        args: argparse로 파싱된 인자 (config, dataset_version 등 포함)
        metrics: 메트릭 딕셔너리
        run_id: 실행 ID
    
    Example:
        >>> # run_gate0.py에서
        >>> from src.utils.context_packet import generate_context_packet_simple
        >>> generate_context_packet_simple(results_dir, args, metrics, run_id)
    """
    return generate_context_packet(
        results_dir=results_dir,
        gate=getattr(args, 'gate', 'gate0'),
        dataset_version=args.dataset_version,
        method=getattr(args, 'method', 'latent_sindy'),
        track=args.track,
        n_train=args.n_train,
        seed=args.seed,
        run_id=run_id,
        metrics=metrics,
        config_path=getattr(args, 'config', None)
    )


# ============================================================
# 테스트/검증
# ============================================================

if __name__ == "__main__":
    import tempfile
    
    print("=" * 60)
    print("context_packet.py 검증")
    print("=" * 60)
    
    # 테스트용 결과 디렉토리 생성
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        
        # 가짜 산출물 생성
        (results_dir / "manifest.json").write_text("{}")
        (results_dir / "metrics.json").write_text("{}")
        (results_dir / "sindy_coefficients.csv").write_text("term,coeff\n")
        
        figures_dir = results_dir / "figures"
        figures_dir.mkdir()
        (figures_dir / "F00_condition_distribution.png").write_text("")
        (figures_dir / "F00_condition_distribution.pdf").write_text("")
        (figures_dir / "F01_rollout_example.png").write_text("")
        (figures_dir / "F01_rollout_example.pdf").write_text("")
        # F02는 일부러 누락
        
        # Context Packet 생성
        print("\n[Context Packet 생성 테스트]")
        
        test_run_id = paths.generate_run_id("test")
        test_metrics = {
            'r2_train': 0.987,
            'r2_test': 0.952,
            'rollout_nrmse_20': 0.034,
            'n_selected_terms': 8
        }
        
        cp_path = generate_context_packet(
            results_dir=results_dir,
            gate="gate0",
            dataset_version="cartpole_ood_v1",
            method="latent_sindy",
            track="standardized",
            n_train=10,
            seed=0,
            run_id=test_run_id,
            metrics=test_metrics
        )
        
        # 내용 확인
        print("\n[생성된 Context Packet 내용 (앞부분)]")
        content = cp_path.read_text(encoding='utf-8')
        print(content[:1500] + "\n...")
    
    print("\n" + "=" * 60)
    print("✅ context_packet.py 검증 완료")
    print("=" * 60)