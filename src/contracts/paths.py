"""
SSOT: 경로 규칙 - 이 파일을 통해서만 경로 생성
수정 금지 파일 (S00에서 확정 후 절대 변경 안 함)
"""
from pathlib import Path
from datetime import datetime
import subprocess
import os

# 프로젝트 루트: 이 파일 기준 상대 경로 (src/contracts/paths.py → 2단계 위)
# 환경변수 PHD_PROJECT_ROOT가 있으면 그것을 우선 사용
ROOT = Path(os.environ.get('PHD_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
DATA_ROOT = ROOT / "data"
RESULTS_ROOT = ROOT / "results"


def get_git_sha() -> str:
    """Git SHA 가져오기 (없으면 'nogit')"""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha
    except:
        return "nogit"


def generate_run_id(note: str = "base") -> str:
    """run_id 생성: YYYYMMDD_HHMMSS_{gitsha}_{note}"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = get_git_sha()
    return f"{timestamp}_{git_sha}_{note}"


def get_results_dir(
    dataset_version: str,
    gate: str,
    track: str,
    method: str,
    n_train: int,
    seed: int,
    run_id: str
) -> Path:
    """
    결과 디렉토리 경로 반환 (자동 생성)
    경로: results/{dataset}/{gate}/{track}/{method}/n{N}/seed{S}/{run_id}/
    """
    path = (RESULTS_ROOT / dataset_version / gate / track / 
            method / f"n{n_train}" / f"seed{seed}" / run_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    return path


def get_dataset_path(dataset_version: str, system: str = "cartpole") -> Path:
    """dataset.npz 경로"""
    return DATA_ROOT / system / dataset_version / "dataset.npz"


def get_meta_path(dataset_version: str, system: str = "cartpole") -> Path:
    """meta.json 경로"""
    return DATA_ROOT / system / dataset_version / "meta.json"


def get_norm_stats_path(dataset_version: str, system: str = "cartpole") -> Path:
    """norm_stats.json 경로"""
    return DATA_ROOT / system / dataset_version / "norm_stats.json"


def get_context_packet_dir() -> Path:
    """Context Packet 디렉토리"""
    path = ROOT / "docs" / "context_packets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_context_packet_path(run_id: str) -> Path:
    """Context Packet 파일 경로"""
    return get_context_packet_dir() / f"CP_{run_id}.md"