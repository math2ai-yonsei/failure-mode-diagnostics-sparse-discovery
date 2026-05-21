"""
Phase 3.5 Runner: Stable-core Selection Experiment

핵심 목표:
- Teacher (n=10) vs Oracle (n=50) support 비교
- Stable-core / Fragile-pool 분류
- Structure Evaluation 수행

산출물:
- manifest.json: 실험 메타데이터
- metrics.json: Primary metrics
- structure_eval.json: 구조 평가 결과
- core_mining.json: Core mining 결과
- pool_metadata.json: Support 3종 + z-score 정보

Author: Claude (Phase 3.5 Day2)
Updated: Phase 3.5 Option B - Modern Schema
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (src 모듈 import용)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import yaml

# Selection 모듈 import (SSOT)
from src.sindy.selection import select_terms, SelectionResult

# === MODERNIZE IMPORT START ===
from phase35_manifest_modernize import (
    get_control_equivalence,
    compute_file_sha256,
    create_code_snapshot
)
# === MODERNIZE IMPORT END ===


# ============================================================
# SSOT Constants
# ============================================================

# 기본 target names (Gate1 manifest에 없을 경우 사용)
DEFAULT_TARGET_NAMES = ["x_dot", "x_ddot", "theta_dot", "theta_ddot"]

# Phase 3.5 하이퍼파라미터 (사전 고정)
DEFAULT_TAU_HI = 0.5
DEFAULT_Z0 = 2.0
DEFAULT_EPS = 1e-12


# ============================================================
# Configuration
# ============================================================

@dataclass
class Phase35Config:
    """Phase 3.5 실험 설정"""
    
    # Dataset
    dataset_version: str = "cartpole_ood_v1"
    
    # Gate1 artifacts
    teacher_run_id: str = ""  # n=10 Teacher
    oracle_run_id: str = ""   # n=50 Oracle
    
    # Track
    track: str = "standardized"
    
    # Core mining params (사전 고정)
    tau_hi: float = DEFAULT_TAU_HI
    z0: float = DEFAULT_Z0
    eps: float = DEFAULT_EPS
    
    # === Day3: Selection 파라미터 ===
    # arm: 'A' (stable_core_only) or 'B' (budget_plus_fragile)
    arm: str = "A"
    # budget: Arm B에서 최대 선택 개수 (Oracle n_active = 9)
    budget: int = 9
    # selection_method: arm에서 자동 결정 (override 가능)
    selection_method: str = ""  # auto-set based on arm
    
    # Experiment
    seed: int = 0
    note: str = "core_mining"
    
    def __post_init__(self):
        """arm에 따라 selection_method 자동 설정"""
        # 항상 arm에 맞게 selection_method 설정 (CLI override 후에도 동작하도록)
        if self.arm == "A":
            self.selection_method = "stable_core_only"
        elif self.arm == "B":
            self.selection_method = "budget_plus_fragile"
        else:
            raise ValueError(f"Unknown arm: {self.arm}. Use 'A' or 'B'")
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Phase35Config':
        """YAML 파일에서 설정 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Phase35Config':
        """Dict에서 설정 로드"""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# ============================================================
# Helper Functions
# ============================================================

def generate_run_id(note: str = "") -> str:
    """run_id 생성: YYYYMMDD_HHMMSS_nogit_{note}"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_part = f"_{note}" if note else ""
    return f"{timestamp}_nogit{note_part}"


def load_gate1_manifest(run_dir: Path) -> Dict[str, Any]:
    """Gate1 manifest.json 로드"""
    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Gate1 manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_feature_names(manifest: Dict) -> List[str]:
    """manifest에서 feature_names 추출"""
    return manifest['library']['feature_names']


def get_target_names(manifest: Dict) -> List[str]:
    """manifest에서 target_names 추출 (없으면 기본값)"""
    # Gate1 manifest에 optimizer.target_names가 없을 수 있음
    if 'optimizer' in manifest and 'target_names' in manifest['optimizer']:
        return manifest['optimizer']['target_names']
    else:
        return DEFAULT_TARGET_NAMES


def get_gate1_run_dir(
    dataset_version: str,
    track: str, 
    n_train: int,
    seed: int,
    run_id: str
) -> Path:
    """Gate1 run 디렉토리 경로 생성"""
    return (_PROJECT_ROOT / 'results' / dataset_version / 'gate1' / 
            track / 'esindy' / f'n{n_train}' / f'seed{seed}' / run_id)


def get_phase35_results_dir(
    dataset_version: str,
    track: str,
    method: str,
    n_train: int,
    seed: int,
    run_id: str
) -> Path:
    """Phase 3.5 results 디렉토리 경로 생성"""
    results_dir = (_PROJECT_ROOT / 'results' / dataset_version / 'phase35' /
                   track / method / f'n{n_train}' / f'seed{seed}' / run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def compute_support_mask(inc_prob: np.ndarray, tau_hi: float) -> np.ndarray:
    """Support mask 계산: inc_prob >= tau_hi"""
    return inc_prob >= tau_hi


# ============================================================
# Phase 3.5 Runner
# ============================================================

class Phase35Runner:
    """
    Phase 3.5 Stable-core Selection Runner
    
    핵심 파이프라인:
    1. Gate1 artifacts 로드 (Teacher, Oracle)
    2. CSV 포맷 검증
    3. Core Mining 수행
    4. Oracle vs Teacher support 비교
    5. Structure Evaluation 수행
    6. 산출물 저장
    """
    
    def __init__(self, config: Phase35Config):
        self.config = config
        self.run_id = generate_run_id(config.note)
        
        # Artifacts will be loaded
        self.teacher_manifest: Optional[Dict] = None
        self.oracle_manifest: Optional[Dict] = None
        self.feature_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        self.target_names_source: str = ""  # 'gate1_manifest' or 'default_fallback'
        
        # Results
        self.results_dir: Optional[Path] = None
        self.core_mining_result = None
        self.oracle_core_result = None
        self.selection_result = None  # Day3: Selection 결과
        self.structure_eval_result = None
        self.comparison: Optional[Dict] = None  # Step 5 결과 저장용
        
    def _banner(self, msg: str):
        """구분선 출력"""
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"{'='*60}")
    
    def _section(self, step: str, title: str):
        """섹션 헤더 출력"""
        print(f"\n[{step}] {title}")
        print("-" * 50)
    
    def run(self) -> Dict[str, Any]:
        """메인 실행 파이프라인"""
        
        self._banner(f"Phase 3.5 Runner: {self.run_id}")
        print(f"  Arm: {self.config.arm} ({self.config.selection_method})")
        if self.config.arm == 'B':
            print(f"  Budget: {self.config.budget}")
        
        # Step 1: Gate1 artifacts 로드
        self._section("1/8", "Loading Gate1 Artifacts")
        self._load_gate1_artifacts()
        
        # Step 2: CSV 포맷 검증 (R1 Smoke Test)
        self._section("2/8", "Validating CSV Format")
        self._validate_csv_format()
        
        # Step 3: Core Mining 수행 (Teacher)
        self._section("3/8", "Teacher Core Mining")
        self._run_teacher_core_mining()
        
        # Step 4: Oracle Core Mining (support 비교용)
        self._section("4/8", "Oracle Core Mining")
        self._run_oracle_core_mining()
        
        # Step 5: Support 비교 분석
        self._section("5/8", "Support Comparison")
        self.comparison = self._compare_supports()  # self에 저장하여 재사용
        
        # Step 6: Selection (Day3)
        self._section("6/8", "Term Selection")
        self._run_selection()
        
        # Step 7: Structure Evaluation
        self._section("7/8", "Structure Evaluation")
        self._run_structure_evaluation()
        
        # Step 8: 산출물 저장
        self._section("8/8", "Saving Artifacts")
        self._save_artifacts()
        
        self._banner(f"Phase 3.5 Complete: {self.run_id}")
        print(f"  Results: {self.results_dir}")
        
        return {
            'run_id': self.run_id,
            'results_dir': str(self.results_dir),
            'comparison': self.comparison,
            'selection': self.selection_result.to_dict() if self.selection_result else None
        }
    
    def _load_gate1_artifacts(self):
        """Gate1 artifacts 로드 (Teacher & Oracle)"""
        cfg = self.config
        
        # Teacher (n=10)
        teacher_dir = get_gate1_run_dir(
            cfg.dataset_version, cfg.track, 10, cfg.seed, cfg.teacher_run_id
        )
        self.teacher_manifest = load_gate1_manifest(teacher_dir)
        print(f"  Teacher: {cfg.teacher_run_id}")
        print(f"    Path: {teacher_dir}")
        print(f"    n_train: {self.teacher_manifest['config']['n_train']}")
        
        # Oracle (n=50)
        oracle_dir = get_gate1_run_dir(
            cfg.dataset_version, cfg.track, 50, cfg.seed, cfg.oracle_run_id
        )
        self.oracle_manifest = load_gate1_manifest(oracle_dir)
        print(f"  Oracle: {cfg.oracle_run_id}")
        print(f"    Path: {oracle_dir}")
        print(f"    n_train: {self.oracle_manifest['config']['n_train']}")
        
        # SSOT 추출 (Teacher manifest 기준)
        self.feature_names = get_feature_names(self.teacher_manifest)
        
        # SSOT: target_names는 CSV 컬럼 헤더에서 추출 (가장 신뢰할 수 있는 소스)
        import pandas as pd
        teacher_csv_path = teacher_dir / 'sindy_coefficients.csv'
        df = pd.read_csv(teacher_csv_path)
        term_col = 'term_name' if 'term_name' in df.columns else df.columns[0]
        csv_target_names = [c for c in df.columns if c != term_col]
        
        # manifest에도 있으면 일치 확인, 없으면 CSV 기준
        if 'optimizer' in self.teacher_manifest and 'target_names' in self.teacher_manifest['optimizer']:
            manifest_target_names = self.teacher_manifest['optimizer']['target_names']
            if csv_target_names != manifest_target_names:
                print(f"    ⚠️ CSV columns {csv_target_names} != manifest {manifest_target_names}")
                print(f"    Using CSV columns as SSOT")
            else:
                self.target_names_source = 'gate1_manifest_verified'
            self.target_names = csv_target_names
        else:
            self.target_names = csv_target_names
            self.target_names_source = 'csv_columns'  # CSV에서 추출했음을 명시
            print(f"    ✅ target_names extracted from CSV columns: {csv_target_names}")
        
        print(f"  SSOT:")
        print(f"    n_features: {len(self.feature_names)}")
        print(f"    n_targets: {len(self.target_names)}")
        print(f"    target_names: {self.target_names}")
        print(f"    target_names_source: {self.target_names_source}")
        
        # Results 디렉토리 생성
        self.results_dir = get_phase35_results_dir(
            cfg.dataset_version, cfg.track, 'stable_core',
            10, cfg.seed, self.run_id
        )
    
    def _validate_csv_format(self):
        """Gate1 CSV 포맷 검증 (R1 Smoke Test) - P0 강화"""
        cfg = self.config
        import pandas as pd
        
        required_files = [
            'sindy_coefficients.csv',
            'coefficient_std.csv',
            'inclusion_probability.csv'
        ]
        
        def validate_run_csvs(run_dir: Path, run_label: str):
            """단일 run의 3개 CSV 검증"""
            for fname in required_files:
                fpath = run_dir / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"Missing Gate1 artifact: {fpath}")
                
                df = pd.read_csv(fpath)
                
                # 1. term_name 컬럼 존재
                if 'term_name' not in df.columns:
                    raise ValueError(f"'term_name' column not found in {fname}")
                
                # 2. row count 확인
                if len(df) != len(self.feature_names):
                    raise ValueError(
                        f"Row count mismatch in {run_label}/{fname}: "
                        f"{len(df)} vs {len(self.feature_names)}"
                    )
                
                # 3. target 컬럼 확인
                csv_targets = [c for c in df.columns if c != 'term_name']
                if set(csv_targets) != set(self.target_names):
                    raise ValueError(
                        f"Target columns mismatch in {run_label}/{fname}: "
                        f"{csv_targets} vs {self.target_names}"
                    )
                
                # 4. P0 FIX: feature set 확인
                csv_features = df['term_name'].tolist()
                if set(csv_features) != set(self.feature_names):
                    raise ValueError(
                        f"Feature mismatch in {run_label}/{fname}: "
                        f"CSV has {set(csv_features)}, expected {set(self.feature_names)}"
                    )
                
                # 5. P0 FIX: 중복 term_name 확인
                if len(csv_features) != len(set(csv_features)):
                    duplicates = [f for f in csv_features if csv_features.count(f) > 1]
                    raise ValueError(
                        f"Duplicate term_name in {run_label}/{fname}: {set(duplicates)}"
                    )
                
                print(f"  ✅ {run_label}/{fname}: {len(df)} rows, {len(csv_targets)} targets")
        
        # Teacher CSV 검증
        teacher_dir = get_gate1_run_dir(
            cfg.dataset_version, cfg.track, 10, cfg.seed, cfg.teacher_run_id
        )
        validate_run_csvs(teacher_dir, "Teacher")
        
        # P0 FIX: Oracle CSV도 검증
        oracle_dir = get_gate1_run_dir(
            cfg.dataset_version, cfg.track, 50, cfg.seed, cfg.oracle_run_id
        )
        validate_run_csvs(oracle_dir, "Oracle")
        
        print(f"  R1 Smoke Test: PASSED")
    
    def _run_teacher_core_mining(self):
        """Teacher Core Mining 수행"""
        from src.sindy.core_mining import StableCoreMiner
        
        cfg = self.config
        teacher_dir = get_gate1_run_dir(
            cfg.dataset_version, cfg.track, 10, cfg.seed, cfg.teacher_run_id
        )
        
        self.core_mining_result = StableCoreMiner.from_gate1_artifacts(
            run_dir=teacher_dir,
            feature_names=self.feature_names,
            target_names=self.target_names,
            tau_hi=cfg.tau_hi,
            z0=cfg.z0,
            eps=cfg.eps
        )
        
        result = self.core_mining_result
        print(f"  Active terms (inc_prob >= {cfg.tau_hi}): {result.n_active_terms}")
        print(f"  Stable-core (z >= {cfg.z0}): {result.n_stable_core}")
        print(f"  Fragile-pool (z < {cfg.z0}): {result.n_fragile_pool}")
    
    def _run_oracle_core_mining(self):
        """Oracle Core Mining 수행 (support 비교용)"""
        from src.sindy.core_mining import StableCoreMiner
        
        cfg = self.config
        oracle_dir = get_gate1_run_dir(
            cfg.dataset_version, cfg.track, 50, cfg.seed, cfg.oracle_run_id
        )
        
        self.oracle_core_result = StableCoreMiner.from_gate1_artifacts(
            run_dir=oracle_dir,
            feature_names=self.feature_names,
            target_names=self.target_names,
            tau_hi=cfg.tau_hi,
            z0=cfg.z0,
            eps=cfg.eps
        )
        
        result = self.oracle_core_result
        print(f"  Active terms (inc_prob >= {cfg.tau_hi}): {result.n_active_terms}")
        print(f"  Stable-core (z >= {cfg.z0}): {result.n_stable_core}")
        print(f"  Fragile-pool (z < {cfg.z0}): {result.n_fragile_pool}")
    
    def _compare_supports(self) -> Dict[str, Any]:
        """Teacher vs Oracle support 비교"""
        teacher = self.core_mining_result
        oracle = self.oracle_core_result
        
        teacher_support = teacher.active_mask
        oracle_support = oracle.active_mask
        
        # Set operations
        both = teacher_support & oracle_support
        teacher_only = teacher_support & ~oracle_support
        oracle_only = ~teacher_support & oracle_support
        
        n_both = int(np.sum(both))
        n_teacher_only = int(np.sum(teacher_only))
        n_oracle_only = int(np.sum(oracle_only))
        
        print(f"  Teacher support: {int(np.sum(teacher_support))} terms")
        print(f"  Oracle support: {int(np.sum(oracle_support))} terms")
        print(f"  Both: {n_both}")
        print(f"  Teacher-only (spurious): {n_teacher_only}")
        print(f"  Oracle-only (missed): {n_oracle_only}")
        
        # Stable-core 중 oracle-true 비율
        stable_core = teacher.stable_core_mask
        oracle_true_in_stable = np.sum(stable_core & oracle_support)
        print(f"\n  Stable-core analysis:")
        print(f"    Stable-core total: {teacher.n_stable_core}")
        print(f"    Oracle-true in stable-core: {oracle_true_in_stable}")
        
        # Fragile-pool 중 oracle-true 비율
        fragile_pool = teacher.fragile_pool_mask
        oracle_true_in_fragile = np.sum(fragile_pool & oracle_support)
        print(f"  Fragile-pool analysis:")
        print(f"    Fragile-pool total: {teacher.n_fragile_pool}")
        print(f"    Oracle-true in fragile-pool: {oracle_true_in_fragile}")
        
        comparison = {
            'teacher_support_count': int(np.sum(teacher_support)),
            'oracle_support_count': int(np.sum(oracle_support)),
            'both_count': n_both,
            'teacher_only_count': n_teacher_only,
            'oracle_only_count': n_oracle_only,
            'stable_core_count': teacher.n_stable_core,
            'oracle_true_in_stable_core': int(oracle_true_in_stable),
            'fragile_pool_count': teacher.n_fragile_pool,
            'oracle_true_in_fragile_pool': int(oracle_true_in_fragile)
        }
        
        return comparison
    
    def _run_selection(self):
        """Day3: Term Selection 수행 (SSOT: src/sindy/selection.py)"""
        cfg = self.config
        teacher = self.core_mining_result
        
        # Selection 모듈 호출 (SSOT - 드리프트 방지)
        self.selection_result = select_terms(
            stable_core_mask=teacher.stable_core_mask,
            fragile_pool_mask=teacher.fragile_pool_mask,
            teacher_support=teacher.active_mask,
            inc_prob=teacher.inc_prob,
            z_scores=teacher.z_scores,
            feature_names=self.feature_names,
            target_names=self.target_names,
            method=cfg.selection_method,
            budget=cfg.budget if cfg.arm == 'B' else None
        )
        
        result = self.selection_result
        print(f"  Method: {result.method}")
        print(f"  Selected: {result.n_selected} terms")
        print(f"    - Stable-core: {result.n_stable_core_selected}")
        print(f"    - Fragile: {result.n_fragile_selected}")
        if cfg.arm == 'B':
            print(f"  Budget: {result.budget}")
    
    def _run_structure_evaluation(self):
        """Structure Evaluation 수행"""
        from src.evaluation.structure_eval import StructureEvaluator
        
        teacher = self.core_mining_result
        oracle = self.oracle_core_result
        
        # Oracle support as ground truth
        oracle_support = oracle.active_mask
        
        evaluator = StructureEvaluator(
            oracle_support=oracle_support,
            feature_names=self.feature_names,
            target_names=self.target_names
        )
        
        # Day3: Selection 적용된 support 사용
        selected_support = self.selection_result.selected_mask if self.selection_result else None
        final_support = selected_support if selected_support is not None else teacher.active_mask
        
        # Evaluate with selection
        self.structure_eval_result = evaluator.evaluate(
            teacher_support=teacher.active_mask,
            final_support=final_support,
            z_before=teacher.z_scores,
            z_after=teacher.z_scores,  # 아직 augmentation 전이므로 동일
            fragile_pool_mask=teacher.fragile_pool_mask,
            z0=self.config.z0,
            selected_support_pre_aug=selected_support
        )
        
        result = self.structure_eval_result
        print(f"  Spurious Reduction: {result.spurious_reduction:.2%}")
        print(f"  Spurious Retention: {result.spurious_retention:.2%}")
        # augmentation 미적용 시 N/A 출력 (측정 불가)
        print(f"  Promotion Rate: N/A (augmentation not applied)")
        print(f"  Delta-z Median: N/A (augmentation not applied)")
    
    def _save_artifacts(self):
        """산출물 저장 (Modern Schema 적용)"""
        cfg = self.config
        teacher = self.core_mining_result
        selection = self.selection_result
        
        # === STEP 0: numpy arrays 먼저 저장 (hash 계산 위해) ===
        np.save(self.results_dir / 'teacher_support.npy', teacher.active_mask)
        np.save(self.results_dir / 'z_before.npy', teacher.z_scores)
        np.save(self.results_dir / 'stable_core_mask.npy', teacher.stable_core_mask)
        np.save(self.results_dir / 'fragile_pool_mask.npy', teacher.fragile_pool_mask)
        if selection:
            np.save(self.results_dir / 'selected_support_pre_aug.npy', selection.selected_mask)
        print(f"  ✅ Saved: .npy arrays ({5 if selection else 4} files)")
        
        # === STEP 1: teacher_support hash 계산 (MODERNIZE) ===
        teacher_support_path = self.results_dir / 'teacher_support.npy'
        teacher_support_sha256 = compute_file_sha256(teacher_support_path)
        print(f"  ✅ Computed teacher_support_sha256: {teacher_support_sha256[:16]}...")
        
        # === STEP 2: code_snapshot 생성 (MODERNIZE) ===
        _project_root = Path(__file__).resolve().parent.parent
        source_files = [
            Path(__file__),
            _project_root / 'src' / 'sindy' / 'selection.py',
            _project_root / 'src' / 'sindy' / 'core_mining.py',
        ]
        code_hash = create_code_snapshot(self.results_dir, source_files)
        print(f"  ✅ Created code_snapshot/ with {len(code_hash)} files")
        
        # === STEP 3: manifest.json (MODERNIZE 필드 포함) ===
        manifest = {
            'phase': 'phase35',
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'dataset_version': cfg.dataset_version,
            'track': cfg.track,
            'method': 'stable_core',
            'n_train': 10,
            'seed': cfg.seed,
            
            'gate1_artifacts': {
                'teacher_run_id': cfg.teacher_run_id,
                'oracle_run_id': cfg.oracle_run_id
            },
            
            'hyperparameters': {
                'tau_hi': cfg.tau_hi,
                'z0': cfg.z0,
                'eps': cfg.eps
            },
            
            'ssot': {
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'target_names_source': self.target_names_source,
                'n_features': len(self.feature_names),
                'n_targets': len(self.target_names)
            },
            
            'definitions': {
                'z_score': '|mean| / (std + eps)',
                'support': f'inc_prob >= {cfg.tau_hi}',
                'stable_core': f'support AND z >= {cfg.z0}',
                'fragile_pool': f'support AND z < {cfg.z0}'
            },
            
            'stage': {
                'current': 'selection_applied',
                'selection_applied': True,
                'augmentation_applied': False
            },
            
            'selection': {
                'arm': cfg.arm,
                'method': cfg.selection_method,
                'budget': cfg.budget if cfg.arm == 'B' else None,
                'n_selected': self.selection_result.n_selected if self.selection_result else None,
                'n_stable_core_selected': self.selection_result.n_stable_core_selected if self.selection_result else None,
                'n_fragile_selected': self.selection_result.n_fragile_selected if self.selection_result else None,
                'ranking': {
                    'type': 'lexicographic',
                    'keys': ['-inc_prob', '-z_score', '+global_idx'],
                    'description': 'inc_prob descending, z_score descending, global_idx ascending (tie-break)'
                }
            },
            
            # === MODERNIZE FIELDS START ===
            'control_equivalence': get_control_equivalence(),
            'teacher_support_sha256': teacher_support_sha256,
            'preflight_qc': {
                'dx_equivalence': {
                    'dx_key_used': 'inherited_from_gate1',
                    'note': 'Day3 uses Gate1 coefficients directly, no dx recomputation'
                }
            },
            'code_hash': code_hash
            # === MODERNIZE FIELDS END ===
        }
        
        manifest_path = self.results_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: manifest.json (with modern schema)")
        
        # === STEP 4: core_mining.json ===
        self.core_mining_result.save_json(self.results_dir / 'core_mining.json')
        print(f"  ✅ Saved: core_mining.json")
        
        # === STEP 5: selection.json ===
        if self.selection_result:
            selection_data = self.selection_result.to_dict()
            with open(self.results_dir / 'selection.json', 'w', encoding='utf-8') as f:
                json.dump(selection_data, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved: selection.json")
        
        # === STEP 6: structure_eval.json ===
        structure_eval_dict = self.structure_eval_result.to_dict()
        
        # NULL 정책 적용 (augmentation 미적용 시)
        if 'primary_metrics' in structure_eval_dict:
            structure_eval_dict['primary_metrics']['promotion_rate'] = None
            structure_eval_dict['primary_metrics']['delta_z_median'] = None
            structure_eval_dict['primary_metrics']['delta_z_mean'] = None
            if 'spurious_reentry' in structure_eval_dict['primary_metrics']:
                structure_eval_dict['primary_metrics']['spurious_reentry'] = None
        
        if 'details' in structure_eval_dict and 'delta_z_details' in structure_eval_dict['details']:
            structure_eval_dict['details']['delta_z_details'] = []
        
        structure_eval_dict['training_effect_available'] = False
        structure_eval_dict['training_effect_note'] = 'Requires augmentation (Day4+)'
        
        with open(self.results_dir / 'structure_eval.json', 'w', encoding='utf-8') as f:
            json.dump(structure_eval_dict, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: structure_eval.json")
        
        # === STEP 7: pool_metadata.json ===
        pool_metadata = {
            'supports': {
                'teacher_support': {
                    'description': f'inc_prob >= {cfg.tau_hi}',
                    'n_active': teacher.n_active_terms,
                    'mask_file': 'teacher_support.npy'
                },
                'selected_support_pre_aug': {
                    'description': f'Selection method: {cfg.selection_method}',
                    'n_active': selection.n_selected if selection else None,
                    'mask_file': 'selected_support_pre_aug.npy' if selection else None,
                    'arm': cfg.arm,
                    'budget': cfg.budget if cfg.arm == 'B' else None
                },
                'final_support_post_aug_raw': {
                    'description': 'Augmented training 후 final model (re-entry 허용)',
                    'n_active': None,
                    'mask_file': None,
                    'note': 'Augmentation 후 생성 예정'
                },
                'final_support_post_aug_proj': {
                    'description': 'final_raw ∩ selected (하드 제약)',
                    'n_active': None,
                    'mask_file': None,
                    'note': 'Augmentation 후 생성 예정'
                }
            },
            'z_scores': {
                'z_before': {
                    'description': 'Teacher 통계 기반 z-score',
                    'source': 'gate1_coefficients',
                    'file': 'z_before.npy'
                },
                'z_after': {
                    'description': 'Final model 통계 기반 z-score',
                    'source': None,
                    'file': None,
                    'note': 'Augmentation 후 생성 예정'
                }
            },
            'core_mining': {
                'n_stable_core': teacher.n_stable_core,
                'n_fragile_pool': teacher.n_fragile_pool,
                'stable_core_file': 'stable_core_mask.npy',
                'fragile_pool_file': 'fragile_pool_mask.npy'
            }
        }
        
        with open(self.results_dir / 'pool_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(pool_metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: pool_metadata.json")
        
        # === STEP 8: metrics.json ===
        structure_dict = self.structure_eval_result.to_dict()
        
        # NULL 정책 완전 통일 (augmentation 미적용 시)
        if 'primary_metrics' in structure_dict:
            structure_dict['primary_metrics']['promotion_rate'] = None
            structure_dict['primary_metrics']['delta_z_median'] = None
            structure_dict['primary_metrics']['delta_z_mean'] = None
            if 'spurious_reentry' in structure_dict['primary_metrics']:
                structure_dict['primary_metrics']['spurious_reentry'] = None
        
        if 'details' in structure_dict and 'delta_z_details' in structure_dict['details']:
            structure_dict['details']['delta_z_details'] = []
        
        structure_dict['training_effect_available'] = False
        structure_dict['training_effect_note'] = 'Requires augmentation (Day4+)'
        
        metrics = {
            'stage': 'selection_applied',
            'augmentation_applied': False,
            'support_comparison': self.comparison,
            'structure': structure_dict,
            'selection_metrics': {
                'arm': cfg.arm,
                'method': cfg.selection_method,
                'n_selected': selection.n_selected if selection else None,
                'n_stable_core_selected': selection.n_stable_core_selected if selection else None,
                'n_fragile_selected': selection.n_fragile_selected if selection else None,
                'budget': cfg.budget if cfg.arm == 'B' else None
            }
        }
        
        with open(self.results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: metrics.json")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Phase 3.5 Stable-core Selection Experiment'
    )
    
    parser.add_argument(
        '--config', type=str, default=None,
        help='YAML config file path'
    )
    parser.add_argument(
        '--dataset_version', type=str, default='cartpole_ood_v1',
        help='Dataset version'
    )
    parser.add_argument(
        '--teacher_run_id', type=str, required=True,
        help='Gate1 Teacher (n=10) run_id'
    )
    parser.add_argument(
        '--oracle_run_id', type=str, required=True,
        help='Gate1 Oracle (n=50) run_id'
    )
    parser.add_argument(
        '--track', type=str, default='standardized',
        help='Track (standardized or author_recommended)'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--note', type=str, default='core_mining',
        help='Run note'
    )
    # === Day3: Selection 파라미터 ===
    parser.add_argument(
        '--arm', type=str, default='A', choices=['A', 'B'],
        help='Experiment arm: A (stable_core_only) or B (budget_plus_fragile)'
    )
    parser.add_argument(
        '--budget', type=int, default=9,
        help='Max terms to select for arm B (default: 9, oracle n_active)'
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # Config 생성
    if args.config:
        config = Phase35Config.from_yaml(Path(args.config))
    else:
        config = Phase35Config()
    
    # CLI 인자로 override
    cli_overrides = {
        'dataset_version': args.dataset_version,
        'teacher_run_id': args.teacher_run_id,
        'oracle_run_id': args.oracle_run_id,
        'track': args.track,
        'seed': args.seed,
        'note': args.note,
        'arm': args.arm,
        'budget': args.budget
    }
    
    for key, value in cli_overrides.items():
        if value is not None:
            setattr(config, key, value)
    
    # BUGFIX: CLI에서 arm을 override하면 __post_init__이 다시 호출되지 않음
    # selection_method를 arm에 맞게 재설정
    config.__post_init__()
    
    # Runner 실행
    runner = Phase35Runner(config)
    
    try:
        result = runner.run()
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())