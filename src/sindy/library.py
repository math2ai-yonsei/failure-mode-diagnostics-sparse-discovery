"""
S04-B: SINDy Library for Cart-Pole System

Provides feature library construction for SINDy regression.
Designed with wrap-safe θ handling (sin/cos instead of θ polynomials).

Usage:
    from src.sindy.library import SINDyLibrary, LIBRARY_CONFIGS
    
    lib = SINDyLibrary(config='gate0_min')
    Theta = lib.fit_transform(x, u)
    
Gate0 Library Design:
    - Constant: 1
    - Linear: x, x_dot, theta_dot, u
    - Trigonometric: sin(θ), cos(θ)
    - Quadratic: x², x·x_dot, x_dot²
    - Input coupling: u·sin(θ), u·cos(θ)
    - NO θ polynomials (wrap boundary issues)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from itertools import combinations_with_replacement


# =============================================================================
# SSOT: Library Configurations (Fixed for Gate0-1)
# =============================================================================

LIBRARY_CONFIGS = {
    'gate0_min': {
        'description': 'Minimal Gate0 baseline (wrap-safe)',
        'poly_degree': 2,
        'use_trig': True,
        'include_u': True,
        'include_u_trig': True,
        'theta_poly': False,  # NO θ polynomials (wrap boundary issues)
    },
    'gate0_full': {
        'description': 'Full Gate0 with cross terms',
        'poly_degree': 2,
        'use_trig': True,
        'include_u': True,
        'include_u_trig': True,
        'theta_poly': False,
        # ============================================================
        # [PLAN B - Gate0 실패 시 즉시 활성화]
        # Cart-Pole EoM에 중요한 3차 물리 항:
        #   - theta_dot^2 * sin(theta)
        #   - theta_dot^2 * cos(theta)
        # 
        # 현재 gate0_min/full은 2차까지만 포함.
        # Gate0에서 rollout 발산 또는 test R² < 0.8 발생 시:
        #   1. _build_library()에 아래 2개 항 추가
        #   2. 또는 custom_config={'include_cubic_trig': True} 옵션 구현
        # 
        # 참고: GPT 교차검토 C급 피드백 (2024-12-29)
        # ============================================================
    },
    'oracle': {
        'description': 'Oracle with analytic derivatives',
        'poly_degree': 2,
        'use_trig': True,
        'include_u': True,
        'include_u_trig': True,
        'theta_poly': False,
        # NOTE: 이 config는 track='author_recommended'와 함께 사용
        # derivative_key='derivative_dx' (analytic derivatives)
        # Library 구조 자체는 gate0_min과 동일
    },
}

# Cart-Pole state indices
STATE_INDICES = {
    'x': 0,
    'x_dot': 1,
    'theta': 2,
    'theta_dot': 3,
}

# Feature names for each state component
STATE_NAMES = ['x', 'x_dot', 'theta', 'theta_dot']


# =============================================================================
# SINDy Library Class
# =============================================================================

class SINDyLibrary:
    """
    SINDy feature library with wrap-safe θ handling.
    
    Constructs feature matrix Θ(x, u) for regression: dx = Θ(x, u) @ ξ
    
    Attributes:
        config_name: Name of library configuration
        feature_names: List of feature names
        n_features: Number of features
    """
    
    def __init__(
        self,
        config: str = 'gate0_min',
        custom_config: Optional[Dict] = None
    ):
        """
        Initialize SINDy library.
        
        Args:
            config: Configuration name from LIBRARY_CONFIGS
            custom_config: Override default config with custom settings
        """
        if config not in LIBRARY_CONFIGS:
            raise ValueError(f"Unknown config: {config}. Available: {list(LIBRARY_CONFIGS.keys())}")
        
        self.config_name = config
        self.config = LIBRARY_CONFIGS[config].copy()
        
        if custom_config:
            self.config.update(custom_config)
        
        # Build feature list
        self.feature_names: List[str] = []
        self._feature_funcs: List[Callable] = []
        self._build_library()
        
        self.n_features = len(self.feature_names)
    
    def _build_library(self) -> None:
        """Build feature library based on configuration."""
        cfg = self.config
        
        # 1. Constant term
        self._add_feature('1', lambda x, u: np.ones(x.shape[0]))
        
        # 2. Linear terms (x, x_dot, theta_dot) - NO theta
        self._add_feature('x', lambda x, u: x[:, 0])
        self._add_feature('x_dot', lambda x, u: x[:, 1])
        self._add_feature('theta_dot', lambda x, u: x[:, 3])
        
        # 3. Trigonometric terms (wrap-safe θ handling)
        if cfg['use_trig']:
            self._add_feature('sin(theta)', lambda x, u: np.sin(x[:, 2]))
            self._add_feature('cos(theta)', lambda x, u: np.cos(x[:, 2]))
        
        # 4. Input term
        if cfg['include_u']:
            self._add_feature('u', lambda x, u: u[:, 0])
        
        # 5. Quadratic terms (excluding θ)
        if cfg['poly_degree'] >= 2:
            # x², x·x_dot, x_dot²
            self._add_feature('x^2', lambda x, u: x[:, 0]**2)
            self._add_feature('x*x_dot', lambda x, u: x[:, 0] * x[:, 1])
            self._add_feature('x_dot^2', lambda x, u: x[:, 1]**2)
            
            # theta_dot quadratic
            self._add_feature('theta_dot^2', lambda x, u: x[:, 3]**2)
            
            # Cross terms with theta_dot
            self._add_feature('x*theta_dot', lambda x, u: x[:, 0] * x[:, 3])
            self._add_feature('x_dot*theta_dot', lambda x, u: x[:, 1] * x[:, 3])
        
        # 6. Trig-linear cross terms
        if cfg['use_trig'] and cfg['poly_degree'] >= 2:
            self._add_feature('x*sin(theta)', lambda x, u: x[:, 0] * np.sin(x[:, 2]))
            self._add_feature('x*cos(theta)', lambda x, u: x[:, 0] * np.cos(x[:, 2]))
            self._add_feature('x_dot*sin(theta)', lambda x, u: x[:, 1] * np.sin(x[:, 2]))
            self._add_feature('x_dot*cos(theta)', lambda x, u: x[:, 1] * np.cos(x[:, 2]))
            self._add_feature('theta_dot*sin(theta)', lambda x, u: x[:, 3] * np.sin(x[:, 2]))
            self._add_feature('theta_dot*cos(theta)', lambda x, u: x[:, 3] * np.cos(x[:, 2]))
        
        # 7. Input coupling with trig
        if cfg['include_u'] and cfg['include_u_trig'] and cfg['use_trig']:
            self._add_feature('u*sin(theta)', lambda x, u: u[:, 0] * np.sin(x[:, 2]))
            self._add_feature('u*cos(theta)', lambda x, u: u[:, 0] * np.cos(x[:, 2]))
        
        # 8. θ polynomials (DISABLED by default for wrap safety)
        if cfg.get('theta_poly', False):
            self._add_feature('theta', lambda x, u: x[:, 2])
            if cfg['poly_degree'] >= 2:
                self._add_feature('theta^2', lambda x, u: x[:, 2]**2)
    
    def _add_feature(self, name: str, func: Callable) -> None:
        """Add a feature to the library."""
        self.feature_names.append(name)
        self._feature_funcs.append(func)
    
    def fit_transform(
        self,
        x: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """
        Compute feature matrix Θ(x, u).
        
        Args:
            x: State array, shape (N, 4) or (N, T, 4)
            u: Input array, shape (N, 1) or (N, T, 1)
        
        Returns:
            Theta: Feature matrix, shape (N, n_features) or (N*T, n_features)
        
        Raises:
            ValueError: If input shapes are invalid or contain NaN/Inf
        """
        # =============================================
        # Fail-fast 입력 검증 (S05 파이프라인 안정성)
        # =============================================
        
        # 1. ndim 검증
        if x.ndim not in (2, 3):
            raise ValueError(
                f"x must be 2D (N, 4) or 3D (N, T, 4), got {x.ndim}D with shape {x.shape}"
            )
        
        if u.ndim != x.ndim:
            raise ValueError(
                f"u.ndim ({u.ndim}) must match x.ndim ({x.ndim})"
            )
        
        # 2. 마지막 차원 검증 (state=4, input=1)
        if x.shape[-1] != 4:
            raise ValueError(
                f"x last dimension must be 4 (state_dim), got {x.shape[-1]}"
            )
        
        if u.shape[-1] != 1:
            raise ValueError(
                f"u last dimension must be 1 (input_dim), got {u.shape[-1]}"
            )
        
        # 3. 앞 차원 정합성 검증
        if x.ndim == 3:
            # (N, T, D) 형태
            if x.shape[:2] != u.shape[:2]:
                raise ValueError(
                    f"x shape {x.shape[:2]} and u shape {u.shape[:2]} "
                    f"must match in (N, T) dimensions"
                )
        else:
            # (N, D) 형태
            if x.shape[0] != u.shape[0]:
                raise ValueError(
                    f"x has {x.shape[0]} samples but u has {u.shape[0]} samples"
                )
        
        # 4. NaN/Inf 검증
        if not np.isfinite(x).all():
            nan_count = np.isnan(x).sum()
            inf_count = np.isinf(x).sum()
            raise ValueError(
                f"x contains invalid values: {nan_count} NaN, {inf_count} Inf"
            )
        
        if not np.isfinite(u).all():
            nan_count = np.isnan(u).sum()
            inf_count = np.isinf(u).sum()
            raise ValueError(
                f"u contains invalid values: {nan_count} NaN, {inf_count} Inf"
            )
        
        # =============================================
        # Feature matrix 계산
        # =============================================
        
        # Flatten if 3D
        if x.ndim == 3:
            N, T, D = x.shape
            x = x.reshape(-1, D)
            u = u.reshape(-1, u.shape[-1])
        
        n_samples = x.shape[0]
        Theta = np.zeros((n_samples, self.n_features), dtype=np.float64)
        
        for i, func in enumerate(self._feature_funcs):
            Theta[:, i] = func(x, u)
        
        return Theta
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def get_config(self) -> Dict:
        """Return library configuration."""
        return {
            'config_name': self.config_name,
            'n_features': self.n_features,
            'feature_names': self.feature_names.copy(),
            **self.config
        }
    
    def __repr__(self) -> str:
        return f"SINDyLibrary(config='{self.config_name}', n_features={self.n_features})"


# =============================================================================
# Convenience Functions
# =============================================================================

def build_library_matrix(
    x: np.ndarray,
    u: np.ndarray,
    config: str = 'gate0_min'
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix with default configuration.
    
    Args:
        x: State array, shape (N, 4) or (N, T, 4)
        u: Input array, shape (N, 1) or (N, T, 1)
        config: Library configuration name
    
    Returns:
        Theta: Feature matrix
        feature_names: List of feature names
    """
    lib = SINDyLibrary(config=config)
    Theta = lib.fit_transform(x, u)
    return Theta, lib.get_feature_names()


def get_derivative_key(track: str) -> str:
    """
    Get derivative key based on track.
    
    Args:
        track: 'standardized' or 'author_recommended'
    
    Returns:
        derivative_key: 'derivative_dx_savgol' or 'derivative_dx'
    """
    if track == 'standardized':
        return 'derivative_dx_savgol'
    elif track == 'author_recommended':
        return 'derivative_dx'
    else:
        raise ValueError(f"Unknown track: {track}. Use 'standardized' or 'author_recommended'")


# =============================================================================
# Manifest Helper
# =============================================================================

def get_library_manifest(
    lib: SINDyLibrary,
    track: str
) -> Dict:
    """
    Generate manifest entry for library configuration.
    
    For recording in manifest.json during experiments.
    
    Args:
        lib: SINDyLibrary instance
        track: Experiment track name
    
    Returns:
        Dict for manifest.json
    """
    return {
        'library_id': lib.config_name,
        'n_features': lib.n_features,
        'feature_names': lib.feature_names,
        'poly_degree': lib.config.get('poly_degree', 2),
        'use_trig': lib.config.get('use_trig', True),
        'include_u': lib.config.get('include_u', True),
        'theta_poly': lib.config.get('theta_poly', False),
        'track': track,
        'derivative_key': get_derivative_key(track),
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SINDy Library Test")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    N, T = 10, 50
    x = np.random.randn(N, T, 4)
    u = np.random.randn(N, T, 1)
    
    # Test library creation
    print("\n[Library Creation]")
    lib = SINDyLibrary(config='gate0_min')
    print(f"  {lib}")
    print(f"  Features: {lib.n_features}")
    
    # Test feature matrix
    print("\n[Feature Matrix]")
    Theta = lib.fit_transform(x, u)
    print(f"  Input shape: x={x.shape}, u={u.shape}")
    print(f"  Output shape: Theta={Theta.shape}")
    print(f"  Expected: ({N*T}, {lib.n_features})")
    
    # Print feature names
    print("\n[Feature Names]")
    for i, name in enumerate(lib.get_feature_names()):
        print(f"  {i:2d}: {name}")
    
    # Test manifest
    print("\n[Manifest Entry]")
    manifest = get_library_manifest(lib, 'standardized')
    for k, v in manifest.items():
        if k != 'feature_names':
            print(f"  {k}: {v}")
    
    # Verify no NaN
    print("\n[Validation]")
    if np.isnan(Theta).any():
        print("  ❌ NaN detected in feature matrix")
    else:
        print("  ✅ No NaN in feature matrix")
    
    # Test derivative key
    print("\n[Derivative Key Mapping]")
    print(f"  standardized → {get_derivative_key('standardized')}")
    print(f"  author_recommended → {get_derivative_key('author_recommended')}")
    
    # Test input validation
    print("\n[Input Validation Test]")
    try:
        bad_x = np.random.randn(10, 3)  # Wrong shape
        bad_u = np.random.randn(10, 1)
        lib.fit_transform(bad_x, bad_u)
    except ValueError as e:
        print(f"  ✅ Caught expected error: {e}")
    
    print("\n" + "=" * 60)
    print("  ✅ SINDy Library Test Complete")
    print("=" * 60)