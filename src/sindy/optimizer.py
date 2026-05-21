"""
S05: STLSQ Optimizer for Cart-Pole System

Implements Sequentially Thresholded Least Squares (STLSQ) for SINDy.
Uses scale-only column normalization (NO mean-centering) to preserve
constant term '1' and simplify coefficient inverse transform.

Usage:
    from src.sindy.optimizer import STLSQOptimizer, ColumnScaler
    
    # 1. Scale feature matrix (train-only fit, scale-only)
    scaler = ColumnScaler()
    scaler.fit(Theta_train)
    Theta_train_scaled = scaler.transform(Theta_train)
    Theta_val_scaled = scaler.transform(Theta_val)
    
    # 2. Fit STLSQ
    optimizer = STLSQOptimizer(threshold=0.01)
    optimizer.fit(Theta_train_scaled, dx_train)
    
    # 3. Get coefficients (scaled and original)
    coeffs_scaled = optimizer.coefficients_
    coeffs_original = optimizer.get_unscaled_coefficients(scaler)

Design Decisions:
    - Scale-only (NO mean-centering): Θ_scaled = Θ / scale
    - Constant columns (std<1e-10): scale=1.0 (preserve '1')
    - Inverse transform: Ξ_orig = Ξ_scaled / scale_Θ * scale_dx
    - No bias correction needed (simplifies Gate0)

Gate0: threshold=0.01 (pipeline verification)
Gate1: threshold grid search [0, 1e-4, ..., 5e-2]
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# =============================================================================
# SSOT: STLSQ Configurations
# =============================================================================

STLSQ_CONFIGS = {
    'gate0': {
        'description': 'Gate0 baseline (single threshold)',
        'threshold': 0.01,
        'max_iter': 10,
        'normalize_columns': True,
    },
    'gate1_grid': {
        'description': 'Gate1 threshold grid search',
        'thresholds': [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2],
        'max_iter': 10,
        'normalize_columns': True,
    },
}


# =============================================================================
# Column Scaler (Scale-Only, NO Mean-Centering)
# =============================================================================

class ColumnScaler:
    """
    Column-wise scaler for feature matrix Θ.
    
    Uses SCALE-ONLY normalization: Θ_scaled = Θ / scale
    This preserves the constant term '1' and simplifies inverse transform.
    
    Why scale-only (not z-score):
        - Constant column '1': (1-1)/1 = 0 (WRONG!) vs 1/1 = 1 (CORRECT)
        - Inverse transform: no bias correction needed
        - Gate0 simplicity: Ξ_orig = Ξ_scaled / scale_Θ * scale_dx
    
    Attributes:
        scale_: Column scales (std), shape (n_features,)
        constant_mask_: Boolean mask for constant columns (scale=1.0)
    """
    
    def __init__(self):
        self.scale_: Optional[np.ndarray] = None
        self.constant_mask_: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def fit(self, Theta: np.ndarray) -> 'ColumnScaler':
        """
        Compute column scales from training data.
        
        Args:
            Theta: Feature matrix, shape (N, n_features)
        
        Returns:
            self (for chaining)
        
        Raises:
            ValueError: If input is invalid
        """
        if Theta.ndim != 2:
            raise ValueError(f"Theta must be 2D, got {Theta.ndim}D")
        
        if not np.isfinite(Theta).all():
            raise ValueError("Theta contains NaN or Inf")
        
        # Scale = std per column
        self.scale_ = np.std(Theta, axis=0)
        
        # Constant columns: scale=1.0 (preserve original values)
        # This handles '1' column and any other constant features
        self.constant_mask_ = self.scale_ < 1e-10
        self.scale_[self.constant_mask_] = 1.0
        
        self._is_fitted = True
        return self
    
    def transform(self, Theta: np.ndarray) -> np.ndarray:
        """
        Apply scale-only normalization: Θ / scale
        
        Args:
            Theta: Feature matrix, shape (N, n_features)
        
        Returns:
            Scaled feature matrix, same shape
        """
        if not self._is_fitted:
            raise ValueError("ColumnScaler not fitted. Call fit() first.")
        
        if Theta.ndim != 2:
            raise ValueError(f"Theta must be 2D, got {Theta.ndim}D")
        
        if Theta.shape[1] != len(self.scale_):
            raise ValueError(
                f"Theta has {Theta.shape[1]} columns, expected {len(self.scale_)}"
            )
        
        return Theta / self.scale_
    
    def fit_transform(self, Theta: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(Theta).transform(Theta)
    
    def inverse_transform(self, Theta_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse scale: Θ = Θ_scaled * scale
        
        Args:
            Theta_scaled: Scaled feature matrix
        
        Returns:
            Original scale feature matrix
        """
        if not self._is_fitted:
            raise ValueError("ColumnScaler not fitted.")
        
        return Theta_scaled * self.scale_
    
    def get_scale_factors(self) -> Dict[str, np.ndarray]:
        """Return scaling factors for coefficient inverse transform."""
        if not self._is_fitted:
            raise ValueError("ColumnScaler not fitted.")
        
        return {
            'scale': self.scale_.copy(),
            'constant_mask': self.constant_mask_.copy(),
        }


# =============================================================================
# STLSQ Optimizer
# =============================================================================

class STLSQOptimizer:
    """
    Sequentially Thresholded Least Squares optimizer.
    
    Algorithm:
        1. Solve least squares: Ξ = (Θᵀ Θ)⁻¹ Θᵀ dx
        2. Threshold small coefficients: |Ξᵢⱼ| < threshold → 0
        3. Repeat with support mask until convergence
    
    Input requirements:
        - Theta: Should be scaled via ColumnScaler (scale-only)
        - dx: For standardized track, use normalized dx (from norm_stats)
              For author_recommended track, use raw analytic dx
    
    Attributes:
        coefficients_: Fitted coefficients, shape (n_features, n_targets)
        support_mask_: Boolean mask of active terms
        n_iter_: Number of iterations until convergence
    """
    
    def __init__(
        self,
        threshold: float = 0.01,
        max_iter: int = 10,
        ridge_alpha: float = 0.0,
    ):
        """
        Initialize STLSQ optimizer.
        
        Args:
            threshold: Sparsity threshold (|coef| < threshold → 0)
                       threshold=0 means pure OLS (no sparsity)
            max_iter: Maximum iterations for sequential thresholding
            ridge_alpha: Ridge regularization (0 = pure least squares)
        """
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        
        self.threshold = threshold
        self.max_iter = max_iter
        self.ridge_alpha = ridge_alpha
        
        # Fitted attributes
        self.coefficients_: Optional[np.ndarray] = None
        self.support_mask_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self._is_fitted = False
    
    def fit(
        self,
        Theta: np.ndarray,
        dx: np.ndarray,
    ) -> 'STLSQOptimizer':
        """
        Fit STLSQ to find sparse coefficients.
        
        Solves: dx = Θ @ Ξ  (Ξ sparse)
        
        Args:
            Theta: Feature matrix, shape (N, n_features)
                   Should be scaled via ColumnScaler
            dx: Target derivatives, shape (N, n_targets) or (N,)
        
        Returns:
            self (for chaining)
        """
        # =============================================
        # Input validation (fail-fast)
        # =============================================
        if Theta.ndim != 2:
            raise ValueError(f"Theta must be 2D, got {Theta.ndim}D")
        
        if dx.ndim == 1:
            dx = dx.reshape(-1, 1)
        
        if dx.ndim != 2:
            raise ValueError(f"dx must be 1D or 2D, got {dx.ndim}D")
        
        if Theta.shape[0] != dx.shape[0]:
            raise ValueError(
                f"Sample count mismatch: Theta={Theta.shape[0]}, dx={dx.shape[0]}"
            )
        
        if not np.isfinite(Theta).all():
            raise ValueError("Theta contains NaN or Inf")
        
        if not np.isfinite(dx).all():
            raise ValueError("dx contains NaN or Inf")
        
        n_samples, n_features = Theta.shape
        n_targets = dx.shape[1]
        
        # =============================================
        # Special case: threshold=0 (pure OLS, no sparsity)
        # =============================================
        if self.threshold == 0:
            if self.ridge_alpha > 0:
                # Ridge regression via augmented system
                n_features = Theta.shape[1]
                A_aug = np.vstack([Theta, np.sqrt(self.ridge_alpha) * np.eye(n_features)])
                coefficients = np.zeros((n_features, n_targets))
                for j in range(n_targets):
                    b_aug = np.concatenate([dx[:, j], np.zeros(n_features)])
                    coefficients[:, j], _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            else:
                coefficients, _, _, _ = np.linalg.lstsq(Theta, dx, rcond=None)
            
            self.coefficients_ = coefficients
            self.support_mask_ = np.ones((n_features, n_targets), dtype=bool)
            self.n_iter_ = 1
            self._is_fitted = True
            return self
        
        # =============================================
        # STLSQ iteration
        # =============================================
        support_mask = np.ones((n_features, n_targets), dtype=bool)
        coefficients = np.zeros((n_features, n_targets))
        
        for iteration in range(self.max_iter):
            prev_support = support_mask.copy()
            
            for j in range(n_targets):
                active_idx = np.where(support_mask[:, j])[0]
                
                if len(active_idx) == 0:
                    coefficients[:, j] = 0
                    continue
                
                Theta_active = Theta[:, active_idx]
                
                # Least squares with optional ridge
                coef_active = self._solve_lstsq(
                    Theta_active, dx[:, j], self.ridge_alpha
                )
                
                # Place coefficients
                coefficients[:, j] = 0
                coefficients[active_idx, j] = coef_active
                
                # Thresholding
                small_mask = np.abs(coefficients[:, j]) < self.threshold
                coefficients[small_mask, j] = 0
                support_mask[:, j] = ~small_mask
            
            self.n_iter_ = iteration + 1
            
            # Convergence check
            if np.array_equal(support_mask, prev_support):
                break
        
        self.coefficients_ = coefficients
        self.support_mask_ = support_mask
        self._is_fitted = True
        
        return self
    
    def _solve_lstsq(
        self,
        A: np.ndarray,
        b: np.ndarray,
        ridge_alpha: float
    ) -> np.ndarray:
        """
        Solve least squares with optional ridge regularization.
        
        Uses lstsq for stability (handles near-singular cases).
        """
        if ridge_alpha > 0:
            # Augmented system for ridge regression
            # [A; sqrt(α)I] @ x = [b; 0]
            n_features = A.shape[1]
            A_aug = np.vstack([A, np.sqrt(ridge_alpha) * np.eye(n_features)])
            b_aug = np.concatenate([b, np.zeros(n_features)])
            coef, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
        else:
            coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return coef
    
    def predict(self, Theta: np.ndarray) -> np.ndarray:
        """
        Predict derivatives: dx_pred = Θ @ Ξ
        
        Args:
            Theta: Feature matrix, shape (N, n_features)
                   Must be scaled the same way as training data
        
        Returns:
            Predicted derivatives, shape (N, n_targets)
        """
        if not self._is_fitted:
            raise ValueError("Optimizer not fitted. Call fit() first.")
        
        return Theta @ self.coefficients_
    
    def get_unscaled_coefficients(
        self,
        feature_scaler: ColumnScaler,
        target_scale: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Transform coefficients back to original (physical) units.
        
        Scale-only inverse formula:
            Θ_orig = Θ_scaled * scale_Θ
            dx_orig = dx_scaled * scale_dx
            
            dx_orig = Θ_orig @ Ξ_orig
            dx_scaled * scale_dx = (Θ_scaled * scale_Θ) @ Ξ_orig
            dx_scaled = Θ_scaled @ (scale_Θ * Ξ_orig / scale_dx)
            
            Therefore: Ξ_scaled = scale_Θ * Ξ_orig / scale_dx
            Inverse:   Ξ_orig = Ξ_scaled / scale_Θ * scale_dx
        
        Args:
            feature_scaler: ColumnScaler used for Theta
            target_scale: Target (dx) scales, shape (n_targets,)
                          If None, assumes dx was not scaled (raw)
        
        Returns:
            Unscaled coefficients in physical units
        """
        if not self._is_fitted:
            raise ValueError("Optimizer not fitted.")
        
        scale_Theta = feature_scaler.get_scale_factors()['scale']
        
        # Ξ_orig = Ξ_scaled / scale_Θ
        coeffs_unscaled = self.coefficients_ / scale_Theta[:, np.newaxis]
        
        # Apply target scale if dx was normalized
        if target_scale is not None:
            target_scale = np.asarray(target_scale)
            n_targets = self.coefficients_.shape[1]
            if target_scale.shape != (n_targets,):
                raise ValueError(
                    f"target_scale shape {target_scale.shape} does not match "
                    f"n_targets ({n_targets},)"
                )
            coeffs_unscaled = coeffs_unscaled * target_scale[np.newaxis, :]
        
        return coeffs_unscaled
    
    def get_active_terms(
        self,
        feature_names: List[str]
    ) -> Dict[int, List[str]]:
        """
        Get active (non-zero) terms per target.
        
        Args:
            feature_names: Feature names from SINDyLibrary
        
        Returns:
            Dict mapping target index to list of active term names
        """
        if not self._is_fitted:
            raise ValueError("Optimizer not fitted.")
        
        active_terms = {}
        n_targets = self.coefficients_.shape[1]
        
        for j in range(n_targets):
            active_idx = np.where(self.support_mask_[:, j])[0]
            active_terms[j] = [feature_names[i] for i in active_idx]
        
        return active_terms
    
    def get_sparsity_info(self) -> Dict:
        """Get sparsity statistics."""
        if not self._is_fitted:
            raise ValueError("Optimizer not fitted.")
        
        n_features, n_targets = self.coefficients_.shape
        total = n_features * n_targets
        nonzero = int(np.sum(self.support_mask_))
        
        return {
            'n_features': n_features,
            'n_targets': n_targets,
            'n_total': total,
            'n_nonzero': nonzero,
            'n_zero': total - nonzero,
            'sparsity': float(1 - nonzero / total),
            'n_iter': self.n_iter_,
            'threshold': self.threshold,
        }


# =============================================================================
# Coefficient I/O
# =============================================================================

def save_coefficients_csv(
    coefficients: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    save_path: Path,
) -> Path:
    """
    Save coefficients to CSV (Gate0 required artifact).
    
    Format:
        term_name,x_dot,x_ddot,theta_dot,theta_ddot
        1,0.00000000,0.00000000,0.00000000,0.00000000
        x,0.10000000,-0.20000000,0.00000000,0.50000000
        ...
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Shape validation (fail-fast)
    if coefficients.shape[0] != len(feature_names):
        raise ValueError(
            f"coefficients has {coefficients.shape[0]} rows but "
            f"feature_names has {len(feature_names)} elements"
        )
    if coefficients.shape[1] != len(target_names):
        raise ValueError(
            f"coefficients has {coefficients.shape[1]} columns but "
            f"target_names has {len(target_names)} elements"
        )
    
    # newline='' for Windows CSV compatibility
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        # Header: term_name for consistency with Gate conventions
        header = 'term_name,' + ','.join(target_names)
        f.write(header + '\n')
        
        for i, name in enumerate(feature_names):
            row_values = ','.join(f'{v:.8f}' for v in coefficients[i, :])
            f.write(f'{name},{row_values}\n')
    
    print(f"  ✅ Saved: {save_path}")
    return save_path

def load_coefficients_csv(load_path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load coefficients from CSV.
    
    Returns:
        (coefficients, feature_names, target_names)
    """
    load_path = Path(load_path)
    
    with open(load_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split(',')
    target_names = header[1:]
    
    feature_names = []
    coefficients = []
    
    for line in lines[1:]:
        parts = line.strip().split(',')
        feature_names.append(parts[0])
        coefficients.append([float(v) for v in parts[1:]])
    
    return np.array(coefficients), feature_names, target_names


# =============================================================================
# Manifest Helper
# =============================================================================

def get_optimizer_manifest(
    optimizer: STLSQOptimizer,
    scaler: ColumnScaler,
    feature_names: List[str],
) -> Dict:
    """Generate manifest entry for optimizer configuration."""
    sparsity = optimizer.get_sparsity_info()
    active = optimizer.get_active_terms(feature_names)
    
    return {
        'optimizer': 'STLSQ',
        'threshold': optimizer.threshold,
        'max_iter': optimizer.max_iter,
        'ridge_alpha': optimizer.ridge_alpha,
        'n_iter_converged': optimizer.n_iter_,
        'sparsity': sparsity['sparsity'],
        'n_nonzero': sparsity['n_nonzero'],
        'n_total': sparsity['n_total'],
        'active_terms_per_target': {str(k): v for k, v in active.items()},
        'n_constant_columns': int(np.sum(scaler.constant_mask_)),
        'scaler_type': 'scale_only',
    }


# =============================================================================
# Target Names Helper
# =============================================================================

TARGET_NAMES = ['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  STLSQ Optimizer Test (Scale-Only)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 1. Create synthetic data with known sparse structure
    print("\n[1. Synthetic Data]")
    n_samples, n_features, n_targets = 500, 10, 4
    
    # True sparse coefficients
    true_coef = np.zeros((n_features, n_targets))
    true_coef[0, :] = [0.5, -0.3, 0.2, 0.1]   # constant term (MUST be preserved)
    true_coef[1, :] = [1.0, -0.5, 0.0, 0.3]   # feature 1
    true_coef[3, :] = [0.0, 0.8, -1.0, 0.0]   # feature 3
    
    Theta = np.random.randn(n_samples, n_features)
    Theta[:, 0] = 1.0  # Constant column
    dx = Theta @ true_coef + 0.01 * np.random.randn(n_samples, n_targets)
    
    print(f"  Theta shape: {Theta.shape}")
    print(f"  dx shape: {dx.shape}")
    print(f"  True nonzero: {np.sum(true_coef != 0)}")
    
    # 2. Scale-only normalization
    print("\n[2. Column Scaling (Scale-Only)]")
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta)
    
    print(f"  Constant columns: {np.sum(scaler.constant_mask_)}")
    print(f"  Constant col '1' preserved: {np.allclose(Theta_scaled[:, 0], 1.0)}")
    
    # 3. Fit STLSQ
    print("\n[3. STLSQ Fit]")
    optimizer = STLSQOptimizer(threshold=0.05)
    optimizer.fit(Theta_scaled, dx)
    
    sparsity = optimizer.get_sparsity_info()
    print(f"  Iterations: {sparsity['n_iter']}")
    print(f"  Nonzero: {sparsity['n_nonzero']} / {sparsity['n_total']}")
    print(f"  Sparsity: {sparsity['sparsity']:.1%}")
    
    # 4. Prediction quality
    print("\n[4. Prediction]")
    dx_pred = optimizer.predict(Theta_scaled)
    
    r2_per_target = 1 - np.var(dx - dx_pred, axis=0) / np.var(dx, axis=0)
    print(f"  R² per target: {[f'{r:.4f}' for r in r2_per_target]}")
    print(f"  R² mean: {np.mean(r2_per_target):.4f}")
    
    # 5. Coefficient recovery (unscaled)
    print("\n[5. Coefficient Recovery]")
    coeffs_recovered = optimizer.get_unscaled_coefficients(scaler)
    
    # Compare with true (note: slight difference due to noise)
    coef_error = np.abs(true_coef - coeffs_recovered)
    print(f"  Max coefficient error: {coef_error.max():.6f}")
    print(f"  Mean coefficient error: {coef_error.mean():.6f}")
    
    # Support recovery
    true_support = true_coef != 0
    pred_support = optimizer.support_mask_
    support_acc = np.mean(true_support == pred_support)
    print(f"  Support accuracy: {support_acc:.1%}")
    
    # 6. CSV round-trip test
    print("\n[6. CSV Save/Load]")
    import tempfile
    
    feature_names = ['1'] + [f'f{i}' for i in range(1, n_features)]
    target_names = ['dx0', 'dx1', 'dx2', 'dx3']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'coefficients.csv'
        save_coefficients_csv(coeffs_recovered, feature_names, target_names, csv_path)
        
        loaded, feat_loaded, tgt_loaded = load_coefficients_csv(csv_path)
        match = np.allclose(coeffs_recovered, loaded)
        print(f"  Round-trip match: {match}")
    
    # 7. threshold=0 test (pure OLS)
    print("\n[7. threshold=0 (Pure OLS)]")
    ols_opt = STLSQOptimizer(threshold=0)
    ols_opt.fit(Theta_scaled, dx)
    print(f"  All terms active: {np.all(ols_opt.support_mask_)}")
    print(f"  n_iter: {ols_opt.n_iter_}")
    
    print("\n" + "=" * 60)
    print("  ✅ STLSQ Optimizer Test Complete")
    print("=" * 60)