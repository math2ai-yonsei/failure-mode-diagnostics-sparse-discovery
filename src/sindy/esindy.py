"""
S07: E-SINDy (Ensemble SINDy) Implementation

Bootstrap-based ensemble learning for robust coefficient estimation
and uncertainty quantification.

Key Design Decisions (per GPT cross-review):
    - Trajectory-level bootstrap (NOT row-level) to preserve temporal correlation
    - Coefficients aggregated in UNSCALED (physical) units
    - Inclusion probability with explicit threshold (abs > 0)

Usage:
    from src.sindy.esindy import ESINDyEnsemble, threshold_sweep
    
    # Basic usage
    ensemble = ESINDyEnsemble(n_bootstrap=20, threshold=0.01)
    ensemble.fit(Theta_scaled, dx, n_trajectories=10, T=101, scaler=scaler)
    
    # Threshold sweep
    results = threshold_sweep(Theta, dx, thresholds, n_traj=10, T=101, ...)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Handle both module import and direct execution
try:
    from src.sindy.optimizer import STLSQOptimizer, ColumnScaler
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from src.sindy.optimizer import STLSQOptimizer, ColumnScaler


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class ESINDyResult:
    """E-SINDy fitting result container."""
    # Aggregated coefficients (UNSCALED, physical units)
    coefficients_mean: np.ndarray      # (n_features, n_targets)
    coefficients_std: np.ndarray       # (n_features, n_targets)
    inclusion_probability: np.ndarray  # (n_features, n_targets)
    
    # Individual bootstrap results (UNSCALED)
    individual_coefficients: List[np.ndarray]  # List of (n_features, n_targets)
    
    # Metadata
    n_bootstrap: int
    threshold: float
    bootstrap_unit: str = 'trajectory'  # 'trajectory' or 'row'


# =============================================================================
# E-SINDy Ensemble Class
# =============================================================================

class ESINDyEnsemble:
    """
    Ensemble-SINDy with trajectory-level bootstrap aggregation.
    
    Algorithm:
        1. Generate n_bootstrap trajectory-level bootstrap samples
        2. Fit STLSQ on each sample
        3. Convert each to unscaled coefficients
        4. Aggregate: mean, std, inclusion probability
    
    Why trajectory bootstrap (not row bootstrap)?
        - Preserves temporal correlation within trajectories
        - Avoids artificial data duplication artifacts
        - Uncertainty reflects data generation process, not sampling noise
    
    Attributes:
        coefficients_mean_: Ensemble mean (UNSCALED, physical units)
        coefficients_std_: Ensemble std (UNSCALED)
        inclusion_probability_: Fraction of models with nonzero coef
    """
    
    def __init__(
        self,
        n_bootstrap: int = 20,
        threshold: float = 0.01,
        max_iter: int = 10,
        ridge_alpha: float = 0.0,
        random_state: Optional[int] = None,
        inclusion_eps: float = 0.0,  # |coef| > eps means "included"
    ):
        """
        Initialize E-SINDy ensemble.
        
        Args:
            n_bootstrap: Number of bootstrap samples (>= 2)
            threshold: STLSQ sparsity threshold
            max_iter: Maximum STLSQ iterations
            ridge_alpha: Ridge regularization
            random_state: Random seed for reproducibility
            inclusion_eps: Threshold for inclusion probability (default: 0)
        """
        if n_bootstrap < 2:
            raise ValueError(f"n_bootstrap must be >= 2, got {n_bootstrap}")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.max_iter = max_iter
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.inclusion_eps = inclusion_eps
        
        # Fitted attributes (all in UNSCALED units)
        self.coefficients_mean_: Optional[np.ndarray] = None
        self.coefficients_std_: Optional[np.ndarray] = None
        self.inclusion_probability_: Optional[np.ndarray] = None
        self._individual_coefficients: List[np.ndarray] = []  # Unscaled
        self._is_fitted = False
    
    def fit(
        self,
        Theta: np.ndarray,
        dx: np.ndarray,
        n_trajectories: int,
        T: int,
        scaler: ColumnScaler,
        target_scale: Optional[np.ndarray] = None,
    ) -> 'ESINDyEnsemble':
        """
        Fit ensemble via trajectory-level bootstrap.
        
        Args:
            Theta: Scaled feature matrix, shape (n_traj * T, n_features)
            dx: Normalized target derivatives, shape (n_traj * T, n_targets)
            n_trajectories: Number of trajectories in data
            T: Time steps per trajectory
            scaler: ColumnScaler used for Theta (for unscaling)
            target_scale: Target (dx) std for unscaling, shape (n_targets,)
                          If None, assumes dx is not normalized
        
        Returns:
            self (for chaining)
        """
        # =============================================
        # Input validation
        # =============================================
        if Theta.ndim != 2:
            raise ValueError(f"Theta must be 2D, got {Theta.ndim}D")
        if dx.ndim == 1:
            dx = dx.reshape(-1, 1)
        if dx.ndim != 2:
            raise ValueError(f"dx must be 1D or 2D, got {dx.ndim}D")
        
        n_samples = Theta.shape[0]
        expected_samples = n_trajectories * T
        
        if n_samples != expected_samples:
            raise ValueError(
                f"Theta has {n_samples} samples, expected {n_trajectories} * {T} = {expected_samples}"
            )
        if dx.shape[0] != n_samples:
            raise ValueError(
                f"Sample count mismatch: Theta={n_samples}, dx={dx.shape[0]}"
            )
        
        if not np.isfinite(Theta).all():
            raise ValueError("Theta contains NaN or Inf")
        if not np.isfinite(dx).all():
            raise ValueError("dx contains NaN or Inf")
        
        n_features = Theta.shape[1]
        n_targets = dx.shape[1]
        
        # =============================================
        # Trajectory-level bootstrap
        # =============================================
        rng = np.random.default_rng(self.random_state)
        self._individual_coefficients = []
        
        for b in range(self.n_bootstrap):
            # Bootstrap: sample trajectory indices with replacement
            traj_idx = rng.choice(n_trajectories, n_trajectories, replace=True)
            
            # Convert trajectory indices to row indices
            row_indices = []
            for ti in traj_idx:
                start = ti * T
                end = (ti + 1) * T
                row_indices.extend(range(start, end))
            row_indices = np.array(row_indices)
            
            Theta_boot = Theta[row_indices]
            dx_boot = dx[row_indices]
            
            # Fit STLSQ
            optimizer = STLSQOptimizer(
                threshold=self.threshold,
                max_iter=self.max_iter,
                ridge_alpha=self.ridge_alpha,
            )
            optimizer.fit(Theta_boot, dx_boot)
            
            # Convert to unscaled coefficients immediately
            coef_unscaled = optimizer.get_unscaled_coefficients(
                scaler, target_scale
            )
            self._individual_coefficients.append(coef_unscaled)
        
        # =============================================
        # Aggregate (all in unscaled units)
        # =============================================
        coef_stack = np.stack(self._individual_coefficients, axis=0)  # (B, F, T)
        
        self.coefficients_mean_ = np.mean(coef_stack, axis=0)
        self.coefficients_std_ = np.std(coef_stack, axis=0)
        
        # Inclusion probability: fraction of bootstrap samples with |coef| > eps
        self.inclusion_probability_ = np.mean(
            np.abs(coef_stack) > self.inclusion_eps, axis=0
        )
        
        self._is_fitted = True
        return self
    
    def predict(
        self,
        Theta: np.ndarray,
        scaler: ColumnScaler,
        target_scale: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict using ensemble mean coefficients.
        
        Note: Since coefficients_mean_ is in unscaled units,
              we need to reverse the scaling for prediction.
        
        Args:
            Theta: Scaled feature matrix
            scaler: ColumnScaler used for Theta
            target_scale: Target std if dx was normalized
        
        Returns:
            Predictions in NORMALIZED units (same scale as input dx)
        """
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        # Convert mean coefficients back to scaled units for prediction
        scale_Theta = scaler.get_scale_factors()['scale']
        
        # Reverse: coef_unscaled = coef_scaled / scale_Theta * target_scale
        # So: coef_scaled = coef_unscaled * scale_Theta / target_scale
        coef_scaled = self.coefficients_mean_ * scale_Theta[:, np.newaxis]
        
        if target_scale is not None:
            coef_scaled = coef_scaled / target_scale[np.newaxis, :]
        
        return Theta @ coef_scaled
    
    def get_result(self) -> ESINDyResult:
        """Return structured result object."""
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        return ESINDyResult(
            coefficients_mean=self.coefficients_mean_.copy(),
            coefficients_std=self.coefficients_std_.copy(),
            inclusion_probability=self.inclusion_probability_.copy(),
            individual_coefficients=[c.copy() for c in self._individual_coefficients],
            n_bootstrap=self.n_bootstrap,
            threshold=self.threshold,
            bootstrap_unit='trajectory',
        )
    
    def get_sparsity_info(self, inclusion_threshold: float = 0.5) -> Dict:
        """
        Get sparsity statistics based on inclusion probability.
        
        Args:
            inclusion_threshold: Terms with inclusion_prob > this are "active"
        
        Returns:
            Dict with sparsity metrics
        """
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        n_features, n_targets = self.coefficients_mean_.shape
        total = n_features * n_targets
        
        # Nonzero based on inclusion probability
        active_mask = self.inclusion_probability_ > inclusion_threshold
        n_active = int(np.sum(active_mask))
        
        return {
            'n_features': n_features,
            'n_targets': n_targets,
            'n_total': total,
            'n_active': n_active,
            'n_zero': total - n_active,
            'sparsity': float(1 - n_active / total),
            'n_bootstrap': self.n_bootstrap,
            'threshold': self.threshold,
            'inclusion_threshold': inclusion_threshold,
            'mean_coef_std': float(np.mean(self.coefficients_std_)),
        }
    
    def get_active_terms(
        self,
        feature_names: List[str],
        inclusion_threshold: float = 0.5,
    ) -> Dict[int, List[Tuple[str, float, float]]]:
        """
        Get active terms per target with uncertainty info.
        
        Returns:
            Dict mapping target index to list of (name, coef_mean, coef_std)
        """
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        active_terms = {}
        n_targets = self.coefficients_mean_.shape[1]
        
        for j in range(n_targets):
            active_idx = np.where(self.inclusion_probability_[:, j] > inclusion_threshold)[0]
            terms = []
            for i in active_idx:
                terms.append((
                    feature_names[i],
                    float(self.coefficients_mean_[i, j]),
                    float(self.coefficients_std_[i, j]),
                ))
            active_terms[j] = terms
        
        return active_terms


# =============================================================================
# Threshold Sweep
# =============================================================================

def threshold_sweep(
    Theta_train: np.ndarray,
    dx_train: np.ndarray,
    Theta_val: np.ndarray,
    dx_val: np.ndarray,
    thresholds: List[float],
    n_trajectories_train: int,
    n_trajectories_val: int,
    T: int,
    scaler: ColumnScaler,
    target_scale: Optional[np.ndarray] = None,
    n_bootstrap: int = 20,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """
    Run E-SINDy across multiple thresholds with train/val evaluation.
    
    Args:
        Theta_train: Scaled train features (n_train * T, F)
        dx_train: Normalized train targets (n_train * T, D)
        Theta_val: Scaled val features (n_val * T, F)
        dx_val: Normalized val targets (n_val * T, D)
        thresholds: List of threshold values
        n_trajectories_train: Number of train trajectories
        n_trajectories_val: Number of val trajectories
        T: Time steps per trajectory
        scaler: ColumnScaler for unscaling
        target_scale: Target std for unscaling
        n_bootstrap: Bootstrap ensemble size
        random_state: Random seed
    
    Returns:
        List of result dicts per threshold
    """
    results = []
    
    for thresh in thresholds:
        ensemble = ESINDyEnsemble(
            n_bootstrap=n_bootstrap,
            threshold=thresh,
            random_state=random_state,
        )
        ensemble.fit(
            Theta_train, dx_train,
            n_trajectories=n_trajectories_train,
            T=T,
            scaler=scaler,
            target_scale=target_scale,
        )
        
        # Train R²
        dx_pred_train = ensemble.predict(Theta_train, scaler, target_scale)
        r2_train = _compute_r2(dx_train, dx_pred_train)
        
        # Val R²
        dx_pred_val = ensemble.predict(Theta_val, scaler, target_scale)
        r2_val = _compute_r2(dx_val, dx_pred_val)
        
        sparsity_info = ensemble.get_sparsity_info()
        
        results.append({
            'threshold': thresh,
            'train_r2_mean': float(np.mean(r2_train)),
            'train_r2_per_target': r2_train.tolist(),
            'val_r2_mean': float(np.mean(r2_val)),
            'val_r2_per_target': r2_val.tolist(),
            'sparsity': sparsity_info['sparsity'],
            'n_active': sparsity_info['n_active'],
            'mean_coef_std': sparsity_info['mean_coef_std'],
        })
    
    return results


def select_best_threshold(
    sweep_results: List[Dict],
    metric: str = 'val_r2_mean',
    tie_break_sparsity: bool = True,
    tie_break_uncertainty: bool = True,
    tie_tolerance: float = 0.001,
) -> Dict:
    """
    Select best threshold from sweep results.
    
    Selection criteria (in order):
        1. Maximize val R² (primary)
        2. If tied (within tolerance): prefer higher sparsity
        3. If still tied: prefer lower uncertainty (mean_coef_std)
    
    Args:
        sweep_results: Results from threshold_sweep()
        metric: Primary metric to maximize
        tie_break_sparsity: Use sparsity as first tie-breaker
        tie_break_uncertainty: Use uncertainty as second tie-breaker
        tie_tolerance: Tolerance for considering metrics "tied"
    
    Returns:
        Copy of best result dict with 'best_reason' field added
    """
    if not sweep_results:
        raise ValueError("sweep_results is empty")
    
    # Sort by primary metric (descending)
    sorted_results = sorted(
        sweep_results,
        key=lambda x: x[metric],
        reverse=True
    )
    
    best = sorted_results[0].copy()  # Copy to avoid modifying original
    best_metric = best[metric]
    
    # Find candidates within tolerance
    candidates = [
        r.copy() for r in sorted_results
        if abs(r[metric] - best_metric) <= tie_tolerance
    ]
    
    if len(candidates) > 1 and tie_break_sparsity:
        # Prefer higher sparsity (more sparse)
        candidates = sorted(candidates, key=lambda x: x['sparsity'], reverse=True)
        best = candidates[0]
        
        # Further tie-break by uncertainty
        best_sparsity = best['sparsity']
        sparsity_tied = [
            c for c in candidates
            if abs(c['sparsity'] - best_sparsity) <= 0.01
        ]
        
        if len(sparsity_tied) > 1 and tie_break_uncertainty:
            # Prefer lower uncertainty
            sparsity_tied = sorted(sparsity_tied, key=lambda x: x['mean_coef_std'])
            best = sparsity_tied[0]
            best['best_reason'] = f'{metric} + sparsity + low_uncertainty'
        else:
            best['best_reason'] = f'{metric} + sparsity'
    else:
        best['best_reason'] = metric
    
    return best


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute R² per target."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1 - ss_res / ss_tot


# =============================================================================
# I/O Helpers
# =============================================================================

def save_coefficients_std_csv(
    coefficients_std: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    save_path,
) -> None:
    """Save coefficient std to CSV."""
    from pathlib import Path
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        header = 'term_name,' + ','.join(target_names)
        f.write(header + '\n')
        
        for i, name in enumerate(feature_names):
            row = ','.join(f'{v:.8f}' for v in coefficients_std[i, :])
            f.write(f'{name},{row}\n')
    
    print(f"  ✅ Saved: {save_path}")


def save_inclusion_prob_csv(
    inclusion_prob: np.ndarray,
    feature_names: List[str],
    target_names: List[str],
    save_path,
) -> None:
    """Save inclusion probability to CSV."""
    from pathlib import Path
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        header = 'term_name,' + ','.join(target_names)
        f.write(header + '\n')
        
        for i, name in enumerate(feature_names):
            row = ','.join(f'{v:.4f}' for v in inclusion_prob[i, :])
            f.write(f'{name},{row}\n')
    
    print(f"  ✅ Saved: {save_path}")


def save_threshold_sweep_csv(
    sweep_results: List[Dict],
    save_path,
) -> None:
    """Save threshold sweep results to CSV."""
    from pathlib import Path
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    columns = [
        'threshold', 'train_r2_mean', 'val_r2_mean',
        'sparsity', 'n_active', 'mean_coef_std'
    ]
    
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(columns) + '\n')
        
        for r in sweep_results:
            row = [
                f"{r['threshold']:.6f}",
                f"{r['train_r2_mean']:.6f}",
                f"{r['val_r2_mean']:.6f}",
                f"{r['sparsity']:.4f}",
                f"{r['n_active']}",
                f"{r['mean_coef_std']:.6f}",
            ]
            f.write(','.join(row) + '\n')
    
    print(f"  ✅ Saved: {save_path}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  E-SINDy Test (Trajectory Bootstrap)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 1. Create synthetic trajectory data
    print("\n[1. Synthetic Trajectory Data]")
    n_traj, T, n_features, n_targets = 10, 50, 8, 4
    
    # True sparse coefficients
    true_coef = np.zeros((n_features, n_targets))
    true_coef[0, :] = [0.5, -0.3, 0.2, 0.1]   # constant
    true_coef[1, :] = [1.0, -0.5, 0.0, 0.3]   # feature 1
    true_coef[3, :] = [0.0, 0.8, -1.0, 0.0]   # feature 3
    
    # Generate data as (n_traj * T, F)
    Theta_raw = np.random.randn(n_traj * T, n_features)
    Theta_raw[:, 0] = 1.0  # Constant column
    dx = Theta_raw @ true_coef + 0.05 * np.random.randn(n_traj * T, n_targets)
    
    print(f"  n_traj: {n_traj}, T: {T}")
    print(f"  Theta shape: {Theta_raw.shape}")
    print(f"  True nonzero: {np.sum(true_coef != 0)}")
    
    # 2. Scale features
    print("\n[2. Column Scaling]")
    scaler = ColumnScaler()
    Theta_scaled = scaler.fit_transform(Theta_raw)
    print(f"  Scaled Theta shape: {Theta_scaled.shape}")
    
    # 3. Fit E-SINDy with trajectory bootstrap
    print("\n[3. E-SINDy Fit (Trajectory Bootstrap)]")
    ensemble = ESINDyEnsemble(
        n_bootstrap=20,
        threshold=0.05,
        random_state=42,
    )
    ensemble.fit(
        Theta_scaled, dx,
        n_trajectories=n_traj,
        T=T,
        scaler=scaler,
        target_scale=None,  # dx not normalized in this test
    )
    
    sparsity = ensemble.get_sparsity_info()
    print(f"  n_bootstrap: {sparsity['n_bootstrap']}")
    print(f"  Active terms: {sparsity['n_active']} / {sparsity['n_total']}")
    print(f"  Sparsity: {sparsity['sparsity']:.1%}")
    print(f"  Mean coef std: {sparsity['mean_coef_std']:.6f}")
    
    # 4. Coefficient recovery
    print("\n[4. Coefficient Recovery]")
    coef_error = np.abs(true_coef - ensemble.coefficients_mean_)
    print(f"  Max error: {coef_error.max():.6f}")
    print(f"  Mean error: {coef_error.mean():.6f}")
    
    # Check support recovery
    true_support = true_coef != 0
    pred_support = ensemble.inclusion_probability_ > 0.5
    support_acc = np.mean(true_support == pred_support)
    print(f"  Support accuracy: {support_acc:.1%}")
    
    # 5. Threshold sweep
    print("\n[5. Threshold Sweep]")
    thresholds = [0, 0.01, 0.05, 0.1]
    
    # Split data for sweep test
    n_train = 8
    n_val = 2
    train_end = n_train * T
    
    sweep_results = threshold_sweep(
        Theta_train=Theta_scaled[:train_end],
        dx_train=dx[:train_end],
        Theta_val=Theta_scaled[train_end:],
        dx_val=dx[train_end:],
        thresholds=thresholds,
        n_trajectories_train=n_train,
        n_trajectories_val=n_val,
        T=T,
        scaler=scaler,
        target_scale=None,
        n_bootstrap=10,
        random_state=42,
    )
    
    print(f"  {'Threshold':>10} {'Train R²':>10} {'Val R²':>10} {'Sparsity':>10}")
    for r in sweep_results:
        print(f"  {r['threshold']:>10.4f} {r['train_r2_mean']:>10.4f} {r['val_r2_mean']:>10.4f} {r['sparsity']:>10.1%}")
    
    # 6. Best threshold selection
    print("\n[6. Best Threshold Selection]")
    best = select_best_threshold(sweep_results)
    print(f"  Best threshold: {best['threshold']}")
    print(f"  Reason: {best['best_reason']}")
    print(f"  Val R²: {best['val_r2_mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("  ✅ E-SINDy Test Complete")
    print("=" * 60)