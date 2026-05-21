"""
Gate2: Base Augmentor Abstract Class

Defines the interface for all augmentation methods.
Ensures physics-consistent augmentation with dx-x parity.

Key Design Principles:
    1. Augmentation is train-only (val/test NEVER augmented)
    2. dx must be physically consistent with x (re-simulation required)
    3. Reproducibility via seed-based RNG
    4. Same train subset idx as Gate1 (fair comparison)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class AugmentationResult:
    """Container for augmentation results."""
    
    # Augmented trajectories
    x: np.ndarray           # (n_aug, T, state_dim)
    u: np.ndarray           # (n_aug, T, input_dim)
    dx: np.ndarray          # (n_aug, T, state_dim) - physically consistent
    params: np.ndarray      # (n_aug, n_params)
    
    # Metadata
    n_original: int         # Number of original trajectories
    n_augmented: int        # Number of augmented trajectories
    aug_method: str         # Augmentation method name
    aug_config: Dict        # Augmentation configuration
    
    # Per-trajectory metadata
    source_idx: np.ndarray  # (n_aug,) - index of original trajectory
    aug_type: List[str]     # (n_aug,) - type of augmentation applied
    
    def __post_init__(self):
        """Validate shapes."""
        assert self.x.shape[0] == self.n_augmented
        assert self.u.shape[0] == self.n_augmented
        assert self.dx.shape[0] == self.n_augmented
        assert self.params.shape[0] == self.n_augmented
        assert len(self.source_idx) == self.n_augmented
        assert len(self.aug_type) == self.n_augmented


@dataclass
class AugmentorConfig:
    """Base configuration for augmentors."""
    
    # Augmentation ratio: n_aug = n_original * aug_ratio
    aug_ratio: float = 1.0
    
    # Random seed for reproducibility
    seed: int = 42
    
    # System parameters
    system: str = 'cartpole'
    dt: float = 0.02
    T: int = 101
    
    # Method name (set by subclass)
    method: str = 'base'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for manifest."""
        return {
            'aug_ratio': self.aug_ratio,
            'seed': self.seed,
            'system': self.system,
            'dt': self.dt,
            'T': self.T,
            'method': self.method,
        }


class BaseAugmentor(ABC):
    """
    Abstract base class for physics-consistent augmentation.
    
    Subclasses must implement:
        - _augment_single(): Augment a single trajectory
        - _get_method_config(): Return method-specific config dict
    
    Guarantees:
        - dx is physically consistent with x (via re-simulation)
        - Reproducible with seed
        - train-only augmentation
    """
    
    def __init__(self, config: AugmentorConfig):
        """
        Initialize augmentor.
        
        Args:
            config: Augmentor configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._is_fitted = False
    
    @abstractmethod
    def _augment_single(
        self,
        x: np.ndarray,
        u: np.ndarray,
        params: np.ndarray,
        traj_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Augment a single trajectory.
        
        Args:
            x: Original state trajectory, shape (T, state_dim)
            u: Original input trajectory, shape (T, input_dim)
            params: Original parameters, shape (n_params,)
            traj_idx: Index of trajectory in original dataset
        
        Returns:
            (aug_x, aug_u, aug_dx, aug_params, aug_type)
            - aug_x: Augmented state, shape (T, state_dim)
            - aug_u: Augmented input, shape (T, input_dim)
            - aug_dx: Augmented derivative (physically consistent), shape (T, state_dim)
            - aug_params: Augmented parameters, shape (n_params,)
            - aug_type: Type of augmentation applied (str)
        """
        pass
    
    @abstractmethod
    def _get_method_config(self) -> Dict:
        """Return method-specific configuration for manifest."""
        pass
    
    def augment(
        self,
        x: np.ndarray,
        u: np.ndarray,
        params: np.ndarray,
        n_aug: Optional[int] = None,
    ) -> AugmentationResult:
        """
        Augment training trajectories.
        
        Args:
            x: Original states, shape (n_traj, T, state_dim)
            u: Original inputs, shape (n_traj, T, input_dim)
            params: Original parameters, shape (n_traj, n_params)
            n_aug: Number of augmented trajectories (default: n_traj * aug_ratio)
        
        Returns:
            AugmentationResult containing augmented data
        """
        # Input validation
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (n_traj, T, state_dim), got {x.ndim}D")
        if u.ndim != 3:
            raise ValueError(f"u must be 3D (n_traj, T, input_dim), got {u.ndim}D")
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (n_traj, n_params), got {params.ndim}D")
        
        n_original = x.shape[0]
        T = x.shape[1]
        state_dim = x.shape[2]
        input_dim = u.shape[2]
        n_params = params.shape[1]
        
        # Determine number of augmented trajectories
        if n_aug is None:
            n_aug = int(n_original * self.config.aug_ratio)
        n_aug = max(1, n_aug)  # At least 1
        
        # Pre-allocate arrays
        aug_x = np.zeros((n_aug, T, state_dim), dtype=np.float64)
        aug_u = np.zeros((n_aug, T, input_dim), dtype=np.float64)
        aug_dx = np.zeros((n_aug, T, state_dim), dtype=np.float64)
        aug_params = np.zeros((n_aug, n_params), dtype=np.float64)
        source_idx = np.zeros(n_aug, dtype=np.int64)
        aug_types = []
        
        # Generate augmented trajectories
        for i in range(n_aug):
            # Randomly select source trajectory
            src_idx = self.rng.integers(0, n_original)
            source_idx[i] = src_idx
            
            # Augment single trajectory
            ax, au, adx, ap, atype = self._augment_single(
                x[src_idx], u[src_idx], params[src_idx], src_idx
            )
            
            aug_x[i] = ax
            aug_u[i] = au
            aug_dx[i] = adx
            aug_params[i] = ap
            aug_types.append(atype)
        
        # Build config dict for manifest
        aug_config = self.config.to_dict()
        aug_config.update(self._get_method_config())
        aug_config['n_aug'] = n_aug
        
        return AugmentationResult(
            x=aug_x,
            u=aug_u,
            dx=aug_dx,
            params=aug_params,
            n_original=n_original,
            n_augmented=n_aug,
            aug_method=self.config.method,
            aug_config=aug_config,
            source_idx=source_idx,
            aug_type=aug_types,
        )
    
    def get_manifest_entry(self) -> Dict:
        """Generate manifest entry for this augmentor."""
        config_dict = self.config.to_dict()
        config_dict.update(self._get_method_config())
        return {
            'augmentor': self.__class__.__name__,
            'method': self.config.method,
            'config': config_dict,
        }


def get_train_subset_idx(
    n_total: int,
    n_train: int,
    seed: int,
) -> np.ndarray:
    """
    Get train subset indices using same logic as Gate1.
    
    CRITICAL: This must match Gate1ESINDyRunner._load_data() exactly
    for fair comparison.
    
    Args:
        n_total: Total number of training trajectories in dataset
        n_train: Number of trajectories to use
        seed: Random seed
    
    Returns:
        Sorted array of trajectory indices
    """
    rng = np.random.default_rng(seed)
    n_use = min(n_train, n_total)
    
    if n_use < n_total:
        idx = rng.choice(n_total, n_use, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n_total)
    
    return idx