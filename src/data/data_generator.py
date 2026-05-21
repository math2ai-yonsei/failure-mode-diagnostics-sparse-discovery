"""
S02: Data Generator with Condition-Separated Split

Generates OOD evaluation dataset for Cart-Pole system.
- Train/Val: lighter masses (in-distribution)
- Test: heavier masses (out-of-distribution)

Usage:
    python src/data/data_generator.py --version cartpole_ood_v1
"""
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from itertools import product

# Project imports
from src.contracts import paths
from src.simulators import CartPoleSimulator
from src.utils.seed_utils import set_global_seed


# =============================================================================
# Controller Functions
# =============================================================================

def make_zero_controller() -> Callable:
    """Zero input controller."""
    def controller(t: float, state: np.ndarray) -> float:
        return 0.0
    return controller


def make_random_smooth_controller(
    amplitude: float,
    frequency: float,
    duration: float,
    dt: float,
    seed: Optional[int] = None
) -> Callable:
    """
    Random smooth controller using filtered noise.
    
    Pre-generates control signal for entire trajectory to ensure
    consistency between integration and post-hoc recording.
    """
    rng = np.random.RandomState(seed)
    
    # Generate time array
    n_steps = int(duration / dt) + 1
    t_array = np.linspace(0, duration, n_steps)
    
    # Generate random signal
    raw_signal = rng.randn(n_steps) * amplitude
    
    # Apply low-pass filter via moving average (simple smoothing)
    window_size = max(1, int(1.0 / (frequency * dt)))
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        smooth_signal = np.convolve(raw_signal, kernel, mode='same')
    else:
        smooth_signal = raw_signal
    
    # Clip to amplitude bounds
    smooth_signal = np.clip(smooth_signal, -amplitude, amplitude)
    
    def controller(t: float, state: np.ndarray) -> float:
        # Find closest time index
        idx = int(round(t / dt))
        idx = np.clip(idx, 0, len(smooth_signal) - 1)
        return float(smooth_signal[idx])
    
    return controller


def make_sinusoidal_controller(
    amplitude: float,
    frequency: float,
    phase: float = 0.0
) -> Callable:
    """Sinusoidal input controller."""
    def controller(t: float, state: np.ndarray) -> float:
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return controller


# =============================================================================
# Data Generator Class
# =============================================================================

class CartPoleDataGenerator:
    """
    Generates Cart-Pole trajectories with condition-separated splits.
    
    Key features:
    - OOD evaluation: train/val vs test have different parameter ranges
    - Quality filtering: rejects trajectories that exceed bounds
    - Analytic dx computation from dynamics
    - Metadata tracking for reproducibility
    """
    
    def __init__(self, config_path: Path):
        """
        Args:
            config_path: Path to cartpole.yaml configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Extract settings
        self.physics = self.config['physics']
        self.sim_config = self.config['simulation']
        self.init_state_config = self.config['initial_state']
        self.controller_config = self.config['controller']
        self.conditions = self.config['conditions']
        self.gen_config = self.config['data_generation']
        
        # Quality filter thresholds
        self.quality_filters = self.gen_config['quality_filters']
        
        # Statistics tracking
        self.stats = {
            'total_attempts': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'rejection_reasons': {}
        }
    
    def _load_config(self) -> Dict:
        """Load YAML configuration."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_controller(self, seed: int) -> Callable:
        """Create controller based on configuration."""
        ctrl_type = self.controller_config['type']
        
        if ctrl_type == 'zero':
            return make_zero_controller()
        
        elif ctrl_type == 'random_smooth':
            params = self.controller_config['random_smooth']
            return make_random_smooth_controller(
                amplitude=params['amplitude'],
                frequency=params['frequency'],
                duration=self.sim_config['duration'],
                dt=self.sim_config['dt'],
                seed=seed
            )
        
        elif ctrl_type == 'sinusoidal':
            params = self.controller_config.get('sinusoidal', {})
            return make_sinusoidal_controller(
                amplitude=params.get('amplitude', 5.0),
                frequency=params.get('frequency', 1.0),
                phase=params.get('phase', 0.0)
            )
        
        else:
            raise ValueError(f"Unknown controller type: {ctrl_type}")
    
    def _sample_initial_state(self, rng: np.random.RandomState) -> np.ndarray:
        """Sample random initial state within configured bounds."""
        cfg = self.init_state_config
        return np.array([
            rng.uniform(cfg['x']['min'], cfg['x']['max']),
            rng.uniform(cfg['x_dot']['min'], cfg['x_dot']['max']),
            rng.uniform(cfg['theta']['min'], cfg['theta']['max']),
            rng.uniform(cfg['theta_dot']['min'], cfg['theta_dot']['max'])
        ], dtype=np.float64)
    
    def _check_quality(self, x: np.ndarray, u: np.ndarray) -> Tuple[bool, str]:
        """
        Check if trajectory passes quality filters.
        
        Args:
            x: State trajectory (T, 4)
            u: Input trajectory (T, 1)
        
        Returns:
            (passed, reason): True if passed, reason string if failed
        """
        # Check cart position
        if np.abs(x[:, 0]).max() > self.quality_filters['max_x']:
            return False, 'max_x_exceeded'
        
        # Check pole angle
        if np.abs(x[:, 2]).max() > self.quality_filters['max_theta']:
            return False, 'max_theta_exceeded'
        
        # Check velocities
        if np.abs(x[:, 1]).max() > self.quality_filters['max_velocity']:
            return False, 'max_x_dot_exceeded'
        
        if np.abs(x[:, 3]).max() > self.quality_filters['max_velocity']:
            return False, 'max_theta_dot_exceeded'
        
        # Check for NaN
        if np.isnan(x).any() or np.isnan(u).any():
            return False, 'nan_detected'
        
        return True, 'passed'
    
    def _compute_dx_analytic(
        self, 
        sim: CartPoleSimulator, 
        x: np.ndarray, 
        u: np.ndarray
    ) -> np.ndarray:
        """
        Compute analytic state derivatives using dynamics.
        
        Args:
            sim: Simulator instance with correct parameters
            x: State trajectory (T, 4)
            u: Input trajectory (T, 1)
        
        Returns:
            dx: Derivative trajectory (T, 4)
        """
        T = x.shape[0]
        dx = np.zeros_like(x)
        
        for t_idx in range(T):
            state = x[t_idx]
            input_force = u[t_idx, 0]
            # dynamics() returns [x_dot, x_ddot, theta_dot, theta_ddot]
            dx[t_idx] = sim.dynamics(0.0, state, input_force)
        
        return dx
    
    def _generate_single_trajectory(
        self,
        m_cart: float,
        m_pole: float,
        seed: int
    ) -> Optional[Dict]:
        """
        Generate a single trajectory with given parameters.
        
        Args:
            m_cart: Cart mass
            m_pole: Pole mass
            seed: Random seed for this trajectory
        
        Returns:
            Dict with x, u, dx, params or None if all attempts failed
        """
        rng = np.random.RandomState(seed)
        max_attempts = self.quality_filters['max_attempts']
        
        # Create simulator with this condition's parameters
        sim_params = {
            'm_cart': m_cart,
            'm_pole': m_pole,
            'L': self.physics['L'],
            'g': self.physics['g'],
            'b_cart': self.physics['b_cart'],
            'b_pole': self.physics['b_pole']
        }
        sim = CartPoleSimulator(params=sim_params)
        
        for attempt in range(max_attempts):
            self.stats['total_attempts'] += 1
            
            # Sample initial state
            x0 = self._sample_initial_state(rng)
            
            # Create controller (new seed for each attempt)
            ctrl_seed = seed * 1000 + attempt
            controller = self._create_controller(ctrl_seed)
            
            # Simulate
            t_span = (0.0, self.sim_config['duration'])
            t, x, u = sim.simulate(
                x0=x0,
                t_span=t_span,
                dt=self.sim_config['dt'],
                controller=controller
            )
            
            # Quality check
            passed, reason = self._check_quality(x, u)
            
            if passed:
                self.stats['total_accepted'] += 1
                
                # Compute analytic derivatives
                dx = self._compute_dx_analytic(sim, x, u)
                
                return {
                    'x': x,       # (T, 4)
                    'u': u,       # (T, 1)
                    'dx': dx,     # (T, 4)
                    'params': np.array([m_cart, m_pole], dtype=np.float64)
                }
            else:
                self.stats['total_rejected'] += 1
                self.stats['rejection_reasons'][reason] = \
                    self.stats['rejection_reasons'].get(reason, 0) + 1
        
        # All attempts failed
        return None
    
    def generate_split(
        self,
        split_name: str,
        conditions: Dict[str, List[float]],
        n_traj_per_condition: int,
        base_seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trajectories for a split (train/val/test).
        
        Args:
            split_name: Name of split for logging
            conditions: Dict with 'm_cart' and 'm_pole' lists
            n_traj_per_condition: Number of trajectories per condition
            base_seed: Base random seed
        
        Returns:
            x: (N, T, 4)
            u: (N, T, 1)
            dx: (N, T, 4)
            params: (N, 2)
            cond_id: (N,)
        """
        print(f"\n  Generating {split_name} split...")
        
        m_carts = conditions['m_cart']
        m_poles = conditions['m_pole']
        
        # All condition combinations
        condition_combos = list(product(m_carts, m_poles))
        n_conditions = len(condition_combos)
        
        print(f"    Conditions: {n_conditions} combinations")
        print(f"    Trajectories per condition: {n_traj_per_condition}")
        
        all_x, all_u, all_dx, all_params, all_cond_id = [], [], [], [], []
        
        for cond_idx, (m_cart, m_pole) in enumerate(condition_combos):
            for traj_idx in range(n_traj_per_condition):
                # Unique seed for each trajectory
                seed = base_seed + cond_idx * 10000 + traj_idx
                
                result = self._generate_single_trajectory(m_cart, m_pole, seed)
                
                if result is not None:
                    all_x.append(result['x'])
                    all_u.append(result['u'])
                    all_dx.append(result['dx'])
                    all_params.append(result['params'])
                    all_cond_id.append(cond_idx)
        
        print(f"    Generated: {len(all_x)} trajectories")
        
        # Defense: check for empty results
        if len(all_x) == 0:
            raise RuntimeError(
                f"No trajectories generated for {split_name} split! "
                f"Stats: {self.stats}. "
                f"Consider relaxing quality_filters or checking simulator stability."
            )
        
        # Stack arrays
        x = np.stack(all_x, axis=0)      # (N, T, 4)
        u = np.stack(all_u, axis=0)      # (N, T, 1)
        dx = np.stack(all_dx, axis=0)    # (N, T, 4)
        params = np.stack(all_params, axis=0)  # (N, 2)
        cond_id = np.array(all_cond_id, dtype=np.int64)  # (N,)
        
        print(f"    Generated: {len(x)} trajectories")
        
        return x, u, dx, params, cond_id
    
    def generate_dataset(self, version: str) -> Dict:
        """
        Generate complete dataset with train/val/test splits.
        
        Args:
            version: Dataset version string (e.g., 'cartpole_ood_v1')
        
        Returns:
            Dict containing all arrays and metadata
        """
        print("=" * 60)
        print(f"  Cart-Pole Dataset Generation: {version}")
        print("=" * 60)
        
        # Set master seed
        master_seed = self.gen_config['master_seed']
        set_global_seed(master_seed)
        
        # Generate each split
        train_x, train_u, train_dx, train_params, train_cond_id = self.generate_split(
            split_name='train',
            conditions=self.conditions['train'],
            n_traj_per_condition=self.gen_config['n_traj_per_condition']['train'],
            base_seed=master_seed
        )
        
        val_x, val_u, val_dx, val_params, val_cond_id = self.generate_split(
            split_name='val',
            conditions=self.conditions['val'],
            n_traj_per_condition=self.gen_config['n_traj_per_condition']['val'],
            base_seed=master_seed + 100000
        )
        
        test_x, test_u, test_dx, test_params, test_cond_id = self.generate_split(
            split_name='test',
            conditions=self.conditions['test'],
            n_traj_per_condition=self.gen_config['n_traj_per_condition']['test'],
            base_seed=master_seed + 200000
        )
        
        # Create time array
        T = self.sim_config['T']
        dt = self.sim_config['dt']
        t = np.linspace(0, self.sim_config['duration'], T)
        
        # Subsample if configured
        n_train_target = self.gen_config['n_trajectories']['train']
        n_val_target = self.gen_config['n_trajectories']['val']
        n_test_target = self.gen_config['n_trajectories']['test']
        
        rng = np.random.RandomState(master_seed + 999)
        
        if len(train_x) > n_train_target:
            idx = rng.choice(len(train_x), n_train_target, replace=False)
            train_x, train_u, train_dx = train_x[idx], train_u[idx], train_dx[idx]
            train_params, train_cond_id = train_params[idx], train_cond_id[idx]
        
        if len(val_x) > n_val_target:
            idx = rng.choice(len(val_x), n_val_target, replace=False)
            val_x, val_u, val_dx = val_x[idx], val_u[idx], val_dx[idx]
            val_params, val_cond_id = val_params[idx], val_cond_id[idx]
        
        if len(test_x) > n_test_target:
            idx = rng.choice(len(test_x), n_test_target, replace=False)
            test_x, test_u, test_dx = test_x[idx], test_u[idx], test_dx[idx]
            test_params, test_cond_id = test_params[idx], test_cond_id[idx]
        
        print(f"\n  Final dataset sizes:")
        print(f"    Train: {len(train_x)} trajectories")
        print(f"    Val:   {len(val_x)} trajectories")
        print(f"    Test:  {len(test_x)} trajectories")
        
        # Compute acceptance rate
        accept_rate = self.stats['total_accepted'] / max(1, self.stats['total_attempts'])
        print(f"\n  Quality filter statistics:")
        print(f"    Acceptance rate: {accept_rate:.1%}")
        print(f"    Rejection reasons: {self.stats['rejection_reasons']}")
        
        # Build dataset dict
        dataset = {
            'train_x': train_x,
            'val_x': val_x,
            'test_x': test_x,
            'train_u': train_u,
            'val_u': val_u,
            'test_u': test_u,
            'train_dx': train_dx,
            'val_dx': val_dx,
            'test_dx': test_dx,
            'train_params': train_params,
            'val_params': val_params,
            'test_params': test_params,
            'train_cond_id': train_cond_id,
            'val_cond_id': val_cond_id,
            'test_cond_id': test_cond_id,
            't': t,
            'dt': np.float64(dt)
        }
        
        # Build metadata
        metadata = {
            'version': version,
            'system': 'cartpole',
            'created_at': datetime.now().isoformat(),
            'config_path': str(self.config_path),
            'physics': self.physics,
            'simulation': self.sim_config,
            'conditions': self.conditions,
            'data_generation': self.gen_config,
            'statistics': {
                'n_train': len(train_x),
                'n_val': len(val_x),
                'n_test': len(test_x),
                'T': T,
                'dt': dt,
                'state_dim': 4,
                'input_dim': 1,
                'param_dim': 2,
                'acceptance_rate': accept_rate,
                'rejection_reasons': self.stats['rejection_reasons']
            },
            'state_definition': self.config['state_definition'],
            'param_definition': self.config['param_definition']
        }
        
        return {'data': dataset, 'meta': metadata}
    
    def save_dataset(self, dataset_dict: Dict, version: str) -> Tuple[Path, Path]:
        """
        Save dataset to files.
        
        Args:
            dataset_dict: Output from generate_dataset()
            version: Dataset version string
        
        Returns:
            (npz_path, meta_path)
        """
        # Get paths from SSOT
        npz_path = paths.get_dataset_path(version, system='cartpole')
        meta_path = paths.get_meta_path(version, system='cartpole')
        
        # Create directory
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save npz
        np.savez_compressed(npz_path, **dataset_dict['data'])
        print(f"\n  ✅ Saved: {npz_path}")
        
        # Save metadata
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict['meta'], f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: {meta_path}")
        
        return npz_path, meta_path


# =============================================================================
# Convenience Function
# =============================================================================

def generate_dataset(
    version: str,
    config_path: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Generate and save Cart-Pole dataset.
    
    Args:
        version: Dataset version (e.g., 'cartpole_ood_v1')
        config_path: Path to config file (default: configs/systems/cartpole.yaml)
    
    Returns:
        (npz_path, meta_path)
    """
    if config_path is None:
        config_path = paths.ROOT / 'configs' / 'systems' / 'cartpole.yaml'
    
    generator = CartPoleDataGenerator(config_path)
    dataset_dict = generator.generate_dataset(version)
    return generator.save_dataset(dataset_dict, version)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Cart-Pole OOD evaluation dataset'
    )
    parser.add_argument(
        '--version', '-v',
        type=str,
        default='cartpole_ood_v1',
        help='Dataset version string (default: cartpole_ood_v1)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file (default: configs/systems/cartpole.yaml)'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    npz_path, meta_path = generate_dataset(args.version, config_path)
    
    print("\n" + "=" * 60)
    print("  Dataset generation complete!")
    print("=" * 60)
    print(f"\n  Files:")
    print(f"    {npz_path}")
    print(f"    {meta_path}")
    
    # Quick validation
    print("\n  Validating with schema_dataset_lite...")
    from src.contracts.schema_dataset_lite import validate_dataset_lite
    validate_dataset_lite(npz_path)


if __name__ == '__main__':
    main()