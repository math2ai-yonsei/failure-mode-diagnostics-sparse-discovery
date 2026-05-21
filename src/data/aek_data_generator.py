"""
AEK Data Generator with Condition-Separated Split

Generates OOD evaluation dataset for AEK Self-balancing Motorcycle system.
- Train/Val: in-distribution I_w_C values
- Test: out-of-distribution I_w_C (heavier flywheel)

OOD parameter: I_w_C (wheel spin inertia)
    When I_w_C changes, simulator recomputes: m_w, I_p, M_total, h_cm

Controller strategy:
    Stabilizing PD + random excitation (physically realistic).
    The real AEK motorcycle uses a balance controller; purely random torque
    cannot stabilize this inverted pendulum (tau_max << gravity torque).
    PD keeps the system upright; excitation provides state diversity for SINDy.

    Sign convention (critical):
        phi_ddot = (M*g*h*sin(phi) - tau) / I_p
        Since tau enters with NEGATIVE sign, stabilizing PD requires:
            tau = +K_p * phi + K_d * phi_dot   (NOT the usual -K convention)
        This produces positive tau when phi > 0, so -tau opposes gravity.

Usage:
    python -m src.data.aek_data_generator --version aek_ood_v1
    python -m src.data.aek_data_generator --version aek_ood_v1 --seed 42
"""
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

# Project imports
from src.contracts import paths
from src.simulators import AEKSimulator
from src.utils.seed_utils import set_global_seed


# =============================================================================
# Controller Functions (AEK-specific: torque input)
# =============================================================================

def make_zero_controller() -> Callable:
    """Zero torque controller."""
    def controller(t: float, state: np.ndarray) -> float:
        return 0.0
    return controller


def make_stabilizing_excitation_controller(
    tau_max: float,
    K_p: float,
    K_d: float,
    excitation_amplitude: float,
    excitation_frequency: float,
    duration: float,
    dt: float,
    seed: Optional[int] = None
) -> Callable:
    """
    Stabilizing PD controller with random excitation for SINDy data generation.

    Physical motivation:
    - The AEK motorcycle is an inverted pendulum with very limited torque
      (tau_max=0.02 N*m vs gravity torque ~0.28 N*m/rad at small angles)
    - Pure random torque cannot stabilize the system
    - A PD controller keeps phi near zero; excitation explores the state space
    - This mimics the real system which has a balance controller

    Sign convention derivation:
        phi_ddot = (M*g*h*sin(phi) - tau) / I_p

        Gravity: +M*g*h*sin(phi) destabilizes (pushes phi away from 0)
        Motor:   -tau in numerator means positive tau REDUCES phi_ddot

        To stabilize:
        - When phi > 0 (leaning right), need tau > 0 to push phi_ddot negative
        - When phi < 0 (leaning left),  need tau < 0 to push phi_ddot positive

        Therefore: tau = +K_p * phi + K_d * phi_dot
        (opposite of standard -K convention because tau enters with minus sign)

    Control law:
        tau = clip(+K_p * phi + K_d * phi_dot + excitation(t), -tau_max, tau_max)

    Args:
        tau_max: Motor torque saturation limit (N*m)
        K_p: Proportional gain for lean angle stabilization
        K_d: Derivative gain for lean rate damping
        excitation_amplitude: Amplitude of random excitation (N*m)
        excitation_frequency: Cutoff frequency for excitation smoothing (Hz)
        duration: Total simulation time (s)
        dt: Time step (s)
        seed: Random seed for excitation signal
    """
    rng = np.random.RandomState(seed)

    # Pre-generate smooth excitation signal
    n_steps = int(duration / dt) + 1
    raw_signal = rng.randn(n_steps) * excitation_amplitude

    # Low-pass filter via moving average
    window_size = max(1, int(1.0 / (excitation_frequency * dt)))
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        excitation_signal = np.convolve(raw_signal, kernel, mode='same')
    else:
        excitation_signal = raw_signal

    # Clip excitation to its own amplitude
    excitation_signal = np.clip(
        excitation_signal, -excitation_amplitude, excitation_amplitude
    )

    def controller(t: float, state: np.ndarray) -> float:
        phi = state[0]
        phi_dot = state[1]

        # PD stabilization (+ sign because tau enters EOM with minus sign)
        # phi_ddot = (M*g*h*sin(phi) - tau) / I_p
        # Positive tau when phi>0 -> -tau reduces phi_ddot -> stabilizing
        tau_pd = K_p * phi + K_d * phi_dot

        # Add excitation for state-space exploration
        idx = int(round(t / dt))
        idx = min(max(idx, 0), len(excitation_signal) - 1)
        tau_exc = excitation_signal[idx]

        # Total torque with motor saturation
        tau = np.clip(tau_pd + tau_exc, -tau_max, tau_max)
        return float(tau)

    return controller


# =============================================================================
# AEK Data Generator Class
# =============================================================================

class AEKDataGenerator:
    """
    Generates AEK Self-balancing Motorcycle trajectories with
    condition-separated splits for OOD evaluation.

    Key features:
    - OOD via I_w_C variation (flywheel swap)
    - Stabilizing PD + excitation controller (physically realistic)
    - Quality filtering: rejects trajectories that exceed physical bounds
    - Analytic dx computation from dynamics (no finite-diff issues)
    - Metadata tracking for reproducibility
    """

    def __init__(self, config_path: Path):
        """
        Args:
            config_path: Path to aek.yaml configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Extract settings from YAML
        self.physics = self.config['physics']
        self.sim_config = self.config['simulation']
        self.ood_config = self.config['ood']

        # Simulation parameters
        self.dt = self.sim_config['dt']
        self.T_steps = self.sim_config['T_steps']
        self.T_duration = self.sim_config['T_duration']

        # Input range
        self.tau_max = self.sim_config['input_range']['tau'][1]  # 0.02

        # -----------------------------------------------------------------
        # Initial condition ranges (tightened for AEK instability)
        # -----------------------------------------------------------------
        # YAML has phi: [-0.15, 0.15] but that's too large for tau_max=0.02.
        # Linearized gravity torque: M*g*h*phi = 0.277*phi N*m
        # At phi=0.15: gravity torque = 0.042 N*m > tau_max=0.02
        # -> controller cannot stabilize beyond phi ~ 0.07 rad
        # We use [-0.03, 0.03] (well within controllable region)
        self.ic_ranges = {
            'phi': [-0.03, 0.03],           # ~1.7 deg (controllable)
            'phi_dot': [-0.5, 0.5],         # reduced from [-1.0, 1.0]
            'theta_w': [-1.0, 1.0],         # wheel angle (no stability issue)
            'theta_w_dot': [-10.0, 10.0],   # wheel speed (no stability issue)
        }

        # -----------------------------------------------------------------
        # Controller config: PD stabilization + random excitation
        # -----------------------------------------------------------------
        # Gain design (linearized at phi=0):
        #   phi_ddot = (M*g*h/I_p)*phi - tau/I_p
        #   With tau = +K_p*phi + K_d*phi_dot:
        #   phi_ddot = (M*g*h - K_p)/I_p * phi - K_d/I_p * phi_dot
        #
        #   M*g*h = 0.3643 * 9.807 * 0.0774 = 0.2765 N*m/rad
        #   Stability requires K_p > M*g*h = 0.277 N*m/rad
        #
        #   K_p = 0.4: (M*g*h - K_p)/I_p = (0.277-0.4)/0.00286 = -43 < 0 (stable!)
        #   K_d = 0.01: damping term = K_d/I_p = 3.49 rad/s
        #
        #   BUT: at phi=0.05, K_p*phi = 0.02 = tau_max (saturation onset)
        #   Beyond phi~0.05 the controller clips and gravity wins.
        #   IC range phi=[-0.03, 0.03] keeps well within linear control region.
        #
        # Excitation: 0.005 N*m (25% of tau_max), 2Hz filtered noise
        self.controller_config = {
            'type': 'stabilizing_excitation',
            'K_p': 0.4,
            'K_d': 0.01,
            'excitation_amplitude': 0.005,   # N*m (25% of tau_max)
            'excitation_frequency': 2.0,     # Hz
        }

        # Quality filter thresholds (AEK-specific)
        self.quality_filters = {
            'max_phi': 0.5,               # ~28 deg (generous)
            'max_phi_dot': 30.0,          # rad/s
            'max_theta_w_dot': 200.0,     # rad/s
            'max_attempts': 30,           # retries per trajectory
        }

        # Dataset split config
        self.n_traj = self.sim_config['n_trajectories']

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

    def _create_simulator(self, I_w_C: float) -> AEKSimulator:
        """
        Create AEK simulator with given I_w_C.
        Simulator recomputes all derived quantities automatically.
        """
        sim_params = {
            'm_r': self.physics['rod']['mass_kg'],
            'R': self.physics['wheel']['radius_m'],
            'r': self.physics['rod']['cross_section_radius_m'],
            'l': self.physics['rod']['length_m'],
            'I_w_C': I_w_C,
            'g': self.physics['g'],
            'tau_max': self.tau_max,
        }
        return AEKSimulator(params=sim_params)

    def _create_controller(self, seed: int) -> Callable:
        """Create stabilizing + excitation controller."""
        cfg = self.controller_config
        return make_stabilizing_excitation_controller(
            tau_max=self.tau_max,
            K_p=cfg['K_p'],
            K_d=cfg['K_d'],
            excitation_amplitude=cfg['excitation_amplitude'],
            excitation_frequency=cfg['excitation_frequency'],
            duration=self.T_duration,
            dt=self.dt,
            seed=seed
        )

    def _sample_initial_state(self, rng: np.random.RandomState) -> np.ndarray:
        """Sample random initial state within controllable bounds."""
        ic = self.ic_ranges
        return np.array([
            rng.uniform(ic['phi'][0], ic['phi'][1]),
            rng.uniform(ic['phi_dot'][0], ic['phi_dot'][1]),
            rng.uniform(ic['theta_w'][0], ic['theta_w'][1]),
            rng.uniform(ic['theta_w_dot'][0], ic['theta_w_dot'][1]),
        ], dtype=np.float64)

    def _check_quality(self, x: np.ndarray, u: np.ndarray) -> Tuple[bool, str]:
        """
        Check if trajectory passes quality filters.

        Args:
            x: State trajectory (T, 4)
            u: Input trajectory (T, 1)

        Returns:
            (passed, reason)
        """
        # Check lean angle (phi)
        if np.abs(x[:, 0]).max() > self.quality_filters['max_phi']:
            return False, 'max_phi_exceeded'

        # Check lean angular velocity
        if np.abs(x[:, 1]).max() > self.quality_filters['max_phi_dot']:
            return False, 'max_phi_dot_exceeded'

        # Check wheel angular velocity
        if np.abs(x[:, 3]).max() > self.quality_filters['max_theta_w_dot']:
            return False, 'max_theta_w_dot_exceeded'

        # Check for NaN/Inf
        if np.isnan(x).any() or np.isnan(u).any():
            return False, 'nan_detected'
        if np.isinf(x).any() or np.isinf(u).any():
            return False, 'inf_detected'

        return True, 'passed'

    def _compute_dx_analytic(
        self,
        sim: AEKSimulator,
        x: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """
        Compute analytic state derivatives using dynamics equations.
        No finite-difference -> no phi unwrap issue (P0-1 satisfied).
        """
        T = x.shape[0]
        dx = np.zeros_like(x)

        for t_idx in range(T):
            state = x[t_idx]
            tau = u[t_idx, 0]
            dx[t_idx] = sim.dynamics(0.0, state, tau)

        return dx

    def _generate_single_trajectory(
        self,
        I_w_C: float,
        seed: int
    ) -> Optional[Dict]:
        """
        Generate a single trajectory with given I_w_C.

        Returns:
            Dict with x, u, dx, params or None if all attempts failed
        """
        rng = np.random.RandomState(seed)
        max_attempts = self.quality_filters['max_attempts']

        sim = self._create_simulator(I_w_C)

        for attempt in range(max_attempts):
            self.stats['total_attempts'] += 1

            x0 = self._sample_initial_state(rng)

            # New controller seed per attempt (different excitation)
            ctrl_seed = seed * 1000 + attempt
            controller = self._create_controller(ctrl_seed)

            try:
                t_span = (0.0, self.T_duration)
                t, x, u = sim.simulate(
                    x0=x0,
                    t_span=t_span,
                    dt=self.dt,
                    controller=controller,
                    method='RK45'
                )

                passed, reason = self._check_quality(x, u)

                if passed:
                    self.stats['total_accepted'] += 1
                    dx = self._compute_dx_analytic(sim, x, u)

                    return {
                        'x': x,
                        'u': u,
                        'dx': dx,
                        'params': np.array([I_w_C], dtype=np.float64)
                    }
                else:
                    self.stats['total_rejected'] += 1
                    self.stats['rejection_reasons'][reason] = \
                        self.stats['rejection_reasons'].get(reason, 0) + 1

            except RuntimeError:
                self.stats['total_rejected'] += 1
                self.stats['rejection_reasons']['integration_failure'] = \
                    self.stats['rejection_reasons'].get('integration_failure', 0) + 1

        return None

    def generate_split(
        self,
        split_name: str,
        I_w_C_values: List[float],
        n_traj_per_condition: int,
        base_seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trajectories for a split (train/val/test).

        Returns:
            x: (N, T, 4), u: (N, T, 1), dx: (N, T, 4),
            params: (N, 1), cond_id: (N,)
        """
        print(f"\n  Generating {split_name} split...")
        print(f"    Conditions: {len(I_w_C_values)} I_w_C values: {I_w_C_values}")
        print(f"    Trajectories per condition: {n_traj_per_condition}")

        all_x, all_u, all_dx, all_params, all_cond_id = [], [], [], [], []

        for cond_idx, I_w_C in enumerate(I_w_C_values):
            for traj_idx in range(n_traj_per_condition):
                seed = base_seed + cond_idx * 10000 + traj_idx

                result = self._generate_single_trajectory(I_w_C, seed)

                if result is not None:
                    all_x.append(result['x'])
                    all_u.append(result['u'])
                    all_dx.append(result['dx'])
                    all_params.append(result['params'])
                    all_cond_id.append(cond_idx)

        print(f"    Generated: {len(all_x)} trajectories")

        if len(all_x) == 0:
            raise RuntimeError(
                f"No trajectories generated for {split_name} split! "
                f"Stats: {self.stats}. "
                f"Consider relaxing quality_filters or checking simulator stability."
            )

        x = np.stack(all_x, axis=0)
        u = np.stack(all_u, axis=0)
        dx = np.stack(all_dx, axis=0)
        params = np.stack(all_params, axis=0)
        cond_id = np.array(all_cond_id, dtype=np.int64)

        return x, u, dx, params, cond_id

    def generate_dataset(self, version: str, master_seed: int = 42) -> Dict:
        """
        Generate complete dataset with train/val/test splits.

        Split design (Tier-A main):
        - Train: I_w_C in {6.95e-5, 8.6875e-5}, 5 per condition = 10
        - Val:   I_w_C in {6.95e-5, 8.6875e-5}, ceil(5/2)=3 per cond -> subsample to 5
        - Test:  I_w_C = 1.04e-4, 10
        """
        print("=" * 60)
        print(f"  AEK Dataset Generation: {version}")
        print("=" * 60)

        set_global_seed(master_seed)

        # Conditions from YAML (Tier-A main)
        train_I_w_C = self.ood_config['main']['train_conditions']
        test_I_w_C = self.ood_config['main']['test_conditions']
        val_I_w_C = train_I_w_C

        n_train_target = self.n_traj['train']
        n_val_target = self.n_traj['val']
        n_test_target = self.n_traj['test']

        n_train_per_cond = n_train_target // len(train_I_w_C)
        n_val_per_cond = (n_val_target + len(val_I_w_C) - 1) // len(val_I_w_C)
        n_test_per_cond = n_test_target // len(test_I_w_C)

        # Generate splits
        train_x, train_u, train_dx, train_params, train_cond_id = self.generate_split(
            split_name='train',
            I_w_C_values=train_I_w_C,
            n_traj_per_condition=n_train_per_cond,
            base_seed=master_seed
        )

        val_x, val_u, val_dx, val_params, val_cond_id = self.generate_split(
            split_name='val',
            I_w_C_values=val_I_w_C,
            n_traj_per_condition=n_val_per_cond,
            base_seed=master_seed + 100000
        )

        test_x, test_u, test_dx, test_params, test_cond_id = self.generate_split(
            split_name='test',
            I_w_C_values=test_I_w_C,
            n_traj_per_condition=n_test_per_cond,
            base_seed=master_seed + 200000
        )

        # Time array
        t = np.linspace(0, self.T_duration, self.T_steps)

        # Subsample to target sizes if overgenerated
        rng = np.random.RandomState(master_seed + 999)

        if len(val_x) > n_val_target:
            idx = rng.choice(len(val_x), n_val_target, replace=False)
            idx.sort()
            val_x, val_u, val_dx = val_x[idx], val_u[idx], val_dx[idx]
            val_params, val_cond_id = val_params[idx], val_cond_id[idx]

        if len(train_x) > n_train_target:
            idx = rng.choice(len(train_x), n_train_target, replace=False)
            idx.sort()
            train_x, train_u, train_dx = train_x[idx], train_u[idx], train_dx[idx]
            train_params, train_cond_id = train_params[idx], train_cond_id[idx]

        if len(test_x) > n_test_target:
            idx = rng.choice(len(test_x), n_test_target, replace=False)
            idx.sort()
            test_x, test_u, test_dx = test_x[idx], test_u[idx], test_dx[idx]
            test_params, test_cond_id = test_params[idx], test_cond_id[idx]

        print(f"\n  Final dataset sizes:")
        print(f"    Train: {len(train_x)} trajectories")
        print(f"    Val:   {len(val_x)} trajectories")
        print(f"    Test:  {len(test_x)} trajectories")

        accept_rate = self.stats['total_accepted'] / max(1, self.stats['total_attempts'])
        print(f"\n  Quality filter statistics:")
        print(f"    Acceptance rate: {accept_rate:.1%}")
        if self.stats['rejection_reasons']:
            print(f"    Rejection reasons: {self.stats['rejection_reasons']}")

        # Verify T consistency
        assert train_x.shape[1] == self.T_steps, \
            f"T mismatch: got {train_x.shape[1]}, expected {self.T_steps}"

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
            'dt': np.float64(self.dt)
        }

        # Build metadata
        metadata = {
            'version': version,
            'system': 'aek',
            'created_at': datetime.now().isoformat(),
            'config_path': str(self.config_path),
            'master_seed': master_seed,
            'ood_tier': 'main (Tier-A)',
            'ood_parameter': 'I_w_C',
            'ood_extrapolation_ratio': self.ood_config['main']['extrapolation_ratio'],
            'conditions': {
                'train_I_w_C': train_I_w_C,
                'val_I_w_C': val_I_w_C,
                'test_I_w_C': test_I_w_C,
            },
            'simulation': {
                'dt': self.dt,
                'T_steps': self.T_steps,
                'T_duration': self.T_duration,
                'integrator': self.sim_config['integrator'],
            },
            'initial_conditions_actual': self.ic_ranges,
            'initial_conditions_yaml': self.sim_config['initial_conditions'],
            'controller': self.controller_config,
            'controller_note': (
                'PD stabilization + random excitation. '
                'tau = +K_p*phi + K_d*phi_dot + excitation(t), clipped to tau_max. '
                'Positive sign because tau enters EOM as -tau/I_p '
                '(opposite of standard -K convention). '
                'Pure random torque cannot stabilize AEK inverted pendulum '
                '(tau_max=0.02 N*m << gravity torque ~0.28 N*m/rad).'
            ),
            'quality_filters': {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.quality_filters.items()
            },
            'statistics': {
                'n_train': len(train_x),
                'n_val': len(val_x),
                'n_test': len(test_x),
                'T': self.T_steps,
                'dt': self.dt,
                'state_dim': 4,
                'input_dim': 1,
                'param_dim': 1,
                'acceptance_rate': accept_rate,
                'rejection_reasons': self.stats['rejection_reasons']
            },
            'state_definition': {
                'index_0': 'phi (rad) - lean angle',
                'index_1': 'phi_dot (rad/s) - lean angular velocity',
                'index_2': 'theta_w (rad) - wheel angle (relative)',
                'index_3': 'theta_w_dot (rad/s) - wheel angular velocity (relative)',
            },
            'input_definition': {
                'index_0': 'tau (N*m) - motor torque',
            },
            'param_definition': {
                'index_0': 'I_w_C (kg*m^2) - wheel spin inertia (OOD knob)',
            },
        }

        return {'data': dataset, 'meta': metadata}

    def save_dataset(self, dataset_dict: Dict, version: str) -> Tuple[Path, Path]:
        """Save dataset to files."""
        npz_path = paths.get_dataset_path(version, system='aek')
        meta_path = paths.get_meta_path(version, system='aek')

        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(npz_path, **dataset_dict['data'])
        print(f"\n  ✅ Saved: {npz_path}")

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict['meta'], f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: {meta_path}")

        return npz_path, meta_path


# =============================================================================
# Convenience Function
# =============================================================================

def generate_dataset(
    version: str,
    config_path: Optional[Path] = None,
    seed: int = 42
) -> Tuple[Path, Path]:
    """Generate and save AEK dataset."""
    if config_path is None:
        config_path = paths.ROOT / 'configs' / 'systems' / 'aek.yaml'

    generator = AEKDataGenerator(config_path)
    dataset_dict = generator.generate_dataset(version, master_seed=seed)
    return generator.save_dataset(dataset_dict, version)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate AEK Self-balancing Motorcycle OOD dataset'
    )
    parser.add_argument(
        '--version', '-v', type=str, default='aek_ood_v1',
        help='Dataset version string (default: aek_ood_v1)'
    )
    parser.add_argument(
        '--config', '-c', type=str, default=None,
        help='Path to config file (default: configs/systems/aek.yaml)'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=42,
        help='Master random seed (default: 42)'
    )

    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None
    npz_path, meta_path = generate_dataset(args.version, config_path, args.seed)

    print("\n" + "=" * 60)
    print("  AEK Dataset generation complete!")
    print("=" * 60)
    print(f"\n  Files:")
    print(f"    {npz_path}")
    print(f"    {meta_path}")

    # Preflight validation
    print("\n  Validating with schema_dataset_lite...")
    from src.contracts.schema_dataset_lite import validate_dataset_lite
    validate_dataset_lite(npz_path)


if __name__ == '__main__':
    main()