"""
Simulators package for PhD project.
Provides physics-based trajectory generation for dynamical systems.
"""

from .base_simulator import BaseSimulator
from .cartpole_simulator import CartPoleSimulator
from .aek_simulator import AEKSimulator

__all__ = ['BaseSimulator', 'CartPoleSimulator', 'AEKSimulator']