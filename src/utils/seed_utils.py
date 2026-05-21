"""
Seed utilities for reproducibility.
"""
import random
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seed for numpy and random module.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for all random number generators.
    
    This is an alias for set_seed() for clarity in data generation context.
    
    Args:
        seed: Random seed value
    """
    set_seed(seed)


def get_rng(seed: int) -> np.random.RandomState:
    """
    Create a new RandomState with given seed.
    
    Use this when you need an independent random stream.
    
    Args:
        seed: Random seed value
        
    Returns:
        numpy RandomState instance
    """
    return np.random.RandomState(seed)


if __name__ == '__main__':
    # Quick test
    set_global_seed(42)
    print(f"✅ set_global_seed(42) 호출 성공")
    print(f"   np.random.rand(): {np.random.rand():.6f}")
    
    rng = get_rng(42)
    print(f"   get_rng(42).rand(): {rng.rand():.6f}")