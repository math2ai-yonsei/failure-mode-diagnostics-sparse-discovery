"""
Pytest configuration for PhD_project tests.

Sets up matplotlib backend, Python path, and common fixtures.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path (BEFORE any src imports)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set matplotlib backend BEFORE any matplotlib imports
# This ensures headless operation in CI/pytest environments
os.environ["MPLBACKEND"] = "Agg"

import pytest
import numpy as np


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root) -> Path:
    """Return data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def cartpole_dataset_path(data_dir) -> Path:
    """Return cartpole dataset path if exists."""
    path = data_dir / "cartpole" / "cartpole_ood_v1" / "dataset.npz"
    if not path.exists():
        pytest.skip("Cartpole dataset not found")
    return path