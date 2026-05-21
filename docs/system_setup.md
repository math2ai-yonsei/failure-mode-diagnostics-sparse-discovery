# System Setup

This document describes how to set up the Python environment needed to run the
code in this repository.

## Requirements

- Python 3.10 or newer.
- The six third-party packages listed in `requirements.txt`. All other imports
  in the codebase are from the Python standard library.
- Git (used by the path convention to record a short commit hash in run IDs;
  the code runs without Git, recording `nogit` instead).

## Creating the environment

The project was developed on Windows with VS Code and PowerShell. From the
repository root:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`
instead.

## Project root

Paths are generated through a single module (`src/contracts/paths.py`). It
locates the project root automatically as two directory levels above that
file. If you run scripts from an unusual location, set the project root
explicitly with an environment variable:

```
$env:PHD_PROJECT_ROOT = "C:\path\to\repository"     # PowerShell
export PHD_PROJECT_ROOT=/path/to/repository          # macOS / Linux
```

## Verifying the installation

```
python -c "import numpy, scipy, sklearn, matplotlib, yaml; print('environment OK')"
pytest -q
```

The test suite includes a simulator check (`test_aek_simulator.py`). A clean
run confirms the environment is correctly configured.

## Exact version pinning

The bounds in `requirements.txt` are conservative minimums and list the direct
dependencies only. For an exact reproduction of the environment used in the
paper, generate a separate pinned lock file in the working environment:

```
pip freeze > requirements-lock.txt
```

Keep `requirements.txt` as the human-readable direct-dependency list; do not
overwrite it with the `pip freeze` output.
