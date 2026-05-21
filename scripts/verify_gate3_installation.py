#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Installation Verification Script
"""

import sys
from pathlib import Path

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_pass(msg: str):
    print(f"  ✅ {msg}")

def check_fail(msg: str):
    print(f"  ❌ {msg}")

def check_warn(msg: str):
    print(f"  ⚠️  {msg}")

def find_dataset_path(project_root: Path, dataset_version: str) -> Path:
    """
    Find dataset path (support multiple directory structures)
    
    Checks in order:
    1. data/{dataset_version}/dataset.npz
    2. data/cartpole/{dataset_version}/dataset.npz
    """
    # Priority 1: data/{dataset_version}/dataset.npz
    path1 = project_root / 'data' / dataset_version / 'dataset.npz'
    if path1.exists():
        return path1
    
    # Priority 2: data/cartpole/{dataset_version}/dataset.npz
    path2 = project_root / 'data' / 'cartpole' / dataset_version / 'dataset.npz'
    if path2.exists():
        return path2
    
    return None

def main():
    print("\n" + "="*60)
    print("  Gate3 Installation Verification")
    print("="*60)
    
    errors = []
    warnings = []
    
    # 1. Python version
    print_section("1. Python Environment")
    py_version = sys.version_info
    if py_version >= (3, 8):
        check_pass(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        check_fail(f"Python {py_version.major}.{py_version.minor} (requires >= 3.8)")
        errors.append("Python version too old")
    
    # 2. Required Libraries
    print_section("2. Required Libraries")
    
    required_libs = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('torch', 'torch'),
        ('yaml', 'yaml'),
        ('matplotlib', 'matplotlib'),
    ]
    
    for lib_name, import_name in required_libs:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            check_pass(f"{lib_name} ({version})")
        except ImportError:
            check_fail(f"{lib_name} not installed")
            errors.append(f"Missing library: {lib_name}")
    
    # 3. Project Structure
    print_section("3. Project Structure")
    
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    if not (project_root / 'src').exists():
        possible_roots = [
            Path('C:/python_work/PhD_project'),
            Path.cwd(),
            script_path.parent,
        ]
        for root in possible_roots:
            if (root / 'src').exists():
                project_root = root
                break
    
    check_pass(f"Project root: {project_root}")
    
    required_dirs = [
        'src/contracts',
        'src/sindy',
        'src/augmentation',
        'src/experiments',
        'src/generative',
        'configs/experiments',
        'experiments',
        'scripts',
        'results',
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            check_pass(f"Directory: {dir_path}")
        else:
            check_warn(f"Directory missing: {dir_path}")
            warnings.append(f"Missing directory: {dir_path}")
    
    # 4. SSOT Files
    print_section("4. SSOT Files (Contracts)")
    
    ssot_files = [
        'src/contracts/paths.py',
        'src/contracts/plot_style.py',
        'src/contracts/schema_dataset_lite.py',
    ]
    
    for file_path in ssot_files:
        full_path = project_root / file_path
        if full_path.exists():
            check_pass(f"SSOT: {file_path}")
        else:
            check_fail(f"SSOT missing: {file_path}")
            errors.append(f"Missing SSOT: {file_path}")
    
    # 5. Gate1/Gate2 Dependencies
    print_section("5. Gate1/Gate2 Dependencies")
    
    gate_deps = [
        ('src/sindy/esindy.py', 'E-SINDy module'),
        ('src/experiments/gate1_esindy_runner.py', 'Gate1 runner'),
        ('src/augmentation/physics_augmentor.py', 'Gate2 augmentor'),
        ('src/experiments/gate2_aug_runner.py', 'Gate2 runner'),
    ]
    
    for file_path, desc in gate_deps:
        full_path = project_root / file_path
        if full_path.exists():
            check_pass(f"{desc}: {file_path}")
        else:
            check_warn(f"{desc} missing: {file_path}")
            warnings.append(f"Missing: {file_path}")
    
    # 6. Gate3 Files
    print_section("6. Gate3 Files")
    
    gate3_files = [
        ('configs/experiments/gate3_cartpole.yaml', 'Gate3 config'),
        ('experiments/run_gate3.py', 'Gate3 CLI'),
        ('src/generative/vae.py', 'VAE model'),
        ('src/generative/alignment.py', 'Teacher alignment'),
        ('src/generative/filtering.py', 'Filtering module'),
        ('src/augmentation/generative_augmentor.py', 'Generative augmentor'),
        ('src/augmentation/generative_baselines.py', 'Baseline methods'),
        ('src/experiments/gate3_gen_runner.py', 'Gate3 runner'),
    ]
    
    for file_path, desc in gate3_files:
        full_path = project_root / file_path
        if full_path.exists():
            check_pass(f"{desc}: {file_path}")
        else:
            check_fail(f"{desc} missing: {file_path}")
            errors.append(f"Missing Gate3 file: {file_path}")
    
    # 7. Gate3 Config Parse Test
    print_section("7. Gate3 Config Validation")
    
    gate3_config = project_root / 'configs/experiments/gate3_cartpole.yaml'
    if gate3_config.exists():
        try:
            import yaml
            with open(gate3_config, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'seeds' in config:
                check_pass("Config: seeds block present")
            else:
                check_warn("Config: seeds block missing")
                warnings.append("seeds block missing in config")
            
            if config.get('gate') == 'gate3':
                check_pass("Config: gate=gate3 confirmed")
            
        except Exception as e:
            check_fail(f"Config parse error: {e}")
            errors.append(f"Config parse error: {e}")
    
    # 8. Dataset
    print_section("8. Dataset Availability")
    
    dataset_version = 'cartpole_ood_v1'
    dataset_path = find_dataset_path(project_root, dataset_version)
    
    if dataset_path:
        check_pass(f"Dataset found: {dataset_path.relative_to(project_root)}")
        
        # Validate dataset keys
        try:
            import numpy as np
            data = np.load(dataset_path)
            required_keys = ['train_x', 'val_x', 'test_x', 'train_dx', 'val_dx', 'test_dx', 'dt']
            missing_keys = [k for k in required_keys if k not in data.keys()]
            
            if not missing_keys:
                check_pass(f"Dataset keys: OK ({len(data.keys())} keys)")
                check_pass(f"Dataset train_x shape: {data['train_x'].shape}")
            else:
                check_warn(f"Dataset missing keys: {missing_keys}")
                warnings.append(f"Dataset missing keys: {missing_keys}")
        except Exception as e:
            check_warn(f"Dataset load error: {e}")
            warnings.append(f"Dataset load error: {e}")
    else:
        check_fail(f"Dataset not found for {dataset_version}")
        check_warn(f"  Searched: data/{dataset_version}/dataset.npz")
        check_warn(f"  Searched: data/cartpole/{dataset_version}/dataset.npz")
        errors.append("Dataset not found")
    
    # 9. Gate1 Results
    print_section("9. Gate1 Results (Teacher Source)")
    
    gate1_results = project_root / 'results' / dataset_version / 'gate1'
    if gate1_results.exists():
        run_dirs = list(gate1_results.rglob('*/metrics.json'))
        if run_dirs:
            check_pass(f"Gate1 results: {len(run_dirs)} runs found")
            # Show latest run
            latest = sorted(run_dirs, key=lambda x: x.parent.name)[-1]
            check_pass(f"Latest run: {latest.parent.name}")
        else:
            check_warn("No Gate1 runs found")
            warnings.append("No Gate1 results")
    else:
        check_warn("Gate1 results directory not found")
        warnings.append("Gate1 results missing")
    
    # 10. Gate2 Results
    print_section("10. Gate2 Results (Matched Baseline)")
    
    gate2_results = project_root / 'results' / dataset_version / 'gate2'
    if gate2_results.exists():
        run_dirs = list(gate2_results.rglob('*/metrics.json'))
        if run_dirs:
            check_pass(f"Gate2 results: {len(run_dirs)} runs found")
        else:
            check_warn("No Gate2 runs found")
            warnings.append("No Gate2 results")
    else:
        check_warn("Gate2 results directory not found")
        warnings.append("Gate2 results missing")
    
    # Summary
    print_section("Summary")
    
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"     - {e}")
    
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"     - {w}")
    
    if not errors:
        print("\n  ✅ Gate3 installation verification PASSED")
        print("     Ready to proceed with Phase -1")
        return 0
    else:
        print("\n  ❌ Gate3 installation verification FAILED")
        print("     Please fix errors before proceeding")
        return 1

if __name__ == '__main__':
    sys.exit(main())