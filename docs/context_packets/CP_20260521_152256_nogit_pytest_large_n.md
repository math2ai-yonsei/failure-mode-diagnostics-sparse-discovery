# Context Packet: 20260521_152256_nogit_pytest_large_n

## 실험 정보

| 항목 | 값 |
|------|-----|
| Gate | gate0 |
| Dataset | cartpole_ood_v1 |
| Track | standardized |
| Method | sindy |
| n_train | 1000 |
| seed | 0 |
| threshold | 0.01 |

## 실행 명령어

```bash
python experiments/run_gate0.py ^
  --config configs/experiments/gate0_cartpole.yaml ^
  --dataset_version cartpole_ood_v1 ^
  --n_train 1000 --seed 0 ^
  --track standardized --note pytest_large_n
```

## Results 경로

```
C:\python_work\PhD_project_release\results\cartpole_ood_v1\gate0\standardized\sindy\n1000\seed0\20260521_152256_nogit_pytest_large_n
```

## Metrics 요약

| Split | R² (mean) |
|-------|-----------|
| Train | 0.9018 |
| Val | 0.8973 |
| Test | 0.8683 |

- Sparsity: 60.7%
- Nonzero terms: 33 / 84

## 산출물

- manifest.json ✅
- metrics.json ✅
- sindy_coefficients.csv ✅
- F00_condition_distribution.png/pdf ✅
- F01_rollout_example.png/pdf ✅
- F02_coeff_heatmap.png/pdf ✅

## 다음 작업

- [ ] seed=1로 재실행하여 Gate0 통과 조건(2 seeds) 달성
- [ ] n_train=20으로 실험 확장
- [ ] Gate1 (E-SINDy) 진입 준비

---
*Generated: 2026-05-21T15:22:57.465593*
