# Context Packet: 20260521_152253_nogit_pytest_author

## 실험 정보

| 항목 | 값 |
|------|-----|
| Gate | gate0 |
| Dataset | cartpole_ood_v1 |
| Track | author_recommended |
| Method | sindy |
| n_train | 10 |
| seed | 0 |
| threshold | 0.01 |

## 실행 명령어

```bash
python experiments/run_gate0.py ^
  --config configs/experiments/gate0_cartpole.yaml ^
  --dataset_version cartpole_ood_v1 ^
  --n_train 10 --seed 0 ^
  --track author_recommended --note pytest_author
```

## Results 경로

```
C:\python_work\PhD_project_release\results\cartpole_ood_v1\gate0\author_recommended\sindy\n10\seed0\20260521_152253_nogit_pytest_author
```

## Metrics 요약

| Split | R² (mean) |
|-------|-----------|
| Train | 0.9167 |
| Val | 0.8069 |
| Test | 0.8408 |

- Sparsity: 52.4%
- Nonzero terms: 40 / 84

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
*Generated: 2026-05-21T15:22:55.027007*
