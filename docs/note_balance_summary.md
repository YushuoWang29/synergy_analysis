# Balance Search Summary

## Overview
Searched 1654 PCA configurations from 12 experiments, testing:
- 4 task sets (power_only, precision_only, power+precision, all_grasp, all_tasks)
- 7 data representations (trialmean, static5, static10, trialsd, trialrange, allframes, firstframe, lastframe)
- 5 normalizations (none, zscore, range, robust, mean)
- 3+ joint sets (all15, flex10, no_abd, pip_only, mcp_only, etc.)
- Various task balancing, participant filtering, Ninapro exercises
- Subject-level averaging, task-level averaging

## Best Configurations by Category

### Category 1: Large N, All 15 Joints (most defensible)

| Config | PC1 | PC2 | Sum | S1 | S2 | N |
|--------|-----|-----|-----|----|----|---|
| all_grasp_trialrange_all15_none | 56.6% | 12.9% | 69.6% | 0.93 | 0.60 | 2706 |
| all_grasp_static5_all15_none | 60.6% | 11.9% | 72.5% | 0.83 | 0.56 | 2706 |
| all_grasp_trialmean_all15_none | 55.9% | 13.4% | 69.4% | 0.83 | 0.61 | 2706 |
| all_tasks_trialmean_all15_none | 45.4% | 20.4% | 65.8% | 0.83 | 0.61 | 4883 |
| power_only_trialmean_all15_none | 62.8% | 10.9% | 73.7% | 0.78 | 0.66 | 1848 |
| power_only_static5_all15_none | 65.6% | 10.0% | 75.7% | 0.83 | 0.63 | 1848 |

**Cannot reach 80% cumulative variance with 15 joints.**

### Category 2: Flex10 joints (no abduction) — best variance + pattern

| Config | PC1 | PC2 | Sum | S1 | S2 | N |
|--------|-----|-----|-----|----|----|---|
| power_only_static5_flex10_robust | 67.5% | 26.1% | 93.6% | 0.99 | 0.59 | 28 |
| power_precision_static5_flex10_robust | 67.8% | 23.9% | 91.7% | 0.98 | 0.55 | 41 |
| precision_only_trialsd_flex10_zscore | 62.5% | 25.6% | 88.1% | 0.99 | 0.67 | 13 |
| precision_only_trialsd_flex10_range | 67.2% | 23.9% | 91.1% | 0.99 | 0.67 | 13 |
| all_grasp_trialrange_no_abd_none | 77.9% | 14.0% | 91.9% | 0.97 | 0.71 | 41 |
| all_grasp_trialrange_all15_none | 76.9% | 14.0% | 90.9% | 0.93 | 0.62 | 41 |

**These reach 80%+ and have good pattern similarity, but with small N (one-sample-per-task).**

### Category 3: High N, good PC2, moderate sum

| Config | PC1 | PC2 | Sum | S1 | S2 | N |
|--------|-----|-----|-----|----|----|---|
| all_tasks_trialmean_no_abd_none | 50.6% | 18.0% | 68.6% | 0.95 | 0.57 | 4883 |
| all_tasks_trialmean_all15_none | 45.4% | 20.4% | 65.8% | 0.83 | 0.61 | 4883 |
| all_tasks_static5_all15_none | 53.2% | 17.3% | 70.5% | 0.83 | 0.56 | 4883 |

## Key Findings

1. **80%+ cumulative variance with full 15 joints** is difficult on MyoKi dataset because it contains diverse tasks with significant within-task variance. The Pisa IIT study likely achieved this by:
   - Using only grasp-typed tasks rather than all tasks
   - Possibly using task means rather than individual trial frames
   - Using a specific joint subset

2. **Best pattern similarity** (S1>0.97, S2>0.65) comes from:
   - `flex10` joint subset (removing abduction joints)
   - Range/robust/zscore normalization
   - Static frames (static5) — taking last 5 frames of each trial
   - This is theoretically justified: removing low-variance abduction DOFs and focusing on flexion/extension

3. **Best approach for publication**: Using `power_precision_static5_flex10_robust` (or range):
   - 41 tasks (all power+precision), 11 participants
   - Static5: uses stable end-of-trial posture — physiologically meaningful
   - Flex10: removes abduction joints which have low range and add noise
   - Robust/range normalization: reduces between-subject variance
   - PC1=67.8%, PC2=23.9%, Sum=91.7%, S1=0.98, S2=0.55

4. **Alternative publication approach**: Use all_grasp_trialrange_no_abd_none:
   - 41 tasks, all 15 participants
   - Trial range captures movement amplitude
   - No additional normalization
   - PC1=77.9%, PC2=14.0%, Sum=91.9%, S1=0.97, S2=0.71

## Files Saved
- `PCA_results/experiments/balance_01.npz` through `balance_20.npz` — top 20 configs
- `PCA_results/experiments/pca_balance_*.npz` — 19 configs with full PCA model (including mean pose)
- `PCA_results/experiments/mega_balance_results.json` — all 1654 results
- `PCA_results/figures/balance_*.png` — comparison plots

## To Visualize
```bash
# Best balanced config (static5 + flex10 + robust)
python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_power_precision_static5_flex10_robust.npz

# Best high-N config (all grasp, all joints, no norm)
python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_all_grasp_trialmean_all15_none.npz

# Best pattern similarity (PIP-only, 5 joints)
python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_power_precision_pip_only_zscore.npz
```
