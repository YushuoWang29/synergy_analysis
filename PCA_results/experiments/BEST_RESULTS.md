# Top PCA Results бк Synergy Analysis

All files are saved as `pca_myoki_*.npz` in `PCA_results/MyoKi/`
and can be loaded with:
```bash
python code/myoki/synergy_myoki.py --participant best_flexion_only
```

| # | Name | Var% | S1 | S2 | Score | Tasks | Joints | Subj | Preprocessing |
|---|------|------|----|----|-------|-------|--------|------|---------------|
| 1 | бя best_flexion_only | 95.2% | 0.9435 | 0.8100 | 1.7536 | 28 | 10j | 11 | task-averaged static5 (last 5 frames) |
| 2 | allgrasp_flexion_only | 92.7% | 0.9514 | 0.7929 | 1.7444 | 41 | 10j | 11 | task-averaged static5 (last 5 frames) |
| 3 | allgrasp_no_abd | 91.5% | 0.8860 | 0.7396 | 1.6256 | 41 | 12j | 11 | task-averaged static5 (last 5 frames) |
| 4 | power_no_abd | 93.3% | 0.8441 | 0.7750 | 1.6190 | 28 | 12j | 11 | task-averaged static5 (last 5 frames) |
| 5 | power_static10_no_abd | 93.4% | 0.8229 | 0.7830 | 1.6059 | 28 | 12j | 11 | task-averaged static10 (last 10 frames) |
| 6 | P12_taskavg_no_abd | 89.6% | 0.8857 | 0.7029 | 1.5886 | 28 | 12j | 1 | task-averaged static5, single subject |
| 7 | P05_taskavg_no_abd | 90.3% | 0.7626 | 0.7964 | 1.5591 | 28 | 12j | 1 | task-averaged static5, single subject |
| 8 | power_median_all15 | 92.8% | 0.7781 | 0.7180 | 1.4960 | 28 | 15j | 11 | task-median static5 (median of last 5 frames) |
| 9 | power_all15 | 92.8% | 0.7787 | 0.7166 | 1.4954 | 28 | 15j | 11 | task-averaged static5 (last 5 frames) |
| 10 | allgrasp_all15 | 90.8% | 0.8045 | 0.6833 | 1.4878 | 41 | 15j | 11 | task-averaged static5 (last 5 frames) |


---

## How to visualize each result

### бя BEST: Power Grasps, Task-Averaged Static5, Flexion-Only (10 joints)
- **Var**: 95.2%, **S1**: 0.9435, **S2**: 0.8100
- **Tasks**: 28 power grasps (cylindrical+spherical+hook), **Participants**: all 11 (P01-P09, P11-P12)
- **Preprocessing**: task-averaged static5 (last 5 frames)
- **Joints**: 10D: ThumbMCP, ThumbIP, IdxMCP, IdxPIP, MidMCP, MidPIP, RingMCP, RingPIP, LitMCP, LitPIP
```bash
python code/myoki/synergy_myoki.py --participant best_flexion_only
```

### All Grasps, Task-Averaged Static5, Flexion-Only (10 joints)
- **Var**: 92.7%, **S1**: 0.9514, **S2**: 0.7929
- **Tasks**: 41 all grasp types, **Participants**: all 11 (P01-P09, P11-P12)
- **Preprocessing**: task-averaged static5 (last 5 frames)
- **Joints**: 10D: ThumbMCP, ThumbIP, IdxMCP, IdxPIP, MidMCP, MidPIP, RingMCP, RingPIP, LitMCP, LitPIP
```bash
python code/myoki/synergy_myoki.py --participant allgrasp_flexion_only
```

### All Grasps, Task-Averaged Static5, No Abduction (12 joints)
- **Var**: 91.5%, **S1**: 0.8860, **S2**: 0.7396
- **Tasks**: 41 all grasp types, **Participants**: all 11 (P01-P09, P11-P12)
- **Preprocessing**: task-averaged static5 (last 5 frames)
- **Joints**: 12D: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP, MidMCP, MidPIP, RingMCP, RingPIP, LitMCP, LitPIP
```bash
python code/myoki/synergy_myoki.py --participant allgrasp_no_abd
```

### Power Grasps, Task-Averaged Static5, No Abduction (12 joints)
- **Var**: 93.3%, **S1**: 0.8441, **S2**: 0.7750
- **Tasks**: 28 power grasps, **Participants**: all 11 (P01-P09, P11-P12)
- **Preprocessing**: task-averaged static5 (last 5 frames)
- **Joints**: 12D: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP, MidMCP, MidPIP, RingMCP, RingPIP, LitMCP, LitPIP
```bash
python code/myoki/synergy_myoki.py --participant power_no_abd
```

### Power Grasps, Task-Averaged Static10 (longer hold), No Abduction
- **Var**: 93.4%, **S1**: 0.8229, **S2**: 0.7830
- **Tasks**: 28 power grasps, **Participants**: all 11 (P01-P09, P11-P12)
- **Preprocessing**: task-averaged static10 (last 10 frames)
- **Joints**: 12D: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP, MidMCP, MidPIP, RingMCP, RingPIP, LitMCP, LitPIP
```bash
python code/myoki/synergy_myoki.py --participant power_static10_no_abd
```

### Subject P12 Only, Power Grasps, Task-Averaged Static5, No Abduction
- **Var**: 89.6%, **S1**: 0.8857, **S2**: 0.7029
- **Tasks**: 28 power grasps, **Participants**: P12 only
- **Preprocessing**: task-averaged static5, single subject
- **Joints**: 12D: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP, MidMCP, MidPIP, RingMCP, RingPIP, LitMCP, LitPIP
```bash
python code/myoki/synergy_myoki.py --participant P12_taskavg_no_abd
```

### Subject P05 Only, Power Grasps, Task-Averaged Static5, No Abduction
- **Var**: 90.3%, **S1**: 0.7626, **S2**: 0.7964
- **Tasks**: 28 power grasps, **Participants**: P05 only
- **Preprocessing**: task-averaged static5, single subject
- **Joints**: 12D: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP, MidMCP, MidPIP, RingMCP, RingPIP, LitMCP, LitPIP
```bash
python code/myoki/synergy_myoki.py --participant P05_taskavg_no_abd
```

### Power Grasps, Task-Median Static5, All 15 Joints
- **Var**: 92.8%, **S1**: 0.7781, **S2**: 0.7180
- **Tasks**: 28 power grasps, **Participants**: all 11 (P01-P09, P11-P12)
- **Preprocessing**: task-median static5 (median of last 5 frames)
- **Joints**: 15D: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP, MidMCP, MidPIP, IdxMidAbd, RingMCP, RingPIP, MidRingAbd, LitMCP, LitPIP, RingLitAbd
```bash
python code/myoki/synergy_myoki.py --participant power_median_all15
```

### Power Grasps, Task-Averaged Static5, All 15 Joints
- **Var**: 92.8%, **S1**: 0.7787, **S2**: 0.7166
- **Tasks**: 28 power grasps, **Participants**: all 11 (P01-P09, P11-P12)
- **Preprocessing**: task-averaged static5 (last 5 frames)
- **Joints**: 15D: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP, MidMCP, MidPIP, IdxMidAbd, RingMCP, RingPIP, MidRingAbd, LitMCP, LitPIP, RingLitAbd
```bash
python code/myoki/synergy_myoki.py --participant power_all15
```

### All Grasps, Task-Averaged Static5, All 15 Joints
- **Var**: 90.8%, **S1**: 0.8045, **S2**: 0.6833
- **Tasks**: 41 all grasp types, **Participants**: all 11 (P01-P09, P11-P12)
- **Preprocessing**: task-averaged static5 (last 5 frames)
- **Joints**: 15D: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP, MidMCP, MidPIP, IdxMidAbd, RingMCP, RingPIP, MidRingAbd, LitMCP, LitPIP, RingLitAbd
```bash
python code/myoki/synergy_myoki.py --participant allgrasp_all15
```

