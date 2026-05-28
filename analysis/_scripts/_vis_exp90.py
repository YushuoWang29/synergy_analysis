"""
Replicate V5 experiment #90 and save viewer-compatible file, then launch.
v5_power_only_power_joints_task_std
Var: 78.2%+14.6%=92.8%  S1=0.9620 S2=0.6060  Bal=2.4853
"""
import sys, os
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(__file__))
from save_best_results import (get_participant_files, load_myoki_trials, 
                                make_pisa_target, ALL_JOINT_NAMES)

# Config for experiment #90
POWER_GRASP_IDS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31]
# power_joints from mega_experiment_v5.py
POWER_JOINTS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13, 0, 3]

SAVE_NAME = 'v5_power_only_power_joints_task_std'

# Load data
files = get_participant_files()
trials = load_myoki_trials(files, POWER_GRASP_IDS)
print(f"Loaded {len(trials)} trials")

# task_std preprocessing: std within each trial
by_task = {}
for t, tid, subj in trials:
    if tid not in by_task:
        by_task[tid] = []
    by_task[tid].append(np.std(t, axis=0))  # std across frames

# Average per task
task_means = {}
for tid, std_list in by_task.items():
    task_means[tid] = np.mean(std_list, axis=0)

# Build data matrices
task_data_15d = np.array(list(task_means.values()))  # (n_tasks, 15)
task_data = task_data_15d[:, POWER_JOINTS]  # (n_tasks, 12)

print(f"Task data: {task_data.shape}, {len(by_task)} tasks")
print(f"Full 15D data: {task_data_15d.shape}")

# PCA
n_comp = min(15, *task_data.shape)
pca = PCA(n_components=n_comp)
pca.fit(task_data)

var_pc1 = pca.explained_variance_ratio_[0]
var_pc2 = pca.explained_variance_ratio_[1]
print(f"\nPCA Results:")
print(f"  PC1: {var_pc1*100:.1f}%")
print(f"  PC2: {var_pc2*100:.1f}%")
print(f"  Sum: {(var_pc1+var_pc2)*100:.1f}%")

# Procrustes to Pisa targets
pisa_s1, pisa_s2, pisa_target = make_pisa_target(POWER_JOINTS)
pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
PC = np.column_stack([pc1, pc2])
R, _ = orthogonal_procrustes(PC, pisa_target)
rotated = PC @ R
s1_sim = float(np.dot(rotated[:, 0], pisa_s1))
s2_sim = float(np.dot(rotated[:, 1], pisa_s2))
print(f"  S1={s1_sim:.4f}, S2={s2_sim:.4f}")

# Build full 15-D components for viewer
full_components = np.zeros((pca.n_components_, 15))
for i in range(pca.n_components_):
    full_components[i, POWER_JOINTS] = pca.components_[i]

full_mean = np.mean(task_data_15d, axis=0)

# Print PC patterns
print(f"\nPC1 (selected joints): {np.round(pca.components_[0], 3)}")
print(f"PC2 (selected joints): {np.round(pca.components_[1], 3)}")

jnames = [ALL_JOINT_NAMES[i] for i in POWER_JOINTS]
print(f"\nJoint names for selected: {jnames}")
print(f"Rotated PC1: {np.round(rotated[:, 0], 3)}")
print(f"Rotated PC2: {np.round(rotated[:, 1], 3)}")

# Save viewer file
viewer_path = f'PCA_results/MyoKi/pca_myoki_{SAVE_NAME}.npz'
np.savez(viewer_path,
         components=full_components,
         mean=full_mean,
         explained_variance=pca.explained_variance_ratio_,
         data_label=f'V5 #{SAVE_NAME} | PC1={var_pc1*100:.1f}% PC2={var_pc2*100:.1f}% S1={s1_sim:.3f} S2={s2_sim:.3f}')

print(f"\n✅ Saved: {viewer_path}")
print(f"\nRun viewer:")
print(f"  python code/myoki/synergy_myoki.py --participant {SAVE_NAME}")
