"""
Quick Ninapro PCA test using best-practice approach from MyoKi experiments.
Uses task-averaged static-postures with flexion-only joint selection.
"""
import os, glob
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = 'PCA_results/experiments'
FIG_DIR = 'Ninapro_figures'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Ninapro 20 hand joints (indices 0-19, no wrist)
ALL_JOINT_NAMES = [
    "CMC1_f","CMC1_a","MCP1","IP1",
    "MCP2_f","MCP2_a","PIP2","DIP2",
    "MCP3_f","PIP3","DIP3",
    "MCP4_f","MCP4_a","PIP4","DIP4",
    "CMC5","MCP5_f","MCP5_a","PIP5","DIP5",
]

# For flexion-only: thumb MCP+IP, all fingers MCP+PIP (no DIP, no abd)
# Ninapro indices: MCP1(2), IP1(3), MCP2_f(4), PIP2(6), MCP3_f(7), PIP3(8), 
# MCP4_f(9), PIP4(11), MCP5_f(13), PIP5(15)
FLEXION_IDS = [2, 3, 4, 6, 7, 8, 9, 11, 13, 15]
FLEXION_NAMES = [ALL_JOINT_NAMES[i] for i in FLEXION_IDS]

# No abduction: all except MCP2_a(5), MCP4_a(10), MCP5_a(14)
NO_ABD_IDS = [i for i in range(20) if i not in [5, 10, 14]]
NO_ABD_NAMES = [ALL_JOINT_NAMES[i] for i in NO_ABD_IDS]

# Pisa targets for different joint subsets
def make_pisa_target(jids):
    """Create Pisa target for given joint indices."""
    n = len(jids)
    s1 = np.ones(n) / np.linalg.norm(np.ones(n))
    # Ninapro Pisa pattern (20 joints): [2,2,1,1, 1,1,1,1, 1,1,1, -1,-1,-1,-1, -2,-2,-2,-2,-2]
    raw = np.array([2,2,1,1, 1,1,1,1, 1,1,1, -1,-1,-1,-1, -2,-2,-2,-2,-2])
    s2 = raw[jids] / np.linalg.norm(raw[jids])
    return s1, s2, np.column_stack([s1, s2])

def load_ninapro_by_task(files, ex='E2'):
    """Load Ninapro data grouped by trial/task."""
    all_trials = []
    for fpath in files:
        mat = sio.loadmat(fpath)
        angles = mat['angles']
        tasks = mat['trial_tasks'].flatten().astype(int)
        n_cols = min(20, angles.shape[1])
        angles = angles[:, :n_cols]
        
        # Extract trials by task boundaries
        diff = np.diff(tasks, prepend=-999, append=-999)
        starts = np.where(diff != 0)[0]
        for i in range(len(starts)-1):
            s, e = starts[i], starts[i+1]
            trial = angles[s:e]
            tid = tasks[s]
            if len(trial) >= 5 and tid > 0:  # valid task
                all_trials.append((trial, tid, os.path.basename(fpath).split('_')[0]))
    return all_trials

def run_test(label, trials, jids, joint_names):
    """Run task-averaged PCA analysis."""
    # Task-averaged static postures (last 5 frames)
    by_task = {}
    for trial, tid, subj in trials:
        post = np.mean(trial[-5:], axis=0)[jids]
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(post)
    
    task_data = np.array([np.mean(v, axis=0) for v in by_task.values()])
    n_joints = len(jids)
    pisa_target = make_pisa_target(jids)[2]
    
    if len(task_data) < 3 or task_data.shape[1] < 2:
        return None
    
    pca = PCA(n_components=min(15, *task_data.shape))
    pca.fit(task_data)
    evr = pca.explained_variance_ratio_
    
    # Normalize PCs
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
    
    # Procrustes
    PC = np.column_stack([pc1, pc2])
    R, _ = orthogonal_procrustes(PC, pisa_target)
    rotated = PC @ R
    
    s1_sim = np.dot(rotated[:, 0], pisa_target[:, 0])
    s2_sim = np.dot(rotated[:, 1], pisa_target[:, 1])
    
    print(f"  {label} [{len(task_data)} tasks, {n_joints} joints]: "
          f"Var={sum(evr[:2])*100:.1f}%, "
          f"S1={s1_sim:.4f}, S2={s2_sim:.4f}, Score={s1_sim+s2_sim:.4f}")
    
    # Print joint weights
    print(f"    Joint weights (Rot S1 vs Pisa S1, Rot S2 vs Pisa S2):")
    for i in range(n_joints):
        print(f"      {joint_names[i]:12s}: RotS1={rotated[i,0]:+7.4f} PisaS1={pisa_target[i,0]:+7.4f}  "
              f"RotS2={rotated[i,1]:+7.4f} PisaS2={pisa_target[i,1]:+7.4f}")
    
    return {
        'label': label, 'n_tasks': len(task_data), 'n_joints': n_joints,
        'var': sum(evr[:2]), 's1_sim': s1_sim, 's2_sim': s2_sim,
        'rotated': rotated, 'pisa_target': pisa_target,
        'evr': evr, 'joint_names': joint_names,
    }

def main():
    print("=" * 70)
    print("NINAPRO PCA TEST (Best Practice Approach)")
    print("=" * 70)
    
    # Find preprocessed files
    files = sorted(glob.glob('Ninapro Data/preprocessed/S*_E2_preprocessed.mat'))
    print(f"\nE2 files: {len(files)}")
    
    # Load E2 trials
    e2_trials = load_ninapro_by_task(files, 'E2')
    print(f"E2 total trials: {len(e2_trials)}")
    
    results = []
    
    # Test with different joint subsets
    configs = [
        ('ninapro_E2_all20', list(range(20)), ALL_JOINT_NAMES),
        ('ninapro_E2_no_abd', NO_ABD_IDS, NO_ABD_NAMES),
        ('ninapro_E2_flexion_only', FLEXION_IDS, FLEXION_NAMES),
    ]
    
    for label, jids, jnames in configs:
        res = run_test(label, e2_trials, jids, jnames)
        if res:
            results.append(res)
    
    # Also test E3
    files_e3 = sorted(glob.glob('Ninapro Data/preprocessed/S*_E3_preprocessed.mat'))
    print(f"\nE3 files: {len(files_e3)}")
    if files_e3:
        e3_trials = load_ninapro_by_task(files_e3, 'E3')
        print(f"E3 total trials: {len(e3_trials)}")
        for label, jids, jnames in configs:
            label_e3 = label.replace('E2', 'E3')
            res = run_test(label_e3, e3_trials, jids, jnames)
            if res:
                results.append(res)
    
    # Combined E12
    files_e1 = sorted(glob.glob('Ninapro Data/preprocessed/S*_E1_preprocessed.mat'))
    if files_e1:
        e1_trials = load_ninapro_by_task(files_e1, 'E1')
        combined_trials = e1_trials + e2_trials
        print(f"\nE1+E2 combined: {len(combined_trials)} trials")
        res = run_test('ninapro_E12_flexion_only', combined_trials, FLEXION_IDS, FLEXION_NAMES)
        if res: results.append(res)
    
    print(f"\n{'='*120}")
    print(f"{'NINAPRO RESULTS SUMMARY':^120}")
    print(f"{'='*120}")
    print(f"{'Experiment':<35} {'Tasks':>6} {'Joints':>8} {'Var%':>8} {'S1':>8} {'S2':>8} {'Score':>8}")
    print("-" * 120)
    
    for res in sorted(results, key=lambda r: r['s1_sim']+r['s2_sim'], reverse=True):
        print(f"{res['label']:<35} {res['n_tasks']:>6} {res['n_joints']:>8} "
              f"{res['var']*100:>7.1f}% {res['s1_sim']:>+7.4f} {res['s2_sim']:>+7.4f} "
              f"{res['s1_sim']+res['s2_sim']:>7.4f}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
