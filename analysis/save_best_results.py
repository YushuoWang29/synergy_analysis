"""
Save all top PCA results as synergy-viewer-compatible files + a manifest.
The synergy_myoki.py script expects .npz files with:
  - components: (n_components, n_joints) eigenpostures
  - mean: (n_joints,) mean posture
  - explained_variance: (n_components,) variance ratios
  - data_label: string descriptor (used in figure title)

This script:
  1. Re-runs the top 10 experiments to generate PCA models with FULL 15-D mean
     (so the myoki_to_qpos mapping works correctly for visualization)
  2. Saves them to PCA_results/MyoKi/pca_myoki_BEST_*.npz
  3. Creates a manifest with all metadata
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes
import openpyxl

SAVE_DIR = 'PCA_results/MyoKi'
EXPERIMENT_DIR = 'PCA_results/experiments'
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# ============================================================
# Data loading helpers
# ============================================================
FLEXION_IDS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]  # 10 joints
NO_ABD_IDS  = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13]  # 12 joints
ALL15_IDS   = list(range(15))  # 15 joints

ALL_JOINT_NAMES = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
    "RingMCP","RingPIP","MidRingAbd",
    "LitMCP","LitPIP","RingLitAbd"
]

def load_task_categories():
    """Load task categorization from Excel."""
    wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
    ws = wb.active
    categories = {}
    action_cols = {'cylindrical': 11, 'spherical': 12, 'hook': 13, 'tripod': 14,
                   'pinch': 15, 'lumbrical': 16, 'complex': 17, 'none_action': 18}
    task_names = {}
    for row in ws.iter_rows(min_row=4, values_only=False):
        tid = row[0].value
        if tid is None: continue
        tid_int = int(tid)
        task_names[tid_int] = str(row[1].value).strip() if row[1].value else f"Task {tid_int}"
        actions = []
        for act_name, col_idx in action_cols.items():
            if row[col_idx].value is not None and str(row[col_idx].value).strip() == 'x':
                actions.append(act_name)
        categories[tid_int] = actions
    return categories, task_names

categories, task_names = load_task_categories()

# Define task ID sets
POWER_TYPES = {'cylindrical', 'spherical', 'hook'}
ALL_GRASP_TYPES = {'cylindrical', 'spherical', 'hook', 'tripod', 'pinch', 'lumbrical'}

POWER_GRASP_IDS = sorted([tid for tid, acts in categories.items()
                          if any(a in POWER_TYPES for a in acts) and 'none_action' not in acts])
ALL_GRASP_IDS = sorted([tid for tid, acts in categories.items()
                        if any(a in ALL_GRASP_TYPES for a in acts) and 'none_action' not in acts])

def load_myoki_trials(files, task_ids, resample_n=100):
    """Load MyoKi trials, resampled to fixed length."""
    all_trials = []
    for fpath in files:
        mat = sio.loadmat(fpath)
        glove = np.nan_to_num(mat['glove_calibrated'])
        tasks = mat['task'].flatten().astype(int)
        hand_data = glove[:, :15]
        for tid in task_ids:
            mask = (tasks == tid)
            diff = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0]
            for s, e in zip(starts, ends):
                trial = hand_data[s:e, :]
                if len(trial) >= 5:
                    x_old = np.linspace(0, 1, len(trial))
                    x_new = np.linspace(0, 1, resample_n)
                    f = interp1d(x_old, trial, axis=0, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                    resampled = f(x_new)
                    all_trials.append((resampled, tid, os.path.basename(fpath)))
    return all_trials

def get_participant_files():
    """Get all valid participant files."""
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    return [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]

def make_pisa_target(jids):
    """Create Pisa/IIT synergy targets for given joint indices (15-D)."""
    n = len(jids)
    s1 = np.ones(n) / np.linalg.norm(np.ones(n))
    # Pisa pattern for 15 joints: [2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2]
    raw = np.array([2, 2, 2, 2, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2])
    s2 = raw[jids] / np.linalg.norm(raw[jids])
    return s1, s2, np.column_stack([s1, s2])

# ============================================================
# Experiment definitions with detailed metadata
# ============================================================
EXPERIMENTS = [
    {
        "name": "best_flexion_only",
        "display_name": "★ BEST: Power Grasps, Task-Averaged Static5, Flexion-Only (10 joints)",
        "task_ids": POWER_GRASP_IDS,
        "task_type": "power grasps (cylindrical+spherical+hook)",
        "jids": FLEXION_IDS,
        "joint_names": [ALL_JOINT_NAMES[i] for i in FLEXION_IDS],
        "n_joints": 10,
        "preprocess": "task-averaged static5 (last 5 frames)",
        "participants": "all 11 (P01-P09, P11-P12)",
        "expected_var": 95.2,
        "expected_s1": 0.9435,
        "expected_s2": 0.8100,
    },
    {
        "name": "allgrasp_flexion_only",
        "display_name": "All Grasps, Task-Averaged Static5, Flexion-Only (10 joints)",
        "task_ids": ALL_GRASP_IDS,
        "task_type": "all grasp types",
        "jids": FLEXION_IDS,
        "joint_names": [ALL_JOINT_NAMES[i] for i in FLEXION_IDS],
        "n_joints": 10,
        "preprocess": "task-averaged static5 (last 5 frames)",
        "participants": "all 11 (P01-P09, P11-P12)",
        "expected_var": 92.7,
        "expected_s1": 0.9514,
        "expected_s2": 0.7929,
    },
    {
        "name": "power_no_abd",
        "display_name": "Power Grasps, Task-Averaged Static5, No Abduction (12 joints)",
        "task_ids": POWER_GRASP_IDS,
        "task_type": "power grasps",
        "jids": NO_ABD_IDS,
        "joint_names": [ALL_JOINT_NAMES[i] for i in NO_ABD_IDS],
        "n_joints": 12,
        "preprocess": "task-averaged static5 (last 5 frames)",
        "participants": "all 11 (P01-P09, P11-P12)",
        "expected_var": 93.3,
        "expected_s1": 0.8441,
        "expected_s2": 0.7750,
    },
    {
        "name": "power_all15",
        "display_name": "Power Grasps, Task-Averaged Static5, All 15 Joints",
        "task_ids": POWER_GRASP_IDS,
        "task_type": "power grasps",
        "jids": ALL15_IDS,
        "joint_names": ALL_JOINT_NAMES,
        "n_joints": 15,
        "preprocess": "task-averaged static5 (last 5 frames)",
        "participants": "all 11 (P01-P09, P11-P12)",
        "expected_var": 92.8,
        "expected_s1": 0.7787,
        "expected_s2": 0.7166,
    },
    {
        "name": "allgrasp_no_abd",
        "display_name": "All Grasps, Task-Averaged Static5, No Abduction (12 joints)",
        "task_ids": ALL_GRASP_IDS,
        "task_type": "all grasp types",
        "jids": NO_ABD_IDS,
        "joint_names": [ALL_JOINT_NAMES[i] for i in NO_ABD_IDS],
        "n_joints": 12,
        "preprocess": "task-averaged static5 (last 5 frames)",
        "participants": "all 11 (P01-P09, P11-P12)",
        "expected_var": 91.5,
        "expected_s1": 0.8860,
        "expected_s2": 0.7396,
    },
    {
        "name": "allgrasp_all15",
        "display_name": "All Grasps, Task-Averaged Static5, All 15 Joints",
        "task_ids": ALL_GRASP_IDS,
        "task_type": "all grasp types",
        "jids": ALL15_IDS,
        "joint_names": ALL_JOINT_NAMES,
        "n_joints": 15,
        "preprocess": "task-averaged static5 (last 5 frames)",
        "participants": "all 11 (P01-P09, P11-P12)",
        "expected_var": 90.8,
        "expected_s1": 0.8045,
        "expected_s2": 0.6833,
    },
    {
        "name": "power_median_all15",
        "display_name": "Power Grasps, Task-Median Static5, All 15 Joints",
        "task_ids": POWER_GRASP_IDS,
        "task_type": "power grasps",
        "jids": ALL15_IDS,
        "joint_names": ALL_JOINT_NAMES,
        "n_joints": 15,
        "preprocess": "task-median static5 (median of last 5 frames)",
        "participants": "all 11 (P01-P09, P11-P12)",
        "expected_var": 92.1,
        "expected_s1": 0.8070,
        "expected_s2": 0.6944,
    },
    {
        "name": "power_static10_no_abd",
        "display_name": "Power Grasps, Task-Averaged Static10 (longer hold), No Abduction",
        "task_ids": POWER_GRASP_IDS,
        "task_type": "power grasps",
        "jids": NO_ABD_IDS,
        "joint_names": [ALL_JOINT_NAMES[i] for i in NO_ABD_IDS],
        "n_joints": 12,
        "preprocess": "task-averaged static10 (last 10 frames)",
        "participants": "all 11 (P01-P09, P11-P12)",
        "expected_var": 93.4,
        "expected_s1": 0.8229,
        "expected_s2": 0.7830,
    },
    {
        "name": "P05_taskavg_no_abd",
        "display_name": "Subject P05 Only, Power Grasps, Task-Averaged Static5, No Abduction",
        "task_ids": POWER_GRASP_IDS,
        "task_type": "power grasps",
        "jids": NO_ABD_IDS,
        "joint_names": [ALL_JOINT_NAMES[i] for i in NO_ABD_IDS],
        "n_joints": 12,
        "preprocess": "task-averaged static5, single subject",
        "participants": "P05 only",
        "expected_var": 90.9,
        "expected_s1": 0.7880,
        "expected_s2": 0.7978,
    },
    {
        "name": "P12_taskavg_no_abd",
        "display_name": "Subject P12 Only, Power Grasps, Task-Averaged Static5, No Abduction",
        "task_ids": POWER_GRASP_IDS,
        "task_type": "power grasps",
        "jids": NO_ABD_IDS,
        "joint_names": [ALL_JOINT_NAMES[i] for i in NO_ABD_IDS],
        "n_joints": 12,
        "preprocess": "task-averaged static5, single subject",
        "participants": "P12 only",
        "expected_var": 88.6,
        "expected_s1": 0.8855,
        "expected_s2": 0.7274,
    },
]

def run_experiment(exp):
    """Run PCA experiment and save synergy-viewer-compatible file."""
    files = get_participant_files()
    if exp.get('participants', '').startswith('P'):
        subj = exp['participants'][:3]
        files = [f for f in files if subj in os.path.basename(f)]
    
    trials = load_myoki_trials(files, exp['task_ids'])
    
    # Build task-averaged data
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        post = np.mean(t[-5:], axis=0) if 'static10' not in exp.get('preprocess', '') else np.mean(t[-10:], axis=0)
        if 'median' in exp.get('preprocess', '').lower():
            post = np.median(t[-5:], axis=0)
        by_task[tid].append(post)
    
    # Use selected joints
    jids = exp['jids']
    task_data = np.array([np.mean(v, axis=0)[jids] for v in by_task.values()])
    
    # Also compute full 15-D means for synergy viewer compatibility
    all_task_data_full = np.array([np.mean(v, axis=0) for v in by_task.values()])
    full_mean = np.mean(all_task_data_full, axis=0)
    
    # PCA on selected joints
    pca = PCA(n_components=min(15, *task_data.shape))
    pca.fit(task_data)
    
    # Compute Procrustes rotation
    pisa_s1, pisa_s2, pisa_target = make_pisa_target(jids)
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
    PC = np.column_stack([pc1, pc2])
    R, _ = orthogonal_procrustes(PC, pisa_target)
    rotated = PC @ R
    s1_sim = float(np.dot(rotated[:, 0], pisa_s1))
    s2_sim = float(np.dot(rotated[:, 1], pisa_s2))
    
    print(f"\n{'='*60}")
    print(f"{exp['name']}")
    print(f"  Tasks: {len(by_task)}, Joints: {len(jids)}, Subjects: {exp['participants']}")
    print(f"  Var: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")
    print(f"  S1={s1_sim:.4f}, S2={s2_sim:.4f}, Score={s1_sim+s2_sim:.4f}")
    
    # ====== Save TWO files ======
    
    # 1) Selected-joints PCA file (compact - for reference)
    select_path = os.path.join(EXPERIMENT_DIR, f'final_{exp["name"]}.npz')
    np.savez(select_path,
             components=pca.components_,
             mean=pca.mean_,
             explained_variance=pca.explained_variance_ratio_,
             pc1=pc1, pc2=pc2,
             rotated_s1=rotated[:, 0], rotated_s2=rotated[:, 1],
             rotation_matrix=R, s1_sim=s1_sim, s2_sim=s2_sim,
             joint_names=np.array(exp['joint_names'], dtype=object),
             scores=pca.transform(task_data)[:, :2] @ R.T,
             description=exp['display_name'])
    
    # 2) Synergy-viewer compatible file (uses FULL 15-D mean + padded components)
    # The synergy viewer expects:
    #   - components: (n_components, 15) eigenpostures in FULL 15-D space
    #   - mean: (15,) mean posture
    #   - explained_variance: (n_components,)
    #   - data_label: string
    # 
    # For joint subsets, we need to reconstruct FULL 15-D vectors.
    # We do this by creating components that are 0 in the excluded joints.
    full_components = np.zeros((pca.n_components_, 15))
    for i in range(pca.n_components_):
        full_components[i, jids] = pca.components_[i]
    
    viewer_path = os.path.join(SAVE_DIR, f'pca_myoki_{exp["name"]}.npz')
    np.savez(viewer_path,
             components=full_components,
             mean=full_mean,
             explained_variance=pca.explained_variance_ratio_,
             data_label=f'{exp["display_name"]} | S1={s1_sim:.3f} S2={s2_sim:.3f} Var={sum(pca.explained_variance_ratio_[:2])*100:.1f}%')
    
    print(f"  Saved: {select_path}")
    print(f"  Saved (viewer): {viewer_path}")
    
    return {
        'name': exp['name'],
        'display_name': exp['display_name'],
        'task_type': exp['task_type'],
        'n_tasks': len(by_task),
        'n_joints': len(jids),
        'n_subjects': 1 if exp['participants'].startswith('P') else 11,
        'participants': exp['participants'],
        'preprocess': exp['preprocess'],
        'var_pc1': float(pca.explained_variance_ratio_[0]),
        'var_pc2': float(pca.explained_variance_ratio_[1]),
        'var_sum': float(sum(pca.explained_variance_ratio_[:2])),
        's1_sim': s1_sim,
        's2_sim': s2_sim,
        'rot_sum': s1_sim + s2_sim,
        'viewer_file': f'pca_myoki_{exp["name"]}.npz',
        'reference_file': f'final_{exp["name"]}.npz',
        'joints_used': exp['joint_names'],
        'task_ids': [int(t) for t in by_task.keys()],
        'task_names': {int(t): task_names.get(int(t), f"Task {t}")
                       for t in by_task.keys() if int(t) in task_names},
    }

def main():
    print("=" * 70)
    print("SAVING BEST PCA RESULTS — synergy-viewer compatible + manifest")
    print("=" * 70)
    
    results = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp)
        results.append(result)
    
    # Sort by score
    results.sort(key=lambda r: r['rot_sum'], reverse=True)
    
    # Save manifest
    manifest_path = os.path.join(EXPERIMENT_DIR, 'best_results_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Also save a human-readable markdown
    md_path = os.path.join(EXPERIMENT_DIR, 'BEST_RESULTS.md')
    with open(md_path, 'w') as f:
        f.write("# Top PCA Results — Synergy Analysis\n\n")
        f.write("All files are saved as `pca_myoki_*.npz` in `PCA_results/MyoKi/`\n")
        f.write("and can be loaded with:\n")
        f.write("```bash\n")
        f.write("python code/myoki/synergy_myoki.py --participant best_flexion_only\n")
        f.write("```\n\n")
        f.write("| # | Name | Var% | S1 | S2 | Score | Tasks | Joints | Subj | Preprocessing |\n")
        f.write("|---|------|------|----|----|-------|-------|--------|------|---------------|\n")
        for i, r in enumerate(results, 1):
            star = "★ " if i == 1 else ""
            f.write(f"| {i} | {star}{r['name']} "
                    f"| {r['var_sum']*100:.1f}% "
                    f"| {r['s1_sim']:.4f} "
                    f"| {r['s2_sim']:.4f} "
                    f"| {r['rot_sum']:.4f} "
                    f"| {r['n_tasks']} "
                    f"| {r['n_joints']}j "
                    f"| {r['n_subjects']} "
                    f"| {r['preprocess']} |\n")
        f.write("\n\n---\n\n")
        f.write("## How to visualize each result\n\n")
        for r in results:
            iface = "--participant " + r['name']
            f.write(f"### {r['display_name']}\n")
            f.write(f"- **Var**: {r['var_sum']*100:.1f}%, **S1**: {r['s1_sim']:.4f}, **S2**: {r['s2_sim']:.4f}\n")
            f.write(f"- **Tasks**: {r['n_tasks']} {r['task_type']}, **Participants**: {r['participants']}\n")
            f.write(f"- **Preprocessing**: {r['preprocess']}\n")
            f.write(f"- **Joints**: {r['n_joints']}D: {', '.join(r['joints_used'])}\n")
            f.write("```bash\n")
            f.write(f"python code/myoki/synergy_myoki.py {iface}\n")
            f.write("```\n\n")
    
    print(f"\n{'='*70}")
    print(f"Manifest saved: {manifest_path}")
    print(f"Markdown guide: {md_path}")
    print(f"\nTo visualize any result, run:")
    for r in results[:3]:
        print(f"  python code/myoki/synergy_myoki.py --participant {r['name']}")
    print(f"\nDone! {len(results)} results saved.")

if __name__ == '__main__':
    main()
