"""
Export the TOP configurations from mega_balance_results.json as proper PCA models.
Re-runs PCA with EXACTLY the same pipeline as the balance search to ensure matching results.
Saves with mean pose for synergy_myoki.py compatibility.
"""
import os, sys, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes
import openpyxl

# ─── Load task info ───
def load_task_info():
    wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
    ws = wb.active
    categories, task_names = {}, {}
    action_cols = {'cylindrical': 11, 'spherical': 12, 'hook': 13, 'tripod': 14, 'pinch': 15, 'lumbrical': 16, 'complex': 17, 'none_action': 18}
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

categories, task_names = load_task_info()
power_types = {'cylindrical', 'spherical', 'hook'}
precision_types = {'tripod', 'pinch', 'lumbrical'}
grasp_types = power_types | precision_types

POWER_IDS = sorted([tid for tid, acts in categories.items() if any(a in power_types for a in acts) and 'none_action' not in acts])
ALL_GRASP_IDS = sorted([tid for tid, acts in categories.items() if any(a in grasp_types for a in acts) and 'none_action' not in acts])
PRECISION_IDS = sorted([tid for tid, acts in categories.items() if any(a in precision_types for a in acts) and 'none_action' not in acts and not any(a in power_types for a in acts)])
NON_GRASP_IDS = sorted([tid for tid, acts in categories.items() if 'none_action' in acts])
COMPLEX_IDS = sorted([tid for tid, acts in categories.items() if 'complex' in acts])
ALL_TASKS = sorted(categories.keys())
FLEXION_IDS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]
NO_ABD_IDS = [1,2,3,4,5,6,7,9,10,12,13]

def get_pickup_ids():
    wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
    ws = wb.active
    ids = set()
    for row in ws.iter_rows(min_row=4, values_only=False):
        tid = row[0].value; name = str(row[1].value).strip() if row[1].value else ''
        if tid is not None and 'place it on' in name.lower() and 'back' not in name.lower():
            ids.add(int(tid))
    return sorted(ids)
PICKUP_IDS = get_pickup_ids()

# ─── Load trials ───
def load_myoki_trials(files, task_ids, resample_n=100):
    all_trials = []
    for fpath in files:
        mat = sio.loadmat(fpath)
        glove = np.nan_to_num(mat['glove_calibrated'])
        tasks = mat['task'].flatten().astype(int)
        hand_data = glove[:, :15]
        for tid in task_ids:
            mask = (tasks == tid)
            diff = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(diff > 0)[0]; ends = np.where(diff < 0)[0]
            for s, e in zip(starts, ends):
                trial = hand_data[s:e, :]
                if len(trial) >= 5:
                    x_old = np.linspace(0, 1, len(trial))
                    x_new = np.linspace(0, 1, resample_n)
                    f = interp1d(x_old, trial, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
                    all_trials.append((f(x_new), tid, os.path.basename(fpath).replace('.mat', '')))
    return all_trials

def get_participant_files():
    pf = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid = [f'P{p:02d}' for p in [1,2,3,4,5,6,7,8,9,11,12]]
    return [f for f in pf if any(p in os.path.basename(f) for p in valid)]
FILES = get_participant_files()

# ─── Processing functions ───
def extract_static5(trials): return [(np.mean(t[-5:], axis=0), tid, subj) for t, tid, subj in trials]
def extract_trial_means(trials): return [(np.mean(t, axis=0), tid, subj) for t, tid, subj in trials]
def extract_trial_std(trials): return [(np.std(t, axis=0), tid, subj) for t, tid, subj in trials]
def extract_trial_range(trials): return [(np.max(t, axis=0)-np.min(t, axis=0), tid, subj) for t, tid, subj in trials]

# Normalization
def zscore_norm(data):
    m = np.mean(data, axis=0); s = np.std(data, axis=0); s[s<1e-6]=1.0; return (data-m)/s
def robust_norm(data):
    med = np.median(data, axis=0); q75, q25 = np.percentile(data, [75,25], axis=0); iqr=q75-q25; iqr[iqr<1e-6]=1.0; return (data-med)/iqr
def range_norm(data):
    mins = np.min(data,axis=0); maxs = np.max(data,axis=0); r=maxs-mins; r[r<1e-6]=1.0; return (data-mins)/r
def none_norm(data): return data
NORM_FUNCS = {'none': none_norm, 'zscore': zscore_norm, 'range': range_norm, 'robust': robust_norm}

# ─── Configuration definitions ───
# Each entry: (short_name, task_ids_fn, rep_fn, norm_name, jids, n_tasks, n_participants, description)
CONFIGS = [
    ("all_grasp_static5_flex10_robust", lambda: ALL_GRASP_IDS, extract_static5, "robust", FLEXION_IDS, 41, 11),
    ("power_precision_static5_flex10_robust", lambda: sorted(set(POWER_IDS)|set(PRECISION_IDS)), extract_static5, "robust", FLEXION_IDS, 41, 11),
    ("power_only_static5_flex10_robust", lambda: POWER_IDS, extract_static5, "robust", FLEXION_IDS, 28, 11),
    ("precision_only_trialsd_flex10_zscore", lambda: PRECISION_IDS, extract_trial_std, "zscore", FLEXION_IDS, 13, 11),
    ("precision_only_trialsd_flex10_range", lambda: PRECISION_IDS, extract_trial_std, "range", FLEXION_IDS, 13, 11),
    ("all_grasp_trialrange_no_abd_none", lambda: ALL_GRASP_IDS, extract_trial_range, "none", NO_ABD_IDS, 41, 11),
    ("power_only_trialrange_no_abd_none", lambda: POWER_IDS, extract_trial_range, "none", NO_ABD_IDS, 28, 11),
    ("all_grasp_trialrange_all15_none", lambda: ALL_GRASP_IDS, extract_trial_range, "none", None, 41, 11),
    ("all_grasp_trialrange_no_abd_robust", lambda: ALL_GRASP_IDS, extract_trial_range, "robust", NO_ABD_IDS, 41, 11),
    ("all_grasp_trialmean_all15_none", lambda: ALL_GRASP_IDS, extract_trial_means, "none", None, 41, 11),
    ("all_grasp_static5_all15_none", lambda: ALL_GRASP_IDS, extract_static5, "none", None, 41, 11),
    ("all_tasks_trialmean_all15_none", lambda: sorted(set(ALL_TASKS)-{0}), extract_trial_means, "none", None, 74, 11),
    ("all_tasks_trialmean_no_abd_none", lambda: sorted(set(ALL_TASKS)-{0}), extract_trial_means, "none", NO_ABD_IDS, 74, 11),
    ("power_only_trialmean_all15_none", lambda: POWER_IDS, extract_trial_means, "none", None, 28, 11),
    ("power_only_static5_all15_none", lambda: POWER_IDS, extract_static5, "none", None, 28, 11),
    ("precision_only_trialmean_no_abd_zscore", lambda: PRECISION_IDS, extract_trial_means, "zscore", NO_ABD_IDS, 13, 11),
    ("precision_only_trialsd_all15_zscore", lambda: PRECISION_IDS, extract_trial_std, "zscore", None, 13, 11),
    ("precision_only_trialmean_all15_none", lambda: PRECISION_IDS, extract_trial_means, "none", None, 13, 11),
    ("precision_only_static5_all15_none", lambda: PRECISION_IDS, extract_static5, "none", None, 13, 11),
    ("power_only_static5_no_abd_robust", lambda: POWER_IDS, extract_static5, "robust", NO_ABD_IDS, 28, 11),
]

# Pisa target for comparison
N_JOINTS = 15
PISA_S1 = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
PISA_S2_RAW = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_S2 = PISA_S2_RAW / np.linalg.norm(PISA_S2_RAW)

def make_pisa_jids(jids):
    """Get Pisa target vectors for a joint subset."""
    n = len(jids)
    s1 = np.ones(n) / np.linalg.norm(np.ones(n))
    raw = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
    s2 = raw[jids] / (np.linalg.norm(raw[jids]) + 1e-10)
    return s1, s2

def evaluate_pc_similarity(pc1, pc2, jids):
    """Rotate PCs to best match Pisa targets and compute similarity."""
    if jids is not None:
        s1, s2 = make_pisa_jids(np.array(jids))
    else:
        s1, s2 = PISA_S1, PISA_S2
    
    # Only compare using the available joints
    n = len(pc1)
    s1_u = s1[:n]; s2_u = s2[:n]
    
    PC = np.column_stack([pc1, pc2])
    target = np.column_stack([s1_u, s2_u])
    R, _ = orthogonal_procrustes(PC, target)
    rotated = PC @ R
    s1_sim = float(np.dot(rotated[:, 0], s1_u) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(s1_u) + 1e-10))
    s2_sim = float(np.dot(rotated[:, 1], s2_u) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(s2_u) + 1e-10))
    return s1_sim, s2_sim

def main():
    out_dir = "PCA_results/experiments"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 180)
    print("EXPORTING TOP PCA CONFIGS AS FULL MODELS")
    print("=" * 180)
    
    # ─── Table header ───
    hdr = f"{'#':<3} {'Config Name':<50} {'PC1%':>6} {'PC2%':>6} {'Sum%':>6} {'S1_sim':>7} {'S2_sim':>7} {'DoF':>4} {'Tasks':>5} {'Subj':>5} {'Task division':<40} {'Preprocessing'}"
    print(hdr)
    print("-" * 180)
    
    results = []
    for idx, (name, task_fn, rep_fn, norm_name, jids, n_tasks, n_part) in enumerate(CONFIGS):
        task_ids = task_fn()
        
        # Load and process
        trials = load_myoki_trials(FILES, task_ids)
        processed = rep_fn(trials)
        if not processed:
            continue
        data_array = np.vstack([p[0] for p in processed])
        
        # Normalize
        data = NORM_FUNCS[norm_name](data_array)
        
        # Select joints
        if jids is not None:
            data_subset = data[:, jids]
            n_j = len(jids)
        else:
            data_subset = data
            n_j = 15
        
        # PCA on subset
        n_comp = min(n_j, data_subset.shape[0], 10)
        pca = PCA(n_components=n_comp)
        pca.fit(data_subset)
        evr = pca.explained_variance_ratio_
        pc1_sub = pca.components_[0]; pc2_sub = pca.components_[1]
        
        # Compute Pisa similarity on the actual PC components used
        s1_sim, s2_sim = evaluate_pc_similarity(pc1_sub, pc2_sub, jids)
        
        # Build 15-joint components
        full_components = np.zeros((n_comp, 15))
        full_mean = np.zeros(15)
        
        if jids is not None:
            full_components[:, jids] = pca.components_[:, :n_j]
            full_mean[jids] = np.mean(data_array[:, jids], axis=0)
        else:
            full_components[:, :n_j] = pca.components_
            full_mean = np.mean(data_array, axis=0)
        
        # Describe task division
        if name.startswith("all_grasp"): task_desc = "All grasp tasks (41) — power:28 prec:13"
        elif name.startswith("power_precision"): task_desc = "Power+Precision (41) — power:28 prec:13"
        elif name.startswith("power_only"): task_desc = "Power grip only (28)"
        elif name.startswith("all_tasks"): task_desc = "All tasks (74) — grasp:41 non-grasp:6 rest:27"
        elif name.startswith("precision_only"): task_desc = "Precision grip only (13) — tripod+pinch+lumbrical"
        else: task_desc = "?"
        
        # Describe preprocessing
        rep_desc = {'extract_static5': 'Last 5 frames (static hold)', 'extract_trial_means': 'Trial mean', 
                    'extract_trial_std': 'Trial std (motion variability)', 'extract_trial_range': 'Trial range (motion amplitude)'}
        
        if jids is None: joint_desc = "All 15 joints (ThumbAbd+ThumbMCP+ThumbIP+ThumbOpp+IdxMCP+IdxPIP+MidMCP+MidPIP+IdxMidAbd+RingMCP+RingPIP+MidRingAbd+LitMCP+LitPIP+RingLitAbd)"
        elif jids == FLEXION_IDS: joint_desc = "Flexion joints only (10) — MCP+PIP all fingers, ThumbAbd/ThumbOpp/IdxMidAbd/MidRingAbd/RingLitAbd removed"
        elif jids == NO_ABD_IDS: joint_desc = "No abduction (12) — IdxMidAbd+MidRingAbd+RingLitAbd removed"
        
        norm_desc = {'none': 'None (raw sensor values)', 'zscore': 'Z-score (per-joint mean=0, std=1)', 
                     'range': 'Range [0,1] (per-joint min-max)', 'robust': 'Robust IQR (median-centered, IQR-scaled)'}
        
        prep = f"{rep_desc.get(rep_fn.__name__, rep_fn.__name__)} | Joints: {joint_desc} | Norm: {norm_desc.get(norm_name, norm_name)}"
        
        # Build --participant shortcut
        # The synergy_myoki.py expects files in PCA_results/MyoKi/ or needs --pca_model
        save_name = f"pca_balance_{name}.npz"
        save_path = os.path.join(out_dir, save_name)
        
        # Save full model
        pc1_full = full_components[0]; pc2_full = full_components[1]
        np.savez(save_path,
                 components=full_components,
                 mean=full_mean,
                 explained_variance=evr,
                 data_label=f"balance_{name}",
                 pc1=pc1_full, pc2=pc2_full,
                 n_samples=data_subset.shape[0],
                 n_joints=n_j,
                 norm=norm_name)
        
        # Print row
        pc1_var = evr[0]*100; pc2_var = evr[1]*100; sum_var = sum(evr[:2])*100
        star = "★" if pc2_var > 10 and s1_sim > 0.9 else " "
        participant_arg = f"--pca_model {save_path}"
        print(f"{star}{idx+1:<2} {name:<50} {pc1_var:>5.1f}% {pc2_var:>5.1f}% {sum_var:>5.1f}% {s1_sim:>+6.3f} {s2_sim:>+6.3f} {n_j:>4} {n_tasks:>5} {n_part:>5} {task_desc:<40} {prep[:60]}")
        
        results.append({
            'name': name, 'file': save_name, 'participant_arg': participant_arg,
            'pc1_var': pc1_var, 'pc2_var': pc2_var, 'sum_var': sum_var,
            's1_sim': s1_sim, 's2_sim': s2_sim,
            'dof': n_j, 'n_tasks': n_tasks, 'n_subjects': n_part,
            'task_desc': task_desc, 'prep': prep
        })
    
    # Save manifest
    with open(os.path.join(out_dir, 'top_export_manifest.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 180)
    print("QUICK VISUALIZATION COMMANDS")
    print("=" * 180)
    print()
    print("python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_all_grasp_static5_flex10_robust.npz")
    print("python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_power_precision_static5_flex10_robust.npz")
    print("python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_all_grasp_trialmean_all15_none.npz")
    print("python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_precision_only_trialsd_flex10_zscore.npz")
    print("python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_all_grasp_trialrange_no_abd_none.npz")

if __name__ == '__main__':
    main()
