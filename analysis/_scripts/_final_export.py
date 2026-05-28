"""
FINAL EXPORT: Read mega_balance_results.json, extract best configs,
Reconstruct full 15-D PCA models, save to PCA_results/MyoKi/ for --participant support.
"""
import os, json, glob
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
import scipy.io as sio
from scipy.interpolate import interp1d
import openpyxl

RESULTS_FILE = 'PCA_results/experiments/mega_balance_results.json'
SAVE_DIR = 'PCA_results/MyoKi'
os.makedirs(SAVE_DIR, exist_ok=True)

N_JOINTS = 15
JOINT_NAMES = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp","IdxMCP","IdxPIP",
               "MidMCP","MidPIP","IdxMidAbd","RingMCP","RingPIP",
               "MidRingAbd","LitMCP","LitPIP","RingLitAbd"]

PISA_S1_15 = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
PISA_S2_15 = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_S2_15 = PISA_S2_15 / np.linalg.norm(PISA_S2_15)

# Joint indices
FLEXION_IDS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]
NO_ABD_IDS  = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13]
ALL15_IDS   = list(range(15))
PIP_ONLY_IDS = [2, 5, 7, 10, 13]
MCP_ONLY_IDS = [1, 4, 6, 9, 12]

JOINT_SETS = {
    'all15': ALL15_IDS, 'flex10': FLEXION_IDS, 'no_abd': NO_ABD_IDS,
    'pip_only': PIP_ONLY_IDS, 'mcp_only': MCP_ONLY_IDS,
}

# Load task info
wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
ws = wb.active
categories, task_names = {}, {}
action_cols = {'cylindrical': 11, 'spherical': 12, 'hook': 13, 'tripod': 14,
               'pinch': 15, 'lumbrical': 16, 'complex': 17, 'none_action': 18}
for row in ws.iter_rows(min_row=4, values_only=False):
    tid = row[0].value
    if tid is None: continue
    ti = int(tid)
    task_names[ti] = str(row[1].value).strip() if row[1].value else str(ti)
    acts = []
    for name, ci in action_cols.items():
        if row[ci].value is not None and str(row[ci].value).strip() == 'x':
            acts.append(name)
    categories[ti] = acts

POWER_IDS = sorted([t for t, a in categories.items()
    if any(x in {'cylindrical','spherical','hook'} for x in a) and 'none_action' not in a])
PRECISION_IDS = sorted([t for t, a in categories.items()
    if any(x in {'tripod','pinch','lumbrical'} for x in a) and 'none_action' not in a and
    not any(x in {'cylindrical','spherical','hook'} for x in a)])
ALL_GRASP_IDS = sorted(set(POWER_IDS) | set(PRECISION_IDS))
ALL_TASKS_IDS = sorted(set(categories.keys()) - {0})

def load_trials(files, task_ids):
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
                    xo = np.linspace(0, 1, len(trial))
                    xn = np.linspace(0, 1, 100)
                    f = interp1d(xo, trial, axis=0, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                    all_trials.append((f(xn), tid, os.path.basename(fpath)))
    return all_trials

def get_participant_files():
    pf = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid = [f'P{p:02d}' for p in [1,2,3,4,5,6,7,8,9,11,12]]
    return [f for f in pf if any(p in os.path.basename(f) for p in valid)]

FILES = get_participant_files()

def parse_config(label):
    """Parse experiment label to extract parameters."""
    parts = label.split('_')
    info = {}
    
    # Experiment number
    if 'e1' in parts: info['exp'] = 'e1'; info['desc'] = 'BasicPCA'
    elif 'e2' in parts: info['exp'] = 'e2'; info['desc'] = 'TaskAveraged'
    elif 'e3' in parts: info['exp'] = 'e3'; info['desc'] = 'NormCompare'
    elif 'e4' in parts: info['exp'] = 'e4'; info['desc'] = 'Ninapro'
    elif 'e9' in parts: info['exp'] = 'e9'; info['desc'] = 'OneSamplePerTask'
    elif 'e12' in parts: info['exp'] = 'e12'; info['desc'] = 'JointSubset'
    else: info['exp'] = '?'; info['desc'] = 'Mixed'
    
    # Task type
    if 'power_only' in label: info['task'] = 'Power (28)'
    elif 'precision_only' in label: info['task'] = 'Precision (13)'
    elif 'power_precision' in label: info['task'] = 'Power+Precision (41)'
    elif 'all_grasp' in label: info['task'] = 'All grasp (41)'
    elif 'all_tasks' in label: info['task'] = 'All tasks (74)'
    elif 'all_grasp_nongrasp' in label: info['task'] = 'Grasp+non-grasp'
    else: info['task'] = '?'
    
    # Data representation
    if 'trialmean' in label: info['rep'] = 'Trial mean'
    elif 'trialrange' in label: info['rep'] = 'Trial range'
    elif 'trialsd' in label: info['rep'] = 'Trial std'
    elif 'static5' in label: info['rep'] = 'Static (last 5)'
    elif 'static10' in label: info['rep'] = 'Static (last 10)'
    elif 'allframes' in label: info['rep'] = 'All frames'
    elif 'firstframe' in label: info['rep'] = 'First frame'
    elif 'lastframe' in label: info['rep'] = 'Last frame'
    else: info['rep'] = '?'
    
    # Joints
    if 'all15' in label: info['joint_set'] = 'all15'; info['jids'] = ALL15_IDS
    elif 'flex10' in label: info['joint_set'] = 'flex10'; info['jids'] = FLEXION_IDS
    elif 'no_abd' in label: info['joint_set'] = 'no_abd'; info['jids'] = NO_ABD_IDS
    elif 'pip_only' in label: info['joint_set'] = 'pip_only'; info['jids'] = PIP_ONLY_IDS
    elif 'mcp_only' in label: info['joint_set'] = 'mcp_only'; info['jids'] = MCP_ONLY_IDS
    else: info['joint_set'] = 'all15'; info['jids'] = ALL15_IDS
    
    # Normalization
    if '_none' in label: info['norm'] = 'None'
    elif '_zscore' in label: info['norm'] = 'Z-score'
    elif '_range' in label: info['norm'] = 'Range [0,1]'
    elif '_robust' in label: info['norm'] = 'Robust (IQR)'
    elif '_mean' in label: info['norm'] = 'Mean center'
    else: info['norm'] = 'None'
    
    return info

def create_pca_model(label, result):
    """Reconstruct a full 15-D PCA model from result metadata."""
    info = parse_config(label)
    jids = info['jids']
    
    # Determine task IDs
    if info['task'] == 'Power (28)':
        task_ids = POWER_IDS
    elif info['task'] == 'All grasp (41)':
        task_ids = ALL_GRASP_IDS
    elif info['task'] == 'Power+Precision (41)':
        task_ids = sorted(set(POWER_IDS) | set(PRECISION_IDS))
    elif info['task'] == 'All tasks (74)':
        task_ids = ALL_TASKS_IDS
    elif info['task'] == 'Precision (13)':
        task_ids = PRECISION_IDS
    else:
        return None
    
    # Load trials
    trials = load_trials(FILES, task_ids)
    
    # Apply data representation
    if info['rep'] == 'Trial mean':
        data_fn = lambda t: np.mean(t, axis=0)
        aggr = 'task_mean'
    elif info['rep'] == 'Trial range':
        data_fn = lambda t: np.max(t, axis=0) - np.min(t, axis=0)
        aggr = 'task_mean'
    elif info['rep'] == 'Trial std':
        data_fn = lambda t: np.std(t, axis=0)
        aggr = 'task_mean'
    elif info['rep'] == 'Static (last 5)':
        data_fn = lambda t: np.mean(t[-5:], axis=0)
        aggr = 'task_mean'
    elif info['rep'] == 'Static (last 10)':
        data_fn = lambda t: np.mean(t[-10:], axis=0)
        aggr = 'task_mean'
    elif info['rep'] == 'First frame':
        data_fn = lambda t: t[0]
        aggr = 'task_mean'
    elif info['rep'] == 'Last frame':
        data_fn = lambda t: t[-1]
        aggr = 'task_mean'
    else:
        return None
    
    # Aggregate by task
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(data_fn(t))
    
    # One sample per task (mean across trials)
    X_full = np.array([np.mean(v, axis=0) for v in by_task.values()])
    
    # Apply normalization
    norm = info['norm']
    if norm == 'Z-score':
        m = np.mean(X_full, axis=0); s = np.std(X_full, axis=0); s[s<1e-6] = 1.0
        X_norm = (X_full - m) / s
    elif norm == 'Range [0,1]':
        mins = np.min(X_full, axis=0); maxs = np.max(X_full, axis=0); r = maxs - mins; r[r<1e-6] = 1.0
        X_norm = (X_full - mins) / r
    elif norm == 'Robust (IQR)':
        med = np.median(X_full, axis=0); q75, q25 = np.percentile(X_full, [75,25], axis=0); iqr = q75 - q25; iqr[iqr<1e-6] = 1.0
        X_norm = (X_full - med) / iqr
    else:
        X_norm = X_full  # None norm
    
    # Select joints
    X_j = X_norm[:, jids]
    
    # PCA on selected joints
    n_comp = min(len(jids), X_j.shape[0], 10)
    pca = PCA(n_components=n_comp)
    pca.fit(X_j)
    evr = pca.explained_variance_ratio_
    
    # Build full 15-D components
    full_comp = np.zeros((n_comp, 15))
    full_mean = np.zeros(15)
    for i in range(n_comp):
        full_comp[i, jids] = pca.components_[i]
    full_mean = np.mean(X_full, axis=0)
    
    # Sign-flip PC1 so thumb positive
    if full_comp[0, 1] < 0:
        full_comp[0] *= -1
    if n_comp > 1 and full_comp[1, 1] < 0:
        full_comp[1] *= -1
    
    # Compute Pisa similarity
    pc1 = full_comp[0]
    pc2 = full_comp[1] if n_comp > 1 else np.zeros(15)
    
    from scipy.linalg import orthogonal_procrustes
    PC = np.column_stack([pc1, pc2])
    target = np.column_stack([PISA_S1_15, PISA_S2_15])
    R, _ = orthogonal_procrustes(PC, target)
    rot = PC @ R
    s1_sim = float(np.dot(rot[:, 0], PISA_S1_15) / (np.linalg.norm(rot[:, 0]) * np.linalg.norm(PISA_S1_15) + 1e-10))
    s2_sim = float(np.dot(rot[:, 1], PISA_S2_15) / (np.linalg.norm(rot[:, 1]) * np.linalg.norm(PISA_S2_15) + 1e-10))
    
    # Build participant shortcut name
    short_name = label.replace('experiment_', '').replace('experiment', '').replace('_none', '').replace('_zscore', '').replace('_range', '').replace('_robust', '')
    # Keep it reasonable
    short_name = label[:50]
    
    desc = (f"{info['task']} | {info['rep']} | {info['joint_set']} | {info['norm']} | "
            f"PC1={evr[0]*100:.1f}% PC2={evr[1]*100:.1f}% "
            f"S1={s1_sim:.3f} S2={s2_sim:.3f}")
    
    return full_comp, full_mean, evr, desc, s1_sim, s2_sim

def main():
    # Load the mega_balance_results
    with open(RESULTS_FILE, 'r') as f:
        all_results = json.load(f)
    
    print(f"Loaded {len(all_results)} results from {RESULTS_FILE}")
    
    # Score and find best
    def score(r):
        s1 = abs(r.get('rot_s1_sim', r.get('pc1_uniform_sim', 0)))
        s2 = abs(r.get('rot_s2_sim', r.get('pc2_gradient_sim', 0)))
        pc2 = r.get('pc2_var', 0)
        pc1 = r.get('pc1_var', 0)
        total = r.get('pc1_pc2_sum', 0)
        
        pattern_score = s1 + s2
        pc2_bonus = min(1.0, (pc2 - 0.08) * 8) if pc2 >= 0.08 else 0
        total_bonus = min(0.5, (total - 0.80) * 3) if total >= 0.80 else 0
        pc1_penalty = max(0, (pc1 - 0.90) * 10) if pc1 > 0.90 else 0
        
        return pattern_score + pc2_bonus + total_bonus - pc1_penalty
    
    all_results.sort(key=score, reverse=True)
    
    # Define the top configs to export by label pattern matching
    top_configs = [
        "power_only_static5_flex10_robust",
        "power_precision_static5_flex10_robust",
        "precision_only_trialsd_flex10_zscore",
        "precision_only_trialsd_flex10_range",
        "all_grasp_trialrange_no_abd_none",
        "power_only_trialrange_no_abd_none",
        "all_grasp_trialrange_all15_none",
        "precision_only_trialsd_all15_zscore",
        "all_grasp_trialrange_no_abd_robust",
        "all_grasp_trialmean_all15_none",
        "all_tasks_trialmean_no_abd_none",
        "power_only_static5_no_abd_robust",
        "all_grasp_static5_flex10_robust",
        "power_precision_pip_only_zscore",
    ]
    
    exported = []
    
    print(f"\n{'='*160}")
    print(f"{'#':<3} {'--participant arg':<55} {'PC1%':>6} {'PC2%':>6} {'Sum%':>6} {'S1':>7} {'S2':>7} {'DoF':>4} {'Tasks':>6} {'Subj':>5} {'Task set':<25} {'Rep':<18} {'Joints':<18} {'Norm':<14}")
    print(f"{'='*160}")
    
    for idx, label_pattern in enumerate(top_configs):
        # Find matching results
        matches = [r for r in all_results if label_pattern in r['label']]
        if not matches:
            continue
        
        best = matches[0]  # Already sorted by score
        
        # Reconstruct model
        result = create_pca_model(best['label'], best)
        if result is None:
            continue
        
        full_comp, full_mean, evr, desc, s1_sim, s2_sim = result
        pc1_var = evr[0] * 100
        pc2_var = evr[1] * 100 if len(evr) > 1 else 0
        
        # Save to PCA_results/MyoKi/
        safe_name = label_pattern.replace('experiment_', '').replace('_none', '').replace('_zscore', '').replace('_range', '').replace('_robust', '')
        save_name = f"pca_myoki_{safe_name}.npz"
        save_path = os.path.join(SAVE_DIR, save_name)
        
        np.savez(save_path,
                 components=full_comp,
                 mean=full_mean,
                 explained_variance=evr,
                 data_label=desc)
        
        # Determine joint count
        info = parse_config(best['label'])
        n_j = len(info['jids'])
        
        # Task and subject count
        n_tasks = best.get('n_samples', '?')
        n_subj = 11  # all participants
        
        star = "★" if pc2_var > 10 and s1_sim > 0.9 else " "
        print(f"{star}{idx+1:<2} --participant {safe_name:<45} {pc1_var:>5.1f}% {pc2_var:>5.1f}% {pc1_var+pc2_var:>5.1f}% {s1_sim:>+6.3f} {s2_sim:>+6.3f} {n_j:>4} {n_tasks:>6} {n_subj:>5} {info['task']:<25} {info['rep']:<18} {info['joint_set']:<18} {info['norm']:<14}")
        
        exported.append({
            'participant_arg': f"--participant {safe_name}",
            'pca_model_arg': f"--pca_model {save_path}",
            'name': safe_name,
            'pc1_var': pc1_var,
            'pc2_var': pc2_var,
            'sum_var': pc1_var + pc2_var,
            's1_sim': s1_sim,
            's2_sim': s2_sim,
            'dof': n_j,
            'n_tasks': n_tasks,
            'n_subjects': n_subj,
            'task_set': info['task'],
            'rep': info['rep'],
            'joints': info['joint_set'],
            'norm': info['norm'],
        })
    
    print()
    print("=" * 160)
    print("VISUALIZATION COMMANDS")
    print("=" * 160)
    print()
    print("# Method 1: Using --participant (looks in PCA_results/MyoKi/)")
    for e in exported[:5]:
        print(f"  python code/myoki/synergy_myoki.py {e['participant_arg']}")
    print()
    print("# Method 2: Using --pca_model (direct path)")
    for e in exported[:5]:
        print(f"  python code/myoki/synergy_myoki.py {e['pca_model_arg']}")
    print()
    
    # Save manifest
    manifest = {
        'generated': 'final_export',
        'total_exported': len(exported),
        'results': exported,
    }
    with open(os.path.join(SAVE_DIR, 'best_export_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {SAVE_DIR}/best_export_manifest.json")

if __name__ == '__main__':
    main()
