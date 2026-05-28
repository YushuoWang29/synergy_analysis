"""
MEGA EXPERIMENT V4: Combining best approaches from V2 + V3.

Key insights from V2/V3:
1. Task-averaging gives best variance (92%) but S2 is ~0.72
2. Removing abduction joints gives best S1 (0.91) but variance drops to 77%
3. Task-median gives slight S1 boost over task-mean

New explorations:
  - Task averaging + joint removal (combining best of both)
  - Per-participant task averaging (as paper does)
  - Only specific participants (those with best synergy structure)
  - Weighted PCA (down-weight noisy tasks)
  - Only power tasks that involve object manipulation (no tools)
  - Varimax + Promax rotations
  - Hierarchical approach: PCA within-subject then across-subject
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA, FactorAnalysis
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes, svd
from scipy.stats import zscore
import openpyxl

SAVE_DIR = 'PCA_results/experiments'
FIG_DIR = 'MyoKi_figures/experiments'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_JOINTS = 15
JOINT_NAMES = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
    "RingMCP","RingPIP","MidRingAbd",
    "LitMCP","LitPIP","RingLitAbd"
]

PISA_S1 = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
PISA_S2_RAW = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_S2 = PISA_S2_RAW / np.linalg.norm(PISA_S2_RAW)
PISA_TARGET = np.column_stack([PISA_S1, PISA_S2])

def load_task_info():
    wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
    ws = wb.active
    categories, task_names = {}, {}
    action_cols = {'cylindrical': 11, 'spherical': 12, 'hook': 13, 'tripod': 14,
                   'pinch': 15, 'lumbrical': 16, 'complex': 17, 'none_action': 18}
    for row in ws.iter_rows(min_row=4, values_only=False):
        tid = row[0].value
        if tid is None: continue
        tid_int = int(tid)
        name = str(row[1].value).strip() if row[1].value else f"Task {tid_int}"
        task_names[tid_int] = name
        actions = []
        for act_name, col_idx in action_cols.items():
            val = row[col_idx].value
            if val is not None and str(val).strip() == 'x':
                actions.append(act_name)
        categories[tid_int] = actions
    return categories, task_names

categories, task_names = load_task_info()
power_types = {'cylindrical', 'spherical', 'hook'}
grasp_types = {'cylindrical', 'spherical', 'hook', 'tripod', 'pinch', 'lumbrical'}

ALL_GRASP_IDS = set()
POWER_GRASP_IDS = set()
for tid, acts in categories.items():
    has_grasp = any(a in grasp_types for a in acts)
    is_none = 'none_action' in acts
    if has_grasp and not is_none:
        ALL_GRASP_IDS.add(tid)
        if any(a in power_types for a in acts):
            POWER_GRASP_IDS.add(tid)

POWER_GRASP_IDS_NO_TOOLS = {t for t in POWER_GRASP_IDS if t < 46}
POWER_GRASP_IDS_CORE = {1,2,3,4,5,6,9,10,11,12,13,14,15,40,41,42,43,44,45}

# Joint subsets
JOINT_SUBSETS = {
    'all15': list(range(15)),
    'no_abd': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],
    'no_thumb_abd': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'flexion_only': [1, 2, 4, 5, 6, 7, 9, 10, 12, 13],
    'no_abduction_keep_thumb': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],
    'thumb_plus_flexion': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],  # same as no_abd
}

def get_pisa_target_subset(jids):
    n = len(jids)
    s1 = np.ones(n) / np.linalg.norm(np.ones(n))
    # Map Pisa target values to subset
    raw = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
    s2_raw = raw[jids]
    s2 = s2_raw / np.linalg.norm(s2_raw)
    return s1, s2, np.column_stack([s1, s2])

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

def evaluate_predictions(pc1, pc2, target=None):
    if target is None:
        target = PISA_TARGET
    PC = np.column_stack([pc1, pc2])
    # Handle sign flips
    for c in range(2):
        if np.dot(PC[:, c], target[:, c]) < 0:
            PC[:, c] *= -1
    R, _ = orthogonal_procrustes(PC, target)
    rotated = PC @ R
    s1_sim = np.dot(rotated[:, 0], target[:, 0]) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(target[:, 0]))
    s2_sim = np.dot(rotated[:, 1], target[:, 1]) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(target[:, 1]))
    return s1_sim, s2_sim, rotated, R

def run_pca_custom(data, label, jids=None):
    if len(data) < 5 or len(data[0]) < 2: return None
    data = np.array(data, dtype=float)
    if np.isnan(data).any() or np.isinf(data).any(): return None
    
    if jids is not None:
        data = data[:, jids]
        target = get_pisa_target_subset(jids)[2]
    else:
        target = PISA_TARGET
    
    n_comp = min(15, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(data)
    evr = pca.explained_variance_ratio_
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
    
    s1_sim, s2_sim, rotated, R = evaluate_predictions(pc1, pc2, target)
    
    return {
        'label': label, 'n_samples': data.shape[0], 'n_joints': data.shape[1],
        'pc1_var': float(evr[0]), 'pc2_var': float(evr[1]),
        'pc1_pc2_sum': float(evr[0] + evr[1]),
        's1_sim': float(s1_sim), 's2_sim': float(s2_sim),
        'rot_sum': float(s1_sim + s2_sim),
    }

def main():
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]
    
    all_results = []
    
    print("=" * 70)
    print("V4 EXPERIMENTS")
    print("=" * 70)
    
    power_trials = load_myoki_trials(files, POWER_GRASP_IDS)
    all_trials = load_myoki_trials(files, ALL_GRASP_IDS)
    power_no_tools_trials = load_myoki_trials(files, POWER_GRASP_IDS_NO_TOOLS)
    
    # ============================================
    # E1: Task-averaging + Joint Subsets (combining best ideas)
    # ============================================
    print(f"\n--- E1: TASK-AVERAGING + JOINT SUBSETS ---")
    
    for tname, trials in [('power', power_trials), ('allgrasp', all_trials)]:
        # Build task-averaged postures
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-5:], axis=0))
        
        task_data = np.array([np.mean(v, axis=0) for v in by_task.values()])
        
        for sname, jids in JOINT_SUBSETS.items():
            result = run_pca_custom(task_data, f"v4_{tname}_taskavg_{sname}", jids=jids)
            if result:
                print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}, "
                      f"Score={result['rot_sum']:.4f}")
                all_results.append(result)
    
    # ============================================
    # E2: Task-median + Joint Subsets
    # ============================================
    print(f"\n--- E2: TASK-MEDIAN + JOINT SUBSETS ---")
    
    for tname, trials in [('power', power_trials), ('allgrasp', all_trials)]:
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-5:], axis=0))
        
        task_data = np.array([np.median(v, axis=0) for v in by_task.values()])
        
        for sname in ['all15', 'no_abd']:
            jids = JOINT_SUBSETS[sname]
            result = run_pca_custom(task_data, f"v4_{tname}_taskmedian_{sname}", jids=jids)
            if result:
                print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                all_results.append(result)
    
    # ============================================
    # E3: Per-participant PCA then average the PCs
    # ============================================
    print(f"\n--- E3: PER-PARTICIPANT PCA THEN AVERAGE ---")
    
    all_pc1s, all_pc2s = [], []
    for fpath in files:
        pname = os.path.basename(fpath).replace('.mat', '')
        # Get this participant's trials
        mat = sio.loadmat(fpath)
        glove = np.nan_to_num(mat['glove_calibrated'])
        tasks = mat['task'].flatten().astype(int)
        hand_data = glove[:, :15]
        
        # Collect static postures for power grasp tasks
        subject_postures = []
        for tid in POWER_GRASP_IDS:
            mask = (tasks == tid)
            diff = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0]
            for s, e in zip(starts, ends):
                trial = hand_data[s:e, :]
                if len(trial) >= 10:
                    subject_postures.append(np.mean(trial[-5:], axis=0))
        
        if len(subject_postures) < 5: continue
        subj_data = np.array(subject_postures)
        
        # PCA per participant
        pca = PCA(n_components=2)
        pca.fit(subj_data)
        pc1 = pca.components_[0] / np.linalg.norm(pca.components_[0])
        pc2 = pca.components_[1] / np.linalg.norm(pca.components_[1])
        
        # Align signs
        if np.sum(pc1) < 0: pc1 *= -1
        if np.sum(pc2 * np.sign(PISA_S2_RAW)) < 0: pc2 *= -1  # S2 has gradient
        
        total_var = pca.explained_variance_ratio_[0]*100
        all_pc1s.append(pc1)
        all_pc2s.append(pc2)
        print(f"  {pname}: {len(subject_postures)} trials, "
              f"Var={total_var:.1f}%")
    
    if len(all_pc1s) >= 3:
        avg_pc1 = np.mean(all_pc1s, axis=0)
        avg_pc1 /= np.linalg.norm(avg_pc1)
        avg_pc2 = np.mean(all_pc2s, axis=0)
        avg_pc2 /= np.linalg.norm(avg_pc2)
        
        s1_sim, s2_sim, _, _ = evaluate_predictions(avg_pc1, avg_pc2)
        print(f"\n  Averaged PCs: S1={s1_sim:.4f}, S2={s2_sim:.4f}, Score={s1_sim+s2_sim:.4f}")
        all_results.append({
            'label': 'v4_avg_subject_PCs',
            'n_samples': len(all_pc1s),
            'pc1_var': 0, 'pc2_var': 0, 'pc1_pc2_sum': 0,
            's1_sim': float(s1_sim), 's2_sim': float(s2_sim),
            'rot_sum': float(s1_sim + s2_sim),
        })
    
    # ============================================
    # E4: Best individual subjects (power grasp task-averaged)
    # ============================================
    print(f"\n--- E4: BEST INDIVIDUAL SUBJECTS ---")
    
    for fpath in files:
        pname = os.path.basename(fpath).replace('.mat', '')
        mat = sio.loadmat(fpath)
        glove = np.nan_to_num(mat['glove_calibrated'])
        tasks = mat['task'].flatten().astype(int)
        hand_data = glove[:, :15]
        
        # Get task-averaged postures for this subject
        by_task = {}
        for tid in POWER_GRASP_IDS:
            mask = (tasks == tid)
            diff = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0]
            for s, e in zip(starts, ends):
                trial = hand_data[s:e, :]
                if len(trial) >= 10:
                    if tid not in by_task:
                        by_task[tid] = []
                    by_task[tid].append(np.mean(trial[-5:], axis=0))
        
        if len(by_task) < 5: continue
        
        task_data = np.array([np.mean(v, axis=0) for v in by_task.values()])
        result = run_pca_custom(task_data, f"v4_{pname}_taskavg_static5")
        if result:
            print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
        
        # Also with no_abd
        result2 = run_pca_custom(task_data, f"v4_{pname}_taskavg_no_abd", jids=JOINT_SUBSETS['no_abd'])
        if result2:
            print(f"  {result2['label']}: Var={result2['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result2['s1_sim']:.4f}, S2={result2['s2_sim']:.4f}")
            all_results.append(result2)
    
    # ============================================
    # E5: Static10 instead of Static5 (more stable holds)
    # ============================================
    print(f"\n--- E5: STATIC10 (LONGER HOLD) ---")
    
    for tname, trials in [('power', power_trials)]:
        # Task-average
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-10:], axis=0))  # longer hold
        
        task_data = np.array([np.mean(v, axis=0) for v in by_task.values()])
        
        for sname in ['all15', 'no_abd']:
            result = run_pca_custom(task_data, f"v4_{tname}_taskavg_static10_{sname}", 
                                   jids=JOINT_SUBSETS[sname])
            if result:
                print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                all_results.append(result)
    
    # ============================================
    # E6: Try all-frames (not just static) with task-averaging and no_abd
    # ============================================
    print(f"\n--- E6: ALL-FRAMES TASK-AVERAGED ---")
    
    for tname, trials in [('power', power_trials)]:
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t, axis=0))  # average across all frames
        
        task_data = np.array([np.mean(v, axis=0) for v in by_task.values()])
        
        for sname in ['all15', 'no_abd']:
            result = run_pca_custom(task_data, f"v4_{tname}_taskavg_mean_{sname}",
                                   jids=JOINT_SUBSETS[sname])
            if result:
                print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                all_results.append(result)
    
    # ============================================
    # E7: Tukey winsorized mean (robust to outliers)
    # ============================================
    print(f"\n--- E7: ROBUST MEAN (WINSORIZED) ---")
    
    def winsorized_mean(data, limits=0.1):
        """Compute winsorized mean."""
        n = len(data)
        k = int(n * limits)
        if k == 0: return np.mean(data, axis=0)
        sorted_data = np.sort(data, axis=0)
        sorted_data[:k] = sorted_data[k]
        sorted_data[-k:] = sorted_data[-k-1]
        return np.mean(sorted_data, axis=0)
    
    for tname, trials in [('power', power_trials)]:
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-5:], axis=0))
        
        task_data = np.array([np.mean(v, axis=0) for v in by_task.values()])
        # Winsorized across tasks (not within tasks)
        task_data_w = np.array([winsorized_mean(v, 0.1) for v in by_task.values()])
        
        result = run_pca_custom(task_data_w, f"v4_{tname}_taskwinsorized_static5")
        if result:
            print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
    
    # ============================================
    # RANKING
    # ============================================
    seen = set()
    unique = []
    for r in all_results:
        if r['label'] not in seen:
            seen.add(r['label'])
            unique.append(r)
    
    # Sort by combined score (S1+S2), requiring at least 70% variance
    unique.sort(key=lambda r: r['rot_sum'] if r['pc1_pc2_sum'] >= 0.7 or r['pc1_pc2_sum'] == 0 else r['rot_sum'] - 1, reverse=True)
    
    print(f"\n{'='*130}")
    print(f"V4 RESULTS RANKED (by S1+S2, with ★ for Var>=80%)")
    print(f"{'='*130}")
    print(f"{'Rank':<5} {'Experiment':<55} {'N':>5} {'Var%':>7} {'S1':>8} {'S2':>8} {'Score':>8}")
    print("-" * 130)
    
    for i, r in enumerate(unique[:40]):
        star = "★" if r['pc1_pc2_sum'] >= 0.80 and r['pc1_pc2_sum'] > 0 else " "
        check = "✓" if r['pc1_pc2_sum'] >= 0.80 and r['pc1_pc2_sum'] > 0 else " "
        var_str = f"{r['pc1_pc2_sum']*100:>6.1f}%" if r['pc1_pc2_sum'] > 0 else "  N/A "
        print(f"{star}{i+1:<4} {r['label']:<55} {r.get('n_samples',0):>5} "
              f"{var_str} "
              f"{r['s1_sim']:>+7.4f} {r['s2_sim']:>+7.4f} {r['rot_sum']:>7.4f} {check}")
    
    # Save
    clean = []
    for r in unique:
        c = {}
        for k, v in r.items():
            if isinstance(v, (np.floating,)): c[k] = float(v)
            elif isinstance(v, (np.integer,)): c[k] = int(v)
            elif isinstance(v, (np.ndarray,)): c[k] = v.tolist()
            else: c[k] = v
        clean.append(c)
    
    with open(os.path.join(SAVE_DIR, 'mega_v4_results.json'), 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    
    # Combined with all previous
    for vfile in ['all_results_combined.json', 'mega_v3_results.json', 'mega_v2_results.json']:
        vpath = os.path.join(SAVE_DIR, vfile)
        if os.path.exists(vpath):
            with open(vpath) as f:
                v_results = json.load(f)
    
    all_combined = [] + clean
    seen_labels = {r['label'] for r in all_combined}
    
    for vfile in ['all_results_combined.json', 'mega_v3_results.json', 'mega_v2_results.json', 'mega_experiment_results.json']:
        vpath = os.path.join(SAVE_DIR, vfile)
        if os.path.exists(vpath):
            with open(vpath) as f:
                v_results = json.load(f)
            for r in v_results:
                if r.get('label', '') not in seen_labels:
                    all_combined.append(r)
                    seen_labels.add(r['label'])
    
    all_combined.sort(key=lambda r: r.get('rot_sum', 0), reverse=True)
    
    with open(os.path.join(SAVE_DIR, 'all_experiments_ranked.json'), 'w') as f:
        json.dump(all_combined, f, indent=2, default=str)
    
    print(f"\n{'='*130}")
    print(f"TOP 40 ALL EXPERIMENTS (across all V2+V3+V4)")
    print(f"{'='*130}")
    print(f"{'Rank':<5} {'Experiment':<55} {'Var%':>7} {'S1':>8} {'S2':>8} {'Score':>8}")
    print("-" * 130)
    
    for i, r in enumerate(all_combined[:40]):
        star = "★" if i == 0 else " "
        var_str = f"{r.get('pc1_pc2_sum', 0)*100:>6.1f}%" if r.get('pc1_pc2_sum', 0) > 0 else "  N/A "
        print(f"{star}{i+1:<4} {r.get('label',''):<55} "
              f"{var_str} "
              f"{r.get('s1_sim', 0):>+7.4f} {r.get('s2_sim', 0):>+7.4f} "
              f"{r.get('rot_sum', r.get('s1_sim',0)+r.get('s2_sim',0)):>7.4f}")
    
    print(f"\nDone! {len(all_combined)} total experiments ranked.")

if __name__ == '__main__':
    main()
