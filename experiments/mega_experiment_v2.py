"""
MEGA EXPERIMENT V2: Focused refinements to push closer to Pisa/IIT targets.

Based on V1 results, the best approach is:
  - MyoKi power grasps, static hold phase (last 5-10 frames), trial means, no normalization
  - But we need to push S2 similarity higher and variance above 80%

New strategies:
  1. Per-participant normalization before pooling
  2. Individual rest posture subtraction
  3. Selective participant subsets
  4. Within-trial centering (subtract trial's own mean)
  5. ReLU-like filtering (positive-only for PC1)
  6. Mix of power + precision grasps
  7. Median instead of mean for trial stats
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# =============================
# Helpers
# =============================
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
grasp_types = {'cylindrical', 'spherical', 'hook', 'tripod', 'pinch', 'lumbrical'}
power_types = {'cylindrical', 'spherical', 'hook'}
precision_types = {'tripod', 'pinch', 'lumbrical'}

ALL_GRASP_IDS = set()
POWER_GRASP_IDS = set()
PRECISION_GRASP_IDS = set()
for tid, acts in categories.items():
    has_grasp = any(a in grasp_types for a in acts)
    is_none = 'none_action' in acts
    if has_grasp and not is_none:
        ALL_GRASP_IDS.add(tid)
        if any(a in power_types for a in acts):
            POWER_GRASP_IDS.add(tid)
        if any(a in precision_types for a in acts) and not any(a in power_types for a in acts):
            PRECISION_GRASP_IDS.add(tid)

def load_myoki_trials(files, task_ids, resample_n=100):
    """Load and resample all trials."""
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

def get_static_postures(trials, n_last_frames=5):
    """Get static hold posture from last N frames of each trial."""
    data_list = []
    trial_labels = []
    for t, tid, subj in trials:
        static = np.mean(t[-n_last_frames:], axis=0)
        data_list.append(static)
        trial_labels.append((tid, subj))
    return np.array(data_list), trial_labels

def get_trial_means(trials):
    """Get full trial means."""
    return np.array([np.mean(t, axis=0) for t, tid, subj in trials])

def evaluate_predictions(pc1, pc2):
    """Compute metrics after procrustes rotation."""
    PC = np.column_stack([pc1, pc2])
    R, _ = orthogonal_procrustes(PC, PISA_TARGET)
    rotated = PC @ R
    
    s1_sim = np.dot(rotated[:, 0], PISA_S1) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(PISA_S1))
    s2_sim = np.dot(rotated[:, 1], PISA_S2) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(PISA_S2))
    
    return s1_sim, s2_sim, rotated, R

def run_pca(data, label):
    """Run PCA and return result dict."""
    if len(data) < 5:
        return None
    data = np.array(data, dtype=float)
    if np.isnan(data).any() or np.isinf(data).any():
        return None
    
    n_comp = min(15, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(data)
    evr = pca.explained_variance_ratio_
    
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
    
    s1_sim, s2_sim, rotated, R = evaluate_predictions(pc1, pc2)
    
    result = {
        'label': label,
        'n_samples': data.shape[0],
        'pc1_var': float(evr[0]),
        'pc2_var': float(evr[1]),
        'pc1_pc2_sum': float(evr[0] + evr[1]),
        's1_sim': float(s1_sim),
        's2_sim': float(s2_sim),
        'rot_sum': float(s1_sim + s2_sim),
        'pc1': pc1.tolist(),
        'pc2': pc2.tolist(),
        'rot_s1': rotated[:, 0].tolist(),
        'rot_s2': rotated[:, 1].tolist(),
        'R_mat': R.tolist(),
    }
    return result

# =============================
# EXPERIMENTS
# =============================

def run_all_v2():
    """Run all V2 experiments."""
    all_results = []
    
    # Load participants
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]
    
    print(f"{'='*70}")
    print(f"V2 EXPERIMENTS: {len(files)} participants")
    print(f"{'='*70}")
    
    # Load trials for power grasps
    power_trials = load_myoki_trials(files, POWER_GRASP_IDS)
    all_trials = load_myoki_trials(files, ALL_GRASP_IDS)
    print(f"Power trials: {len(power_trials)}")
    print(f"All grasp trials: {len(all_trials)}")
    
    # ============================================
    # EXPERIMENT 1: Per-participant normalization
    # ============================================
    print(f"\n--- E1: PER-PARTICIPANT NORMALIZED POOLING ---")
    
    # For each participant, normalize data individually, then pool
    for task_name, trials_dict in [('power', power_trials), ('allgrasp', all_trials)]:
        for norm in ['none', 'zscore', 'range']:
            for rep in ['mean', 'static5', 'static10']:
                # Group by participant
                by_subj = {}
                for t, tid, subj in trials_dict:
                    if subj not in by_subj:
                        by_subj[subj] = []
                    by_subj[subj].append((t, tid))
                
                pooled = []
                for subj, st in by_subj.items():
                    if rep == 'mean':
                        local_data = np.array([np.mean(t, axis=0) for t, _ in st])
                    elif rep == 'static5':
                        local_data = np.array([np.mean(t[-5:], axis=0) for t, _ in st])
                    else:
                        local_data = np.array([np.mean(t[-10:], axis=0) for t, _ in st])
                    
                    # Normalize per participant
                    if norm == 'zscore':
                        m = np.mean(local_data, axis=0)
                        s = np.std(local_data, axis=0)
                        s[s < 1e-6] = 1.0
                        local_data = (local_data - m) / s
                    elif norm == 'range':
                        mins = np.min(local_data, axis=0)
                        maxs = np.max(local_data, axis=0)
                        r = maxs - mins
                        r[r < 1e-6] = 1.0
                        local_data = (local_data - mins) / r
                    
                    pooled.append(local_data)
                
                if pooled:
                    data = np.vstack(pooled)
                    label = f"v2_{task_name}_perpnorm{norm}_{rep}"
                    result = run_pca(data, label)
                    if result:
                        print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                              f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                        all_results.append(result)
    
    # ============================================
    # EXPERIMENT 2: Rest posture subtraction per participant
    # ============================================
    print(f"\n--- E2: REST POSTURE SUBTRACTION ---")
    
    for task_name, trials_dict in [('power', power_trials), ('allgrasp', all_trials)]:
        # Compute rest posture per participant
        by_subj_data = {}
        for fpath in files:
            subj = os.path.basename(fpath).replace('.mat', '')
            mat = sio.loadmat(fpath)
            glove = np.nan_to_num(mat['glove_calibrated'])
            tasks = mat['task'].flatten().astype(int)
            hand_data = glove[:, :15]
            rest_mask = (tasks == 0)
            if rest_mask.sum() > 0:
                by_subj_data[subj] = np.mean(hand_data[rest_mask], axis=0)
            else:
                by_subj_data[subj] = np.mean(hand_data, axis=0)
        
        by_subj = {}
        for t, tid, subj in trials_dict:
            if subj not in by_subj:
                by_subj[subj] = []
            by_subj[subj].append((t, tid))
        
        for rep in ['mean', 'static5']:
            pooled_restsub = []
            for subj, st in by_subj.items():
                if rep == 'mean':
                    local_data = np.array([np.mean(t, axis=0) for t, _ in st])
                else:
                    local_data = np.array([np.mean(t[-5:], axis=0) for t, _ in st])
                
                # Subtract rest posture
                rp = by_subj_data.get(subj, np.zeros(15))
                local_data = local_data - rp[np.newaxis, :]
                pooled_restsub.append(local_data)
            
            if pooled_restsub:
                data = np.vstack(pooled_restsub)
                label = f"v2_{task_name}_restsub_{rep}"
                result = run_pca(data, label)
                if result:
                    print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                          f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                    all_results.append(result)
    
    # ============================================
    # EXPERIMENT 3: Within-trial centering
    # ============================================
    print(f"\n--- E3: WITHIN-TRIAL CENTERING ---")
    
    power_tm = np.array([np.mean(t, axis=0) for t, tid, subj in power_trials])
    power_static5 = np.array([np.mean(t[-5:], axis=0) for t, tid, subj in power_trials])
    
    # Subtract each trial's own mean → shape-only
    trial_full = np.array([t for t, tid, subj in power_trials])
    trial_means_all = np.array([np.mean(t, axis=0) for t in trial_full])
    trial_mean_centered = power_tm - trial_means_all
    
    for label, data in [
        ('v2_trial_centered_power_mean', trial_mean_centered),
    ]:
        result = run_pca(data, label)
        if result:
            print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
    
    # ============================================
    # EXPERIMENT 4: Best 3 participants combined (carefully)
    # ============================================
    print(f"\n--- E4: BEST PARTICIPANTS COMBINED ---")
    
    # Find best participants from individual analysis
    indiv_scores = {}
    for fpath in files:
        subj = os.path.basename(fpath).replace('.mat', '')
        pt = [x for x in power_trials if x[2] == subj + '.mat']
        if len(pt) < 10: continue
        data_p = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in pt])
        r = run_pca(data_p, subj)
        if r:
            indiv_scores[subj] = r
    
    best_ps = sorted(indiv_scores.keys(), key=lambda s: indiv_scores[s]['rot_sum'], reverse=True)
    print(f"Best participants: {best_ps[:5]}")
    
    for n_top in [2, 3, 4, 5]:
        selected = best_ps[:n_top]
        selected_files = [f for f in files if any(p in os.path.basename(f) for p in selected)]
        selected_trials = load_myoki_trials(selected_files, POWER_GRASP_IDS)
        
        for rep in ['mean', 'static5']:
            if rep == 'mean':
                data = np.array([np.mean(t, axis=0) for t, tid, _ in selected_trials])
            else:
                data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in selected_trials])
            
            # Try with and without per-participant range normalization
            for norm_label, norm_data in [('raw', data)]:
                label = f"v2_top{n_top}_{rep}_{norm_label}"
                result = run_pca(norm_data, label)
                if result:
                    print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                          f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                    all_results.append(result)
    
    # ============================================
    # EXPERIMENT 5: All tasks BUT with per-task averaging
    # ============================================
    print(f"\n--- E5: PER-TASK AVERAGING ---")
    
    # Average all trials of the same task together (across participants)
    # This reduces noise from individual variation
    for task_name, trials_dict in [('power', power_trials), ('allgrasp', all_trials)]:
        by_task = {}
        for t, tid, subj in trials_dict:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-5:], axis=0))
        
        task_avgs = np.array([np.mean(v, axis=0) for v in by_task.values()])
        
        label = f"v2_{task_name}_taskavg_static5"
        result = run_pca(task_avgs, label)
        if result:
            print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
    
    # ============================================
    # EXPERIMENT 6: Only participants with highest variance
    # ============================================
    print(f"\n--- E6: HIGH-VARIANCE PARTICIPANTS ---")
    
    indiv_vars = {s: indiv_scores[s]['pc1_pc2_sum'] for s in indiv_scores}
    sorted_by_var = sorted(indiv_vars.items(), key=lambda x: x[1], reverse=True)
    print(f"By variance: {sorted_by_var[:5]}")
    
    for n_top in [3, 5]:
        selected = [s for s, _ in sorted_by_var[:n_top]]
        selected_files = [f for f in files if any(p in os.path.basename(f) for p in selected)]
        selected_trials = load_myoki_trials(selected_files, POWER_GRASP_IDS)
        
        data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in selected_trials])
        label = f"v2_topvar{n_top}_static5"
        result = run_pca(data, label)
        if result:
            print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
    
    # ============================================
    # EXPERIMENT 7: Mix power + precision grasps
    # ============================================
    print(f"\n--- E7: MIXED POWER+PRECISION ---")
    
    # The Pisa paper includes all grasps. Precision grasps add gradient information.
    power_p_precision_ids = POWER_GRASP_IDS | PRECISION_GRASP_IDS
    mixed_trials = load_myoki_trials(files, power_p_precision_ids)
    print(f"Mixed trials: {len(mixed_trials)}")
    
    for rep in ['mean', 'static5', 'static10']:
        if rep == 'mean':
            data = np.array([np.mean(t, axis=0) for t, tid, _ in mixed_trials])
        elif rep == 'static5':
            data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in mixed_trials])
        else:
            data = np.array([np.mean(t[-10:], axis=0) for t, tid, _ in mixed_trials])
        
        for norm in ['none', 'range_per_subj']:
            if norm == 'range_per_subj':
                by_subj = {}
                for t, tid, subj in mixed_trials:
                    if subj not in by_subj:
                        by_subj[subj] = []
                    by_subj[subj].append((t, tid))
                
                pooled = []
                for subj, st in by_subj.items():
                    ld = np.array([np.mean(t[-5:], axis=0) for t, _ in st]) if rep == 'static5' else np.array([np.mean(t, axis=0) for t, _ in st])
                    mins = np.min(ld, axis=0)
                    maxs = np.max(ld, axis=0)
                    r = maxs - mins
                    r[r < 1e-6] = 1.0
                    ld = (ld - mins) / r
                    pooled.append(ld)
                data = np.vstack(pooled)
            
            label = f"v2_mixedPowerPrec_{rep}_{norm}"
            result = run_pca(data, label)
            if result:
                print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                all_results.append(result)
    
    # ============================================
    # EXPERIMENT 8: Remove thumb opposition (noise)
    # ============================================
    print(f"\n--- E8: REMOVE THUMB OPPOSITION ---")
    
    no_thumbopp_idx = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    power_static5 = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in power_trials])
    
    data_noopp = power_static5[:, no_thumbopp_idx]
    
    # Redefine S2 for this subspace
    n_j = len(no_thumbopp_idx)
    s1_sub = np.ones(n_j) / np.linalg.norm(np.ones(n_j))
    s2_raw_sub = np.array([2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
    s2_sub = s2_raw_sub / np.linalg.norm(s2_raw_sub)
    
    pca = PCA(n_components=2)
    pca.fit(data_noopp)
    pc1 = pca.components_[0] / np.linalg.norm(pca.components_[0])
    pc2 = pca.components_[1] / np.linalg.norm(pca.components_[1])
    
    PC = np.column_stack([pc1, pc2])
    target = np.column_stack([s1_sub, s2_sub])
    R, _ = orthogonal_procrustes(PC, target)
    rotated = PC @ R
    
    s1_sim = np.dot(rotated[:, 0], s1_sub)
    s2_sim = np.dot(rotated[:, 1], s2_sub)
    
    label = f"v2_power_noThumbOpp_static5"
    print(f"  {label}: S1={s1_sim:.4f}, S2={s2_sim:.4f}")
    
    # ============================================
    # EXPERIMENT 9: Multiple static frames - use ALL static hold frames
    # not averaged → more data points
    # ============================================
    print(f"\n--- E9: ALL STATIC HOLD FRAMES (NOT AVERAGED) ---")
    
    all_static_frames = []
    for t, tid, subj in power_trials:
        static_frames = t[-10:]  # last 10 frames
        for frame in static_frames:
            all_static_frames.append(frame)
    
    data = np.array(all_static_frames)
    label = f"v2_power_allstaticframes"
    result = run_pca(data, label)
    if result:
        print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
              f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
        all_results.append(result)
    
    # ============================================
    # EXPERIMENT 10: Only grasp start (transition from rest)
    # ============================================
    print(f"\n--- E10: GRASP TRANSITION (FIRST FRAMES AFTER REST) ---")
    
    all_start_frames = []
    for t, tid, subj in power_trials:
        start_frames = t[:10]  # first 10 frames
        for frame in start_frames:
            all_start_frames.append(frame)
    
    data = np.array(all_start_frames)
    label = f"v2_power_transition_allframes"
    result = run_pca(data, label)
    if result:
        print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
              f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
        all_results.append(result)
    
    # ============================================
    # EXPERIMENT 11: P07 alone (best participant) with variations
    # ============================================
    print(f"\n--- E11: P07 DEEP DIVE ---")
    
    p07_file = [f for f in files if 'P07' in f]
    if p07_file:
        p07_trials = load_myoki_trials(p07_file, POWER_GRASP_IDS)
        print(f"  P07: {len(p07_trials)} power trials")
        
        for rep in ['mean', 'static5', 'static10']:
            if rep == 'mean':
                data = np.array([np.mean(t, axis=0) for t, tid, _ in p07_trials])
            elif rep == 'static5':
                data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in p07_trials])
            else:
                data = np.array([np.mean(t[-10:], axis=0) for t, tid, _ in p07_trials])
            
            for norm in ['none', 'range']:
                d = data.copy()
                if norm == 'range':
                    mins = np.min(d, axis=0)
                    maxs = np.max(d, axis=0)
                    r = maxs - mins
                    r[r < 1e-6] = 1.0
                    d = (d - mins) / r
                
                label = f"v2_P07_power_{rep}_{norm}"
                result = run_pca(d, label)
                if result:
                    print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                          f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                    all_results.append(result)
    
    # ============================================
    # EXPERIMENT 12: PISA-style "mean center but use trial averages"
    # The Pisa paper averages across participants for each grasp type
    # ============================================
    print(f"\n--- E12: PISA-STYLE AVERAGING ---")
    
    # Average across all participants for each task ID → one posture per task
    by_task = {}
    for t, tid, subj in power_trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(np.mean(t[-5:], axis=0))
    
    task_vectors = {}
    for tid, postures in by_task.items():
        task_vectors[tid] = np.mean(postures, axis=0)
    
    task_data = np.array(list(task_vectors.values()))
    task_ids = list(task_vectors.keys())
    print(f"  {len(task_data)} task-average postures")
    
    label = "v2_PisaStyle_taskavg"
    result = run_pca(task_data, label)
    if result:
        print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
              f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
        all_results.append(result)
    
    # ============================================
    # EXPERIMENT 13: Bilateral normalization (per-joint, per-participant)
    # Scale each joint to [-1, 1] based on min/max per participant
    # ============================================
    print(f"\n--- E13: BILATERAL JOINT NORMALIZATION ([-1,1] per participant) ---")
    
    by_subj_power = {}
    for t, tid, subj in power_trials:
        if subj not in by_subj_power:
            by_subj_power[subj] = []
        by_subj_power[subj].append((t, tid))
    
    pooled_bilat = []
    for subj, st in by_subj_power.items():
        local_data = np.array([np.mean(t[-5:], axis=0) for t, _ in st])
        # Center at mid-range
        mins = np.min(local_data, axis=0)
        maxs = np.max(local_data, axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-6] = 1.0
        # Scale to [-1, 1]: 2*(x - min)/(max-min) - 1
        local_bilat = 2 * (local_data - mins) / ranges - 1
        pooled_bilat.append(local_bilat)
    
    data_bilat = np.vstack(pooled_bilat)
    label = "v2_power_bilateralNorm_static5"
    result = run_pca(data_bilat, label)
    if result:
        print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
              f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
        all_results.append(result)
    
    # ============================================
    # EXPERIMENT 14: Per-participant z-score applied to combined data
    # Different from E1: normalize EACH TRIAL's features separately
    # ============================================
    print(f"\n--- E14: PER-TRIAL FEATURE NORMALIZATION ---")
    
    for task_name, trials_dict in [('power', power_trials)]:
        for rep in ['mean', 'static5']:
            if rep == 'mean':
                raw_data = np.array([np.mean(t, axis=0) for t, tid, _ in trials_dict])
            else:
                raw_data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in trials_dict])
            
            # Per-trial: normalize each trial's joints to z-scores
            trial_normed = np.zeros_like(raw_data)
            for i in range(len(raw_data)):
                m = np.mean(raw_data[i])
                s = np.std(raw_data[i])
                if s > 1e-6:
                    trial_normed[i] = (raw_data[i] - m) / s
                else:
                    trial_normed[i] = raw_data[i]
            
            label = f"v2_{task_name}_perTrialZscore_{rep}"
            result = run_pca(trial_normed, label)
            if result:
                print(f"  {label}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                all_results.append(result)
    
    # ============================================
    # EXPERIMENT 15: Remove the thumb (which has extra DOFs)
    # ============================================
    print(f"\n--- E15: EXCLUDE THUMB ENTIRELY ---")
    
    no_thumb = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    power_static5 = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in power_trials])
    
    data_nothumb = power_static5[:, no_thumb]
    n_j = len(no_thumb)
    s1_nt = np.ones(n_j) / np.linalg.norm(np.ones(n_j))
    s2_nt_raw = np.array([1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
    s2_nt = s2_nt_raw / np.linalg.norm(s2_nt_raw)
    
    pca = PCA(n_components=2)
    pca.fit(data_nothumb)
    evr = pca.explained_variance_ratio_
    pc1 = pca.components_[0] / np.linalg.norm(pca.components_[0])
    pc2 = pca.components_[1] / np.linalg.norm(pca.components_[1])
    
    PC = np.column_stack([pc1, pc2])
    target = np.column_stack([s1_nt, s2_nt])
    R, _ = orthogonal_procrustes(PC, target)
    rotated = PC @ R
    
    s1_sim = np.dot(rotated[:, 0], s1_nt)
    s2_sim = np.dot(rotated[:, 1], s2_nt)
    
    result = {
        'label': 'v2_power_noThumb_static5',
        'n_samples': len(data_nothumb),
        'n_joints': n_j,
        'pc1_var': float(evr[0]),
        'pc2_var': float(evr[1]),
        'pc1_pc2_sum': float(evr[0] + evr[1]),
        's1_sim': float(s1_sim),
        's2_sim': float(s2_sim),
        'rot_sum': float(s1_sim + s2_sim),
    }
    print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
          f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
    all_results.append(result)
    
    return all_results


# =============================
# RANKING
# =============================
def rank_results(all_results):
    """Rank all results and print table."""
    if not all_results:
        print("No results!")
        return []
    
    # Deduplicate
    seen = set()
    unique = []
    for r in all_results:
        if r['label'] not in seen:
            seen.add(r['label'])
            unique.append(r)
    
    unique.sort(key=lambda r: r['rot_sum'], reverse=True)
    
    print(f"\n{'='*150}")
    print(f"V2 RESULTS RANKED (by S1+S2)")
    print(f"{'='*150}")
    print(f"{'Rank':<5} {'Experiment':<50} {'N':>7} {'Var%':>7} {'S1':>8} {'S2':>8} {'Score':>8}")
    print("-" * 150)
    
    for i, r in enumerate(unique):
        star = "★" if i == 0 else " "
        print(f"{star}{i+1:<4} {r['label']:<50} {r.get('n_samples',0):>7} "
              f"{r['pc1_pc2_sum']*100:>6.1f}% "
              f"{r['s1_sim']:>+7.4f} {r['s2_sim']:>+7.4f} {r['rot_sum']:>7.4f}")
    
    # Save
    clean = []
    for r in unique:
        c = {}
        for k, v in r.items():
            if isinstance(v, (np.floating,)): c[k] = float(v)
            elif isinstance(v, (np.integer,)): c[k] = int(v)
            elif isinstance(v, (np.ndarray,)): c[k] = v.tolist()
            elif isinstance(v, bool): c[k] = bool(v)
            else: c[k] = v
        clean.append(c)
    
    with open(os.path.join(SAVE_DIR, 'mega_v2_results.json'), 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"\nSaved {len(unique)} results.")
    
    return unique


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    print("=" * 70)
    print("MEGA EXPERIMENT V2")
    print("=" * 70)
    
    results = run_all_v2()
    ranked = rank_results(results)
    
    # Combined ranking with V1 results
    if os.path.exists(os.path.join(SAVE_DIR, 'mega_experiment_results.json')):
        with open(os.path.join(SAVE_DIR, 'mega_experiment_results.json')) as f:
            v1_results = json.load(f)
        
        combined = v1_results + [r for r in ranked if r['label'] not in {x['label'] for x in v1_results}]
        combined.sort(key=lambda r: r.get('rot_sum', r.get('s1_sim', 0) + r.get('s2_sim', 0)), reverse=True)
        
        print(f"\n{'='*150}")
        print(f"COMBINED V1+V2 RANKING")
        print(f"{'='*150}")
        print(f"{'Rank':<5} {'Experiment':<50} {'Var%':>7} {'S1':>8} {'S2':>8} {'Score':>8}")
        print("-" * 150)
        
        for i, r in enumerate(combined[:30]):
            s1 = r.get('s1_sim', r.get('rot_s1_sim', 0))
            s2 = r.get('s2_sim', r.get('rot_s2_sim', 0))
            sc = s1 + s2
            var_sum = r.get('pc1_pc2_sum', 0)
            star = "★" if i == 0 else " "
            print(f"{star}{i+1:<4} {r['label']:<50} "
                  f"{var_sum*100 if var_sum < 1 else var_sum:>6.1f}% "
                  f"{s1:>+7.4f} {s2:>+7.4f} {sc:>7.4f}")
        
        with open(os.path.join(SAVE_DIR, 'mega_combined_results.json'), 'w') as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\nCombined results saved.")
    
    print("\nDone!")
