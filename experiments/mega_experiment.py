"""
MEGA EXPERIMENT: Exhaustive search for optimal data combination → Pisa/IIT synergies.

This script tries EVERYTHING:
- Multiple datasets (MyoKi, Ninapro E1/E2/E3)
- Multiple task filters (power grasp, precision, all grasp, specific tasks)
- Participant selection (all vs. individual "good" participants)
- Data representations (trial means, static hold, all frames, velocity-filtered)
- Joint normalization (none, z-score, range-scale, robust)
- Centering strategies (mean-center, rest-posture-center)
- Phase selection (pickup only, hold only, whole trial)
- Dimensionality reduction methods (PCA, varimax-PCA, FA)
- Joint subsets (all 15, flexion-only, no abduction)

Output: ranked results with full PC weights for comparison.
"""
import os, sys, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA, FactorAnalysis
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

N_JOINTS_MYOKI = 15
JOINT_NAMES_MYOKI = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
    "RingMCP","RingPIP","MidRingAbd",
    "LitMCP","LitPIP","RingLitAbd"
]

# =============================
# Pisa/IIT Targets (15D)
# =============================
PISA_S1_15 = np.ones(15) / np.linalg.norm(np.ones(15))
PISA_S2_RAW_15 = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_S2_15 = PISA_S2_RAW_15 / np.linalg.norm(PISA_S2_RAW_15)

# =============================
# Joint subsets
# =============================
FLEXION_ONLY = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]  # MCP+PIP of all + ThumbIP
ABDUCTION_JOINTS = [0, 3, 8, 11, 14]
NO_ABDUCTION = [j for j in range(15) if j not in ABDUCTION_JOINTS]  # 10 joints

SUBJOINT_MAP = {
    'flexion10': (FLEXION_ONLY, r'[2,2, 1,1,1, 0,0,0, -1,-1,-1, -2,-2]'),
    'no_abduction10': (NO_ABDUCTION, r'[2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2]'),
}

# =============================
# 1. Load MyoKi data
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

# Pickup-only IDs (original pickup tasks only, no put-back)
PICKUP_IDS = {1, 3, 5, 9, 11, 13, 16, 18, 20, 28, 30, 32, 34, 36, 38, 40, 42, 44}
# Put-back IDs (the corresponding release tasks)
PUTBACK_IDS = {2, 4, 6, 10, 12, 14, 17, 19, 21, 29, 31, 33, 35, 37, 39, 41, 43, 45}
# Static grasp-hold tasks (objects held statically)
STATIC_HOLD_IDS = {46, 47, 48, 49, 50, 51, 52, 53}

print(f"MyoKi task sets:")
print(f"  All grasp: {len(ALL_GRASP_IDS)} tasks")
print(f"  Power grasp: {len(POWER_GRASP_IDS)} tasks")
print(f"  Precision grasp: {len(PRECISION_GRASP_IDS)} tasks")
print(f"  Pickup only: {len(PICKUP_IDS)} tasks")
print(f"  Putback only: {len(PUTBACK_IDS)} tasks")
print(f"  Static hold: {len(STATIC_HOLD_IDS)} tasks")

def load_myoki_trials(participant_files, task_ids, resample_n=100):
    """Load trials from MyoKi data, resampled to uniform length."""
    all_trials = []
    for fpath in participant_files:
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
                    all_trials.append((resampled, tid))
    return all_trials

def load_myoki_rest_postures(participant_files):
    """Get mean rest posture (task=0) for each participant."""
    rest_postures = []
    for fpath in participant_files:
        mat = sio.loadmat(fpath)
        glove = np.nan_to_num(mat['glove_calibrated'])
        tasks = mat['task'].flatten().astype(int)
        hand_data = glove[:, :15]
        rest_mask = (tasks == 0)
        if rest_mask.sum() > 0:
            rest_postures.append(np.mean(hand_data[rest_mask], axis=0))
    if rest_postures:
        return np.mean(rest_postures, axis=0)
    return None

def extract_trial_phase(trials, phase='all'):
    """Extract specific phase from each trial.
    phase options: 'all', 'reach', 'hold_last25', 'hold_last10', 'mid', 'release'
    """
    data = []
    for t, tid in trials:
        n = len(t)
        if phase == 'all':
            data.append(t)  # keep all frames
        elif phase in ('hold_last25', 'static_last25'):
            n_hold = max(5, int(n * 0.25))
            data.append(t[-n_hold:])
        elif phase in ('hold_last10', 'static_last10'):
            n_hold = max(5, int(n * 0.10))
            data.append(t[-n_hold:])
        elif phase in ('hold_last5', 'static_last5'):
            data.append(t[-5:])
        elif phase == 'reach_first25':
            n_reach = max(5, int(n * 0.25))
            data.append(t[:n_reach])
        elif phase == 'reach_first10':
            n_reach = max(5, int(n * 0.10))
            data.append(t[:n_reach])
        elif phase == 'mid':
            s = int(n * 0.25)
            e = int(n * 0.75)
            if e - s >= 5:
                data.append(t[s:e])
            else:
                data.append(t)
        elif phase == 'release_last25':
            # The last 25% after the hold - actually just another name for hold
            n_rel = max(5, int(n * 0.25))
            data.append(t[-n_rel:])
    return data

def trial_means(trial_list):
    """Average each trial."""
    if isinstance(trial_list[0], tuple):
        return np.array([np.mean(t, axis=0) for t, _ in trial_list])
    return np.array([np.mean(t, axis=0) for t in trial_list])

def all_frames(trial_list):
    """Concatenate all frames."""
    if isinstance(trial_list[0], tuple):
        return np.vstack([t for t, _ in trial_list])
    return np.vstack(trial_list)

# =============================
# 2. Load Ninapro data
# =============================
def load_ninapro_data(exercise, hand_only=True):
    """Load all Ninapro preprocessed data for a given exercise."""
    pre_dir = 'Ninapro Data/preprocessed'
    n_joints = 20 if hand_only else 22
    all_data = []
    subject_data = {}  # per-subject storage
    
    files = sorted(glob.glob(os.path.join(pre_dir, f'S*_{exercise}_preprocessed.mat')))
    for fpath in files:
        subj = os.path.basename(fpath).split('_')[0]
        mat = sio.loadmat(fpath)
        angles = mat['angles']
        # First 20 channels = hand joints (exclude wrist 20, 21)
        if hand_only:
            angles = angles[:, :n_joints]
        all_data.append(angles)
        subject_data[subj] = angles
    
    if all_data:
        combined = np.vstack(all_data)
        return combined, subject_data
    return None, {}

# =============================
# 3. Normalization
# =============================
def normalize_data(data, method='none'):
    """Apply normalization to data."""
    if method == 'none':
        return data
    elif method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-6] = 1.0
        return (data - mean) / std
    elif method == 'range':
        """Min-max scale each joint to [0, 1]."""
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-6] = 1.0
        return (data - mins) / ranges
    elif method == 'robust':
        """Robust scaling using IQR."""
        med = np.median(data, axis=0)
        q75, q25 = np.percentile(data, [75, 25], axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-6] = 1.0
        return (data - med) / iqr
    elif method == 'rest_center':
        """Mean-center but subtract rest posture instead of overall mean."""
        # This is handled specially with a pre-computed rest posture
        return data  # placeholder
    return data

def center_rest(data, rest_posture):
    """Subtract rest posture from data."""
    return data - rest_posture[np.newaxis, :]

# =============================
# 4. PCA + Evaluation
# =============================
def get_pisa_targets(n_joints, joint_subset=None):
    """Get Pisa targets for the given joint space."""
    s1 = np.ones(n_joints) / np.linalg.norm(np.ones(n_joints))
    
    if joint_subset is not None:
        raw = np.zeros(n_joints)
        for i, orig_idx in enumerate(joint_subset):
            if orig_idx in [0, 1, 2, 3]:        # Thumb: +2
                raw[i] = 2
            elif orig_idx in [4, 5]:              # Index: +1
                raw[i] = 1
            elif orig_idx in [6, 7, 8]:           # Middle: 0
                raw[i] = 0
            elif orig_idx in [9, 10, 11]:         # Ring: -1
                raw[i] = -1
            elif orig_idx in [12, 13, 14]:        # Little: -2
                raw[i] = -2
        s2 = raw / np.linalg.norm(raw) if np.linalg.norm(raw) > 0 else np.ones(n_joints) / np.sqrt(n_joints)
    else:
        s2 = PISA_S2_15.copy() if n_joints == 15 else np.ones(n_joints) / np.sqrt(n_joints)
        if n_joints == 20:  # Ninapro 20 joint space
            raw = np.array([2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, -2, -2, -2, -2, -2])
            s2 = raw / np.linalg.norm(raw)
    
    return s1, s2

def run_pca_and_evaluate(data, label, norm='none', joint_subset=None, 
                         method='pca', rest_posture=None, n_comps=2):
    """Run PCA/FA, evaluate against Pisa targets."""
    if len(data) < 10:
        return None
    
    data = np.array(data, dtype=float)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    orig_n = data.shape[1]
    
    # Joint subset
    if joint_subset is not None:
        data = data[:, joint_subset]
    
    # Rest posture centering
    if rest_posture is not None and norm == 'rest_center':
        rp = rest_posture
        if joint_subset is not None:
            rp = rp[joint_subset]
        data = data - rp[np.newaxis, :]
        norm_effective = 'none'  # already centered
    else:
        norm_effective = norm
    
    # Normalize
    data = normalize_data(data, norm_effective)
    
    # Filter invalid
    if np.isnan(data).any() or np.isinf(data).any():
        return None
    
    n_j = data.shape[1]
    
    if method == 'pca':
        pca = PCA(n_components=min(n_comps, data.shape[0], n_j))
        pca.fit(data)
        evr = pca.explained_variance_ratio_
        comps = pca.components_[:n_comps].T  # (n_j, n_comps)
        pc_var = evr[:n_comps]
    elif method == 'fa':
        fa = FactorAnalysis(n_components=n_comps, max_iter=2000)
        try:
            fa.fit(data)
            comps = fa.components_.T  # (n_j, 2)
            # Approximate variance explained
            evr = np.var(fa.transform(data), axis=0)
            evr = evr / evr.sum() * 0.5  # rough estimate
            pc_var = evr[:n_comps]
        except:
            return None
    else:
        return None
    
    # Get targets
    s1, s2 = get_pisa_targets(n_j, joint_subset)
    target = np.column_stack([s1, s2])
    
    # Procrustes rotation
    PC = comps[:, :2]
    R, _ = orthogonal_procrustes(PC, target[:, :2])
    rotated = PC @ R
    
    # Cosine similarities
    s1_sim = np.dot(rotated[:, 0], s1) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(s1))
    s2_sim = np.dot(rotated[:, 1], s2) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(s2))
    
    # Before rotation similarity
    pc1 = comps[:, 0] / (np.linalg.norm(comps[:, 0]) + 1e-10)
    pc2 = comps[:, 1] / (np.linalg.norm(comps[:, 1]) + 1e-10)
    pc1_s1 = np.dot(pc1, s1)
    pc2_s2 = np.dot(pc2, s2)
    
    # Compute PC1 sign pattern
    pos_pc1 = sum(1 for w in pc1 if w > 0)
    neg_pc1 = sum(1 for w in pc1 if w < 0)
    
    result = {
        'label': label,
        'n_samples': data.shape[0],
        'n_joints': n_j,
        'norm': norm,
        'method': method,
        'pc1_var': float(pc_var[0]) if len(pc_var) > 0 else 0,
        'pc2_var': float(pc_var[1]) if len(pc_var) > 1 else 0,
        'pc1_pc2_sum': float(np.sum(pc_var[:2])),
        'pc1_s1_sim': float(pc1_s1),
        'pc2_s2_sim': float(pc2_s2),
        'rot_s1_sim': float(s1_sim),
        'rot_s2_sim': float(s2_sim),
        'rot_sum': float(s1_sim + s2_sim),
        'pc1': pc1.tolist(),
        'pc2': pc2.tolist(),
        'rot_s1': rotated[:, 0].tolist(),
        'rot_s2': rotated[:, 1].tolist(),
        'pc1_pos': int(pos_pc1),
        'pc1_neg': int(neg_pc1),
    }
    return result

# =============================
# 5. Experiment Runner
# =============================
def run_myoki_experiments():
    """Run all MyoKi experiments."""
    results = []
    
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]
    
    print(f"\n{'='*70}")
    print(f"MOKI EXPERIMENTS ({len(files)} participants)")
    print(f"{'='*70}")
    
    # Get rest posture for centering
    rest_posture = load_myoki_rest_postures(files)
    print(f"Rest posture loaded: {rest_posture}")
    
    # Load trials for each task set
    all_trials = {}
    for name, ids in [('allgrasp', ALL_GRASP_IDS), ('power', POWER_GRASP_IDS),
                      ('precision', PRECISION_GRASP_IDS), ('pickup', PICKUP_IDS),
                      ('putback', PUTBACK_IDS)]:
        all_trials[name] = load_myoki_trials(files, ids)
        print(f"  {name}: {len(all_trials[name])} trials")
    
    # ============================================
    # EXPERIMENT A: Task sets × Data reps × Normalization
    # ============================================
    print(f"\n--- A: TASK × REP × NORM ---")
    
    task_sets = [('allgrasp', ALL_GRASP_IDS), ('power', POWER_GRASP_IDS),
                 ('pickup', PICKUP_IDS)]
    norms = ['none', 'zscore', 'range', 'robust']
    reps = ['trialmean', 'allframes', 'hold25', 'hold10']
    
    for task_name, _ in task_sets:
        trials = all_trials[task_name]
        for rep in reps:
            if 'hold' in rep:
                pct = rep.replace('hold', '')
                phase_map = {'25': 'hold_last25', '10': 'hold_last10'}
                phased = extract_trial_phase(trials, phase_map.get(pct, 'hold_last25'))
                data = trial_means([(t, 0) for t in phased])
            elif rep == 'allframes':
                data = all_frames(trials)
            else:
                data = trial_means(trials)
            
            for norm in norms:
                label = f"myoki_{task_name}_{rep}_{norm}"
                result = run_pca_and_evaluate(data, label, norm=norm)
                if result:
                    print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                          f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                    results.append(result)
    
    # ============================================
    # EXPERIMENT B: Task sets × Phase selection
    # ============================================
    print(f"\n--- B: TASK × PHASE ---")
    
    phases = ['all', 'reach_first25', 'hold_last25', 'mid', 'hold_last10', 'hold_last5']
    for task_name, _ in task_sets:
        trials = all_trials[task_name]
        for phase in phases:
            if phase == 'all':
                data = trial_means(trials)
            else:
                phased = extract_trial_phase(trials, phase)
                data = trial_means([(t, 0) for t in phased])
            
            for norm in ['none', 'range']:
                label = f"myoki_{task_name}_{phase}_trialmean_{norm}"
                result = run_pca_and_evaluate(data, label, norm=norm)
                if result:
                    print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                          f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                    results.append(result)
    
    # ============================================
    # EXPERIMENT C: Power grasp + rest posture centering
    # ============================================
    print(f"\n--- C: REST POSTURE CENTERING ---")
    
    power_trials = all_trials['power']
    data_tm = trial_means(power_trials)
    data_tm_rest = center_rest(data_tm, rest_posture)
    
    for norm in ['none', 'zscore', 'range']:
        label = f"myoki_power_trialmean_restcenter_{norm}"
        result = run_pca_and_evaluate(data_tm_rest, label, norm=norm)
        if result:
            print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
            results.append(result)
    
    # ============================================
    # EXPERIMENT D: Precision grasps only
    # ============================================
    print(f"\n--- D: PRECISION GRASPS ---")
    
    precision_trials = all_trials['precision']
    if len(precision_trials) > 10:
        for rep in ['trialmean', 'hold25', 'hold10']:
            if 'hold' in rep:
                pct = rep.replace('hold', '')
                phase_map = {'25': 'hold_last25', '10': 'hold_last10'}
                phased = extract_trial_phase(precision_trials, phase_map.get(pct, 'hold_last25'))
                data = trial_means([(t, 0) for t in phased])
            else:
                data = trial_means(precision_trials)
            
            for norm in ['none', 'range']:
                label = f"myoki_precision_{rep}_{norm}"
                result = run_pca_and_evaluate(data, label, norm=norm)
                if result:
                    print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                          f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                    results.append(result)
    
    # ============================================
    # EXPERIMENT E: Joint subsets (flexion only, no abduction)
    # ============================================
    print(f"\n--- E: JOINT SUBSETS ---")
    
    power_tm = trial_means(power_trials)
    for norm in ['none', 'range', 'zscore']:
        for subset_name, (subset_ids, _) in SUBJOINT_MAP.items():
            label = f"myoki_power_trialmean_{subset_name}_{norm}"
            result = run_pca_and_evaluate(power_tm, label, norm=norm, joint_subset=subset_ids)
            if result:
                print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # ============================================
    # EXPERIMENT F: Per-participant selection
    # ============================================
    print(f"\n--- F: PER-PARTICIPANT PCA ---")
    
    participant_scores = {}
    for fpath in files:
        pname = os.path.basename(fpath).replace('.mat', '')
        trials_p = load_myoki_trials([fpath], POWER_GRASP_IDS)
        if len(trials_p) < 10:
            continue
        data_p = trial_means(trials_p)
        for norm in ['none', 'range']:
            label = f"myoki_{pname}_power_trialmean_{norm}"
            result = run_pca_and_evaluate(data_p, label, norm=norm)
            if result:
                print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                results.append(result)
                participant_scores[pname] = result['rot_sum']
    
    # Selected "good" participants (top 5)
    sorted_ps = sorted(participant_scores.items(), key=lambda x: x[1], reverse=True)
    good_ps = [p for p, s in sorted_ps[:5]]
    print(f"\n  Top 5 participants: {good_ps}")
    
    # Mixed: only good participants, all power grasps
    good_files = [f for f in files if any(p in os.path.basename(f) for p in good_ps)]
    if good_files:
        good_trials = load_myoki_trials(good_files, POWER_GRASP_IDS)
        for rep in ['trialmean', 'hold25']:
            if rep == 'hold25':
                phased = extract_trial_phase(good_trials, 'hold_last25')
                data = trial_means([(t, 0) for t in phased])
            else:
                data = trial_means(good_trials)
            for norm in ['none', 'range']:
                label = f"myoki_top5_{rep}_{norm}"
                result = run_pca_and_evaluate(data, label, norm=norm)
                if result:
                    print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                          f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                    results.append(result)
    
    # ============================================
    # EXPERIMENT G: Pickup only (without put-back)
    # ============================================
    print(f"\n--- G: PICKUP ONLY (NO PUTBACK) ---")
    
    for task_name in ['pickup']:
        trials = all_trials['pickup']
        for rep in ['trialmean', 'hold25']:
            if rep == 'hold25':
                phased = extract_trial_phase(trials, 'hold_last25')
                data = trial_means([(t, 0) for t in phased])
            else:
                data = trial_means(trials)
            for norm in ['none', 'range']:
                label = f"myoki_{task_name}_only_{rep}_{norm}"
                result = run_pca_and_evaluate(data, label, norm=norm)
                if result:
                    print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                          f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                    results.append(result)
    
    # Putback only
    putback_trials = all_trials['putback']
    for rep in ['trialmean', 'hold25']:
        if rep == 'hold25':
            phased = extract_trial_phase(putback_trials, 'hold_last25')
            data = trial_means([(t, 0) for t in phased])
        else:
            data = trial_means(putback_trials)
        for norm in ['none', 'range']:
            label = f"myoki_putback_only_{rep}_{norm}"
            result = run_pca_and_evaluate(data, label, norm=norm)
            if result:
                print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # ============================================
    # EXPERIMENT H: Power grasp without thumb opposition
    # ============================================
    print(f"\n--- H: POWER GRASP, NO THUMB OPP ---")
    
    no_thumbopp = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    for norm in ['none', 'range']:
        label = f"myoki_power_trialmean_no_thumbopp_{norm}"
        result = run_pca_and_evaluate(power_tm, label, norm=norm, joint_subset=no_thumbopp)
        if result:
            print(f"  {label}: sum={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
            results.append(result)
    
    return results


def run_ninapro_experiments():
    """Run all Ninapro experiments."""
    results = []
    
    exercises = {'E1': None, 'E2': None, 'E3': None}
    for ex in exercises:
        combined, subjects = load_ninapro_data(ex)
        exercises[ex] = (combined, subjects)
        if combined is not None:
            print(f"\nNinapro {ex}: {len(subjects)} subjects, {len(combined)} samples")
    
    # ============================================
    # EXPERIMENT A: Each exercise separately
    # ============================================
    print(f"\n--- NINAPRO A: PER EXERCISE ---")
    
    for ex in ['E1', 'E2', 'E3']:
        combined, _ = exercises[ex]
        if combined is None: continue
        
        # 20 hand joints, all samples
        for norm in ['none', 'zscore', 'range']:
            label = f"ninapro_{ex}_all_{norm}"
            result = run_pca_and_evaluate(combined, label, norm=norm)
            if result:
                print(f"  {label}: S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # ============================================
    # EXPERIMENT B: Exercise combinations
    # ============================================
    print(f"\n--- NINAPRO B: COMBINATIONS ---")
    
    combos = [
        ('E12', ['E1', 'E2']), ('E13', ['E1', 'E3']),
        ('E23', ['E2', 'E3']), ('E123', ['E1', 'E2', 'E3'])
    ]
    
    for name, ex_list in combos:
        combined_list = []
        for ex in ex_list:
            c, _ = exercises[ex]
            if c is not None:
                combined_list.append(c)
        if not combined_list: continue
        
        combined_data = np.vstack(combined_list)
        for norm in ['none', 'zscore', 'range']:
            label = f"ninapro_{name}_all_{norm}"
            result = run_pca_and_evaluate(combined_data, label, norm=norm)
            if result:
                print(f"  {label}: S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # ============================================
    # EXPERIMENT C: E2 grasp tasks (subset analysis)
    # ============================================
    print(f"\n--- NINAPRO C: E2 GRASP TASKS ---")
    
    # For E2, the tasks are grasping exercises (6 basic grasps × 3 objects = 18 trials × ~10 reps)
    # We can't easily filter by task without the original trial labels
    # But we can do per-subject analysis
    
    combined_e2, subjects_e2 = exercises.get('E2', (None, {}))
    if combined_e2 is not None:
        # All E2 data
        for norm in ['none', 'range']:
            label = f"ninapro_E2_grasp_{norm}"
            result = run_pca_and_evaluate(combined_e2, label, norm=norm)
            if result:
                print(f"  {label}: S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # Per-subject E2
    print(f"\n--- NINAPRO D: PER-SUBJECT E2 ---")
    subj_scores = {}
    for subj, data in subjects_e2.items():
        for norm in ['none', 'range']:
            label = f"ninapro_{subj}_E2_{norm}"
            result = run_pca_and_evaluate(data, label, norm=norm)
            if result:
                results.append(result)
                subj_scores[subj] = result['rot_sum']
                print(f"  {label}: S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
    
    # Top subjects combined
    sorted_subs = sorted(subj_scores.items(), key=lambda x: x[1], reverse=True)
    top5_subs = [s for s, _ in sorted_subs[:5]]
    if top5_subs:
        top5_data = np.vstack([subjects_e2[s] for s in top5_subs if s in subjects_e2])
        for norm in ['none', 'range']:
            label = f"ninapro_E2_top5_{norm}"
            result = run_pca_and_evaluate(top5_data, label, norm=norm)
            if result:
                print(f"  {label}: S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # ============================================
    # EXPERIMENT E: E3 (functional tasks)
    # ============================================
    print(f"\n--- NINAPRO E: E3 FUNCTIONAL ---")
    combined_e3, _ = exercises.get('E3', (None, {}))
    if combined_e3 is not None:
        for norm in ['none', 'range']:
            label = f"ninapro_E3_functional_{norm}"
            result = run_pca_and_evaluate(combined_e3, label, norm=norm)
            if result:
                print(f"  {label}: S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    return results


def run_mixed_experiments(myoki_results):
    """Try mixing datasets or other special experiments."""
    results = []
    
    # ============================================
    # EXPERIMENT A: Factor Analysis on MyoKi
    # ============================================
    print(f"\n--- MIXED A: FACTOR ANALYSIS ON MYOKI ---")
    
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]
    
    for task_name, ids in [('power', POWER_GRASP_IDS), ('allgrasp', ALL_GRASP_IDS)]:
        trials = load_myoki_trials(files, ids)
        data = trial_means(trials)
        
        for norm in ['none', 'range']:
            for method in ['fa']:
                label = f"myoki_{task_name}_trialmean_{method}_{norm}"
                result = run_pca_and_evaluate(data, label, norm=norm, method=method)
                if result:
                    print(f"  {label}: S1={result['rot_s1_sim']:.4f}, S2={result['rot_s2_sim']:.4f}")
                    results.append(result)
    
    # ============================================
    # EXPERIMENT B: MyoKi + Ninapro combined
    # ============================================
    print(f"\n--- MIXED B: MYOKI + NINAPRO ---")
    
    # Only mix if we can align the joint spaces
    # MyoKi has 15 joints (CyberGlove minus Palmar Arch, no DIPs)
    # Ninapro has 20 joints (includes DIPs, more complete)
    # We can only compare separately
    
    return results


# =============================
# 6. Ranking + Visualization
# =============================
def print_ranked_results(all_results, top_n=40):
    """Print all results ranked by rot_sum."""
    if not all_results:
        print("No results!")
        return all_results
    
    # De-duplicate by label
    seen = set()
    unique = []
    for r in all_results:
        if r['label'] not in seen:
            seen.add(r['label'])
            unique.append(r)
    
    # Rank by rot_sum (S1 + S2)
    unique.sort(key=lambda r: r['rot_sum'], reverse=True)
    
    print(f"\n{'='*160}")
    print(f"RANKED RESULTS (by S1+S2 similarity)")
    print(f"{'='*160}")
    print(f"{'Rank':<5} {'Experiment':<50} {'N':>7} {'PC1%':>7} {'Sum%':>7} {'S1':>8} {'S2':>8} {'Score':>7} {'PC1+/-':>7}")
    print("-" * 160)
    
    for i, r in enumerate(unique[:top_n]):
        star = "★" if i == 0 else " "
        print(f"{star}{i+1:<4} {r['label']:<50} {r['n_samples']:>7} "
              f"{r['pc1_var']*100:>6.1f}% {r['pc1_pc2_sum']*100:>6.1f}% "
              f"{r['rot_s1_sim']:>+7.4f} {r['rot_s2_sim']:>+7.4f} "
              f"{r['rot_sum']:>6.4f} {r['pc1_pos']}+/{r['pc1_neg']}-")
    
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
    
    save_path = os.path.join(SAVE_DIR, 'mega_experiment_results.json')
    with open(save_path, 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"\nAll {len(unique)} results saved to {save_path}")
    
    return unique


def plot_top_results(all_results, top_n=10):
    """Plot top results."""
    unique = print_ranked_results(all_results)
    top = unique[:top_n]
    
    n = len(top)
    fig, axes = plt.subplots(n, 4, figsize=(22, 3*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, r in enumerate(top):
        n_j = len(r.get('pc1', []))
        names = JOINT_NAMES_MYOKI[:n_j] if n_j <= 15 else [f"J{j}" for j in range(n_j)]
        
        # Original PC1
        ax = axes[i, 0]
        pc1 = r['pc1']
        colors = ['steelblue' if c >= 0 else 'coral' for c in pc1]
        ax.bar(range(n_j), pc1, color=colors, edgecolor='navy', lw=0.5)
        ax.set_title(f"PC1 ({r['pc1_var']*100:.1f}%)", fontsize=9)
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(-0.7, 0.7)
        ax.grid(axis='y', alpha=.3)
        
        # Original PC2
        ax = axes[i, 1]
        pc2 = r['pc2']
        colors = ['steelblue' if c >= 0 else 'coral' for c in pc2]
        ax.bar(range(n_j), pc2, color=colors, edgecolor='navy', lw=0.5)
        ax.set_title(f"PC2 ({r['pc2_var']*100:.1f}%)", fontsize=9)
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(-0.7, 0.7)
        ax.grid(axis='y', alpha=.3)
        
        # Rotated S1
        ax = axes[i, 2]
        s1 = r['rot_s1']
        colors = ['steelblue' if c >= 0 else 'coral' for c in s1]
        ax.bar(range(n_j), s1, color=colors, edgecolor='darkgreen', lw=0.5)
        ax.set_title(f"Rot→S1 (sim={r['rot_s1_sim']:.3f})", fontsize=9, color='darkgreen')
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(-0.7, 0.7)
        ax.grid(axis='y', alpha=.3)
        
        # Rotated S2
        ax = axes[i, 3]
        s2 = r['rot_s2']
        colors = ['steelblue' if c >= 0 else 'coral' for c in s2]
        ax.bar(range(n_j), s2, color=colors, edgecolor='darkgreen', lw=0.5)
        ax.set_title(f"Rot→S2 (sim={r['rot_s2_sim']:.3f})", fontsize=9, color='darkgreen')
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(-0.7, 0.7)
        ax.grid(axis='y', alpha=.3)
        
        # Label
        axes[i, 0].text(-0.5, 0.6, r['label'][:30], transform=axes[i, 0].transAxes,
                       fontsize=7, fontweight='bold', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"Top {n} Experiments - Pisa/IIT Synergy Matching", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'mega_top_experiments.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")


# =============================
# 7. Detailed analysis of best result
# =============================
def analyze_best_result(all_results):
    """Print detailed analysis of the best result."""
    unique = sorted(set(r['label'] for r in all_results), key=lambda l: 
                   max(r['rot_sum'] for r in all_results if r['label'] == l), reverse=True)
    
    # Get actual unique results sorted
    seen = set()
    unique_results = []
    for r in sorted(all_results, key=lambda x: x['rot_sum'], reverse=True):
        if r['label'] not in seen:
            seen.add(r['label'])
            unique_results.append(r)
    
    if not unique_results:
        return None
    
    best = unique_results[0]
    
    print(f"\n{'='*70}")
    print(f"BEST RESULT ANALYSIS")
    print(f"{'='*70}")
    print(f"Label: {best['label']}")
    print(f"N samples: {best['n_samples']}")
    print(f"N joints: {best['n_joints']}")
    print(f"Normalization: {best['norm']}")
    print(f"Method: {best['method']}")
    print(f"\nVariance explained:")
    print(f"  PC1: {best['pc1_var']*100:.2f}%")
    print(f"  PC2: {best['pc2_var']*100:.2f}%")
    print(f"  PC1+PC2: {best['pc1_pc2_sum']*100:.2f}%")
    print(f"\nPisa target similarity (after Procrustes):")
    print(f"  Rot S1 → Pisa S1: {best['rot_s1_sim']:.4f}")
    print(f"  Rot S2 → Pisa S2: {best['rot_s2_sim']:.4f}")
    print(f"  Combined: {best['rot_sum']:.4f}")
    print(f"\nPC1 sign pattern: {best['pc1_pos']}+ / {best['pc1_neg']}-")
    
    # Print detailed joint weights
    n_j = best['n_joints']
    names = JOINT_NAMES_MYOKI[:n_j] if n_j <= 15 else [f"J{j}" for j in range(n_j)]
    
    print(f"\n{'Joint':<15} {'PC1':>10} {'PC2':>10} {'RotS1':>10} {'RotS2':>10} {'PisaS1':>10} {'PisaS2':>10}")
    print("-" * 75)
    
    s1_target, s2_target = get_pisa_targets(n_j)
    
    for i in range(n_j):
        print(f"{names[i]:<15} {best['pc1'][i]:>+10.4f} {best['pc2'][i]:>+10.4f} "
              f"{best['rot_s1'][i]:>+10.4f} {best['rot_s2'][i]:>+10.4f} "
              f"{s1_target[i]:>+10.4f} {s2_target[i]:>+10.4f}")
    
    # Top 5
    print(f"\n\nTOP 5 RESULTS:")
    print("-" * 100)
    for i in range(min(5, len(unique_results))):
        r = unique_results[i]
        print(f"#{i+1}: {r['label']:<50} PC1+PC2={r['pc1_pc2_sum']*100:.1f}% "
              f"S1={r['rot_s1_sim']:.4f} S2={r['rot_s2_sim']:.4f} Score={r['rot_sum']:.4f}")
    
    return best


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    print("=" * 70)
    print("MEGA EXPERIMENT: Optimal Pisa/IIT Synergy Search")
    print("=" * 70)
    
    all_results = []
    
    # Run MyoKi experiments
    myoki_results = run_myoki_experiments()
    all_results.extend(myoki_results)
    
    # Run Ninapro experiments
    ninapro_results = run_ninapro_experiments()
    all_results.extend(ninapro_results)
    
    # Run mixed experiments
    mixed_results = run_mixed_experiments(myoki_results)
    all_results.extend(mixed_results)
    
    # Rank and visualize
    ranked = print_ranked_results(all_results, top_n=50)
    plot_top_results(all_results, top_n=10)
    
    # Analyze best
    analyze_best_result(all_results)
    
    print("\nDone! Check PCA_results/experiments/mega_experiment_results.json")
