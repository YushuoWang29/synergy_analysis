"""
MEGA EXPERIMENT V3: Targeted refinements to maximize Pisa/IIT match.

Focus areas:
  1. Varimax rotation instead of Procrustes (Pisa paper uses Varimax)
  2. Different joint subsets (remove noisy abduction joints)
  3. Even more aggressive per-task averaging variants
  4. Filter by task type (only power grasps, not tools)
  5. Weighted trials (more weight to cleaner tasks)
  6. Different PCA variants (SVD, correlation matrix)
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA, FactorAnalysis
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes
from scipy.stats import zscore
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

# Subsets
POWER_GRASP_IDS_NO_TOOLS = {t for t in POWER_GRASP_IDS if t < 46}  # Remove tool-use tasks (46-56)
POWER_GRASP_IDS_CORE_GRASPS = {1,2,3,4,5,6,9,10,11,12,13,14,15,40,41,42,43,44,45}
COMPLEX_GRASP_IDS = set()
for t in ALL_GRASP_IDS:
    acts = categories.get(t, [])
    if 'complex' in acts:
        COMPLEX_GRASP_IDS.add(t)

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

def evaluate_predictions(pc1, pc2):
    PC = np.column_stack([pc1, pc2])
    R, _ = orthogonal_procrustes(PC, PISA_TARGET)
    rotated = PC @ R
    s1_sim = np.dot(rotated[:, 0], PISA_S1) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(PISA_S1))
    s2_sim = np.dot(rotated[:, 1], PISA_S2) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(PISA_S2))
    return s1_sim, s2_sim, rotated, R

def run_pca(data, label):
    if len(data) < 5: return None
    data = np.array(data, dtype=float)
    if np.isnan(data).any() or np.isinf(data).any(): return None
    n_comp = min(15, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(data)
    evr = pca.explained_variance_ratio_
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
    s1_sim, s2_sim, rotated, R = evaluate_predictions(pc1, pc2)
    return {
        'label': label, 'n_samples': data.shape[0],
        'pc1_var': float(evr[0]), 'pc2_var': float(evr[1]),
        'pc1_pc2_sum': float(evr[0] + evr[1]),
        's1_sim': float(s1_sim), 's2_sim': float(s2_sim),
        'rot_sum': float(s1_sim + s2_sim),
        'pc1': pc1.tolist(), 'pc2': pc2.tolist(),
        'rot_s1': rotated[:, 0].tolist(), 'rot_s2': rotated[:, 1].tolist(),
    }

def run_varimax_pca(data, label):
    """PCA with Varimax rotation (more like Pisa paper)."""
    if len(data) < 5: return None
    data = np.array(data, dtype=float)
    if np.isnan(data).any() or np.isinf(data).any(): return None
    n_comp = min(15, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(data)
    evr = pca.explained_variance_ratio_
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
    
    # Varimax rotation of the first 2 components
    # Use scipy's orthogonal rotation
    from scipy.stats import ortho_group
    # Perform varimax iteratively
    loadings = pca.components_[:2].T  # (15, 2)
    # Varimax normalization
    # Normalize rows
    h2 = np.sum(loadings**2, axis=1, keepdims=True)
    h2[h2 < 1e-10] = 1.0
    loadings_norm = loadings / np.sqrt(h2)
    
    # Iterative varimax
    from scipy.linalg import svd
    n_rows, n_cols = loadings_norm.shape
    rotation_matrix = np.eye(n_cols)
    converged = False
    max_iter = 100
    for iteration in range(max_iter):
        # Rotate
        rotated_norm = loadings_norm @ rotation_matrix
        # Compute B = d(rotated_norm^2)/d(rotated_norm) = 2*rotated_norm
        # In varimax: Λ = rotated_norm * (rotated_norm^2 - mean(rotated_norm^2, axis=0))
        mean_v = np.mean(rotated_norm**2, axis=0, keepdims=True)
        A = rotated_norm * (rotated_norm**2 - mean_v)
        # SVD
        U, _, Vt = svd(loadings_norm.T @ A, full_matrices=False)
        new_rotation = U @ Vt
        if np.allclose(rotation_matrix, new_rotation, atol=1e-6):
            converged = True
        rotation_matrix = new_rotation
        if converged:
            break
    
    # Apply varimax rotation
    rotated_loadings = loadings @ rotation_matrix
    
    s1_sim, s2_sim, _, _ = evaluate_predictions(rotated_loadings[:, 0], rotated_loadings[:, 1])
    
    return {
        'label': f"{label}_varimax", 'n_samples': data.shape[0],
        'pc1_var': float(evr[0]), 'pc2_var': float(evr[1]),
        'pc1_pc2_sum': float(evr[0] + evr[1]),
        's1_sim': float(s1_sim), 's2_sim': float(s2_sim),
        'rot_sum': float(s1_sim + s2_sim),
        'pc1': rotated_loadings[:, 0].tolist(), 'pc2': rotated_loadings[:, 1].tolist(),
        'rot_s1': rotated_loadings[:, 0].tolist(), 'rot_s2': rotated_loadings[:, 1].tolist(),
    }

def main():
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]
    
    all_results = []
    
    print("=" * 70)
    print("V3 EXPERIMENTS")
    print("=" * 70)
    
    power_trials = load_myoki_trials(files, POWER_GRASP_IDS)
    all_trials = load_myoki_trials(files, ALL_GRASP_IDS)
    power_no_tools_trials = load_myoki_trials(files, POWER_GRASP_IDS_NO_TOOLS)
    
    print(f"Power trials: {len(power_trials)}")
    print(f"All grasp trials: {len(all_trials)}")
    print(f"Power no tools: {len(power_no_tools_trials)}")
    
    # ============================================
    # E1: Varimax rotation on best configs
    # ============================================
    print(f"\n--- E1: VARIMAX ROTATION ---")
    
    for task_name, trials_dict in [('power', power_trials), ('allgrasp', all_trials), ('power_no_tools', power_no_tools_trials)]:
        for rep in ['mean', 'static5']:
            if rep == 'mean':
                data = np.array([np.mean(t, axis=0) for t, tid, _ in trials_dict])
            else:
                data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in trials_dict])
            
            result = run_varimax_pca(data, f"v3_{task_name}_{rep}")
            if result:
                print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                all_results.append(result)
            
            # Also regular PCA for comparison
            result2 = run_pca(data, f"v3_{task_name}_{rep}_pca")
            if result2:
                print(f"  {result2['label']}: Var={result2['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result2['s1_sim']:.4f}, S2={result2['s2_sim']:.4f}")
                all_results.append(result2)
    
    # ============================================
    # E2: Task averaging with different variants
    # ============================================
    print(f"\n--- E2: TASK AVERAGING VARIANTS ---")
    
    for task_name, trials_dict in [('power', power_trials), ('allgrasp', all_trials), ('power_no_tools', power_no_tools_trials)]:
        # Task-average static5 (best so far)
        by_task = {}
        for t, tid, subj in trials_dict:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-5:], axis=0))
        
        task_data = np.array([np.mean(v, axis=0) for v in by_task.values()])
        
        for label, d in [
            (f"v3_{task_name}_taskavg_static5_pca", task_data),
        ]:
            result = run_pca(d, label)
            result2 = run_varimax_pca(d, f"v3_{task_name}_taskavg_static5")
            if result:
                print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                      f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
                all_results.append(result)
            if result2:
                all_results.append(result2)
        
        # Try: median instead of mean for task postures
        task_data_median = np.array([np.median(v, axis=0) for v in by_task.values()])
        result = run_pca(task_data_median, f"v3_{task_name}_taskmedian_static5")
        if result:
            print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
    
    # ============================================
    # E3: Correlation-matrix PCA (uses standardized data)
    # ============================================
    print(f"\n--- E3: CORRELATION-BASED PCA ---")
    
    for task_name, trials_dict in [('power', power_trials), ('power_no_tools', power_no_tools_trials)]:
        # Get data
        data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in trials_dict])
        
        # Correlation PCA: standardize then do PCA
        data_z = zscore(data, axis=0, ddof=1)
        data_z = np.nan_to_num(data_z)
        
        result = run_pca(data_z, f"v3_{task_name}_corrPCA_static5")
        if result:
            print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
        
        # Task-averaged correlation PCA
        by_task = {}
        for t, tid, subj in trials_dict:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-5:], axis=0))
        
        # Standardize per participant before task-averaging
        by_subj = {}
        for t, tid, subj in trials_dict:
            if subj not in by_subj:
                by_subj[subj] = []
            by_subj[subj].append((np.mean(t[-5:], axis=0), tid))
        
        task_means_z = {}
        task_counts = {}
        for subj, st in by_subj.items():
            subj_data = np.array([x[0] for x in st])
            subj_tids = [x[1] for x in st]
            # Z-score within subject
            subj_mean = np.mean(subj_data, axis=0)
            subj_std = np.std(subj_data, axis=0)
            subj_std[subj_std < 1e-6] = 1.0
            subj_z = (subj_data - subj_mean) / subj_std
            
            for i, tid in enumerate(subj_tids):
                if tid not in task_means_z:
                    task_means_z[tid] = []
                    task_counts[tid] = 0
                task_means_z[tid].append(subj_z[i])
        
        task_avg_z = np.array([np.mean(v, axis=0) for v in task_means_z.values()])
        result = run_pca(task_avg_z, f"v3_{task_name}_taskavg_zscored_static5")
        if result:
            print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
    
    # ============================================
    # E4: Remove 'noisy' joints (abductions)
    # ============================================
    print(f"\n--- E4: JOINT SUBSET EXPERIMENTS ---")
    
    joint_subsets = {
        'no_abd': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],  # Remove 3 abduction joints
        'flexion_only': [1, 2, 4, 5, 6, 7, 9, 10, 12, 13],    # No abduction, no thumb abd
    }
    
    for sname, jids in joint_subsets.items():
        n_j = len(jids)
        s1_sub = np.ones(n_j) / np.linalg.norm(np.ones(n_j))
        s2_sub_raw = np.array([2,2, 2,2, 1,1, 0,0, -1,-1, -2,-2]) if sname == 'no_abd' else np.array([2,2, 1,1, 0,0, -1,-1, -2,-2])
        if len(s2_sub_raw) != n_j:
            s2_sub_raw = s2_sub_raw[:n_j]
        s2_sub = s2_sub_raw / np.linalg.norm(s2_sub_raw)
        
        for task_name, trials_dict in [('power', power_trials)]:
            data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in trials_dict])[:, jids]
            
            # Run PCA
            pca = PCA(n_components=2)
            pca.fit(data)
            evr = pca.explained_variance_ratio_
            pc1 = pca.components_[0] / np.linalg.norm(pca.components_[0])
            pc2 = pca.components_[1] / np.linalg.norm(pca.components_[1])
            
            PC = np.column_stack([pc1, pc2])
            target = np.column_stack([s1_sub, s2_sub])
            R, _ = orthogonal_procrustes(PC, target)
            rotated = PC @ R
            
            s1_sim = np.dot(rotated[:, 0], s1_sub)
            s2_sim = np.dot(rotated[:, 1], s2_sub)
            
            result = {
                'label': f"v3_{task_name}_{sname}_static5",
                'n_samples': len(data), 'n_joints': n_j,
                'pc1_var': float(evr[0]), 'pc2_var': float(evr[1]),
                'pc1_pc2_sum': float(evr[0] + evr[1]),
                's1_sim': float(s1_sim), 's2_sim': float(s2_sim),
                'rot_sum': float(s1_sim + s2_sim),
            }
            all_results.append(result)
            print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
    
    # ============================================
    # E5: All grasp tasks, but task-average and use Varimax
    # ============================================
    print(f"\n--- E5: ALL GRASPS, TASK-AVERAGED, VARIMAX ---")
    
    # All grasp tasks, last 10 frames
    by_task_all = {}
    for t, tid, subj in all_trials:
        if tid not in by_task_all:
            by_task_all[tid] = []
        by_task_all[tid].append(np.mean(t[-10:], axis=0))
    
    for rep in ['static5', 'static10']:
        by_task = {}
        for t, tid, subj in all_trials:
            if tid not in by_task:
                by_task[tid] = []
            n_frames = -5 if rep == 'static5' else -10
            by_task[tid].append(np.mean(t[n_frames:], axis=0))
        
        task_data = np.array([np.mean(v, axis=0) for v in by_task.values()])
        result = run_varimax_pca(task_data, f"v3_allgrasp_taskavg_{rep}")
        if result:
            print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
    
    # ============================================
    # E6: Factor Analysis instead of PCA
    # ============================================
    print(f"\n--- E6: FACTOR ANALYSIS ---")
    
    for task_name, trials_dict in [('power', power_trials), ('allgrasp', all_trials)]:
        for rep in ['mean', 'static5']:
            if rep == 'mean':
                data = np.array([np.mean(t, axis=0) for t, tid, _ in trials_dict])
            else:
                data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in trials_dict])
            
            # Factor Analysis with 2 factors
            try:
                fa = FactorAnalysis(n_components=2, max_iter=1000)
                fa.fit(data)
                
                # Get the factor loadings
                loadings = fa.components_.T  # (15, 2)
                # Normalize columns
                for c in range(2):
                    loadings[:, c] = loadings[:, c] / np.linalg.norm(loadings[:, c])
                
                s1_sim, s2_sim, rotated, R = evaluate_predictions(loadings[:, 0], loadings[:, 1])
                
                result = {
                    'label': f"v3_{task_name}_FA{rep}",
                    'n_samples': len(data),
                    'pc1_var': 0, 'pc2_var': 0, 'pc1_pc2_sum': 0,
                    's1_sim': float(s1_sim), 's2_sim': float(s2_sim),
                    'rot_sum': float(s1_sim + s2_sim),
                }
                all_results.append(result)
                print(f"  {result['label']}: S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            except Exception as e:
                print(f"  FA failed for {task_name}_{rep}: {e}")
    
    # ============================================
    # E7: Power grasp tasks only (no tool tasks 46-56)
    # ============================================
    print(f"\n--- E7: PURE POWER GRASPS (NO TOOLS) ---")
    
    for rep in ['mean', 'static5']:
        if rep == 'mean':
            data = np.array([np.mean(t, axis=0) for t, tid, _ in power_no_tools_trials])
        else:
            data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in power_no_tools_trials])
        
        result = run_pca(data, f"v3_power_no_tools_{rep}")
        if result:
            print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
                  f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
            all_results.append(result)
    
    # Task-average version
    by_task_no_tools = {}
    for t, tid, subj in power_no_tools_trials:
        if tid not in by_task_no_tools:
            by_task_no_tools[tid] = []
        by_task_no_tools[tid].append(np.mean(t[-5:], axis=0))
    
    task_data_no_tools = np.array([np.mean(v, axis=0) for v in by_task_no_tools.values()])
    result = run_pca(task_data_no_tools, "v3_power_no_tools_taskavg_static5")
    if result:
        print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
              f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
        all_results.append(result)
    
    # ============================================
    # E8: PCA on covariance vs correlation matrix
    # ============================================
    print(f"\n--- E8: MANUAL PCA ON COVARIANCE MATRIX ---")
    
    data = np.array([np.mean(t[-5:], axis=0) for t, tid, _ in power_trials])
    
    # Covariance-based PCA (default)
    cov = np.cov(data, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    
    pc1_cov = eig_vecs[:, 0] / np.linalg.norm(eig_vecs[:, 0])
    pc2_cov = eig_vecs[:, 1] / np.linalg.norm(eig_vecs[:, 1])
    var_sum = eig_vals[:2].sum() / eig_vals.sum()
    
    s1_sim, s2_sim, _, _ = evaluate_predictions(pc1_cov, pc2_cov)
    result = {
        'label': "v3_power_eig_cov_static5",
        'n_samples': len(data),
        'pc1_var': float(eig_vals[0]/eig_vals.sum()),
        'pc2_var': float(eig_vals[1]/eig_vals.sum()),
        'pc1_pc2_sum': float(var_sum),
        's1_sim': float(s1_sim), 's2_sim': float(s2_sim),
        'rot_sum': float(s1_sim + s2_sim),
    }
    all_results.append(result)
    print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
          f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
    
    # ============================================
    # E9: Per-participant mean-centering then task-averaging
    # ============================================
    print(f"\n--- E9: PER-PARTICIPANT MEAN-CENTERED TASK AVERAGING ---")
    
    # Remove each participant's overall mean from their data
    by_subj = {}
    for t, tid, subj in power_trials:
        post = np.mean(t[-5:], axis=0)
        if subj not in by_subj:
            by_subj[subj] = []
        by_subj[subj].append((post, tid))
    
    task_means_centered = {}
    for subj, st in by_subj.items():
        posts = np.array([x[0] for x in st])
        tids = [x[1] for x in st]
        subj_mean = np.mean(posts, axis=0)
        for i, tid in enumerate(tids):
            centered = posts[i] - subj_mean
            if tid not in task_means_centered:
                task_means_centered[tid] = []
            task_means_centered[tid].append(centered)
    
    task_data_centered = np.array([np.mean(v, axis=0) for v in task_means_centered.values()])
    result = run_pca(task_data_centered, "v3_power_centered_taskavg_static5")
    if result:
        print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
              f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
        all_results.append(result)
    
    # ============================================
    # E10: Only the strongest grasp types (no hammer/saw)
    # ============================================
    print(f"\n--- E10: CORE GRASP TASKS ONLY ---")
    
    core_trials = load_myoki_trials(files, POWER_GRASP_IDS_CORE_GRASPS)
    print(f"Core grasp trials: {len(core_trials)}")
    
    by_task_core = {}
    for t, tid, subj in core_trials:
        if tid not in by_task_core:
            by_task_core[tid] = []
        by_task_core[tid].append(np.mean(t[-5:], axis=0))
    
    task_data_core = np.array([np.mean(v, axis=0) for v in by_task_core.values()])
    result = run_pca(task_data_core, "v3_power_core_taskavg_static5")
    if result:
        print(f"  {result['label']}: Var={result['pc1_pc2_sum']*100:.1f}%, "
              f"S1={result['s1_sim']:.4f}, S2={result['s2_sim']:.4f}")
        all_results.append(result)
    
    result2 = run_varimax_pca(task_data_core, "v3_power_core_taskavg_static5_varimax")
    if result2:
        all_results.append(result2)
    
    # ============================================
    # RANKING
    # ============================================
    seen = set()
    unique = []
    for r in all_results:
        if r['label'] not in seen:
            seen.add(r['label'])
            unique.append(r)
    
    unique.sort(key=lambda r: r['rot_sum'], reverse=True)
    
    print(f"\n{'='*150}")
    print(f"V3 RESULTS RANKED (by S1+S2)")
    print(f"{'='*150}")
    print(f"{'Rank':<5} {'Experiment':<55} {'N':>7} {'Var%':>7} {'S1':>8} {'S2':>8} {'Score':>8}")
    print("-" * 150)
    
    for i, r in enumerate(unique):
        star = "★" if i == 0 else " "
        var_str = f"{r['pc1_pc2_sum']*100:>6.1f}%" if r['pc1_pc2_sum'] > 0 else "  N/A "
        print(f"{star}{i+1:<4} {r['label']:<55} {r.get('n_samples',0):>7} "
              f"{var_str} "
              f"{r['s1_sim']:>+7.4f} {r['s2_sim']:>+7.4f} {r['rot_sum']:>7.4f}")
    
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
    
    with open(os.path.join(SAVE_DIR, 'mega_v3_results.json'), 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"\nSaved {len(unique)} results.")
    
    # Combined with V2
    for vfile in ['mega_v2_results.json', 'mega_experiment_results.json']:
        vpath = os.path.join(SAVE_DIR, vfile)
        if os.path.exists(vpath):
            with open(vpath) as f:
                v_results = json.load(f)
            combined = v_results + [r for r in clean if r['label'] not in {x.get('label','') for x in v_results}]
            combined.sort(key=lambda r: r.get('rot_sum', r.get('s1_sim',0)+r.get('s2_sim',0)), reverse=True)
            
            outpath = os.path.join(SAVE_DIR, f'all_results_combined.json')
            with open(outpath, 'w') as f:
                json.dump(combined, f, indent=2, default=str)
            
            print(f"\nTop 30 ALL COMBINED:")
            print(f"{'Rank':<5} {'Experiment':<55} {'Var%':>7} {'S1':>8} {'S2':>8} {'Score':>8}")
            print("-" * 150)
            for i, r in enumerate(combined[:30]):
                s1 = r.get('s1_sim', 0)
                s2 = r.get('s2_sim', 0)
                va = r.get('pc1_pc2_sum', 0)
                star = "★" if i == 0 else " "
                print(f"{star}{i+1:<4} {r['label']:<55} "
                      f"{va*100 if va < 1 else va:>6.1f}% "
                      f"{s1:>+7.4f} {s2:>+7.4f} {s1+s2:>7.4f}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
