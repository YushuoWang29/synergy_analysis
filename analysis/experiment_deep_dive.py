"""
Deeper experiments: z-score normalization, temporal segmentation, joint subsets.
Goal: Find configuration where PC1≈uniform (open/close) and PC2≈gradient (thumb-to-little).
"""
import os, sys, glob, json
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

# Pisa/IIT targets (15D)
PISA_S1 = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
PISA_S2_RAW = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_S2 = PISA_S2_RAW / np.linalg.norm(PISA_S2_RAW)
PISA_TARGET = np.column_stack([PISA_S1, PISA_S2])

# Joint type classification
FLEXION_JOINTS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]  # MCP+PIP of all fingers + ThumbIP
ABDUCTION_JOINTS = [0, 3, 8, 11, 14]  # ThumbAbd, ThumbOpp, IdxMidAbd, MidRingAbd, RingLitAbd
THUMB_JOINTS = [0, 1, 2, 3]  # All thumb joints
FINGER_JOINTS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # All non-thumb

# =============================
# Data loading
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

POWER_GRASP_IDS = set()
ALL_GRASP_IDS = set()
for tid, acts in categories.items():
    has_grasp = any(a in grasp_types for a in acts)
    is_none = 'none_action' in acts
    if has_grasp and not is_none:
        ALL_GRASP_IDS.add(tid)
        if any(a in power_types for a in acts):
            POWER_GRASP_IDS.add(tid)

PICKUP_IDS = {1, 3, 5, 9, 11, 13, 16, 18, 20, 28, 30, 32, 34, 36, 38, 40, 42, 44}

def load_all_trials(participant_files, task_ids):
    """Load all trials with resampling."""
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
                    # Resample to 100 frames
                    x_old = np.linspace(0, 1, len(trial))
                    x_new = np.linspace(0, 1, 100)
                    f = interp1d(x_old, trial, axis=0, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
                    resampled = f(x_new)
                    all_trials.append((resampled, tid))
    return all_trials

def run_pca_and_evaluate(data, label, zscore=False, joint_subset=None):
    """
    Run PCA with optional z-score normalization and joint subset selection.
    Evaluate against Pisa targets.
    """
    if len(data) < 5: return None
    data = np.array(data)
    
    if joint_subset is not None:
        data = data[:, joint_subset]
    
    if zscore:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 0.001] = 1.0
        data = (data - mean) / std
    
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"  WARN: NaN in data after processing")
        return None
    
    pca = PCA(n_components=min(15, data.shape[0], data.shape[1]))
    pca.fit(data)
    evr = pca.explained_variance_ratio_
    
    pc1 = pca.components_[0] / np.linalg.norm(pca.components_[0])
    pc2 = pca.components_[1] / np.linalg.norm(pca.components_[1])
    
    # Define targets in the current joint space
    n_j = data.shape[1]
    s1 = np.ones(n_j) / np.linalg.norm(np.ones(n_j))
    
    if joint_subset is not None:
        # Build gradient target for the selected joints
        # Map back to original indices to decide values
        raw = np.zeros(n_j)
        for i, orig_idx in enumerate(joint_subset):
            if orig_idx in [0, 1, 2, 3]:  # thumb: +2
                raw[i] = 2
            elif orig_idx in [4, 5]:  # index: +1
                raw[i] = 1
            elif orig_idx in [6, 7, 8]:  # middle: 0
                raw[i] = 0
            elif orig_idx in [9, 10, 11]:  # ring: -1
                raw[i] = -1
            elif orig_idx in [12, 13, 14]:  # little: -2
                raw[i] = -2
        s2 = raw / np.linalg.norm(raw) if np.linalg.norm(raw) > 0 else np.ones(n_j) / np.sqrt(n_j)
    else:
        s2_raw = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
        if n_j < 15:
            s2_raw = s2_raw[joint_subset]
        s2 = s2_raw / np.linalg.norm(s2_raw)
    
    target = np.column_stack([s1, s2])
    
    PC = np.column_stack([pc1, pc2])
    R, _ = orthogonal_procrustes(PC, target)
    rotated = PC @ R
    
    s1_sim = np.dot(rotated[:, 0], s1) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(s1))
    s2_sim = np.dot(rotated[:, 1], s2) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(s2))
    
    # Also compute original Pisa alignment (without rotation)
    pc1_s1 = np.dot(pc1, s1)
    pc2_s2 = np.dot(pc2, s2)
    
    return {
        'label': label,
        'n_samples': data.shape[0],
        'n_joints': n_j,
        'pc1_var': float(evr[0]),
        'pc2_var': float(evr[1]),
        'pc1_pc2_sum': float(evr[0] + evr[1]),
        'pc1_s1_sim': float(pc1_s1),
        'pc2_s2_sim': float(pc2_s2),
        'rot_s1_sim': float(s1_sim),
        'rot_s2_sim': float(s2_sim),
        'pc1': pc1.tolist() if len(pc1) <= 15 else [],
        'pc2': pc2.tolist() if len(pc2) <= 15 else [],
        'rot_s1': rotated[:, 0].tolist() if len(rotated[:, 0]) <= 15 else [],
        'rot_s2': rotated[:, 1].tolist() if len(rotated[:, 1]) <= 15 else [],
        'zscore': zscore,
        'joint_subset': str(joint_subset),
    }


# =============================
# Main experiment runner
# =============================
def main():
    # Load participants
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid = [f'P{p:02d}' for p in [1,2,3,4,5,6,7,8,9,11,12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid)]
    
    print(f"Loading data for {len(files)} participants...")
    
    # Load trials
    print(f"\n--- Loading all grasp trials ---")
    all_grasp_trials = load_all_trials(files, ALL_GRASP_IDS)
    print(f"  All grasp: {len(all_grasp_trials)} trials")
    
    print(f"\n--- Loading power grasp trials ---")
    power_trials = load_all_trials(files, POWER_GRASP_IDS)
    print(f"  Power: {len(power_trials)} trials")
    
    print(f"\n--- Loading pickup-only grasp trials ---")
    pickup_trials = load_all_trials(files, PICKUP_IDS)
    print(f"  Pickup: {len(pickup_trials)} trials")
    
    # Task sets
    task_sets = {
        'all_grasp': (all_grasp_trials, ALL_GRASP_IDS),
        'power_grasp': (power_trials, POWER_GRASP_IDS),
        'pickup_grasp': (pickup_trials, PICKUP_IDS),
    }
    
    results = []
    
    # ============================================================
    # EXPERIMENT ROUND 1: Z-score normalization
    # ============================================================
    print(f"\n{'='*70}")
    print(f"ROUND 1: Z-SCORE NORMALIZATION")
    print(f"{'='*70}")
    
    for task_name, (trials, _) in task_sets.items():
        for zscore in [False, True]:
            for rep in ['trial_mean', 'all_frames', 'static_last25', 'static_last50']:
                label = f"{task_name}_{rep}_z{zscore}"
                
                if rep == 'trial_mean':
                    data = np.array([np.mean(t, axis=0) for t, _ in trials])
                elif rep == 'all_frames':
                    data = np.vstack([t for t, _ in trials])
                elif rep == 'static_last25':
                    data = np.array([np.mean(t[-25:], axis=0) for t, _ in trials])
                elif rep == 'static_last50':
                    data = np.array([np.mean(t[-50:], axis=0) for t, _ in trials])
                
                result = run_pca_and_evaluate(data, label, zscore=zscore)
                if result is None: continue
                
                print(f"  {label}: PC1={result['pc1_var']*100:.1f}%, "
                      f"PC2={result['pc2_var']*100:.1f}%, "
                      f"S1={result['rot_s1_sim']:.4f}, "
                      f"S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # ============================================================
    # EXPERIMENT ROUND 2: Temporal segmentation
    # ============================================================
    print(f"\n{'='*70}")
    print(f"ROUND 2: TEMPORAL SEGMENTATION")
    print(f"{'='*70}")
    
    for task_name, (trials, _) in task_sets.items():
        for seg_name, frac_start, frac_end in [
            ('full', 0.0, 1.0),
            ('reach_first25', 0.0, 0.25),
            ('reach_first50', 0.0, 0.50),
            ('hold_last25', 0.75, 1.0),
            ('mid_phase', 0.25, 0.75),
        ]:
            for zscore in [True, False]:
                # For each trial, use the specified temporal segment
                if seg_name == 'full':
                    data = np.array([np.mean(t, axis=0) for t, _ in trials])
                else:
                    seg_trials = []
                    for t, _ in trials:
                        s = int(len(t) * frac_start)
                        e = int(len(t) * frac_end)
                        if e - s >= 5:
                            seg_trials.append(np.mean(t[s:e], axis=0))
                        else:
                            seg_trials.append(np.mean(t, axis=0))
                    data = np.array(seg_trials)
                
                label = f"{task_name}_{seg_name}_trialmean_z{zscore}"
                result = run_pca_and_evaluate(data, label, zscore=zscore)
                if result is None: continue
                
                print(f"  {label}: PC1={result['pc1_var']*100:.1f}%, "
                      f"PC2={result['pc2_var']*100:.1f}%, "
                      f"S1={result['rot_s1_sim']:.4f}, "
                      f"S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # ============================================================
    # EXPERIMENT ROUND 3: Joint subsets
    # ============================================================
    print(f"\n{'='*70}")
    print(f"ROUND 3: JOINT SUBSETS")
    print(f"{'='*70}")
    
    # For these, use power grasp + trial mean + z-score as baseline
    for task_name, (trials, _) in [('power_grasp', (power_trials, POWER_GRASP_IDS))]:
        trial_means = np.array([np.mean(t, axis=0) for t, _ in trials])
        
        for zscore in [True, False]:
            for joint_set_name, joint_ids in [
                ('all15', None),
                ('flexion10', FLEXION_JOINTS),
                ('all_no_thumb11', FINGER_JOINTS),
                ('flexion_no_thumb7', [j for j in FLEXION_JOINTS if j not in THUMB_JOINTS]),
                ('no_abduction10', [j for j in range(N_JOINTS) if j not in ABDUCTION_JOINTS]),
                ('no_thumbopp14', [j for j in range(N_JOINTS) if j != 3]),  # Exclude thumb opposition
                ('mcp_pip_only9', [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]),
            ]:
                label = f"{task_name}_trialmean_{joint_set_name}_z{zscore}"
                result = run_pca_and_evaluate(trial_means, label, zscore=zscore, joint_subset=joint_ids)
                if result is None: continue
                
                print(f"  {label}: PC1={result['pc1_var']*100:.1f}%, "
                      f"PC2={result['pc2_var']*100:.1f}%, "
                      f"S1={result['rot_s1_sim']:.4f}, "
                      f"S2={result['rot_s2_sim']:.4f}")
                results.append(result)
    
    # ============================================================
    # EXPERIMENT ROUND 4: Hand-crafted synergies (not PCA)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"ROUND 4: VARIMAX / FACTOR ANALYSIS APPROACH")
    print(f"{'='*70}")
    
    from sklearn.decomposition import FactorAnalysis
    
    for task_name, (trials, _) in [('power_grasp', (power_trials, POWER_GRASP_IDS))]:
        trial_means = np.array([np.mean(t, axis=0) for t, _ in trials])
        
        for zscore in [True, False]:
            for joint_subset in [None, FLEXION_JOINTS]:
                data = trial_means.copy()
                if joint_subset is not None:
                    data = data[:, joint_subset]
                
                # Factor Analysis
                fa = FactorAnalysis(n_components=2, max_iter=1000)
                try:
                    fa_data = data.copy()
                    if zscore:
                        m = np.mean(fa_data, axis=0)
                        s = np.std(fa_data, axis=0)
                        s[s < 0.001] = 1.0
                        fa_data = (fa_data - m) / s
                    comps = fa.fit_transform(fa_data)
                    
                    # Get loadings
                    loadings = fa.components_  # (2, n_features)
                    
                    # Normalize and compare to targets
                    n_j = data.shape[1]
                    s1 = np.ones(n_j) / np.linalg.norm(np.ones(n_j))
                    
                    if joint_subset is not None:
                        raw = np.zeros(n_j)
                        for i, orig_idx in enumerate(joint_subset):
                            if orig_idx in [0,1,2,3]: raw[i] = 2
                            elif orig_idx in [4,5]: raw[i] = 1
                            elif orig_idx in [6,7,8]: raw[i] = 0
                            elif orig_idx in [9,10,11]: raw[i] = -1
                            elif orig_idx in [12,13,14]: raw[i] = -2
                        s2 = raw / np.linalg.norm(raw) if np.linalg.norm(raw) > 0 else np.ones(n_j)/np.sqrt(n_j)
                    else:
                        s2_raw = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
                        s2 = s2_raw / np.linalg.norm(s2_raw)
                    
                    f1 = loadings[0] / (np.linalg.norm(loadings[0]) + 1e-10)
                    f2 = loadings[1] / (np.linalg.norm(loadings[1]) + 1e-10)
                    
                    target = np.column_stack([s1, s2])
                    PC = np.column_stack([f1, f2])
                    R_fa, _ = orthogonal_procrustes(PC, target)
                    rotated_fa = PC @ R_fa
                    
                    s1_fa = np.dot(rotated_fa[:, 0], s1)
                    s2_fa = np.dot(rotated_fa[:, 1], s2)
                    
                    # Varimax-like: compute variance of squared loadings
                    varimax_val = np.var(f1**2) + np.var(f2**2)
                    
                    label = f"{task_name}_fa2_{'z' if zscore else 'raw'}_{str(len(joint_subset)+2) if joint_subset else '15'}"
                    
                    print(f"  FA Label: {label}")
                    print(f"    PC1→S1: {s1_fa:.4f}, PC2→S2: {s2_fa:.4f}")
                    
                    results.append({
                        'label': label,
                        'n_samples': data.shape[0],
                        'n_joints': n_j,
                        'pc1_var': 0, 'pc2_var': 0, 'pc1_pc2_sum': 0,
                        'pc1_s1_sim': 0, 'pc2_s2_sim': 0,
                        'rot_s1_sim': float(s1_fa),
                        'rot_s2_sim': float(s2_fa),
                        'pc1': f1.tolist() if len(f1) <= 15 else [],
                        'pc2': f2.tolist() if len(f2) <= 15 else [],
                        'rot_s1': rotated_fa[:, 0].tolist() if len(rotated_fa[:, 0]) <= 15 else [],
                        'rot_s2': rotated_fa[:, 1].tolist() if len(rotated_fa[:, 1]) <= 15 else [],
                        'zscore': zscore,
                        'method': 'FA',
                    })
                except Exception as e:
                    print(f"  FA failed: {e}")
    
    # ============================================================
    # ROUND 5: Only velocity frames (dynamic movement phase)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"ROUND 5: ONLY VELOCITY FRAMES (DYNAMIC PHASE)")
    print(f"{'='*70}")
    
    for task_name, (trials, _) in task_sets.items():
        vel_trial_means = []
        for t, _ in trials:
            # Compute velocity
            grads = np.gradient(t, axis=0)
            vel = np.linalg.norm(grads, axis=1)
            # Use only high velocity frames
            high_vel_mask = vel > np.percentile(vel, 70)
            if high_vel_mask.sum() >= 5:
                vel_trial_means.append(np.mean(t[high_vel_mask], axis=0))
            else:
                # Fallback to all
                vel_trial_means.append(np.mean(t, axis=0))
        
        data = np.array(vel_trial_means)
        for zscore in [True, False]:
            label = f"{task_name}_velocity_frames_z{zscore}"
            result = run_pca_and_evaluate(data, label, zscore=zscore)
            if result is None: continue
            print(f"  {label}: PC1={result['pc1_var']*100:.1f}%, "
                  f"PC2={result['pc2_var']*100:.1f}%, "
                  f"S1={result['rot_s1_sim']:.4f}, "
                  f"S2={result['rot_s2_sim']:.4f}")
            results.append(result)
    
    # ============================================================
    # PRINT COMPARISON TABLE
    # ============================================================
    print(f"\n{'='*150}")
    print(f"ALL EXPERIMENTS - RANKED BY S1+S2 SIMILARITY")
    print(f"{'='*150}")
    
    def score(r):
        return r.get('rot_s1_sim', 0) + r.get('rot_s2_sim', 0)
    
    results.sort(key=score, reverse=True)
    
    header = f"{'Experiment':<55} {'N':>7} {'PC1':>8} {'PC2':>8} {'Sum':>8} {'R1→S1':>8} {'R2→S2':>8} {'Score':>8}"
    print(header)
    print("-" * 150)
    
    for i, r in enumerate(results):
        s = score(r)
        line = (f"{r['label']:<55} {r['n_samples']:>7} "
                f"{r['pc1_var']*100 if r['pc1_var'] > 0 else 0:>7.1f}% "
                f"{r['pc2_var']*100 if r['pc2_var'] > 0 else 0:>7.1f}% "
                f"{r['pc1_pc2_sum']*100 if r['pc1_pc2_sum'] > 0 else 0:>7.1f}% "
                f"{r['rot_s1_sim']:>+7.4f} {r['rot_s2_sim']:>+7.4f} {s:>7.4f}")
        if i == 0:
            print(f"★ {line}")
        else:
            print(f"  {line}")
    
    # Save
    clean = []
    for r in results:
        c = {}
        for k, v in r.items():
            if isinstance(v, (np.floating,)): c[k] = float(v)
            elif isinstance(v, (np.integer,)): c[k] = int(v)
            elif isinstance(v, (np.ndarray,)): c[k] = v.tolist()
            else: c[k] = str(v) if not isinstance(v, str) else v
        clean.append(c)
    
    with open(os.path.join(SAVE_DIR, 'experiment_deep_results.json'), 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"\nResults saved.")
    
    # Plot top 8
    plot_top_results(results[:8], FIG_DIR)
    
    print("\nDone!")

def plot_top_results(top_results, fig_dir):
    """Plot PC1, PC2, rotated S1, rotated S2 for top results."""
    n = len(top_results)
    if n == 0: return
    
    # Determine max joints to plot
    max_j = max(len(r.get('pc1', [])) for r in top_results) if top_results else 15
    
    fig, axes = plt.subplots(n, 4, figsize=(20, 3*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, r in enumerate(top_results):
        n_j = len(r.get('pc1', []))
        joint_labels = JOINT_NAMES[:n_j] if n_j <= 15 else [f"J{j}" for j in range(n_j)]
        
        # PC1
        ax = axes[i, 0]
        pc1 = r.get('pc1', [])
        if pc1:
            colors = ['steelblue' if c >= 0 else 'coral' for c in pc1]
            ax.bar(range(n_j), pc1, color=colors, edgecolor='navy', lw=0.5)
            ax.set_title(f"PC1 ({r['pc1_var']*100:.1f}%)" if r['pc1_var'] > 0 else "Loading 1")
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(joint_labels, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
        ax.set_ylim(-0.7, 0.7)
        
        # PC2
        ax = axes[i, 1]
        pc2 = r.get('pc2', [])
        if pc2:
            colors = ['steelblue' if c >= 0 else 'coral' for c in pc2]
            ax.bar(range(n_j), pc2, color=colors, edgecolor='navy', lw=0.5)
            ax.set_title(f"PC2 ({r['pc2_var']*100:.1f}%)" if r['pc2_var'] > 0 else "Loading 2")
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(joint_labels, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
        ax.set_ylim(-0.7, 0.7)
        
        # Rotated S1
        ax = axes[i, 2]
        s1 = r.get('rot_s1', [])
        if s1:
            colors = ['steelblue' if c >= 0 else 'coral' for c in s1]
            ax.bar(range(n_j), s1, color=colors, edgecolor='green', lw=0.5)
            ax.set_title(f"Rot→S1 ({r['rot_s1_sim']:.3f})")
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(joint_labels, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
        ax.set_ylim(-0.7, 0.7)
        
        # Rotated S2
        ax = axes[i, 3]
        s2 = r.get('rot_s2', [])
        if s2:
            colors = ['steelblue' if c >= 0 else 'coral' for c in s2]
            ax.bar(range(n_j), s2, color=colors, edgecolor='green', lw=0.5)
            ax.set_title(f"Rot→S2 ({r['rot_s2_sim']:.3f})")
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(joint_labels, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
        ax.set_ylim(-0.7, 0.7)
    
    plt.suptitle(f"Top {n} Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'top_deep_experiments.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved: {fig_path}")


if __name__ == '__main__':
    main()
