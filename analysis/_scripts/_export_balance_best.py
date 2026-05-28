"""
Export the best balance search results as full PCA models usable by synergy_myoki.py.
Re-runs PCA on the actual data to get mean pose, then saves in the expected format.
"""
import os, sys, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
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
def extract_static10(trials): return [(np.mean(t[-10:], axis=0), tid, subj) for t, tid, subj in trials]
def extract_first_frame(trials): return [(t[0], tid, subj) for t, tid, subj in trials]

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
# Each entry: (label_suffix, task_ids_fn, rep_fn, norm_name, jids)
CONFIGS = [
    # === BEST OVERALL (balanced) ===
    ("power_only_static5_flex10_robust", lambda: POWER_IDS, extract_static5, "robust", FLEXION_IDS),
    ("precision_only_trialsd_flex10_zscore", lambda: PRECISION_IDS, extract_trial_std, "zscore", FLEXION_IDS),
    ("precision_only_trialsd_flex10_range", lambda: PRECISION_IDS, extract_trial_std, "range", FLEXION_IDS),
    ("precision_only_trialmean_no_abd_zscore", lambda: PRECISION_IDS, extract_trial_means, "zscore", NO_ABD_IDS),
    ("power_precision_static5_flex10_robust", lambda: sorted(set(POWER_IDS)|set(PRECISION_IDS)), extract_static5, "robust", FLEXION_IDS),
    ("precision_only_trialmean_no_abd_range", lambda: PRECISION_IDS, extract_trial_means, "range", NO_ABD_IDS),
    
    # === FULLY EXPLAINABLE: ALL GRASP, ALL 15 JOINTS, REASONABLE NORMALIZATION ===
    ("all_grasp_trialmean_all15_none", lambda: ALL_GRASP_IDS, extract_trial_means, "none", None),
    ("all_grasp_trialmean_all15_zscore", lambda: ALL_GRASP_IDS, extract_trial_means, "zscore", None),
    ("all_grasp_static5_all15_none", lambda: ALL_GRASP_IDS, extract_static5, "none", None),
    ("all_grasp_trialmean_all15_range", lambda: ALL_GRASP_IDS, extract_trial_means, "range", None),
    ("all_grasp_trialsd_all15_none", lambda: ALL_GRASP_IDS, extract_trial_std, "none", None),
    ("all_grasp_trialrange_all15_none", lambda: ALL_GRASP_IDS, extract_trial_range, "none", None),
    
    # === STANDARD CONFIGS (all tasks, all participants) ===
    ("all_tasks_trialmean_no_abd_none", lambda: sorted(set(ALL_TASKS)-{0}), extract_trial_means, "none", NO_ABD_IDS),
    ("all_tasks_trialmean_all15_none", lambda: sorted(set(ALL_TASKS)-{0}), extract_trial_means, "none", None),
    ("all_tasks_static5_all15_none", lambda: sorted(set(ALL_TASKS)-{0}), extract_static5, "none", None),
    
    # === POWER + ALL 15 JOINTS, NO NORM ===
    ("power_only_trialmean_all15_none", lambda: POWER_IDS, extract_trial_means, "none", None),
    ("power_only_static5_all15_none", lambda: POWER_IDS, extract_static5, "none", None),
    
    # === PRECISION + ALL 15 JOINTS ===
    ("precision_only_trialmean_all15_none", lambda: PRECISION_IDS, extract_trial_means, "none", None),
    ("precision_only_static5_all15_none", lambda: PRECISION_IDS, extract_static5, "none", None),
]

def run_and_save_configs():
    print("="*70)
    print("EXPORTING BALANCE SEARCH BEST CONFIGS")
    print("="*70)
    
    out_dir = "PCA_results/experiments"
    os.makedirs(out_dir, exist_ok=True)
    
    results_list = []
    
    for idx, (label_suffix, task_fn, rep_fn, norm_name, jids) in enumerate(CONFIGS):
        task_ids = task_fn()
        print(f"\n[{idx+1}/{len(CONFIGS)}] {label_suffix} ({len(task_ids)} tasks)...")
        
        trials = load_myoki_trials(FILES, task_ids)
        print(f"  Trials: {len(trials)}")
        
        processed = rep_fn(trials)
        if not processed:
            print(f"  SKIP: no processed data")
            continue
        
        data_array = np.vstack([p[0] for p in processed])
        
        # Apply normalization
        data = NORM_FUNCS[norm_name](data_array)
        
        # Select joints
        if jids is not None:
            data = data[:, jids]
            n_j = len(jids)
        else:
            n_j = 15
        
        print(f"  Data shape: {data.shape}")
        
        # PCA
        n_comp = min(n_j, data.shape[0], 10)
        pca = PCA(n_components=n_comp)
        pca.fit(data)
        
        # Build 15-joint PC vectors
        full_components = np.zeros((n_comp, 15))
        full_mean = np.zeros(15)
        
        if jids is not None:
            full_components[:, jids] = pca.components_[:, :n_j]
            # Mean in original space: compute mean of original data (before normalization)
            orig_data = none_norm(data_array)[:, jids] if norm_name != 'none' else data_array[:, jids]
            full_mean[jids] = np.mean(orig_data, axis=0)
        else:
            full_components[:, :n_j] = pca.components_
            full_mean = np.mean(data_array, axis=0)
        
        evr = pca.explained_variance_ratio_
        label = f"balance_{label_suffix}"
        
        print(f"  PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, Sum={sum(evr[:2])*100:.1f}%")
        
        # Save full PCA model
        save_name = f"pca_balance_{label_suffix}.npz"
        save_path = os.path.join(out_dir, save_name)
        np.savez(save_path,
                 components=full_components,
                 mean=full_mean,
                 explained_variance=evr,
                 data_label=label,
                 pc1=full_components[0],
                 pc2=full_components[1],
                 n_samples=data.shape[0],
                 n_joints=n_j,
                 norm=norm_name,
                 task_ids=task_ids)
        
        # Also save a simplified version for quick visualization
        simple_name = f"balance_simple_{label_suffix}.npz"
        simple_path = os.path.join(out_dir, simple_name)
        np.savez(simple_path,
                 components=full_components[:2],
                 mean=full_mean,
                 explained_variance=evr[:2],
                 data_label=label)
        
        print(f"  Saved: {save_name}")
        
        results_list.append({
            'label': label,
            'file': save_name,
            'pc1_var': float(evr[0]),
            'pc2_var': float(evr[1]),
            'total_var': float(sum(evr[:2])),
            'n_samples': int(data.shape[0]),
            'n_joints': n_j,
            'norm': norm_name
        })
    
    # Save manifest
    import json
    manifest_path = os.path.join(out_dir, 'balance_export_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Exported {len(results_list)} configurations")
    print(f"Manifest: {manifest_path}")
    
    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Config':<50} {'PC1%':>7} {'PC2%':>7} {'Sum%':>7} {'N':>6} {'J':>3}")
    print(f"{'-'*90}")
    
    by_balance = sorted(results_list, key=lambda r: r['pc1_var'] + r['pc2_var'] * 3, reverse=True)
    for r in by_balance:
        star = "★" if r['pc2_var'] > 0.08 else " "
        print(f"{star} {r['label']:<50} {r['pc1_var']*100:>6.1f}% {r['pc2_var']*100:>6.1f}% {r['total_var']*100:>6.1f}% {r['n_samples']:>6} {r['n_joints']:>3}")
    
    print(f"\n{'='*70}")
    print("TIP: To visualize, run:")
    print(f'  python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_all_grasp_trialmean_all15_none.npz')

if __name__ == '__main__':
    run_and_save_configs()
