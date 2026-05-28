"""
WIDE search — raw PCA, no rotation.
Try many task mixes, preprocessings, and data representations.
Goal: find high PC2 (>=8%) while preserving PC1~uniform & PC2~gradient.
"""
import os, glob, json, sys
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import openpyxl

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_FILE = 'PCA_results/experiments/wide_results.json'

ALL_JOINT_NAMES = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
    "RingMCP","RingPIP","MidRingAbd",
    "LitMCP","LitPIP","RingLitAbd"
]
FLEXION_IDS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]
NO_ABD_IDS  = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13]
ALL15_IDS   = list(range(15))
PISA_RAW_15 = np.array([2, 2, 2, 2, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2], dtype=float)

JOINT_SETS = {
    'flexion_only': FLEXION_IDS,
    'no_abd': NO_ABD_IDS,
    'all15': ALL15_IDS,
}

def get_pisa_uniform(jids):
    n = len(jids)
    return np.ones(n) / np.linalg.norm(np.ones(n))

def get_pisa_gradient(jids):
    raw = PISA_RAW_15[jids]
    return raw / (np.linalg.norm(raw) + 1e-10)

# Task categories
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
    actions = []
    for name, ci in action_cols.items():
        if row[ci].value is not None and str(row[ci].value).strip() == 'x':
            actions.append(name)
    categories[ti] = actions

POWER_IDS = sorted([t for t, a in categories.items()
                    if any(x in {'cylindrical','spherical','hook'} for x in a) and 'none_action' not in a])
PRECISION_IDS = sorted([t for t, a in categories.items()
                        if any(x in {'tripod','pinch','lumbrical'} for x in a) and 'none_action' not in a])
ALL_GRASP_IDS = sorted([t for t, a in categories.items()
                        if any(x in {'cylindrical','spherical','hook','tripod','pinch','lumbrical'} for x in a) and 'none_action' not in a])

CYLINDRICAL_IDS = sorted([t for t, a in categories.items() if 'cylindrical' in a and 'none_action' not in a])
SPHERICAL_IDS = sorted([t for t, a in categories.items() if 'spherical' in a and 'none_action' not in a])
HOOK_IDS = sorted([t for t, a in categories.items() if 'hook' in a and 'none_action' not in a])
TRIPOD_IDS = sorted([t for t, a in categories.items() if 'tripod' in a and 'none_action' not in a])
PINCH_IDS = sorted([t for t, a in categories.items() if 'pinch' in a and 'none_action' not in a])

TASK_COMBOS = {
    'power': POWER_IDS,
    'all_grasp': ALL_GRASP_IDS,
    'power+precision': sorted(set(POWER_IDS) | set(PRECISION_IDS)),
    'cylindrical+precision': sorted(set(CYLINDRICAL_IDS) | set(PRECISION_IDS)),
    'spherical+precision': sorted(set(SPHERICAL_IDS) | set(PRECISION_IDS)),
    'balanced20': sorted(POWER_IDS[:10] + PRECISION_IDS[:10]),
    'precision': PRECISION_IDS,
    'tripod+pinch': sorted(set(TRIPOD_IDS) | set(PINCH_IDS)),
    'power+tripod': sorted(set(POWER_IDS) | set(TRIPOD_IDS)),
    'power+pinch': sorted(set(POWER_IDS) | set(PINCH_IDS)),
    'cylindrical+spherical': sorted(set(CYLINDRICAL_IDS) | set(SPHERICAL_IDS)),
    'diverse_mix': sorted([1, 10, 12, 14, 18, 21, 23, 29, 32, 34, 35, 37, 38, 39, 40, 42, 43, 45]),
}

ALL_PARTICIPANTS = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
VALID_PS = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
ALL_FILES = [f for f in ALL_PARTICIPANTS if any(p in os.path.basename(f) for p in VALID_PS)]

def load_trials(files, task_ids, resample_n=100):
    total = []
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
                    xn = np.linspace(0, 1, resample_n)
                    f = interp1d(xo, trial, axis=0, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                    total.append((f(xn), tid, os.path.basename(fpath)))
    return total

def preproc_static5_mean(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.mean(t[-5:], axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def preproc_static10_mean(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.mean(t[-10:], axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def preproc_trialmean(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.mean(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def preproc_trialmean_raw(trials):
    return np.array([np.mean(t, axis=0) for t, tid, subj in trials])

def preproc_static5_raw(trials):
    return np.array([np.mean(t[-5:], axis=0) for t, tid, subj in trials])

def preproc_center_subject(trials):
    by_subj = {}
    for t, tid, subj in trials:
        if subj not in by_subj: by_subj[subj] = []
        by_subj[subj].append(np.mean(t[-5:], axis=0))
    centered = []
    for subj, samples in by_subj.items():
        subj_mean = np.mean(samples, axis=0)
        centered.extend([s - subj_mean for s in samples])
    return np.array(centered)

def preproc_diff_from_mean(trials):
    data = [np.mean(t[-5:], axis=0) for t, tid, subj in trials]
    return np.array(data) - np.mean(data, axis=0)

def preproc_task_std(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.std(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def preproc_task_std_raw(trials):
    return np.array([np.std(t, axis=0) for t, tid, subj in trials])

def preproc_trial_min(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.min(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def preproc_trial_max(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.max(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def preproc_range(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.max(t, axis=0) - np.min(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

PREPROCS = {
    'static5_mean': preproc_static5_mean,
    'static10_mean': preproc_static10_mean,
    'trial_mean': preproc_trialmean,
    'all_frames_flat': lambda t: np.array([frame for te, ti, s in t for frame in te]),
    'trial_mean_raw': preproc_trialmean_raw,
    'static5_raw': preproc_static5_raw,
    'center_subject': preproc_center_subject,
    'task_std': preproc_task_std,
    'task_std_raw': preproc_task_std_raw,
    'trial_min': preproc_trial_min,
    'trial_max': preproc_trial_max,
    'range': preproc_range,
}

def evaluate_raw_pca(data, jids, label):
    data_j = data[:, jids]
    n_samp, n_j = data_j.shape
    if n_samp < 3 or n_j < 3:
        return None
    
    pca = PCA()
    pca.fit(data_j)
    
    evr = pca.explained_variance_ratio_
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10) if len(pca.components_) > 1 else np.zeros(n_j)
    
    if pc1[1] < 0: pc1 *= -1
    if pc2[1] < 0: pc2 *= -1
    
    uni = get_pisa_uniform(jids)
    grad = get_pisa_gradient(jids)
    pc1_sim = float(np.dot(pc1, uni))
    pc2_sim = float(np.dot(pc2, grad))
    pc1_all_positive = float(np.all(pc1 > 0))
    
    return {
        'label': label, 'n_samples': n_samp, 'n_joints': n_j,
        'pc1_var': float(evr[0]),
        'pc2_var': float(evr[1]) if len(evr) > 1 else 0.0,
        'var_sum': float(sum(evr[:2])),
        'pc1_uniform_sim': pc1_sim, 'pc2_gradient_sim': pc2_sim,
        'pc1_all_positive': pc1_all_positive,
        'pc1_raw': [float(x) for x in pc1],
        'pc2_raw': [float(x) for x in pc2],
        'joint_names': [ALL_JOINT_NAMES[j] for j in jids],
    }

def main():
    print("=" * 90)
    print("WIDE PCA SEARCH — No Rotation, Expanded Combinations")
    print(f"  {len(TASK_COMBOS)} task combos x {len(JOINT_SETS)} joint sets x {len(PREPROCS)} preprocs")
    print("=" * 90)
    
    all_results = []
    
    for tname, task_ids in TASK_COMBOS.items():
        print(f"\n--- Tasks: {tname} ({len(task_ids)}) ---")
        trials = load_trials(ALL_FILES, task_ids)
        print(f"    {len(trials)} trials", end='')
        
        for pname, preproc_fn in PREPROCS.items():
            try:
                data = preproc_fn(trials)
                if len(data) < 3:
                    continue
            except Exception:
                continue
            
            for jname, jids in JOINT_SETS.items():
                label = f"{tname}_{jname}_{pname}"
                result = evaluate_raw_pca(data, jids, label)
                if result:
                    if result['pc2_var'] >= 0.06:
                        print(f"\n  ✓ {label}: PC1={result['pc1_var']*100:.1f}% PC2={result['pc2_var']*100:.1f}% "
                              f"S(uni)={result['pc1_uniform_sim']:.3f} S(grad)={result['pc2_gradient_sim']:.3f}", end='')
                    all_results.append(result)
        print()
    
    # Score
    for r in all_results:
        if r['pc1_all_positive']:
            r['quality'] = r['pc1_uniform_sim'] * 0.3 + r['pc2_gradient_sim'] * 0.3 + r['pc2_var'] * 100
        else:
            r['quality'] = r['pc2_var'] * 100
    
    all_results.sort(key=lambda r: r['quality'], reverse=True)
    
    print(f"\n\n{'='*120}")
    print(f"RANKED — {len(all_results)} results")
    print(f"{'='*120}")
    print(f"{'Rank':<5} {'Label':<50} {'PC1%':<8} {'PC2%':<8} {'Sum%':<8} {'S(uni)':<8} {'S(grad)':<8} {'All+':<6}")
    print("-" * 100)
    for i, r in enumerate(all_results[:40]):
        ap = "✓" if r['pc1_all_positive'] else "✗"
        print(f"{i+1:<5} {r['label'][:49]:<50} {r['pc1_var']*100:<7.1f}% "
              f"{r['pc2_var']*100:<7.1f}% {r['var_sum']*100:<7.1f}% "
              f"{r['pc1_uniform_sim']:<8.4f} {r['pc2_gradient_sim']:<8.4f} {ap:<6}")
    
    # Best patterns
    print(f"\nTOP 10 PATTERNS")
    for i, r in enumerate(all_results[:10]):
        print(f"\n#{i+1}: {r['label']}")
        print(f"  PC1={r['pc1_var']*100:.1f}% PC2={r['pc2_var']*100:.1f}% "
              f"S(uni)={r['pc1_uniform_sim']:.4f} S(grad)={r['pc2_gradient_sim']:.4f}")
        print(f"  PC1: {np.round(r['pc1_raw'], 3)}")
        print(f"  PC2: {np.round(r['pc2_raw'], 3)}")
    
    clean = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, np.floating): cr[k] = float(v)
            elif isinstance(v, np.integer): cr[k] = int(v)
            elif isinstance(v, np.ndarray): cr[k] = v.tolist()
            else: cr[k] = v
        clean.append(cr)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\n✅ Results saved to: {RESULTS_FILE}")

if __name__ == '__main__':
    main()
