"""
Export the best candidates so far to synergy-viewer compatible files.
Raw PCA, no rotation, full 15-D space for visualization.
"""
import os, glob
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import openpyxl

SAVE_DIR = 'PCA_results/MyoKi'
os.makedirs(SAVE_DIR, exist_ok=True)

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

# Task categories
wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
ws = wb.active
categories = {}
action_cols = {'cylindrical': 11, 'spherical': 12, 'hook': 13, 'tripod': 14,
               'pinch': 15, 'lumbrical': 16, 'complex': 17, 'none_action': 18}
for row in ws.iter_rows(min_row=4, values_only=False):
    tid = row[0].value
    if tid is None: continue
    ti = int(tid)
    actions = []
    for name, ci in action_cols.items():
        if row[ci].value is not None and str(row[ci].value).strip() == 'x':
            actions.append(name)
    categories[ti] = actions

POWER_IDS = sorted([t for t, a in categories.items()
                    if any(x in {'cylindrical','spherical','hook'} for x in a) and 'none_action' not in a])
TRIPOD_IDS = sorted([t for t, a in categories.items() if 'tripod' in a and 'none_action' not in a])
PINCH_IDS = sorted([t for t, a in categories.items() if 'pinch' in a and 'none_action' not in a])
PRECISION_IDS = sorted([t for t, a in categories.items()
                        if any(x in {'tripod','pinch','lumbrical'} for x in a) and 'none_action' not in a])

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

def save(name, display_name, jids, task_ids, preproc_fn):
    trials = load_trials(ALL_FILES, task_ids)
    data = preproc_fn(trials)
    
    # Data on selected joints
    data_j = data[:, jids]
    
    # Full 15-D mean for viewer
    full_mean = np.mean(data, axis=0)
    
    # PCA on selected joints
    pca = PCA()
    pca.fit(data_j)
    
    evr = pca.explained_variance_ratio_
    pc1_s = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    
    # Build full 15-D components (padding excluded joints with 0)
    n_comp = pca.n_components_
    full_comp = np.zeros((n_comp, 15))
    for i in range(n_comp):
        full_comp[i, jids] = pca.components_[i]
    
    # Sign-flip PC1 so thumb positive
    if full_comp[0, 1] < 0:
        full_comp[0] *= -1
    if n_comp > 1 and full_comp[1, 1] < 0:
        full_comp[1] *= -1
    
    # Compute Pisa similarity (no rotation)
    n_j = len(jids)
    uni = np.ones(n_j) / np.linalg.norm(np.ones(n_j))
    grad = PISA_RAW_15[jids] / np.linalg.norm(PISA_RAW_15[jids])
    pc1_sim = float(np.dot(pc1_s, uni))
    
    if n_comp > 1:
        pc2_s = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
        if pc2_s[1] < 0: pc2_s *= -1
        pc2_sim = float(np.dot(pc2_s, grad))
    else:
        pc2_sim = 0
    
    pc2_var = evr[1]*100 if len(evr) > 1 else 0
    desc = (f"{display_name} | "
            f"PC1={evr[0]*100:.1f}% PC2={pc2_var:.1f}% "
            f"S(uni)={pc1_sim:.3f} S(grad)={pc2_sim:.3f}")
    
    fpath = os.path.join(SAVE_DIR, f'pca_myoki_{name}.npz')
    np.savez(fpath,
             components=full_comp,
             mean=full_mean,
             explained_variance=evr,
             data_label=desc)
    
    print(f"✅ {fpath}")
    print(f"   {desc}")
    print(f"   PC1 (all joints): {np.round(full_comp[0], 3)}")
    if n_comp > 1:
        print(f"   PC2 (all joints): {np.round(full_comp[1], 3)}")
    
    return {
        'name': name,
        'file': f'pca_myoki_{name}.npz',
        'desc': desc,
        'pc1_var': float(evr[0]),
        'pc2_var': float(evr[1]) if len(evr) > 1 else 0,
        'var_sum': float(sum(evr[:2])),
        'pc1_sim': pc1_sim,
        'pc2_sim': pc2_sim,
    }

# ============================================
# Preprocessing functions
# ============================================
def static5_mean(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.mean(t[-5:], axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def static10_mean(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.mean(t[-10:], axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def trial_mean(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.mean(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def all_frames_flat(trials):
    return np.array([frame for te, ti, s in trials for frame in te])

def trial_min(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.min(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def trial_max(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.max(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def range_preproc(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.max(t, axis=0) - np.min(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def task_std(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.std(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

# ============================================
# Best candidates from wide search
# ============================================
CANDIDATES = [
    # Category 1: POSTURE-BASED (what Pisa paper does)
    {
        'name': 'power_flexion_only_static5',
        'desc': 'Power grasps, task-avg static5, flexion-only',
        'jids': FLEXION_IDS,
        'tasks': POWER_IDS,
        'preproc': static5_mean,
    },
    {
        'name': 'all_grasp_flexion_only_static10',
        'desc': 'All grasps, task-avg static10, flexion-only',
        'jids': FLEXION_IDS,
        'tasks': list(set(POWER_IDS) | set(PRECISION_IDS)),
        'preproc': static10_mean,
    },
    {
        'name': 'precision_flexion_only_static5',
        'desc': 'Precision tasks, task-avg static5, flexion-only',
        'jids': FLEXION_IDS,
        'tasks': PRECISION_IDS,
        'preproc': static5_mean,
    },
    
    # Category 2: RANGE-BASED (PC2 high)
    {
        'name': 'power_no_abd_range',
        'desc': 'Power grasps, task-avg range, no abd (S=0.97/0.70)',
        'jids': NO_ABD_IDS,
        'tasks': POWER_IDS,
        'preproc': range_preproc,
    },
    {
        'name': 'power_no_abd_task_std',
        'desc': 'Power grasps, task-avg std, no abd (S=0.96/0.61)',
        'jids': NO_ABD_IDS,
        'tasks': POWER_IDS,
        'preproc': task_std,
    },
    
    # Category 3: ALL FRAMES (many samples, high PC2)
    {
        'name': 'precision_flexion_only_allframes',
        'desc': 'Precision, all frames flat, flexion-only',
        'jids': FLEXION_IDS,
        'tasks': PRECISION_IDS,
        'preproc': all_frames_flat,
    },
    {
        'name': 'cyl_precision_flexion_only_allframes',
        'desc': 'Cyl+precision, all frames flat, flexion-only',
        'jids': FLEXION_IDS,
        'tasks': sorted(list(set([t for t, a in categories.items() if 'cylindrical' in a and 'none_action' not in a]) | set(PRECISION_IDS))),
        'preproc': all_frames_flat,
    },
    {
        'name': 'tripod_pinch_flexion_only_allframes',
        'desc': 'Tripod+pinch, all frames flat, flexion-only',
        'jids': FLEXION_IDS,
        'tasks': sorted(set(TRIPOD_IDS) | set(PINCH_IDS)),
        'preproc': all_frames_flat,
    },
    
    # Category 4: TRIAL MAX (peak flexion)
    {
        'name': 'precision_flexion_only_trialmax',
        'desc': 'Precision, task-avg trial-max, flexion-only',
        'jids': FLEXION_IDS,
        'tasks': PRECISION_IDS,
        'preproc': trial_max,
    },
    
    # Category 5: MIXED power+precision
    {
        'name': 'balanced20_flexion_only_task_std',
        'desc': 'Balanced20, task-avg std, flexion-only',
        'jids': FLEXION_IDS,
        'tasks': sorted(POWER_IDS[:10] + PRECISION_IDS[:10]),
        'preproc': task_std,
    },
    {
        'name': 'balanced20_no_abd_task_std',
        'desc': 'Balanced20, task-avg std, no abd',
        'jids': NO_ABD_IDS,
        'tasks': sorted(POWER_IDS[:10] + PRECISION_IDS[:10]),
        'preproc': task_std,
    },
]

def main():
    print("=" * 70)
    print("EXPORTING BEST CANDIDATES FOR VISUALIZATION")
    print("(RAW PCA, no rotation)")
    print("=" * 70)
    
    all_meta = []
    for c in CANDIDATES:
        print(f"\n--- {c['name']} ---")
        try:
            meta = save(c['name'], c['desc'], c['jids'], c['tasks'], c['preproc'])
            all_meta.append(meta)
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Name':<40} {'PC1%':<8} {'PC2%':<8} {'Sum%':<8} {'S(uni)':<8} {'S(grad)':<8}")
    print("-" * 80)
    for m in sorted(all_meta, key=lambda x: x['var_sum'] / 2 + x['pc1_sim'] * 3 + x['pc2_sim'] * 3, reverse=True):
        print(f"{m['name']:<40} {m['pc1_var']*100:<7.1f}% {m['pc2_var']*100:<7.1f}% "
              f"{m['var_sum']*100:<7.1f}% {m['pc1_sim']:<8.4f} {m['pc2_sim']:<8.4f}")
    
    print(f"\nRun viewer:")
    for m in all_meta[:3]:
        print(f"  python code/myoki/synergy_myoki.py --participant {m['name']}")

if __name__ == '__main__':
    main()
