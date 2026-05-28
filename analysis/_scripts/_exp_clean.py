"""
CLEAN experiment — RAW PCA only, NO rotation, NO Procrustes.
Report only the actual eigendecomposition results.

For each experiment, we evaluate:
  PC1: is it all-same-sign (co-activation synergy)?
  PC2: does it follow a gradient pattern (thumb→little)?

We measure similarity to Pisa targets by:
  PC1_sim = cos(PC1,  1/√n)   — alignment with uniform vector
  PC2_sim = cos(PC2,  grad/‖grad‖) — alignment with gradient vector

These are done on the RAW PC vectors (with sign flip if needed).
"""
import os, glob, json, sys
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_FILE = 'PCA_results/experiments/clean_results.json'

# ============================================================
# Constants
# ============================================================
ALL_JOINT_NAMES = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
    "RingMCP","RingPIP","MidRingAbd",
    "LitMCP","LitPIP","RingLitAbd"
]

FLEXION_IDS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]   # 10 joints
NO_ABD_IDS  = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13]  # 12 joints
ALL15_IDS   = list(range(15))

# Pisa targets in full 15-D
PISA_RAW_15 = np.array([2, 2, 2, 2, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2], dtype=float)

def get_pisa_uniform(jids):
    """Uniform vector (all 1s) for given joint subset."""
    n = len(jids)
    return np.ones(n) / np.linalg.norm(np.ones(n))

def get_pisa_gradient(jids):
    """Gradient vector for given joint subset."""
    raw = PISA_RAW_15[jids]
    return raw / np.linalg.norm(raw)

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

# ============================================================
# Task definitions
# ============================================================
import openpyxl
wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
ws = wb.active
categories, task_names = {}, {}
action_cols = {'cylindrical': 11, 'spherical': 12, 'hook': 13, 'tripod': 14,
               'pinch': 15, 'lumbrical': 16, 'complex': 17, 'none_action': 18}
for row in ws.iter_rows(min_row=4, values_only=False):
    tid = row[0].value
    if tid is None: continue
    ti = int(tid)
    task_names[ti] = str(row[1].value).strip() if row[1].value else f"Task {ti}"
    actions = []
    for name, ci in action_cols.items():
        if row[ci].value is not None and str(row[ci].value).strip() == 'x':
            actions.append(name)
    categories[ti] = actions

POWER_TYPES = {'cylindrical', 'spherical', 'hook'}
ALL_GRASP_TYPES = {'cylindrical', 'spherical', 'hook', 'tripod', 'pinch', 'lumbrical'}

POWER_IDS = sorted([t for t, a in categories.items()
                    if any(x in POWER_TYPES for x in a) and 'none_action' not in a])
ALL_GRASP_IDS = sorted([t for t, a in categories.items()
                        if any(x in ALL_GRASP_TYPES for x in a) and 'none_action' not in a])

# ============================================================
# Preprocessing methods
# ============================================================
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

def preproc_trial_mean(trials):
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        by_task[tid].append(np.mean(t, axis=0))
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def preproc_all_frames_stacked(trials):
    """Stack all frames across all trials per task, then average per task."""
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task: by_task[tid] = []
        for frame in t:
            by_task[tid].append(frame)
    return np.array([np.mean(v, axis=0) for v in by_task.values()])

def preproc_all_frames_flat(trials):
    """All frames individually (not averaged)."""
    all_frames = []
    for t, tid, subj in trials:
        for frame in t:
            all_frames.append(frame)
    return np.array(all_frames)

# ============================================================
# Evaluation — NO ROTATION, just raw PC check
# ============================================================
def evaluate_raw_pca(data, jids, label):
    """Run PCA, return raw results + similarity to Pisa patterns (no rotation)."""
    data_j = data[:, jids]
    n_samp, n_j = data_j.shape
    
    if n_samp < 5 or n_j < 3:
        return None
    
    pca = PCA()
    pca.fit(data_j)
    
    evr = pca.explained_variance_ratio_
    pc1 = pca.components_[0]
    pc2 = pca.components_[1] if len(pca.components_) > 1 else np.zeros(n_j)
    
    # Normalize PC vectors
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-10)
    pc2 = pc2 / (np.linalg.norm(pc2) + 1e-10)
    
    # Sign-flip PC1 so that thumb flexions are positive (convention)
    if pc1[1] < 0:  # ThumbMCP is index 1, flip if negative
        pc1 *= -1
    # Sign-flip PC2 so that thumb has positive value (matching gradient convention)
    if pc2[1] < 0:
        pc2 *= -1
    
    # Get Pisa targets for these joints
    uni = get_pisa_uniform(jids)
    grad = get_pisa_gradient(jids)
    
    # Dot products (cosine similarity since vectors are normalized)
    pc1_sim = float(np.dot(pc1, uni))
    pc2_sim = float(np.dot(pc2, grad))
    
    # Check if PC1 is all-same-sign (co-activation)
    pc1_all_same_sign = np.all(pc1 > 0) or np.all(pc1 < 0)
    pc1_positive_ratio = np.mean(pc1 > 0)
    
    # Also report how "gradient-like" PC2 is: correlation with position index
    # Position index: thumb=0, index=1, middle=2, ring=3, little=4
    # Map joint indices to finger ordering
    finger_order = {
        0: 0, 1: 0, 2: 0, 3: 0,    # Thumb
        4: 1, 5: 1,                # Index
        6: 2, 7: 2,                # Middle  
        8: 1,                      # Idx-Mid abd
        9: 3, 10: 3,               # Ring
        11: 2,                     # Mid-Ring abd
        12: 4, 13: 4,              # Little
        14: 3                      # Ring-Lit abd
    }
    finger_pos = np.array([finger_order.get(j, -1) for j in jids])
    valid = finger_pos >= 0
    if np.sum(valid) >= 3:
        corr = np.corrcoef(finger_pos[valid], pc2[valid])[0, 1]
        # Gradient should have negative correlation (thumb high → little low)
        grad_strength = -corr if not np.isnan(corr) else 0
    else:
        grad_strength = 0
    
    return {
        'label': label,
        'n_tasks': n_samp,
        'n_joints': n_j,
        'jids': [int(j) for j in jids],
        'pc1_var': float(evr[0]),
        'pc2_var': float(evr[1]) if len(evr) > 1 else 0.0,
        'var_sum': float(sum(evr[:2])),
        'pc1_uniform_sim': pc1_sim,         # How close to all-ones
        'pc2_gradient_sim': pc2_sim,         # How close to gradient
        'pc1_all_same_sign': bool(pc1_all_same_sign),
        'pc1_positive_ratio': float(pc1_positive_ratio),
        'gradient_strength': float(grad_strength),
        'pc1_raw': [float(x) for x in pc1],
        'pc2_raw': [float(x) for x in pc2],
        'joint_names': [ALL_JOINT_NAMES[j] for j in jids],
    }

# ============================================================
# Experiment definitions
# ============================================================
ALL_PARTICIPANTS = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
VALID_PS = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
FILES = [f for f in ALL_PARTICIPANTS if any(p in os.path.basename(f) for p in VALID_PS)]

TASK_COMBOS = {
    'power_only': POWER_IDS,
    'all_grasp': ALL_GRASP_IDS,
}

JOINT_SETS = {
    'all15': ALL15_IDS,
    'no_abd': NO_ABD_IDS,
    'flexion_only': FLEXION_IDS,
}

PREPROCS = {
    'static5_mean': preproc_static5_mean,
    'static10_mean': preproc_static10_mean,
    'trial_mean': preproc_trial_mean,
    'all_frames_stacked': preproc_all_frames_stacked,
    'all_frames_flat': preproc_all_frames_flat,
}

def main():
    print("=" * 80)
    print("CLEAN PCA EXPERIMENT — NO ROTATION, RAW RESULTS ONLY")
    print("=" * 80)
    
    all_results = []
    
    for tname, task_ids in TASK_COMBOS.items():
        print(f"\n--- Tasks: {tname} ({len(task_ids)} tasks) ---")
        trials = load_trials(FILES, task_ids)
        print(f"  Loaded {len(trials)} trials")
        
        for pname, preproc_fn in PREPROCS.items():
            try:
                data = preproc_fn(trials)
                print(f"  Preproc {pname}: {data.shape}")
            except Exception as e:
                print(f"  Preproc {pname}: FAILED ({e})")
                continue
            
            for jname, jids in JOINT_SETS.items():
                label = f"{tname}_{jname}_{pname}"
                result = evaluate_raw_pca(data, jids, label)
                if result:
                    all_results.append(result)
                    
                    # Print interesting results
                    if result['pc2_var'] >= 0.06:
                        print(f"    ✓ {label}: "
                              f"PC1={result['pc1_var']*100:.1f}% "
                              f"PC2={result['pc2_var']*100:.1f}% "
                              f"S(uni)={result['pc1_uniform_sim']:.3f} "
                              f"S(grad)={result['pc2_gradient_sim']:.3f} "
                              f"GradStr={result['gradient_strength']:.3f}")
    
    # ============================================================
    # RANKING — based on simple criteria
    # ============================================================
    print(f"\n{'='*120}")
    print("RANKING — ALL RESULTS (raw PCA, no rotation)")
    print(f"{'='*120}")
    
    # Metric: good solution = high var_sum + high pc2_var + high pc1_uniform_sim + high pc2_gradient_sim
    for r in all_results:
        # Balanced score: rewards high PC2 + good similarity + high total var
        r['score'] = (r['pc2_var'] * 100) + (r['pc1_uniform_sim'] * 10) + (r['pc2_gradient_sim'] * 10) + r['var_sum']
    
    all_results.sort(key=lambda r: r['score'], reverse=True)
    
    print(f"{'Rank':<5} {'Label':<45} {'PC1%':<8} {'PC2%':<8} {'Sum%':<8} {'S(uni)':<8} {'S(grad)':<8} {'GS':<6} {'Sc':<8}")
    print("-" * 110)
    for i, r in enumerate(all_results):
        print(f"{i+1:<5} {r['label']:<45} {r['pc1_var']*100:<7.1f}% "
              f"{r['pc2_var']*100:<7.1f}% {r['var_sum']*100:<7.1f}% "
              f"{r['pc1_uniform_sim']:<8.4f} {r['pc2_gradient_sim']:<8.4f} "
              f"{r['gradient_strength']:<6.3f} {r['score']:<8.3f}")
    
    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved: {RESULTS_FILE}")
    
    # ============================================================
    # SHOW PC PATTERNS FOR TOP RESULTS
    # ============================================================
    print(f"\n{'='*120}")
    print("TOP 5 — RAW PC PATTERNS")
    print(f"{'='*120}")
    for i, r in enumerate(all_results[:5]):
        print(f"\n--- #{i+1}: {r['label']} ---")
        print(f"  Variance: PC1={r['pc1_var']*100:.1f}%, PC2={r['pc2_var']*100:.1f}%")
        print(f"  Similarity: PC1~uniform={r['pc1_uniform_sim']:.4f}, PC2~gradient={r['pc2_gradient_sim']:.4f}")
        print(f"  PC1 signs: {r['pc1_positive_ratio']*100:.0f}% positive")
        jn = r['joint_names']
        pc1v = np.array(r['pc1_raw'])
        pc2v = np.array(r['pc2_raw'])
        print(f"  Joints: {jn}")
        print(f"  PC1:    {np.round(pc1v, 3)}")
        print(f"  PC2:    {np.round(pc2v, 3)}")
        
        # Check for gradient
        pc2_order = pc2v[np.argsort([jn.index(n) for n in jn])] if len(jn) > 3 else pc2v
        print(f"  Gradient check: values={np.round(pc2v, 3)}")

if __name__ == '__main__':
    main()
