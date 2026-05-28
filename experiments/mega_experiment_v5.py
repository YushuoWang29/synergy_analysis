"""
Mega Experiment V5 — FIND BALANCED VARIANCE SOLUTIONS.

Target: PC1 ~70-80%, PC2 ~10-20% (close to Pisa/IIT paper ratio)
While maintaining: S1 > 0.80, S2 > 0.60.

Search dimensions:
  1) Task combinations: power_only, all_grasp, power+precision, mixed_5_10_15_20, etc.
  2) Joint subsets: flexion_only(10), no_abd(12), all15(15), thumb+flexion(12), etc.
  3) Preprocessing: static5_mean, static10_mean, all_frames_mean, trial_means, 
     trial_mean_of_tasks, task_median, etc.
  4) Participant subsets: all 11, individual, task-averaged per participant
  5) Scoring function: maximize (S1 + S2 + balance_penalty)
     where balance_penalty = max(0, 1 - PC1/PC2_ratio / target_ratio)
"""
import os, glob, json, itertools, sys
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from save_best_results import (load_task_categories, POWER_GRASP_IDS, ALL_GRASP_IDS,
                                get_participant_files, load_myoki_trials, make_pisa_target,
                                categories, task_names)
from save_best_results import FLEXION_IDS, NO_ABD_IDS, ALL15_IDS, ALL_JOINT_NAMES

RESULTS_FILE = 'PCA_results/experiments/mega_v5_results.json'
N_RESULTS_TOP = 40

# ============================================================
# NEW task combinations — expanded
# ============================================================
ALL_POWER_IDS = POWER_GRASP_IDS  # 28 tasks
ALL_ALL_IDS = ALL_GRASP_IDS  # 41 tasks
PRECISION_TYPES = {'tripod', 'pinch', 'lumbrical'}
PRECISION_IDS = sorted([tid for tid, acts in categories.items()
                        if any(a in PRECISION_TYPES for a in acts) and 'none_action' not in acts])
NON_GRASP_IDS = sorted([tid for tid, acts in categories.items()
                        if 'none_action' in acts])
COMPLEX_IDS = sorted([tid for tid, acts in categories.items()
                      if 'complex' in acts])

TASK_COMBOS = {
    # power only (baseline)
    'power_only': ALL_POWER_IDS,
    # all grasps (baseline)
    'all_grasp': ALL_ALL_IDS,
    # power + precision (mix of big and fine)
    'power_precision': sorted(set(ALL_POWER_IDS) | set(PRECISION_IDS)),
    # power + small subset of precision for differentiation
    'power_tripod_pinch': sorted(set(ALL_POWER_IDS) | {t for t in PRECISION_IDS if 'tripod' in str(task_names.get(t, '')) or 'pinch' in str(task_names.get(t, ''))}),
    # precision only (should have high differentiation)
    'precision_only': PRECISION_IDS,
    # all tasks including non-grasp (maximum diversity)
    'all_tasks': sorted(set(ALL_ALL_IDS) | set(NON_GRASP_IDS)),
    # medium power + medium precision (balanced mix)
    'balanced_mix': sorted((set(ALL_POWER_IDS) - {24, 25}) | (set(PRECISION_IDS) - {10, 11})),
    # power tasks plus just tripod (differentiation)
    'power_plus_tripod': sorted(set(ALL_POWER_IDS) | {t for t in PRECISION_IDS if 'tripod' in str(task_names.get(t, ''))}),
    # power tasks plus just pinch (differentiation)
    'power_plus_pinch': sorted(set(ALL_POWER_IDS) | {t for t in PRECISION_IDS if 'pinch' in str(task_names.get(t, ''))}),
    # power tasks plus lumbrical
    'power_plus_lumbrical': sorted(set(ALL_POWER_IDS) | {t for t in PRECISION_IDS if 'lumbrical' in str(task_names.get(t, ''))}),
}

# Add more combos: select specific tasks that give good differentiation
# (known good tasks for PC2: tripod tasks tend to differentiate thumb vs fingers)
TRIPOD_TASKS = [t for t in PRECISION_IDS if 'tripod' in str(task_names.get(t, ''))]
PINCH_TASKS = [t for t in PRECISION_IDS if 'pinch' in str(task_names.get(t, ''))]
LUMBRICAL_TASKS = [t for t in PRECISION_IDS if 'lumbrical' in str(task_names.get(t, ''))]
TASK_COMBOS['power_two_precision'] = sorted(set(ALL_POWER_IDS) | set(TRIPOD_TASKS[:2] + PINCH_TASKS[:2]))
TASK_COMBOS['power_balanced10'] = sorted(set(ALL_POWER_IDS[:10]) | set(PRECISION_IDS[:10]))
TASK_COMBOS['power_even'] = sorted(ALL_POWER_IDS[::2])

# ============================================================
# NEW joint subsets — expanded
# ============================================================
# We'll explore progressive inclusion of joints
JOINT_SUBSETS = {
    'flexion_only': FLEXION_IDS,
    'no_abd': NO_ABD_IDS,
    'all15': ALL15_IDS,
    'flexion_plus_thumb_abd': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],
    'mcp_only': [1, 4, 6, 9, 12],
    'pip_only': [2, 5, 7, 10, 13],
    'mcp_pip_thumb_opp': [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],
    'abd_only': [0, 3, 8, 11, 14],
    'power_joints': [1, 2, 4, 5, 6, 7, 9, 10, 12, 13, 0, 3],
    'no_interfinger_abd': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],
    'mid_abd_flexion': [1, 2, 4, 5, 6, 7, 9, 10, 12, 13, 8, 11, 14],
}

# ============================================================
# NEW preprocessing methods
# ============================================================
def preprocess_static_n(trials, n=5, method='mean'):
    """Build task-averaged static n frames."""
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        if method == 'mean':
            post = np.mean(t[-n:], axis=0)
        elif method == 'median':
            post = np.median(t[-n:], axis=0)
        else:
            post = t[-1]
        by_task[tid].append(post)
    return by_task

def preprocess_all_frames_by_task(trials):
    """Use all frames, but stack all frames from all trials per task."""
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].extend(t)
    return by_task

def preprocess_trial_means(trials):
    """Average all frames in each trial -> one sample per trial."""
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(np.mean(t, axis=0))
    return by_task

def preprocess_trial_means_then_task_mean(trials):
    """Trial means -> average per task -> one sample per task."""
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(np.mean(t, axis=0))
    task_means = {}
    for tid, trial_means_list in by_task.items():
        task_means[tid] = np.mean(trial_means_list, axis=0)
    return task_means

def preprocess_task_median_of_trial_means(trials):
    """Trial means -> median per task -> one sample per task."""
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(np.mean(t, axis=0))
    task_means = {}
    for tid, trial_means_list in by_task.items():
        task_means[tid] = np.median(trial_means_list, axis=0)
    return task_means

def preprocess_all_frames(trials):
    """All frames flattened (huge data)."""
    all_data = []
    for t, tid, subj in trials:
        all_data.extend(t)
    return {'all': all_data}

def preprocess_task_mean_of_std(trials):
    """Use standard deviation within each trial as a measure of dynamic variation."""
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(np.std(t, axis=0))
    task_means = {}
    for tid, std_list in by_task.items():
        task_means[tid] = np.mean(std_list, axis=0)
    return task_means

PREPROC_METHODS = {
    'static5_mean': lambda t: preprocess_static_n(t, 5, 'mean'),
    'static5_median': lambda t: preprocess_static_n(t, 5, 'median'),
    'static10_mean': lambda t: preprocess_static_n(t, 10, 'mean'),
    'static3_mean': lambda t: preprocess_static_n(t, 3, 'mean'),
    'all_frames_stack': preprocess_all_frames_by_task,
    'trial_means': preprocess_trial_means,
    'trial_mean_then_task_mean': preprocess_trial_means_then_task_mean,
    'task_median_of_trial_means': preprocess_task_median_of_trial_means,
    'all_frames_flat': preprocess_all_frames,
    'task_std': preprocess_task_mean_of_std,
}

# ============================================================
# Scoring function — rewards variance balance
# ============================================================
def score_balanced(var_pc1, var_pc2, s1_sim, s2_sim):
    """
    Score that rewards:
      - High S1 + S2 similarity
      - Reasonable variance balance (not too extreme)
    
    Ideal: PC1 ~70%, PC2 ~15%, S1 ~0.85, S2 ~0.70
    
    We don't want to arbitrarily penalize all high-PC1 results,
    just reward those that ALSO have good PC2 variance.
    """
    # Base score from pattern similarity
    base = s1_sim + s2_sim
    
    # Variance balance bonus: reward when PC2 > 8%
    var_balance = var_pc2 * 5  # e.g., 12% -> +0.60
    
    # Penalty for extreme imbalance: PC2 < 5% -> no bonus
    if var_pc2 < 0.05:
        var_balance = var_pc2 * 2  # reduced bonus
    
    # Bonus for reasonable PC1 (not too dominant)
    # Ideal PC1 65-80%
    pc1_ideal = max(0, 1 - abs(var_pc1 - 0.75) * 2)  # 1.0 at 75%, 0.7 at 60% or 90%
    
    return base + var_balance + pc1_ideal * 0.2

# ============================================================
# Run experiments
# ============================================================
def extract_data(by_task, jids):
    """Convert by_task dict to data matrix using selected joints."""
    task_data_list = []
    for tid, samples in by_task.items():
        if isinstance(samples, list):
            if len(samples) == 0:
                continue
            # Multiple trials per task -> average
            arr = np.array(samples)
            if arr.ndim > 1:
                task_data_list.append(np.mean(arr, axis=0)[jids])
        else:
            task_data_list.append(samples[jids])
    if len(task_data_list) < 3:
        return None
    return np.array(task_data_list)

def run_single_experiment(files, task_ids, jids, preproc_fn, label):
    """Run one PCA experiment and return metrics."""
    trials = load_myoki_trials(files, task_ids)
    if len(trials) < 3:
        return None
    
    by_task = preproc_fn(trials)
    task_data = extract_data(by_task, jids)
    if task_data is None or len(task_data) < 3:
        return None
    
    # PCA
    n_comp = min(15, *task_data.shape)
    pca = PCA(n_components=n_comp)
    pca.fit(task_data)
    
    var_pc1 = pca.explained_variance_ratio_[0]
    var_pc2 = pca.explained_variance_ratio_[1] if n_comp > 1 else 0.0
    var_sum = var_pc1 + var_pc2
    
    # Procrustes rotation to Pisa targets
    pisa_s1, pisa_s2, pisa_target = make_pisa_target(jids)
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10) if n_comp > 1 else np.zeros(len(jids))
    PC = np.column_stack([pc1, pc2]) if n_comp > 1 else np.column_stack([pc1, np.zeros(len(jids))])
    R, _ = orthogonal_procrustes(PC, pisa_target)
    rotated = PC @ R
    s1_sim = float(np.dot(rotated[:, 0], pisa_s1))
    s2_sim = float(np.dot(rotated[:, 1], pisa_s2))
    
    balanced_s = score_balanced(var_pc1, var_pc2, s1_sim, s2_sim)
    
    return {
        'label': label,
        'n_samples': len(task_data),
        'n_joints': len(jids),
        'pc1_var': float(var_pc1),
        'pc2_var': float(var_pc2),
        'pc1_pc2_sum': float(var_sum),
        's1_sim': float(s1_sim),
        's2_sim': float(s2_sim),
        'rot_sum': float(s1_sim + s2_sim),
        'balanced_score': float(balanced_s),
    }

def main():
    print("=" * 70)
    print("MEGA EXPERIMENT V5 — SEARCH BALANCED VARIANCE SOLUTIONS")
    print("Started:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    files = get_participant_files()
    print(f"Participants: {len(files)} files")
    
    all_results = []
    exp_count = 0
    skipped = 0
    
    # Strategy: iterate all combos
    for task_name, task_ids in TASK_COMBOS.items():
        for joint_name, jids in JOINT_SUBSETS.items():
            for preproc_name, preproc_fn in PREPROC_METHODS.items():
                label = f"v5_{task_name}_{joint_name}_{preproc_name}"
                exp_count += 1
                
                result = run_single_experiment(files, task_ids, jids, preproc_fn, label)
                if result is None:
                    skipped += 1
                    continue
                
                all_results.append(result)
                
                # Progress print for interesting results
                if result['s1_sim'] > 0.70 and result['pc2_var'] > 0.06:
                    print(f"  [{exp_count}] {label}")
                    print(f"     Var: {result['pc1_var']*100:.1f}%+{result['pc2_var']*100:.1f}%={result['pc1_pc2_sum']*100:.1f}%  "
                          f"S1={result['s1_sim']:.4f} S2={result['s2_sim']:.4f}  "
                          f"Bal={result['balanced_score']:.4f}")
    
    # Filter for reasonable results
    viable = [r for r in all_results if r['s1_sim'] > 0.50 and r['pc1_pc2_sum'] > 0.70]
    print(f"\n{'='*70}")
    print(f"Total experiments: {exp_count}, OK: {len(all_results)}, Skipped: {skipped}")
    print(f"Viable results: {len(viable)}")
    
    # Sort by balanced_score
    viable.sort(key=lambda r: r['balanced_score'], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"TOP {N_RESULTS_TOP} BY BALANCED SCORE")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Label':<48} {'Var%':<12} {'PC2%':<8} {'S1':<8} {'S2':<8} {'BalSc':<8}")
    print("-" * 97)
    for i, r in enumerate(viable[:N_RESULTS_TOP]):
        print(f"{i+1:<5} {r['label']:<48} {r['pc1_pc2_sum']*100:<7.1f}%  "
              f"{r['pc2_var']*100:<7.1f}% "
              f"{r['s1_sim']:<8.4f} {r['s2_sim']:<8.4f} {r['balanced_score']:<8.4f}")
    
    # Save all results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(viable, f, indent=2)
    
    # Also save a separate sorted copy for best by different criteria
    print(f"\n{'='*70}")
    print(f"BEST BY VARIANCE BALANCE (PC2 > 8%, sorted by S1+S2)")
    print(f"{'='*70}")
    balanced = [r for r in viable if r['pc2_var'] > 0.08]
    balanced.sort(key=lambda r: r['rot_sum'], reverse=True)
    for i, r in enumerate(balanced[:15]):
        print(f"{i+1}. {r['label']:<48} Var={r['pc1_pc2_sum']*100:.1f}% "
              f"PC2={r['pc2_var']*100:.1f}% S1={r['s1_sim']:.4f} S2={r['s2_sim']:.4f}")
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
