"""
Experiment: Search for optimal preprocessing → PCA matching Pisa/IIT synergies.

Tests combinations of:
  - Task filtering: grasp-only vs all-tasks vs power-grasps vs no filter
  - Data representation: trial-means vs all-frames vs static-hold vs full-trial
  - Joint space: 15D vs 16D vs 14D
  - Velocity filtering: with vs without

Output: Comprehensive comparison table + best params
"""
import os, sys, glob, json, time
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = 'PCA_results/experiments'
FIG_DIR = 'MyoKi_figures/experiments'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_JOINTS_FULL = 15
JOINT_NAMES = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
    "RingMCP","RingPIP","MidRingAbd",
    "LitMCP","LitPIP","RingLitAbd"
]

# Pisa/IIT Targets
PISA_SYN1 = np.ones(N_JOINTS_FULL) / np.linalg.norm(np.ones(N_JOINTS_FULL))
PISA_SYN2_RAW = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_SYN2 = PISA_SYN2_RAW / np.linalg.norm(PISA_SYN2_RAW)

# =============================
# 1. Load task categories
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

# Define task subsets
PICKUP_GRASP_IDS = {1, 3, 5, 9, 11, 13, 16, 18, 20, 28, 30, 32, 34, 36, 38, 40, 42, 44}
ALL_GRASP_IDS = set()
POWER_GRASP_IDS = set()
for tid, acts in categories.items():
    has_grasp = any(a in grasp_types for a in acts)
    is_none = 'none_action' in acts
    is_power = any(a in power_types for a in acts)
    if has_grasp and not is_none:
        ALL_GRASP_IDS.add(tid)
        if is_power:
            POWER_GRASP_IDS.add(tid)

# "Put back" tasks (the ones NOT in PICKUP_GRASP_IDS but in ALL_GRASP_IDS)
PUTBACK_GRASP_IDS = ALL_GRASP_IDS - PICKUP_GRASP_IDS

# "Precision" grasp tasks (tripod, pinch, lumbrical but not power)
PRECISION_GRASP_IDS = set()
for tid in ALL_GRASP_IDS:
    acts = categories.get(tid, [])
    is_power = any(a in power_types for a in acts)
    has_precision = any(a in {'tripod', 'pinch', 'lumbrical'} for a in acts)
    if has_precision and not is_power:
        PRECISION_GRASP_IDS.add(tid)

print("=" * 70)
print("TASK SUBSETS DEFINED")
print("=" * 70)
print(f"  All tasks (1-74):                 {len(categories)} tasks")
print(f"  All grasp tasks:                  {len(ALL_GRASP_IDS)} tasks: {sorted(ALL_GRASP_IDS)}")
print(f"  Pickup-only grasp tasks:          {len(PICKUP_GRASP_IDS)} tasks: {sorted(PICKUP_GRASP_IDS)}")
print(f"  Put-back grasp tasks:             {len(PUTBACK_GRASP_IDS)} tasks: {sorted(PUTBACK_GRASP_IDS)}")
print(f"  Power grasps (cyl+spherical+hook): {len(POWER_GRASP_IDS)} tasks: {sorted(POWER_GRASP_IDS)}")
print(f"  Precision grasps (tripod+pinch):   {len(PRECISION_GRASP_IDS)} tasks: {sorted(PRECISION_GRASP_IDS)}")

# Also define from the task descriptions which are "static hold" vs "dynamic"
# Tasks 46-53: Independent finger actions (pointing, etc.)
# Tasks 54-59: Wrist actions (hammer, saw - but wrist is removed so these are just orientation changes)
# Tasks 60-67: Fine manipulation (write, type, etc.)
# Tasks 68-74: Functional (squeeze, twist, etc.)
STATIC_GRASP_IDS = set()  # Grasp tasks involving static holding
DYNAMIC_GRASP_IDS = set() # Grasp tasks with dynamic manipulation
for tid in ALL_GRASP_IDS:
    name = task_names.get(tid, '').lower()
    # Tasks that are "put back" or "pickup" of objects are more static holding
    # While tasks like "squeeze", "twist", "tear" are dynamic even though they're grasps
    if any(w in name for w in ['squeeze', 'twist', 'tear', 'pull', 'push', 'rotate', 'roll', 'crumple', 'peel', 'cut', 'write', 'type', 'draw', 'screw', 'wipe', 'hammer', 'saw']):
        DYNAMIC_GRASP_IDS.add(tid)
    else:
        STATIC_GRASP_IDS.add(tid)

print(f"  Static grasp tasks:              {len(STATIC_GRASP_IDS)} tasks")
print(f"  Dynamic grasp tasks:             {len(DYNAMIC_GRASP_IDS)} tasks")

# =============================
# 2. Load raw data for experiments
# =============================
def load_raw_participant(mat_path):
    """Load raw MyoKi data, return (angles_15d, tasks_array)"""
    mat = sio.loadmat(mat_path)
    glove = np.nan_to_num(mat['glove_calibrated'])  # (N, 18)
    tasks = mat['task'].flatten().astype(int)
    # Strip palmar arch (ch15) + wrist (16,17): keep 15 hand-only channels
    hand_data = glove[:, :15]
    return hand_data, tasks

def extract_task_data(hand_data, tasks, task_ids, exclude_rest=True):
    """
    Extract frames belonging to specific task_ids.
    Returns list of (trial_data, task_id) tuples.
    """
    trials = []
    if exclude_rest:
        task_ids = [t for t in task_ids if t != 0]
    for tid in task_ids:
        task_mask = (tasks == tid)
        diff = np.diff(task_mask.astype(int), prepend=0, append=0)
        starts = np.where(diff > 0)[0]
        ends = np.where(diff < 0)[0]
        for s, e in zip(starts, ends):
            trial = hand_data[s:e, :]
            if len(trial) >= 5:
                trials.append((trial, tid))
    return trials

def resample_trial(trial, target_frames=100):
    """Resample trial to uniform length."""
    if len(trial) == target_frames:
        return trial
    x_old = np.linspace(0, 1, len(trial))
    x_new = np.linspace(0, 1, target_frames)
    f = interp1d(x_old, trial, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(x_new)

def remove_static_frames(trial, vel_thresh=0.5):
    """Remove frames with low velocity."""
    if len(trial) < 5:
        return trial
    grads = np.gradient(trial, axis=0)
    vel_norm = np.linalg.norm(grads, axis=1)
    moving = trial[vel_norm > vel_thresh]
    if len(moving) < 10:
        return trial
    return moving

# =============================
# 3. Experiment functions
# =============================
def run_trial_mean_pca(trials, label=""):
    """Average each trial → run PCA."""
    all_means = np.array([np.mean(t, axis=0) for t in trials])
    return run_pca_and_evaluate(all_means, label)

def run_all_frames_pca(trials, label=""):
    """Concatenate all frames → run PCA."""
    all_frames = np.vstack(trials)
    return run_pca_and_evaluate(all_frames, label)

def run_static_last_n_pca(trials, frac=0.25, label=""):
    """Use last N% frames of each trial."""
    static = []
    for t in trials:
        n_last = max(5, int(len(t) * frac))
        static.append(np.mean(t[-n_last:], axis=0))
    return run_pca_and_evaluate(np.array(static), label)

def run_static_velocity_filter_pca(trials, label=""):
    """Use frames with velocity < thresh (static hold phase)."""
    all_static = []
    for t in trials:
        if len(t) < 10:
            all_static.append(np.mean(t, axis=0))
            continue
        grads = np.gradient(t, axis=0)
        vel = np.linalg.norm(grads, axis=1)
        # Keep frames with low velocity
        static_mask = vel < np.percentile(vel, 30)  # bottom 30%
        if static_mask.sum() > 5:
            all_static.append(np.mean(t[static_mask], axis=0))
        else:
            all_static.append(np.mean(t, axis=0))
    return run_pca_and_evaluate(np.array(all_static), label)

def run_pca_and_evaluate(data, label=""):
    """Run PCA, compute metrics, return dict with results."""
    if len(data) < 5:
        return None
    if np.isnan(data).any() or np.isinf(data).any():
        return None
    
    pca = PCA(n_components=min(15, data.shape[0], data.shape[1]))
    pca.fit(data)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    
    # Metrics
    pc1_var = evr[0]
    pc2_var = evr[1]
    pc1_pc2_sum = pc1_var + pc2_var
    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    
    # PC1 shape metric: how close to uniform?
    pc1_vec = pca.components_[0]
    pc2_vec = pca.components_[1]
    pc1_uniform_sim = np.dot(pc1_vec, PISA_SYN1) / (np.linalg.norm(pc1_vec) * np.linalg.norm(PISA_SYN1))
    
    # Procrustes rotation → Pisa targets
    from scipy.linalg import orthogonal_procrustes
    PC = np.column_stack([pc1_vec, pc2_vec])  # (15, 2)
    R, _ = orthogonal_procrustes(PC, np.column_stack([PISA_SYN1, PISA_SYN2]))
    rotated = PC @ R
    
    # Cosine similarity to targets
    s1_sim = np.dot(rotated[:, 0], PISA_SYN1) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(PISA_SYN1))
    s2_sim = np.dot(rotated[:, 1], PISA_SYN2) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(PISA_SYN2))
    
    # Rotated variance explained
    scores = pca.transform(data)[:, :2] @ R.T
    rotated_var = np.var(scores, axis=0)
    rv_ratio = rotated_var / rotated_var.sum()
    rv_scaled = rv_ratio * pc1_pc2_sum
    
    result = {
        'label': label,
        'n_samples': data.shape[0],
        'pc1_var': pc1_var,
        'pc2_var': pc2_var,
        'pc1_pc2_sum': pc1_pc2_sum,
        'n_90': n_90,
        'pc1_uniform_sim': pc1_uniform_sim,
        'rot_pc1_s1_sim': s1_sim,
        'rot_pc2_s2_sim': s2_sim,
        'rot_pc1_var': rv_scaled[0],
        'rot_pc2_var': rv_scaled[1],
        'pc1': pc1_vec.tolist(),
        'pc2': pc2_vec.tolist(),
        'rotated_s1': rotated[:, 0].tolist(),
        'rotated_s2': rotated[:, 1].tolist(),
    }
    
    return result

# =============================
# 4. Run experiments
# =============================
def generate_experiments():
    """Define all experiment configurations."""
    experiments = []
    
    # Experiment dimensions:
    # A. Task sets
    task_sets = {
        'pickup_grasp': PICKUP_GRASP_IDS,
        'all_grasp': ALL_GRASP_IDS,
        'power_grasp': POWER_GRASP_IDS,
        'static_grasp': STATIC_GRASP_IDS,
    }
    
    # B. Data representations  
    reps = ['trial_mean', 'all_frames', 'static_last25pct', 'static_last50pct', 'static_vel_filter']
    
    # C. Joint space variations
    joint_variants = ['15d_no_palmar']
    
    for task_name, task_ids in task_sets.items():
        for rep in reps:
            for joint in joint_variants:
                exp_label = f"{task_name}_{rep}_{joint}"
                experiments.append({
                    'task_name': task_name,
                    'task_ids': task_ids,
                    'rep': rep,
                    'joint': joint,
                    'label': exp_label
                })
    
    return experiments

def collect_data(participant_files):
    """Collect all trials from all participants, organized by task."""
    all_trials = []  # List of (trial_data, task_id)
    for fpath in participant_files:
        basename = os.path.basename(fpath).replace('.mat', '')
        print(f"  Loading {basename}...", end=" ")
        hand_data, tasks = load_raw_participant(fpath)
        trials = extract_task_data(hand_data, tasks, ALL_GRASP_IDS)
        # Resample all to uniform length
        for t, tid in trials:
            t_res = resample_trial(t)
            all_trials.append((t_res, tid))
        print(f"{len(trials)} trials")
    return all_trials

def filter_trials_by_task(all_trials, task_ids):
    return [(t, tid) for t, tid in all_trials if tid in task_ids]

def run_experiments(participant_files):
    print("=" * 70)
    print("COLLECTING TRIALS FROM ALL PARTICIPANTS")
    print("=" * 70)
    all_trials = collect_data(participant_files)
    print(f"\nTotal trials collected: {len(all_trials)}")
    
    experiments = generate_experiments()
    
    print(f"\n{'='*70}")
    print(f"RUNNING {len(experiments)} EXPERIMENTS")
    print(f"{'='*70}")
    
    results = []
    
    for exp in experiments:
        label = exp['label']
        print(f"\n--- Experiment: {label} ---")
        
        # Filter trials
        trials_with_tasks = filter_trials_by_task(all_trials, exp['task_ids'])
        trials = [t for t, tid in trials_with_tasks]
        
        if len(trials) < 5:
            print(f"  SKIP: only {len(trials)} trials")
            continue
        
        print(f"  Trials: {len(trials)}")
        
        # Apply data representation
        if exp['rep'] == 'trial_mean':
            result = run_trial_mean_pca(trials, label)
        elif exp['rep'] == 'all_frames':
            result = run_all_frames_pca(trials, label)
        elif exp['rep'] == 'static_last25pct':
            result = run_static_last_n_pca(trials, 0.25, label)
        elif exp['rep'] == 'static_last50pct':
            result = run_static_last_n_pca(trials, 0.50, label)
        elif exp['rep'] == 'static_vel_filter':
            result = run_static_velocity_filter_pca(trials, label)
        else:
            continue
        
        if result is None:
            print(f"  SKIP: PCA failed")
            continue
        
        print(f"  PC1: {result['pc1_var']:.2%}, PC2: {result['pc2_var']:.2%}, "
              f"Sum: {result['pc1_pc2_sum']:.2%}")
        print(f"  PC1 uniform sim: {result['pc1_uniform_sim']:.4f}")
        print(f"  Rot PC1→S1: {result['rot_pc1_s1_sim']:.4f}, "
              f"Rot PC2→S2: {result['rot_pc2_s2_sim']:.4f}")
        
        results.append(result)
    
    return results

# =============================
# 5. Print comparison table
# =============================
def print_results_table(results):
    print("\n" + "=" * 150)
    print("EXPERIMENT RESULTS - COMPARISON TABLE")
    print("=" * 150)
    
    header = f"{'Experiment':<40} {'N':>7} {'PC1':>8} {'PC2':>8} {'Sum':>8} {'Uni':>7} {'R1→S1':>8} {'R2→S2':>8}"
    print(header)
    print("-" * 150)
    
    # Sort by best combination of R1→S1 + R2→S2 similarity
    def score(r):
        return r['rot_pc1_s1_sim'] + r['rot_pc2_s2_sim']
    results.sort(key=score, reverse=True)
    
    for i, r in enumerate(results):
        line = (f"{r['label']:<40} {r['n_samples']:>7} "
                f"{r['pc1_var']*100:>7.1f}% "
                f"{r['pc2_var']*100:>7.1f}% "
                f"{r['pc1_pc2_sum']*100:>7.1f}% "
                f"{r['pc1_uniform_sim']:>+7.4f} "
                f"{r['rot_pc1_s1_sim']:>+7.4f} "
                f"{r['rot_pc2_s2_sim']:>+7.4f}")
        if i == 0:
            print(f"★ {line}")
        else:
            print(f"  {line}")
    
    # Save results
    save_path = os.path.join(SAVE_DIR, 'experiment_results.json')
    # Convert numpy types
    clean_results = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.ndarray,)):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        clean_results.append(clean)
    with open(save_path, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")
    
    return results

# =============================
# 6. Plot best results
# =============================
def plot_best_results(results, top_n=5):
    """Plot PC1/PC2 bar charts for top results."""
    def score(r):
        return r['rot_pc1_s1_sim'] + r['rot_pc2_s2_sim']
    results.sort(key=score, reverse=True)
    
    fig, axes = plt.subplots(top_n, 4, figsize=(24, 4*top_n))
    if top_n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(top_n, len(results))):
        r = results[i]
        ax1 = axes[i, 0]  # Original PC1
        comp = r['pc1']
        colors = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax1.bar(range(N_JOINTS_FULL), comp, color=colors, edgecolor='navy', lw=0.5)
        ax1.set_title(f"{r['label']} - PC1 ({r['pc1_var']*100:.1f}%)")
        ax1.set_xticks(range(N_JOINTS_FULL)); ax1.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=6)
        ax1.axhline(0, color='black', lw=0.5); ax1.grid(axis='y', alpha=.3)
        ax1.set_ylim(-0.6, 0.6)
        
        ax2 = axes[i, 1]  # Original PC2
        comp2 = r['pc2']
        colors2 = ['steelblue' if c >= 0 else 'coral' for c in comp2]
        ax2.bar(range(N_JOINTS_FULL), comp2, color=colors2, edgecolor='navy', lw=0.5)
        ax2.set_title(f"{r['label']} - PC2 ({r['pc2_var']*100:.1f}%)")
        ax2.set_xticks(range(N_JOINTS_FULL)); ax2.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=6)
        ax2.axhline(0, color='black', lw=0.5); ax2.grid(axis='y', alpha=.3)
        ax2.set_ylim(-0.6, 0.6)
        
        ax3 = axes[i, 2]  # Rotated S1
        s1 = r['rotated_s1']
        c3 = ['steelblue' if s >= 0 else 'coral' for s in s1]
        ax3.bar(range(N_JOINTS_FULL), s1, color=c3, edgecolor='darkgreen', lw=0.5)
        ax3.set_title(f"Rotated S1 (sim={r['rot_pc1_s1_sim']:.3f})")
        ax3.set_xticks(range(N_JOINTS_FULL)); ax3.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=6)
        ax3.axhline(0, color='black', lw=0.5); ax3.grid(axis='y', alpha=.3)
        ax3.set_ylim(-0.6, 0.6)
        
        ax4 = axes[i, 2]  # Rotated S2
        s2 = r['rotated_s2']
        c4 = ['steelblue' if s >= 0 else 'coral' for s in s2]
        ax4.bar(range(N_JOINTS_FULL), s2, color=c4, edgecolor='darkgreen', lw=0.5)
        ax4.set_title(f"Rotated S2 (sim={r['rot_pc2_s2_sim']:.3f})")
        ax4.set_xticks(range(N_JOINTS_FULL)); ax4.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=6)
        ax4.axhline(0, color='black', lw=0.5); ax4.grid(axis='y', alpha=.3)
        ax4.set_ylim(-0.6, 0.6)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'best_experiments.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved: {fig_path}")

# =============================
# 7. Also test: Ninapro tasks E2+E3
# =============================
def check_ninapro_preprocessing():
    """Check if we can also test Ninapro data."""
    # Ninapro E2 has grasping tasks, E3 has functional tasks
    # Both are relevant
    print(f"\n{'='*70}")
    print(f"CHECKING NINAPRO DATA AVAILABILITY")
    print(f"{'='*70}")
    
    for ex in ['E1', 'E2', 'E3']:
        files = sorted(glob.glob(f'Ninapro Data/{ex}/S*_B.mat'))
        files2 = sorted(glob.glob(f'Ninapro Data/preprocessed/{ex}/*.npy'))
        print(f"  {ex}: {len(files)} preprocessed .mat files, {len(files2)} .npy files")

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    # Load participants
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    # Use all participants that have preprocessed data (P01-P09, P11-P12)
    valid_participants = [f'P{p:02d}' for p in [1,2,3,4,5,6,7,8,9,11,12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_participants)]
    print(f"Using {len(files)} participants")
    
    results = run_experiments(files)
    sorted_results = print_results_table(results)
    plot_best_results(sorted_results, top_n=10)
    
    check_ninapro_preprocessing()
    
    print("\nDone! Check PCA_results/experiments/ for detailed results.")
