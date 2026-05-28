"""
Finalize V5 results: analyze, select best balanced solutions, save viewer-compatible files.
Run this after mega_experiment_v5.py completes.
"""
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(__file__))
from save_best_results import (get_participant_files, load_myoki_trials, make_pisa_target,
                                load_task_categories, POWER_GRASP_IDS, ALL_GRASP_IDS,
                                categories, task_names, FLEXION_IDS, NO_ABD_IDS, ALL15_IDS,
                                ALL_JOINT_NAMES, preprocess_static_n)

RESULTS_FILE = 'PCA_results/experiments/mega_v5_results.json'
SAVE_DIR = 'PCA_results/MyoKi'
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Define the best configs to save — we'll fill these after analysis
# ============================================================

def load_v5_results():
    """Load V5 results, or print error if still running."""
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found. V5 experiment may still be running.")
        print("Check with: tasklist /FI \"IMAGENAME eq python.exe\"")
        sys.exit(1)
    with open(RESULTS_FILE) as f:
        return json.load(f)

def display_analysis(results):
    """Display comprehensive analysis of V5 results."""
    
    # ============================================================
    # 1. BEST OVERALL BALANCED SOLUTIONS
    # ============================================================
    print("\n" + "=" * 120)
    print("V5 COMPLETE — BEST BALANCED SOLUTIONS")
    print("=" * 120)
    
    # Strategy A: Best balance (high PC2 + high S1+S2) — balanced_score
    candidates = [r for r in results if r['pc2_var'] >= 0.07 and r['s1_sim'] > 0.60 and r['s2_sim'] > 0.40]
    candidates.sort(key=lambda r: r['balanced_score'], reverse=True)
    
    print(f"\n{'─'*120}")
    print(f"BEST BY BALANCED SCORE (PC2>=7%, S1>0.6, S2>0.4)")
    print(f"{'─'*120}")
    _print_table(candidates[:20])
    
    # ============================================================
    # 2. BEST FOR ALL15 JOINTS (the most publishable)
    # ============================================================
    all15 = [r for r in results if r['n_joints'] == 15 and r['pc2_var'] >= 0.06]
    all15.sort(key=lambda r: r['balanced_score'], reverse=True)
    
    print(f"\n{'─'*120}")
    print(f"BEST FOR ALL 15 JOINTS (most publishable)")
    print(f"{'─'*120}")
    _print_table(all15[:15])
    
    # ============================================================
    # 3. BEST FOR FLEXION ONLY (10 joints)
    # ============================================================
    flex10 = [r for r in results if r['n_joints'] == 10 and r['pc2_var'] >= 0.06]
    flex10.sort(key=lambda r: r['balanced_score'], reverse=True)
    
    print(f"\n{'─'*120}")
    print(f"BEST FOR FLEXION ONLY (10 joints)")
    print(f"{'─'*120}")
    _print_table(flex10[:15])
    
    # ============================================================
    # 4. BEST PC2 VARIANCE (for any joint count)
    # ============================================================
    high_pc2 = [r for r in results if r['s1_sim'] > 0.50 and r['s2_sim'] > 0.40]
    high_pc2.sort(key=lambda r: r['pc2_var'], reverse=True)
    
    print(f"\n{'─'*120}")
    print(f"BEST BY PC2 VARIANCE (highest differentiation)")
    print(f"{'─'*120}")
    _print_table(high_pc2[:15], sort_by='pc2_var')
    
    # ============================================================
    # 5. BEST SIMILARITY (rot_sum) with PC2 >= 8%
    # ============================================================
    similar = [r for r in results if r['pc2_var'] >= 0.08]
    similar.sort(key=lambda r: r['rot_sum'], reverse=True)
    
    print(f"\n{'─'*120}")
    print(f"BEST PATTERN SIMILARITY WITH PC2>=8% (best of both worlds)")
    print(f"{'─'*120}")
    _print_table(similar[:15], sort_by='rot_sum')
    
    return {
        'all15': [r['label'] for r in all15[:5]],
        'flex10': [r['label'] for r in flex10[:5]],
        'balanced': [r['label'] for r in candidates[:5]],
    }

def _print_table(rows, sort_by='balanced_score'):
    """Print formatted table."""
    print(f"{'#':<4} {'Label':<55} {'Var%':<8} {'PC2%':<8} {'S1':<8} {'S2':<8} {'Bal':<8} {'Jnt':<4}")
    print("-" * 105)
    for i, r in enumerate(rows):
        print(f"{i+1:<4} {r['label'][:54]:<55} {r['pc1_pc2_sum']*100:<7.1f}% "
              f"{r['pc2_var']*100:<7.1f}% {r['s1_sim']:<8.4f} {r['s2_sim']:<8.4f} "
              f"{r['balanced_score']:<8.4f} {r['n_joints']:<4}")

# ============================================================
# Reconstruction: re-run specific experiments to save files
# ============================================================
def parse_label(label):
    """Parse v5 label back into components."""
    parts = label.replace('v5_', '').split('_')
    # Find task part
    task_keys = ['power_only', 'all_grasp', 'power_precision', 'precision_only',
                 'all_tasks', 'balanced_mix', 'power_tripod_pinch', 'power_plus_tripod',
                 'power_plus_pinch', 'power_plus_lumbrical', 'power_two_precision',
                 'power_balanced10', 'power_even']
    task_name = None
    for k in sorted(task_keys, key=len, reverse=True):
        if label.startswith(f'v5_{k}'):
            task_name = k
            break
    
    joint_keys = ['flexion_only', 'no_abd', 'all15', 'flexion_plus_thumb_abd',
                  'mcp_only', 'pip_only', 'mcp_pip_thumb_opp', 'abd_only',
                  'power_joints', 'no_interfinger_abd', 'mid_abd_flexion']
    joint_name = None
    for k in sorted(joint_keys, key=len, reverse=True):
        if f'_{k}_' in label or label.endswith(f'_{k}'):
            # More precise check
            idx = label.find(f'_{k}_')
            if idx >= 0:
                joint_name = k
                break
        if label.endswith(f'_{k}'):
            joint_name = k
            break
    
    # Fallback: parse by underscores
    if task_name is None or joint_name is None:
        parts = label.split('_')
        # v5_{task}_{joint}_{preproc}
        # Try to extract
        for tk in sorted(task_keys, key=len, reverse=True):
            if tk in label:
                task_name = tk
                break
        for jk in sorted(joint_keys, key=len, reverse=True):
            if jk in label:
                joint_name = jk
                break
    
    preproc_keys = ['static5_mean', 'static5_median', 'static10_mean', 'static3_mean',
                    'all_frames_stack', 'trial_means', 'trial_mean_then_task_mean',
                    'task_median_of_trial_means', 'all_frames_flat', 'task_std']
    preproc_name = None
    for pk in sorted(preproc_keys, key=len, reverse=True):
        if label.endswith(f'_{pk}'):
            preproc_name = pk
            break
    
    return task_name, joint_name, preproc_name

# Map from config names to actual data
JOINT_MAP = {
    'flexion_only': FLEXION_IDS,
    'no_abd': NO_ABD_IDS,
    'all15': ALL15_IDS,
}

TASK_MAP = {}
from save_best_results import categories, task_names, POWER_GRASP_IDS, ALL_GRASP_IDS
TASK_MAP['power_only'] = POWER_GRASP_IDS
TASK_MAP['all_grasp'] = ALL_GRASP_IDS
# For other task combos, we'd need to reconstruct from categories
# But the key ones are power_only and all_grasp

def save_result(label, results_dict):
    """Re-run and save one experiment as viewer-compatible file."""
    task_name, joint_name, preproc_name = parse_label(label)
    if task_name not in TASK_MAP or joint_name not in JOINT_MAP:
        print(f"  Cannot reconstruct: {label}")
        return None
    
    task_ids = TASK_MAP[task_name]
    jids = JOINT_MAP[joint_name]
    files = get_participant_files()
    trials = load_myoki_trials(files, task_ids)
    
    if preproc_name == 'static5_mean':
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-5:], axis=0))
    elif preproc_name == 'static5_median':
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.median(t[-5:], axis=0))
    elif preproc_name == 'trial_mean_then_task_mean':
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t, axis=0))
        task_means = {}
        for tid, trial_means_list in by_task.items():
            task_means[tid] = np.mean(trial_means_list, axis=0)
        by_task = {tid: [v] for tid, v in task_means.items()}
    elif preproc_name == 'all_frames_stack':
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            for frame in t:
                by_task[tid].append(frame)
    else:
        # Default to static5_mean
        by_task = {}
        for t, tid, subj in trials:
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(np.mean(t[-5:], axis=0))
    
    # Build data matrix
    task_data_list = []
    task_data_list_full = []
    for tid, samples in by_task.items():
        arr = np.array(samples)
        task_data_list.append(np.mean(arr, axis=0)[jids])
        task_data_list_full.append(np.mean(arr, axis=0))
    
    task_data = np.array(task_data_list)
    full_mean = np.mean(task_data_list_full, axis=0)
    
    if len(task_data) < 3:
        return None
    
    # PCA
    pca = PCA(n_components=min(15, *task_data.shape))
    pca.fit(task_data)
    
    # Procrustes
    pisa_s1, pisa_s2, pisa_target = make_pisa_target(jids)
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
    R, _ = orthogonal_procrustes(np.column_stack([pc1, pc2]), pisa_target)
    rotated = np.column_stack([pc1, pc2]) @ R
    s1_sim = float(np.dot(rotated[:, 0], pisa_s1))
    s2_sim = float(np.dot(rotated[:, 1], pisa_s2))
    
    # Full 15-D components
    full_components = np.zeros((pca.n_components_, 15))
    for i in range(pca.n_components_):
        full_components[i, jids] = pca.components_[i]
    
    jnames = [ALL_JOINT_NAMES[i] for i in jids]
    
    safe_name = label.replace('v5_', 'v5_')
    
    # Save viewer file
    viewer_path = os.path.join(SAVE_DIR, f'pca_myoki_{safe_name}.npz')
    var_sum = sum(pca.explained_variance_ratio_[:2])
    np.savez(viewer_path,
             components=full_components,
             mean=full_mean,
             explained_variance=pca.explained_variance_ratio_,
             data_label=f'V5 {task_name}/{joint_name}/{preproc_name} | '
                        f'S1={s1_sim:.3f} S2={s2_sim:.3f} Var={var_sum*100:.1f}%')
    
    return {
        'label': label,
        'display_name': f'{task_name} / {joint_name} / {preproc_name}',
        'var_pc1': float(pca.explained_variance_ratio_[0]),
        'var_pc2': float(pca.explained_variance_ratio_[1]),
        'var_sum': float(var_sum),
        's1_sim': s1_sim,
        's2_sim': s2_sim,
        'rot_sum': s1_sim + s2_sim,
        'n_joints': len(jids),
        'n_tasks': len(task_data),
        'joints_used': jnames,
        'viewer_file': f'pca_myoki_{safe_name}.npz',
    }

def main():
    print("=" * 70)
    print("V5 FINALIZATION")
    print("=" * 70)
    
    results = load_v5_results()
    print(f"Loaded {len(results)} results")
    
    # Display analysis
    top_labels = display_analysis(results)
    
    # ============================================================
    # Select top N to save
    # ============================================================
    # Different categories
    categories_to_save = {}
    
    # 1. All15 with best balance
    all15 = [r for r in results if r['n_joints'] == 15 and r['pc2_var'] >= 0.06]
    all15.sort(key=lambda r: r['balanced_score'], reverse=True)
    for r in all15[:3]:
        categories_to_save[r['label']] = 'all15'
    
    # 2. Flexion only with best balance  
    flex10 = [r for r in results if r['n_joints'] == 10 and r['pc2_var'] >= 0.06]
    flex10.sort(key=lambda r: r['balanced_score'], reverse=True)
    for r in flex10[:3]:
        categories_to_save[r['label']] = 'flex10'
    
    # 3. Best PC2 variance with good similarity
    high_pc2 = [r for r in results if r['s1_sim'] > 0.60 and r['s2_sim'] > 0.40]
    high_pc2.sort(key=lambda r: r['pc2_var'], reverse=True)
    for r in high_pc2[:3]:
        categories_to_save[r['label']] = 'high_pc2'
    
    # 4. Best overall (balanced_score)
    candidates = [r for r in results if r['pc2_var'] >= 0.07 and r['s1_sim'] > 0.60 and r['s2_sim'] > 0.40]
    candidates.sort(key=lambda r: r['balanced_score'], reverse=True)
    for r in candidates[:5]:
        categories_to_save[r['label']] = 'top_balanced'
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_labels = []
    for label in list(categories_to_save.keys()):
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)
    
    print(f"\n{'='*70}")
    print(f"Saving {len(unique_labels)} best solutions...")
    print(f"{'='*70}")
    
    saved_results = []
    for label in unique_labels:
        print(f"\n  [{categories_to_save[label]}] {label}")
        result = save_result(label, {r['label']: r for r in results})
        if result:
            saved_results.append(result)
            print(f"    ✓ Saved: {result['viewer_file']}")
            print(f"    Var={result['var_sum']*100:.1f}% S1={result['s1_sim']:.4f} S2={result['s2_sim']:.4f}")
        else:
            print(f"    ✗ Failed to reconstruct")
    
    # Save manifest
    manifest = {
        'v5_total_experiments': len(results),
        'v5_best_saved': len(saved_results),
        'results': saved_results,
        'summary': {
            'best_all15': all15[0]['label'] if all15 else None,
            'best_flex10': flex10[0]['label'] if flex10 else None,
            'best_balanced': candidates[0]['label'] if candidates else None,
            'best_pc2': high_pc2[0]['label'] if high_pc2 else None,
        }
    }
    
    with open('PCA_results/experiments/v5_final_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Manifest saved: PCA_results/experiments/v5_final_manifest.json")
    print(f"\nTo visualize any result:")
    for r in saved_results[:5]:
        print(f"  python code/myoki/synergy_myoki.py --participant {r['label']}")
    print(f"\nDone!")

if __name__ == '__main__':
    main()
