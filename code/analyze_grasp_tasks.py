"""
Analyze MyoKi PCA by task category (grasp types vs non-grasp).
Also verify joint mapping for the little finger CMC joint.
"""

import os, sys, glob
import numpy as np
import scipy.io as sio
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from myoki.myoki_config import MYOKI_JOINT_NAMES, MYOKI_TO_SHADOW, SHADOW_JOINT_NAMES, DIP_FROM_PIP_MAPPING

PCA_DIR = 'PCA_results/MyoKi'
FIG_DIR = 'MyoKi_figures'
N_JOINTS = 16
TARGET_FRAMES = 100
DATA_DIR = 'MyoKi Dataset/preprocessed'


def load_task_categories():
    """Load task ID → category mapping from Task_categorization.xlsx."""
    wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
    ws = wb.active
    
    categories = {}  # task_id → list of action types
    action_cols = {
        'cylindrical': 11,  # column index (0-based)
        'spherical': 12,
        'hook': 13,
        'tripod': 14,
        'pinch': 15,
        'lumbrical': 16,
        'complex': 17,
        'none_action': 18,
    }
    
    task_names = {}
    
    for row in ws.iter_rows(min_row=4, values_only=False):
        tid = row[0].value
        if tid is None:
            continue
        tid = int(tid)
        task_names[tid] = str(row[1].value).strip() if row[1].value else f"Task {tid}"
        
        actions = []
        # Action columns: J(11) to R(18) in Excel, which are values_only index 11-18
        for i, (action_name, col_idx) in enumerate(action_cols.items()):
            val = row[col_idx].value
            if val is not None and str(val).strip() == 'x':
                actions.append(action_name)
        categories[tid] = actions
    
    return categories, task_names


def is_grasp_task(categories, task_id):
    """Check if a task is primarily a grasping task."""
    if task_id not in categories:
        return False
    actions = categories[task_id]
    grasp_types = {'cylindrical', 'spherical', 'hook', 'tripod', 'pinch', 'lumbrical'}
    # It's a grasp task if it has at least one grasp action type AND is not 'none'
    has_grasp = any(a in grasp_types for a in actions)
    is_none = 'none_action' in actions
    return has_grasp and not is_none


def is_thumb_independent_task(categories, task_id, task_names):
    """Check if a task involves independent thumb use (like typing, writing, etc.)"""
    name = task_names.get(task_id, '').lower()
    thumb_keywords = ['thumb', 'smartphone', 'type', 'write', 'mouse', 'click', 'stir', 'spoon']
    return any(kw in name for kw in thumb_keywords)


def load_and_filter_trials(participant_ids, filter_func, categories, task_names):
    """Load preprocessed data and filter trials by a given criterion."""
    all_means = []
    all_task_ids = []
    
    for pid in participant_ids:
        fpath = os.path.join(DATA_DIR, f'{pid}_preprocessed.mat')
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found")
            continue
        
        mat = sio.loadmat(fpath)
        angles = mat['angles']
        trial_tasks = mat['trial_tasks'].flatten().astype(int)
        num_trials = int(mat['num_trials'].item() if hasattr(mat['num_trials'], 'item') else mat['num_trials'])
        
        n_frames = angles.shape[0]
        expected = num_trials * TARGET_FRAMES
        
        if n_frames == expected:
            trial_data = angles.reshape(num_trials, TARGET_FRAMES, N_JOINTS)
            trial_means = np.mean(trial_data, axis=1)
        else:
            trial_means = []
            trial_task_ids = []
            for ti in range(num_trials):
                s = int(ti * n_frames / num_trials)
                e = int((ti + 1) * n_frames / num_trials)
                trial_means.append(np.mean(angles[s:e], axis=0))
            trial_means = np.array(trial_means)
        
        # Filter
        kept_means = []
        kept_ids = []
        for i, tid in enumerate(trial_tasks):
            tid_int = int(tid)
            if filter_func(categories, tid_int):
                kept_means.append(trial_means[i])
                kept_ids.append(tid_int)
        
        print(f"  {pid}: {len(trial_means)} trials → {len(kept_means)} kept (task filter)")
        all_means.append(np.array(kept_means))
        all_task_ids.extend(kept_ids)
    
    return np.vstack(all_means), all_task_ids


def run_pca_on_data(angles, label, save_dir, fig_dir, task_names_dict=None):
    """Run PCA and save results."""
    print(f"\n{'='*60}")
    print(f"PCA analysis: {label}")
    print(f"{'='*60}")
    print(f"Data shape: {angles.shape}")

    n_comp = min(angles.shape[1], N_JOINTS)
    pca = PCA(n_components=n_comp)
    pca.fit(angles)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"\nExplained variance:")
    for i in range(min(6, n_comp)):
        print(f"  PC{i+1}: {evr[i]:.2%} (cum: {cum[i]:.2%})")
    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    n_95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"  90% → {n_90} PCs,  95% → {n_95} PCs")

    # Save
    safe_label = label.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(save_dir, f'pca_grasp_{safe_label}.npz')
    np.savez(save_path,
             components=pca.components_,
             mean=pca.mean_,
             explained_variance=evr,
             explained_variance_cumsum=cum,
             n_samples=angles.shape[0],
             n_features=angles.shape[1],
             data_label=label)
    print(f"Saved: {save_path}")

    # === Eigenposture plot ===
    joint_names_short = [
        "Thumb\nAbd", "Thumb\nMCP", "Thumb\nIP", "Thumb\nOpp",
        "Idx\nMCP", "Idx\nPIP",
        "Mid\nMCP", "Mid\nPIP", "IdxMid\nAbd",
        "Ring\nMCP", "Ring\nPIP", "MidRing\nAbd",
        "Lit\nMCP", "Lit\nPIP", "RingLit\nAbd", "Palmar\nArch"
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, ax in enumerate(axes):
        comp = pca.components_[i]
        colors = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax.bar(range(N_JOINTS), comp, color=colors, edgecolor='navy', lw=0.5)
        ax.set_title(f'PC{i+1} ({evr[i]:.1%})', fontsize=12)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(joint_names_short, rotation=45, ha='right', fontsize=7)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
    plt.suptitle(f'Eigenpostures — {label}', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'eigenpostures_grasp_{safe_label}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure: {fig_path}")
    
    # Print weights
    print(f"\n  PC1 weights:")
    for j in range(N_JOINTS):
        print(f"    {MYOKI_JOINT_NAMES[j]:25s}: {pca.components_[0][j]:+7.4f}")
    print(f"  PC2 weights:")
    for j in range(N_JOINTS):
        print(f"    {MYOKI_JOINT_NAMES[j]:25s}: {pca.components_[1][j]:+7.4f}")

    return pca


def main():
    categories, task_names = load_task_categories()
    
    # Print task categories for reference
    print("=" * 70)
    print("TASK CATEGORIES FROM EXCEL")
    print("=" * 70)
    for tid in sorted(categories.keys()):
        if tid == 0:
            continue
        actions = categories[tid]
        is_grasp = is_grasp_task(categories, tid)
        grasp_tag = " [GRASP]" if is_grasp else " [OTHER]"
        acts_str = ", ".join(actions)
        print(f"  Task {tid:2d}: {task_names.get(tid, '?'):55s} | {acts_str:30s} {grasp_tag}")
    
    print()
    
    # Get all participants
    all_participants = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07']
    
    # === 1. PCA on GRASP tasks only ===
    print("\n" + "=" * 70)
    print("PCA ON GRASP TASKS ONLY")
    print("=" * 70)
    
    grasp_means, grasp_ids = load_and_filter_trials(
        all_participants, is_grasp_task, categories, task_names)
    
    print(f"\nTotal grasp trials: {len(grasp_means)}")
    print(f"Task IDs: {sorted(set(grasp_ids))}")
    
    grasp_pca = run_pca_on_data(
        grasp_means, 
        f'GraspOnly (N={len(grasp_means)})', 
        PCA_DIR, FIG_DIR, task_names)
    
    # === 2. PCA on NON-GRASP tasks for comparison ===
    print("\n" + "=" * 70)
    print("PCA ON NON-GRASP TASKS")
    print("=" * 70)
    
    def non_grasp_filter(cat, tid):
        return tid != 0 and tid in categories and not is_grasp_task(cat, tid)
    
    other_means, other_ids = load_and_filter_trials(
        all_participants, non_grasp_filter, categories, task_names)
    
    print(f"\nTotal non-grasp trials: {len(other_means)}")
    other_pca = run_pca_on_data(
        other_means,
        f'NonGrasp (N={len(other_means)})',
        PCA_DIR, FIG_DIR, task_names)
    
    # === 3. Comparison figure ===
    print("\n" + "=" * 70)
    print("COMPARISON FIGURE")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    joint_names_short = [
        "Thumb\nAbd", "Thumb\nMCP", "Thumb\nIP", "Thumb\nOpp",
        "Idx\nMCP", "Idx\nPIP",
        "Mid\nMCP", "Mid\nPIP", "IdxMid\nAbd",
        "Ring\nMCP", "Ring\nPIP", "MidRing\nAbd",
        "Lit\nMCP", "Lit\nPIP", "RingLit\nAbd", "Palmar\nArch"
    ]
    x = np.arange(N_JOINTS)
    
    datasets = [
        (grasp_pca, 'Grasp Tasks', 0),
        (other_pca, 'Non-Grasp Tasks', 1),
    ]
    
    # Also load all_trialmeans for comparison
    all_data = np.load(os.path.join(PCA_DIR, 'pca_myoki_all_trialmeans.npz'))
    
    for col, (pca_obj, label, _) in enumerate(datasets):
        # PC1
        ax = axes[0, col]
        comp = pca_obj.components_[0]
        ev = pca_obj.explained_variance_ratio_[0]
        colors_comp = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax.bar(x, comp, color=colors_comp, edgecolor='navy', lw=0.5)
        ax.set_title(f'{label} — PC1 ({ev:.1%})', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(joint_names_short, rotation=45, ha='right', fontsize=7)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.2)
        
        # PC2
        ax = axes[1, col]
        comp = pca_obj.components_[1]
        ev = pca_obj.explained_variance_ratio_[1]
        colors_comp = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax.bar(x, comp, color=colors_comp, edgecolor='navy', lw=0.5)
        ax.set_title(f'{label} — PC2 ({ev:.1%})', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(joint_names_short, rotation=45, ha='right', fontsize=7)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.2)
    
    # Add variance comparison table as text
    text_str = "Variance Comparison:\n"
    text_str += f"{'':>15s} {'PC1':>8s} {'PC2':>8s} {'PC1+PC2':>10s}\n"
    text_str += '-' * 45 + '\n'
    for pca_obj, label, _ in datasets:
        ev = pca_obj.explained_variance_ratio_
        text_str += f"{label:>15s} {ev[0]:>7.1%} {ev[1]:>7.1%} {ev[0]+ev[1]:>8.1%}\n"
    all_ev = all_data['explained_variance']
    text_str += f"{'All Tasks':>15s} {all_ev[0]:>7.1%} {all_ev[1]:>7.1%} {all_ev[0]+all_ev[1]:>8.1%}\n"
    
    # Remove the empty subplot
    fig.delaxes(axes[1,1])
    # Add text in the empty space
    text_ax = fig.add_subplot(2, 2, 4)
    text_ax.text(0.1, 0.5, text_str, transform=text_ax.transAxes,
                 fontsize=12, fontfamily='monospace', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
    text_ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'pca_grasp_vs_nongrasp.png'), dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR}/pca_grasp_vs_nongrasp.png")
    
    # === 4. Joint mapping analysis ===
    print("\n" + "=" * 70)
    print("JOINT MAPPING ANALYSIS: Little Finger CMC")
    print("=" * 70)
    
    print("\nShadow Hand LFJ5 (Little CMC) definition:")
    print("  Joint name: rh_LFJ5")
    print("  Axis: (0.573576, 0, 0.819152) → rotation about diagonal axis")
    print("  Range: [0, 0.785398] rad = [0°, 45°]")
    print("  Class: metacarpal → 'Let the little finger oppose/pivot towards palm'")
    
    print("\nMyoKi mapping:")
    myo_idx = [k for k, v in MYOKI_TO_SHADOW.items() if v == 14]
    print(f"  MyoKi sensor(s) mapping to SH_LFJ5: {myo_idx}")
    for idx in myo_idx:
        print(f"    MyoKi#{idx}: {MYOKI_JOINT_NAMES[idx]}")
    
    print("\nThe Palmar Arch (#16 in CyberGlove) measures the TRANSVERSE ARCH")
    print("of the palm (the cupping motion). It's NOT a direct measurement")
    print("of the little finger CMC joint. The CyberGlove has NO dedicated")
    print("sensor for little finger opposition/anteversion.")
    print()
    print("Human hand anatomy: When you cup your palm (palmar arch),")
    print("the little finger CMC also flexes and rotates medially.")
    print("So Palmar Arch angle ≈ correlated with little finger CMC motion,")
    print("but they are NOT the same thing.")
    print()
    print("Santello 1998 used a 15-DOF model where:")
    print("  - Little finger CMC was NOT modeled separately")
    print("  - The palmar arch was treated separately")
    print("  - Little finger MCP and PIP were the only joints used")
    print()
    print("RECOMMENDATION: Consider whether we should:")
    print("  A) Keep Palmar Arch → LFJ5 as is (simple, works for gross motion)")
    print("  B) Don't drive LFJ5 at all (set to 0, since CyberGlove lacks direct measurement)")
    print("  C) Derive little finger opposition from a combination of palmar arch")
    print("     and little finger abduction angle")


if __name__ == '__main__':
    main()
