"""
Task-Level Range-Based PCA on MyoKi Dataset (expanded to 23 participants).

Methodology (matching experiment_report_allgrasp_range_all15):
1. Load all 23 participants' preprocessed data (P01-P15, P17-P24)
2. Extract only grasping tasks (41 tasks total)
3. For each trial, compute angular range (max-min) per joint
4. Average trials across all participants for each task → 41x15 matrix
5. Covariance-based PCA (no normalization, preserving natural scale)
6. Save model and generate figures

Usage:
    python code/myoki/pca_task_range_myoki.py
"""

import os, sys, glob
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SAVE_DIR = 'PCA_results/MyoKi'
FIG_DIR  = 'MyoKi_figures'
DATA_DIR = 'MyoKi Dataset/preprocessed'
N_JOINTS = 15
TARGET_FRAMES = 100

# Grasp-only task IDs (same 41 tasks as the original report)
GRASP_TASK_IDS = sorted([
    1, 2, 3, 4, 5, 6,      # Cylindrical: cup
    9, 10,                    # Cylindrical: empty bottle
    11, 12,                   # Cylindrical: half-full bottle
    13, 14,                   # Cylindrical: full bottle
    15,                       # Cylindrical: pour water
    16, 17, 18, 19, 20, 21,  # Tripod: tea cup
    28, 29,                   # Spherical: ball d=65mm
    30, 31,                   # Spherical: ball d=85mm
    32, 33,                   # Spherical: ball d=100mm
    40, 41,                   # Hook: laptop bag 0.75kg
    42, 43,                   # Hook: laptop bag 1.5kg
    44, 45,                   # Hook: laptop bag 3.0kg
    46, 47, 48,              # Complex: wipe dust
    49, 50, 51,              # Complex: hammer
    54, 55,                   # Complex: screw
    56,                       # Complex: saw
    61,                       # Pinch: write with pen
])

GRASP_CATEGORIES = {
    1: 'Cylindrical', 2: 'Cylindrical', 3: 'Cylindrical', 4: 'Cylindrical',
    5: 'Cylindrical', 6: 'Cylindrical', 9: 'Cylindrical', 10: 'Cylindrical',
    11: 'Cylindrical', 12: 'Cylindrical', 13: 'Cylindrical', 14: 'Cylindrical',
    15: 'Cylindrical',
    16: 'Tripod', 17: 'Tripod', 18: 'Tripod', 19: 'Tripod',
    20: 'Tripod', 21: 'Tripod',
    28: 'Spherical', 29: 'Spherical', 30: 'Spherical', 31: 'Spherical',
    32: 'Spherical', 33: 'Spherical',
    40: 'Hook', 41: 'Hook', 42: 'Hook', 43: 'Hook', 44: 'Hook', 45: 'Hook',
    46: 'Complex', 47: 'Complex', 48: 'Complex',
    49: 'Complex', 50: 'Complex', 51: 'Complex',
    54: 'Complex', 55: 'Complex', 56: 'Complex',
    61: 'Pinch',
}

JOINT_NAMES = [
    "ThumbAbd", "ThumbMCP", "ThumbIP", "ThumbOpp",
    "IdxMCP", "IdxPIP",
    "MidMCP", "MidPIP", "IdxMidAbd",
    "RingMCP", "RingPIP", "MidRingAbd",
    "LitMCP", "LitPIP", "RingLitAbd",
]

JOINT_NAMES_SHORT = [
    "Thumb\nAbd", "Thumb\nMCP", "Thumb\nIP", "Thumb\nOpp",
    "Idx\nMCP", "Idx\nPIP",
    "Mid\nMCP", "Mid\nPIP", "IdxMid\nAbd",
    "Ring\nMCP", "Ring\nPIP", "MidRing\nAbd",
    "Lit\nMCP", "Lit\nPIP", "RingLit\nAbd",
]

# Participants: P01-P15 and P17-P24 (exclude P16 left-handed)
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 16)] + [f'P{i:02d}' for i in range(17, 25)]


def load_and_compute_task_ranges(participant_ids, grasp_task_ids, data_dir, target_frames):
    """For each participant, compute trial-level ranges, then average per task."""
    # task_id → list of trial range vectors
    task_trial_ranges = {tid: [] for tid in grasp_task_ids}
    task_trial_counts = {tid: 0 for tid in grasp_task_ids}
    total_trials_loaded = 0

    for pid in participant_ids:
        fpath = os.path.join(data_dir, f'{pid}_preprocessed.mat')
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue

        mat = sio.loadmat(fpath)
        angles = mat['angles']                          # (N, 15)
        trial_tasks = mat['trial_tasks'].flatten().astype(int)
        num_trials = int(mat['num_trials'].item() if hasattr(mat['num_trials'], 'item') else mat['num_trials'])

        # Reshape into trials
        n_frames = angles.shape[0]
        expected = num_trials * target_frames

        if n_frames != expected:
            # Irregular shape: split by uniform segments
            trial_data_list = []
            for ti in range(num_trials):
                s = int(ti * n_frames / num_trials)
                e = int((ti + 1) * n_frames / num_trials)
                trial_data_list.append(angles[s:e])
            trial_data = trial_data_list
        else:
            trial_data = angles.reshape(num_trials, target_frames, N_JOINTS)

        # Compute range per trial and accumulate
        for ti in range(num_trials):
            tid = int(trial_tasks[ti]) if ti < len(trial_tasks) else 0
            if tid not in grasp_task_ids:
                continue
            trial_angles = trial_data[ti] if isinstance(trial_data, np.ndarray) else trial_data[ti]
            trial_range = trial_angles.max(axis=0) - trial_angles.min(axis=0)
            task_trial_ranges[tid].append(trial_range)
            task_trial_counts[tid] += 1
            total_trials_loaded += 1

    # Average per task
    X = np.zeros((len(grasp_task_ids), N_JOINTS))
    for i, tid in enumerate(grasp_task_ids):
        if task_trial_ranges[tid]:
            X[i] = np.mean(task_trial_ranges[tid], axis=0)
        else:
            print(f"  WARNING: No trials for task {tid}")
            X[i] = np.zeros(N_JOINTS)

    return X, task_trial_counts, total_trials_loaded


def run_pca_on_task_matrix(X, label, save_dir, fig_dir):
    """Run PCA on the task-level range matrix and generate figures."""
    print(f"\n{'='*60}")
    print(f"PCA analysis: {label}")
    print(f"Data matrix: {X.shape} ({len(GRASP_TASK_IDS)} tasks × {N_JOINTS} joints)")
    print(f"{'='*60}")

    # Covariance-based PCA (all variables in same physical units: degrees)
    n_comp = min(X.shape[1], N_JOINTS)
    pca = PCA(n_components=n_comp)
    pca.fit(X)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"\nVariance explained:")
    for i in range(min(10, n_comp)):
        print(f"  PC{i+1}: {evr[i]:.2%} (cum: {cum[i]:.2%})")

    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    n_95 = int(np.searchsorted(cum, 0.95) + 1)
    n_99 = int(np.searchsorted(cum, 0.99) + 1)
    print(f"  90% → {n_90} PCs,  95% → {n_95} PCs,  99% → {n_99} PCs")

    # Save PCA model
    safe_label = label.replace(' ', '_')
    save_path = os.path.join(save_dir, f'pca_myoki_{safe_label}.npz')
    np.savez(save_path,
             components=pca.components_,
             mean=pca.mean_,
             explained_variance=evr,
             explained_variance_cumsum=cum,
             n_tasks=X.shape[0],
             n_joints=X.shape[1],
             data_label=label)
    print(f"Saved: {save_path}")

    # --- Print PC weights ---
    print(f"\n  PC1 weights ({evr[0]:.1%}):")
    for j in range(N_JOINTS):
        print(f"    {JOINT_NAMES[j]:12s}: {pca.components_[0][j]:+7.4f}")
    print(f"  PC2 weights ({evr[1]:.1%}):")
    for j in range(N_JOINTS):
        print(f"    {JOINT_NAMES[j]:12s}: {pca.components_[1][j]:+7.4f}")

    # --- Variance plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = axes[0]
    ax1.bar(range(1, n_comp+1), evr, color='steelblue', edgecolor='navy')
    ax1.set_xlabel('Principal Component'), ax1.set_ylabel('Variance Explained')
    ax1.set_title(f'Individual PC Variance — {label}')
    ax1.set_xticks(range(1, n_comp+1))
    ax1.grid(axis='y', alpha=0.3)
    for i in range(min(5, n_comp)):
        ax1.text(i+1, evr[i]+0.003, f'{evr[i]:.1%}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    ax2.plot(range(1, n_comp+1), cum, 'o-', color='darkorange', lw=2, markersize=8)
    ax2.axhline(0.9, color='gray', ls='--', alpha=.7, label='90%')
    ax2.axhline(0.95, color='gray', ls=':', alpha=.7, label='95%')
    ax2.axhline(0.99, color='gray', ls='-.', alpha=.7, label='99%')
    ax2.set_xlabel('Number of PCs'), ax2.set_ylabel('Cumulative Variance')
    ax2.set_title(f'Cumulative Variance — {label}')
    ax2.set_xticks(range(1, n_comp+1))
    ax2.grid(alpha=.3), ax2.legend()
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'variance_{safe_label}.png')
    plt.savefig(fig_path, dpi=150), plt.close()
    print(f"Figure: {fig_path}")

    # --- Eigenposture plot (PC1, PC2, PC3) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, ax in enumerate(axes):
        comp = pca.components_[i]
        colors = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax.bar(range(N_JOINTS), comp, color=colors, edgecolor='navy', lw=0.5)
        ax.set_title(f'PC{i+1} ({evr[i]:.1%})', fontsize=12)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES_SHORT, rotation=45, ha='right', fontsize=7)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
        # Add zero line indicator
        ax.set_ylim(-0.6, 0.6)
    plt.suptitle(f'Eigenpostures — {label} (Task-Level Range PCA)', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'eigenpostures_{safe_label}.png')
    plt.savefig(fig_path, dpi=150), plt.close()
    print(f"Figure: {fig_path}")

    # --- PCA scatter plot (PC1 vs PC2) with categories ---
    fig, ax = plt.subplots(figsize=(10, 8))
    scores = pca.transform(X)
    category_colors = {
        'Cylindrical': '#1f77b4',
        'Spherical': '#ff7f0e',
        'Hook': '#2ca02c',
        'Tripod': '#d62728',
        'Pinch': '#9467bd',
        'Complex': '#8c564b',
    }
    categories_display = {
        'Cylindrical': 'Cylindrical',
        'Spherical': 'Spherical',
        'Hook': 'Hook',
        'Tripod': 'Tripod (3-finger)',
        'Pinch': 'Pinch',
        'Complex': 'Complex',
    }

    for cat_name, color in category_colors.items():
        task_indices = [i for i, tid in enumerate(GRASP_TASK_IDS)
                        if GRASP_CATEGORIES[tid] == cat_name]
        if task_indices:
            ax.scatter(scores[task_indices, 0], scores[task_indices, 1],
                       c=color, label=categories_display.get(cat_name, cat_name),
                       s=60, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Label each point with task ID
    for i, tid in enumerate(GRASP_TASK_IDS):
        ax.annotate(str(tid), (scores[i, 0], scores[i, 1]),
                    fontsize=7, alpha=0.7, ha='center', va='bottom')

    ax.set_xlabel(f'PC1 ({evr[0]:.1%})')
    ax.set_ylabel(f'PC2 ({evr[1]:.1%})')
    ax.set_title(f'PCA Task Space — {label} (Task-Level Range)')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'pc_scatter_{safe_label}.png')
    plt.savefig(fig_path, dpi=150), plt.close()
    print(f"Figure: {fig_path}")

    return pca, scores


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("TASK-LEVEL RANGE PCA ON MyoKi DATASET (23 participants)")
    print("=" * 70)
    print(f"Participants: {', '.join(PARTICIPANTS)}")
    print(f"Grasp tasks: {len(GRASP_TASK_IDS)} tasks")
    print(f"Data directory: {DATA_DIR}")

    # Load and process
    print(f"\n{'='*60}")
    print("Loading preprocessed data and computing trial ranges...")
    X, task_trial_counts, total_trials = load_and_compute_task_ranges(
        PARTICIPANTS, GRASP_TASK_IDS, DATA_DIR, TARGET_FRAMES)

    print(f"\nTotal trials loaded: {total_trials}")
    print(f"Task-level matrix: {X.shape}")
    print(f"\nTrial counts per task:")
    for tid in GRASP_TASK_IDS:
        cat = GRASP_CATEGORIES[tid]
        n = task_trial_counts[tid]
        print(f"  Task {tid:2d} ({cat:12s}): {n:4d} trials")

    # Run PCA
    label = f'all_grasp_trialrange_none_all23'
    pca, scores = run_pca_on_task_matrix(X, label, SAVE_DIR, FIG_DIR)

    # Summary table
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"\n{'='*70}")
    print(f"{'Dataset':<45s} {'PC1':>8s} {'PC2':>8s} {'PC1+PC2':>10s}")
    print(f"{'-'*70}")
    print(f"{'All 23 participants (task-level range)':<45s} {evr[0]:>7.1%} {evr[1]:>7.1%} {evr[0]+evr[1]:>8.1%}")
    print(f"\nDone! Results saved to {SAVE_DIR}/")


if __name__ == '__main__':
    main()
