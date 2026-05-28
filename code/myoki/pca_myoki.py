"""
PCA analysis on MyoKi preprocessed data — grasp-only, static-hold phase.

Key methodology:
  1. Uses the LAST N frames of each trial (static hold phase)
  2. Excludes "put back" tasks
  3. 15-D joint space (palmar arch removed)
  4. Also performs PROCRUSTES ROTATION to align PC1/PC2 to Pisa/IIT SoftHand 2 synergies

Usage:
    python pca_myoki.py
    python pca_myoki.py --participants P01 P03
"""

import os, sys, glob, argparse
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SAVE_DIR = 'PCA_results/MyoKi'
FIG_DIR  = 'MyoKi_figures'
N_JOINTS = 15
TARGET_FRAMES = 100
LAST_FRAMES_FRAC = 0.25

# Pisa/IIT SoftHand 2 Target Synergies
# Synergy 1: All fingers close together → uniform [1,...1]
# Synergy 2: Gradient thumb(+) to little(-) → [+2,...,-2]
PISA_SYNERGY_1_TARGET = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
PISA_SYNERGY_2_TARGET = np.array([
    +2.0, +2.0, +2.0, +2.0,
    +1.0, +1.0,
     0.0,  0.0,  0.0,
    -1.0, -1.0, -1.0,
    -2.0, -2.0, -2.0,
], dtype=float)
PISA_SYNERGY_2_TARGET = PISA_SYNERGY_2_TARGET / np.linalg.norm(PISA_SYNERGY_2_TARGET)
PISA_TARGET_MATRIX = np.column_stack([PISA_SYNERGY_1_TARGET, PISA_SYNERGY_2_TARGET])

JOINT_NAMES = [
    "ThumbAbd", "ThumbMCP", "ThumbIP", "ThumbOpp",
    "IdxMCP", "IdxPIP",
    "MidMCP", "MidPIP", "IdxMidAbd",
    "RingMCP", "RingPIP", "MidRingAbd",
    "LitMCP", "LitPIP", "RingLitAbd",
]


def compute_static_posture(preproc_path, last_frames_frac=LAST_FRAMES_FRAC):
    mat = sio.loadmat(preproc_path)
    angles = mat['angles']
    trial_tasks = mat['trial_tasks'].flatten().astype(int)
    num_trials = int(mat['num_trials'].item() if hasattr(mat['num_trials'], 'item') else mat['num_trials'])
    n_frames = angles.shape[0]
    expected_frames = num_trials * TARGET_FRAMES
    if n_frames != expected_frames:
        static_poses, trial_ids = [], []
        for ti in range(num_trials):
            s = int(ti * n_frames / num_trials)
            e = int((ti + 1) * n_frames / num_trials)
            if e - s > 10:
                n_last = max(5, int((e - s) * last_frames_frac))
                static_poses.append(np.mean(angles[e - n_last:e], axis=0))
            else:
                static_poses.append(np.mean(angles[s:e], axis=0))
            trial_ids.append(trial_tasks[ti] if ti < len(trial_tasks) else 0)
        return np.array(static_poses), np.array(trial_ids)
    trial_data = angles.reshape(num_trials, TARGET_FRAMES, N_JOINTS)
    n_last = max(5, int(TARGET_FRAMES * last_frames_frac))
    static_poses = np.mean(trial_data[:, -n_last:, :], axis=1)
    return static_poses, trial_tasks


def filter_grasp_and_pickup_only(trial_means, trial_tasks):
    pickup_grasp_task_ids = {1, 3, 5, 9, 11, 13, 16, 18, 20, 28, 30, 32, 34, 36, 38, 40, 42, 44}
    mask = np.isin(trial_tasks, list(pickup_grasp_task_ids))
    filtered = trial_means[mask], trial_tasks[mask]
    print(f"  Filter: {len(trial_means)} trials -> {len(filtered[0])} (excluded {len(trial_means)-len(filtered[0])})")
    return filtered


def apply_procrustes_rotation(pc_vectors, target_vectors):
    """
    pc_vectors: (n_features, n_components) — PC loading vectors as columns
    target_vectors: (n_features, n_targets) — target synergy vectors as columns
    Returns: rotated_vectors (n_features, n_targets), R (n_components, n_targets)
    """
    M = pc_vectors.T @ target_vectors
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    rotated_vectors = pc_vectors @ R
    return rotated_vectors, R


def plot_pisa_comparison(pca_components, pca_evr, rotated_components, label, save_dir, fig_dir):
    """2 rows × 3 cols: Original | Rotated | Pisa Target"""
    fig, axes = plt.subplots(2, 3, figsize=(21, 7))
    # Col 0: Original
    for r in range(2):
        ax = axes[r, 0]
        comp = pca_components[r]
        colors = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax.bar(range(N_JOINTS), comp, color=colors, edgecolor='navy', lw=0.5)
        ax.set_title(f'Original — PC{r+1} ({pca_evr[r]:.1%})', fontsize=11)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
        ax.set_ylim(-0.6, 0.6)
    # Col 1: Rotated
    for r in range(2):
        ax = axes[r, 1]
        comp = rotated_components[r]
        colors = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax.bar(range(N_JOINTS), comp, color=colors, edgecolor='navy', lw=0.5)
        ax.set_title(f'Rotated→Pisa — PC{r+1}', fontsize=11)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
        ax.set_ylim(-0.6, 0.6)
    # Col 2: Target
    for r in range(2):
        ax = axes[r, 2]
        target = [PISA_SYNERGY_1_TARGET, PISA_SYNERGY_2_TARGET][r]
        colors = ['steelblue' if c >= 0 else 'coral' for c in target]
        ax.bar(range(N_JOINTS), target, color=colors, edgecolor='darkgreen', lw=1.0)
        ax.set_title(f'Pisa/IIT Target — Syn{r+1}', fontsize=11)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
        ax.set_ylim(-0.6, 0.6)
    plt.suptitle(f'Pisa Rotation — {label}', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'pisa_rotation_{label}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure: {fig_path}")


def run_pca_on_data(angles, label, save_dir, fig_dir):
    n_comp = min(angles.shape[1], N_JOINTS)
    pca = PCA(n_components=n_comp)
    pca.fit(angles)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"\nData shape: {angles.shape}")
    for i in range(min(6, n_comp)):
        print(f"  PC{i+1}: {evr[i]:.2%} (cum: {cum[i]:.2%})")
    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    n_95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"  90% -> {n_90} PCs,  95% -> {n_95} PCs")

    # Save standard PCA
    save_path = os.path.join(save_dir, f'pca_myoki_{label}.npz')
    np.savez(save_path, components=pca.components_, mean=pca.mean_,
             explained_variance=evr, explained_variance_cumsum=cum,
             n_samples=angles.shape[0], n_features=angles.shape[1], data_label=label)
    print(f"Saved: {save_path}")

    # Procrustes rotation
    pc_vectors = pca.components_[:2].T  # (15, 2)
    rotated_vectors, R = apply_procrustes_rotation(pc_vectors, PISA_TARGET_MATRIX)
    scores = pca.transform(angles)
    scores_rotated = scores[:, :2] @ R.T
    rotated_var = np.var(scores_rotated, axis=0)
    rotated_var_ratio = rotated_var / rotated_var.sum()
    total_var_pc1pc2 = evr[:2].sum()
    rotated_normalized = rotated_var_ratio * total_var_pc1pc2

    print(f"\n--- Procrustes Rotation → Pisa ---")
    print(f"R:\n{R}")
    print(f"Rot PC1: {rotated_normalized[0]:.2%}  Rot PC2: {rotated_normalized[1]:.2%}  "
          f"(PC1+PC2={total_var_pc1pc2:.1%})")
    for i in range(2):
        sim = np.dot(rotated_vectors[:, i], PISA_TARGET_MATRIX[:, i])
        print(f"  Rot PC{i+1} vs Pisa Syn{i+1}: cos_sim={sim:.4f}")

    # Save rotated model
    rotated_path = os.path.join(save_dir, f'pca_myoki_{label}_pisa_rotated.npz')
    np.savez(rotated_path, components=rotated_vectors.T, mean=pca.mean_,
             rotation_matrix=R, explained_variance=rotated_normalized,
             explained_variance_original=evr[:2], total_pc1pc2_variance=total_var_pc1pc2,
             n_samples=angles.shape[0], n_features=angles.shape[1],
             data_label=f"{label}_pisa_rotated")
    print(f"Saved: {rotated_path}")

    # Print PC weights
    for name, vec in [("Original PC1", pca.components_[0]),
                      ("Rot PC1→Pisa Syn1", rotated_vectors[:, 0]),
                      ("Original PC2", pca.components_[1]),
                      ("Rot PC2→Pisa Syn2", rotated_vectors[:, 1])]:
        print(f"\n  {name}:")
        for j in range(N_JOINTS):
            print(f"    {JOINT_NAMES[j]:12s}: {vec[j]:+7.4f}")

    # Variance plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = axes[0]
    ax1.bar(range(1, n_comp+1), evr, color='steelblue', edgecolor='navy')
    ax1.bar(1, rotated_normalized[0], color='limegreen', alpha=0.5, edgecolor='darkgreen', lw=1.5)
    ax1.bar(2, rotated_normalized[1], color='limegreen', alpha=0.5, edgecolor='darkgreen', lw=1.5, label='Pisa-rotated')
    ax1.set_xlabel('PC'), ax1.set_ylabel('Variance Ratio')
    ax1.set_title(f'Variance — {label}'), ax1.set_xticks(range(1, n_comp+1))
    ax1.grid(axis='y', alpha=0.3), ax1.legend(fontsize=8)
    for i in range(min(3, n_comp)):
        ax1.text(i+1, evr[i]+0.005, f'{evr[i]:.1%}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    ax2.plot(range(1, n_comp+1), cum, 'o-', color='darkorange', lw=2, markersize=8)
    ax2.axhline(0.9, color='gray', ls='--', alpha=.7, label='90%')
    ax2.axhline(0.95, color='gray', ls=':', alpha=.7, label='95%')
    ax2.set_xlabel('Number of PCs'), ax2.set_ylabel('Cumulative Variance')
    ax2.set_title(f'Cumulative — {label}'), ax2.set_xticks(range(1, n_comp+1))
    ax2.grid(alpha=.3), ax2.legend()
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'variance_{label}.png')
    plt.savefig(fig_path, dpi=150), plt.close()
    print(f"Figure: {fig_path}")

    # Eigenpostures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, ax in enumerate(axes):
        comp = pca.components_[i]
        colors = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax.bar(range(N_JOINTS), comp, color=colors, edgecolor='navy', lw=0.5)
        ax.set_title(f'PC{i+1} ({evr[i]:.1%})', fontsize=12)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=7)
        ax.axhline(0, color='black', lw=0.5), ax.grid(axis='y', alpha=.3)
    plt.suptitle(f'Eigenpostures — {label}', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'eigenpostures_{label}.png')
    plt.savefig(fig_path, dpi=150), plt.close()
    print(f"Figure: {fig_path}")

    # Pisa rotation comparison
    plot_pisa_comparison(pca.components_, evr, rotated_vectors.T, label, save_dir, fig_dir)
    return pca


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='MyoKi Dataset/preprocessed')
    parser.add_argument('--participants', nargs='+', default=None)
    parser.add_argument('--last_frames', type=float, default=LAST_FRAMES_FRAC)
    args = parser.parse_args()

    save_dir, fig_dir = SAVE_DIR, FIG_DIR
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.data_dir, 'P*_preprocessed.mat')))
    if args.participants:
        files = [f for f in all_files if any(p in os.path.basename(f) for p in args.participants)]
    else:
        files = all_files
    if not files:
        print(f"No files found in {args.data_dir}"), sys.exit(1)

    print(f"Found {len(files)} participant(s), using last {args.last_frames:.0%} frames/trial")
    print(f"Excluding 'put back' tasks, applying Pisa/IIT Procrustes rotation")

    all_static_poses = []
    for fpath in files:
        basename = os.path.basename(fpath).replace('_preprocessed.mat', '')
        print(f"\n{'--'*30}\nProcessing {basename}...")
        poses, tasks = compute_static_posture(fpath, args.last_frames)
        poses, tasks = filter_grasp_and_pickup_only(poses, tasks)
        all_static_poses.append(poses)
        run_pca_on_data(poses, basename, save_dir, fig_dir)

    combined = np.vstack(all_static_poses)
    print(f"\n{'#'*60}\nCombined PCA\n{'#'*60}")
    run_pca_on_data(combined, 'all_grasp_static', save_dir, fig_dir)

    # Summary
    print(f"\n{'='*70}\n{'Dataset':<25} {'PC1':>8} {'PC2':>8} {'PC1+PC2':>10} {'N':>10}")
    print(f"{'-'*60}")
    for fpath in files:
        basename = os.path.basename(fpath).replace('_preprocessed.mat', '')
        poses, tasks = compute_static_posture(fpath, args.last_frames)
        poses, tasks = filter_grasp_and_pickup_only(poses, tasks)
        pca_tmp = PCA(n_components=2); pca_tmp.fit(poses)
        print(f"{basename:<25} {pca_tmp.explained_variance_ratio_[0]:>7.1%} "
              f"{pca_tmp.explained_variance_ratio_[1]:>7.1%} "
              f"{pca_tmp.explained_variance_ratio_[:2].sum():>8.1%} {len(poses):>10}")
    pca_comb = PCA(n_components=2); pca_comb.fit(combined)
    print(f"{'ALL (static):':<25} {pca_comb.explained_variance_ratio_[0]:>7.1%} "
          f"{pca_comb.explained_variance_ratio_[1]:>7.1%} "
          f"{pca_comb.explained_variance_ratio_[:2].sum():>8.1%} {len(combined):>10}")
    print(f"\nDone!")


if __name__ == '__main__':
    main()
