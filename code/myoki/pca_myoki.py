"""
PCA analysis on MyoKi preprocessed data (16-D hand space, wrist excluded).

Usage:
    python pca_myoki.py [--data_file all_participants_preprocessed.mat]
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
N_JOINTS = 16  # hand only


def run_pca(data_path, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Loading data: {data_path}")
    mat = sio.loadmat(data_path)
    angles = mat['angles']  # (N, 16)
    print(f"Data shape: {angles.shape}, Joint space: {angles.shape[1]}-D (hand only, wrist removed)")

    mean_pose = np.mean(angles, axis=0)
    print(f"Mean pose (16-D): {mean_pose}")

    # PCA
    n_comp = min(angles.shape[1], N_JOINTS)
    pca = PCA(n_components=n_comp)
    pca.fit(angles)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"\nExplained variance: {evr}")
    print(f"Cumulative: {cum}")

    # Save model
    np.savez(os.path.join(save_dir, 'pca_myoki_all.npz'),
             components=pca.components_,
             mean=pca.mean_,
             explained_variance=evr,
             explained_variance_cumsum=cum,
             n_samples=angles.shape[0],
             n_features=angles.shape[1])
    print(f"Saved: {save_dir}/pca_myoki_all.npz")

    # ---- Variance plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = axes[0]
    bars = ax1.bar(range(1, n_comp+1), evr, color='steelblue', edgecolor='navy')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual PC Variance (MyoKi 16-D hand)')
    ax1.set_xticks(range(1, n_comp+1))
    ax1.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        if i < 5:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                     f'{evr[i]:.1%}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    ax2.plot(range(1, n_comp+1), cum, 'o-', color='darkorange', linewidth=2, markersize=8)
    ax2.axhline(0.9, color='gray', ls='--', alpha=.7, label='90%')
    ax2.axhline(0.95, color='gray', ls=':', alpha=.7, label='95%')
    ax2.set_xlabel('Number of PCs')
    ax2.set_ylabel('Cumulative Variance')
    ax2.set_title('Cumulative Variance (MyoKi 16-D hand)')
    ax2.set_xticks(range(1, n_comp+1))
    ax2.grid(alpha=.3)
    ax2.legend()
    n_90, n_95 = np.searchsorted(cum, 0.90)+1, np.searchsorted(cum, 0.95)+1
    for n_, label in [(n_90, '90%'), (n_95, '95%')]:
        if n_ <= n_comp:
            ax2.annotate(f'{n_} PCs → {label}', xy=(n_, cum[n_-1]),
                         xytext=(n_+1, cum[n_-1]-0.05),
                         arrowprops=dict(arrowstyle='->'), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'myoki_variance.png'), dpi=150)
    plt.close()

    # ---- First 3 eigenpostures ----
    joint_labels = [f"ch{i+1}" for i in range(N_JOINTS)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        comp = pca.components_[i]
        colors = ['steelblue' if c >= 0 else 'coral' for c in comp]
        ax.bar(range(1, N_JOINTS+1), comp, color=colors, edgecolor='navy', lw=0.5)
        ax.set_title(f'Eigenposture {i+1} (Var: {evr[i]:.1%})')
        ax.set_xlabel('MyoKi Joint Index')
        ax.set_ylabel('Weight')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xticks(range(1, N_JOINTS+1, 2))
        ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'myoki_eigenpostures.png'), dpi=150)
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("PCA Summary (MyoKi 16-D hand space)")
    print("="*60)
    for i in range(min(6, n_comp)):
        print(f"  PC{i+1}: {evr[i]:.2%} (cum: {cum[i]:.2%})")
    print(f"  90% → {n_90} PCs,  95% → {n_95} PCs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='MyoKi Dataset/preprocessed')
    parser.add_argument('--data_file', default='all_participants_preprocessed.mat')
    args = parser.parse_args()
    path = os.path.join(args.data_dir, args.data_file)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run preprocess_myoki.py first.")
        sys.exit(1)
    run_pca(path)


if __name__ == '__main__':
    main()
