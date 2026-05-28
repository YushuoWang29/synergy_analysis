"""
PCA on Ninapro preprocessed data — per-exercise, combined, with Pisa/IIT Procrustes rotation.
Uses the new preprocessing pipeline (glove field, static-hold, 20 hand joints).
"""
import os, glob, sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SAVE = 'PCA_results/Ninapro'
FIG = 'Ninapro_figures'
PRE = 'Ninapro Data/preprocessed'

# Joint names for 20 hand-only joints
JOINT_NAMES = [
    "CMC1_f", "CMC1_a", "MCP1", "IP1",
    "MCP2_f", "MCP2_a", "PIP2", "DIP2",
    "MCP3_f", "PIP3", "DIP3",
    "MCP4_f", "MCP4_a", "PIP4", "DIP4",
    "CMC5", "MCP5_f", "MCP5_a", "PIP5", "DIP5",
]
N_JOINTS = 20

# Pisa/IIT targets (20-D)
P1 = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
P2_RAW = [2,2,1,1,  1,1,1,1,  1,1,1,  -1,-1,-1,-1,  -2,-2,-2,-2,-2]
P2 = np.array(P2_RAW, dtype=float) / np.linalg.norm(P2_RAW)
PT = np.column_stack([P1, P2])


def procrustes(pc, t):
    M = pc.T @ t
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return pc @ R, R


def load_ex(ex):
    """Load all preprocessed files for a given exercise (E1, E2, E3)."""
    fs = sorted(glob.glob(os.path.join(PRE, f'S*_{ex}_preprocessed.mat')))
    d = {}
    for f in fs:
        s = os.path.basename(f).split('_')[0]
        m = sio.loadmat(f)
        a = m['angles']
        # Keep only first 20 joints (hand-only)
        if a.shape[1] >= N_JOINTS:
            a = a[:, :N_JOINTS]
        d[s] = (a, m['trial_tasks'].flatten())
    return d


def run(label, a, save_dir, fig_dir, nm=JOINT_NAMES):
    n = min(a.shape[1], 20)
    p = PCA(n_components=n)
    p.fit(a)
    e = p.explained_variance_ratio_
    c = np.cumsum(e)
    
    print(f'\n{"="*60}')
    print(f'{label} [{len(a)} samples]')
    for i in range(min(6, n)):
        print(f'  PC{i+1}: {e[i]:.2%}  (cum: {c[i]:.2%})')
    n90 = int(np.searchsorted(c, 0.9)+1)
    n95 = int(np.searchsorted(c, 0.95)+1)
    print(f'  90%: {n90} PCs, 95%: {n95} PCs')
    
    # Save standard PCA
    fp = os.path.join(save_dir, f'pca_{label}.npz')
    np.savez(fp, components=p.components_, mean=p.mean_,
             explained_variance=e, explained_variance_cumsum=c,
             n_samples=len(a), data_label=label)
    print(f'Saved: {fp}')
    
    # Procrustes rotation
    k = min(2, p.components_.shape[0])
    pv = p.components_[:k].T
    rv, R = procrustes(pv, PT[:, :k])
    scores = p.transform(a)
    scores_rot = scores[:, :k] @ R.T
    rv_var = np.var(scores_rot, axis=0)
    rv_ratio = rv_var / rv_var.sum()
    tv = e[:k].sum()
    rn = rv_ratio * tv
    
    print(f'\n--- Pisa rotation (Procrustes) ---')
    print(f'R:\n{R}')
    for i in range(k):
        sim = np.dot(rv[:, i], PT[:, i])
        print(f'  Rot PC{i+1}: {rn[i]:.2%}  vs Pisa Syn{i+1}: cos_sim={sim:.4f}')
    
    # Save rotated
    fr = os.path.join(save_dir, f'pca_{label}_pisa.npz')
    np.savez(fr, components=rv.T, mean=p.mean_, rotation=R,
             explained_variance=rn, original=e[:k], total=tv,
             n_samples=len(a), data_label=f'{label}_pisa')
    print(f'Saved: {fr}')
    
    # Print detailed weights
    for name, vec in [('PC1', p.components_[0]), ('RotPC1', rv[:, 0]),
                      ('PC2', p.components_[1]), ('RotPC2', rv[:, 1])]:
        print(f'\n  {name}:')
        for j in range(N_JOINTS):
            print(f'    {nm[j]:12s}: {vec[j]:+7.4f}')
    
    # === Figures ===
    # 1. Variance
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(range(1, n+1), e, color='steelblue', edgecolor='navy')
    ax[0].bar(1, rn[0], color='limegreen', alpha=0.5, edgecolor='darkgreen', lw=1.5)
    if k > 1:
        ax[0].bar(2, rn[1], color='limegreen', alpha=0.5, edgecolor='darkgreen', lw=1.5, label='Pisa-rot')
    ax[0].legend(fontsize=8)
    ax[0].set_xticks(range(1, min(n+1, 21)))
    ax[0].grid(axis='y', alpha=0.3)
    ax[0].set_title(f'{label} Variance')
    for i in range(min(3, n)):
        ax[0].text(i+1, e[i]+0.005, f'{e[i]:.1%}', ha='center', va='bottom', fontsize=8)
    ax[1].plot(range(1, n+1), c, 'o-', color='darkorange', lw=2, ms=8)
    ax[1].axhline(0.9, color='gray', ls='--', alpha=.7)
    ax[1].axhline(0.95, color='gray', ls=':', alpha=.7)
    ax[1].set_title(f'{label} Cumulative')
    ax[1].set_xticks(range(1, min(n+1, 21)))
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'variance_{label}.png'), dpi=150)
    plt.close()
    
    # 2. Eigenpostures (top 3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for i, ax in enumerate(axes):
        co = p.components_[i]
        cl = ['steelblue' if v >= 0 else 'coral' for v in co]
        ax.bar(range(N_JOINTS), co, color=cl, edgecolor='navy', lw=0.5)
        ax.set_title(f'PC{i+1} ({e[i]:.1%})')
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(nm, rotation=45, ha='right', fontsize=7)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=.3)
    plt.suptitle(f'Eigenpostures - {label}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'eigenpostures_{label}.png'), dpi=150)
    plt.close()
    
    # 3. Pisa comparison (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(24, 7))
    for r in range(2):
        for c, (v, t) in enumerate([
            (p.components_[r], f'PC{r+1} ({e[r]:.1%})'),
            (rv[:, r], f'Rot PC{r+1}'),
            (PT[:, r], f'Pisa Syn{r+1}'),
        ]):
            cl = ['steelblue' if x >= 0 else 'coral' for x in v]
            axes[r, c].bar(range(N_JOINTS), v, color=cl,
                          edgecolor='navy' if c < 2 else 'darkgreen',
                          lw=0.5 if c < 2 else 1.0)
            axes[r, c].set_title(t, fontsize=11)
            axes[r, c].set_xticks(range(N_JOINTS))
            axes[r, c].set_xticklabels(nm, rotation=45, ha='right', fontsize=6)
            axes[r, c].axhline(0, color='black', lw=0.5)
            axes[r, c].grid(axis='y', alpha=.3)
    plt.suptitle(f'Pisa Rotation - {label}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'pisa_{label}.png'), dpi=150)
    plt.close()
    
    return p, e, c


if __name__ == '__main__':
    os.makedirs(SAVE, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    
    e1 = load_ex('E1')
    e2 = load_ex('E2')
    e3 = load_ex('E3')
    
    n1 = sum(len(v[0]) for v in e1.values()) if e1 else 0
    n2 = sum(len(v[0]) for v in e2.values()) if e2 else 0
    n3 = sum(len(v[0]) for v in e3.values()) if e3 else 0
    print(f'E1: {len(e1)} subjects, {n1} samples')
    print(f'E2: {len(e2)} subjects, {n2} samples')
    print(f'E3: {len(e3)} subjects, {n3} samples')
    
    # Per exercise
    for en, ed in [('E1', e1), ('E2', e2), ('E3', e3)]:
        if ed:
            run(en, np.vstack([v[0] for v in ed.values()]), SAVE, FIG)
    
    # Combinations
    for label, datasets in [
        ('E12', [e1, e2]), ('E13', [e1, e3]),
        ('E23', [e2, e3]), ('E123', [e1, e2, e3])
    ]:
        items = [v[0] for d in datasets for v in d.values() if d]
        if items:
            run(label, np.vstack(items), SAVE, FIG)
    
    # Per-subject summary
    print(f'\n{"="*70}')
    print(f'{"Subj":<8} {"PC1":>8} {"PC2":>8} {"Sum":>8} {"N":>8}')
    print(f'{"-"*45}')
    for en, ed in [('E2', e2), ('E3', e3)]:
        if not ed: continue
        for s in sorted(ed.keys()):
            q = PCA(n_components=2).fit(ed[s][0])
            print(f'{s}_{en:<4} {q.explained_variance_ratio_[0]:>7.1%} '
                  f'{q.explained_variance_ratio_[1]:>7.1%} '
                  f'{q.explained_variance_ratio_[:2].sum():>7.1%} {len(ed[s][0]):>7}')
    
    print(f'\nDone! Results in {SAVE}, figures in {FIG}')
