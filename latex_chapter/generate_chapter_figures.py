#!/usr/bin/env python3
"""
Generate all figures for the chapter on PCA-based hand synergy analysis.
Output: PNG figures in latex_chapter/figures/
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA

# Add project root to path for importing (needed for potential myoki_config import)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

PCA_DIR = os.path.join(PROJECT_ROOT, 'PCA_results', 'MyoKi')

# Set Chinese font (Microsoft YaHei available on Windows)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 200

N_JOINTS = 15
JOINT_NAMES_SHORT = [
    "ThumbAbd", "ThumbMCP", "ThumbIP", "ThumbOpp",
    "IdxMCP", "IdxPIP",
    "MidMCP", "MidPIP", "IdxMidAbd",
    "RingMCP", "RingPIP", "MidRingAbd",
    "LitMCP", "LitPIP", "RingLitAbd",
]

# ======================================================================
# Fig 1: Variance Explained
# ======================================================================
def fig_variance_curve():
    """PC variance explained - individual bars + cumulative curve."""
    pca_path = os.path.join(PCA_DIR, 'pca_myoki_all_grasp_static.npz')
    d = np.load(pca_path, allow_pickle=True)
    ev = d['explained_variance']
    cum = d['explained_variance_cumsum']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Individual variance
    colors_ind = ['#2E86C1'] * 5 + ['#85C1E9'] * 10
    ax1.bar(range(1, len(ev)+1), ev*100, color=colors_ind, edgecolor='#1B4F72', 
            alpha=0.85, linewidth=0.8, width=0.7)
    # Highlight first 3
    ax1.bar(range(1, 4), ev[:3]*100, color='#E74C3C', edgecolor='#922B21', 
            alpha=0.85, linewidth=0.8, width=0.7)
    ax1.set_xlabel('主成分编号', fontsize=12)
    ax1.set_ylabel('方差解释比率 (%)', fontsize=12)
    ax1.set_title('(a) 各主成分方差解释比率', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(1, len(ev)+1))
    ax1.set_xticklabels([f'PC{i}' for i in range(1, len(ev)+1)], fontsize=7, rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    # Annotate first 3
    for i in range(3):
        ax1.text(i+1, ev[i]*100 + 1.5, f'{ev[i]*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # Cumulative variance
    ax2.plot(range(1, len(cum)+1), cum*100, 'o-', color='#E74C3C', linewidth=2.5, markersize=6)
    ax2.fill_between(range(1, len(cum)+1), cum*100, alpha=0.1, color='#E74C3C')
    ax2.axhline(90, color='gray', ls='--', alpha=0.7, linewidth=1.5, label='90% 阈值')
    ax2.axhline(95, color='gray', ls=':', alpha=0.7, linewidth=1.5, label='95% 阈值')
    # Mark key points
    for nc in [3, 6, 9]:
        if nc <= len(cum):
            ax2.plot(nc, cum[nc-1]*100, 'o', color='#2ECC71', ms=8, zorder=5)
            ax2.annotate(f'{cum[nc-1]*100:.1f}%', (nc, cum[nc-1]*100),
                        textcoords='offset points', xytext=(10, 10), fontsize=10,
                        fontweight='bold', color='#1E8449',
                        arrowprops=dict(arrowstyle='->', color='#1E8449', lw=1.5))
    ax2.set_xlabel('主成分数量', fontsize=12)
    ax2.set_ylabel('累计方差解释比率 (%)', fontsize=12)
    ax2.set_title('(b) 累计方差解释曲线', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, len(cum)+1, 2))
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0.5, len(cum)+0.5)
    
    plt.suptitle('图1: PCA方差解释分析（抓取静态姿势数据）', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig01_variance.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")
    return path


# ======================================================================
# Fig 2: First 3 eigenpostures
# ======================================================================
def fig_eigenpostures():
    """Bar plots of the first 3 PC loading vectors."""
    pca_path = os.path.join(PCA_DIR, 'pca_myoki_all_grasp_static.npz')
    d = np.load(pca_path, allow_pickle=True)
    ev = d['explained_variance']
    comps = d['components']  # (n_components, 15)
    
    # Define finger group colors
    finger_colors = {
        0: '#E74C3C',  1: '#E74C3C',  2: '#E74C3C',  3: '#E74C3C',  # Thumb
        4: '#3498DB',  5: '#3498DB',                                     # Index
        6: '#2ECC71',  7: '#2ECC71',  8: '#2ECC71',                     # Middle
        9: '#F39C12',  10: '#F39C12', 11: '#F39C12',                    # Ring
        12: '#9B59B6', 13: '#9B59B6', 14: '#9B59B6',                    # Little
    }
    group_labels = {
        0: '拇指', 4: '食指', 6: '中指', 9: '无名指', 12: '小指'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, ax in enumerate(axes):
        comp = comps[i]
        colors = [finger_colors[j] for j in range(N_JOINTS)]
        bars = ax.bar(range(N_JOINTS), comp, color=colors, edgecolor='#2C3E50', linewidth=0.8, alpha=0.85)
        ax.set_title(f'PC{i+1}（方差解释: {ev[i]*100:.1f}%）', fontsize=13, fontweight='bold')
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES_SHORT, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='black', lw=1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('负载权重', fontsize=11)
        ax.set_ylim(-0.7, 0.7)
    
    # Add finger group labels on top
    axes[0].text(7, 0.62, '拇指', fontsize=9, color='#E74C3C', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#E74C3C', alpha=0.8))
    axes[0].text(12, 0.62, '食指     中指     无名指     小指', fontsize=9, color='#2C3E50', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Legend for finger groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='拇指'),
        Patch(facecolor='#3498DB', edgecolor='#2C3E50', label='食指'),
        Patch(facecolor='#2ECC71', edgecolor='#2C3E50', label='中指'),
        Patch(facecolor='#F39C12', edgecolor='#2C3E50', label='无名指'),
        Patch(facecolor='#9B59B6', edgecolor='#2C3E50', label='小指'),
    ]
    axes[2].legend(handles=legend_elements, fontsize=9, loc='upper right')
    
    plt.suptitle('图2: 前三主成分特征姿势（Eigenpostures）', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig02_eigenpostures.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")
    return path


# ======================================================================
# Fig 3: PC1-PC2 Scatter plot with shadow hand concept
# ======================================================================
def fig_pc1_pc2_scatter():
    """PC1 vs PC2 scatter plot showing the distribution of all grasp postures."""
    pca_path = os.path.join(PCA_DIR, 'pca_myoki_all_grasp_static.npz')
    d = np.load(pca_path, allow_pickle=True)
    ev = d['explained_variance']
    comps = d['components']
    mean = d['mean']
    n_samples = d['n_samples']
    
    # We need the scores; load the preprocessed data to project
    data_path = os.path.join(PROJECT_ROOT, 'MyoKi Dataset', 'preprocessed')
    all_data = []
    for f in sorted(glob.glob(os.path.join(data_path, 'P*_preprocessed.mat'))):
        import scipy.io as sio
        mat = sio.loadmat(f)
        angles = mat['angles']
        trial_tasks = mat['trial_tasks'].flatten().astype(int)
        num_trials = int(mat['num_trials'].item() if hasattr(mat['num_trials'], 'item') else mat['num_trials'])
        # Get static postures
        n_frames = angles.shape[0]
        target_frames = 100
        last_frames_frac = 0.25
        if n_frames == num_trials * target_frames:
            trial_data = angles.reshape(num_trials, target_frames, N_JOINTS)
            n_last = max(5, int(target_frames * last_frames_frac))
            static_poses = np.mean(trial_data[:, -n_last:, :], axis=1)
        else:
            static_poses = []
            for ti in range(num_trials):
                s = int(ti * n_frames / num_trials)
                e = int((ti + 1) * n_frames / num_trials)
                if e - s > 10:
                    n_last = max(5, int((e - s) * last_frames_frac))
                    static_poses.append(np.mean(angles[e - n_last:e], axis=0))
                else:
                    static_poses.append(np.mean(angles[s:e], axis=0))
            static_poses = np.array(static_poses)
        # Filter grasp tasks only (pickup tasks)
        pickup_ids = {1, 3, 5, 9, 11, 13, 16, 18, 20, 28, 30, 32, 34, 36, 38, 40, 42, 44}
        mask = np.isin(trial_tasks, list(pickup_ids))
        all_data.append(static_poses[mask])
    
    X = np.vstack(all_data)
    print(f"  Projected data shape: {X.shape}")
    
    # Project onto PC1-PC2
    X_centered = X - mean
    scores = X_centered @ comps[:2].T  # (N, 2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with density coloring
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c='#2E86C1', s=15, 
                         alpha=0.5, edgecolors='none', label='各抓取姿势')
    
    # Add density contours
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(scores.T)
        xmin, xmax = scores[:, 0].min() - 20, scores[:, 0].max() + 20
        ymin, ymax = scores[:, 1].min() - 20, scores[:, 1].max() + 20
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kde(positions).T, xx.shape)
        ax.contour(xx, yy, f, levels=5, colors='#E74C3C', alpha=0.5, linewidths=1)
    except:
        pass
    
    # Axes
    ax.axhline(0, color='gray', lw=1, alpha=0.5)
    ax.axvline(0, color='gray', lw=1, alpha=0.5)
    ax.set_xlabel(f'PC1（方差解释: {ev[0]*100:.1f}%）', fontsize=13)
    ax.set_ylabel(f'PC2（方差解释: {ev[1]*100:.1f}%）', fontsize=13)
    ax.set_title('图3: 所有抓取静态姿势在PC1-PC2平面的投影', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.2)
    
    # Annotate extreme postures
    extremes_idx = []
    for target_angle in [0, np.pi/2, np.pi, -np.pi/2]:
        dir_vec = np.array([np.cos(target_angle), np.sin(target_angle)])
        proj = scores @ dir_vec
        extremes_idx.append(np.argmax(proj))
    
    labels = ['全力抓取（握拳）', '捏取姿势', '全伸展（张开）', '侧捏姿势']
    colors_star = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']
    for idx, label, color in zip(extremes_idx, labels, colors_star):
        ax.plot(scores[idx, 0], scores[idx, 1], '*', markersize=15, color=color, zorder=10)
        ax.annotate(label, (scores[idx, 0], scores[idx, 1]),
                   textcoords='offset points', xytext=(15, 15), fontsize=10,
                   fontweight='bold', color=color,
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Legend
    ax.legend(fontsize=10, loc='upper right')
    
    # Annotation for synergy direction interpretation
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    textstr = f'数据点: N={len(scores)}\nPC1+PC2累计方差: {(ev[0]+ev[1])*100:.1f}%\nPC1: 开-合协同\nPC2: 拇指-手指对立'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig03_pc1_pc2_scatter.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")
    return path


# ======================================================================
# Fig 4: Pisa rotation comparison
# ======================================================================
def fig_pisa_rotation():
    """Compare original PCs vs Pisa-rotated vs Pisa/IIT SoftHand targets."""
    pca_path = os.path.join(PCA_DIR, 'pca_myoki_all_grasp_static.npz')
    rot_path = os.path.join(PCA_DIR, 'pca_myoki_all_grasp_static_pisa_rotated.npz')
    
    d = np.load(pca_path, allow_pickle=True)
    dr = np.load(rot_path, allow_pickle=True)
    
    ev = d['explained_variance']
    orig_comps = d['components'][:2]  # (2, 15)
    rot_comps = dr['components']  # (2, 15)
    
    # Pisa targets
    pisa_1 = np.ones(15) / np.linalg.norm(np.ones(15))
    pisa_2 = np.array([2, 2, 2, 2, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2], dtype=float)
    pisa_2 = pisa_2 / np.linalg.norm(pisa_2)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    titles_col = ['原始PC', 'Pisa旋转后', 'Pisa/IIT目标协同']
    comps_to_plot = [
        [orig_comps[0], orig_comps[1]],
        [rot_comps[0], rot_comps[1]],
        [pisa_1, pisa_2]
    ]
    col_titles = ['协同1（PC1对应）', '协同2（PC2对应）']
    
    for col in range(3):
        for row in range(2):
            ax = axes[row, col]
            comp = comps_to_plot[col][row]
            colors = ['steelblue' if c >= 0 else 'coral' for c in comp]
            ax.bar(range(15), comp, color=colors, edgecolor='navy', lw=0.6)
            if row == 0:
                ax.set_title(f'{titles_col[col]}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(15))
            ax.set_xticklabels(JOINT_NAMES_SHORT, rotation=45, ha='right', fontsize=6.5)
            ax.axhline(0, color='black', lw=0.8)
            ax.grid(axis='y', alpha=0.2)
            ax.set_ylim(-0.7, 0.7)
            
            if col == 2 and row < 2:
                ax.set_ylabel(col_titles[row], fontsize=10, fontweight='bold')
    
    plt.suptitle('图4: PCA协同与Pisa/IIT SoftHand理论协同对比（Procrustes旋转对齐）', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig04_pisa_rotation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")
    return path


# ======================================================================
# Fig 5: Per-participant comparison
# ======================================================================
def fig_participant_comparison():
    """Bar chart comparing PC1, PC2, PC1+PC2 across all participants."""
    participants = ['P01','P02','P03','P04','P05','P06','P07','P08','P09','P11','P12']
    pc1_vals = []
    pc2_vals = []
    n_samples_list = []
    labels = []
    
    for pid in participants:
        path = os.path.join(PCA_DIR, f'pca_myoki_{pid}_all_tasks_static.npz')
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        ev = d['explained_variance']
        n = d['n_samples']
        pc1_vals.append(ev[0]*100)
        pc2_vals.append(ev[1]*100)
        n_samples_list.append(n)
        labels.append(pid)
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    
    width = 0.35
    bars1 = ax.bar(x - width/2, pc1_vals, width, label='PC1', color='#E74C3C', 
                   edgecolor='#922B21', alpha=0.85)
    bars2 = ax.bar(x + width/2, pc2_vals, width, label='PC2', color='#3498DB', 
                   edgecolor='#1B4F72', alpha=0.85)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(pc1_vals, pc2_vals)):
        ax.text(i - width/2, v1 + 0.8, f'{v1:.1f}%', ha='center', fontsize=7.5, fontweight='bold')
        ax.text(i + width/2, v2 + 0.8, f'{v2:.1f}%', ha='center', fontsize=7.5, fontweight='bold')
    
    # Add PC1+PC2 combined bars on secondary y-axis
    ax2 = ax.twinx()
    combined = [a+b for a, b in zip(pc1_vals, pc2_vals)]
    ax2.plot(x, combined, 'o-', color='#2ECC71', linewidth=2.5, markersize=10, 
             label='PC1+PC2', zorder=5)
    for i, v in enumerate(combined):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=8.5, fontweight='bold', 
                color='#1E8449')
    
    ax.set_xlabel('被试编号', fontsize=12)
    ax.set_ylabel('方差解释比率 (%)', fontsize=12)
    ax2.set_ylabel('PC1+PC2合计 (%)', fontsize=12, color='#2ECC71')
    ax.set_title('图5: 各被试PCA结果对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(pc1_vals), max(pc2_vals)) * 1.5)
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig05_participant_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")
    return path


# ======================================================================
# Fig 6: Conceptual figure - hand synergy principle
# ======================================================================
def fig_synergy_concept():
    """Conceptual illustration: high-D hand → low-D synergy space."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Left: High-dimensional hand posture
    ax = axes[0]
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Palm
    palm = plt.Circle((0, 0), 0.9, fill=False, color='#2C3E50', linewidth=2.5)
    ax.add_patch(palm)
    
    # Fingers
    finger_angles = [np.pi/2.8, np.pi/4.5, np.pi/6.5, 0, -np.pi/5]
    finger_names = ['拇指', '食指', '中指', '无名指', '小指']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    for i, (theta, name, c) in enumerate(zip(finger_angles, finger_names, colors)):
        length = 1.2 if i == 0 else 1.0
        dx = length * np.cos(theta)
        dy = length * np.sin(theta)
        # Finger segments (2 phalanges)
        ax.plot([0, dx*0.5], [0, dy*0.5], color=c, linewidth=4, alpha=0.85)
        ax.plot([dx*0.5, dx], [dy*0.5, dy], color=c, linewidth=3.5, alpha=0.85)
        # Joint markers
        ax.plot(dx*0.5, dy*0.5, 'o', color='white', markersize=6, markeredgecolor=c, markeredgewidth=2)
        ax.plot(dx, dy, 'o', color=c, markersize=6)
        ax.text(dx*1.2, dy*1.2, name, fontsize=9, ha='center', va='center', 
                color=c, fontweight='bold')
    
    ax.set_title('高维手部姿态空间\n（15自由度）', fontsize=12, fontweight='bold')
    
    # Middle: PCA arrow
    ax_mid = axes[1]
    ax_mid.axis('off')
    ax_mid.set_xlim(-1, 1)
    ax_mid.set_ylim(-1, 1)
    
    # Large arrow
    ax_mid.annotate('', xy=(0.3, 0), xytext=(-0.3, 0),
                   arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=5),
                   fontsize=16)
    ax_mid.text(0, -0.2, '主成分分析\n(PCA)', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#FAD7A0', edgecolor='#E67E22', pad=0.5))
    # Labels
    ax_mid.text(-0.6, 0.4, '高维数据', ha='center', fontsize=10, fontweight='bold', color='#2C3E50')
    ax_mid.text(0.6, 0.4, '低维表示', ha='center', fontsize=10, fontweight='bold', color='#2C3E50')
    
    # Right: Low-dimensional synergy space
    ax2 = axes[2]
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Draw synergy axes
    ax2.plot([0, 1.5], [0, 0], color='#E74C3C', linewidth=3)
    ax2.plot([0, -0.8], [0, 1.2], color='#3498DB', linewidth=3)
    ax2.plot([0, -0.6], [0, -1.3], color='#2ECC71', linewidth=3)
    
    ax2.text(1.6, 0, 'PC1', fontsize=12, color='#E74C3C', fontweight='bold')
    ax2.text(-1.0, 1.3, 'PC2', fontsize=12, color='#3498DB', fontweight='bold')
    ax2.text(-0.7, -1.45, 'PC3', fontsize=12, color='#2ECC71', fontweight='bold')
    
    # Data points in synergy space
    np.random.seed(42)
    points = np.random.randn(40, 3) * 0.35
    ax2.scatter(points[:, 0], points[:, 1], c='#7F8C8D', s=25, alpha=0.6, edgecolors='white', linewidths=0.3)
    
    # Label examples
    ax2.text(0.8, 0.6, '握拳', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='#FADBD8', edgecolor='#E74C3C'))
    ax2.text(-0.6, 0.8, '捏取', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='#2ECC71'))
    ax2.text(0.2, -0.7, '伸展', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#D4E6F1', edgecolor='#3498DB'))
    
    ax2.set_title('低维协同空间\n（3-6维即可解释~90%方差）', fontsize=12, fontweight='bold')
    
    plt.suptitle('图6: 手部姿态协同分析原理示意图', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig06_synergy_concept.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")
    return path


# ======================================================================
# Fig 7: MyoKi task examples relevant to different PC extremes
# ======================================================================
def fig_task_examples():
    """Show the mean posture for 4 representative grasp tasks projected in PC space."""
    pca_path = os.path.join(PCA_DIR, 'pca_myoki_all_grasp_static.npz')
    d = np.load(pca_path, allow_pickle=True)
    ev = d['explained_variance']
    comps = d['components']
    mean = d['mean']
    
    # Load data to extract task-specific means
    import scipy.io as sio
    data_path = os.path.join(PROJECT_ROOT, 'MyoKi Dataset', 'preprocessed')
    
    # Selected tasks: 1 (cup pick up), 28 (ball pick up), 34 (book pick up), 40 (bag lift)
    task_info = {
        1: '拿杯子（精确抓取）',
        28: '拿球（球状抓取）',
        34: '拿书（捏取）',
        40: '提包（力量抓取）',
    }
    
    task_means = {}
    for pid in ['P01','P03','P05']:
        fpath = os.path.join(data_path, f'{pid}_preprocessed.mat')
        if not os.path.exists(fpath):
            continue
        mat = sio.loadmat(fpath)
        angles = mat['angles']
        trial_tasks = mat['trial_tasks'].flatten().astype(int)
        num_trials = int(mat['num_trials'].item() if hasattr(mat['num_trials'], 'item') else mat['num_trials'])
        
        n_frames = angles.shape[0]
        target_frames = 100
        if n_frames == num_trials * target_frames:
            trial_data = angles.reshape(num_trials, target_frames, N_JOINTS)
            n_last = max(5, int(target_frames * 0.25))
            static_poses = np.mean(trial_data[:, -n_last:, :], axis=1)
        else:
            static_poses = []
            for ti in range(num_trials):
                s = int(ti * n_frames / num_trials)
                e = int((ti + 1) * n_frames / num_trials)
                if e - s > 10:
                    n_last = max(5, int((e - s) * 0.25))
                    static_poses.append(np.mean(angles[e - n_last:e], axis=0))
                else:
                    static_poses.append(np.mean(angles[s:e], axis=0))
            static_poses = np.array(static_poses)
        
        for tid, tname in task_info.items():
            mask = trial_tasks == tid
            if np.any(mask):
                if tid not in task_means:
                    task_means[tid] = []
                task_means[tid].append(static_poses[mask])
    
    if not task_means:
        print("  WARNING: Could not extract task-specific data")
        return None
    
    # Average task means across participants
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    
    for idx, (tid, tname) in enumerate(task_info.items()):
        ax = axes_flat[idx]
        if tid in task_means:
            task_data = np.vstack(task_means[tid])
            task_mean = np.mean(task_data, axis=0)
            
            # Project onto PC1-PC2
            score = (task_mean - mean) @ comps[:2].T
            
            # Plot bar chart of posture
            colors = ['#E74C3C']*4 + ['#3498DB']*2 + ['#2ECC71']*3 + ['#F39C12']*3 + ['#9B59B6']*3
            bars = ax.bar(range(15), task_mean, color=colors, edgecolor='#2C3E50', linewidth=0.8, alpha=0.85)
            ax.set_title(f'任务 {tid}: {tname}\n(PC1={score[0]:.1f}, PC2={score[1]:.1f})', 
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(15))
            ax.set_xticklabels(JOINT_NAMES_SHORT, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('关节角度 (度)', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('图7: 代表性抓取任务的静态手部姿态', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig07_task_examples.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")
    return path


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 60)
    print("  Generating Chapter Figures for Hand Synergy PCA Analysis")
    print("=" * 60)
    
    fig_variance_curve()
    fig_eigenpostures()
    fig_pc1_pc2_scatter()
    fig_pisa_rotation()
    fig_participant_comparison()
    fig_synergy_concept()
    fig_task_examples()
    
    print(f"\nAll figures saved to: {FIG_DIR}")

if __name__ == '__main__':
    main()
