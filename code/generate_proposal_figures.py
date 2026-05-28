#!/usr/bin/env python3
"""
generate_proposal_figures.py
============================
Generate figures for the biostatistics course proposal on hand synergy analysis.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCA_DIR      = os.path.join(PROJECT_ROOT, 'PCA_results')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'Ninapro Data')
FIG_OUT_DIR  = os.path.join(PROJECT_ROOT, 'latex_report')  # same dir as final .tex
os.makedirs(FIG_OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 200,
})

# ======================================================================
# Fig 1: Dataset overview — subjects per exercise
# ======================================================================
def fig_dataset_overview():
    """Bar chart showing number of subjects and gestures per exercise."""
    # From the dataset: 
    # E1: subjects S28-S67 (40 subjects), originally had S1-S2 in wrong description
    # Actually from the data files: E1 has subjects S28-S67 (40 subjects performing basic movements)
    # E2: subjects S1-S27 (exc S25), S68-S77 (36 subjects, 9 hand gestures)
    # E3: subjects S1-S27 (exc S25), S68-S77 (36 subjects, 12 Dofs force)
    exercises = ['E1', 'E2', 'E3']
    n_subjects = [40, 36, 36]
    n_gestures = [17, 9, 12]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    
    bars1 = ax1.bar(exercises, n_subjects, color=['#3498DB', '#E74C3C', '#2ECC71'], 
                    edgecolor='k', linewidth=0.8, alpha=0.85, width=0.5)
    for bar, n in zip(bars1, n_subjects):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(n),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Subjects')
    ax1.set_title('Subjects per Exercise', fontsize=11)
    ax1.set_ylim(0, 50)
    ax1.grid(axis='y', alpha=0.3)
    
    bars2 = ax2.bar(exercises, n_gestures, color=['#3498DB', '#E74C3C', '#2ECC71'],
                    edgecolor='k', linewidth=0.8, alpha=0.85, width=0.5)
    for bar, n in zip(bars2, n_gestures):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(n),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Gestures')
    ax2.set_title('Gestures per Exercise', fontsize=11)
    ax2.set_ylim(0, 22)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Ninapro Database Overview', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_OUT_DIR, 'proposal_dataset_overview.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK]  {os.path.basename(path)}")
    return path


# ======================================================================
# Fig 2: Preprocessing pipeline flowchart (conceptual)
# ======================================================================
def fig_preprocessing_pipeline():
    """A visual block diagram of the preprocessing pipeline."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Define box style
    def add_box(x, y, w, h, text, color='#AED6F1', text_color='black'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                             facecolor=color, edgecolor='#2C3E50', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=8.5, color=text_color, fontweight='bold')
    
    # Row 1: Data sources
    add_box(0.3, 2.1, 2.5, 0.7, 'Raw .mat Files\n(22 joint angles\n+ restimulus labels)', '#D5F5E3')
    add_box(3.3, 2.1, 2.5, 0.7, 'E1/E2/E3\nClassification\n(from filename)', '#D5F5E3')
    
    # Arrow
    ax.annotate('', xy=(5.8, 2.45), xytext=(3.3, 2.45),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Row 1b: E2 special handling
    add_box(6.3, 2.1, 2.8, 0.7, 'E2: Discard labels\n≥9 (extra gestures)', '#FADBD8')
    
    # Arrow down
    ax.annotate('', xy=(7.7, 1.7), xytext=(7.7, 2.1),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Row 2: Segmentation
    add_box(0.3, 1.0, 2.5, 0.7, 'Trial Segmentation\nBased on restimulus\ndifferential', '#FDEBD0')
    ax.annotate('', xy=(2.8, 1.35), xytext=(0.3, 1.35),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Row 2b: Static removal
    add_box(3.3, 1.0, 2.5, 0.7, 'Static Frame Removal\nVelocity threshold\n= 0.05', '#FDEBD0')
    ax.annotate('', xy=(5.8, 1.35), xytext=(3.3, 1.35),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Row 2c: Resampling
    add_box(6.3, 1.0, 2.8, 0.7, 'Temporal Resampling\nLinear interpolation\nto 100 frames/trial', '#FDEBD0')
    
    # Arrow down
    ax.annotate('', xy=(7.7, 0.55), xytext=(7.7, 1.0),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Row 3: Output
    add_box(5.5, 0.0, 3.5, 0.55, 'Processed Data: Sxx_Ex_B.mat\n(N × 20 finger joint angles)', '#D4E6F1')
    
    ax.set_title('Data Preprocessing Pipeline', fontsize=12, fontweight='bold', pad=5)
    plt.tight_layout()
    path = os.path.join(FIG_OUT_DIR, 'proposal_preprocessing_pipeline.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK]  {os.path.basename(path)}")
    return path


# ======================================================================
# Fig 3: Preliminary PCA results — variance explained with synergy diagram
# ======================================================================
def fig_preliminary_pca():
    """Show preliminary PCA results: variance curve + first 3 eigen-postures side by side."""
    fp = os.path.join(PCA_DIR, 'pca_model_123.npz')
    if not os.path.isfile(fp):
        return None
    d = np.load(fp)
    ev = d['explained_variance']          # ratio
    cum = np.cumsum(ev)
    comps = d['components'][:3]           # first 3 PCs
    
    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1.2],
                          hspace=0.35, wspace=0.3)
    
    # (a) Cumulative variance curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.step(range(1, len(cum)+1), cum*100, where='mid', color='#E74C3C', linewidth=2)
    ax1.axhline(90, ls='--', color='gray', alpha=0.5, label='90% threshold')
    for nc in [3, 5]:
        if nc <= len(cum):
            ax1.plot(nc, cum[nc-1]*100, 'o', color='#2ECC71', ms=6)
            ax1.annotate(f'{cum[nc-1]*100:.1f}%', (nc, cum[nc-1]*100),
                         textcoords='offset points', xytext=(5, 5), fontsize=7)
    ax1.set_xlabel('Number of PCs', fontsize=9)
    ax1.set_ylabel('Cumulative Variance (%)', fontsize=9)
    ax1.set_title('(a) Cumulative Variance', fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    
    # (b) Individual variance (scree)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(range(1, len(ev)+1), ev*100, color='#3498DB', alpha=0.8, 
            edgecolor='k', linewidth=0.5, width=0.7)
    ax2.plot(range(1, len(ev)+1), ev*100, 'o-', color='#E74C3C', markersize=3, linewidth=1)
    ax2.set_xlabel('Principal Component', fontsize=9)
    ax2.set_ylabel('Explained Variance (%)', fontsize=9)
    ax2.set_title('(b) Scree Plot', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # (c) First 3 eigen-postures
    ax3 = fig.add_subplot(gs[:, 1])
    finger_groups = {'Thumb': [19], 'Index': [2,3,4,5], 'Middle': [6,7,8,9], 
                     'Ring': [10,11,12,13], 'Little': [14,15,16,17,18]}
    group_colors = {'Thumb': '#E74C3C', 'Index': '#3498DB', 'Middle': '#2ECC71',
                    'Ring': '#F39C12', 'Little': '#9B59B6'}
    
    joint_labels_short = ['TH','MC','AB','IP',
                          'AB','MC','IP','DP',
                          'AB','MC','IP','DP',
                          'AB','MC','IP','DP',
                          'CM','AB','MC','IP']
    
    y_offset = 0
    for i in range(3):
        comp = comps[i]
        colors = []
        for j in range(20):
            found = False
            for gname, idxs in finger_groups.items():
                if j in idxs:
                    colors.append(group_colors[gname])
                    found = True
                    break
            if not found:
                colors.append('#888888')
        
        for j in range(20):
            ax3.barh(y_offset + j*0.7, comp[j], height=0.6, left=0,
                    color=colors[j], edgecolor='k', linewidth=0.3, alpha=0.85)
        
        ax3.text(0.02, y_offset + 13.5, f'PC{i+1} ({ev[i]*100:.1f}%)', 
                fontsize=9, fontweight='bold', va='bottom')
        y_offset += 16
    
    ax3.set_yticks([])
    ax3.set_xlabel('Weight', fontsize=9)
    ax3.axvline(0, color='gray', linewidth=0.5)
    ax3.set_title('(c) First 3 Eigen-Postures', fontsize=10)
    
    # Legend for finger groups
    legend_elements = [mpatches.Patch(facecolor=group_colors[g], edgecolor='k', label=g) 
                       for g in finger_groups.keys()]
    ax3.legend(handles=legend_elements, fontsize=6, loc='lower right')
    
    plt.suptitle('Preliminary PCA Results (Full Dataset E1+E2+E3)', 
                fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_OUT_DIR, 'proposal_preliminary_pca.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK]  {os.path.basename(path)}")
    return path


# ======================================================================
# Fig 4: Synergy concept diagram
# ======================================================================
def fig_synergy_concept():
    """Conceptual figure: high-dim hand posture → low-dim synergy space."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Left: High-dimensional hand
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw a simplified hand sketch
    # Palm
    palm = plt.Circle((0, 0), 0.8, fill=False, color='#2C3E50', linewidth=2)
    ax.add_patch(palm)
    # Fingers (lines)
    finger_angles = [np.pi/3, np.pi/4.5, np.pi/6, 0, -np.pi/6]
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    for i, (theta, name, c) in enumerate(zip(finger_angles, finger_names, colors)):
        dx = 1.0 * np.cos(theta)
        dy = 1.0 * np.sin(theta)
        ax.plot([0, dx], [0, dy], color=c, linewidth=3, alpha=0.8)
        # Joints
        mid_x, mid_y = dx*0.5, dy*0.5
        ax.plot(mid_x, mid_y, 'o', color=c, markersize=5)
        ax.plot(dx, dy, 'o', color=c, markersize=5)
        ax.text(dx*1.15, dy*1.15, name, fontsize=6, ha='center', va='center', color=c)
    
    ax.set_title('High-Dimensional Posture\n(20 finger DoFs)', fontsize=10, fontweight='bold')
    
    # Arrow in middle: "PCA"
    ax_arrow = axes[0]
    # Right: Low-dimensional synergy space
    ax2 = axes[1]
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Draw 3D coordinate system
    ax2.plot([0, 1.2], [0, 0], color='#E74C3C', linewidth=2)
    ax2.plot([0, -0.6], [0, 0.9], color='#3498DB', linewidth=2)
    ax2.plot([0, -0.4], [0, -1.0], color='#2ECC71', linewidth=2)
    
    ax2.text(1.25, 0, 'PC1', fontsize=9, color='#E74C3C', fontweight='bold')
    ax2.text(-0.7, 0.95, 'PC2', fontsize=9, color='#3498DB', fontweight='bold')
    ax2.text(-0.5, -1.1, 'PC3', fontsize=9, color='#2ECC71', fontweight='bold')
    
    # Data points in synergy space
    np.random.seed(42)
    points_3d = np.random.randn(30, 3) * 0.3
    ax2.scatter(points_3d[:,0], points_3d[:,1], c='#7F8C8D', s=15, alpha=0.6)
    
    # Label
    ax2.text(0, -1.4, 'Synergy Space\n(3-5 dimensions)', 
            ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('From High-Dimensional Postures to Low-Dimensional Synergies', 
                fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_OUT_DIR, 'proposal_synergy_concept.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK]  {os.path.basename(path)}")
    return path


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 50)
    print("  Generating Proposal Figures")
    print("=" * 50)
    
    fig_dataset_overview()
    fig_preprocessing_pipeline()
    fig_preliminary_pca()
    fig_synergy_concept()
    
    print("\nAll figures generated in:", FIG_OUT_DIR)


if __name__ == '__main__':
    main()
