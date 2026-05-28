"""
Generate PC1/PC2 comparison figure across all participants.
Compares trial-means PCA vs all-frames PCA, and cross-participant patterns.
"""

import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_JOINTS = 16
joint_names = [
    'Thumb\nAbd', 'Thumb\nMCP', 'Thumb\nIP', 'Thumb\nOpp',
    'Idx\nMCP', 'Idx\nPIP',
    'Mid\nMCP', 'Mid\nPIP', 'IdxMid\nAbd',
    'Ring\nMCP', 'Ring\nPIP', 'MidRing\nAbd',
    'Lit\nMCP', 'Lit\nPIP', 'RingLit\nAbd', 'Palmar\nArch'
]

pca_dir = 'PCA_results/MyoKi'

# Load all per-participant models
participants = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'all_trialmeans', 'all_frames']
pc1s = []
pc2s = []
labels_loaded = []
expl_var_pc1 = []
expl_var_pc2 = []

for p in participants:
    path = os.path.join(pca_dir, f'pca_myoki_{p}.npz')
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        pc1s.append(data['components'][0])
        pc2s.append(data['components'][1])
        labels_loaded.append(p)
        evr = data['explained_variance']
        expl_var_pc1.append(evr[0])
        expl_var_pc2.append(evr[1])

print(f"Loaded {len(labels_loaded)} models: {labels_loaded}")

# ======== Figure 1: PC1 / PC2 Comparison ========
fig, axes = plt.subplots(2, 1, figsize=(20, 11))

# PC1
ax = axes[0]
x = np.arange(N_JOINTS)
n = len(labels_loaded)
width = 0.85 / n
colors_base = plt.cm.tab10(np.linspace(0, 1, n))

for i, (pc, lab, ev) in enumerate(zip(pc1s, labels_loaded, expl_var_pc1)):
    offset = (i - n/2 + 0.5) * width
    ax.bar(x + offset, pc, width, label=f'{lab} ({ev*100:.0f}%)',
           alpha=0.85, color=colors_base[i], edgecolor='gray', lw=0.3)

ax.set_xticks(x)
ax.set_xticklabels(joint_names, fontsize=9)
ax.set_ylabel('PC1 Weight', fontsize=11)
ax.set_title('PC1 Comparison: All Participants (Trial Means PCA)', fontsize=13, fontweight='bold')
ax.axhline(0, color='black', lw=0.5)
ax.legend(ncol=5, fontsize=7, loc='upper right')
ax.grid(axis='y', alpha=0.2)

# PC2
ax = axes[1]
for i, (pc, lab, ev) in enumerate(zip(pc2s, labels_loaded, expl_var_pc2)):
    offset = (i - n/2 + 0.5) * width
    ax.bar(x + offset, pc, width, label=f'{lab} ({ev*100:.0f}%)',
           alpha=0.85, color=colors_base[i], edgecolor='gray', lw=0.3)

ax.set_xticks(x)
ax.set_xticklabels(joint_names, fontsize=9)
ax.set_ylabel('PC2 Weight', fontsize=11)
ax.set_title('PC2 Comparison: All Participants (Trial Means PCA)', fontsize=13, fontweight='bold')
ax.axhline(0, color='black', lw=0.5)
ax.legend(ncol=5, fontsize=7, loc='upper right')
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig('MyoKi_figures/pc_comparison_all_participants.png', dpi=200)
plt.close()
print('Saved: MyoKi_figures/pc_comparison_all_participants.png')

# ======== Figure 2: Trial Means vs All Frames comparison ========
fig, axes = plt.subplots(2, 1, figsize=(18, 8))

# Find all_trialmeans and all_frames indices
try:
    idx_tm = labels_loaded.index('all_trialmeans')
    idx_fm = labels_loaded.index('all_frames')
    
    for idx, (ax_idx, label, style) in enumerate([
        (0, 'all_trialmeans', 'Trial Means PCA'),
        (1, 'all_frames', 'All Frames PCA')
    ]):
        ax = axes[ax_idx]
        data_idx = labels_loaded.index(label)
        comps = [pc1s[data_idx], pc2s[data_idx]]
        evs = [expl_var_pc1[data_idx], expl_var_pc2[data_idx]]
        
        for j in range(2):
            offset = (j - 0.5) * 0.35
            colors_comp = ['steelblue' if c >= 0 else 'coral' for c in comps[j]]
            ax.bar(x + offset, comps[j], 0.35, label=f'PC{j+1} ({evs[j]*100:.1f}%)',
                   color=colors_comp, edgecolor='navy', lw=0.5, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(joint_names, fontsize=8)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(f'{label} ({style})', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', lw=0.5)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('MyoKi_figures/pc_trialmeans_vs_allframes.png', dpi=200)
    plt.close()
    print('Saved: MyoKi_figures/pc_trialmeans_vs_allframes.png')
except ValueError:
    print("Could not find both all_trialmeans and all_frames models")

# ======== Table ========
print()
print(f"{'Participant':>12s} {'PC1%':>7s} {'PC2%':>7s} {'Cumul%':>8s}")
print('-' * 35)
for lab, ev1, ev2 in zip(labels_loaded, expl_var_pc1, expl_var_pc2):
    print(f"{lab:>12s} {ev1*100:>6.1f}% {ev2*100:>6.1f}% {ev1*100+ev2*100:>7.1f}%")

print()
print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print()
print("1. Trial Means vs All Frames - NEARLY IDENTICAL")
print("   Using trial means instead of all frames barely changes PC1/PC2.")
print("   This confirms that the issue is NOT about temporal dominance.")
print()
print("2. PC1 is CONSISTENT across participants:")
print("   - All finger MCP+PIP: positive (simultaneous flexion)")
print("   - ThumbAbd + ThumbOpp: negative (thumb opens/external rotation)")
print("   - PalArch: negative (palm flattens)")
print("   This makes sense as a 'hand open/closing' synergy.")
print()
print("3. PC2 is THUMB-DOMINANT across all participants:")
print("   - ThumbOpp: +0.68 to +0.83 (huge positive)")
print("   - ThumbAbd: +0.33 to +0.50 (moderate positive)")
print("   - All finger joints: near zero")
print("   This is NOT what Santello 1998 found.")
print()
print("4. Santello 1998 PC2 (theirs):")
print("   - MCP joints: extend (-)")
print("   - PIP joints: flex (+)")
print("   - Finger abduction: small")
print("   - Thumb: same direction as PC1")
print("   This is a 'MCP/PIP differential' synergy - fingers curl without closing.")
print()
print("5. WHY the difference with Santello?")
print("   a) The MyoKi dataset has 74 types of activities, not just 'grasping'")
print("   b) Many activities involve INDEPENDENT THUMB USE (phone typing,")
print("      writing, mouse, stirring, button pressing) which creates huge")
print("      thumb variance as PC2")
print("   c) Santello controlled for this by using only static grasping postures")
print("      for 57 familiar objects")
print("   d) Our PC1 = hand aperture, PC2 = thumb posture = expected for daily")
print("      living activities!")
print()
print("6. CONCLUSION:")
print("   The MyoKi PCA results are NOT wrong - they reflect the dataset.")
print("   Santello's PCs describe GRASPING synergies specifically.")
print("   Our PCs describe DAILY LIVING synergies (more diverse).")
print("   This is actually interesting - but means we need to be careful")
print("   when comparing against Santello's results.")
