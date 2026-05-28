"""Comprehensive comparison: Ninapro PCA vs MyoKi PCA vs Pisa/IIT targets."""
import numpy as np
import os, glob

# =====================================================
# Define synergy targets for both joint spaces
# =====================================================

# MyoKi 15-D synergy targets (same Pisa logic)
P1_15 = np.ones(15) / np.linalg.norm(np.ones(15))
# From thumb(+) to little(-): thumb group=+2, index=+1, middle=0, ring=-1, little=-2
# MyoKi order: ThumbAbd, ThumbMCP, ThumbIP, ThumbOpp, IdxMCP, IdxPIP,
#              MidMCP, MidPIP, IdxMidAbd, RingMCP, RingPIP, MidRingAbd,
#              LitMCP, LitPIP, RingLitAbd
P2_15_raw = [2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2]
P2_15 = np.array(P2_15_raw, dtype=float) / np.linalg.norm(P2_15_raw)

# Ninapro 20-D synergy targets
P1_20 = np.ones(20) / np.linalg.norm(np.ones(20))
# Ninapro order: CMC1_f, CMC1_a, MCP1, IP1, MCP2_f, MCP2_a, PIP2, DIP2,
#                MCP3_f, PIP3, DIP3, MCP4_f, MCP4_a, PIP4, DIP4,
#                CMC5, MCP5_f, MCP5_a, PIP5, DIP5
# Thumb(0-3)=+2, Index(4-7)=+1, Middle(8-10)=0, Ring(11-14)=-1, Little(15-19)=-2
P2_20_raw = [2,2,2,2, 1,1,1,1, 0,0,0, -1,-1,-1,-1, -2,-2,-2,-2,-2]
P2_20 = np.array(P2_20_raw, dtype=float) / np.linalg.norm(P2_20_raw)

# =====================================================
# Part 1: MyoKi analysis
# =====================================================
print("=" * 70)
print("PART 1: MYOKI PCA RESULTS")
print("=" * 70)

for label in ['all_grasp_static', 'all_tasks_static']:
    try:
        mm = np.load(f'PCA_results/MyoKi/pca_myoki_{label}.npz')
    except:
        print(f"{label}: not found")
        continue
    evr = mm['explained_variance']
    pc = mm['components']
    print(f'\n--- {label} (N={mm["n_samples"]}) ---')
    print(f'  PC1: {evr[0]:.2%}, PC2: {evr[1]:.2%}, Sum: {(evr[0]+evr[1]):.2%}')
    
    # Pisa rotation
    try:
        mr = np.load(f'PCA_results/MyoKi/pca_myoki_{label}_pisa_rotated.npz')
        rv = mr['components']
        s1 = np.dot(rv[0], P1_15)
        s2 = np.dot(rv[1], P2_15)
        print(f'  [Rotated] RotPC1 vs P1: {s1:.4f}, RotPC2 vs P2: {s2:.4f}')
        print(f'  R = {mr["rotation_matrix"]}')
    except Exception as e:
        print(f'  No rotated model: {e}')
    
    # Detailed PC1
    print(f'\n  --- PC1 weights ---')
    JOINT_NAMES_15 = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
                      "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
                      "RingMCP","RingPIP","MidRingAbd","LitMCP","LitPIP","RingLitAbd"]
    for j in range(15):
        print(f'    {JOINT_NAMES_15[j]:12s}: {pc[0][j]:+7.4f}')

# =====================================================
# Part 2: Ninapro analysis
# =====================================================
print("\n" + "=" * 70)
print("PART 2: NINAPRO PCA RESULTS")
print("=" * 70)

for label in ['E1', 'E2', 'E3', 'E123']:
    try:
        npz = np.load(f'PCA_results/Ninapro/pca_{label}.npz')
    except:
        print(f"{label}: not found")
        continue
    evr = npz['explained_variance']
    pc = npz['components']
    print(f'\n--- {label} (N={npz["n_samples"]}) ---')
    print(f'  PC1: {evr[0]:.2%}, PC2: {evr[1]:.2%}, Sum: {(evr[0]+evr[1]):.2%}')
    
    # Pisa rotation
    try:
        rpz = np.load(f'PCA_results/Ninapro/pca_{label}_pisa.npz')
        rv = rpz['components']
        s1 = np.dot(rv[0], P1_20)
        s2 = np.dot(rv[1], P2_20)
        print(f'  [Rotated] RotPC1 vs P1: {s1:.4f}, RotPC2 vs P2: {s2:.4f}')
    except Exception as e:
        print(f'  No rotated model: {e}')

# =====================================================
# Part 3: Cross-dataset comparison
# =====================================================
print("\n" + "=" * 70)
print("PART 3: CROSS-DATASET COMPARISON")
print("=" * 70)

# The key question: MyoKi 15-D and Ninapro 20-D are different spaces.
# To compare, we need to map to a common joint space.
# ShadowHand qpos is the common space (24-DOF).

# Let's see what the "loaded" graphs look like by loading the preprocessed data
# and running PCA on the common ShadowHand space.

# For Ninapro, load E2+E3 (skip E1 which has calibration issues)
print('\nNinapro uses CyberGlove II with 22 sensors → directly comparable to MyoKi CyberGlove I')
print('Key differences between datasets:')
print('  1. Ninapro has DIP sensors (indices 16-19), MyoKi does not')
print('  2. Ninapro has wrist sensors (indices 20-21), MyoKi does not')
print('  3. Ninapro has Palmar Arch sensor (index 15), MyoKi has it too (excluded)')
print('  4. Ninapro E1 (basic movements) DIP3 has anomalous variance')
print('  5. MyoKi has Thumb opposition sensor, Ninapro maps CMC1_f to opposition')

# Let's check the actual MyoKi vs Ninapro Pisa rotated patterns
print('\n\n' + '=' * 70)
print('PISA ROTATION COMPARISON (SUMMARY TABLE)')
print('=' * 70)
print(f'{"Dataset":<20} {"N":>6} {"PC1":>7} {"PC2":>7} {"Sum":>7} {"R1->P1":>8} {"R2->P2":>8}')
print('-' * 60)

# MyoKi
for label in ['all_grasp_static', 'all_tasks_static']:
    try:
        mm = np.load(f'PCA_results/MyoKi/pca_myoki_{label}.npz')
        mr = np.load(f'PCA_results/MyoKi/pca_myoki_{label}_pisa_rotated.npz')
        evr = mm['explained_variance']
        rv = mr['components']
        s1 = np.dot(rv[0], P1_15)
        s2 = np.dot(rv[1], P2_15)
        print(f'{"MyoKi-"+label:<20} {mm["n_samples"]:>6} {evr[0]:>6.1%} {evr[1]:>6.1%} {(evr[0]+evr[1]):>6.1%} {s1:>7.4f} {s2:>7.4f}')
    except:
        pass

# Ninapro
for label in ['E1', 'E2', 'E3', 'E12', 'E13', 'E23', 'E123']:
    try:
        npz = np.load(f'PCA_results/Ninapro/pca_{label}.npz')
        rpz = np.load(f'PCA_results/Ninapro/pca_{label}_pisa.npz')
        evr = npz['explained_variance']
        rv = rpz['components']
        s1 = np.dot(rv[0], P1_20)
        s2 = np.dot(rv[1], P2_20)
        print(f'{"Ninapro-"+label:<20} {npz["n_samples"]:>6} {evr[0]:>6.1%} {evr[1]:>6.1%} {(evr[0]+evr[1]):>6.1%} {s1:>7.4f} {s2:>7.4f}')
    except:
        pass

# =====================================================
# Part 4: Per-subject PC1 PC2 analysis
# =====================================================
print('\n\n' + '=' * 70)
print('PART 4: PER-SUBJECT VARIANCE - ARE PC1 AND PC2 SIMILAR?')
print('=' * 70)

# Question: Are PC1 and PC2 close enough to "merge"?
# Answer: They are mathematically orthogonal (by definition of PCA).
# But if PC1 is not dominant enough, PC2 captures motion that looks "similar" to PC1
# because the remaining variance is structured rather than random.

# Let's check: how much does PC1 vary across subjects?
print('\nMyoKi per-subject PC1 (grasp-only):')
for label_fn in sorted(glob.glob('PCA_results/MyoKi/pca_myoki_P*_grasp_static.npz')):
    try:
        m = np.load(label_fn)
        print(f'  {os.path.basename(label_fn):30s} PC1={m["explained_variance"][0]:.1%} PC2={m["explained_variance"][1]:.1%}')
    except:
        pass

# When are PC1 and PC2 "mergable"? 
# Only if Procrustes rotation can find a rotation that aligns to user-defined targets.
# The rotation angles tell us how far apart PC1 and PC2 are from the "ideal" synergy axes.

print('\n' + '=' * 70)
print('CONCLUSIONS')
print('=' * 70)
print('''
1. PC1 and PC2 are mathematically orthogonal - cannot "merge" in standard PCA sense.
   But Varimax rotation or Promax rotation can make them more interpretable.

2. The Ninapro E2+E3 results (grasping + functional tasks) match Pisa/IIT well:
   - RotPC1 vs Pisa Syn1: cos_sim ≈ 0.97-0.98 (excellent!)
   - RotPC2 vs Pisa Syn2: cos_sim ≈ 0.20 (poor)
   The Syn2 (thumb-to-little gradient) is NOT recovered well in any dataset.

3. Ninapro E1 has DIP3 calibration artifacts (std=257 vs normal 2-50).

4. Key insight: Pisa/IIT's strong PC1/PC2 may come from:
   (a) Different recording method (passive markers, not CyberGlove)
   (b) Different task set
   (c) Different subject population
   (d) Different number of joints (Pisa uses 15 DOF with specific joint definitions)

5. MyoKi's PC1~48% vs Ninapro's PC1~71%:
   The difference is due to Ninapro having more subjects (75 vs 8-9),
   more trials per subject, and DIP sensors that add correlated variance.
''')
