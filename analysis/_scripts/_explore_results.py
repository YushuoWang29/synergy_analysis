"""Analyze saved MyoKi PCA results to compare PC1 vs PC2 similarity."""
import numpy as np
import os, glob

results_dir = 'PCA_results/MyoKi'
fig_dir = 'MyoKi_figures'

# Load all PCA results
files = sorted(glob.glob(os.path.join(results_dir, 'pca_myoki_*.npz')))
for f in files:
    basename = os.path.basename(f)
    if '_pisa' in basename:
        continue
    data = np.load(f, allow_pickle=True)
    comps = data['components']
    evr = data['explained_variance']
    label = data.get('data_label', basename)
    
    # Check similarity between PC1 and PC2
    pc1 = comps[0]
    pc2 = comps[1]
    cos_sim = np.dot(pc1, pc2) / (np.linalg.norm(pc1) * np.linalg.norm(pc2))
    angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
    
    # Check if PC1 is roughly uniform
    uniform = np.ones_like(pc1) / np.linalg.norm(np.ones_like(pc1))
    pc1_uniform_sim = np.dot(pc1, uniform)
    
    print(f"\n{'='*60}")
    print(f"File: {basename}")
    print(f"Label: {label}")
    print(f"PC1 variance: {evr[0]:.2%}")
    print(f"PC2 variance: {evr[1]:.2%}")
    print(f"PC1+PC2: {(evr[0]+evr[1]):.2%}")
    print(f"PC1 · PC2 cos_sim: {cos_sim:+.4f}  (angle: {angle:.1f}°)")
    print(f"PC1 · Uniform cos_sim: {pc1_uniform_sim:+.4f}")
    
    # Also check rotated versions
    pisa_file = f.replace('.npz', '_pisa_rotated.npz')
    if os.path.exists(pisa_file):
        pd = np.load(pisa_file, allow_pickle=True)
        rcomps = pd['components']
        revr = pd['explained_variance']
        print(f"  Rotated PC1: {revr[0]:.2%}")
        print(f"  Rotated PC2: {revr[1]:.2%}")
        
        # Check rotated vs target
        target1 = np.ones(15) / np.linalg.norm(np.ones(15))
        target2 = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
        target2 = target2 / np.linalg.norm(target2)
        
        r1_sim = np.dot(rcomps[0], target1)
        r2_sim = np.dot(rcomps[1], target2)
        print(f"  RotPC1 · PisaSyn1: {r1_sim:+.4f}")
        print(f"  RotPC2 · PisaSyn2: {r2_sim:+.4f}")

# Load pisa_myoki_all_grasp_static and print detailed component weights
print("\n\n=== DETAILED: ALL GRASP STATIC ===")
f = os.path.join(results_dir, 'pca_myoki_all_grasp_static.npz')
if os.path.exists(f):
    d = np.load(f)
    pc1, pc2 = d['components'][0], d['components'][1]
    names = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
             "IdxMCP","IdxPIP",
             "MidMCP","MidPIP","IdxMidAbd",
             "RingMCP","RingPIP","MidRingAbd",
             "LitMCP","LitPIP","RingLitAbd"]
    print("\nPC1 weights:")
    for i in range(15):
        print(f"  {names[i]:12s}: {pc1[i]:+7.4f}")
    print(f"\nPC2 weights:")
    for i in range(15):
        print(f"  {names[i]:12s}: {pc2[i]:+7.4f}")
    print(f"\nPC1 · PC2 = {np.dot(pc1, pc2):+.4f}")
    print(f"PC1 var: {d['explained_variance'][0]:.2%}")
    print(f"PC2 var: {d['explained_variance'][1]:.2%}")
