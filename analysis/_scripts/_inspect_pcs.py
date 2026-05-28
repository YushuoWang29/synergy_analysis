"""Inspect PC1/PC2 vectors from NPZ files and compare to Pisa targets."""
import numpy as np
import sys, os, glob
from scipy.linalg import orthogonal_procrustes

JN = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp","IdxMCP","IdxPIP",
      "MidMCP","MidPIP","IdxMidAbd","RingMCP","RingPIP",
      "MidRingAbd","LitMCP","LitPIP","RingLitAbd"]

PS1 = np.ones(15) / np.linalg.norm(np.ones(15))
PS2 = np.array([2,2,2,2,1,1,0,0,0,-1,-1,-1,-2,-2,-2], dtype=float)
PS2 = PS2 / np.linalg.norm(PS2)

def inspect(path):
    d = np.load(path, allow_pickle=True)
    fc = d['components']
    fm = d['mean']
    ev = d['explained_variance']
    
    pc1 = fc[0]; pc2 = fc[1] if fc.shape[0] > 1 else np.zeros(15)
    # Sign fix
    if pc1[1] < 0: pc1 *= -1
    if pc2[1] < 0: pc2 *= -1
    pc1n = pc1 / np.linalg.norm(pc1)
    pc2n = pc2 / np.linalg.norm(pc2)
    
    # Simple cosine sim
    s1 = float(np.dot(pc1n, PS1))
    s2 = float(np.dot(pc2n, PS2))
    
    # Procrustes sim
    P = np.column_stack([pc1n, pc2n])
    T = np.column_stack([PS1, PS2])
    R, _ = orthogonal_procrustes(P, T)
    rot = P @ R
    rs1 = float(np.dot(rot[:,0], PS1))
    rs2 = float(np.dot(rot[:,1], PS2))
    
    print(f"\n{'='*80}")
    print(f"File: {os.path.basename(path)}")
    print(f"{'='*80}")
    print(f"  PC1: {ev[0]*100:.1f}%, PC2: {ev[1]*100:.1f}%, Sum: {(ev[0]+ev[1])*100:.1f}%")
    print(f"  Cosine S1={s1:.4f}, S2={s2:.4f}")
    print(f"  Procrustes S1={rs1:.4f}, S2={rs2:.4f}")
    print()
    
    print(f"  {'Joint':<15} {'PC1':>8} {'PC2':>8} {'PisaS1':>8} {'PisaS2':>8}")
    print(f"  {'-'*55}")
    for i in range(15):
        print(f"  {JN[i]:<15} {pc1[i]:>+8.3f} {pc2[i]:>+8.3f} {PS1[i]:>+8.3f} {PS2[i]:>+8.3f}")
    
    # Pisa pattern match
    print(f"\n  Pattern matches:")
    print(f"  PC1 all positive: {np.all(pc1 > 0)}")
    print(f"  PC1 monotonic thumb-to-pinky (flexion): checking...")
    
    # Check index finger (5 joints: MCP, PIP) patterns
    print(f"\n  Gradient pattern PC2 thumb-to-pinky:")
    print(f"  Thumb (Abd,MCP,IP,Opp): {pc2[0]:+.3f}, {pc2[1]:+.3f}, {pc2[2]:+.3f}, {pc2[3]:+.3f}")
    print(f"  Index (MCP,PIP):       {pc2[4]:+.3f}, {pc2[5]:+.3f}")
    print(f"  Middle (MCP,PIP):      {pc2[6]:+.3f}, {pc2[7]:+.3f}")
    print(f"  Ring (MCP,PIP):        {pc2[9]:+.3f}, {pc2[10]:+.3f}")
    print(f"  Pinky (MCP,PIP):       {pc2[12]:+.3f}, {pc2[13]:+.3f}")
    
    return {
        'file': os.path.basename(path),
        'pc1': ev[0]*100, 'pc2': ev[1]*100, 'sum': (ev[0]+ev[1])*100,
        's1_cos': s1, 's2_cos': s2,
        's1_proc': rs1, 's2_proc': rs2,
        'all_pos': np.all(pc1 > 0),
        'pc1_vec': pc1.tolist(),
        'pc2_vec': pc2.tolist(),
    }

if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob('PCA_results/MyoKi/pca_myoki_s3_*.npz'))
    results = [inspect(p) for p in paths]
    
    # Quick ranking
    def score(r):
        return r['s1_proc'] + r['s2_proc'] + r['pc2']/100 + min(r['sum']/100, 0.95)
    
    results.sort(key=score, reverse=True)
    
    print(f"\n{'='*80}")
    print(f"RANKED BY QUALITY")
    print(f"{'='*80}")
    print(f"{'#':<3} {'File':<55} {'PC1':>6} {'PC2':>6} {'Sum':>6} {'S1_proc':>8} {'S2_proc':>8}")
    print(f"{'-'*90}")
    for i, r in enumerate(results[:15]):
        s = "★" if r['pc2'] > 10 and r['s1_proc'] > 0.8 and r['s2_proc'] > 0.5 else " "
        print(f"{s}{i+1:<2} {r['file'][:54]:<55} {r['pc1']:>5.1f}% {r['pc2']:>5.1f}% {r['sum']:>5.1f}% {r['s1_proc']:>+7.4f} {r['s2_proc']:>+7.4f}")
