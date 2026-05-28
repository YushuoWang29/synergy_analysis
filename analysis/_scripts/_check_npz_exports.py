"""
Check all exported NPZ files and produce a detailed summary table.
"""
import os, json, glob
import numpy as np
from scipy.linalg import orthogonal_procrustes

EXPERIMENTS_DIR = 'PCA_results/experiments'
PCA_MODELS_DIR = 'PCA_results/experiments'

# Pisa target vectors for pattern comparison
N_JOINTS = 15
PISA_S1 = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
PISA_S2_RAW = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_S2 = PISA_S2_RAW / np.linalg.norm(PISA_S2_RAW)

JOINT_NAMES = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp","IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd","RingMCP","RingPIP","MidRingAbd","LitMCP","LitPIP","RingLitAbd"]

def compute_pisa_similarity(pc_15d):
    """Compute Procrustes-rotated similarity to Pisa targets."""
    pc1 = pc_15d[0]
    pc2 = pc_15d[1]
    PC = np.column_stack([pc1, pc2])
    target = np.column_stack([PISA_S1, PISA_S2])
    R, _ = orthogonal_procrustes(PC, target)
    rotated = PC @ R
    s1_sim = float(np.dot(rotated[:, 0], PISA_S1) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(PISA_S1) + 1e-10))
    s2_sim = float(np.dot(rotated[:, 1], PISA_S2) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(PISA_S2) + 1e-10))
    return s1_sim, s2_sim, rotated

def describe_config(label):
    """Parse a label to extract meaningful description."""
    desc = {}
    
    # Task set
    if 'all_grasp' in label: desc['task'] = 'All grasp tasks (41)'
    elif 'all_tasks' in label: desc['task'] = 'All tasks (74)'
    elif 'power_precision' in label: desc['task'] = 'Power+Precision (41)'
    elif 'power_only' in label: desc['task'] = 'Power grip (28)'
    elif 'precision_only' in label: desc['task'] = 'Precision grip (13)'
    elif 'complex_grasp' in label: desc['task'] = 'Complex+grasp'
    elif 'pickup_only' in label: desc['task'] = 'Pickup (21)'
    elif 'all_grasp_nongrasp' in label: desc['task'] = 'Grasp+non-grasp'
    elif 'ninapro' in label or 'Ninapro' in label: desc['task'] = 'Ninapro'
    else: desc['task'] = label.split('_')[1] if len(label.split('_'))>1 else '?'
    
    # Data representation
    if 'trialmean' in label: desc['rep'] = 'Trial mean'
    elif 'static5' in label: desc['rep'] = 'Static (last 5 frames)'
    elif 'static10' in label: desc['rep'] = 'Static (last 10 frames)'
    elif 'trialsd' in label: desc['rep'] = 'Trial std'
    elif 'trialrange' in label: desc['rep'] = 'Trial range'
    elif 'allframes' in label: desc['rep'] = 'All frames'
    elif 'firstframe' in label: desc['rep'] = 'First frame'
    elif 'lastframe' in label: desc['rep'] = 'Last frame'
    else: desc['rep'] = '?'
    
    # Joints
    if 'all15' in label: desc['joints'] = '15 joints (all)'
    elif 'flex10' in label: desc['joints'] = 'Flexion only (10)'
    elif 'no_abd' in label: desc['joints'] = 'No abduction (12)'
    elif 'pip_only' in label: desc['joints'] = 'PIP only (5)'
    elif 'mcp_only' in label: desc['joints'] = 'MCP only (5)'
    elif 'mcp_pip_no_thumb' in label: desc['joints'] = 'MCP+PIP no thumb (8)'
    elif 'thumb_vs_little' in label: desc['joints'] = 'Thumb+little (4)'
    elif 'thumb_idx_lit' in label: desc['joints'] = 'Thumb+idx+little (6)'
    elif 'thumb_opp_flex' in label: desc['joints'] = 'Thumb opp+flex (11)'
    else: desc['joints'] = '?'
    
    # Normalization
    if '_none' in label: desc['norm'] = 'None'
    elif '_zscore' in label: desc['norm'] = 'Z-score'
    elif '_range' in label: desc['norm'] = 'Range [0,1]'
    elif '_robust' in label: desc['norm'] = 'Robust (IQR)'
    elif '_mean' in label: desc['norm'] = 'Mean centering'
    else: desc['norm'] = '?'
    
    # Participant/averaging
    if 'taskavg' in label: desc['avg'] = 'Task-averaged'
    elif 'subjavg' in label: desc['avg'] = 'Subject-averaged'
    else: desc['avg'] = 'Trial-level'
    
    # Top3 participant selection
    if 'top3pc2' in label: desc['sel'] = 'Top 3 (P02,P07,P09)'
    else: desc['sel'] = 'All 11 participants'
    
    # Experiment number
    if 'e1_' in label: desc['exp'] = 'Basic PCA'
    elif 'e2_' in label: desc['exp'] = 'Task-averaged'
    elif 'e3_' in label: desc['exp'] = 'Norm comparison'
    elif 'e4_' in label: desc['exp'] = 'Ninapro'
    elif 'e6_' in label: desc['exp'] = 'Participant filter'
    elif 'e7_' in label: desc['exp'] = 'Static+dynamic'
    elif 'e8_' in label: desc['exp'] = 'Task balance'
    elif 'e9_' in label: desc['exp'] = 'One sample/task'
    elif 'e10_' in label: desc['exp'] = 'Rest subtract'
    elif 'e11_' in label: desc['exp'] = 'All frames'
    elif 'e12_' in label: desc['exp'] = 'Joint subset'
    else: desc['exp'] = '?'
    
    return desc

def get_participant_count(label):
    """Estimate participant count from label."""
    if 'top3pc2' in label: return 3
    if 'P0' in label or 'P1' in label:
        parts = set()
        for p in ['P01','P02','P03','P04','P05','P06','P07','P08','P09','P11','P12']:
            if p in label: parts.add(p)
        return max(1, len(parts))
    if 'ninapro' in label.lower() or 'Ninapro' in label: return '~40'
    return 11  # default for MyoKi all participants

# The best configs extracted from mega_balance_search results
BEST_CONFIGS = [
    # (name, PC1%, PC2%, Sum%, S1_sim, S2_sim, N_tasks, jids_count, participants)
    # From the mega_balance_results.json analysis
    ("power_only_static5_flex10_robust", 67.5, 26.1, 93.6, 0.9935, 0.5907, 28, 10, 11),
    ("power_precision_static5_flex10_robust", 67.8, 23.9, 91.7, 0.9848, 0.5451, 41, 10, 11),
    ("precision_only_trialsd_flex10_zscore", 62.5, 25.6, 88.1, 0.9903, 0.6690, 13, 10, 11),
    ("precision_only_trialsd_flex10_range", 67.2, 23.9, 91.1, 0.9943, 0.6695, 13, 10, 11),
    ("all_grasp_trialrange_no_abd_none", 77.9, 14.0, 91.9, 0.9690, 0.7050, 41, 12, 11),
    ("power_only_trialrange_no_abd_none", 79.6, 14.8, 94.4, 0.9696, 0.7060, 28, 12, 11),
    ("all_grasp_trialrange_all15_none", 76.9, 14.0, 90.9, 0.9255, 0.6235, 41, 15, 11),
    ("power_only_trialmean_all15_none", 62.8, 10.9, 73.7, 0.7821, 0.6572, 28, 15, 11),
    ("all_grasp_trialmean_all15_none", 55.9, 13.4, 69.4, 0.8328, 0.6277, 41, 15, 11),
    ("all_grasp_static5_all15_none", 60.6, 11.9, 72.5, 0.8328, 0.6277, 41, 15, 11),
    ("all_tasks_trialmean_all15_none", 45.4, 20.4, 65.8, 0.8319, 0.6064, 74, 15, 11),
    ("all_tasks_trialmean_no_abd_none", 46.4, 20.9, 67.3, 0.9268, 0.6754, 74, 12, 11),
    ("precision_only_trialmean_no_abd_zscore", 70.2, 28.4, 98.6, 0.8670, 0.3661, 13, 12, 11),
    ("power_precision_pip_only_zscore", 58.9, 21.2, 80.1, 0.9977, 0.8731, 41, 5, 11),
    ("precision_only_trialmean_all15_none", 75.1, 24.3, 99.5, 0.7448, 0.2831, 13, 15, 11),
    ("power_only_static5_all15_none", 65.6, 10.0, 75.7, 0.8328, 0.6277, 28, 15, 11),
    ("precision_only_static5_all15_none", 51.7, 14.3, 66.0, 0.7921, 0.4743, 13, 15, 11),
    ("all_grasp_trialrange_no_abd_robust", 79.2, 12.3, 91.5, 0.9514, 0.7105, 41, 12, 11),
    ("precision_only_trialsd_all15_zscore", 65.5, 21.0, 86.5, 0.9921, 0.5908, 13, 15, 11),
    ("power_only_static5_no_abd_robust", 69.1, 23.7, 92.7, 0.7493, 0.6256, 28, 12, 11),
]

print("=" * 180)
print("BEST PCA CONFIGURATIONS — COMPLETE TABLE")
print("=" * 180)

# Header
hdr = f"{'#':<3} {'Config Name':<55} {'PC1%':>6} {'PC2%':>6} {'Sum%':>6} {'S1':>6} {'S2':>6} {'N_f':>5} {'N_j':>4} {'Task Set':<30} {'Rep':<22} {'Joints':<22} {'Norm':<16} {'--participant'}"
print(hdr)
print("-" * 180)

for i, (name, pc1, pc2, psum, s1, s2, n_tasks, n_j, n_part) in enumerate(BEST_CONFIGS):
    label = f"balance_{name}"
    
    # Build --participant argument
    # The synergy_myoki.py --pca_model flag takes a path directly
    # But we can also create a --participant shortcut
    pca_path = f"PCA_results/experiments/pca_balance_{name}.npz"
    participant_arg = f"--pca_model {pca_path}"
    
    # Describe details
    desc = describe_config(name)
    
    star = "★" if pc2 > 8 and s1 > 0.9 else " "
    print(f"{star}{i+1:<2} {name:<55} {pc1:>5.1f}% {pc2:>5.1f}% {psum:>5.1f}% {s1:>+5.3f} {s2:>+5.3f} {n_tasks:>5} {n_j:>4} {desc['task']:<30} {desc['rep']:<22} {desc['joints']:<22} {desc['norm']:<16} {participant_arg}")

print()
print("=" * 180)
print("AVAILABLE NPZ FILES FOR VISUALIZATION")
print("=" * 180)
print()
print("Using --pca_model flag:")
print()

for i, (name, pc1, pc2, psum, s1, s2, n_tasks, n_j, n_part) in enumerate(BEST_CONFIGS):
    pca_path = f"PCA_results/experiments/pca_balance_{name}.npz"
    if os.path.exists(pca_path.replace('PCA_results/experiments/', '')):
        exists = "✓"
    else:
        exists = "✗"
    print(f"  {exists} python code/myoki/synergy_myoki.py --pca_model {pca_path}")

print()
print("=" * 180)
print("TOP 5 RECOMMENDED")
print("=" * 180)
print()
print("1. Best overall (flex10 + robust norm, static posture):")
print(f"   python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_power_precision_static5_flex10_robust.npz")
print(f"   PC1={67.8:.1f}%, PC2={23.9:.1f}%, Sum={91.7:.1f}%, S1={0.98:.3f}, S2={0.55:.3f}")
print()
print("2. All grasp, all 15 joints (most explainable):")
print(f"   python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_all_grasp_trialmean_all15_none.npz")
print(f"   PC1={55.9:.1f}%, PC2={13.4:.1f}%, Sum={69.4:.1f}%, S1={0.83:.3f}, S2={0.63:.3f}")
print()
print("3. Precision-only, trialsd, flex10, zscore (best S2 match):")
print(f"   python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_precision_only_trialsd_flex10_zscore.npz")
print(f"   PC1={62.5:.1f}%, PC2={25.6:.1f}%, Sum={88.1:.1f}%, S1={0.99:.3f}, S2={0.67:.3f}")
print()
print("4. PIP-only, 5 joints (extreme Pisa similarity):")
print(f"   python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_power_precision_pip_only_zscore.npz")
print(f"   PC1={58.9:.1f}%, PC2={21.2:.1f}%, Sum={80.1:.1f}%, S1={0.998:.3f}, S2={0.87:.3f}")
print()
print("5. All grasp, trialrange, no_abd, none (high N, good 15-joint):")
print(f"   python code/myoki/synergy_myoki.py --pca_model PCA_results/experiments/pca_balance_all_grasp_trialrange_all15_none.npz")
print(f"   PC1={56.6:.1f}%, PC2={12.9:.1f}%, Sum={69.6:.1f}%, S1={0.93:.3f}, S2={0.60:.3f}")

# Verify which pca_balance files actually exist
print()
print("=" * 180)
print("VERIFYING EXPORTED FILES")
print("=" * 180)

for f in sorted(glob.glob('PCA_results/experiments/pca_balance_*.npz')):
    try:
        d = np.load(f, allow_pickle=True)
        keys = list(d.keys())
        has_components = 'components' in keys
        has_mean = 'mean' in keys
        has_evr = 'explained_variance' in keys
        comps = d['components']
        evr = d['explained_variance']
        n_pcs = comps.shape[0]
        
        # Compute Pisa similarity from the 15-joint components
        s1_sim, s2_sim, _ = compute_pisa_similarity(comps[:2])
        
        status = "✓" if (has_components and has_mean and has_evr) else "✗"
        print(f"  {status} {os.path.basename(f):<60} PC1={evr[0]*100:>5.1f}% PC2={evr[1]*100:>5.1f}% S1={s1_sim:.3f} S2={s2_sim:.3f} components={comps.shape}")
    except Exception as e:
        print(f"  ✗ {os.path.basename(f):<60} ERROR: {e}")
