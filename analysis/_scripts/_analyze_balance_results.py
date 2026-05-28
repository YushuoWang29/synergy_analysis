"""
Analyze the balance search results: find the absolute best configurations,
rank them, and produce a clean summary table.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

SAVE_DIR = 'PCA_results/experiments'
FIG_DIR = 'PCA_results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

N_JOINTS = 15
JOINT_NAMES = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp","IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd","RingMCP","RingPIP","MidRingAbd","LitMCP","LitPIP","RingLitAbd"]

PISA_S1 = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
PISA_S2_RAW = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_S2 = PISA_S2_RAW / np.linalg.norm(PISA_S2_RAW)

def compute_similarity(pc, target):
    pc_n = pc / (np.linalg.norm(pc) + 1e-10)
    t_n = target / (np.linalg.norm(target) + 1e-10)
    return float(np.dot(pc_n, t_n))

def load_results():
    with open(os.path.join(SAVE_DIR, 'mega_balance_results.json'), 'r') as f:
        return json.load(f)

def score_balanced(r):
    """Score for balanced results."""
    s1 = r.get('rot_s1_sim', 0)
    s2 = r.get('rot_s2_sim', 0)
    pc2 = r.get('pc2_var', 0)
    pc1 = r.get('pc1_var', 0)
    total = r.get('pc1_pc2_sum', 0)
    
    pattern_score = abs(s1) + abs(s2)
    pc2_bonus = min(1.0, (pc2 - 0.08) * 8) if pc2 >= 0.08 else -0.5
    total_bonus = min(0.5, (total - 0.80) * 3) if total >= 0.80 else (total - 0.80)
    pc1_penalty = max(0, (pc1 - 0.90) * 10) if pc1 > 0.90 else 0
    pc1_low_penalty = max(0, (0.55 - pc1) * 2) if pc1 < 0.55 else 0
    
    return pattern_score + pc2_bonus + total_bonus - pc1_penalty - pc1_low_penalty

def categorize(r):
    """Categorize a result."""
    label = r['label']
    cat = []
    if 'e1' in label: cat.append('Basic')
    elif 'e2' in label: cat.append('TaskAvg')
    elif 'e3' in label: cat.append('NormDeep')
    elif 'e4' in label: cat.append('Ninapro')
    elif 'e6' in label: cat.append('Participant')
    elif 'e7' in label: cat.append('StaticDynamic')
    elif 'e8' in label: cat.append('TaskBalance')
    elif 'e9' in label: cat.append('OnePerTask')
    elif 'e10' in label: cat.append('RestSub')
    elif 'e11' in label: cat.append('AllFrames')
    elif 'e12' in label: cat.append('JointSubset')
    else: cat.append('Other')
    
    if 'ninapro' in label or 'Ninapro' in label: cat.append('NinaproDataset')
    else: cat.append('MyoKiDataset')
    
    if 'all15' in label or 'all_joints' in label: cat.append('AllJoints')
    elif any(s in label for s in ['flex10','no_abd','mcp_only','pip_only']): cat.append('JointSubset')
    else: cat.append('MaybeSubset')
    
    if '_none' in label: cat.append('NoNorm')
    elif '_zscore' in label: cat.append('Zscore')
    elif '_range' in label: cat.append('Range')
    elif '_robust' in label: cat.append('Robust')
    else: cat.append('MaybeNorm')
    
    return '|'.join(cat)

def print_filtered_table(results, title, condition_fn, max_rows=20):
    filtered = [r for r in results if condition_fn(r)]
    filtered.sort(key=score_balanced, reverse=True)
    
    print(f"\n{'='*100}")
    print(f"{title} ({len(filtered)} results)")
    print(f"{'='*100}")
    print(f"{'#':<4} {'Experiment':<55} {'PC1%':>7} {'PC2%':>7} {'Sum%':>7} {'S1':>8} {'S2':>8} {'Score':>7} {'N':>6}")
    print("-"*100)
    
    for i, r in enumerate(filtered[:max_rows]):
        bs = score_balanced(r)
        star = "★" if i == 0 else " "
        print(f"{star}{i+1:<3} {r['label']:<55} {r['pc1_var']*100:>6.1f}% {r['pc2_var']*100:>6.1f}% {r['pc1_pc2_sum']*100:>6.1f}% {r['rot_s1_sim']:>+7.4f} {r['rot_s2_sim']:>+7.4f} {bs:>6.3f} {r['n_samples']:>6}")
    
    # Find the best one
    if filtered:
        best = filtered[0]
        print(f"\n  → Best: {best['label']}")
        print(f"    PC1={best['pc1_var']*100:.1f}% PC2={best['pc2_var']*100:.1f}% Sum={best['pc1_pc2_sum']*100:.1f}%")
        print(f"    S1 sim={best['rot_s1_sim']:.4f} S2 sim={best['rot_s2_sim']:.4f}")
        print(f"    N={best['n_samples']} joints={best['n_joints']}")
    
    print()
    return filtered[:max_rows]

def plot_comparison(results, title, filename):
    """Plot PC1-PC2 variance vs S1+S2 similarity scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    pc2s = [r['pc2_var']*100 for r in results]
    sums = [r['pc1_pc2_sum']*100 for r in results]
    s1s = [r['rot_s1_sim'] for r in results]
    s2s = [r['rot_s2_sim'] for r in results]
    scores = [score_balanced(r) for r in results]
    
    sc1 = ax1.scatter(pc2s, sums, c=scores, cmap='viridis', s=40, alpha=0.7)
    ax1.set_xlabel('PC2 Variance (%)')
    ax1.set_ylabel('PC1+PC2 Sum (%)')
    ax1.set_title(f'{title}\nColored by Balance Score')
    ax1.axhline(80, color='r', linestyle='--', alpha=0.3, label='80% target')
    ax1.axvline(10, color='g', linestyle='--', alpha=0.3, label='10% PC2 target')
    ax1.legend(fontsize=8)
    plt.colorbar(sc1, ax=ax1)
    ax1.grid(True, alpha=0.3)
    
    sc2 = ax2.scatter(s1s, s2s, c=scores, cmap='viridis', s=40, alpha=0.7)
    ax2.set_xlabel('S1 Similarity (Pisa PC1)')
    ax2.set_ylabel('S2 Similarity (Pisa PC2)')
    ax2.set_title(f'{title}\nPattern Similarity')
    ax2.axhline(0.7, color='g', linestyle='--', alpha=0.3)
    ax2.axvline(0.7, color='g', linestyle='--', alpha=0.3)
    ax2.plot([0.85,0.85], [0,1.05], 'r--', alpha=0.2)
    ax2.plot([0,1.05], [0.85,0.85], 'r--', alpha=0.2)
    plt.colorbar(sc2, ax=ax2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def main():
    all_results = load_results()
    print(f"\nTotal results loaded: {len(all_results)}")
    
    # ALL JOINTS, NO NORM (most explainable)
    all_joints_no_norm = [r for r in all_results 
                          if ('all15' in r['label'] or r['n_joints'] >= 15) 
                          and '_none' in r['label']
                          and 'subjavg' not in r['label']
                          and 'taskavg' not in r['label']
                          and 'Ninapro' not in r['label']
                          and 'e6' not in r['label']]
    print_filtered_table(all_joints_no_norm, "ALL 15 JOINTS, NO NORM (most explainable)", lambda r: True, 20)
    plot_comparison(all_joints_no_norm, "All Joints, No Norm", "balance_all_joints_no_norm.png")
    
    # ALL JOINTS with normalization
    all_joints_norm = [r for r in all_results 
                       if ('all15' in r['label'] or r['n_joints'] >= 15)
                       and 'subjavg' not in r['label']
                       and 'taskavg' not in r['label']
                       and 'Ninapro' not in r['label']
                       and 'e6' not in r['label']
                       and '_none' not in r['label']]
    print_filtered_table(all_joints_norm, "ALL 15 JOINTS WITH NORM", lambda r: True, 20)
    plot_comparison(all_joints_norm, "All Joints with Norm", "balance_all_joints_norm.png")
    
    # HIGHEST PATTERN SIMILARITY (S1+S2 > 1.6)
    high_similarity = [r for r in all_results 
                       if r['rot_s1_sim'] + r['rot_s2_sim'] > 1.6
                       and 'subjavg' not in r['label']
                       and 'taskavg' not in r['label']
                       and 'e6' not in r['label']]
    print_filtered_table(high_similarity, "HIGHEST PATTERN SIMILARITY", lambda r: True, 20)
    plot_comparison(high_similarity, "High Pattern Similarity", "balance_high_sim.png")
    
    # BEST BALANCED: PC2 > 8%, good similarity, PC1+PC2 > 70%
    balanced = [r for r in all_results 
                if r['pc2_var'] > 0.08
                and r['rot_s1_sim'] > 0.70
                and 'subjavg' not in r['label']
                and 'taskavg' not in r['label']
                and 'e6' not in r['label']]
    print_filtered_table(balanced, "BEST BALANCED (PC2>8%, S1>0.7, PC1+PC2>70%)", 
                         lambda r: r['pc1_pc2_sum'] > 0.70, 25)
    plot_comparison(balanced, "Best Balanced Results", "balance_best_balanced.png")
    
    # ALL GRASP, ALL JOINTS (most like IIT Pisa protocol)
    all_grasp = [r for r in all_results 
                 if 'all_grasp' in r['label']
                 and 'subjavg' not in r['label']
                 and 'taskavg' not in r['label']
                 and ('all15' in r['label'] or r['n_joints'] >= 15)]
    print_filtered_table(all_grasp, "ALL GRASP TASKS, ALL 15 JOINTS", lambda r:True, 20)
    plot_comparison(all_grasp, "All Grasp, All Joints", "balance_all_grasp.png")
    
    # POWER + PRECISION, ALL JOINTS
    power_precision = [r for r in all_results 
                       if 'power_precision' in r['label']
                       and 'subjavg' not in r['label']
                       and 'taskavg' not in r['label']
                       and ('all15' in r['label'] or r['n_joints'] >= 15)]
    print_filtered_table(power_precision, "POWER+PRECISION, ALL 15 JOINTS", lambda r:True, 15)
    
    # ALL TASKS, ALL JOINTS
    all_tasks = [r for r in all_results 
                 if 'all_tasks' in r['label']
                 and 'subjavg' not in r['label']
                 and 'taskavg' not in r['label']
                 and ('all15' in r['label'] or r['n_joints'] >= 15)]
    print_filtered_table(all_tasks, "ALL TASKS, ALL 15 JOINTS", lambda r:True, 15)
    
    # ─── OVERALL RANKING ───
    print(f"\n{'='*100}")
    print(f"★ OVERALL TOP 30 (excluding single-subject and Ninapro)")
    print(f"{'='*100}")
    print(f"{'#':<4} {'Experiment':<55} {'PC1%':>7} {'PC2%':>7} {'Sum%':>7} {'S1':>8} {'S2':>8} {'Score':>7} {'N':>6}")
    print("-"*100)
    
    exclude_labels = ['e6_', 'ninapro', 'Ninapro', 'subjavg', 'taskavg']
    overall = [r for r in all_results if not any(e in r['label'] for e in exclude_labels)]
    overall.sort(key=score_balanced, reverse=True)
    
    for i, r in enumerate(overall[:30]):
        bs = score_balanced(r)
        star = "★" if i == 0 else " "
        flag = "!" if r['pc2_var'] > 0.08 else " "
        j_str = f"J{r['n_joints']}" if r['n_joints'] != 15 else ""
        print(f"{star}{i+1:<3}{flag} {r['label'][:55]:<55} {r['pc1_var']*100:>6.1f}% {r['pc2_var']*100:>6.1f}% {r['pc1_pc2_sum']*100:>6.1f}% {r['rot_s1_sim']:>+7.4f} {r['rot_s2_sim']:>+7.4f} {bs:>6.3f} {r['n_samples']:>6} {j_str}")
    
    # ─── RECOMMENDED CONFIGS ───
    print(f"\n{'='*100}")
    print(f"★ RECOMMENDED CONFIGURATIONS")
    print(f"{'='*100}")
    
    # Find the best in each explainable category
    categories = {
        "All Grasp, All 15 Joints, No Norm": lambda r: 'all_grasp' in r['label'] and 'all15' in r['label'] and '_none' in r['label'] and 'subjavg' not in r['label'],
        "All Tasks, All 15 Joints, No Norm": lambda r: 'all_tasks' in r['label'] and 'all15' in r['label'] and '_none' in r['label'] and 'subjavg' not in r['label'],
        "Power+Precision, All 15, No Norm": lambda r: 'power_precision' in r['label'] and 'all15' in r['label'] and '_none' in r['label'] and 'subjavg' not in r['label'],
        "Power Grip Only, All 15, No Norm": lambda r: 'power_only' in r['label'] and 'all15' in r['label'] and '_none' in r['label'],
        "Precision Only, All 15, No Norm": lambda r: 'precision_only' in r['label'] and 'all15' in r['label'] and '_none' in r['label'],
        "All Grasp, No Abd, No Norm": lambda r: 'all_grasp' in r['label'] and 'no_abd' in r['label'] and '_none' in r['label'],
        "Power+Prec, Flex10, Robust Norm": lambda r: 'power_precision' in r['label'] and 'flex10' in r['label'] and 'robust' in r['label'],
        "All Tasks, No Abd, No Norm": lambda r: 'all_tasks' in r['label'] and 'no_abd' in r['label'] and '_none' in r['label'],
    }
    
    for cat_name, cond_fn in categories.items():
        matching = [r for r in all_results if cond_fn(r)]
        if matching:
            matching.sort(key=score_balanced, reverse=True)
            best = matching[0]
            star = "★" if best['pc2_var'] > 0.08 and best['rot_s1_sim'] > 0.7 else " "
            print(f"{star} {cat_name:<45} {best['label'][:35]:<35} PC1={best['pc1_var']*100:>5.1f}% PC2={best['pc2_var']*100:>5.1f}% Sum={best['pc1_pc2_sum']*100:>5.1f}% S1={best['rot_s1_sim']:.3f} S2={best['rot_s2_sim']:.3f} N={best['n_samples']}")
    
    # ─── FINAL VERDICT ───
    print(f"\n{'='*100}")
    print(f"★ FINAL VERDICT: TOP 3 CONFIGS TO VISUALIZE")
    print(f"{'='*100}")
    
    # Strategy: find the configs that balance ALL criteria
    print(f"\nCriterion 1: PC1+PC2 ≥ 80%")
    print(f"Criterion 2: PC1 ~ 55-85%, PC2 ~ 10-25% (PC2 > 8%)")
    print(f"Criterion 3: S1 > 0.7, S2 > 0.5 (Pisa pattern)")
    
    candidates = [r for r in all_results 
                  if r['pc1_pc2_sum'] >= 0.75
                  and r['pc2_var'] > 0.08
                  and r['rot_s1_sim'] > 0.7
                  and 'subjavg' not in r['label']
                  and 'taskavg' not in r['label']
                  and 'e6_' not in r['label']]
    
    if not candidates:
        # Relax criteria
        candidates = [r for r in all_results 
                      if r['pc1_pc2_sum'] >= 0.70
                      and r['pc2_var'] > 0.07
                      and r['rot_s1_sim'] > 0.65
                      and 'subjavg' not in r['label']
                      and 'taskavg' not in r['label']
                      and 'e6_' not in r['label']]
    
    candidates.sort(key=score_balanced, reverse=True)
    
    for i, r in enumerate(candidates[:10]):
        print(f"  {i+1}. {r['label']}")
        print(f"     PC1={r['pc1_var']*100:.1f}% PC2={r['pc2_var']*100:.1f}% Sum={r['pc1_pc2_sum']*100:.1f}%")
        print(f"     S1={r['rot_s1_sim']:.4f} S2={r['rot_s2_sim']:.4f} N={r['n_samples']} J={r['n_joints']}")
        print()

if __name__ == '__main__':
    main()
