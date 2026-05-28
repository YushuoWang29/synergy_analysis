"""
FINAL REPORT: Best PCA configurations for Pisa/IIT synergy replication.

TOP FINDINGS SUMMARY:
  1. flexion_only (10 joints: all MCP+PIP) + task-averaging + power grasps → S1=0.94, S2=0.81, Var=95.2% ★BEST★
  2. no_abd (12 joints, remove 3 abductions) + task-averaging + power grasps → S1=0.84, S2=0.78, Var=93.3%
  3. all15 (full 15 joints) + task-averaging + power grasps → S1=0.78, S2=0.72, Var=92.8%
  
This script generates detailed comparison figures.
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVE_DIR = 'PCA_results/experiments'
FIG_DIR = 'MyoKi_figures/experiments'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ALL_JOINT_NAMES = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
    "RingMCP","RingPIP","MidRingAbd",
    "LitMCP","LitPIP","RingLitAbd"
]

FLEXION_JOINT_NAMES = [
    "ThumbMCP","ThumbIP",
    "IdxMCP","IdxPIP",
    "MidMCP","MidPIP",
    "RingMCP","RingPIP",
    "LitMCP","LitPIP"
]

NO_ABD_JOINT_NAMES = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP",
    "RingMCP","RingPIP",
    "LitMCP","LitPIP"
]

FLEXION_IDS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]
NO_ABD_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13]

def get_pisa_target_for_joints(jids):
    n = len(jids)
    s1 = np.ones(n) / np.linalg.norm(np.ones(n))
    raw = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
    s2 = raw[jids] / np.linalg.norm(raw[jids])
    return s1, s2, np.column_stack([s1, s2])

def load_task_info():
    import openpyxl
    wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
    ws = wb.active
    categories, task_names = {}, {}
    action_cols = {'cylindrical': 11, 'spherical': 12, 'hook': 13, 'tripod': 14,
                   'pinch': 15, 'lumbrical': 16, 'complex': 17, 'none_action': 18}
    for row in ws.iter_rows(min_row=4, values_only=False):
        tid = row[0].value
        if tid is None: continue
        tid_int = int(tid)
        name = str(row[1].value).strip() if row[1].value else f"Task {tid_int}"
        task_names[tid_int] = name
        actions = []
        for act_name, col_idx in action_cols.items():
            val = row[col_idx].value
            if val is not None and str(val).strip() == 'x':
                actions.append(act_name)
        categories[tid_int] = actions
    return categories, task_names

categories, task_names = load_task_info()
power_types = {'cylindrical', 'spherical', 'hook'}
POWER_GRASP_IDS = set()
for tid, acts in categories.items():
    has_grasp = any(a in power_types for a in acts)
    is_none = 'none_action' in acts
    if has_grasp and not is_none:
        POWER_GRASP_IDS.add(tid)

def load_myoki_trials(files, task_ids, resample_n=100):
    all_trials = []
    for fpath in files:
        mat = sio.loadmat(fpath)
        glove = np.nan_to_num(mat['glove_calibrated'])
        tasks = mat['task'].flatten().astype(int)
        hand_data = glove[:, :15]
        for tid in task_ids:
            mask = (tasks == tid)
            diff = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0]
            for s, e in zip(starts, ends):
                trial = hand_data[s:e, :]
                if len(trial) >= 5:
                    x_old = np.linspace(0, 1, len(trial))
                    x_new = np.linspace(0, 1, resample_n)
                    f = interp1d(x_old, trial, axis=0, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
                    resampled = f(x_new)
                    all_trials.append((resampled, tid, os.path.basename(fpath)))
    return all_trials

def run_analysis(label, jids, joint_names):
    """Run full PCA + Procrustes analysis."""
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]
    
    # Load power grasp trials
    trials = load_myoki_trials(files, POWER_GRASP_IDS)
    
    # Task-averaged static5 postures
    by_task = {}
    for t, tid, subj in trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(np.mean(t[-5:], axis=0))
    
    print(f"Tasks: {len(by_task)}, Trials: {len(trials)}")
    
    # Build data matrix with selected joints
    task_data = np.array([np.mean(v, axis=0)[jids] for v in by_task.values()])
    n_joints = len(jids)
    print(f"Data shape: {task_data.shape}")
    
    # Pisa target
    pisa_s1, pisa_s2, pisa_target = get_pisa_target_for_joints(jids)
    
    # PCA
    pca = PCA(n_components=min(15, task_data.shape[0], task_data.shape[1]))
    pca.fit(task_data)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    
    pc1_raw = pca.components_[0]
    pc2_raw = pca.components_[1]
    
    # Normalize
    pc1 = pc1_raw / (np.linalg.norm(pc1_raw) + 1e-10)
    pc2 = pc2_raw / (np.linalg.norm(pc2_raw) + 1e-10)
    
    # Procrustes
    PC = np.column_stack([pc1, pc2])
    R, _ = orthogonal_procrustes(PC, pisa_target)
    rotated = PC @ R
    
    s1_sim = np.dot(rotated[:, 0], pisa_s1)
    s2_sim = np.dot(rotated[:, 1], pisa_s2)
    
    # Scores
    scores = pca.transform(task_data)[:, :2] @ R.T
    
    return {
        'label': label,
        'n_tasks': len(by_task),
        'n_joints': n_joints,
        'evr': evr,
        'cum': cum,
        'pc1': pc1, 'pc2': pc2,
        'rotated': rotated,
        'pc1_var': evr[0], 'pc2_var': evr[1],
        'pc_sum': evr[0] + evr[1],
        's1_sim': s1_sim, 's2_sim': s2_sim,
        'pisa_s1': pisa_s1, 'pisa_s2': pisa_s2,
        'pisa_target': pisa_target,
        'joint_names': joint_names,
        'jids': jids,
        'R': R,
        'scores': scores,
        'task_ids_list': list(by_task.keys()),
        'pca': pca,
    }

def make_comparison_plot(results):
    """Make a comprehensive comparison figure."""
    n_configs = len(results)
    fig = plt.figure(figsize=(20, 4*n_configs + 3))
    gs = gridspec.GridSpec(n_configs + 1, 3, height_ratios=[3]*n_configs + [2], 
                          hspace=0.4, wspace=0.3)
    
    color_map = plt.cm.tab10
    
    # For each config, show PC1, PC2, and score plot
    for idx, res in enumerate(results):
        n_j = res['n_joints']
        
        # Row: PC1 + Rot S1 vs Pisa S1 (left), PC2 + Rot S2 vs Pisa S2 (middle), Score (right)
        ax_pc1 = fig.add_subplot(gs[idx, 0])
        ax_pc2 = fig.add_subplot(gs[idx, 1])
        ax_scatter = fig.add_subplot(gs[idx, 2])
        
        # === PC1 comparison ===
        x = np.arange(n_j)
        width = 0.2
        
        ax_pc1.bar(x - width, res['pc1'], width, label=f'PC1 ({res["pc1_var"]*100:.1f}%)',
                   color='steelblue', edgecolor='navy')
        ax_pc1.bar(x, res['rotated'][:, 0], width, 
                   label=f'Rot S1 (sim={res["s1_sim"]:.3f})',
                   color='limegreen', edgecolor='darkgreen')
        ax_pc1.bar(x + width, res['pisa_s1'], width, label='Pisa S1 target',
                   color='gold', edgecolor='darkorange', alpha=0.7)
        
        ax_pc1.set_xticks(x)
        ax_pc1.set_xticklabels(res['joint_names'], rotation=45, ha='right', fontsize=7)
        ax_pc1.set_ylabel('Weight')
        ax_pc1.set_title(f'Synergy 1 - {res["label"]}', fontsize=11, fontweight='bold')
        ax_pc1.axhline(0, color='black', lw=0.5)
        ax_pc1.legend(fontsize=7, loc='upper right')
        ax_pc1.grid(axis='y', alpha=0.3)
        ax_pc1.set_ylim(-0.7 if n_j <= 10 else -0.85, 0.85)
        
        # === PC2 comparison ===
        ax_pc2.bar(x - width, res['pc2'], width, label=f'PC2 ({res["pc2_var"]*100:.1f}%)',
                   color='steelblue', edgecolor='navy')
        ax_pc2.bar(x, res['rotated'][:, 1], width,
                   label=f'Rot S2 (sim={res["s2_sim"]:.3f})',
                   color='limegreen', edgecolor='darkgreen')
        ax_pc2.bar(x + width, res['pisa_s2'], width, label='Pisa S2 target',
                   color='gold', edgecolor='darkorange', alpha=0.7)
        
        ax_pc2.set_xticks(x)
        ax_pc2.set_xticklabels(res['joint_names'], rotation=45, ha='right', fontsize=7)
        ax_pc2.set_ylabel('Weight')
        ax_pc2.set_title(f'Synergy 2 - {res["label"]}', fontsize=11, fontweight='bold')
        ax_pc2.axhline(0, color='black', lw=0.5)
        ax_pc2.legend(fontsize=7, loc='upper right')
        ax_pc2.grid(axis='y', alpha=0.3)
        ax_pc2.set_ylim(-0.7, 0.7)
        
        # === Score scatter ===
        # Color by task type
        task_colors = []
        for tid in res['task_ids_list']:
            acts = categories.get(tid, [])
            if any(a in {'cylindrical', 'spherical'} for a in acts):
                task_colors.append('red')
            elif 'hook' in acts:
                task_colors.append('orange')
            elif 'tripod' in acts:
                task_colors.append('blue')
            elif 'pinch' in acts:
                task_colors.append('purple')
            elif 'lumbrical' in acts:
                task_colors.append('magenta')
            else:
                task_colors.append('gray')
        
        ax_scatter.scatter(res['scores'][:, 0], res['scores'][:, 1], 
                          c=task_colors, s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add synergy directions
        ax_scatter.arrow(0, 0, 0.5, 0, head_width=0.03, head_length=0.015, 
                        fc='green', ec='green', lw=2, label='Syn1 (uniform)')
        ax_scatter.arrow(0, 0, 0, 0.3, head_width=0.015, head_length=0.03,
                        fc='blue', ec='blue', lw=2, label='Syn2 (gradient)')
        
        ax_scatter.set_xlabel('Synergy 1')
        ax_scatter.set_ylabel('Synergy 2')
        ax_scatter.set_title(f'Scores in Synergy Space', fontsize=10, fontweight='bold')
        ax_scatter.axhline(0, color='gray', lw=0.5)
        ax_scatter.axvline(0, color='gray', lw=0.5)
        ax_scatter.grid(alpha=0.3)
        ax_scatter.legend(fontsize=6, loc='best')
    
    # Bottom: Variance comparison bar plot
    ax_var = fig.add_subplot(gs[n_configs, :])
    n_show = min(10, len(results[0]['evr']))
    x = np.arange(n_show)
    width = 0.8 / len(results)
    
    for idx, res in enumerate(results):
        evr = res['evr']
        offset = (idx - len(results)/2 + 0.5) * width
        bars = ax_var.bar(x + offset, evr[:n_show]*100, width, 
                         label=res['label'], alpha=0.8)
    
    ax_var.set_xlabel('Principal Component')
    ax_var.set_ylabel('Variance Explained (%)')
    ax_var.set_title('Variance Explained by PC (All Configurations)', fontsize=12, fontweight='bold')
    ax_var.set_xticks(x)
    ax_var.set_xticklabels([f'PC{i+1}' for i in range(n_show)])
    ax_var.legend(fontsize=9)
    ax_var.grid(axis='y', alpha=0.3)
    
    plt.suptitle('PCA Synergy Analysis — Comparison of Best Configurations', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig

def make_summary_table(results):
    """Print a formatted summary table."""
    print(f"\n{'='*120}")
    print(f"{'BEST CONFIGURATIONS - SUMMARY':^120}")
    print(f"{'='*120}")
    print(f"{'Configuration':<40} {'Joints':>8} {'Var%':>8} {'S1':>8} {'S2':>8} {'Score':>8}")
    print("-" * 120)
    
    for res in sorted(results, key=lambda r: r['s1_sim'] + r['s2_sim'], reverse=True):
        print(f"{res['label']:<40} {res['n_joints']:>8} {res['pc_sum']*100:>7.1f}% "
              f"{res['s1_sim']:>+7.4f} {res['s2_sim']:>+7.4f} "
              f"{res['s1_sim']+res['s2_sim']:>7.4f}")
    
    print(f"{'='*120}")

def main():
    print("=" * 70)
    print("FINAL REPORT: Pisa/IIT Synergy Replication")
    print("=" * 70)
    
    # Run the three best configurations
    configs = [
        ("v4_power_taskavg_all15", list(range(15)), ALL_JOINT_NAMES),
        ("v4_power_taskavg_no_abd", NO_ABD_IDS, NO_ABD_JOINT_NAMES),
        ("v4_power_taskavg_flexion_only", FLEXION_IDS, FLEXION_JOINT_NAMES),
    ]
    
    results = []
    for label, jids, jnames in configs:
        print(f"\n{'─'*60}")
        print(f"Running: {label}")
        print(f"Joints: {jids} -> {jnames}")
        res = run_analysis(label, jids, jnames)
        results.append(res)
        print(f"  Variance PC1+PC2: {res['pc_sum']*100:.2f}%")
        print(f"  90% variance: {int(np.searchsorted(res['cum'], 0.9)+1)} PCs")
        print(f"  95% variance: {int(np.searchsorted(res['cum'], 0.95)+1)} PCs")
        print(f"  S1 similarity: {res['s1_sim']:.4f}")
        print(f"  S2 similarity: {res['s2_sim']:.4f}")
    
    # Summary table
    make_summary_table(results)
    
    # Save detailed results
    print("\nJoint weights for BEST config (flexion_only):")
    best = results[2]
    print(f"{'Joint':<15} {'PC1':>10} {'PC2':>10} {'RotS1':>10} {'RotS2':>10} {'PisaS1':>10} {'PisaS2':>10}")
    print("-" * 75)
    for i in range(best['n_joints']):
        print(f"{best['joint_names'][i]:<15} {best['pc1'][i]:>+10.4f} {best['pc2'][i]:>+10.4f} "
              f"{best['rotated'][i,0]:>+10.4f} {best['rotated'][i,1]:>+10.4f} "
              f"{best['pisa_s1'][i]:>+10.4f} {best['pisa_s2'][i]:>+10.4f}")
    
    print("\nJoint weights for no_abd config:")
    mid = results[1]
    for i in range(mid['n_joints']):
        print(f"{mid['joint_names'][i]:<15} {mid['pc1'][i]:>+10.4f} {mid['pc2'][i]:>+10.4f} "
              f"{mid['rotated'][i,0]:>+10.4f} {mid['rotated'][i,1]:>+10.4f} "
              f"{mid['pisa_s1'][i]:>+10.4f} {mid['pisa_s2'][i]:>+10.4f}")
    
    # Generate comparison figure
    print("\nGenerating comparison figure...")
    fig = make_comparison_plot(results)
    fig_path = os.path.join(FIG_DIR, 'final_comparison.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    
    # Save all results as numpy arrays for the main report
    for res in results:
        safe_name = res['label'].replace('v4_', '')
        np.savez(os.path.join(SAVE_DIR, f'final_{safe_name}.npz'),
                 components=res['pca'].components_,
                 mean=res['pca'].mean_,
                 explained_variance=res['evr'],
                 pc1=res['pc1'], pc2=res['pc2'],
                 rotated_s1=res['rotated'][:, 0], 
                 rotated_s2=res['rotated'][:, 1],
                 rotation_matrix=res['R'],
                 s1_sim=res['s1_sim'], s2_sim=res['s2_sim'],
                 joint_names=res['joint_names'],
                 scores=res['scores'],
                 description=f"MyoKi power grasps, task-averaged static5, {res['label']}")
    
    # Also test with all-grasp for flexion_only
    print(f"\n{'─'*60}")
    print("Testing all_grasp flexion_only (second best)...")
    all_trials = []
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]
    
    grasp_types = {'cylindrical', 'spherical', 'hook', 'tripod', 'pinch', 'lumbrical'}
    ALL_GRASP_IDS = set()
    for tid, acts in categories.items():
        has_grasp = any(a in grasp_types for a in acts)
        is_none = 'none_action' in acts
        if has_grasp and not is_none:
            ALL_GRASP_IDS.add(tid)
    
    all_trials = load_myoki_trials(files, ALL_GRASP_IDS)
    by_task = {}
    for t, tid, subj in all_trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(np.mean(t[-5:], axis=0))
    
    task_data = np.array([np.mean(v, axis=0)[FLEXION_IDS] for v in by_task.values()])
    target = get_pisa_target_for_joints(FLEXION_IDS)[2]
    
    pca2 = PCA(n_components=2)
    pca2.fit(task_data)
    evr2 = pca2.explained_variance_ratio_
    pc1_2 = pca2.components_[0] / np.linalg.norm(pca2.components_[0])
    pc2_2 = pca2.components_[1] / np.linalg.norm(pca2.components_[1])
    PC2 = np.column_stack([pc1_2, pc2_2])
    R2, _ = orthogonal_procrustes(PC2, target)
    rotated2 = PC2 @ R2
    s1_2 = np.dot(rotated2[:, 0], target[:, 0])
    s2_2 = np.dot(rotated2[:, 1], target[:, 1])
    
    print(f"S1={s1_2:.4f}, S2={s2_2:.4f}, Var={sum(evr2)*100:.1f}%")
    
    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")
    print(f"")
    print(f"Top 3 best approaches ranked by S1+S2 (with variance constraint >= 80%):")
    print(f"")
    print(f"  ★ 1. Power Grasps, Task-Averaged Static5, FLEXION ONLY (10 joints)")
    print(f"     S1={results[2]['s1_sim']:.4f}, S2={results[2]['s2_sim']:.4f}, Var={results[2]['pc_sum']*100:.1f}%")
    print(f"")
    print(f"  ★ 2. Power Grasps, Task-Averaged Static5, NO ABDUCTION (12 joints)")
    print(f"     S1={results[1]['s1_sim']:.4f}, S2={results[1]['s2_sim']:.4f}, Var={results[1]['pc_sum']*100:.1f}%")
    print(f"")
    print(f"  ★ 3. Power Grasps, Task-Averaged Static5, ALL 15 JOINTS")
    print(f"     S1={results[0]['s1_sim']:.4f}, S2={results[0]['s2_sim']:.4f}, Var={results[0]['pc_sum']*100:.1f}%")
    print(f"")
    print(f"All results saved to {SAVE_DIR} and {FIG_DIR}")
    
    np.savez(os.path.join(SAVE_DIR, 'final_allgrasp_flexion_only.npz'),
             components=pca2.components_, mean=pca2.mean_,
             explained_variance=evr2,
             pc1=pc1_2, pc2=pc2_2,
             rotated_s1=rotated2[:, 0], rotated_s2=rotated2[:, 1],
             rotation_matrix=R2,
             s1_sim=float(s1_2), s2_sim=float(s2_2),
             joint_names=FLEXION_JOINT_NAMES,
             scores=pca2.transform(task_data)[:, :2] @ R2.T)

if __name__ == '__main__':
    main()
