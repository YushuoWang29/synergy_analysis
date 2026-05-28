"""
Show best result in detail: v2_power_taskavg_static5.
Also generate detailed figures for comparison with Pisa/IIT paper.
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVE_DIR = 'PCA_results/experiments'
FIG_DIR = 'MyoKi_figures/experiments'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_JOINTS = 15
JOINT_NAMES = [
    "ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
    "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
    "RingMCP","RingPIP","MidRingAbd",
    "LitMCP","LitPIP","RingLitAbd"
]

PISA_S1 = np.ones(N_JOINTS) / np.linalg.norm(np.ones(N_JOINTS))
PISA_S2_RAW = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PISA_S2 = PISA_S2_RAW / np.linalg.norm(PISA_S2_RAW)
PISA_TARGET = np.column_stack([PISA_S1, PISA_S2])

# Load task info
def load_task_info():
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
grasp_types = {'cylindrical', 'spherical', 'hook', 'tripod', 'pinch', 'lumbrical'}
power_types = {'cylindrical', 'spherical', 'hook'}

ALL_GRASP_IDS = set()
POWER_GRASP_IDS = set()
for tid, acts in categories.items():
    has_grasp = any(a in grasp_types for a in acts)
    is_none = 'none_action' in acts
    if has_grasp and not is_none:
        ALL_GRASP_IDS.add(tid)
        if any(a in power_types for a in acts):
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


def main():
    participant_files = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
    valid_ps = [f'P{p:02d}' for p in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    files = [f for f in participant_files if any(p in os.path.basename(f) for p in valid_ps)]
    
    # =============================
    # COMPUTE BEST RESULT: v2_power_taskavg_static5
    # =============================
    print("Loading power trials...")
    power_trials = load_myoki_trials(files, POWER_GRASP_IDS)
    print(f"  {len(power_trials)} power trials")
    
    # Per-task averaging (Pisa style)
    by_task = {}
    for t, tid, subj in power_trials:
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(np.mean(t[-5:], axis=0))
    
    print(f"\nTasks with postures ({len(by_task)} tasks):")
    for tid in sorted(by_task.keys()):
        name = task_names.get(tid, f"Task {tid}")
        n = len(by_task[tid])
        print(f"  Task {tid:2d} ({name:25s}): {n} trials")
    
    task_postures = np.array([np.mean(v, axis=0) for v in by_task.values()])
    task_ids_list = list(by_task.keys())
    
    print(f"\nTask-average posture matrix: {task_postures.shape}")
    
    # === DO PCA ON TASK-AVERAGED POSTURES ===
    pca = PCA(n_components=min(15, task_postures.shape[0], task_postures.shape[1]))
    pca.fit(task_postures)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    
    pc1 = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-10)
    pc2 = pca.components_[1] / (np.linalg.norm(pca.components_[1]) + 1e-10)
    
    # Procrustes rotation
    PC = np.column_stack([pc1, pc2])
    R, _ = orthogonal_procrustes(PC, PISA_TARGET)
    rotated = PC @ R
    
    s1_sim = np.dot(rotated[:, 0], PISA_S1) / (np.linalg.norm(rotated[:, 0]) * np.linalg.norm(PISA_S1))
    s2_sim = np.dot(rotated[:, 1], PISA_S2) / (np.linalg.norm(rotated[:, 1]) * np.linalg.norm(PISA_S2))
    
    # =============================
    # DETAILED PRINT
    # =============================
    print(f"\n{'='*70}")
    print(f"BEST RESULT: Power Grasps, Task-Averaged Postures, Static Hold")
    print(f"{'='*70}")
    print(f"N tasks: {len(task_postures)}")
    print(f"N subjects: {len(files)}")
    print(f"N total trials: {len(power_trials)}")
    print(f"\nVariance explained:")
    print(f"  PC1: {evr[0]*100:.2f}%")
    print(f"  PC2: {evr[1]*100:.2f}%")
    print(f"  PC1+PC2: {(evr[0]+evr[1])*100:.2f}%")
    print(f"  90% variance: {int(np.searchsorted(cum, 0.9)+1)} PCs")
    print(f"  95% variance: {int(np.searchsorted(cum, 0.95)+1)} PCs")
    print(f"\nPisa target similarity (after Procrustes):")
    print(f"  Rot S1 → Pisa S1 (1:1:1:1:1 synergy): {s1_sim:.4f}")
    print(f"  Rot S2 → Pisa S2 (-2:-1:0:1:2 gradient): {s2_sim:.4f}")
    print(f"  Combined score: {s1_sim + s2_sim:.4f}")
    
    # Print joint weights
    print(f"\n{'Joint':<15} {'PC1':>10} {'PC2':>10} {'RotS1':>10} {'RotS2':>10} {'PisaS1':>10} {'PisaS2':>10}")
    print("-" * 75)
    for i in range(N_JOINTS):
        print(f"{JOINT_NAMES[i]:<15} {pc1[i]:>+10.4f} {pc2[i]:>+10.4f} "
              f"{rotated[i,0]:>+10.4f} {rotated[i,1]:>+10.4f} "
              f"{PISA_S1[i]:>+10.4f} {PISA_S2[i]:>+10.4f}")
    
    # =============================
    # GENERATE FIGURES
    # =============================
    print(f"\nGenerating figures...")
    
    # FIGURE 1: PC1, PC2, RotS1, RotS2 comparison
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 0.5, 0.5], width_ratios=[1, 1, 1, 1],
                          hspace=0.3, wspace=0.3)
    
    # Row 0: Bar plots
    for i, (vec, title, color_scheme) in enumerate([
        (pc1, f"PC1 ({evr[0]*100:.1f}%)", 'navy'),
        (pc2, f"PC2 ({evr[1]*100:.1f}%)", 'navy'),
        (rotated[:, 0], f"Rotated S1 (sim={s1_sim:.3f})", 'darkgreen'),
        (rotated[:, 1], f"Rotated S2 (sim={s2_sim:.3f})", 'darkgreen'),
    ]):
        ax = fig.add_subplot(gs[0, i])
        colors = ['steelblue' if v >= 0 else 'coral' for v in vec]
        edge = 'darkgreen' if i >= 2 else 'navy'
        ax.bar(range(N_JOINTS), vec, color=colors, edgecolor=edge, lw=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=6)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylim(-0.7, 0.7)
        ax.grid(axis='y', alpha=0.3)
    
    # Row 1: Finger-grouped bar plots (group bars by finger)
    finger_groups = [
        ('Thumb', [0, 1, 2, 3]),
        ('Index', [4, 5]),
        ('Middle', [6, 7, 8]),
        ('Ring', [9, 10, 11]),
        ('Little', [12, 13, 14]),
    ]
    
    for i, vec in enumerate([rotated[:, 0], rotated[:, 1], PISA_S1, PISA_S2]):
        ax = fig.add_subplot(gs[1, i])
        x_positions = []
        labels = []
        colors_list = []
        x = 0
        for fname, jids in finger_groups:
            for jid in jids:
                x_positions.append(x)
                labels.append(f"{fname[0]}")  # T, I, M, R, L
                colors_list.append('steelblue' if vec[jid] >= 0 else 'coral')
                x += 1
            x += 0.5  # gap between fingers
        ax.bar(x_positions, [vec[j] for j in range(N_JOINTS)], color=colors_list, 
               edgecolor='darkgreen', lw=0.5, width=0.7)
        ax.set_title(['Rot S1', 'Rot S2', 'Pisa S1', 'Pisa S2'][i], fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(-0.6, 0.6)
        ax.grid(axis='y', alpha=0.3)
    
    # Row 2: Score projection (tasks in PC space)
    ax = fig.add_subplot(gs[2, :3])
    scores = pca.transform(task_postures)[:, :2] @ R.T
    
    # Color by task type
    task_colors = []
    for tid in task_ids_list:
        acts = categories.get(tid, [])
        if any(a in {'cylindrical', 'spherical'} for a in acts):
            task_colors.append('red')       # power
        elif 'hook' in acts:
            task_colors.append('orange')    # hook
        elif 'tripod' in acts:
            task_colors.append('blue')      # precision
        elif 'pinch' in acts:
            task_colors.append('purple')
        elif 'lumbrical' in acts:
            task_colors.append('magenta')
        else:
            task_colors.append('gray')
    
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=task_colors, s=50, alpha=0.7, edgecolors='black')
    for i, tid in enumerate(task_ids_list):
        name = task_names.get(tid, "").split(" ")[0]
        ax.annotate(f"T{tid}", (scores[i, 0], scores[i, 1]), fontsize=6, alpha=0.7)
    
    # Add synergy directions
    ax.arrow(0, 0, 0.5, 0, head_width=0.05, head_length=0.02, fc='green', ec='green', lw=2, label='Syn1 (uniform)')
    ax.arrow(0, 0, 0, 0.5, head_width=0.02, head_length=0.05, fc='blue', ec='blue', lw=2, label='Syn2 (gradient)')
    
    ax.set_xlabel('Synergy 1 (open/close)', fontsize=10)
    ax.set_ylabel('Synergy 2 (thumb→little gradient)', fontsize=10)
    ax.set_title('Tasks in Synergy Space', fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # Add variance pie chart
    ax2 = fig.add_subplot(gs[2, 3])
    var_labels = [f'PC1\n{evr[0]*100:.1f}%', f'PC2\n{evr[1]*100:.1f}%', f'Others\n{100-(evr[0]+evr[1])*100:.1f}%']
    var_sizes = [evr[0]*100, evr[1]*100, 100-(evr[0]+evr[1])*100]
    explode = (0.05, 0.05, 0)
    wedges, texts, autotexts = ax2.pie(var_sizes, explode=explode, labels=var_labels,
                                        autopct='', colors=['#2196F3', '#4CAF50', '#E0E0E0'],
                                        startangle=90, textprops={'fontsize': 8})
    ax2.set_title('Variance Explained', fontsize=10, fontweight='bold')
    
    plt.suptitle('MyoKi Power Grasps → Pisa/IIT Synergies\n(Task-Averaged, Static Hold Phase, PCA + Procrustes Rotation)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'best_result_summary.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    
    # FIGURE 2: Detailed comparison with Pisa targets (like paper figures)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # PC1 vs Pisa S1
    ax = axes[0]
    x = np.arange(N_JOINTS)
    width = 0.25
    ax.bar(x - width, pc1, width, label=f'PC1 ({evr[0]*100:.1f}%)', color='steelblue', edgecolor='navy')
    ax.bar(x, rotated[:, 0], width, label=f'Rot S1 (sim={s1_sim:.3f})', color='limegreen', edgecolor='darkgreen')
    ax.bar(x + width, PISA_S1, width, label='Pisa S1 (target)', color='gold', edgecolor='darkorange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Weight')
    ax.set_title('Synergy 1: Open/Close (all fingers move together)', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # PC2 vs Pisa S2
    ax = axes[1]
    ax.bar(x - width, pc2, width, label=f'PC2 ({evr[1]*100:.1f}%)', color='steelblue', edgecolor='navy')
    ax.bar(x, rotated[:, 1], width, label=f'Rot S2 (sim={s2_sim:.3f})', color='limegreen', edgecolor='darkgreen')
    ax.bar(x + width, PISA_S2, width, label='Pisa S2 (target)', color='gold', edgecolor='darkorange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Weight')
    ax.set_title('Synergy 2: Thumb→Little Gradient (internal adjustment)', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Variance plot
    ax = axes[2]
    n_show = min(10, len(evr))
    bars = ax.bar(range(1, n_show+1), evr[:n_show]*100, color='steelblue', edgecolor='navy')
    bars[0].set_color('#2196F3')
    bars[1].set_color('#4CAF50')
    for i, v in enumerate(evr[:n_show]):
        ax.text(i+1, v*100+0.5, f'{v*100:.1f}%', ha='center', fontsize=9)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Variance Explained by Each PC', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, n_show+1))
    ax.set_xticklabels([f'PC{i}' for i in range(1, n_show+1)])
    ax.axhline((evr[0]+evr[1])*100, color='red', ls='--', lw=1, alpha=0.5, label=f'PC1+PC2={(evr[0]+evr[1])*100:.1f}%')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Best Result: Power Grasps, Task-Averaged Postures (Static Hold)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'best_result_detailed.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    
    # FIGURE 3: Heatmap comparison
    # Save a simple numpy file with the best PCA result for the main report
    np.savez(os.path.join(SAVE_DIR, 'best_result_power_taskavg.npz'),
             components=pca.components_,
             mean=pca.mean_,
             explained_variance=evr,
             pc1=pc1, pc2=pc2,
             rotated_s1=rotated[:, 0], rotated_s2=rotated[:, 1],
             rotation_matrix=R,
             s1_sim=s1_sim, s2_sim=s2_sim,
             joint_names=JOINT_NAMES,
             description="MyoKi power grasps, task-averaged static hold postures, PCA + Procrustes to Pisa targets")
    print(f"Saved: {os.path.join(SAVE_DIR, 'best_result_power_taskavg.npz')}")
    
    # Also save the P07 best result for comparison
    print(f"\n{'='*70}")
    print(f"SECOND BEST: P07 Individual (high variance + good synergy)")
    print(f"{'='*70}")
    
    p07_trials = [x for x in power_trials if 'P07' in x[2]]
    p07_static10 = np.array([np.mean(t[-10:], axis=0) for t, tid, _ in p07_trials])
    
    pca2 = PCA(n_components=2)
    pca2.fit(p07_static10)
    evr2 = pca2.explained_variance_ratio_
    pc1_2 = pca2.components_[0] / np.linalg.norm(pca2.components_[0])
    pc2_2 = pca2.components_[1] / np.linalg.norm(pca2.components_[1])
    
    PC2 = np.column_stack([pc1_2, pc2_2])
    R2, _ = orthogonal_procrustes(PC2, PISA_TARGET)
    rotated2 = PC2 @ R2
    s1_sim2 = np.dot(rotated2[:, 0], PISA_S1) / (np.linalg.norm(rotated2[:, 0]) * np.linalg.norm(PISA_S1))
    s2_sim2 = np.dot(rotated2[:, 1], PISA_S2) / (np.linalg.norm(rotated2[:, 1]) * np.linalg.norm(PISA_S2))
    
    print(f"Participant P07: {len(p07_trials)} power trials")
    print(f"Variance: PC1={evr2[0]*100:.1f}%, PC2={evr2[1]*100:.1f}%, Sum={(evr2[0]+evr2[1])*100:.1f}%")
    print(f"Rot S1 sim: {s1_sim2:.4f}")
    print(f"Rot S2 sim: {s2_sim2:.4f}")
    
    np.savez(os.path.join(SAVE_DIR, 'best_result_P07.npz'),
             components=pca2.components_,
             mean=pca2.mean_,
             explained_variance=evr2,
             pc1=pc1_2, pc2=pc2_2,
             rotated_s1=rotated2[:, 0], rotated_s2=rotated2[:, 1],
             rotation_matrix=R2,
             s1_sim=s1_sim2, s2_sim=s2_sim2,
             joint_names=JOINT_NAMES,
             description="MyoKi P07 power grasps, static10, PCA + Procrustes")
    
    print(f"\nDone! Best results saved to {SAVE_DIR} and {FIG_DIR}")

if __name__ == '__main__':
    main()
