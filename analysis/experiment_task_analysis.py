"""
Task Analysis - Understand task categories and experiment with different PCA strategies.
"""
import os, sys, glob
import numpy as np
import scipy.io as sio
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

N_JOINTS = 15  # MyoKi hand-only, palmar arch removed

# ============================================================
# 1. Load task categories
# ============================================================
def load_task_categories():
    wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
    ws = wb.active
    categories = {}
    task_names = {}
    action_cols = {
        'cylindrical': 11, 'spherical': 12, 'hook': 13, 'tripod': 14,
        'pinch': 15, 'lumbrical': 16, 'complex': 17, 'none_action': 18,
    }
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

categories, task_names = load_task_categories()

print("=" * 80)
print("ALL TASK CATEGORIES")
print("=" * 80)
grasp_types = {'cylindrical', 'spherical', 'hook', 'tripod', 'pinch', 'lumbrical'}
grasp_tasks = []
non_grasp_tasks = []

for tid in sorted(categories.keys()):
    actions = categories[tid]
    has_grasp = any(a in grasp_types for a in actions)
    is_none = 'none_action' in actions
    is_grasp = has_grasp and not is_none
    acts_str = "+".join(actions)
    tag = "GRASP" if is_grasp else "OTHER"
    if is_grasp:
        grasp_tasks.append(tid)
    else:
        non_grasp_tasks.append(tid)
    print(f"  Task {tid:2d}: {task_names.get(tid,'?'):55s} [{acts_str:30s}] {tag}")

print(f"\nGrasp tasks ({len(grasp_tasks)}): {grasp_tasks}")
print(f"Non-grasp tasks ({len(non_grasp_tasks)}): {non_grasp_tasks}")

# ============================================================
# 2. From note.md: the "put back" tasks and "grasp only" tasks
# ============================================================
# From note.md:
# PCA now only uses tasks 1-6 (cup), 9-14 (bottle), 16-21 (small cup), 
# 28-33 (ball), 34-39 (book), 40-45 (bag) — 36 grasp tasks
# Each has 5 variations = 180 trials/participant

# BUT the "put back" versions (tasks 2,4,6,10,12,14,17,19,21,29,31,33,35,37,39,41,43,45)
# should maybe be excluded since they're reverse actions.

# Let's see the pattern:
# Cup: 1(Pickup) 2(Put back) 3(Pickup) 4(Put back) 5(Pickup) 6(Put back) 7(Wipe) 8(Screw) 
# Let's print task descriptions grouped by object

print("\n" + "=" * 80)
print("TASKS GROUPED BY OBJECT (from Task ID naming convention)")
print("=" * 80)

# Tasks are organized by object groups of 5
current_group = []
for tid in range(1, 75):
    name = task_names.get(tid, '?')
    # Extract the "action" part - everything after the comma or after the colon
    parts = name.split(',')
    action_only = parts[0].strip() if len(parts) > 1 else name
    print(f"  Task {tid:2d}: {name}")
    if tid % 5 == 0:
        print()

# ============================================================
# 3. Check what the current preprocessed data looks like
# ============================================================
print("\n" + "=" * 80)
print("PREPROCESSED DATA CHECK (first participant)")
print("=" * 80)

mat = sio.loadmat('MyoKi Dataset/preprocessed/P01_preprocessed.mat')
angles = mat['angles']
trial_tasks = mat['trial_tasks'].flatten().astype(int)
num_trials = int(mat['num_trials'])
print(f"Angles shape: {angles.shape}")
print(f"Num trials: {num_trials}")
print(f"Frames/trial: {angles.shape[0] // num_trials}")
print(f"Unique tasks: {sorted(np.unique(trial_tasks))}")

# Check how many trials per task
unique_tasks, counts = np.unique(trial_tasks, return_counts=True)
print(f"\nTrials per task:")
for t, c in zip(unique_tasks, counts):
    name = task_names.get(t, '?')
    print(f"  Task {t:2d} ({name}): {c} trials")

# ============================================================
# 4. Define task sets for experiments
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT DESIGN")
print("=" * 80)

# The Pisa/IIT SoftHand 2 paper says:
# Synergy 1 (s1): all fingers close together -> uniform [1,...,1]/norm
# Synergy 2 (s2): gradient from thumb(+) to little(-) -> [+2,...,-2]/norm

# Pisa SoftHand 2 has 19 joints (5 per finger, but missing some like thumb CMC)
# They found: s1 ~80%, s2 ~10% of variance
# They used: reach-to-grasp movements, not static poses

# Key insight from Santello 1998:
# - Used static grasp postures (holding objects)
# - 15 joint angles
# - PC1: hand opening/closing (all joints co-vary) ~60%
# - PC2: MCP/PIP differential (fingers curl without full closure) ~15%
# - PC3: thumb opposition vs finger abduction

# MyoKi dataset issues:
# - Many non-grasp tasks (hammering, sawing, typing, etc.)
# - For grasp tasks, includes both "pick up" AND "put back" (reverse motion)
# - Dynamic tasks have high temporal variance

# HYPOTHESES:
# 1. Filtering to STATIC GRASP ONLY should give more Santello-like PCs
# 2. Using only "pick up" tasks (not "put back") should be cleaner
# 3. Using trial means (average of 100 frames) reduces noise
# 4. Using only the static hold phase (last frames) is best

# Let's check what the current implementation does:
# pca_myoki.py: uses last 25% of frames for static posture, 
#   filters to grasp+pickup tasks only (pickup_grasp_task_ids)

print("\nCurrent pca_myoki.py grasp+pickup task filter:")
pickup_grasp_task_ids = {1, 3, 5, 9, 11, 13, 16, 18, 20, 28, 30, 32, 34, 36, 38, 40, 42, 44}
print(f"  IDs: {sorted(pickup_grasp_task_ids)}")
for tid in sorted(pickup_grasp_task_ids):
    print(f"    Task {tid:2d}: {task_names.get(tid, '?')}")

# The note.md said "36 grasp tasks" but current filter has only 18 tasks.
# Let's check what all the grasp tasks look like.
print("\n\nAll grasp tasks from Excel:")
for tid in sorted(grasp_tasks):
    actions = categories[tid]
    is_pickup = tid in pickup_grasp_task_ids
    tag = "PICKUP" if is_pickup else "PUTBACK/OTHER"
    print(f"  Task {tid:2d} [{tag:>10}]: {task_names.get(tid, '?'):55s} actions={actions}")
