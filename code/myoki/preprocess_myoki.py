"""
MyoKi dataset preprocessing — wrist columns removed.

Pipeline:
1. Load raw .mat file (18-ch CyberGlove)
2. Strip wrist channels 16,17 → keep 16 hand-only channels
3. Segment by task label (task 0 = rest/transition, tasks 1-74 = activities)
4. Remove static frames (velocity threshold)
5. Resample each trial to a fixed number of frames
6. Save processed data (16-D) for PCA

Usage:
    python preprocess_myoki.py
    python preprocess_myoki.py --participants P01 P02 P03
    python preprocess_myoki.py --target_frames 100 --velocity_thresh 0.5
"""

import os, sys, glob, argparse
import numpy as np
import scipy.io as sio

DEFAULT_RAW_DIR = 'MyoKi Dataset/raw'
DEFAULT_OUT_DIR = 'MyoKi Dataset/preprocessed'
TARGET_FRAMES = 100
VELOCITY_THRESHOLD = 0.5  # deg/frame


def load_task_names():
    """Load task ID → name mapping from Task_categorization.xlsx."""
    task_names = {0: 'Rest / Transition'}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(os.path.join('MyoKi Dataset', 'Task_categorization.xlsx'))
        ws = wb.active
        for row in ws.iter_rows(min_row=4, values_only=True):   # data starts at row 4
            tid = row[0]
            name = row[1]
            if tid is not None and name is not None:
                task_names[int(tid)] = str(name).replace("'", "").strip()
    except Exception as e:
        print(f"  (Could not load task names: {e})")
    return task_names


TASK_NAMES = load_task_names()


def preprocess_participant(mat_path, out_dir,
                           target_frames=TARGET_FRAMES,
                           vel_thresh=VELOCITY_THRESHOLD,
                           exclude_rest=True):
    """
    Preprocess a single participant file.

    Returns dict with keys:
        'angles': (N, 16)  hand-only joint angles
        'num_trials': int
        'tasks_used': list
        'trial_task_ids': which task each trial belongs to (for OSD)
        'info': str
    """
    basename = os.path.basename(mat_path).replace('.mat', '')
    print(f"\n{'='*60}")
    print(f"Processing {basename}...")

    mat = sio.loadmat(mat_path)
    glove_calib = np.nan_to_num(mat['glove_calibrated'])  # (N, 18)
    tasks = mat['task'].flatten().astype(int)

    # --- Strip wrist: keep only first 16 channels ---
    hand_data = glove_calib[:, :16]  # (N, 16)

    print(f"  Total frames: {glove_calib.shape[0]}  →  hand-only: {hand_data.shape}")
    print(f"  Stripped wrist channels (16,17). Remaining: 16 hand joints.")

    processed_trials = []
    trial_task_ids = []
    tasks_used = set()

    unique_tasks = sorted(np.unique(tasks))
    if exclude_rest:
        unique_tasks = [t for t in unique_tasks if t != 0]
    print(f"  Number of tasks (excluding rest): {len(unique_tasks)}")

    for task_id in unique_tasks:
        task_mask = (tasks == task_id)
        diff = np.diff(task_mask.astype(int), prepend=0, append=0)
        starts = np.where(diff > 0)[0]
        ends   = np.where(diff < 0)[0]

        for s, e in zip(starts, ends):
            trial = hand_data[s:e, :]
            if len(trial) < 5:
                continue

            # Static-frame removal
            grads = np.gradient(trial, axis=0)
            vel_norm = np.linalg.norm(grads, axis=1)
            moving = trial[vel_norm > vel_thresh]
            if len(moving) < 10:
                moving = trial  # fallback

            # Resample to uniform length
            x_old = np.linspace(0, 1, len(moving))
            x_new = np.linspace(0, 1, target_frames)
            from scipy.interpolate import interp1d
            f_interp = interp1d(x_old, moving, axis=0, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
            resampled = f_interp(x_new)

            processed_trials.append(resampled)
            trial_task_ids.append(int(task_id))
            tasks_used.add(int(task_id))

    if not processed_trials:
        print(f"  WARNING: No trials extracted for {basename}")
        return None

    final_matrix = np.vstack(processed_trials)          # (total_N, 16)
    num_trials   = len(processed_trials)
    trial_arr    = np.array(trial_task_ids, dtype=int)

    save_name = f"{basename}_preprocessed.mat"
    save_path = os.path.join(out_dir, save_name)
    sio.savemat(save_path, {
        'angles': final_matrix,
        'num_trials': num_trials,
        'tasks_used': np.array(sorted(tasks_used)),
        'trial_tasks': trial_arr,
        'info': (f'MyoKi preprocessed (hand-only 16ch, wrist removed). '
                 f'Static removal thresh={vel_thresh}, resampled to {target_frames} '
                 f'frames/trial, {num_trials} trials.')
    })

    tn = TASK_NAMES
    print(f"  Saved: {save_path}")
    print(f"  Trials: {num_trials}, frames: {final_matrix.shape[0]}, dims: {final_matrix.shape[1]}")
    print(f"  Tasks: {sorted(tasks_used)}")
    for tid in sorted(tasks_used):
        print(f"    Task {tid}: {tn.get(tid, '?')}")

    return {'angles': final_matrix, 'num_trials': num_trials,
            'tasks_used': sorted(tasks_used), 'trial_tasks': trial_arr}


def main(args=None):
    parser = argparse.ArgumentParser(description='Preprocess MyoKi (hand-only 16ch)')
    parser.add_argument('--raw_dir', default=DEFAULT_RAW_DIR)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--participants', nargs='+', default=None)
    parser.add_argument('--target_frames', type=int, default=TARGET_FRAMES)
    parser.add_argument('--velocity_thresh', type=float, default=VELOCITY_THRESHOLD)

    args = parser.parse_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.participants:
        mat_files = [os.path.join(args.raw_dir, f'{p}.mat') for p in args.participants]
        mat_files = [f for f in mat_files if os.path.exists(f)]
    else:
        mat_files = sorted(glob.glob(os.path.join(args.raw_dir, 'P*.mat')))

    if not mat_files:
        print(f"No .mat files found in {args.raw_dir}")
        return

    print(f"Found {len(mat_files)} participant(s):")
    for f in mat_files:
        print(f"  {os.path.basename(f)}")

    all_angles = []
    all_tasks  = []
    total_trials = 0

    for f in mat_files:
        result = preprocess_participant(
            f, args.out_dir,
            target_frames=args.target_frames,
            vel_thresh=args.velocity_thresh,
            exclude_rest=True)
        if result is not None:
            all_angles.append(result['angles'])
            all_tasks.append(result['trial_tasks'])
            total_trials += result['num_trials']

    if all_angles:
        combined   = np.vstack(all_angles)
        comb_tasks = np.concatenate(all_tasks)
        comb_path  = os.path.join(args.out_dir, 'all_participants_preprocessed.mat')
        sio.savemat(comb_path, {
            'angles': combined,
            'num_trials': total_trials,
            'trial_tasks': comb_tasks,
            'info': (f'Combined MyoKi hand-only (16ch, wrist removed): '
                     f'{len(mat_files)} participants, {total_trials} trials, '
                     f'{combined.shape[0]} frames')
        })
        print(f"\n{'='*60}")
        print(f"Combined data saved: {comb_path}")
        print(f"Participants: {len(mat_files)} | Trials: {total_trials} | Frames: {combined.shape[0]}")
        print(f"Joint space: 16-D (hand only)")


if __name__ == '__main__':
    main()
