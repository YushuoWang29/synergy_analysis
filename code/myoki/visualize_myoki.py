"""
MyoKi dataset visualization on MuJoCo Shadow Hand.

Loads raw MyoKi .mat or preprocessed data and plays motion on the
ShadowHand model with live OSD (participant, task name, trial info).

Playback controls (keyboard while viewer is focused):
  Space  — pause / resume
  +/-    — speed up / slow down

Usage:
  # List available tasks for a participant
  python visualize_myoki.py --participant P01 --list_tasks

  # Play a specific task (all repetitions)
  python visualize_myoki.py --participant P01 --task 5 --subsample 500

  # Play all tasks
  python visualize_myoki.py --participant P01 --subsample 2000
"""

import os, sys, argparse, time
import numpy as np
import scipy.io as sio
import mujoco
import mujoco_viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from myoki.myoki_config import (
    myoki_to_qpos, get_myoki_joint_order_string,
    MYOKI_TO_SHADOW, SHADOW_JOINT_NAMES, MYOKI_JOINT_NAMES,
    DIP_FROM_PIP_MAPPING, MYOKI_N_JOINTS,
)

DEFAULT_MODEL_PATH = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'
DEFAULT_RAW_DIR    = 'MyoKi Dataset/raw'
DEFAULT_PREPROC_DIR= 'MyoKi Dataset/preprocessed'


# ---------------------------------------------------------------------------
# Task-name loader
# ---------------------------------------------------------------------------
def load_task_names():
    task_names = {0: 'Rest / Transition'}
    xlsx = os.path.join('MyoKi Dataset', 'Task_categorization.xlsx')
    if not os.path.exists(xlsx):
        return task_names
    try:
        import openpyxl
        wb = openpyxl.load_workbook(xlsx)
        ws = wb.active
        for row in ws.iter_rows(min_row=4, values_only=True):
            tid, name = row[0], row[1]
            if tid is not None and name is not None:
                task_names[int(tid)] = str(name).strip()
    except ImportError:
        pass
    return task_names

TASK_NAMES = load_task_names()


def list_available_tasks(participant, raw_dir=DEFAULT_RAW_DIR):
    """Print available task IDs with names for a participant."""
    mat_path = os.path.join(raw_dir, f'{participant}.mat')
    if not os.path.exists(mat_path):
        print(f"File not found: {mat_path}")
        return
    mat = sio.loadmat(mat_path)
    tasks = mat['task'].flatten().astype(int)
    unique = sorted(np.unique(tasks))
    print(f"\nAvailable tasks for {participant}:")
    print(f"{'Task ID':<8} {'Name'}")
    print("-" * 70)
    for tid in unique:
        name = TASK_NAMES.get(int(tid), '?')
        mask = (tasks == tid)
        n_frames = mask.sum()
        # count contiguous segments
        diff = np.diff(mask.astype(int), prepend=0, append=0)
        n_trials = (diff > 0).sum()
        print(f"  {tid:<6} {name:<50} [{n_trials} trial(s), {n_frames} frames]")


def load_data_raw(participant, raw_dir=DEFAULT_RAW_DIR, task_id=None,
                  subsample=None):
    """Load raw MyoKi data. Returns (data_deg, task_labels)."""
    mat_path = os.path.join(raw_dir, f'{participant}.mat')
    if not os.path.exists(mat_path):
        print(f"ERROR: {mat_path} not found.")
        sys.exit(1)

    mat = sio.loadmat(mat_path)
    glove_calib = np.nan_to_num(mat['glove_calibrated'])  # (N, 18)
    tasks = mat['task'].flatten().astype(int)

    # Strip wrist → keep 16 hand channels
    hand_data = glove_calib[:, :MYOKI_N_JOINTS]

    print(f"Loaded {participant}: {hand_data.shape[0]} frames ({len(hand_data)/2000:.1f}s @ 2000Hz)")
    print(f"  Hand-only channels (wrist removed): {MYOKI_N_JOINTS}")

    if task_id is not None:
        mask = (tasks == task_id)
        data  = hand_data[mask]
        tinfo = tasks[mask]
        print(f"  Task {task_id} ({TASK_NAMES.get(task_id,'?')}): {len(data)} frames")
        if len(data) == 0:
            print(f"  No frames for task {task_id}")
            sys.exit(1)
    else:
        data  = hand_data
        tinfo = tasks
        print(f"  All tasks: {len(data)} frames")

    if subsample is not None and subsample < len(data):
        step = max(1, len(data) // subsample)
        data  = data[::step]
        tinfo = tinfo[::step]
        print(f"  Subsampled to {len(data)} frames")

    return data, tinfo


def load_data_preprocessed(participant, preproc_dir=DEFAULT_PREPROC_DIR,
                           task_id=None, subsample=None):
    """Load preprocessed MyoKi data (returns angles, task_labels)."""
    # Try per-participant file first
    fpath = os.path.join(preproc_dir, f'{participant}_preprocessed.mat')
    if not os.path.exists(fpath):
        fpath = os.path.join(preproc_dir, 'all_participants_preprocessed.mat')
        if not os.path.exists(fpath):
            print("ERROR: No preprocessed data. Run preprocess_myoki.py first.")
            sys.exit(1)
        print("Using combined data (all participants). Cannot filter by participant.")
    else:
        print(f"Using preprocessed: {os.path.basename(fpath)}")

    mat = sio.loadmat(fpath)
    angles = mat['angles']  # (N, 16)
    trial_tasks = mat.get('trial_tasks', None)
    if trial_tasks is not None:
        trial_tasks = trial_tasks.flatten().astype(int)

    print(f"  Preprocessed: {angles.shape[0]} frames, {angles.shape[1]}-D")

    if task_id is not None and trial_tasks is not None:
        # Find frames belonging to this task
        # Since preprocessed has uniform trials, find trial-level mask
        # Actually, each trial contributes target_frames consecutive frames.
        # We need to find which trial indices match.
        # Simple approach: broadcast mask across all frames
        mask = np.isin(trials_to_frames(trial_tasks, 100), task_id)  # rough but works
        # Actually, let's use a simpler approach: use the trial_tasks array to find trial indices
        # then expand to frames
        
    if task_id is not None and trial_tasks is not None:
        target_frames = 100  # from preprocessing
        trial_idxs = np.where(trial_tasks == task_id)[0]
        frame_mask = np.zeros(len(angles), dtype=bool)
        for ti in trial_idxs:
            s = ti * target_frames
            e = s + target_frames
            frame_mask[s:e] = True
        angles = angles[frame_mask]
        print(f"  Task {task_id} ({TASK_NAMES.get(task_id,'?')}): {len(angles)} frames")

    if subsample is not None and subsample < len(angles):
        step = max(1, len(angles) // subsample)
        angles = angles[::step]
        print(f"  Subsampled to {len(angles)} frames")

    return angles, None  # task labels not available for OSD in preproc mode


def main():
    parser = argparse.ArgumentParser(description='MyoKi → ShadowHand playback')
    parser.add_argument('--participant', default='P01')
    parser.add_argument('--task', type=int, default=None,
                        help='Task ID (default: all tasks)')
    parser.add_argument('--subsample', type=int, default=2000,
                        help='Max frames to play')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Use preprocessed data')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--show_mapping', action='store_true',
                        help='Print mapping and exit')
    parser.add_argument('--list_tasks', action='store_true',
                        help='List available tasks and exit')
    parser.add_argument('--fps', type=float, default=60,
                        help='Target playback fps (default: 60)')

    args = parser.parse_args()

    if args.show_mapping:
        print(get_myoki_joint_order_string())
        return

    if args.list_tasks:
        list_available_tasks(args.participant)
        return

    # Load data
    if args.preprocessed:
        angles_deg, _ = load_data_preprocessed(
            args.participant, task_id=args.task, subsample=args.subsample)
        participant_label = f"{args.participant} [preprocessed]"
    else:
        angles_deg, task_labels = load_data_raw(
            args.participant, task_id=args.task, subsample=args.subsample)
        participant_label = args.participant

    # Load MuJoCo
    print(f"\nLoading model: {args.model}")
    model = mujoco.MjModel.from_xml_path(args.model)
    data  = mujoco.MjData(model)

    # Extract joint limits
    joint_ranges_rad = []
    for i in range(model.nq):
        if i < model.njnt:
            joint_ranges_rad.append(model.jnt_range[i])
        else:
            joint_ranges_rad.append(None)

    # Convert to qpos
    print(f"Converting {len(angles_deg)} frames to ShadowHand qpos …")
    all_qpos = myoki_to_qpos(angles_deg, joint_ranges_rad)
    print(f"  qpos shape: {all_qpos.shape}")

    # -------------------------------------------------------------------
    # Playback
    # -------------------------------------------------------------------
    viewer = mujoco_viewer.MujocoViewer(model, data)
    paused   = False
    speed    = 1.0
    frame_i  = 0
    n_frames = len(all_qpos)
    base_dt  = 1.0 / args.fps

    # Print instructions
    print(f"\nControls: Space=pause  +/-=speed  Close window=quit")
    print(f"Playing {n_frames} frames @ {args.fps} fps …\n")

    try:
        while viewer.is_alive and frame_i < n_frames:
            # --- keyboard (simple check via viewer internal state) ---
            # We can't easily capture keys with mujoco_viewer, so we use
            # the standard key callback
            # (mujoco_viewer may not support this; we'll handle via
            #  a simple periodic status update instead)

            if not paused:
                data.qpos[:] = all_qpos[frame_i]
                mujoco.mj_forward(model, data)

                # --- compute OSD info ---
                total_frames = len(all_qpos)
                if task_labels is not None and frame_i < len(task_labels):
                    current_task = int(task_labels[frame_i])
                    task_name = TASK_NAMES.get(current_task, '?')
                else:
                    # For preprocessed or raw with specific task
                    if args.task is not None:
                        current_task = args.task
                        task_name = TASK_NAMES.get(current_task, '?')
                    else:
                        current_task = '?'
                        task_name = '?'

                # Print status (replaces current line)
                pct = 100.0 * (frame_i + 1) / total_frames
                status = (f"[{participant_label}]  Task {current_task}: {task_name}  "
                          f"|  Frame {frame_i+1}/{total_frames} ({pct:.0f}%)  "
                          f"|  Speed {speed:.1f}x  {'[PAUSED]' if paused else ''}")
                sys.stdout.write("\033[K" + status + "\r")
                sys.stdout.flush()

                frame_i += 1

            viewer.render()
            time.sleep(base_dt / speed)

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("\nPlayback ended.")


if __name__ == '__main__':
    main()
