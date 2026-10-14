"""
Main visualization script for hand motion playback on MuJoCo ShadowHand.

Supports both Ninapro and MyoKi datasets.
For MyoKi, uses proper mapping from code/myoki/myoki_config.py (16-D hand-space).

Usage:
    # MyoKi
    python code/visualization.py --dataset myoki --participant P01 --task 5 --subsample 500
    python code/visualization.py --dataset myoki --participant P01 --list_tasks
    
    # Ninapro (legacy)
    python code/visualization.py --dataset ninapro --path Ninapro Data/E2/S5_E2_B.mat
"""

import os, sys, argparse
import mujoco
import mujoco_viewer
import numpy as np
import scipy.io
import time

MODEL_PATH = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'


def main():
    parser = argparse.ArgumentParser(description='Visualize hand motion data on ShadowHand')
    parser.add_argument('--dataset', choices=['ninapro', 'myoki'], default='myoki')
    parser.add_argument('--model', default=MODEL_PATH)
    # MyoKi
    parser.add_argument('--participant', default='P01')
    parser.add_argument('--task', type=int, default=None)
    parser.add_argument('--subsample', type=int, default=2000)
    parser.add_argument('--list_tasks', action='store_true')
    # Ninapro
    parser.add_argument('--path', default='Ninapro Data/E2/S5_E2_B.mat')
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data  = mujoco.MjData(model)

    joint_ranges_rad = []
    for i in range(model.nq):
        joint_ranges_rad.append(
            model.jnt_range[i] if i < model.njnt else None)

    # ================================================================
    # MyoKi
    # ================================================================
    if args.dataset.lower() == 'myoki':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from myoki.myoki_config import myoki_to_qpos, get_myoki_joint_order_string
        from myoki.visualize_myoki import load_data_raw, list_available_tasks

        if args.list_tasks:
            list_available_tasks(args.participant)
            return

        print(get_myoki_joint_order_string())
        angles_deg, _ = load_data_raw(args.participant, task_id=args.task,
                                      subsample=args.subsample)

        print(f"\nConverting to ShadowHand qpos …")
        all_qpos = myoki_to_qpos(angles_deg, joint_ranges_rad)

        viewer = mujoco_viewer.MujocoViewer(model, data)
        print(f"Playing MyoKi {args.participant} …")
        for i in range(len(all_qpos)):
            if not viewer.is_alive:
                break
            data.qpos[:] = all_qpos[i]
            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(0.016)  # ~60fps

    # ================================================================
    # Ninapro (legacy)
    # ================================================================
    elif args.dataset.lower() == 'ninapro':
        mat = scipy.io.loadmat(args.path)
        angles = np.nan_to_num(mat['angles'])
        angles_rad = np.deg2rad(angles)

        mapping = {
            0: 20, 1: 19, 2: 22, 3: 23,
            4: 3,  5: 2,  6: 4,  16: 5,
            7: 7,  8: 8,  17: 9,
            9: 11, 10: 10, 11: 12, 18: 13,
            12: 14, 13: 16, 14: 15, 15: 17, 19: 18,
            20: 1, 21: 0
        }
        sign = {i: 1 for i in range(24)}
        for k in [0, 1]: sign[k] = 0   # wrist off
        for k in [9, 13, 18]: sign[k] = -1

        if args.subsample and args.subsample < len(angles_rad):
            step = max(1, len(angles_rad) // args.subsample)
            angles_rad = angles_rad[::step]

        viewer = mujoco_viewer.MujocoViewer(model, data)
        print(f"Playing Ninapro …")
        for i in range(len(angles_rad)):
            if not viewer.is_alive:
                break
            data.qpos[:] = 0.0
            frame = angles_rad[i]
            for src_idx, mj_idx in mapping.items():
                if src_idx < len(frame):
                    val = frame[src_idx] * sign[mj_idx]
                    rng = joint_ranges_rad[mj_idx]
                    if rng is not None:
                        val = np.clip(val, rng[0], rng[1])
                    data.qpos[mj_idx] = val
            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(0.016)

    viewer.close()
    print("Playback ended.")


if __name__ == '__main__':
    main()
