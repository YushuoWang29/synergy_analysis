"""
Main visualization script for hand motion playback on MuJoCo ShadowHand.

Supports both Ninapro (22-D) and MyoKi (18-D) datasets.
For MyoKi, uses the proper mapping from code/myoki/myoki_config.py.

Usage:
    python code/visualization.py --dataset myoki --participant P01 --task 1 --subsample 2000
    python code/visualization.py --dataset ninapro --path Ninapro Data/E2/S5_E2_B.mat
"""

import os, sys, argparse
import mujoco
import mujoco_viewer
import numpy as np
import scipy.io
import time

# ================== 默认配置 ==================
MODEL_PATH = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'
# ==============================================

def main():
    parser = argparse.ArgumentParser(description='Visualize hand motion data on ShadowHand')
    parser.add_argument('--dataset', choices=['ninapro', 'myoki'], default='ninapro',
                        help='Dataset type (default: ninapro)')
    parser.add_argument('--model', default=MODEL_PATH, help='MuJoCo model path')
    
    # MyoKi specific
    parser.add_argument('--participant', default='P01', help='MyoKi participant (default: P01)')
    parser.add_argument('--task', type=int, default=None, help='MyoKi task ID (default: all)')
    parser.add_argument('--subsample', type=int, default=2000, help='Max frames to display')
    
    # Ninapro specific
    parser.add_argument('--path', default='Ninapro Data/E2/S5_E2_B.mat', help='Ninapro data path')
    
    args = parser.parse_args()
    
    # Load MuJoCo model
    print(f"Loading model: {args.model}")
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    
    print("=== MuJoCo Joint qpos Mapping ===")
    for i in range(model.nq):
        if i < model.njnt:
            name = model.joint(i).name
        else:
            name = "free joint"
        print(f"qpos[{i}] : {name}")
    
    # Pre-extract joint limits
    joint_ranges_rad = []
    for i in range(model.nq):
        if i < model.njnt:
            joint_ranges_rad.append(model.jnt_range[i])
        else:
            joint_ranges_rad.append(None)
    
    # ==================== Load data based on dataset type ====================
    if args.dataset.lower() == "ninapro":
        mat_path = args.path
        mat = scipy.io.loadmat(mat_path)
        angles = np.nan_to_num(mat['angles'])
        angles_rad = np.deg2rad(angles)
        
        # Ninapro 22D -> MuJoCo qpos
        mapping = {
            0: 20, 1: 19, 2: 22, 3: 23,          # thumb
            4: 3,  5: 2,  6: 4,  16: 5,          # index
            7: 7,  8: 8,  17: 9,                 # middle
            9: 11, 10: 10, 11: 12, 18: 13,       # ring
            12: 14, 13: 16, 14: 15, 15: 17, 19: 18,  # little
            20: 1, 21: 0                          # wrist
        }
        sign_correction = {
            0: 0, 1: 0,        # wrist disabled
            2: 1, 3: 1, 4: 1, 5: 1,
            7: 1, 8: 1, 9: -1,
            10: 1, 11: 1, 12: 1, 13: -1,
            14: 1, 15: 1, 16: 1, 17: 1, 18: -1,
            19: 1, 20: 1, 21: 1, 22: 1, 23: 1
        }
        
        print(f"Loaded Ninapro: {angles_rad.shape}")
        
        # Subsampling
        if args.subsample and args.subsample < len(angles_rad):
            step = max(1, len(angles_rad) // args.subsample)
            angles_rad = angles_rad[::step]
            print(f"Subsampled to {len(angles_rad)} frames")
        
        # Playback
        viewer = mujoco_viewer.MujocoViewer(model, data)
        print(f"Playing Ninapro data...")
        
        for i in range(len(angles_rad)):
            if not viewer.is_alive:
                break
            data.qpos[:] = 0.0
            frame = angles_rad[i]
            for src_idx, mj_idx in mapping.items():
                if src_idx < len(frame):
                    val = frame[src_idx] * sign_correction.get(mj_idx, 1)
                    rng = joint_ranges_rad[mj_idx]
                    if rng is not None:
                        val = np.clip(val, rng[0], rng[1])
                    data.qpos[mj_idx] = val
            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(0.01)
    
    elif args.dataset.lower() == "myoki":
        # Use the proper MyoKi config
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from myoki.myoki_config import (
            myoki_to_qpos, get_myoki_joint_order_string,
            MYOKI_TO_SHADOW, SHADOW_JOINT_NAMES, MYOKI_JOINT_NAMES,
            DIP_FROM_PIP_MAPPING
        )
        from myoki.visualize_myoki import load_data_raw, load_data_preprocessed
        
        print(get_myoki_joint_order_string())
        
        # Load raw data
        angles_deg = load_data_raw(args.participant, task_id=args.task, subsample=args.subsample)
        
        print(f"\nConverting to ShadowHand qpos...")
        all_qpos = myoki_to_qpos(angles_deg, joint_ranges_rad)
        
        # Playback
        viewer = mujoco_viewer.MujocoViewer(model, data)
        print(f"Playing MyoKi data...")
        
        for i in range(len(all_qpos)):
            if not viewer.is_alive:
                break
            data.qpos[:] = all_qpos[i]
            mujoco.mj_forward(model, data)
            viewer.render()
            time.sleep(0.005)
    
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)
    
    viewer.close()
    print("Playback ended.")

if __name__ == '__main__':
    main()
