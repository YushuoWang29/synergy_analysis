"""
Interactive camera calibrator for MuJoCo hand rendering.
Opens the viewer; use mouse to adjust camera position.
Press 'P' to print current camera parameters (for use in visualize_pc_grid.py).
Press 'Q' or 'ESC' to quit.
"""

import mujoco
import mujoco_viewer
import numpy as np

MODEL_PATH = "model/mujoco_menagerie/shadow_hand/scene_right.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Set a neutral hand posture
data.qpos[:] = 0.0
mujoco.mj_forward(model, data)

viewer = mujoco_viewer.MujocoViewer(model, data)

print("\n" + "=" * 60)
print("  交互相机校准器")
print("=" * 60)
print("  Mouse controls:")
print("    Left drag   : rotate")
print("    Right drag  : pan")
print("    Scroll      : zoom")
print("    Middle drag : zoom")
print()
print("  Press 'P' to print current camera parameters")
print("  Press 'Q' or 'ESC' to quit")
print("=" * 60 + "\n")

# Override key callback
original_key_callback = viewer._key_callback

def key_callback(keycode):
    if keycode == 112:  # 'p'
        cam = viewer.cam
        print(f"\n>>> Camera params <<<")
        print(f"CAM_DISTANCE = {cam.distance:.4f}")
        print(f"CAM_AZIMUTH  = {cam.azimuth}")
        print(f"CAM_ELEVATION = {cam.elevation}")
        print(f"CAM_CENTER   = [{cam.lookat[0]:.4f}, {cam.lookat[1]:.4f}, {cam.lookat[2]:.4f}]")
        print(f"Type: {'FIXED' if cam.type == mujoco.mjtCamera.mjCAMERA_FIXED else 'FREE'}")
        print()
        return
    if original_key_callback:
        original_key_callback(keycode)

viewer._key_callback = key_callback

while viewer.is_alive:
    viewer.render()

viewer.close()
print("Done.")
