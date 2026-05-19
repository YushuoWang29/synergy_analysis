"""
MyoKi Interactive Synergy Demo.

This script allows interactive exploration of the MyoKi PCA synergy space.
Two modes:
  1) 2D mode: Click on a 2D PC1-PC2 plane to set synergy weights.
  2) Slider mode: Press 'm' to switch to sliders for first 5 PCs.

For each synergy state, the 18-D joint angles are computed as:
    angles = mean + w1*pc1*scale + w2*pc2*scale + ...
Then mapped to ShadowHand qpos and displayed in MuJoCo.

Usage:
    python synergy_myoki.py [--pca_model PCA_results/MyoKi/pca_myoki_all.npz]
                            [--model model/mujoco_menagerie/shadow_hand/scene_right.xml]
"""

import os, sys, argparse, threading, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import mujoco
import mujoco_viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from myoki.myoki_config import (
    myoki_to_qpos, MYOKI_TO_SHADOW, SHADOW_JOINT_NAMES, DIP_FROM_PIP_MAPPING
)

DEFAULT_PCA_PATH = 'PCA_results/MyoKi/pca_myoki_all.npz'
DEFAULT_MODEL_PATH = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'
SYNERGY_SCALE = 80  # scaling factor for PC weights (since PCA is done on degrees)


class SynergyController:
    """Manages the synergy state and MuJoCo visualization."""
    
    def __init__(self, pca_path, model_path):
        # Load PCA model
        pca_data = np.load(pca_path)
        self.pc_vectors = pca_data['components']       # (n_components, 18)
        self.mean_pose = pca_data['mean']               # (18,)
        self.explained_var = pca_data['explained_variance']
        
        print(f"Loaded PCA model: {pca_path}")
        print(f"  Mean pose (18-D): {self.mean_pose}")
        print(f"  PC vectors shape: {self.pc_vectors.shape}")
        print(f"  Explained variance (first 5): {self.explained_var[:5]}")
        
        # Load MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint ranges
        self.joint_ranges_rad = []
        for i in range(self.model.nq):
            if i < self.model.njnt:
                self.joint_ranges_rad.append(self.model.jnt_range[i])
            else:
                self.joint_ranges_rad.append(None)
        
        # Synergy weights
        self.n_pcs = min(5, self.pc_vectors.shape[0])
        self.weights_2d = [0.0, 0.0]
        self.weights_slider = [0.0] * self.n_pcs
        self.mode = '2d'  # '2d' or 'slider'
        
        print(f"\nUsing {self.n_pcs} principal components for synergy control.")
        print("Mode: 2D (click on PC1-PC2 plane). Press 'm' to toggle sliders.")
        
    def compute_pose(self):
        """Compute 18-D joint angles from current synergy weights."""
        pose = self.mean_pose.copy()
        if self.mode == '2d':
            pose += self.weights_2d[0] * self.pc_vectors[0] * SYNERGY_SCALE
            pose += self.weights_2d[1] * self.pc_vectors[1] * SYNERGY_SCALE
        else:
            for i in range(self.n_pcs):
                pose += self.weights_slider[i] * self.pc_vectors[i] * SYNERGY_SCALE
        return pose
    
    def update_mujoco(self):
        """Compute synergy pose and apply to MuJoCo."""
        pose_18d = self.compute_pose()
        qpos = myoki_to_qpos(pose_18d, self.joint_ranges_rad)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)


def run_gui(controller):
    """Run the matplotlib GUI for synergy control."""
    matplotlib.use('TkAgg')
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_title(f"MyoKi Synergy Space\nPC1 ({controller.explained_var[0]:.1%}) vs PC2 ({controller.explained_var[1]:.1%})\n"
                 f"Mode: {controller.mode.upper()} — Press 'm' to switch")
    ax.set_xlabel(f'PC1 ({controller.explained_var[0]:.1%})')
    ax.set_ylabel(f'PC2 ({controller.explained_var[1]:.1%})')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    
    point, = ax.plot([0], [0], 'ro', markersize=12, zorder=5)
    
    # Sliders for PC1-PC5
    slider_axes = []
    sliders = []
    for i in range(controller.n_pcs):
        ax_slider = plt.axes([0.15, 0.02 + i*0.04, 0.7, 0.02])
        slider = Slider(ax_slider, f'PC{i+1}', -6.0, 6.0, valinit=0.0)
        slider_axes.append(ax_slider)
        sliders.append(slider)
        ax_slider.set_visible(False)
    
    def update_2d(event):
        if controller.mode != '2d':
            return
        if event.xdata is not None and event.ydata is not None:
            controller.weights_2d[0] = event.xdata
            controller.weights_2d[1] = event.ydata
            point.set_data([event.xdata], [event.ydata])
            fig.canvas.draw_idle()
    
    def update_slider(val):
        if controller.mode != 'slider':
            return
        for i in range(controller.n_pcs):
            controller.weights_slider[i] = sliders[i].val
    
    for s in sliders:
        s.on_changed(update_slider)
    
    def on_key(event):
        if event.key == 'm':
            if controller.mode == '2d':
                controller.mode = 'slider'
                for ax_s in slider_axes:
                    ax_s.set_visible(True)
                title = f"MyoKi Synergy Space — Mode: SLIDER — Press 'm' to switch"
            else:
                controller.mode = '2d'
                for ax_s in slider_axes:
                    ax_s.set_visible(False)
                title = f"MyoKi Synergy Space\nPC1 ({controller.explained_var[0]:.1%}) vs PC2 ({controller.explained_var[1]:.1%})\n"
                title += f"Mode: 2D — Press 'm' to switch"
            ax.set_title(title)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', update_2d)
    fig.canvas.mpl_connect('motion_notify_event',
                           lambda ev: update_2d(ev) if ev.button == 1 else None)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='MyoKi Interactive Synergy Demo')
    parser.add_argument('--pca_model', default=DEFAULT_PCA_PATH,
                        help=f'PCA model path (default: {DEFAULT_PCA_PATH})')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                        help=f'MuJoCo model path (default: {DEFAULT_MODEL_PATH})')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pca_model):
        print(f"ERROR: PCA model not found: {args.pca_model}")
        print("Run pca_myoki.py first.")
        sys.exit(1)
    
    controller = SynergyController(args.pca_model, args.model)
    
    # Start GUI in a separate thread
    ui_thread = threading.Thread(target=run_gui, args=(controller,), daemon=True)
    ui_thread.start()
    
    # Start MuJoCo viewer
    viewer = mujoco_viewer.MujocoViewer(controller.model, controller.data)
    
    try:
        while viewer.is_alive:
            controller.update_mujoco()
            viewer.render()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("Demo ended.")


if __name__ == '__main__':
    main()
