"""
MyoKi Interactive Synergy Demo (v3 — Static Hold PCA).

Supports loading per-participant or combined PCA models.
Use --participant P01 (or P02, etc.) to explore a specific participant's synergy space.
Use --pca_model to load a specific .npz file directly.

For each synergy state, the 15-D joint angles are computed as:
    angles = mean + w1*pc1 + w2*pc2 + ...
Then mapped to ShadowHand qpos and displayed in MuJoCo.

Usage:
    python synergy_myoki.py --participant P01
    python synergy_myoki.py --participant all_grasp_static
    python synergy_myoki.py --pca_model PCA_results/MyoKi/pca_myoki_P01.npz
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

DEFAULT_PCA_DIR = 'PCA_results/MyoKi'
DEFAULT_MODEL_PATH = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'
SYNERGY_SCALE = 1.0
SLIDER_RANGE = 60

# Layout constants for non-overlap
SLIDER_BOTTOM = 0.06       # bottom of the lowest slider
SLIDER_SPACING = 0.04      # height per slider
PLOT_BOTTOM_2D = 0.08      # plot bottom edge in 2D mode
PLOT_BOTTOM_SLIDER = 0.35  # plot bottom edge in slider mode (leave room for sliders)


class SynergyController:
    """Manages the synergy state and MuJoCo visualization."""

    def __init__(self, pca_path, model_path):
        # Load PCA model
        pca_data = np.load(pca_path, allow_pickle=True)
        self.pc_vectors = pca_data['components']       # (n_components, 15)
        self.mean_pose = pca_data['mean']               # (15,)
        self.explained_var = pca_data['explained_variance']
        self.data_label = str(pca_data.get('data_label', 'unknown'))

        print(f"Loaded PCA model: {pca_path}")
        print(f"  Label: {self.data_label}")
        print(f"  Mean pose (15-D): {[f'{v:.1f}' for v in self.mean_pose]}")
        print(f"  PC vectors shape: {self.pc_vectors.shape}")
        print(f"  Explained variance (first 5): {[f'{v:.1%}' for v in self.explained_var[:5]]}")
        print(f"  PC1+PC2 cumulative: {self.explained_var[:2].sum():.1%}")

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

        self.slider_range = SLIDER_RANGE

        # Synergy weights
        self.n_pcs = min(5, self.pc_vectors.shape[0])
        self.weights_2d = [0.0, 0.0]
        self.weights_slider = [0.0] * self.n_pcs
        self.mode = '2d'  # '2d' or 'slider'

        print(f"\nUsing {self.n_pcs} PCs. Slider range: ±{self.slider_range}°")
        print("Mode: 2D (click on PC1-PC2 plane). Press 'm' to toggle sliders.")

    def compute_pose(self):
        """Compute 15-D joint angles from current synergy weights."""
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
        pose_15d = self.compute_pose()
        qpos = myoki_to_qpos(pose_15d, self.joint_ranges_rad)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)


def run_gui(controller):
    """Run the matplotlib GUI for synergy control — no overlapping elements."""
    matplotlib.use('TkAgg')

    sr = controller.slider_range
    n_pcs = controller.n_pcs

    fig = plt.figure(figsize=(7, 7))

    # Main axis for the 2D PC1-PC2 scatter plot
    ax = fig.add_axes([0.12, PLOT_BOTTOM_2D, 0.78, 0.85])
    ax.set_xlim([-sr, sr])
    ax.set_ylim([-sr, sr])
    ax.set_xlabel(f'PC1 ({controller.explained_var[0]:.1%}) [deg]')
    ax.set_ylabel(f'PC2 ({controller.explained_var[1]:.1%}) [deg]')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title(f"MyoKi Synergy Space — {controller.data_label}\n"
                 f"PC1 ({controller.explained_var[0]:.1%}) vs PC2 ({controller.explained_var[1]:.1%})\n"
                 f"Mode: {controller.mode.upper()} — Press 'm' to switch")

    point, = ax.plot([0], [0], 'ro', markersize=12, zorder=5)

    # Sliders for PC1-PC5 — placed below the plot area
    slider_axes = []
    sliders = []
    for i in range(n_pcs):
        y_pos = SLIDER_BOTTOM + (n_pcs - 1 - i) * SLIDER_SPACING  # top-to-bottom: PC1 first
        ax_slider = fig.add_axes([0.15, y_pos, 0.7, 0.025])
        slider = Slider(ax_slider, f'PC{i+1}', -sr, sr, valinit=0.0, valfmt='%.1f°')
        slider_axes.append(ax_slider)
        sliders.append(slider)
        ax_slider.set_visible(False)  # initially hidden in 2D mode

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
        for i in range(n_pcs):
            controller.weights_slider[i] = sliders[i].val

    for s in sliders:
        s.on_changed(update_slider)

    def on_key(event):
        if event.key == 'm':
            if controller.mode == '2d':
                # Switch to slider mode
                controller.mode = 'slider'
                # Slide plot up to make room for sliders
                ax.set_position([0.12, PLOT_BOTTOM_SLIDER, 0.78, 0.85 - (PLOT_BOTTOM_SLIDER - PLOT_BOTTOM_2D)])
                for ax_s in slider_axes:
                    ax_s.set_visible(True)
                title = f"MyoKi Synergy Space — {controller.data_label}\n"
                title += f"Mode: SLIDER — Press 'm' to switch"
            else:
                # Switch to 2D mode
                controller.mode = '2d'
                # Restore plot position
                ax.set_position([0.12, PLOT_BOTTOM_2D, 0.78, 0.85])
                for ax_s in slider_axes:
                    ax_s.set_visible(False)
                title = f"MyoKi Synergy Space — {controller.data_label}\n"
                title += f"PC1 ({controller.explained_var[0]:.1%}) vs PC2 ({controller.explained_var[1]:.1%})\n"
                title += f"Mode: 2D — Press 'm' to switch"
            ax.set_title(title)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', update_2d)
    fig.canvas.mpl_connect('motion_notify_event',
                           lambda ev: update_2d(ev) if ev.button == 1 else None)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def find_pca_model(participant, pca_dir=DEFAULT_PCA_DIR):
    """Find PCA model file for a given participant label."""
    path = os.path.join(pca_dir, f'pca_myoki_{participant}.npz')
    if os.path.exists(path):
        return path
    for f in os.listdir(pca_dir):
        if participant in f and f.endswith('.npz'):
            return os.path.join(pca_dir, f)
    return None


def main():
    parser = argparse.ArgumentParser(description='MyoKi Interactive Synergy Demo (v3)')
    parser.add_argument('--participant', default=None,
                        help='Participant label (P01, P02, all_grasp_static, etc.)')
    parser.add_argument('--pca_model', default=None,
                        help='Direct path to PCA .npz file')
    parser.add_argument('--pca_dir', default=DEFAULT_PCA_DIR,
                        help=f'PCA model directory (default: {DEFAULT_PCA_DIR})')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                        help=f'MuJoCo model path (default: {DEFAULT_MODEL_PATH})')

    args = parser.parse_args()

    # Determine PCA model path
    if args.pca_model:
        pca_path = args.pca_model
    elif args.participant:
        pca_path = find_pca_model(args.participant, args.pca_dir)
        if pca_path is None:
            print(f"Could not find PCA model for participant '{args.participant}'")
            print("Available models:")
            for f in sorted(os.listdir(args.pca_dir)):
                if f.endswith('.npz'):
                    print(f"  {f}")
            sys.exit(1)
    else:
        pca_path = find_pca_model('all_grasp_static', args.pca_dir)
        if pca_path is None:
            pca_path = find_pca_model('all_grasp', args.pca_dir)
        if pca_path is None:
            pca_path = find_pca_model('all', args.pca_dir)
        if pca_path is None:
            print("No default PCA model found. Please specify --participant or --pca_model")
            print("Available models:")
            for f in sorted(os.listdir(args.pca_dir)):
                if f.endswith('.npz'):
                    print(f"  {f}")
            sys.exit(1)
        print(f"Using default model: {pca_path}")

    if not os.path.exists(pca_path):
        print(f"ERROR: PCA model not found: {pca_path}")
        sys.exit(1)

    controller = SynergyController(pca_path, args.model)

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
