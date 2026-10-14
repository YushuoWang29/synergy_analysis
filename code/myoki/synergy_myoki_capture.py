"""
Screenshot-friendly MyoKi synergy explorer.

This is a thin wrapper around synergy_myoki.py. It uses the latest accepted
MyoKi PCA result by default and loads a derived ShadowHand scene with:
  - white background,
  - no checkerboard floor,
  - forearm and wrist geoms removed while preserving qpos order,
  - MCP abduction/adduction limits tightened to +/-5 degrees.

The original synergy_myoki.py and original ShadowHand XML files are unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mujoco_viewer

from synergy_myoki import SynergyController


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PCA_MODEL = ROOT / "PCA_results" / "MyoKi" / "pca_myoki_all41_taskmean_rms_start_all15_all23.npz"
DEFAULT_CAPTURE_MODEL = ROOT / "model" / "mujoco_menagerie" / "shadow_hand" / "scene_right_capture.xml"
DEFAULT_CAPTURE_RANGE = 120.0
SLIDER_BOTTOM = 0.06
SLIDER_SPACING = 0.04
PLOT_BOTTOM_2D = 0.08
PLOT_BOTTOM_SLIDER = 0.35


def locked_update(controller: SynergyController) -> None:
    lock = getattr(controller, "_capture_lock", None)
    if lock is None:
        controller.update_mujoco()
        return
    with lock:
        controller.update_mujoco()


def run_capture_viewer(controller: SynergyController, stop_event: threading.Event) -> None:
    viewer = mujoco_viewer.MujocoViewer(controller.model, controller.data)
    lock = getattr(controller, "_capture_lock", None)
    try:
        while viewer.is_alive and not stop_event.is_set():
            if lock is None:
                controller.update_mujoco()
                viewer.render()
            else:
                with lock:
                    controller.update_mujoco()
                    viewer.render()
            time.sleep(0.01)
    finally:
        viewer.close()


def run_capture_gui(controller: SynergyController) -> None:
    matplotlib.use("TkAgg")

    sr = controller.slider_range
    n_pcs = controller.n_pcs
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0.12, PLOT_BOTTOM_2D, 0.78, 0.85])
    ax.set_xlim([-sr, sr])
    ax.set_ylim([-sr, sr])
    ax.set_xlabel(f"PC1 ({controller.explained_var[0]:.1%}) [deg]")
    ax.set_ylabel(f"PC2 ({controller.explained_var[1]:.1%}) [deg]")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title(
        f"MyoKi Capture Synergy Space - {controller.data_label}\n"
        f"PC1 ({controller.explained_var[0]:.1%}) vs PC2 ({controller.explained_var[1]:.1%})\n"
        "Mode: 2D - Press 'm' to switch"
    )
    point, = ax.plot([0], [0], "ro", markersize=12, zorder=5)

    slider_axes = []
    sliders = []
    for i in range(n_pcs):
        y_pos = SLIDER_BOTTOM + (n_pcs - 1 - i) * SLIDER_SPACING
        ax_slider = fig.add_axes([0.15, y_pos, 0.7, 0.025])
        slider = Slider(ax_slider, f"PC{i + 1}", -sr, sr, valinit=0.0, valfmt="%.1f deg")
        ax_slider.set_visible(False)
        slider_axes.append(ax_slider)
        sliders.append(slider)

    def push_to_mujoco() -> None:
        locked_update(controller)

    def update_2d(event) -> None:
        if controller.mode != "2d":
            return
        if event.xdata is None or event.ydata is None:
            return
        controller.weights_2d[0] = float(event.xdata)
        controller.weights_2d[1] = float(event.ydata)
        point.set_data([event.xdata], [event.ydata])
        push_to_mujoco()
        fig.canvas.draw_idle()

    def update_slider(_val) -> None:
        if controller.mode != "slider":
            return
        for i in range(n_pcs):
            controller.weights_slider[i] = float(sliders[i].val)
        push_to_mujoco()
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update_slider)

    def on_key(event) -> None:
        if event.key != "m":
            return
        if controller.mode == "2d":
            controller.mode = "slider"
            ax.set_position([0.12, PLOT_BOTTOM_SLIDER, 0.78, 0.85 - (PLOT_BOTTOM_SLIDER - PLOT_BOTTOM_2D)])
            for ax_s in slider_axes:
                ax_s.set_visible(True)
            ax.set_title(
                f"MyoKi Capture Synergy Space - {controller.data_label}\n"
                "Mode: SLIDER - Press 'm' to switch"
            )
        else:
            controller.mode = "2d"
            ax.set_position([0.12, PLOT_BOTTOM_2D, 0.78, 0.85])
            for ax_s in slider_axes:
                ax_s.set_visible(False)
            ax.set_title(
                f"MyoKi Capture Synergy Space - {controller.data_label}\n"
                f"PC1 ({controller.explained_var[0]:.1%}) vs PC2 ({controller.explained_var[1]:.1%})\n"
                "Mode: 2D - Press 'm' to switch"
            )
        push_to_mujoco()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", update_2d)
    fig.canvas.mpl_connect("motion_notify_event", lambda ev: update_2d(ev) if ev.button == 1 else None)
    fig.canvas.mpl_connect("key_press_event", on_key)

    push_to_mujoco()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="MyoKi synergy explorer for clean screenshots")
    parser.add_argument(
        "--pca_model",
        default=os.fspath(DEFAULT_PCA_MODEL),
        help=f"PCA .npz file (default: {DEFAULT_PCA_MODEL})",
    )
    parser.add_argument(
        "--model",
        default=os.fspath(DEFAULT_CAPTURE_MODEL),
        help=f"MuJoCo model path (default: {DEFAULT_CAPTURE_MODEL})",
    )
    parser.add_argument(
        "--range",
        dest="slider_range",
        type=float,
        default=DEFAULT_CAPTURE_RANGE,
        help=f"PC1/PC2 plane and slider half-range in degrees (default: {DEFAULT_CAPTURE_RANGE:g})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pca_model):
        print(f"ERROR: PCA model not found: {args.pca_model}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"ERROR: capture model not found: {args.model}")
        print("Run: python code/myoki/create_capture_shadow_hand.py")
        sys.exit(1)

    controller = SynergyController(args.pca_model, args.model)
    controller.slider_range = args.slider_range
    controller._capture_lock = threading.Lock()
    print(f"Capture control range set to +/-{controller.slider_range:g} deg")

    stop_event = threading.Event()
    viewer_thread = threading.Thread(target=run_capture_viewer, args=(controller, stop_event), daemon=True)
    viewer_thread.start()
    try:
        run_capture_gui(controller)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        viewer_thread.join(timeout=1.0)
        print("Capture demo ended.")


if __name__ == "__main__":
    main()
