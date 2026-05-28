"""
Render PC1/PC2 of best_all_grasp_trialrange_none_all15 as a 2x4 grid of MuJoCo hand images.
Shows PC1 and PC2 at -100%, -33%, +33%, +100% of ±2σ range.

Usage:
    python vis_trialrange_grid.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from myoki.myoki_config import myoki_to_qpos

# --- CONFIG ---
MODEL_PATH = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'
PCA_PATH = 'PCA_results/MyoKi/pca_myoki_best_all_grasp_trialrange_none_all15.npz'
FIG_DIR = 'MyoKi_figures'

WEIGHT_FRACS = [-1.0, -0.33, 0.33, 1.0]
N_SIGMA = 2.0

# PC standard deviations for the trial_range model (estimated from data scale)
# Mean range values are ~7-33°, typical variation ~8-12°
PC_STD = [9.5, 5.5]

# Camera: top-down straight down
CAM_DISTANCE = 0.35
CAM_AZIMUTH = 0
CAM_ELEVATION = -90
CAM_CENTER = np.array([0.30, 0.00, 0.02], dtype=np.float64)


def load_pca_model(path):
    d = np.load(path)
    return d['components'], d['mean'], d['explained_variance']


def apply_posture(model, data, comps, mean, pc_idx, wf, n_sigma, pca_std):
    w_deg = wf * n_sigma * pca_std[pc_idx]
    synergy_15d = mean + w_deg * comps[pc_idx]
    qpos = myoki_to_qpos(synergy_15d)
    for i in range(min(len(qpos), model.nq)):
        if i < model.njnt:
            lo, hi = model.jnt_range[i]
            data.qpos[i] = np.clip(qpos[i], lo, hi)
        else:
            data.qpos[i] = qpos[i]
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    mujoco.mj_forward(model, data)


def render_scene(model, data, cam, scene, context, width, height):
    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()
    mujoco.mjv_updateScene(model, data, option, perturb, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    buf = np.empty((height, width, 3), dtype=np.uint8)
    rect = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(rect, scene, context)
    mujoco.mjr_readPixels(buf, None, rect, context)
    return np.flipud(buf)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Loading PCA model: {PCA_PATH}")
    comps, mean, ev = load_pca_model(PCA_PATH)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    width, height = 640, 480
    glctx = mujoco.GLContext(width, height)
    glctx.make_current()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100.value)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    cam.lookat[:] = CAM_CENTER

    titles_labels = [f'{int(wf*100)}%' for wf in WEIGHT_FRACS]
    imgs_pc1, imgs_pc2 = [], []

    for wf in WEIGHT_FRACS:
        apply_posture(model, data, comps, mean, 0, wf, N_SIGMA, PC_STD)
        imgs_pc1.append(render_scene(model, data, cam, scene, context,
                                      width, height).copy())
        print(f"  PC1 wf={wf:+.2f} ({wf*N_SIGMA*PC_STD[0]:+.0f}°)", flush=True)

        apply_posture(model, data, comps, mean, 1, wf, N_SIGMA, PC_STD)
        imgs_pc2.append(render_scene(model, data, cam, scene, context,
                                      width, height).copy())
        print(f"  PC2 wf={wf:+.2f} ({wf*N_SIGMA*PC_STD[1]:+.0f}°)", flush=True)

    del context, scene, glctx

    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    plt.subplots_adjust(wspace=0.05, hspace=0.06, top=0.92, bottom=0.03, left=0.01, right=0.99)

    for col in range(4):
        axes[0, col].imshow(imgs_pc1[col])
        axes[0, col].axis('off')
        w_val = WEIGHT_FRACS[col] * N_SIGMA * PC_STD[0]
        axes[0, col].set_title(f'PC1  {titles_labels[col]}  ({w_val:+.0f}°)',
                               fontsize=20, fontweight='bold', pad=4)
        axes[1, col].imshow(imgs_pc2[col])
        axes[1, col].axis('off')
        w_val = WEIGHT_FRACS[col] * N_SIGMA * PC_STD[1]
        axes[1, col].set_title(f'PC2  {titles_labels[col]}  ({w_val:+.0f}°)',
                               fontsize=20, fontweight='bold', pad=4)

    ev1, ev2 = ev[0], ev[1] if len(ev) > 1 else 0
    label = 'All Grasps · Trial Range · All 15 Joints'
    plt.suptitle(f'{label}\n'
                 f'PC1 = {ev1:.1%} var | PC2 = {ev2:.1%} var | '
                 f'Scale = ±{int(N_SIGMA)}σ (≈ ±{int(N_SIGMA*PC_STD[0])}°)',
                 fontsize=20, fontweight='bold', y=0.97)

    save_path = os.path.join(FIG_DIR, 'pc_grid_trialrange_none_all15.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n→ Saved: {save_path}")


if __name__ == '__main__':
    main()
