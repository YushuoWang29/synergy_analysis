"""
Render MyoKi PCA synergy eigenpostures as a grid of MuJoCo hand images.
For each PCA model, show PC1 and PC2 at -100%, -33%, +33%, +100% of ±2σ range.

Usage:
    # Batch mode (original): render all predefined models
    D:\anaconda3\envs\hand_synergy\python code/myoki/visualize_pc_grid.py

    # Single model by participant label (auto-prepends 'pca_myoki_'):
    D:\anaconda3\envs\hand_synergy\python code/myoki/visualize_pc_grid.py --participant best_all_grasp_trialrange_none_all15

    # Single model by direct .npz path:
    D:\anaconda3\envs\hand_synergy\python code/myoki/visualize_pc_grid.py --pca PCA_results/MyoKi/pca_myoki_best_all_grasp_trialrange_none_all15.npz
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from myoki.myoki_config import myoki_to_qpos

# --- CONFIG ---
PCA_DIR = 'PCA_results/MyoKi'
FIG_DIR = 'MyoKi_figures'

WEIGHT_FRACS = [-1.0, -0.33, 0.33, 1.0]
N_SIGMA = 2.0

# Standard deviations for different datasets (from all_tasks_static combined PCA)
PC_STD = {
    'grasp_only':   [40.5, 22.7],
    'all_tasks':    [49.1, 30.2],
}
PISA_STD = {
    'grasp_only':   [34.6, 31.0],
    'all_tasks':    [44.0, 37.1],
}

MODEL_PATH = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'

# Camera: top-down, straight down
CAM_DISTANCE = 0.35
CAM_AZIMUTH = 0
CAM_ELEVATION = -90
CAM_CENTER = np.array([0.30, 0.00, 0.02], dtype=np.float64)


def load_pca_model(label):
    path = os.path.join(PCA_DIR, f'{label}.npz')
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


def run_one_model(label, suffix, suffix_label, pca_std, display_label=None):
    pca_label = f'{label}{suffix}'
    print(f"\n=== {pca_label} ===")
    comps, mean, ev = load_pca_model(pca_label)
    if display_label is None:
        display_label = label.replace('pca_myoki_', '').upper()

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
        apply_posture(model, data, comps, mean, 0, wf, N_SIGMA, pca_std)
        imgs_pc1.append(render_scene(model, data, cam, scene, context,
                                      width, height).copy())
        print(f"  PC1 wf={wf:+.2f} ({wf*N_SIGMA*pca_std[0]:+.0f}°)", flush=True)

        apply_posture(model, data, comps, mean, 1, wf, N_SIGMA, pca_std)
        imgs_pc2.append(render_scene(model, data, cam, scene, context,
                                      width, height).copy())
        print(f"  PC2 wf={wf:+.2f} ({wf*N_SIGMA*pca_std[1]:+.0f}°)", flush=True)

    del context, scene, glctx

    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    plt.subplots_adjust(wspace=0.05, hspace=0.06, top=0.92, bottom=0.03, left=0.01, right=0.99)

    for col in range(4):
        axes[0, col].imshow(imgs_pc1[col])
        axes[0, col].axis('off')
        w_val = WEIGHT_FRACS[col] * N_SIGMA * pca_std[0]
        axes[0, col].set_title(f'PC1  {titles_labels[col]}  ({w_val:+.0f}°)',
                               fontsize=20, fontweight='bold', pad=4)
        axes[1, col].imshow(imgs_pc2[col])
        axes[1, col].axis('off')
        w_val = WEIGHT_FRACS[col] * N_SIGMA * pca_std[1]
        axes[1, col].set_title(f'PC2  {titles_labels[col]}  ({w_val:+.0f}°)',
                               fontsize=20, fontweight='bold', pad=4)

    ev1, ev2 = ev[0], ev[1] if len(ev) > 1 else 0
    plt.suptitle(f'{display_label}{suffix_label}\n'
                 f'PC1 = {ev1:.1%} var | PC2 = {ev2:.1%} var | '
                 f'Scale = ±{int(N_SIGMA)}σ (≈ ±{int(N_SIGMA*pca_std[0])}°)',
                 fontsize=20, fontweight='bold', y=0.97)

    save_path = os.path.join(FIG_DIR, f'pc_grid_{pca_label}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Render PCA synergy eigenpostures as a 2x4 grid (PC1 × 4 weights, PC2 × 4 weights)')
    parser.add_argument('--participant', default=None,
                        help='Participant/model label (e.g. best_all_grasp_trialrange_none_all15). '
                             'Auto-prepends "pca_myoki_" and loads from PCA_results/MyoKi/')
    parser.add_argument('--pca', '--pca_model', dest='pca_model', default=None,
                        help='Direct path to a .npz PCA model file')
    parser.add_argument('--pc1_std', type=float, default=None,
                        help='PC1 standard deviation in degrees (for scaling the visualization). '
                             'If omitted, estimates from explained_variance.')
    parser.add_argument('--pc2_std', type=float, default=None,
                        help='PC2 standard deviation in degrees (for scaling the visualization). '
                             'If omitted, estimates from explained_variance.')
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    # ────────────────────────────────────────────────────────────
    # Single-model mode (--participant or --pca)
    # ────────────────────────────────────────────────────────────
    if args.participant is not None or args.pca_model is not None:
        if args.pca_model is not None:
            # Load directly from the provided .npz path
            if not os.path.isfile(args.pca_model):
                print(f"ERROR: PCA model file not found: {args.pca_model}")
                sys.exit(1)
            label = os.path.splitext(os.path.basename(args.pca_model))[0]
            # Temporarily override PCA_DIR to the parent directory of the given file
            global PCA_DIR
            PCA_DIR = os.path.dirname(args.pca_model)
            pca_std = [args.pc1_std, args.pc2_std] if args.pc1_std is not None and args.pc2_std is not None else PC_STD['grasp_only']
            if args.pc1_std is None or args.pc2_std is None:
                # Estimate from explained_variance
                d = np.load(args.pca_model)
                ev = d['explained_variance']
                n_samples = d.get('n_samples', None)
                if n_samples is not None:
                    est_std = np.sqrt(ev * n_samples)
                    print(f"  Estimated PC1 std from data: {est_std[0]:.1f}°")
                    print(f"  Estimated PC2 std from data: {est_std[1]:.1f}°")
                    pca_std = [est_std[0], est_std[1]]
                else:
                    print("  WARNING: No n_samples available in .npz. Falling back to default grasp_only std.")
                    pca_std = PC_STD['grasp_only']
            run_one_model(label, '', '', pca_std, display_label=label)
        else:
            # --participant mode: auto-prepend 'pca_myoki_'
            label = f'pca_myoki_{args.participant}'
            pca_std = [args.pc1_std, args.pc2_std] if args.pc1_std is not None and args.pc2_std is not None else PC_STD['grasp_only']
            if args.pc1_std is None or args.pc2_std is None:
                # Estimate from explained_variance
                try:
                    pca_path = os.path.join(PCA_DIR, f'{label}.npz')
                    d = np.load(pca_path)
                    ev = d['explained_variance']
                    n_samples = d.get('n_samples', None)
                    if n_samples is not None:
                        est_std = np.sqrt(ev * n_samples)
                        print(f"  Estimated PC1 std from data: {est_std[0]:.1f}°")
                        print(f"  Estimated PC2 std from data: {est_std[1]:.1f}°")
                        pca_std = [est_std[0], est_std[1]]
                    else:
                        print("  WARNING: No n_samples available in .npz. Falling back to default grasp_only std.")
                        pca_std = PC_STD['grasp_only']
                except Exception as e:
                    print(f"  WARNING: Could not estimate std from data ({e}). Using default grasp_only std.")
                    pca_std = PC_STD['grasp_only']
            run_one_model(label, '', '', pca_std, display_label=args.participant)

        print(f"\nDone! Figure saved in {FIG_DIR}/")
        return

    # ────────────────────────────────────────────────────────────
    # Batch mode (original behaviour)
    # ────────────────────────────────────────────────────────────
    ############################################################
    # 1) GRASP-ONLY PCA (original - 18 grasp tasks)
    ############################################################
    labels = [
        'all_grasp_static',
        'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07',
        'P08', 'P09', 'P11', 'P12',
    ]
    for label in labels:
        run_one_model(f'pca_myoki_{label}', '', '', PC_STD['grasp_only'])
        run_one_model(f'pca_myoki_{label}', '_pisa_rotated', ' (Pisa-rotated)',
                      PISA_STD['grasp_only'])

    ############################################################
    # 2) ALL-TASKS PCA (74 tasks, no filter)
    ############################################################
    labels_all = [
        'all_tasks_static',
        'P01_all_tasks_static',
        'P02_all_tasks_static',
        'P03_all_tasks_static',
        'P04_all_tasks_static',
        'P05_all_tasks_static',
        'P06_all_tasks_static',
        'P07_all_tasks_static',
        'P08_all_tasks_static',
        'P09_all_tasks_static',
        'P11_all_tasks_static',
        'P12_all_tasks_static',
    ]
    for label in labels_all:
        if 'all_tasks_static' in label and '_' not in label.replace('all_tasks_static', ''):
            # Combined label
            display = 'All participants (all 74 tasks)'
        else:
            participant = label.replace('_all_tasks_static', '').replace('pca_myoki_', '').upper()
            display = f'{participant} (all 74 tasks)'

        run_one_model(f'pca_myoki_{label}', '', '', PC_STD['all_tasks'], display)
        run_one_model(f'pca_myoki_{label}', '_pisa_rotated', ' (Pisa-rotated)',
                      PISA_STD['all_tasks'], display)

    print(f"\nDone! Figures in {FIG_DIR}/")


if __name__ == '__main__':
    main()
