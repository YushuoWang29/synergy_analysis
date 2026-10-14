"""
Save the best defensible West-style PCA variant found in exploration:

* all 41 thesis grasp/manipulation tasks;
* raw resampled joint-angle trajectories, not ROM max-min features;
* trials averaged into one 100-frame representative trajectory per task;
* PCA on the 10 flexion channels that correspond to fold-like DOFs.

The PCA components are expanded back to 15-D for compatibility with
synergy_myoki.py; removed abduction/opposition channels have zero component
weights and retain their mean pose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "MyoKi Dataset" / "preprocessed"
SAVE_DIR = ROOT / "PCA_results" / "MyoKi"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_JOINTS = 15
TARGET_FRAMES = 100
PARTICIPANTS = [f"P{i:02d}" for i in range(1, 16)] + [f"P{i:02d}" for i in range(17, 25)]
TASK_IDS = sorted(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        28,
        29,
        30,
        31,
        32,
        33,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        54,
        55,
        56,
        61,
    ]
)

JOINT_NAMES = np.array(
    [
        "ThumbAbd",
        "ThumbMCP",
        "ThumbIP",
        "ThumbOpp",
        "IdxMCP",
        "IdxPIP",
        "MidMCP",
        "MidPIP",
        "IdxMidAbd",
        "RingMCP",
        "RingPIP",
        "MidRingAbd",
        "LitMCP",
        "LitPIP",
        "RingLitAbd",
    ],
    dtype=object,
)

FLEX10 = np.array([1, 2, 4, 5, 6, 7, 9, 10, 12, 13])
FLEX10_NAMES = JOINT_NAMES[FLEX10]
GRADIENT_TARGET = np.array([-2, -2, -1, -1, 0, 0, 1, 1, 2, 2], dtype=float)


def split_trials(angles: np.ndarray, num_trials: int) -> np.ndarray:
    expected = num_trials * TARGET_FRAMES
    if angles.shape[0] == expected:
        return angles.reshape(num_trials, TARGET_FRAMES, N_JOINTS)
    chunks = []
    for i in range(num_trials):
        s = int(i * angles.shape[0] / num_trials)
        e = int((i + 1) * angles.shape[0] / num_trials)
        trial = angles[s:e, :N_JOINTS]
        old_x = np.linspace(0, 1, len(trial))
        new_x = np.linspace(0, 1, TARGET_FRAMES)
        chunks.append(np.vstack([np.interp(new_x, old_x, trial[:, j]) for j in range(N_JOINTS)]).T)
    return np.asarray(chunks)


def load_task_mean_trajectories() -> tuple[np.ndarray, dict[int, int]]:
    by_task = {tid: [] for tid in TASK_IDS}
    for pid in PARTICIPANTS:
        path = DATA_DIR / f"{pid}_preprocessed.mat"
        if not path.exists():
            continue
        mat = sio.loadmat(path)
        angles = np.asarray(mat["angles"], dtype=float)[:, :N_JOINTS]
        trial_tasks = np.asarray(mat["trial_tasks"]).flatten().astype(int)
        num_trials = int(np.asarray(mat["num_trials"]).squeeze())
        trials = split_trials(angles, num_trials)
        for ti, trial in enumerate(trials):
            tid = int(trial_tasks[ti]) if ti < len(trial_tasks) else 0
            if tid in by_task:
                by_task[tid].append(trial)

    task_counts = {tid: len(arrs) for tid, arrs in by_task.items()}
    missing = [tid for tid, n in task_counts.items() if n == 0]
    if missing:
        raise RuntimeError(f"Missing task data: {missing}")

    # Shape: 41 tasks * 100 frames = 4100 rows, 15 channels.
    return np.vstack([np.mean(by_task[tid], axis=0) for tid in TASK_IDS]), task_counts


def vaf_ci(evr: np.ndarray) -> float:
    cum = np.cumsum(evr)
    x = np.arange(len(cum) + 1)
    y = np.r_[0, cum]
    area_under = np.trapezoid(y, x)
    return float((len(cum) - area_under) / len(cum))


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def main() -> None:
    X15, task_counts = load_task_mean_trajectories()
    X = X15[:, FLEX10]

    pca = PCA(n_components=len(FLEX10), svd_solver="full")
    scores = pca.fit_transform(X)

    # Orient PC1 as average flexion closure and PC2 as radial-to-ulnar gradient.
    if pca.components_[0].sum() < 0:
        pca.components_[0] *= -1
        scores[:, 0] *= -1
    if np.dot(pca.components_[1] - pca.components_[1].mean(), GRADIENT_TARGET - GRADIENT_TARGET.mean()) < 0:
        pca.components_[1] *= -1
        scores[:, 1] *= -1

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    ci = vaf_ci(evr)
    pc2_grad = corr_abs(pca.components_[1], GRADIENT_TARGET)

    components15 = np.zeros((len(FLEX10), N_JOINTS), dtype=float)
    components15[:, FLEX10] = pca.components_
    mean15 = X15.mean(axis=0)
    # Keep the PCA mean for selected channels exact; removed channels use their trajectory mean.
    mean15[FLEX10] = pca.mean_

    print("West-style task-mean flex10 PCA")
    print(f"  X15: {X15.shape}, X: {X.shape}")
    print(f"  PC1: {evr[0]:.4%}")
    print(f"  PC2: {evr[1]:.4%}")
    print(f"  PC1+PC2: {evr[:2].sum():.4%}")
    print(f"  n90/n95/n99: {n90}/{n95}/{n99}")
    print(f"  VAF-CI: {ci:.6f}")
    print(f"  PC2 gradient correlation: {pc2_grad:.4f}")
    print("\nPC1 flex10 weights:")
    for name, val in zip(FLEX10_NAMES, pca.components_[0]):
        print(f"  {name:12s} {val:+.5f}")
    print("\nPC2 flex10 weights:")
    for name, val in zip(FLEX10_NAMES, pca.components_[1]):
        print(f"  {name:12s} {val:+.5f}")

    out_path = SAVE_DIR / "pca_myoki_west_all41_taskmean_flex10_all23.npz"
    np.savez_compressed(
        out_path,
        components=components15,
        mean=mean15,
        explained_variance=evr,
        explained_variance_ratio=evr,
        explained_variance_cumsum=cum,
        singular_values=pca.singular_values_,
        eigenvalues=pca.explained_variance_,
        selected_joint_indices=FLEX10,
        selected_joint_names=FLEX10_NAMES,
        joint_names=JOINT_NAMES,
        task_ids=np.asarray(TASK_IDS, dtype=int),
        task_trial_counts=np.asarray([task_counts[tid] for tid in TASK_IDS], dtype=int),
        n_rows=X.shape[0],
        n_tasks=len(TASK_IDS),
        n_participants=len(PARTICIPANTS),
        n90=n90,
        n95=n95,
        n99=n99,
        vaf_ci=ci,
        pc2_gradient_corr=pc2_grad,
        data_label="west_all41_taskmean_flex10_all23",
        method="Centered PCA/SVD on 41 task-mean 100-frame MyoKi trajectories, flexion channels only; no ROM reduction; no z-score.",
    )

    json_path = SAVE_DIR / "pca_myoki_west_all41_taskmean_flex10_all23_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "npz": str(out_path),
                "X15_shape": list(X15.shape),
                "X_shape": list(X.shape),
                "explained_variance_ratio": [float(x) for x in evr],
                "cumulative_variance": [float(x) for x in cum],
                "n90": n90,
                "n95": n95,
                "n99": n99,
                "vaf_ci": ci,
                "pc2_gradient_corr": pc2_grad,
                "pc1": {str(k): float(v) for k, v in zip(FLEX10_NAMES, pca.components_[0])},
                "pc2": {str(k): float(v) for k, v in zip(FLEX10_NAMES, pca.components_[1])},
                "task_trial_counts": {str(k): int(v) for k, v in task_counts.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved NPZ: {out_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
