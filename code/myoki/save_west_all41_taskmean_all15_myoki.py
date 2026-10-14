"""
Save a transparent West-style PCA model:

* same 41 thesis grasp/manipulation tasks;
* all original 15 hand channels retained;
* raw resampled joint-angle trajectories;
* repeated trials averaged into one 100-frame trajectory per task;
* PCA/SVD on the resulting 4100 x 15 matrix.

No ROM reduction, no channel deletion, no structured weighting.
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

GRADIENT_TARGET = np.array([0, -2, -2, 0, -1, -1, 0, 0, 0, 1, 1, 0, 2, 2, 0], dtype=float)
CLOSURE_TARGET = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=float)


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


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def vaf_ci(evr: np.ndarray) -> float:
    cum = np.cumsum(evr)
    x = np.arange(len(cum) + 1)
    y = np.r_[0, cum]
    area_under = np.trapezoid(y, x)
    return float((len(cum) - area_under) / len(cum))


def load_task_mean_matrix() -> tuple[np.ndarray, dict[int, int]]:
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
        for i, trial in enumerate(trials):
            tid = int(trial_tasks[i]) if i < len(trial_tasks) else 0
            if tid in by_task:
                by_task[tid].append(trial)

    task_counts = {tid: len(v) for tid, v in by_task.items()}
    missing = [tid for tid, count in task_counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"Missing tasks: {missing}")
    return np.vstack([np.mean(by_task[tid], axis=0) for tid in TASK_IDS]), task_counts


def main() -> None:
    X, task_counts = load_task_mean_matrix()
    pca = PCA(n_components=N_JOINTS, svd_solver="full")
    scores = pca.fit_transform(X)

    # Orient signs only; PCA subspace and variance are unchanged.
    if np.dot(pca.components_[0] - pca.components_[0].mean(), CLOSURE_TARGET - CLOSURE_TARGET.mean()) < 0:
        pca.components_[0] *= -1
        scores[:, 0] *= -1
    # Keep PC2 with the sign that is closer to the target gradient, even if the match is moderate.
    if np.dot(pca.components_[1] - pca.components_[1].mean(), GRADIENT_TARGET - GRADIENT_TARGET.mean()) < 0:
        pca.components_[1] *= -1
        scores[:, 1] *= -1

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    ci = vaf_ci(evr)
    pc1_closure = corr_abs(pca.components_[0], CLOSURE_TARGET)
    pc2_gradient = corr_abs(pca.components_[1], GRADIENT_TARGET)

    print("West-style all41 task-mean all15 PCA")
    print(f"  X: {X.shape}")
    print(f"  PC1: {evr[0]:.4%}")
    print(f"  PC2: {evr[1]:.4%}")
    print(f"  PC1+PC2: {evr[:2].sum():.4%}")
    print(f"  n90/n95/n99: {n90}/{n95}/{n99}")
    print(f"  VAF-CI: {ci:.6f}")
    print(f"  PC1 closure corr: {pc1_closure:.4f}")
    print(f"  PC2 gradient corr: {pc2_gradient:.4f}")
    print("\nPC1 weights:")
    for name, val in zip(JOINT_NAMES, pca.components_[0]):
        print(f"  {name:12s} {val:+.5f}")
    print("\nPC2 weights:")
    for name, val in zip(JOINT_NAMES, pca.components_[1]):
        print(f"  {name:12s} {val:+.5f}")

    out_path = SAVE_DIR / "pca_myoki_west_all41_taskmean_all15_all23.npz"
    np.savez_compressed(
        out_path,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=evr,
        explained_variance_ratio=evr,
        explained_variance_cumsum=cum,
        singular_values=pca.singular_values_,
        eigenvalues=pca.explained_variance_,
        scores_pc12=scores[:, :2],
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
        pc1_closure_corr=pc1_closure,
        pc2_gradient_corr=pc2_gradient,
        data_label="west_all41_taskmean_all15_all23",
        method="Centered PCA/SVD on raw 15-channel MyoKi trajectories after averaging repeated trials into one 100-frame trajectory per task; no ROM reduction, no channel deletion, no weighting.",
    )

    json_path = SAVE_DIR / "pca_myoki_west_all41_taskmean_all15_all23_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "npz": str(out_path),
                "X_shape": [int(X.shape[0]), int(X.shape[1])],
                "explained_variance_ratio": [float(x) for x in evr],
                "cumulative_variance": [float(x) for x in cum],
                "n90": n90,
                "n95": n95,
                "n99": n99,
                "vaf_ci": ci,
                "pc1_closure_corr": pc1_closure,
                "pc2_gradient_corr": pc2_gradient,
                "pc1": {str(k): float(v) for k, v in zip(JOINT_NAMES, pca.components_[0])},
                "pc2": {str(k): float(v) for k, v in zip(JOINT_NAMES, pca.components_[1])},
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
