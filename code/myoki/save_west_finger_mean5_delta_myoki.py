"""
Save a West-style finger-level movement synergy model for MyoKi.

Method:
* Use the same 41 thesis grasp/manipulation tasks and all 23 right-handed
  participants.
* Keep complete resampled time series; no ROM max-min reduction.
* For each trial, subtract the mean of its first five frames, so PCA describes
  movement relative to the trial's initial hand posture.
* Convert 10 MCP/PIP flexion channels to five finger-level flexion variables
  by averaging each finger's two flexion channels.
* Run PCA/SVD on the 565600 x 5 frame-by-finger matrix.

For synergy_myoki.py compatibility, the five components are expanded back to
15-D by assigning each finger-level component to its MCP/PIP channels.
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

FINGER_NAMES = np.array(["Thumb", "Index", "Middle", "Ring", "Little"], dtype=object)
FINGER_CHANNELS = [
    (1, 2),    # Thumb MCP/IP
    (4, 5),    # Index MCP/PIP
    (6, 7),    # Middle MCP/PIP
    (9, 10),   # Ring MCP/PIP
    (12, 13),  # Little MCP/PIP
]
GRADIENT_TARGET = np.array([-2, -1, 0, 1, 2], dtype=float)


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


def finger_features(trial_delta: np.ndarray) -> np.ndarray:
    return np.column_stack([trial_delta[:, pair].mean(axis=1) for pair in FINGER_CHANNELS])


def load_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], int]:
    rows = []
    frame_task_ids = []
    frame_participants = []
    initial_poses = []
    task_counts = {tid: 0 for tid in TASK_IDS}
    total_trials = 0

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
            if tid not in task_counts:
                continue
            initial = trial[:5].mean(axis=0, keepdims=True)
            delta = trial - initial
            rows.append(finger_features(delta))
            frame_task_ids.extend([tid] * TARGET_FRAMES)
            frame_participants.extend([pid] * TARGET_FRAMES)
            initial_poses.append(initial.squeeze())
            task_counts[tid] += 1
            total_trials += 1

    return (
        np.vstack(rows),
        np.asarray(frame_task_ids, dtype=int),
        np.asarray(frame_participants, dtype=object),
        task_counts,
        total_trials,
        np.mean(initial_poses, axis=0),
    )


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


def expand_components(components5: np.ndarray) -> np.ndarray:
    components15 = np.zeros((components5.shape[0], N_JOINTS), dtype=float)
    for finger_idx, pair in enumerate(FINGER_CHANNELS):
        for joint_idx in pair:
            components15[:, joint_idx] = components5[:, finger_idx]
    return components15


def main() -> None:
    X, frame_task_ids, frame_participants, task_counts, total_trials, mean_initial15 = load_matrix()
    pca = PCA(n_components=5, svd_solver="full")
    scores = pca.fit_transform(X)

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

    components15 = expand_components(pca.components_)

    print("West-style all-frame finger-mean movement PCA")
    print(f"  X: {X.shape}")
    print(f"  Trials: {total_trials}, Participants: {len(set(frame_participants.tolist()))}")
    print(f"  PC1: {evr[0]:.4%}")
    print(f"  PC2: {evr[1]:.4%}")
    print(f"  PC1+PC2: {evr[:2].sum():.4%}")
    print(f"  n90/n95/n99: {n90}/{n95}/{n99}")
    print(f"  VAF-CI: {ci:.6f}")
    print(f"  PC2 gradient correlation: {pc2_grad:.4f}")
    print("\nPC1 finger weights:")
    for name, val in zip(FINGER_NAMES, pca.components_[0]):
        print(f"  {name:8s} {val:+.5f}")
    print("\nPC2 finger weights:")
    for name, val in zip(FINGER_NAMES, pca.components_[1]):
        print(f"  {name:8s} {val:+.5f}")

    out_path = SAVE_DIR / "pca_myoki_west_all41_delta_fingermean5_all23.npz"
    np.savez_compressed(
        out_path,
        components=components15,
        mean=mean_initial15,
        explained_variance=evr,
        explained_variance_ratio=evr,
        explained_variance_cumsum=cum,
        singular_values=pca.singular_values_,
        eigenvalues=pca.explained_variance_,
        components_finger5=pca.components_,
        mean_finger5=pca.mean_,
        scores_pc12=scores[:, :2],
        finger_names=FINGER_NAMES,
        joint_names=JOINT_NAMES,
        frame_task_ids=frame_task_ids,
        frame_participants=frame_participants,
        task_ids=np.asarray(TASK_IDS, dtype=int),
        task_trial_counts=np.asarray([task_counts[tid] for tid in TASK_IDS], dtype=int),
        n_frames=X.shape[0],
        n_trials=total_trials,
        n_participants=len(set(frame_participants.tolist())),
        n90=n90,
        n95=n95,
        n99=n99,
        vaf_ci=ci,
        pc2_gradient_corr=pc2_grad,
        data_label="west_all41_delta_fingermean5_all23",
        method="Centered PCA/SVD on trial-start-relative full MyoKi time series after averaging MCP/PIP flexion into five finger-level variables; no ROM reduction; no z-score.",
    )

    json_path = SAVE_DIR / "pca_myoki_west_all41_delta_fingermean5_all23_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "npz": str(out_path),
                "X_shape": [int(X.shape[0]), int(X.shape[1])],
                "n_trials": int(total_trials),
                "n_participants": int(len(set(frame_participants.tolist()))),
                "explained_variance_ratio": [float(x) for x in evr],
                "cumulative_variance": [float(x) for x in cum],
                "n90": n90,
                "n95": n95,
                "n99": n99,
                "vaf_ci": ci,
                "pc2_gradient_corr": pc2_grad,
                "pc1": {str(k): float(v) for k, v in zip(FINGER_NAMES, pca.components_[0])},
                "pc2": {str(k): float(v) for k, v in zip(FINGER_NAMES, pca.components_[1])},
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
