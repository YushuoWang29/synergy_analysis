"""
West-style time-series PCA/SVD on the MyoKi grasp-task data.

This script differs from the task-range PCA used in the v0.6 draft:
it does not reduce each trial to max(theta)-min(theta). Instead, it
concatenates the resampled joint-angle time series and runs PCA on the
full frame-by-joint matrix:

    X = [all selected frames] in R^(n_frames x 15)

The resulting components are therefore joint-space movement/posture
synergies, and the cumulative VAF curve is used to compute a normalized
area-above-curve complexity index in the spirit of West et al.

Usage:
    conda activate hand_synergy
    python code/myoki/pca_timeseries_west_myoki.py
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

# Same 23 right-handed participants used in the thesis analysis.
PARTICIPANTS = [f"P{i:02d}" for i in range(1, 16)] + [f"P{i:02d}" for i in range(17, 25)]

# Same 41 grasp/manipulation tasks used in Section 3.1.
GRASP_TASK_IDS = sorted(
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

FLEXION_IDS = np.array([1, 2, 4, 5, 6, 7, 9, 10, 12, 13])
RADIAL_IDS = np.array([0, 1, 2, 3, 4, 5])
ULNAR_IDS = np.array([9, 10, 11, 12, 13, 14])


def split_trials(angles: np.ndarray, num_trials: int) -> np.ndarray:
    """Return trial data as (num_trials, frames, N_JOINTS)."""
    n_frames = angles.shape[0]
    expected = num_trials * TARGET_FRAMES
    if n_frames == expected:
        return angles.reshape(num_trials, TARGET_FRAMES, N_JOINTS)

    # Fallback for older processed files with nonuniform frame count.
    chunks = []
    for i in range(num_trials):
        s = int(i * n_frames / num_trials)
        e = int((i + 1) * n_frames / num_trials)
        trial = angles[s:e, :N_JOINTS]
        if len(trial) == TARGET_FRAMES:
            chunks.append(trial)
        else:
            old_x = np.linspace(0, 1, len(trial))
            new_x = np.linspace(0, 1, TARGET_FRAMES)
            resampled = np.vstack([np.interp(new_x, old_x, trial[:, j]) for j in range(N_JOINTS)]).T
            chunks.append(resampled)
    return np.asarray(chunks)


def load_selected_frames() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Concatenate all frames from selected tasks.

    Returns
    -------
    X:
        Frame-by-joint matrix, shape (n_frames, 15).
    frame_task_ids:
        Task id for each row of X.
    frame_participants:
        Participant label for each row of X.
    task_trial_counts:
        Number of selected trials per task.
    """
    frames = []
    frame_task_ids = []
    frame_participants = []
    task_trial_counts = {tid: 0 for tid in GRASP_TASK_IDS}

    for pid in PARTICIPANTS:
        path = DATA_DIR / f"{pid}_preprocessed.mat"
        if not path.exists():
            print(f"WARNING: missing {path}, skipping")
            continue

        mat = sio.loadmat(path)
        angles = np.asarray(mat["angles"], dtype=float)[:, :N_JOINTS]
        trial_tasks = np.asarray(mat["trial_tasks"]).flatten().astype(int)
        num_trials = int(np.asarray(mat["num_trials"]).squeeze())
        trials = split_trials(angles, num_trials)

        for ti, trial in enumerate(trials):
            tid = int(trial_tasks[ti]) if ti < len(trial_tasks) else 0
            if tid not in task_trial_counts:
                continue
            frames.append(trial)
            frame_task_ids.extend([tid] * len(trial))
            frame_participants.extend([pid] * len(trial))
            task_trial_counts[tid] += 1

    if not frames:
        raise RuntimeError("No selected MyoKi frames were loaded.")

    X = np.vstack(frames)
    return (
        X,
        np.asarray(frame_task_ids, dtype=int),
        np.asarray(frame_participants, dtype=object),
        task_trial_counts,
    )


def vaf_ci(explained_variance_ratio: np.ndarray) -> float:
    """Normalized area above cumulative VAF curve, including the origin."""
    cum = np.cumsum(explained_variance_ratio)
    x = np.arange(len(cum) + 1)
    y = np.r_[0.0, cum]
    area_under = np.trapezoid(y, x)
    return float((len(cum) - area_under) / len(cum))


def orient_components(components: np.ndarray, scores: np.ndarray) -> None:
    """Make the first two components easier to interpret visually."""
    if components[0, FLEXION_IDS].sum() < 0:
        components[0] *= -1
        scores[:, 0] *= -1

    radial_minus_ulnar = components[1, RADIAL_IDS].sum() - components[1, ULNAR_IDS].sum()
    if radial_minus_ulnar < 0:
        components[1] *= -1
        scores[:, 1] *= -1


def compute_task_complexity(X: np.ndarray, frame_task_ids: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Compute West-style PCA/VAF-CI per task for later thesis use."""
    out: dict[str, dict[str, float | int]] = {}
    for tid in GRASP_TASK_IDS:
        Xi = X[frame_task_ids == tid]
        pca = PCA(n_components=N_JOINTS, svd_solver="full")
        pca.fit(Xi)
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        out[str(tid)] = {
            "n_frames": int(Xi.shape[0]),
            "n_trials": int(Xi.shape[0] // TARGET_FRAMES),
            "pc1": float(evr[0]),
            "pc2": float(evr[1]),
            "pc12": float(evr[:2].sum()),
            "n90": int(np.searchsorted(cum, 0.90) + 1),
            "n95": int(np.searchsorted(cum, 0.95) + 1),
            "n99": int(np.searchsorted(cum, 0.99) + 1),
            "vaf_ci": vaf_ci(evr),
        }
    return out


def main() -> None:
    X, frame_task_ids, frame_participants, task_trial_counts = load_selected_frames()
    print(f"Loaded X: {X.shape[0]} frames x {X.shape[1]} joints")
    print(f"Selected trials: {sum(task_trial_counts.values())}")
    print(f"Participants: {len(set(frame_participants.tolist()))}")
    print(f"Tasks: {len(GRASP_TASK_IDS)}")

    pca = PCA(n_components=N_JOINTS, svd_solver="full")
    scores = pca.fit_transform(X)
    orient_components(pca.components_, scores)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    ci = vaf_ci(evr)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)

    print("\nWest-style time-series PCA / VAF")
    print(f"  PC1: {evr[0]:.4%}")
    print(f"  PC2: {evr[1]:.4%}")
    print(f"  PC1+PC2: {evr[:2].sum():.4%}")
    print(f"  n90/n95/n99: {n90}/{n95}/{n99}")
    print(f"  VAF-CI: {ci:.6f}")

    print("\nPC1 weights:")
    for name, val in zip(JOINT_NAMES, pca.components_[0]):
        print(f"  {name:12s} {val:+.5f}")
    print("\nPC2 weights:")
    for name, val in zip(JOINT_NAMES, pca.components_[1]):
        print(f"  {name:12s} {val:+.5f}")

    task_complexity = compute_task_complexity(X, frame_task_ids)

    out_path = SAVE_DIR / "pca_myoki_west_all_grasp_timeseries_all23.npz"
    np.savez_compressed(
        out_path,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=evr,  # kept for synergy_myoki.py compatibility
        explained_variance_ratio=evr,
        explained_variance_cumsum=cum,
        singular_values=pca.singular_values_,
        eigenvalues=pca.explained_variance_,
        scores_pc12=scores[:, :2],
        joint_names=JOINT_NAMES,
        task_ids=np.asarray(GRASP_TASK_IDS, dtype=int),
        frame_task_ids=frame_task_ids,
        frame_participants=frame_participants,
        task_trial_counts=np.asarray([task_trial_counts[tid] for tid in GRASP_TASK_IDS], dtype=int),
        n_frames=X.shape[0],
        n_trials=sum(task_trial_counts.values()),
        n_participants=len(set(frame_participants.tolist())),
        n90=n90,
        n95=n95,
        n99=n99,
        vaf_ci=ci,
        data_label="west_all_grasp_timeseries_all23",
        method="Centered full time-series PCA/SVD on resampled MyoKi joint angles; no ROM reduction; no z-score.",
    )

    json_path = SAVE_DIR / "pca_myoki_west_all_grasp_timeseries_all23_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "npz": str(out_path),
                "X_shape": [int(X.shape[0]), int(X.shape[1])],
                "n_trials": int(sum(task_trial_counts.values())),
                "n_participants": int(len(set(frame_participants.tolist()))),
                "task_ids": GRASP_TASK_IDS,
                "explained_variance_ratio": [float(x) for x in evr],
                "cumulative_variance": [float(x) for x in cum],
                "n90": n90,
                "n95": n95,
                "n99": n99,
                "vaf_ci": ci,
                "pc1": {str(k): float(v) for k, v in zip(JOINT_NAMES, pca.components_[0])},
                "pc2": {str(k): float(v) for k, v in zip(JOINT_NAMES, pca.components_[1])},
                "task_trial_counts": {str(k): int(v) for k, v in task_trial_counts.items()},
                "task_complexity": task_complexity,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved NPZ: {out_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
