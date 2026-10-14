"""
Save a structured 15-D West-style PCA model for MyoKi.

This variant keeps all 15 hand channels but changes the coordinate system:

1. Five finger-level flexion coordinates:
   (MCP + PIP) / sqrt(2) for thumb/index/middle/ring/little.
2. Five within-finger flexion-difference coordinates:
   (MCP - PIP) / sqrt(2) for the same fingers.
3. Five abduction/opposition coordinates:
   ThumbAbd, ThumbOpp, IdxMidAbd, MidRingAbd, RingLitAbd.

The difference and abduction/opposition coordinates are retained, but weighted
by alpha=0.25 because the thesis mechanism primarily realizes fold-like finger
flexion while these channels are secondary/non-actuated morphology variables.

Unlike the 5-D finger-mean model, this is a full 15-D, invertible linear
coordinate transform. No original degree of freedom is removed.
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
ALPHA = 0.25
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

STRUCTURED_NAMES = np.array(
    [
        "ThumbFlexMean",
        "IndexFlexMean",
        "MiddleFlexMean",
        "RingFlexMean",
        "LittleFlexMean",
        "ThumbFlexDiff",
        "IndexFlexDiff",
        "MiddleFlexDiff",
        "RingFlexDiff",
        "LittleFlexDiff",
        "ThumbAbd",
        "ThumbOpp",
        "IdxMidAbd",
        "MidRingAbd",
        "RingLitAbd",
    ],
    dtype=object,
)

PAIRS = [(1, 2), (4, 5), (6, 7), (9, 10), (12, 13)]
ABD_INDICES = [0, 3, 8, 11, 14]
GRADIENT_TARGET = np.r_[[-2, -1, 0, 1, 2], np.zeros(10)].astype(float)
CLOSURE_TARGET = np.r_[np.ones(5), np.zeros(10)].astype(float)


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


def to_structured(x15: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    means = np.column_stack([(x15[:, i] + x15[:, j]) / np.sqrt(2) for i, j in PAIRS])
    diffs = np.column_stack([(x15[:, i] - x15[:, j]) / np.sqrt(2) for i, j in PAIRS])
    abds = x15[:, ABD_INDICES]
    return np.column_stack([means, alpha * diffs, alpha * abds])


def from_structured_delta(z15: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Map structured weighted component rows back to original joint-angle deltas."""
    z15 = np.atleast_2d(z15)
    out = np.zeros((z15.shape[0], N_JOINTS), dtype=float)
    means = z15[:, :5]
    diffs = z15[:, 5:10] / alpha
    abds = z15[:, 10:15] / alpha
    for k, (i, j) in enumerate(PAIRS):
        out[:, i] = (means[:, k] + diffs[:, k]) / np.sqrt(2)
        out[:, j] = (means[:, k] - diffs[:, k]) / np.sqrt(2)
    for k, idx in enumerate(ABD_INDICES):
        out[:, idx] = abds[:, k]
    return out


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


def load_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], int, np.ndarray]:
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
            rows.append(to_structured(delta))
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


def main() -> None:
    X, frame_task_ids, frame_participants, task_counts, total_trials, mean_initial15 = load_matrix()
    pca = PCA(n_components=N_JOINTS, svd_solver="full")
    scores = pca.fit_transform(X)

    if np.dot(pca.components_[0] - pca.components_[0].mean(), CLOSURE_TARGET - CLOSURE_TARGET.mean()) < 0:
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
    pc1_closure = corr_abs(pca.components_[0], CLOSURE_TARGET)
    pc2_gradient = corr_abs(pca.components_[1], GRADIENT_TARGET)
    components_original15 = from_structured_delta(pca.components_)

    print("West-style structured 15-D movement PCA")
    print(f"  X: {X.shape}, alpha={ALPHA}")
    print(f"  Trials: {total_trials}, Participants: {len(set(frame_participants.tolist()))}")
    print(f"  PC1: {evr[0]:.4%}")
    print(f"  PC2: {evr[1]:.4%}")
    print(f"  PC1+PC2: {evr[:2].sum():.4%}")
    print(f"  n90/n95/n99: {n90}/{n95}/{n99}")
    print(f"  VAF-CI: {ci:.6f}")
    print(f"  PC1 closure corr: {pc1_closure:.4f}")
    print(f"  PC2 gradient corr: {pc2_gradient:.4f}")
    print("\nPC1 structured weights:")
    for name, val in zip(STRUCTURED_NAMES, pca.components_[0]):
        print(f"  {name:16s} {val:+.5f}")
    print("\nPC2 structured weights:")
    for name, val in zip(STRUCTURED_NAMES, pca.components_[1]):
        print(f"  {name:16s} {val:+.5f}")

    out_path = SAVE_DIR / "pca_myoki_west_all41_delta_structured15_w025_all23.npz"
    np.savez_compressed(
        out_path,
        components=components_original15,
        mean=mean_initial15,
        explained_variance=evr,
        explained_variance_ratio=evr,
        explained_variance_cumsum=cum,
        singular_values=pca.singular_values_,
        eigenvalues=pca.explained_variance_,
        components_structured15=pca.components_,
        mean_structured15=pca.mean_,
        structured_names=STRUCTURED_NAMES,
        joint_names=JOINT_NAMES,
        alpha=ALPHA,
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
        pc1_closure_corr=pc1_closure,
        pc2_gradient_corr=pc2_gradient,
        data_label="west_all41_delta_structured15_w025_all23",
        method=(
            "Centered PCA/SVD on full trial-start-relative MyoKi time series after an invertible 15-D "
            "structured coordinate transform: five finger flexion means, five MCP-PIP differences, "
            "and five abduction/opposition coordinates. Difference and abduction coordinates are retained "
            "with alpha=0.25 weighting; no ROM reduction; no DOF deletion."
        ),
    )

    json_path = SAVE_DIR / "pca_myoki_west_all41_delta_structured15_w025_all23_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "npz": str(out_path),
                "X_shape": [int(X.shape[0]), int(X.shape[1])],
                "alpha": ALPHA,
                "n_trials": int(total_trials),
                "n_participants": int(len(set(frame_participants.tolist()))),
                "explained_variance_ratio": [float(x) for x in evr],
                "cumulative_variance": [float(x) for x in cum],
                "n90": n90,
                "n95": n95,
                "n99": n99,
                "vaf_ci": ci,
                "pc1_closure_corr": pc1_closure,
                "pc2_gradient_corr": pc2_gradient,
                "pc1_structured": {str(k): float(v) for k, v in zip(STRUCTURED_NAMES, pca.components_[0])},
                "pc2_structured": {str(k): float(v) for k, v in zip(STRUCTURED_NAMES, pca.components_[1])},
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
