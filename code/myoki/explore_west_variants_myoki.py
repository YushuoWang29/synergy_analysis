"""
Explore academically defensible West-style PCA variants for MyoKi.

The goal is to keep time-series PCA/SVD, while testing preprocessing choices
that may better match the thesis mechanism design target:

* raw absolute joint angles vs. trial-relative movements;
* all selected frames vs. trial-averaged/subject-task-averaged trajectories;
* all 15 channels vs. flexion-only channels that correspond to fold DOFs;
* full 41 tasks vs. cleaner grasp-only task subsets.

The script prints a ranked table by PC1+PC2 variance and PC2 radial-ulnar
gradient similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "MyoKi Dataset" / "preprocessed"
TARGET_FRAMES = 100
N_JOINTS = 15
PARTICIPANTS = [f"P{i:02d}" for i in range(1, 16)] + [f"P{i:02d}" for i in range(17, 25)]

TASK_SETS = {
    "all41": sorted(
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
    ),
    "no_complex32": sorted(
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
            61,
        ]
    ),
    "power25": sorted([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45]),
    "cyl_sph_lift19": sorted([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 28, 29, 30, 31, 32, 33]),
}

JOINT_SETS = {
    "all15": np.arange(15),
    "no_interabd12": np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13]),
    "flex10": np.array([1, 2, 4, 5, 6, 7, 9, 10, 12, 13]),
    "mcp_pip8": np.array([4, 5, 6, 7, 9, 10, 12, 13]),
    "finger_mean5": np.arange(5),
}

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

TARGET_GRADIENT = {
    "all15": np.array([0, -2, -2, 0, -1, -1, 0, 0, 0, 1, 1, 0, 2, 2, 0], dtype=float),
    "no_interabd12": np.array([0, -2, -2, 0, -1, -1, 0, 0, 1, 1, 2, 2], dtype=float),
    "flex10": np.array([-2, -2, -1, -1, 0, 0, 1, 1, 2, 2], dtype=float),
    "mcp_pip8": np.array([-1, -1, 0, 0, 1, 1, 2, 2], dtype=float),
    "finger_mean5": np.array([-2, -1, 0, 1, 2], dtype=float),
}


@dataclass
class Trial:
    pid: str
    task_id: int
    angles: np.ndarray  # (100, 15)


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


def load_trials() -> list[Trial]:
    trials: list[Trial] = []
    for pid in PARTICIPANTS:
        path = DATA_DIR / f"{pid}_preprocessed.mat"
        if not path.exists():
            continue
        mat = sio.loadmat(path)
        angles = np.asarray(mat["angles"], dtype=float)[:, :N_JOINTS]
        trial_tasks = np.asarray(mat["trial_tasks"]).flatten().astype(int)
        num_trials = int(np.asarray(mat["num_trials"]).squeeze())
        trial_data = split_trials(angles, num_trials)
        for i, arr in enumerate(trial_data):
            tid = int(trial_tasks[i]) if i < len(trial_tasks) else 0
            trials.append(Trial(pid=pid, task_id=tid, angles=arr))
    return trials


def transform_trial(trial: np.ndarray, representation: str) -> np.ndarray:
    if representation == "raw":
        return trial
    if representation == "delta_start":
        return trial - trial[:5].mean(axis=0, keepdims=True)
    if representation == "trial_center":
        return trial - trial.mean(axis=0, keepdims=True)
    if representation == "delta_then_center":
        x = trial - trial[:5].mean(axis=0, keepdims=True)
        return x - x.mean(axis=0, keepdims=True)
    raise ValueError(representation)


def make_matrix(trials: list[Trial], task_ids: list[int], representation: str, aggregation: str) -> np.ndarray:
    selected = [t for t in trials if t.task_id in task_ids]
    if aggregation == "all_frames":
        return np.vstack([transform_trial(t.angles, representation) for t in selected])

    if aggregation == "trial_mean_traj":
        # Average repeated trials for each task across all participants.
        rows = []
        for tid in task_ids:
            arrs = [transform_trial(t.angles, representation) for t in selected if t.task_id == tid]
            if arrs:
                rows.append(np.mean(arrs, axis=0))
        return np.vstack(rows)

    if aggregation == "subject_task_mean_traj":
        rows = []
        for pid in PARTICIPANTS:
            for tid in task_ids:
                arrs = [transform_trial(t.angles, representation) for t in selected if t.pid == pid and t.task_id == tid]
                if arrs:
                    rows.append(np.mean(arrs, axis=0))
        return np.vstack(rows)

    raise ValueError(aggregation)


def select_features(X15: np.ndarray, joint_name: str, jids: np.ndarray) -> np.ndarray:
    if joint_name != "finger_mean5":
        return X15[:, jids]

    # Thumb: MCP/IP; other fingers: MCP/PIP.
    return np.column_stack(
        [
            X15[:, [1, 2]].mean(axis=1),
            X15[:, [4, 5]].mean(axis=1),
            X15[:, [6, 7]].mean(axis=1),
            X15[:, [9, 10]].mean(axis=1),
            X15[:, [12, 13]].mean(axis=1),
        ]
    )


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den == 0 else float(abs(np.dot(a, b) / den))


def vaf_ci(evr: np.ndarray) -> float:
    cum = np.cumsum(evr)
    x = np.arange(len(cum) + 1)
    y = np.r_[0, cum]
    area_under = np.trapezoid(y, x)
    return float((len(cum) - area_under) / len(cum))


def main() -> None:
    trials = load_trials()
    print(f"Loaded {len(trials)} total trials")

    rows = []
    for task_name, task_ids in TASK_SETS.items():
        for representation in ["raw", "delta_start", "trial_center", "delta_then_center"]:
            for aggregation in ["all_frames", "subject_task_mean_traj", "trial_mean_traj"]:
                X15 = make_matrix(trials, task_ids, representation, aggregation)
                for joint_name, jids in JOINT_SETS.items():
                    X = select_features(X15, joint_name, jids)
                    pca = PCA(n_components=len(jids), svd_solver="full")
                    pca.fit(X)
                    evr = pca.explained_variance_ratio_
                    pc2_grad = corr_abs(pca.components_[1], TARGET_GRADIENT[joint_name])
                    pc1_closure = corr_abs(pca.components_[0], np.ones(len(jids)))
                    rows.append(
                        {
                            "task": task_name,
                            "repr": representation,
                            "agg": aggregation,
                            "joints": joint_name,
                            "n_rows": X.shape[0],
                            "pc1": float(evr[0]),
                            "pc2": float(evr[1]),
                            "pc12": float(evr[:2].sum()),
                            "pc2_grad": pc2_grad,
                            "pc1_close": pc1_closure,
                            "n90": int(np.searchsorted(np.cumsum(evr), 0.90) + 1),
                            "ci": vaf_ci(evr),
                        }
                    )

    # Prefer high cumulative variance and gradient-like PC2.
    rows.sort(key=lambda r: (r["pc12"] >= 0.80, r["pc2_grad"], r["pc12"], r["pc1_close"]), reverse=True)

    print("\nTop variants by PC2 gradient among PC1+PC2 >= 80%:")
    header = f"{'task':14s} {'repr':17s} {'agg':22s} {'joints':14s} {'PC1':>7s} {'PC2':>7s} {'PC12':>7s} {'grad2':>7s} {'close1':>7s} {'n90':>4s} {'rows':>8s}"
    print(header)
    print("-" * len(header))
    for r in [x for x in rows if x["pc12"] >= 0.80][:30]:
        print(
            f"{r['task']:14s} {r['repr']:17s} {r['agg']:22s} {r['joints']:14s} "
            f"{r['pc1']*100:6.1f}% {r['pc2']*100:6.1f}% {r['pc12']*100:6.1f}% "
            f"{r['pc2_grad']:7.3f} {r['pc1_close']:7.3f} {r['n90']:4d} {r['n_rows']:8d}"
        )

    print("\nTop all41 variants:")
    print(header)
    print("-" * len(header))
    all41 = [x for x in rows if x["task"] == "all41"]
    all41.sort(key=lambda r: (r["pc12"] >= 0.80, r["pc2_grad"], r["pc12"], r["pc1_close"]), reverse=True)
    for r in all41[:30]:
        print(
            f"{r['task']:14s} {r['repr']:17s} {r['agg']:22s} {r['joints']:14s} "
            f"{r['pc1']*100:6.1f}% {r['pc2']*100:6.1f}% {r['pc12']*100:6.1f}% "
            f"{r['pc2_grad']:7.3f} {r['pc1_close']:7.3f} {r['n90']:4d} {r['n_rows']:8d}"
        )


if __name__ == "__main__":
    main()
