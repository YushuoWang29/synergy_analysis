"""
Generate Chinese thesis figures for Section 3.1.

The figures are based on the 23-participant MyoKi task-level range PCA
reported in docs/experiment_report_allgrasp_trialrange_none_all23_zh.md.
Outputs are written to ../../doc/figure from the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "synergy_analysis"
DATA_DIR = ANALYSIS / "MyoKi Data" / "preprocessed"
if not DATA_DIR.exists():
    DATA_DIR = ANALYSIS / "MyoKi Dataset" / "preprocessed"
PCA_PATH = ANALYSIS / "PCA_results" / "MyoKi" / "pca_myoki_all_grasp_trialrange_none_all23.npz"
OUT_DIR = ROOT / "doc" / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 180

N_JOINTS = 15
TARGET_FRAMES = 100

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 16)] + [f"P{i:02d}" for i in range(17, 25)]

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

CATEGORY = {
    **{tid: "圆柱抓取" for tid in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]},
    **{tid: "三指捏" for tid in [16, 17, 18, 19, 20, 21]},
    **{tid: "球状抓取" for tid in [28, 29, 30, 31, 32, 33]},
    **{tid: "钩状抓取" for tid in [40, 41, 42, 43, 44, 45]},
    **{tid: "复合抓取" for tid in [46, 47, 48, 49, 50, 51, 54, 55, 56]},
    61: "对指捏",
}

JOINT_LABELS = [
    "拇指\n外展",
    "拇指\nMCP",
    "拇指\nIP",
    "拇指\n对掌",
    "食指\nMCP",
    "食指\nPIP",
    "中指\nMCP",
    "中指\nPIP",
    "食-中\n外展",
    "无名指\nMCP",
    "无名指\nPIP",
    "中-无名\n外展",
    "小指\nMCP",
    "小指\nPIP",
    "无名-小\n外展",
]


def load_task_matrix() -> tuple[np.ndarray, dict[int, int]]:
    task_ranges = {tid: [] for tid in GRASP_TASK_IDS}
    counts = {tid: 0 for tid in GRASP_TASK_IDS}

    for pid in PARTICIPANTS:
        path = DATA_DIR / f"{pid}_preprocessed.mat"
        if not path.exists():
            continue

        mat = sio.loadmat(path)
        angles = np.asarray(mat["angles"], dtype=float)
        trial_tasks = mat["trial_tasks"].flatten().astype(int)
        num_trials = int(np.asarray(mat["num_trials"]).squeeze())

        n_frames = angles.shape[0]
        if n_frames == num_trials * TARGET_FRAMES:
            trial_data = angles.reshape(num_trials, TARGET_FRAMES, N_JOINTS)
        else:
            trial_data = []
            for ti in range(num_trials):
                s = int(ti * n_frames / num_trials)
                e = int((ti + 1) * n_frames / num_trials)
                trial_data.append(angles[s:e, :N_JOINTS])

        for ti in range(num_trials):
            tid = int(trial_tasks[ti]) if ti < len(trial_tasks) else 0
            if tid not in task_ranges:
                continue
            data = trial_data[ti] if isinstance(trial_data, list) else trial_data[ti]
            if len(data) < 5:
                continue
            task_ranges[tid].append(data.max(axis=0) - data.min(axis=0))
            counts[tid] += 1

    matrix = np.zeros((len(GRASP_TASK_IDS), N_JOINTS))
    for i, tid in enumerate(GRASP_TASK_IDS):
        if task_ranges[tid]:
            matrix[i] = np.mean(task_ranges[tid], axis=0)
    return matrix, counts


def load_pca():
    data = np.load(PCA_PATH, allow_pickle=True)
    return data["components"], data["mean"], data["explained_variance"], data["explained_variance_cumsum"]


def save_variance(ev: np.ndarray, cum: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].bar(range(1, len(ev) + 1), ev * 100, color="#4C78A8", edgecolor="#17324D")
    axes[0].set_xlabel("主成分编号")
    axes[0].set_ylabel("方差解释率/%")
    axes[0].set_title("各主成分方差解释率")
    axes[0].set_xticks(range(1, len(ev) + 1))
    axes[0].grid(axis="y", alpha=0.25)
    for i in range(5):
        axes[0].text(i + 1, ev[i] * 100 + 0.8, f"{ev[i] * 100:.1f}", ha="center", fontsize=9)

    axes[1].plot(range(1, len(cum) + 1), cum * 100, "o-", color="#F58518", lw=2)
    for y, label, style in [(90, "90%", "--"), (95, "95%", ":"), (99, "99%", "-.")]:
        axes[1].axhline(y, color="#777777", ls=style, lw=1.2, label=label)
    axes[1].set_xlabel("保留主成分数量")
    axes[1].set_ylabel("累计方差解释率/%")
    axes[1].set_title("累计方差解释率")
    axes[1].set_xticks(range(1, len(cum) + 1))
    axes[1].set_ylim(76, 101)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right")

    fig.suptitle("MyoKi抓取任务角度活动范围PCA的方差解释结果", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "3_1_1_pca_variance.png"
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return out


def save_loadings(components: np.ndarray, ev: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colors = ["#4C78A8" if v >= 0 else "#E45756" for v in components[1]]

    for idx, ax in enumerate(axes):
        vals = components[idx]
        if idx == 0:
            c = "#4C78A8"
            bar_colors = [c] * len(vals)
        else:
            bar_colors = colors
        ax.bar(range(N_JOINTS), vals, color=bar_colors, edgecolor="#333333", linewidth=0.5)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(f"PC{idx + 1}载荷（{ev[idx] * 100:.1f}%）")
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_LABELS, rotation=0, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("主成分载荷")
    fig.suptitle("前两个主成分对应的关节协同模式", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "3_1_2_pca_loadings.png"
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return out


def save_task_space(matrix: np.ndarray, components: np.ndarray, mean: np.ndarray, ev: np.ndarray) -> Path:
    scores = (matrix - mean) @ components[:2].T
    palette = {
        "圆柱抓取": "#4C78A8",
        "球状抓取": "#F58518",
        "钩状抓取": "#54A24B",
        "三指捏": "#E45756",
        "对指捏": "#B279A2",
        "复合抓取": "#8D6D63",
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    for cat, color in palette.items():
        idx = [i for i, tid in enumerate(GRASP_TASK_IDS) if CATEGORY[tid] == cat]
        if not idx:
            continue
        ax.scatter(
            scores[idx, 0],
            scores[idx, 1],
            s=54,
            color=color,
            label=cat,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.88,
        )
    for i, tid in enumerate(GRASP_TASK_IDS):
        ax.text(scores[i, 0], scores[i, 1] + 0.9, str(tid), ha="center", fontsize=7, alpha=0.78)

    ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.axvline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"PC1：整体抓取活动强度（{ev[0] * 100:.1f}%）")
    ax.set_ylabel(f"PC2：桡侧-尺侧活动梯度（{ev[1] * 100:.1f}%）")
    ax.set_title("41个抓取任务在二维协同空间中的分布", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()
    out = OUT_DIR / "3_1_3_pca_task_space.png"
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    components, mean, ev, cum = load_pca()
    matrix, counts = load_task_matrix()
    total_trials = sum(counts.values())
    print(f"Loaded {len([p for p in PARTICIPANTS if (DATA_DIR / (p + '_preprocessed.mat')).exists()])} participants")
    print(f"Task matrix: {matrix.shape}; total trials: {total_trials}")
    paths = [save_variance(ev, cum), save_loadings(components, ev)]
    if total_trials > 0:
        if matrix.shape != (41, 15):
            raise RuntimeError(f"Unexpected task matrix shape: {matrix.shape}")
        paths.append(save_task_space(matrix, components, mean, ev))
    else:
        print("Preprocessed trial data were not found; task-space scatter was not regenerated.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
