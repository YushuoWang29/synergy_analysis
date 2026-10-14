"""
Generate Section 3.1 analysis assets for thesis v0.6.

This script uses the preprocessed MyoKi files only. It reproduces the
task-level range PCA used in v0.5 and adds three evidence layers inspired by
recent synergy literature:

1. cumulative VAF / normalized area-above-curve complexity index;
2. task distribution in the first two PCs;
3. leave-one-subject-out stability of PC directions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "synergy_analysis"
DATA_DIR = ANALYSIS / "MyoKi Dataset" / "preprocessed"
OUT_DIR = ROOT / "doc" / "figure"
STATS_PATH = ROOT / "doc" / "section_3_1_v06_stats.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    **{tid: "提拉抓取" for tid in [40, 41, 42, 43, 44, 45]},
    **{tid: "复合抓取" for tid in [46, 47, 48, 49, 50, 51, 54, 55, 56]},
    61: "对指捏",
}

JOINT_KEYS = [
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
]

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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 180


def split_trials(angles: np.ndarray, num_trials: int) -> list[np.ndarray]:
    n_frames = angles.shape[0]
    expected = num_trials * TARGET_FRAMES
    if n_frames == expected:
        return list(angles.reshape(num_trials, TARGET_FRAMES, N_JOINTS))
    return [angles[int(i * n_frames / num_trials) : int((i + 1) * n_frames / num_trials)] for i in range(num_trials)]


def load_subject_task_ranges() -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, int]]:
    by_subject: dict[str, dict[int, list[np.ndarray]]] = {}
    counts: dict[str, int] = {}

    for pid in PARTICIPANTS:
        path = DATA_DIR / f"{pid}_preprocessed.mat"
        if not path.exists():
            continue
        mat = sio.loadmat(path)
        angles = np.asarray(mat["angles"], dtype=float)[:, :N_JOINTS]
        trial_tasks = np.asarray(mat["trial_tasks"]).flatten().astype(int)
        num_trials = int(np.asarray(mat["num_trials"]).squeeze())
        trials = split_trials(angles, num_trials)

        by_subject[pid] = {tid: [] for tid in GRASP_TASK_IDS}
        counts[pid] = 0
        for ti, trial in enumerate(trials):
            tid = int(trial_tasks[ti]) if ti < len(trial_tasks) else 0
            if tid not in by_subject[pid] or len(trial) < 5:
                continue
            by_subject[pid][tid].append(trial.max(axis=0) - trial.min(axis=0))
            counts[pid] += 1

    return by_subject, counts


def aggregate_matrix(by_subject: dict[str, dict[int, list[np.ndarray]]], participants: list[str]) -> tuple[np.ndarray, dict[int, int]]:
    X = np.zeros((len(GRASP_TASK_IDS), N_JOINTS), dtype=float)
    task_counts: dict[int, int] = {}
    for row, tid in enumerate(GRASP_TASK_IDS):
        ranges: list[np.ndarray] = []
        for pid in participants:
            ranges.extend(by_subject[pid][tid])
        task_counts[tid] = len(ranges)
        if not ranges:
            raise RuntimeError(f"No valid trials for task {tid}")
        X[row] = np.mean(ranges, axis=0)
    return X, task_counts


def fit_pca(X: np.ndarray) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=N_JOINTS)
    scores = pca.fit_transform(X)
    for k in range(2):
        if np.sum(pca.components_[k]) < 0:
            pca.components_[k] *= -1
            scores[:, k] *= -1
    return pca, scores


def vaf_ci(evr: np.ndarray) -> float:
    """Normalized area above the cumulative VAF curve, including the origin."""
    cum = np.cumsum(evr)
    x = np.arange(len(cum) + 1)
    y = np.r_[0.0, cum]
    area_under = np.trapezoid(y, x)
    return float((len(evr) - area_under) / len(evr))


def align_component(vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return vec if float(np.dot(vec, ref)) >= 0 else -vec


def compute_leave_one_out(by_subject: dict[str, dict[int, list[np.ndarray]]], ref_components: np.ndarray) -> list[dict[str, float | str]]:
    pids = sorted(by_subject)
    rows: list[dict[str, float | str]] = []
    for omit in pids:
        keep = [p for p in pids if p != omit]
        X, _ = aggregate_matrix(by_subject, keep)
        pca, _ = fit_pca(X)
        pc1 = align_component(pca.components_[0], ref_components[0])
        pc2 = align_component(pca.components_[1], ref_components[1])
        evr = pca.explained_variance_ratio_
        rows.append(
            {
                "omitted": omit,
                "pc1_cos": float(abs(np.dot(pc1, ref_components[0]))),
                "pc2_cos": float(abs(np.dot(pc2, ref_components[1]))),
                "pc1_var": float(evr[0]),
                "pc2_var": float(evr[1]),
                "pc12_var": float(evr[:2].sum()),
                "vaf_ci": vaf_ci(evr),
            }
        )
    return rows


def save_variance(evr: np.ndarray, ci: float) -> Path:
    cum = np.cumsum(evr)
    n = len(evr)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    axes[0].bar(range(1, n + 1), evr * 100, color="#4C78A8", edgecolor="#26384F", linewidth=0.6)
    axes[0].set_xlabel("主成分编号")
    axes[0].set_ylabel("方差解释率/%")
    axes[0].set_xticks(range(1, n + 1))
    axes[0].set_ylim(0, 86)
    axes[0].grid(axis="y", alpha=0.25)
    for i in range(4):
        axes[0].text(i + 1, evr[i] * 100 + 1.1, f"{evr[i] * 100:.1f}", ha="center", fontsize=10)
    axes[0].text(0.02, 0.94, "(a)", transform=axes[0].transAxes, fontweight="bold", fontsize=13)

    axes[1].plot(range(1, n + 1), cum * 100, "o-", color="#F58518", lw=2, markersize=5)
    for y, label, ls in [(90, "90%", "--"), (95, "95%", ":"), (99, "99%", "-.")]:
        axes[1].axhline(y, color="#666666", ls=ls, lw=1.0, alpha=0.85, label=label)
    axes[1].fill_between(range(1, n + 1), cum * 100, 100, color="#F58518", alpha=0.14)
    axes[1].set_xlabel("保留主成分数量")
    axes[1].set_ylabel("累计方差解释率/%")
    axes[1].set_xticks(range(1, n + 1))
    axes[1].set_ylim(75, 101.5)
    axes[1].grid(alpha=0.24)
    axes[1].legend(loc="lower right", frameon=False)
    axes[1].text(0.02, 0.94, "(b)", transform=axes[1].transAxes, fontweight="bold", fontsize=13)
    axes[1].text(
        0.48,
        0.18,
        f"归一化面积指标 = {ci:.3f}",
        transform=axes[1].transAxes,
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#B0B0B0", "lw": 0.8},
    )

    fig.tight_layout()
    out = OUT_DIR / "3_1_1_pca_variance_v06.png"
    fig.savefig(out, bbox_inches="tight", dpi=240)
    plt.close(fig)
    return out


def save_loadings(components: np.ndarray, evr: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), sharey=True)
    for idx, ax in enumerate(axes):
        vals = components[idx]
        colors = ["#4C78A8" if v >= 0 else "#E45756" for v in vals]
        if idx == 0:
            colors = ["#4C78A8"] * len(vals)
        ax.bar(range(N_JOINTS), vals, color=colors, edgecolor="#303030", linewidth=0.45)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(JOINT_LABELS, fontsize=9.5)
        ax.set_ylim(-0.62, 0.62)
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.02,
            0.92,
            f"({chr(ord('a') + idx)}) PC{idx + 1}, {evr[idx] * 100:.1f}%",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
        )
    axes[0].set_ylabel("主成分载荷")
    fig.tight_layout()
    out = OUT_DIR / "3_1_2_pca_loadings_v06.png"
    fig.savefig(out, bbox_inches="tight", dpi=240)
    plt.close(fig)
    return out


def save_task_space(scores: np.ndarray, evr: np.ndarray) -> Path:
    palette = {
        "圆柱抓取": "#4C78A8",
        "球状抓取": "#F58518",
        "提拉抓取": "#54A24B",
        "三指捏": "#E45756",
        "对指捏": "#B279A2",
        "复合抓取": "#8D6D63",
    }
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    for cat, color in palette.items():
        idx = [i for i, tid in enumerate(GRASP_TASK_IDS) if CATEGORY[tid] == cat]
        ax.scatter(
            scores[idx, 0],
            scores[idx, 1],
            s=55,
            color=color,
            label=cat,
            edgecolor="#202020",
            linewidth=0.45,
            alpha=0.88,
        )
    for i, tid in enumerate(GRASP_TASK_IDS):
        ax.text(scores[i, 0], scores[i, 1] + 0.85, str(tid), ha="center", fontsize=8.5, alpha=0.78)

    ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.axvline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"PC1：整体抓取活动强度（{evr[0] * 100:.1f}%）")
    ax.set_ylabel(f"PC2：桡侧-尺侧活动梯度（{evr[1] * 100:.1f}%）")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=10, frameon=True)
    fig.tight_layout()
    out = OUT_DIR / "3_1_3_pca_task_space_v06.png"
    fig.savefig(out, bbox_inches="tight", dpi=240)
    plt.close(fig)
    return out


def save_stability(loo: list[dict[str, float | str]]) -> Path:
    pc1_cos = np.array([float(r["pc1_cos"]) for r in loo])
    pc2_cos = np.array([float(r["pc2_cos"]) for r in loo])
    pc12 = np.array([float(r["pc12_var"]) for r in loo])
    ci = np.array([float(r["vaf_ci"]) for r in loo])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    bp = axes[0].boxplot([pc1_cos, pc2_cos], patch_artist=True, labels=["PC1", "PC2"], widths=0.45)
    for patch, color in zip(bp["boxes"], ["#4C78A8", "#F58518"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
    axes[0].set_ylabel("与全体受试者模型的余弦相似度")
    axes[0].set_ylim(0.8, 1.01)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(0.02, 0.92, "(a)", transform=axes[0].transAxes, fontweight="bold", fontsize=11)

    x = np.arange(len(loo))
    axes[1].bar(x, pc12 * 100, color="#72B7B2", edgecolor="#355C5A", linewidth=0.4, label="PC1+PC2")
    axes[1].plot(x, ci * 100, "o-", color="#E45756", lw=1.4, markersize=3.8, label="面积指标×100")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(r["omitted"]) for r in loo], rotation=45, ha="right", fontsize=7.5)
    axes[1].set_ylabel("百分比/%")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.24)
    axes[1].legend(loc="lower right", frameon=False)
    axes[1].text(0.02, 0.92, "(b)", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "3_1_4_pca_stability_v06.png"
    fig.savefig(out, bbox_inches="tight", dpi=240)
    plt.close(fig)
    return out


def summarize_by_category(scores: np.ndarray) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for cat in sorted(set(CATEGORY.values())):
        idx = [i for i, tid in enumerate(GRASP_TASK_IDS) if CATEGORY[tid] == cat]
        vals = scores[idx, :]
        summary[cat] = {
            "n_tasks": len(idx),
            "pc1_mean": float(vals[:, 0].mean()),
            "pc2_mean": float(vals[:, 1].mean()),
            "pc1_min": float(vals[:, 0].min()),
            "pc1_max": float(vals[:, 0].max()),
            "pc2_min": float(vals[:, 1].min()),
            "pc2_max": float(vals[:, 1].max()),
        }
    return summary


def main() -> None:
    by_subject, subject_counts = load_subject_task_ranges()
    pids = sorted(by_subject)
    X, task_counts = aggregate_matrix(by_subject, pids)
    pca, scores = fit_pca(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    ci = vaf_ci(evr)
    loo = compute_leave_one_out(by_subject, pca.components_)

    paths = {
        "variance": str(save_variance(evr, ci)),
        "loadings": str(save_loadings(pca.components_, evr)),
        "task_space": str(save_task_space(scores, evr)),
        "stability": str(save_stability(loo)),
    }

    stats = {
        "participants": pids,
        "n_participants": len(pids),
        "n_tasks": len(GRASP_TASK_IDS),
        "n_joints": N_JOINTS,
        "total_trials": int(sum(subject_counts.values())),
        "subject_trial_counts": subject_counts,
        "task_trial_counts": task_counts,
        "explained_variance": evr.tolist(),
        "cumulative_variance": cum.tolist(),
        "vaf_ci": ci,
        "n90": int(np.searchsorted(cum, 0.90) + 1),
        "n95": int(np.searchsorted(cum, 0.95) + 1),
        "n99": int(np.searchsorted(cum, 0.99) + 1),
        "components": pca.components_.tolist(),
        "joint_keys": JOINT_KEYS,
        "joint_labels": [label.replace("\n", "") for label in JOINT_LABELS],
        "scores": scores.tolist(),
        "task_ids": GRASP_TASK_IDS,
        "category_by_task": {str(tid): CATEGORY[tid] for tid in GRASP_TASK_IDS},
        "category_summary": summarize_by_category(scores),
        "leave_one_out": loo,
        "leave_one_out_summary": {
            "pc1_cos_mean": float(np.mean([r["pc1_cos"] for r in loo])),
            "pc1_cos_min": float(np.min([r["pc1_cos"] for r in loo])),
            "pc2_cos_mean": float(np.mean([r["pc2_cos"] for r in loo])),
            "pc2_cos_min": float(np.min([r["pc2_cos"] for r in loo])),
            "pc12_mean": float(np.mean([r["pc12_var"] for r in loo])),
            "pc12_min": float(np.min([r["pc12_var"] for r in loo])),
            "pc12_max": float(np.max([r["pc12_var"] for r in loo])),
            "vaf_ci_mean": float(np.mean([r["vaf_ci"] for r in loo])),
            "vaf_ci_min": float(np.min([r["vaf_ci"] for r in loo])),
            "vaf_ci_max": float(np.max([r["vaf_ci"] for r in loo])),
        },
        "figure_paths": paths,
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"participants: {len(pids)}")
    print(f"task matrix: {X.shape}; total trials: {stats['total_trials']}")
    print(f"PC1={evr[0] * 100:.2f}%, PC2={evr[1] * 100:.2f}%, PC1+PC2={(evr[:2].sum()) * 100:.2f}%")
    print(f"n90={stats['n90']}, n95={stats['n95']}, n99={stats['n99']}, VAF-CI={ci:.4f}")
    print(f"LOO PC1 cos mean/min={stats['leave_one_out_summary']['pc1_cos_mean']:.4f}/{stats['leave_one_out_summary']['pc1_cos_min']:.4f}")
    print(f"LOO PC2 cos mean/min={stats['leave_one_out_summary']['pc2_cos_mean']:.4f}/{stats['leave_one_out_summary']['pc2_cos_min']:.4f}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"stats: {STATS_PATH}")


if __name__ == "__main__":
    main()
