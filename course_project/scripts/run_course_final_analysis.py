#!/usr/bin/env python
"""Course final-project analysis for MyoKi hand synergy statistics.

This script intentionally uses only packages already present in the
``hand_synergy`` conda environment: numpy, scipy, scikit-learn, matplotlib and
Pillow.  It builds a trial-level feature table from the preprocessed MyoKi
files, performs PCA as a representation layer, and then adds statistical
inference around that representation:

1. subject-level bootstrap confidence intervals for variance explained and
   a West-style variance-accounted-for complexity index;
2. family-level complexity summaries inspired by West et al.;
3. blocked fixed-effect ANOVA for task-family effects on PC scores;
4. Benjamini-Hochberg corrected pairwise tests;
5. leave-one-subject-out logistic-regression validation.

Outputs are written under ``course_project/outputs`` and are consumed by
``build_report_pdf.py``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io as sio
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSE_DIR = PROJECT_ROOT / "course_project"
OUT_DIR = COURSE_DIR / "outputs"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = COURSE_DIR / "report"

MYOKI_DIR = PROJECT_ROOT / "MyoKi Dataset"
PREPROC_DIR = MYOKI_DIR / "preprocessed"
TASK_XLSX = MYOKI_DIR / "Task_categorization.xlsx"
PARTICIPANT_XLSX = MYOKI_DIR / "Participant_information.xlsx"

N_FRAMES = 100
RANDOM_SEED = 20260609
BOOTSTRAP_N = int(os.environ.get("SYNERGY_BOOTSTRAP_N", "1000"))

JOINT_NAMES = [
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

FLEXION_IDS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13]
NO_ABD_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13]
ALL_IDS = list(range(15))

ACTION_ORDER = [
    "cylindrical",
    "spherical",
    "hook",
    "tripod",
    "pinch",
    "lumbrical",
    "complex",
]

POWER_TYPES = {"cylindrical", "spherical", "hook"}
PRECISION_TYPES = {"tripod", "pinch", "lumbrical"}
OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
]
FAMILY_COLORS = {
    "Power": "#0072B2",
    "Precision": "#D55E00",
    "Complex": "#009E73",
}
FAMILY_CN = {
    "Power": "力量型",
    "Precision": "精细型",
    "Complex": "复杂型",
}

# Fixed course-project task set.  This matches the documented 41 grasp tasks in
# docs/experiment_report_allgrasp_trialrange_none_all23_zh.md and prevents
# non-grasp actions with broad spreadsheet "Action" labels from entering the
# final-project analysis.
GRASP_TASK_TYPE_OVERRIDES = {
    **{tid: "cylindrical" for tid in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]},
    **{tid: "tripod" for tid in [16, 17, 18, 19, 20, 21]},
    **{tid: "spherical" for tid in [28, 29, 30, 31, 32, 33]},
    **{tid: "hook" for tid in [40, 41, 42, 43, 44, 45]},
    **{tid: "complex" for tid in [46, 47, 48, 49, 50, 51, 54, 55, 56]},
    61: "pinch",
}
COURSE_GRASP_TASK_IDS = sorted(GRASP_TASK_TYPE_OVERRIDES)


@dataclass
class PcaResult:
    pca: PCA
    X_task: np.ndarray
    task_ids: List[int]
    scores_task: np.ndarray
    vaf_ci: float
    n80: int
    n90: int
    n95: int


def ensure_dirs() -> None:
    for path in [OUT_DIR, DATA_DIR, FIG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.unicode_minus": False,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def confidence_ellipse(x: np.ndarray, y: np.ndarray, ax: plt.Axes, color: str, n_std: float = 1.7) -> None:
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        (np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        edgecolor=color,
        lw=1.0,
        alpha=0.12,
    )
    ax.add_patch(ell)
    edge = Ellipse(
        (np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        facecolor="none",
        edgecolor=color,
        lw=1.0,
        alpha=0.85,
    )
    ax.add_patch(edge)


def read_xlsx_first_sheet(path: Path) -> List[List[object]]:
    """Read simple values from the first sheet of an xlsx without openpyxl."""
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared.append("".join(texts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        ns_main = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        first_sheet = workbook.find("a:sheets/a:sheet", ns_main)
        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
        target = None
        for rel in rels.findall("r:Relationship", rel_ns):
            if rel.attrib["Id"] == rel_id:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise RuntimeError(f"Cannot find first worksheet in {path}")
        sheet_path = "xl/" + target.lstrip("/")
        sheet = ET.fromstring(zf.read(sheet_path))

    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: Dict[int, Dict[int, object]] = defaultdict(dict)
    max_row = 0
    max_col = 0
    for c in sheet.findall(".//a:c", ns):
        ref = c.attrib.get("r", "A1")
        match = re.match(r"([A-Z]+)(\d+)", ref)
        if not match:
            continue
        col_letters, row_str = match.groups()
        row = int(row_str)
        col = 0
        for ch in col_letters:
            col = col * 26 + (ord(ch) - 64)
        value = None
        cell_type = c.attrib.get("t")
        if cell_type == "s":
            v = c.find("a:v", ns)
            if v is not None and v.text is not None:
                value = shared[int(v.text)]
        elif cell_type == "inlineStr":
            texts = [t.text or "" for t in c.findall(".//a:t", ns)]
            value = "".join(texts)
        else:
            v = c.find("a:v", ns)
            if v is not None and v.text is not None:
                txt = v.text
                try:
                    f = float(txt)
                    value = int(f) if f.is_integer() else f
                except ValueError:
                    value = txt
        rows[row][col] = value
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    table = []
    for r in range(1, max_row + 1):
        table.append([rows[r].get(c) for c in range(1, max_col + 1)])
    return table


def normalize_marker(value: object) -> bool:
    return str(value).strip().lower() == "x"


def load_task_metadata() -> Dict[int, Dict[str, object]]:
    rows = read_xlsx_first_sheet(TASK_XLSX)
    header1 = rows[0]
    header2 = rows[1]
    action_start = None
    wrist_start = None
    for idx, value in enumerate(header1):
        if value == "Action":
            action_start = idx
        if value == "Wrist":
            wrist_start = idx
    if action_start is None:
        raise RuntimeError("Cannot identify Action columns in Task_categorization.xlsx")
    action_end = wrist_start if wrist_start is not None else len(header1)
    action_cols = {}
    for idx in range(action_start, action_end):
        label = header2[idx]
        if label:
            action_cols[str(label).strip()] = idx

    meta = {}
    for row in rows[3:]:
        if not row or row[0] is None:
            continue
        task_id = int(row[0])
        task_name = str(row[1]) if row[1] is not None else f"Task {task_id}"
        actions = [name for name, idx in action_cols.items() if idx < len(row) and normalize_marker(row[idx])]
        if "none" in actions:
            actions = [a for a in actions if a != "none"]
        if actions:
            primary = next((a for a in ACTION_ORDER if a in actions), actions[0])
        else:
            primary = "none"
        if primary in POWER_TYPES:
            family = "Power"
        elif primary in PRECISION_TYPES:
            family = "Precision"
        elif primary == "complex":
            family = "Complex"
        else:
            family = "Other"
        meta[task_id] = {
            "task_id": task_id,
            "task_name": task_name,
            "actions": actions,
            "grasp_type": primary,
            "grasp_family": family,
        }
    for tid, primary in GRASP_TASK_TYPE_OVERRIDES.items():
        if tid in meta:
            if primary in POWER_TYPES:
                family = "Power"
            elif primary in PRECISION_TYPES:
                family = "Precision"
            else:
                family = "Complex"
            meta[tid]["actions"] = [primary]
            meta[tid]["grasp_type"] = primary
            meta[tid]["grasp_family"] = family
    return meta


def load_participant_metadata() -> Dict[str, Dict[str, object]]:
    rows = read_xlsx_first_sheet(PARTICIPANT_XLSX)
    header = [str(x) if x is not None else "" for x in rows[0]]
    idx = {name: i for i, name in enumerate(header)}
    meta = {}
    for row in rows[1:]:
        if not row or row[0] is None:
            continue
        pid = str(row[idx["Participant"]]).strip()
        meta[pid] = {
            "participant": pid,
            "age": float(row[idx["Age"]]),
            "sex": str(row[idx["Sex"]]).strip(),
            "height_cm": float(row[idx["Height [cm]"]]),
            "weight_kg": float(row[idx["Weight [kg]"]]),
            "handedness": str(row[idx["Handedness"]]).strip(),
        }
    return meta


def build_trial_table(task_meta: Dict[int, Dict[str, object]], participant_meta: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    files = sorted(PREPROC_DIR.glob("P*_preprocessed.mat"))
    for path in files:
        pid = path.name.split("_")[0]
        pinfo = participant_meta.get(pid)
        if pinfo is None or str(pinfo.get("handedness", "")).lower() != "right":
            continue
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        angles = np.asarray(mat["angles"], dtype=float)
        trial_tasks = np.asarray(mat["trial_tasks"], dtype=int).ravel()
        n_trials = int(mat["num_trials"])
        if angles.shape[0] != n_trials * N_FRAMES:
            raise ValueError(f"{path.name}: angles shape {angles.shape} does not match {n_trials} trials")
        trials = angles.reshape(n_trials, N_FRAMES, angles.shape[1])
        for i, task_id in enumerate(trial_tasks):
            if int(task_id) not in COURSE_GRASP_TASK_IDS:
                continue
            tm = task_meta.get(int(task_id))
            if tm is None or tm["grasp_family"] not in {"Power", "Precision", "Complex"}:
                continue
            arr = trials[i]
            static5 = arr[-5:].mean(axis=0)
            static25 = arr[-25:].mean(axis=0)
            trial_range = arr.max(axis=0) - arr.min(axis=0)
            trial_sd = arr.std(axis=0, ddof=1)
            rec: Dict[str, object] = {
                "participant": pid,
                "age": pinfo["age"],
                "sex": pinfo["sex"],
                "task_id": int(task_id),
                "task_name": tm["task_name"],
                "grasp_type": tm["grasp_type"],
                "grasp_family": tm["grasp_family"],
                "trial_index": i,
            }
            for j, name in enumerate(JOINT_NAMES):
                rec[f"range_{name}"] = float(trial_range[j])
                rec[f"static5_{name}"] = float(static5[j])
                rec[f"static25_{name}"] = float(static25[j])
                rec[f"sd_{name}"] = float(trial_sd[j])
            records.append(rec)
    return records


def write_records_csv(records: List[Dict[str, object]], path: Path) -> None:
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def matrix_from_records(records: List[Dict[str, object]], prefix: str = "range", jids: Sequence[int] = ALL_IDS) -> np.ndarray:
    return np.array([[float(r[f"{prefix}_{JOINT_NAMES[j]}"]) for j in jids] for r in records], dtype=float)


def vaf_complexity_index(explained_variance_ratio: Sequence[float]) -> float:
    """West-style normalized area above the cumulative VAF curve.

    The index is near zero when most variance is captured by the first few
    synergies, and increases as more independent degrees of freedom are needed.
    """
    evr = np.asarray(explained_variance_ratio, dtype=float)
    if evr.size == 0:
        return float("nan")
    cum = np.cumsum(evr)
    return float(np.sum(1.0 - cum) / len(evr))


def pc_count_for_vaf(explained_variance_ratio: Sequence[float], threshold: float) -> int:
    cum = np.cumsum(np.asarray(explained_variance_ratio, dtype=float))
    return int(np.searchsorted(cum, threshold, side="left") + 1)


def fit_task_pca(records: List[Dict[str, object]], prefix: str = "range", jids: Sequence[int] = ALL_IDS) -> PcaResult:
    task_ids = sorted({int(r["task_id"]) for r in records})
    rows = []
    for tid in task_ids:
        sub = [r for r in records if int(r["task_id"]) == tid]
        rows.append(matrix_from_records(sub, prefix=prefix, jids=jids).mean(axis=0))
    X_task = np.vstack(rows)
    pca = PCA(n_components=min(len(jids), len(task_ids))).fit(X_task)
    scores_task = pca.transform(X_task)
    evr = pca.explained_variance_ratio_
    return PcaResult(
        pca=pca,
        X_task=X_task,
        task_ids=task_ids,
        scores_task=scores_task,
        vaf_ci=vaf_complexity_index(evr),
        n80=pc_count_for_vaf(evr, 0.80),
        n90=pc_count_for_vaf(evr, 0.90),
        n95=pc_count_for_vaf(evr, 0.95),
    )


def project_trial_scores(records: List[Dict[str, object]], pca_result: PcaResult, prefix: str = "range", jids: Sequence[int] = ALL_IDS) -> np.ndarray:
    X = matrix_from_records(records, prefix=prefix, jids=jids)
    return pca_result.pca.transform(X)


def ci(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return tuple(np.quantile(arr, [alpha / 2, 1 - alpha / 2]).tolist())  # type: ignore[return-value]


def bootstrap_pca(records: List[Dict[str, object]], n_boot: int, rng: np.random.Generator) -> Dict[str, object]:
    by_pid: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in records:
        by_pid[str(r["participant"])].append(r)
    pids = sorted(by_pid)
    boot = defaultdict(list)
    for _ in range(n_boot):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        sample_records: List[Dict[str, object]] = []
        for pid in sampled:
            sample_records.extend(by_pid[str(pid)])
        res = fit_task_pca(sample_records)
        boot["pc1"].append(float(res.pca.explained_variance_ratio_[0]))
        boot["pc2"].append(float(res.pca.explained_variance_ratio_[1]))
        boot["pc1_pc2"].append(float(res.pca.explained_variance_ratio_[:2].sum()))
        boot["vaf_ci"].append(res.vaf_ci)
        boot["n90"].append(res.n90)
    summary = {}
    for key, vals in boot.items():
        summary[key] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "ci95": [float(x) for x in ci(vals)],
            "values": [float(x) for x in vals],
        }
    return summary


def family_complexity(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for fam in ["Power", "Precision", "Complex"]:
        sub = [r for r in records if str(r["grasp_family"]) == fam]
        res = fit_task_pca(sub)
        out.append(
            {
                "family": fam,
                "n_tasks": len(res.task_ids),
                "n_trials": len(sub),
                "pc1": float(res.pca.explained_variance_ratio_[0]),
                "pc2": float(res.pca.explained_variance_ratio_[1]),
                "pc1_pc2": float(res.pca.explained_variance_ratio_[:2].sum()),
                "vaf_ci": res.vaf_ci,
                "n80": res.n80,
                "n90": res.n90,
                "n95": res.n95,
            }
        )
    return out


def one_hot(values: Sequence[object], drop_first: bool = True) -> Tuple[np.ndarray, List[str]]:
    labels = sorted({str(v) for v in values})
    if drop_first:
        labels = labels[1:]
    mat = np.zeros((len(values), len(labels)), dtype=float)
    for i, val in enumerate(values):
        sval = str(val)
        for j, lab in enumerate(labels):
            if sval == lab:
                mat[i, j] = 1.0
    return mat, labels


def design_matrix(records: List[Dict[str, object]], include_family: bool = True, pair: Tuple[str, str] | None = None) -> Tuple[np.ndarray, List[str]]:
    cols = [np.ones(len(records), dtype=float)]
    names = ["Intercept"]
    if include_family:
        if pair is None:
            mat, labs = one_hot([r["grasp_family"] for r in records], drop_first=True)
            cols.append(mat)
            names.extend([f"family={lab}" for lab in labs])
        else:
            a, b = pair
            binary = np.array([1.0 if str(r["grasp_family"]) == b else 0.0 for r in records])[:, None]
            cols.append(binary)
            names.append(f"{b}_vs_{a}")
    subj, labs = one_hot([r["participant"] for r in records], drop_first=True)
    if subj.size:
        cols.append(subj)
        names.extend([f"subject={lab}" for lab in labs])
    return np.column_stack(cols), names


def ols_sse_rank(y: np.ndarray, X: np.ndarray) -> Tuple[float, int, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(np.dot(resid, resid))
    rank = int(np.linalg.matrix_rank(X))
    return sse, rank, beta


def nested_f_test(y: np.ndarray, X_full: np.ndarray, X_reduced: np.ndarray) -> Dict[str, float]:
    sse_f, rank_f, _ = ols_sse_rank(y, X_full)
    sse_r, rank_r, _ = ols_sse_rank(y, X_reduced)
    df_num = rank_f - rank_r
    df_den = len(y) - rank_f
    ss_effect = max(sse_r - sse_f, 0.0)
    ms_effect = ss_effect / df_num
    ms_error = sse_f / df_den
    f_stat = ms_effect / ms_error if ms_error > 0 else np.inf
    p = float(stats.f.sf(f_stat, df_num, df_den))
    partial_eta2 = ss_effect / (ss_effect + sse_f) if (ss_effect + sse_f) > 0 else 0.0
    return {
        "F": float(f_stat),
        "p": p,
        "df_num": int(df_num),
        "df_den": int(df_den),
        "partial_eta2": float(partial_eta2),
        "sse_full": float(sse_f),
        "sse_reduced": float(sse_r),
    }


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for rank_idx in range(n - 1, -1, -1):
        idx = order[rank_idx]
        rank = rank_idx + 1
        val = min(prev, p[idx] * n / rank)
        adjusted[idx] = val
        prev = val
    return [float(min(x, 1.0)) for x in adjusted]


def blocked_anova(records: List[Dict[str, object]], trial_scores: np.ndarray) -> Dict[str, object]:
    out = {"overall": {}, "pairwise": []}
    X_full, _ = design_matrix(records, include_family=True)
    X_reduced, _ = design_matrix(records, include_family=False)
    for pc_idx in [0, 1]:
        y = trial_scores[:, pc_idx]
        out["overall"][f"PC{pc_idx + 1}"] = nested_f_test(y, X_full, X_reduced)

    pairs = list(combinations(["Power", "Precision", "Complex"], 2))
    p_values = []
    entries = []
    for pc_idx in [0, 1]:
        for pair in pairs:
            sub_idx = [i for i, r in enumerate(records) if str(r["grasp_family"]) in pair]
            sub_records = [records[i] for i in sub_idx]
            y = trial_scores[sub_idx, pc_idx]
            Xf, _ = design_matrix(sub_records, include_family=True, pair=pair)
            Xr, _ = design_matrix(sub_records, include_family=False)
            test = nested_f_test(y, Xf, Xr)
            means = {
                fam: float(np.mean([trial_scores[i, pc_idx] for i in sub_idx if str(records[i]["grasp_family"]) == fam]))
                for fam in pair
            }
            entry = {
                "pc": f"PC{pc_idx + 1}",
                "family_a": pair[0],
                "family_b": pair[1],
                "mean_a": means[pair[0]],
                "mean_b": means[pair[1]],
                "difference_b_minus_a": means[pair[1]] - means[pair[0]],
                **test,
            }
            entries.append(entry)
            p_values.append(test["p"])
    adj = bh_fdr(p_values)
    for entry, q in zip(entries, adj):
        entry["q_bh"] = q
    out["pairwise"] = entries
    return out


def leave_subject_out_prediction(records: List[Dict[str, object]]) -> Dict[str, object]:
    X = matrix_from_records(records, prefix="range", jids=ALL_IDS)
    y_labels = np.array([str(r["grasp_family"]) for r in records])
    groups = np.array([str(r["participant"]) for r in records])
    class_names = sorted(np.unique(y_labels).tolist())
    label_to_int = {lab: i for i, lab in enumerate(class_names)}
    y = np.array([label_to_int[v] for v in y_labels])
    variants = {
        "Raw15": None,
        "PC2": 2,
        "PC3": 3,
        "PC5": 5,
    }
    logo = LeaveOneGroupOut()
    summary = {}
    all_confusions = {}
    for name, n_pc in variants.items():
        bal_acc = []
        macro_f1 = []
        accuracy = []
        cm_total = np.zeros((len(class_names), len(class_names)), dtype=int)
        for train_idx, test_idx in logo.split(X, y, groups):
            scaler = StandardScaler().fit(X[train_idx])
            X_train = scaler.transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            if n_pc is not None:
                pca = PCA(n_components=n_pc).fit(X_train)
                X_train_model = pca.transform(X_train)
                X_test_model = pca.transform(X_test)
            else:
                X_train_model = X_train
                X_test_model = X_test
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
            clf.fit(X_train_model, y[train_idx])
            pred = clf.predict(X_test_model)
            bal_acc.append(float(balanced_accuracy_score(y[test_idx], pred)))
            macro_f1.append(float(f1_score(y[test_idx], pred, average="macro")))
            accuracy.append(float(np.mean(pred == y[test_idx])))
            cm_total += confusion_matrix(y[test_idx], pred, labels=list(range(len(class_names))))
        summary[name] = {
            "balanced_accuracy_mean": float(np.mean(bal_acc)),
            "balanced_accuracy_sd": float(np.std(bal_acc, ddof=1)),
            "macro_f1_mean": float(np.mean(macro_f1)),
            "macro_f1_sd": float(np.std(macro_f1, ddof=1)),
            "accuracy_mean": float(np.mean(accuracy)),
            "accuracy_sd": float(np.std(accuracy, ddof=1)),
            "fold_balanced_accuracy": bal_acc,
            "fold_macro_f1": macro_f1,
        }
        all_confusions[name] = cm_total.tolist()
    best = max(summary, key=lambda k: summary[k]["balanced_accuracy_mean"])
    return {
        "class_names": class_names,
        "summary": summary,
        "confusion_matrices": all_confusions,
        "best_variant": best,
    }


def sensitivity_analysis(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    variants = [
        ("All 15 joints", ALL_IDS),
        ("No abduction (12D)", NO_ABD_IDS),
        ("Flexion only (10D)", FLEXION_IDS),
    ]
    out = []
    for label, jids in variants:
        res = fit_task_pca(records, prefix="range", jids=jids)
        out.append(
            {
                "label": label,
                "n_joints": len(jids),
                "pc1": float(res.pca.explained_variance_ratio_[0]),
                "pc2": float(res.pca.explained_variance_ratio_[1]),
                "pc1_pc2": float(res.pca.explained_variance_ratio_[:2].sum()),
                "vaf_ci": res.vaf_ci,
                "n80": res.n80,
                "n90": res.n90,
                "n95": res.n95,
            }
        )
    return out


def save_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_table_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_figure1(records: List[Dict[str, object]], pca_result: PcaResult, trial_scores: np.ndarray, bootstrap: Dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.6, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    counts = Counter(str(r["grasp_family"]) for r in records)
    families = ["Power", "Precision", "Complex"]
    ax_a.bar([FAMILY_CN[f] for f in families], [counts[f] for f in families], color=[FAMILY_COLORS[f] for f in families], edgecolor="black", linewidth=0.4)
    ax_a.set_ylabel("试次数")
    ax_a.set_title("MyoKi 抓取试次")
    for i, f in enumerate(families):
        ax_a.text(i, counts[f] + max(counts.values()) * 0.02, str(counts[f]), ha="center", va="bottom", fontsize=7)
    despine(ax_a)
    panel_label(ax_a, "A")

    evr = pca_result.pca.explained_variance_ratio_[:8] * 100
    ax_b.bar(np.arange(1, len(evr) + 1), evr, color="#4D4D4D", edgecolor="black", linewidth=0.4)
    ax_b.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), color="#D55E00", marker="o", ms=3, lw=1.2)
    ci2 = bootstrap["pc1_pc2"]["ci95"]
    ax_b.axhspan(ci2[0] * 100, ci2[1] * 100, color="#D55E00", alpha=0.12, lw=0)
    ax_b.set_xlabel("主成分序号")
    ax_b.set_ylabel("方差解释率（%）")
    ax_b.set_title("任务均值主成分分析")
    ax_b.set_ylim(0, 105)
    despine(ax_b)
    panel_label(ax_b, "B")

    load = pca_result.pca.components_[:2]
    x = np.arange(len(JOINT_NAMES))
    ax_c.axhline(0, color="black", lw=0.6)
    ax_c.plot(x, load[0], color="#0072B2", marker="o", ms=3, lw=1.0, label="PC1")
    ax_c.plot(x, load[1], color="#D55E00", marker="s", ms=3, lw=1.0, label="PC2")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(JOINT_NAMES, rotation=70, ha="right")
    ax_c.set_ylabel("载荷")
    ax_c.set_title("前两种特征姿态的关节载荷")
    ax_c.legend(frameon=False, ncol=2, loc="upper right")
    despine(ax_c)
    panel_label(ax_c, "C")

    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = np.arange(len(records))
    if len(sample_idx) > 1800:
        sample_idx = rng.choice(sample_idx, size=1800, replace=False)
    for fam in families:
        idx = np.array([i for i in sample_idx if str(records[i]["grasp_family"]) == fam])
        ax_d.scatter(trial_scores[idx, 0], trial_scores[idx, 1], s=6, alpha=0.28, color=FAMILY_COLORS[fam], label=FAMILY_CN[fam], linewidths=0)
        full_idx = np.array([i for i, r in enumerate(records) if str(r["grasp_family"]) == fam])
        confidence_ellipse(trial_scores[full_idx, 0], trial_scores[full_idx, 1], ax_d, FAMILY_COLORS[fam])
    ax_d.set_xlabel("第一主成分得分")
    ax_d.set_ylabel("第二主成分得分")
    ax_d.set_title("试次在协同空间中的投影")
    ax_d.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    despine(ax_d)
    panel_label(ax_d, "D")

    fig.savefig(FIG_DIR / "figure1_pca_overview.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure1_pca_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_figure2(bootstrap: Dict[str, object], complexity: List[Dict[str, object]], anova: Dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.6, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    metrics = [
        ("前两主成分\n方差解释率", bootstrap["pc1_pc2"]["values"], "#0072B2"),
    ]
    positions = np.arange(len(metrics))
    for i, (_, vals, color) in enumerate(metrics):
        arr = np.asarray(vals)
        ax_a.violinplot(arr, positions=[i], widths=0.72, showmeans=False, showextrema=False)
        parts = ax_a.collections[-1]
        parts.set_facecolor(color)
        parts.set_alpha(0.35)
        q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        lo, hi = ci(arr)
        ax_a.plot([i, i], [lo, hi], color="black", lw=1.0)
        ax_a.plot([i - 0.18, i + 0.18], [med, med], color="black", lw=1.4)
        ax_a.plot([i - 0.12, i + 0.12], [q1, q1], color="black", lw=0.8)
        ax_a.plot([i - 0.12, i + 0.12], [q3, q3], color="black", lw=0.8)
    ax_a.set_xticks(positions)
    ax_a.set_xticklabels([m[0] for m in metrics], rotation=20, ha="right")
    all_vals = np.asarray(bootstrap["pc1_pc2"]["values"], dtype=float)
    ax_a.set_ylim(max(0.0, all_vals.min() - 0.02), min(1.0, all_vals.max() + 0.02))
    ax_a.set_ylabel("bootstrap 估计值")
    ax_a.set_title("受试者层面的稳定性")
    despine(ax_a)
    panel_label(ax_a, "A")

    fams = [c["family"] for c in complexity]
    x = np.arange(len(fams))
    vals = [c["vaf_ci"] for c in complexity]
    ax_b.bar(x, vals, color=[FAMILY_COLORS[f] for f in fams], edgecolor="black", linewidth=0.4)
    for i, c in enumerate(complexity):
        ax_b.text(i, vals[i] + 0.006, f"90%:{c['n90']}个主成分", ha="center", va="bottom", fontsize=7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["力量型", "精细型", "复杂型"])
    ax_b.set_ylabel("VAF-CI")
    ax_b.set_title("按抓取家族计算的 VAF-CI")
    ax_b.set_ylim(0, max(vals) * 1.35)
    despine(ax_b)
    panel_label(ax_b, "B")

    pcs = ["PC1", "PC2"]
    eta = [anova["overall"][pc]["partial_eta2"] for pc in pcs]
    q = [anova["overall"][pc]["p"] for pc in pcs]
    ax_c.bar(pcs, eta, color=["#0072B2", "#D55E00"], edgecolor="black", linewidth=0.4)
    for i, pc in enumerate(pcs):
        if q[i] == 0 or q[i] < 1e-300:
            p_label = r"$p<10^{-300}$"
        elif q[i] < 1e-4:
            mantissa, exponent = f"{q[i]:.1e}".split("e")
            p_label = rf"$p={mantissa}\times 10^{{{int(exponent)}}}$"
        else:
            p_label = f"p={q[i]:.4f}"
        ax_c.text(i, eta[i] + 0.01, p_label, ha="center", va="bottom", fontsize=7)
    ax_c.set_ylim(0, max(eta) * 1.35)
    ax_c.set_ylabel("偏 eta 平方")
    ax_c.set_title("blocked ANOVA：抓取家族效应")
    despine(ax_c)
    panel_label(ax_c, "C")

    pairs = [("Power", "Precision"), ("Power", "Complex"), ("Precision", "Complex")]
    heat = np.ones((2, len(pairs)))
    labels = []
    short_cn = {"Power": "力", "Precision": "精", "Complex": "复"}
    for j, pair in enumerate(pairs):
        labels.append(f"{short_cn[pair[0]]}-{short_cn[pair[1]]}")
        for i, pc in enumerate(pcs):
            match = [e for e in anova["pairwise"] if e["pc"] == pc and e["family_a"] == pair[0] and e["family_b"] == pair[1]]
            if match:
                heat[i, j] = -math.log10(max(float(match[0]["q_bh"]), 1e-300))
    im = ax_d.imshow(heat, cmap="viridis", vmin=0)
    ax_d.set_xticks(np.arange(len(pairs)))
    ax_d.set_xticklabels(labels)
    ax_d.set_yticks(np.arange(2))
    ax_d.set_yticklabels(pcs)
    for i in range(2):
        for j in range(len(pairs)):
            ax_d.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", color="white" if heat[i, j] > heat.max() / 2 else "black", fontsize=7)
    ax_d.set_title("两两比较（-log10 假发现率 q）")
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    panel_label(ax_d, "D")

    fig.savefig(FIG_DIR / "figure2_inference.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure2_inference.png", bbox_inches="tight")
    plt.close(fig)


def plot_figure3(prediction: Dict[str, object], sensitivity: List[Dict[str, object]]) -> None:
    fig = plt.figure(figsize=(7.6, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    variants = ["PC2", "PC3", "PC5", "Raw15"]
    variant_labels = ["前2主成分", "前3主成分", "前5主成分", "原始15维"]
    means = [prediction["summary"][v]["balanced_accuracy_mean"] for v in variants]
    sds = [prediction["summary"][v]["balanced_accuracy_sd"] for v in variants]
    ax_a.bar(variant_labels, means, yerr=sds, color=OKABE_ITO[:4], edgecolor="black", linewidth=0.4, capsize=2)
    ax_a.tick_params(axis="x", labelrotation=15)
    ax_a.axhline(1 / len(prediction["class_names"]), color="black", lw=0.8, ls="--")
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("平衡准确率")
    ax_a.set_title("留一受试者预测")
    despine(ax_a)
    panel_label(ax_a, "A")

    best = prediction["best_variant"]
    cm = np.asarray(prediction["confusion_matrices"][best], dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax_b.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labs = [FAMILY_CN.get(lab, lab) for lab in prediction["class_names"]]
    ax_b.set_xticks(np.arange(len(labs)))
    ax_b.set_xticklabels(labs, rotation=20, ha="right")
    ax_b.set_yticks(np.arange(len(labs)))
    ax_b.set_yticklabels(labs)
    ax_b.set_xlabel("预测类别")
    ax_b.set_ylabel("真实类别")
    ax_b.set_title(f"混淆矩阵（{'原始15维' if best == 'Raw15' else best}）")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax_b.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    panel_label(ax_b, "B")

    labels = [s["label"] for s in sensitivity]
    x = np.arange(len(labels))
    width = 0.34
    bars1 = ax_c.bar(x - width / 2, [s["pc1_pc2"] for s in sensitivity], width, label="前两主成分方差解释率", color="#0072B2")
    ax_c2 = ax_c.twinx()
    bars2 = ax_c2.bar(x + width / 2, [s["vaf_ci"] for s in sensitivity], width, label="VAF-CI", color="#D55E00")
    ax_c.set_ylim(0.86, 0.96)
    ax_c2.set_ylim(0, max(s["vaf_ci"] for s in sensitivity) * 1.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["全部15关节", "去除外展", "仅屈曲"], rotation=15, ha="right")
    ax_c.set_ylabel("方差解释率")
    ax_c2.set_ylabel("VAF-CI")
    ax_c.set_title("关节集敏感性分析")
    ax_c.legend([bars1, bars2], ["前两主成分方差解释率", "VAF-CI"], frameon=False, ncol=1, loc="upper center", bbox_to_anchor=(0.48, 1.04))
    despine(ax_c)
    ax_c2.spines["top"].set_visible(False)
    ax_c2.tick_params(direction="out", length=3, width=0.7)
    panel_label(ax_c, "C")

    # Compact text panel with the main statistical takeaways.
    ax_d.axis("off")
    text = (
        f"最佳预测表征：{'原始15维' if best == 'Raw15' else best}\n"
        f"平衡准确率：{prediction['summary'][best]['balanced_accuracy_mean']:.2f} "
        f"+/- {prediction['summary'][best]['balanced_accuracy_sd']:.2f}\n\n"
        "解释：\n"
        "- 低维主成分保留部分任务家族信息。\n"
        "- 原始15维特征提供预测上界参照。\n"
        "- 留出整名受试者检验跨个体泛化。\n\n"
        "统计保护：\n"
        "主成分分析之后加入 bootstrap、\n"
        "blocked ANOVA、假发现率校正和预测验证。"
    )
    ax_d.text(0.0, 0.98, text, va="top", ha="left", fontsize=8, linespacing=1.35)
    panel_label(ax_d, "D")

    fig.savefig(FIG_DIR / "figure3_prediction_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure3_prediction_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def write_latex_report(payload: Dict[str, object]) -> None:
    tex = REPORT_DIR / "final_project_report.tex"
    stats_payload = payload["stats"]
    main = stats_payload["main_pca"]
    anova = stats_payload["anova"]["overall"]
    pred = stats_payload["prediction"]
    complexity = stats_payload["complexity_by_family"]
    best = pred["best_variant"]

    def p_text(p: float) -> str:
        if p == 0:
            return "$<10^{-300}$"
        if p < 1e-4:
            return f"{p:.1e}"
        return f"{p:.4f}"

    pc1_p = p_text(float(anova["PC1"]["p"]))
    pc2_p = p_text(float(anova["PC2"]["p"]))
    text = rf"""\documentclass[10pt,a4paper]{{ctexart}}
\usepackage[margin=1.6cm]{{geometry}}
\usepackage{{graphicx,booktabs,amsmath,caption,float,hyperref}}
\usepackage{{titlesec}}
\setlength{{\parskip}}{{3pt}}
\setlength{{\parindent}}{{1em}}
\titlespacing*{{\section}}{{0pt}}{{4pt}}{{2pt}}
\titlespacing*{{\subsection}}{{0pt}}{{3pt}}{{1pt}}
\captionsetup{{font=small,labelfont=bf}}
\begin{{document}}
\begin{{center}}
{{\Large \bfseries 基于 MyoKi 人手运动学数据的抓取姿态协同统计分析}}\\[4pt]
{{\normalsize 生物统计和生物信息引论 Final Project}}\\[2pt]
\end{{center}}
\noindent\textbf{{摘要}} 本项目将人手抓取姿态协同分析组织为一个统计问题：MyoKi 抓取运动学中是否存在稳定的低维协同结构，该结构是否能解释抓取家族差异，并且是否能跨受试者预测。分析纳入 {payload['n_participants']} 名右利手健康受试者、{payload['n_tasks']} 个抓取任务和 {payload['n_trials']} 个试次。主成分分析作为表征层，后续加入受试者 bootstrap、方差解释复杂度指数（Variance-Accounted-For Complexity Index, VAF-CI）、受试者 blocked ANOVA、Benjamini-Hochberg 假发现率校正和留一受试者交叉验证。前两个主成分解释 {main['pc1_pc2']*100:.1f}\% 任务级方差，整体 VAF-CI 为 {main['vaf_ci']:.3f}，达到 90\% 方差解释率需要 {main['n90']} 个主成分。\\
\textbf{{关键词}} 手部运动学；姿态协同；主成分分析；bootstrap；blocked ANOVA；逻辑回归

\section{{数据来源与研究问题}}
MyoKi 数据集由 CyberGlove II 记录健康受试者在日常生活操作中的手部运动学信号。数据从公开发布网站下载后，按试次边界截取 CyberGlove II 手部关节角时间序列，并通过线性插值把每个试次重采样到 100 帧；随后剔除掌弓通道和两个腕部通道，保留 15 个手指相关关节通道。根据 \texttt{{Task\_categorization.xlsx}} 筛选 41 个抓取任务，并将任务归并为力量型、精细型和复杂型三个抓取家族。研究问题为：1）抓取姿态变化能否由少数协同轴解释；2）方差集中度和 VAF-CI 是否表明任务需要不同程度的个体化关节控制；3）不同抓取家族在协同空间中是否存在统计差异；4）低维协同是否保留预测抓取类别的信息。

\section{{研究方法}}
对每个试次计算 15 个关节的角度活动范围 $\max(\theta_j)-\min(\theta_j)$，得到试次级特征。主成分分析在任务均值矩阵上拟合，以避免重复试次数量主导协方差结构；随后将每个试次投影到同一协同空间。参考 West 等人的方差解释复杂度指数，本研究从累计方差解释曲线计算 VAF-CI：当更多主成分才能解释同样比例的方差时，说明该任务家族需要更强的关节个体化控制。

统计推断包括四部分。首先，以受试者为重采样单位进行 {BOOTSTRAP_N} 次 bootstrap，估计方差解释率和 VAF-CI 的 95\% 置信区间。其次，分别计算三个抓取家族的 VAF-CI 和达到 90\% 方差解释率所需的主成分数。第三，以主成分得分为因变量，用受试者固定效应进行 blocked ANOVA：$\mathrm{{PC}} \sim \mathrm{{grasp\ family}} + \mathrm{{subject}}$。两两比较使用 Benjamini-Hochberg 假发现率校正。第四，使用留一受试者交叉验证训练多项逻辑回归，比较原始 15 维特征、前 2、3、5 个主成分对抓取家族的预测能力。

\section{{主要结果}}
任务级主成分分析显示，第一主成分解释 {main['pc1']*100:.1f}\% 方差，第二主成分解释 {main['pc2']*100:.1f}\%，前两维合计 {main['pc1_pc2']*100:.1f}\%。受试者 bootstrap 得到前两主成分方差解释率 95\% 置信区间为 [{stats_payload['bootstrap']['pc1_pc2']['ci95'][0]*100:.1f}\%, {stats_payload['bootstrap']['pc1_pc2']['ci95'][1]*100:.1f}\%]。VAF-CI 分析显示，整体数据的 VAF-CI 为 {main['vaf_ci']:.3f}；力量型、精细型和复杂型抓取的 VAF-CI 分别为 {complexity[0]['vaf_ci']:.3f}、{complexity[1]['vaf_ci']:.3f} 和 {complexity[2]['vaf_ci']:.3f}。

受试者 blocked ANOVA 显示，抓取家族对第一主成分的影响显著（F({anova['PC1']['df_num']}, {anova['PC1']['df_den']})={anova['PC1']['F']:.1f}, p={pc1_p}, 偏 $\eta^2$={anova['PC1']['partial_eta2']:.3f}），对第二主成分亦显著（F({anova['PC2']['df_num']}, {anova['PC2']['df_den']})={anova['PC2']['F']:.1f}, p={pc2_p}, 偏 $\eta^2$={anova['PC2']['partial_eta2']:.3f}）。留一受试者交叉验证显示，最佳方案为 {best}，平衡准确率为 {pred['summary'][best]['balanced_accuracy_mean']:.2f}$\pm${pred['summary'][best]['balanced_accuracy_sd']:.2f}。

\section{{图表}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.88\linewidth]{{../outputs/figures/figure1_pca_overview.png}}
\caption{{数据结构、任务级主成分分析、关节载荷与试次投影。}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.88\linewidth]{{../outputs/figures/figure2_inference.png}}
\caption{{bootstrap、VAF-CI、blocked ANOVA 和 FDR 校正的两两比较。}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.88\linewidth]{{../outputs/figures/figure3_prediction_sensitivity.png}}
\caption{{留一受试者预测验证、混淆矩阵和关节集敏感性分析。}}
\end{{figure}}

\section{{结论与局限性}}
本项目支持一个统计上更完整的结论：人手抓取运动学中存在稳定的低维姿态协同结构。主成分分析将 15 维运动学数据转化为可解释表征；bootstrap 量化不确定性，VAF-CI 描述方差集中度，blocked ANOVA 检验任务差异，逻辑回归验证预测价值。局限性包括：健康受试者数据不能直接推广至神经损伤或假肢用户；角度活动范围忽略了完整时间序列的相位和速度信息；主成分分析是线性方法；预测验证基于现有任务标签，不能替代真实控制系统或临床性能评估。

\section{{参考文献}}
\begin{{enumerate}}\setlength\itemsep{{2pt}}
\item Santello M, Flanders M, Soechting JF. Postural hand synergies for tool use. \emph{{Journal of Neuroscience}}. 1998;18(23):10105-10115.
\item Jarque-Bou NJ, et al. A database of multimodal myography and hand kinematics during realistic daily life activities. \emph{{Scientific Data}}.
\item Feix T, Romero J, Schmiedmayer H-B, Dollar AM, Kragic D. The GRASP taxonomy of human grasp types. \emph{{IEEE Transactions on Human-Machine Systems}}. 2016;46(1):66-77.
\item West AM, Tessari F, Wang M, Hogan N. The Study of Dexterous Hand Manipulation: A Synergy-Based Complexity Index. \emph{{IEEE Transactions on Medical Robotics and Bionics}}. 2025;7(1):156-163.
\end{{enumerate}}

\section*{{复现说明}}
代码入口：\texttt{{course\_project/scripts/run\_course\_final\_analysis.py}}；课程论文构建：\texttt{{course\_project/scripts/build\_course\_paper.py}}。本科毕业论文中的协同分析章节仅用于理解项目背景，未列入参考文献。
\end{{document}}
"""
    tex.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_plot_style()
    rng = np.random.default_rng(RANDOM_SEED)

    task_meta = load_task_metadata()
    participant_meta = load_participant_metadata()
    records = build_trial_table(task_meta, participant_meta)
    if not records:
        raise RuntimeError("No MyoKi grasp records were built.")

    write_records_csv(records, DATA_DIR / "trial_level_features.csv")

    main_pca = fit_task_pca(records)
    trial_scores = project_trial_scores(records, main_pca)
    for i, r in enumerate(records):
        r["PC1"] = float(trial_scores[i, 0])
        r["PC2"] = float(trial_scores[i, 1])
        r["PC3"] = float(trial_scores[i, 2]) if trial_scores.shape[1] > 2 else 0.0
    write_records_csv(records, DATA_DIR / "trial_level_features_with_pc_scores.csv")

    bootstrap = bootstrap_pca(records, BOOTSTRAP_N, rng)
    complexity = family_complexity(records)
    anova = blocked_anova(records, trial_scores)
    prediction = leave_subject_out_prediction(records)
    sensitivity = sensitivity_analysis(records)

    participant_ids = sorted({str(r["participant"]) for r in records})
    task_ids = sorted({int(r["task_id"]) for r in records})
    family_counts = Counter(str(r["grasp_family"]) for r in records)
    type_counts = Counter(str(r["grasp_type"]) for r in records)
    stats_payload = {
        "random_seed": RANDOM_SEED,
        "main_pca": {
            "pc1": float(main_pca.pca.explained_variance_ratio_[0]),
            "pc2": float(main_pca.pca.explained_variance_ratio_[1]),
            "pc1_pc2": float(main_pca.pca.explained_variance_ratio_[:2].sum()),
            "vaf_ci": main_pca.vaf_ci,
            "n80": main_pca.n80,
            "n90": main_pca.n90,
            "n95": main_pca.n95,
            "task_ids": main_pca.task_ids,
            "components": main_pca.pca.components_[:5].tolist(),
            "explained_variance_ratio": main_pca.pca.explained_variance_ratio_.tolist(),
        },
        "bootstrap": bootstrap,
        "complexity_by_family": complexity,
        "anova": anova,
        "prediction": prediction,
        "sensitivity": sensitivity,
        "family_counts": dict(family_counts),
        "type_counts": dict(type_counts),
    }
    payload = {
        "n_trials": len(records),
        "n_participants": len(participant_ids),
        "participants": participant_ids,
        "n_tasks": len(task_ids),
        "task_ids": task_ids,
        "stats": stats_payload,
    }
    save_json(stats_payload, DATA_DIR / "course_project_stats.json")
    save_json(payload, DATA_DIR / "report_payload.json")
    save_table_csv(sensitivity, DATA_DIR / "sensitivity_analysis.csv")
    save_table_csv(complexity, DATA_DIR / "west_style_complexity_by_family.csv")
    save_table_csv(anova["pairwise"], DATA_DIR / "pairwise_fdr_tests.csv")

    # Save compact tables for quick inspection.
    pred_rows = []
    for label, vals in prediction["summary"].items():
        pred_rows.append({"feature_set": label, **{k: v for k, v in vals.items() if not k.startswith("fold_")}})
    save_table_csv(pred_rows, DATA_DIR / "prediction_summary.csv")
    overall_rows = [{"pc": pc, **vals} for pc, vals in anova["overall"].items()]
    save_table_csv(overall_rows, DATA_DIR / "blocked_anova_summary.csv")

    plot_figure1(records, main_pca, trial_scores, bootstrap)
    plot_figure2(bootstrap, complexity, anova)
    plot_figure3(prediction, sensitivity)
    write_latex_report(payload)

    print(json.dumps({
        "n_trials": len(records),
        "n_participants": len(participant_ids),
        "n_tasks": len(task_ids),
        "pc1_pc2": stats_payload["main_pca"]["pc1_pc2"],
        "vaf_ci": stats_payload["main_pca"]["vaf_ci"],
        "n90": stats_payload["main_pca"]["n90"],
        "best_prediction": prediction["best_variant"],
        "outputs": str(OUT_DIR),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
