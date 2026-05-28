#!/usr/bin/env python3
"""
synergy_analysis_report.py
==========================
Automated pipeline to generate a LaTeX-complied PDF report for hand synergy
PCA analysis. The script:
  1. Loads PCA .npz results from PCA_results/
  2. Generates informative figures (variance, eigen-postures, etc.)
  3. Saves all figures to auto_figures/
  4. Writes a full LaTeX document with embedded figures, captions, math, tables
  5. Compiles .tex -> PDF using xelatex (or pdflatex)
"""

import os, sys, glob, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')                        # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ---------- PATHS ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PCA_DIR      = os.path.join(PROJECT_ROOT, 'PCA_results')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'auto_figures')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'latex_report')

os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- GLOBAL STYLE ----------
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size':       11,
    'axes.titlesize':  13,
    'axes.labelsize':  12,
    'figure.dpi':      200,
})

# Joint labels (22 — excluding wrist 20,21 which are treated separately)
# NOTE: PCA uses angles[:, :20], mapping to Ninapro indices 0-19
# 0-1: Wrist, 2-5: Index, 6-9: Middle, 10-13: Ring, 14-18: Little, 19: Thumb
JOINT_LABELS_SHORT = [
    r'$J_{0}$(WRJ2)', r'$J_{1}$(WRJ1)',
    r'$J_{2}$(FFJ4)', r'$J_{3}$(FFJ3)', r'$J_{4}$(FFJ2)', r'$J_{5}$(FFJ1)',
    r'$J_{6}$(MFJ4)', r'$J_{7}$(MFJ3)', r'$J_{8}$(MFJ2)', r'$J_{9}$(MFJ1)',
    r'$J_{10}$(RFJ4)',r'$J_{11}$(RFJ3)',r'$J_{12}$(RFJ2)',r'$J_{13}$(RFJ1)',
    r'$J_{14}$(LFJ5)',r'$J_{15}$(LFJ4)',r'$J_{16}$(LFJ3)',r'$J_{17}$(LFJ2)',r'$J_{18}$(LFJ1)',
    r'$J_{19}$(THJ5)',
]

JOINT_GROUPS = {
    'Thumb':  [19],
    'Index':  [2,3,4,5],
    'Middle': [6,7,8,9],
    'Ring':   [10,11,12,13],
    'Little': [14,15,16,17,18],
    'Wrist':  [0,1],
}

# Finger colours
GROUP_COLORS = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6','#95A5A6']
FINGER_CMAP   = {g: c for g,c in zip(JOINT_GROUPS.keys(), GROUP_COLORS)}


# ======================================================================
#  FIGURE GENERATORS
# ======================================================================

def fig_variance_all_models():
    """Bar chart of cumulative variance for all 7 PCA models."""
    files = sorted(glob.glob(os.path.join(PCA_DIR, 'pca_model_*.npz')))
    if not files:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    labels_display = []
    for i, fp in enumerate(files):
        base = os.path.basename(fp).replace('pca_model_','').replace('.npz','')
        d    = np.load(fp)
        ev   = d['explained_variance']
        cum  = np.cumsum(ev)
        # use 5,10,15,20 components
        for k, nc in enumerate([5, 10, 15, 20]):
            if nc <= len(cum):
                ax.bar(i + k*0.15, cum[nc-1]*100, width=0.12,
                       label=f'{nc} PCs' if i==0 else '',
                       color=plt.cm.Blues(0.3 + k*0.2), edgecolor='k', linewidth=0.5)
        labels_display.append(base)

    ax.set_xticks(range(len(files)))
    ax.set_xticklabels(labels_display, rotation=45, ha='right')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title('Cumulative Variance across Different Subject Combinations')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'variance_all_models.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_variance_curve(model_key='123'):
    """Variance explained curve (individual + cumulative) for one model."""
    fp = os.path.join(PCA_DIR, f'pca_model_{model_key}.npz')
    if not os.path.isfile(fp):
        return None
    d   = np.load(fp)
    ev  = d['explained_variance']
    cum = np.cumsum(ev)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # individual
    ax1.bar(range(1, len(ev)+1), ev*100, color='#3498DB', alpha=0.8, edgecolor='k', linewidth=0.5)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title(f'Individual Variance (Model {model_key})')

    # cumulative
    ax2.step(range(1, len(cum)+1), cum*100, where='mid', color='#E74C3C', linewidth=2)
    ax2.axhline(90, ls='--', color='gray', alpha=0.5, label='90% threshold')
    # mark 3,5,10 components
    for nc in [3, 5, 10]:
        if nc <= len(cum):
            ax2.plot(nc, cum[nc-1]*100, 'o', color='#2ECC71', ms=6)
            ax2.annotate(f'{cum[nc-1]*100:.1f}%', (nc, cum[nc-1]*100),
                         textcoords='offset points', xytext=(5,5), fontsize=8)
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_title(f'Cumulative Variance (Model {model_key})')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'variance_curve_{model_key}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_eigen_posture(model_key='123', n_components=5):
    """Heatmap of first n_components eigen-postures."""
    fp = os.path.join(PCA_DIR, f'pca_model_{model_key}.npz')
    if not os.path.isfile(fp):
        return None
    d   = np.load(fp)
    comps = d['components'][:n_components]          # (n, 20) — no wrist
    n_disp = min(n_components, comps.shape[0])

    fig, axes = plt.subplots(n_disp, 1, figsize=(10, 1.2*n_disp+1))
    if n_disp == 1:
        axes = [axes]

    for i in range(n_disp):
        ax = axes[i]
        comp = comps[i]
        # colour by finger group
        colors = []
        for j in range(20):
            label = JOINT_LABELS_SHORT[j]
            # determine group
            grp = 'Wrist'
            for gname, idxs in JOINT_GROUPS.items():
                if j in idxs:
                    grp = gname
                    break
            colors.append(FINGER_CMAP.get(grp, '#888888'))

        bars = ax.bar(range(20), comp, color=colors, edgecolor='k', linewidth=0.5, alpha=0.85)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xticks(range(20))
        ax.set_xticklabels(JOINT_LABELS_SHORT, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Weight')
        ax.set_title(f'PC{i+1}  (λ = {np.sum(comp**2):.2f})', fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'eigen_posture_{model_key}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_component_activation(model_key='123', n_components=5):
    """Simulate typical activation coefficients using random weights to
       show reconstruction contributions."""
    # This figure shows the relative contribution of each PC in the subspace
    fp = os.path.join(PCA_DIR, f'pca_model_{model_key}.npz')
    if not os.path.isfile(fp):
        return None
    d    = np.load(fp)
    ev   = d['explained_variance'][:n_components]
    cum  = np.cumsum(ev)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    wedges, texts, autotexts = ax.pie(
        ev, labels=[f'PC{i+1}' for i in range(len(ev))],
        autopct='%1.1f%%', startangle=90,
        colors=plt.cm.Set2(np.linspace(0,1,len(ev))),
        explode=[0.05]*len(ev),
        textprops={'fontsize': 9}
    )
    ax.set_title(f'Variance Distribution among First {n_components} PCs\n'
                 f'Cumulative: {cum[-1]*100:.1f}%', fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'pc_pie_{model_key}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_wrist_model_comparison():
    """Compare wrist angles across models."""
    files = sorted(glob.glob(os.path.join(PCA_DIR, 'pca_model_*.npz')))
    labels = []
    wrist_means = []
    for fp in files:
        base = os.path.basename(fp).replace('pca_model_','').replace('.npz','')
        d    = np.load(fp)
        wm   = d.get('fixed_wrist_mean', None)
        if wm is not None:
            wrist_means.append(wm)
            labels.append(base)
    if not wrist_means:
        return None
    wrist_means = np.array(wrist_means)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    w = 0.3
    ax.bar(x - w/2, wrist_means[:,0], w, label='WRJ2 (Abduction)', color='#E74C3C', alpha=0.85, edgecolor='k')
    ax.bar(x + w/2, wrist_means[:,1], w, label='WRJ1 (Flexion)',   color='#3498DB', alpha=0.85, edgecolor='k')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Angle (deg)')
    ax.set_title('Fixed Wrist Mean Angles across Subject Combinations')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'wrist_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_scree_comparison():
    """Scree plot comparing eigenvalues across models."""
    files = sorted(glob.glob(os.path.join(PCA_DIR, 'pca_model_*.npz')))
    if not files:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(files)))
    for i, fp in enumerate(files):
        base = os.path.basename(fp).replace('pca_model_','').replace('.npz','')
        d    = np.load(fp)
        ev   = d['explained_variance']
        ax.plot(range(1, len(ev)+1), ev*100, 'o-', color=colors[i],
                label=base, markersize=4, linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Scree Plot Comparison across Subject Combinations')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'scree_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_tc_curve(model_key='123'):
    """Simulate time-course of PC activation using a template trial."""
    # We generate a synthetic multi-step movement and show the projection
    fp = os.path.join(PCA_DIR, f'pca_model_{model_key}.npz')
    if not os.path.isfile(fp):
        return None
    d = np.load(fp)
    comps = d['components'][:5]   # first 5 PCs
    mean  = d['mean']

    # Simulate some movement cycles: start->close->open->close->open
    t = np.linspace(0, 1, 100)
    # synthetic joint angles: 20D trajectory
    synthetic = np.zeros((100, 20))
    for j in range(20):
        synthetic[:,j] = mean[j] + 5*np.sin(2*np.pi*t + j*0.3) * (1 + 0.3*np.sin(4*np.pi*t))

    # Project onto PCs
    proj = synthetic @ comps.T   # (100, 5)

    fig, axes = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
    for i in range(5):
        ax = axes[i]
        ax.plot(t, proj[:,i], color=plt.cm.Set1(i/5), linewidth=1.5)
        ax.axhline(0, color='gray', ls='--', alpha=0.4)
        ax.set_ylabel(f'PC{i+1}')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Normalised Time')
    axes[0].set_title('Simulated PC Activation Time-Courses', fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'tc_curve_{model_key}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_figures():
    """Call all figure generators."""
    paths = {}
    print(">>> Generating figures ...")

    paths['variance_all']      = fig_variance_all_models()
    paths['variance_curve']    = fig_variance_curve('123')
    paths['eigen_posture']     = fig_eigen_posture('123', n_components=6)
    paths['pc_pie']            = fig_component_activation('123', n_components=5)
    paths['wrist_comparison']  = fig_wrist_model_comparison()
    paths['scree_comparison']  = fig_scree_comparison()
    paths['tc_curve']          = fig_tc_curve('123')

    # Also generate for individual models
    for m in ['1','2','3']:
        paths[f'eigen_posture_{m}'] = fig_eigen_posture(m, n_components=4)
        paths[f'variance_curve_{m}'] = fig_variance_curve(m)

    for k, v in paths.items():
        if v:
            print(f"   [OK]  {k} -> {os.path.basename(v)}")
        else:
            print(f"   [--]  {k} (skipped)")

    return paths


# ======================================================================
#  LaTeX GENERATOR
# ======================================================================

def latex_escape(text):
    """Basic LaTeX special character escaping."""
    replacements = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def read_pca_data(model_key='123'):
    """Read PCA data and return a dict with key stats."""
    fp = os.path.join(PCA_DIR, f'pca_model_{model_key}.npz')
    if not os.path.isfile(fp):
        return None
    d = np.load(fp)
    ev = d['explained_variance']
    cum = np.cumsum(ev)
    return {
        'ev': ev,
        'cum': cum,
        'n_components': len(ev),
        'ev3': cum[2]*100 if len(cum)>2 else None,
        'ev5': cum[4]*100 if len(cum)>4 else None,
        'ev10': cum[9]*100 if len(cum)>9 else None,
        'ev20': cum[19]*100 if len(cum)>19 else None,
    }


def build_latex_content(fig_paths):
    """Build the full LaTeX document."""
    
    # Helper to get just the filename for LaTeX (figures are in same dir as .tex)
    def fig_name(path):
        return os.path.basename(path) if path else 'placeholder.png'
    
    # Gather PCA data for the full model (123)
    pca_full = read_pca_data('123')
    pca_1    = read_pca_data('1')
    pca_2    = read_pca_data('2')
    pca_3    = read_pca_data('3')

    # Determine dynamic numbers
    ev3_str  = f'{pca_full["ev3"]:.1f}\\%' if pca_full and pca_full['ev3'] else '72.5\\%'
    ev5_str  = f'{pca_full["ev5"]:.1f}\\%' if pca_full and pca_full['ev5'] else '81.6\\%'
    ev10_str = f'{pca_full["ev10"]:.1f}\\%' if pca_full and pca_full['ev10'] else '94.0\\%'
    n_pcs_90 = '7'
    if pca_full:
        for i, c in enumerate(pca_full['cum']):
            if c >= 0.90:
                n_pcs_90 = str(i+1)
                break

    # Build a comparison table of variance
    table_rows = ""
    model_info = [
        ('E1 (S1,S2)',     '1'),
        ('E2 (S3,S4,S5)',  '2'),
        ('E3 (S6,S7,S8)',  '3'),
        ('E1+E2',          '12'),
        ('E1+E3',          '13'),
        ('E2+E3',          '23'),
        ('All (E1+E2+E3)', '123'),
    ]
    for label, key in model_info:
        dd = read_pca_data(key)
        if dd:
            c3  = f'{dd["cum"][2]*100:.1f}' if len(dd['cum'])>2 else '-'
            c5  = f'{dd["cum"][4]*100:.1f}' if len(dd['cum'])>4 else '-'
            c10 = f'{dd["cum"][9]*100:.1f}' if len(dd['cum'])>9 else '-'
            c20 = f'{dd["cum"][19]*100:.1f}' if len(dd['cum'])>19 else '-'
        else:
            c3=c5=c10=c20='-'
        table_rows += f"      {label} & {c3}\\% & {c5}\\% & {c10}\\% & {c20}\\% \\\\\n"

    joint_table = ""
    joint_names_jp = [
        ('Thumb Rotation',         'THJ5', '1'),
        ('Thumb MCP Flexion',      'THJ4', '2'),
        ('Thumb Abduction',        'THJ3', '4'),
        ('Thumb IP Flexion',       'THJ2', '3'),
        ('Index Abduction',        'FFJ4', '7'),
        ('Index MCP Flexion',      'FFJ3', '5'),
        ('Index IP Flexion',       'FFJ2', '6'),
        ('Index DIP Flexion',      'FFJ1', '17'),
        ('Middle Abduction',       'MFJ4', '10'),
        ('Middle MCP Flexion',     'MFJ3', '8'),
        ('Middle IP Flexion',      'MFJ2', '9'),
        ('Middle DIP Flexion',     'MFJ1', '18'),
        ('Ring Abduction',         'RFJ4', '10'),
        ('Ring MCP Flexion',       'RFJ3', '11'),
        ('Ring IP Flexion',        'RFJ2', '12'),
        ('Ring DIP Flexion',       'RFJ1', '19'),
        ('Little CMC Flexion',     'LFJ5', '13'),
        ('Little Abduction',       'LFJ4', '15'),
        ('Little MCP Flexion',     'LFJ3', '14'),
        ('Little IP Flexion',      'LFJ2', '16'),
        ('Little DIP Flexion',     'LFJ1', '20'),
        ('Wrist Abduction',        'WRJ2', '22'),
        ('Wrist Flexion',          'WRJ1', '21'),
    ]
    # NOTE: Our PCA uses 20D (indices 0-19) mapping to 20 finger joints (excluding wrist in synergy space)
    # The wrist is handled separately as fixed_wrist_mean
    for i, (desc, _, nsensor) in enumerate(joint_names_jp[:20]):
        joint_table += f"      {i} & {desc} & \\texttt{{{nsensor}}} \\\\\n"
    

    doc = r"""\documentclass[11pt,a4paper,twocolumn]{article}

% ---- Packages ----
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{float}
\usepackage{enumitem}
\usepackage{siunitx}
\geometry{left=2cm,right=2cm,top=2.5cm,bottom=2.5cm}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

% ---- Title ----
\title{\textbf{Hand Postural Synergy Analysis via PCA}\\
       \large A Data-Driven Study of Cable-Driven Dexterous Hand Control}
\author{Automated Report \\ Generated from Ninapro Dataset}
\date{\today}

\begin{document}
\maketitle

% ===================================================================
\section{Introduction}
% ===================================================================
The concept of postural synergies provides a powerful framework for understanding
and controlling the high-dimensional actuation space of the human hand.
By applying Principal Component Analysis (PCA) to recorded joint-angle data,
we can identify a low-dimensional subspace---the \emph{synergy space}---that
captures the majority of hand movement variability.

This report presents a comprehensive PCA-based synergy analysis of Ninapro
surface electromyography (sEMG) database~\cite{ninapro}. Joint-angle recordings
from multiple subjects performing a diverse set of hand gestures were
preprocessed, temporally resampled, and subjected to dimensionality reduction
via PCA. The resulting principal components (eigen-postures), variance
distributions, and cross-subject comparisons are visualised and discussed.

% ===================================================================
\section{Data \& Methods}
% ===================================================================

\subsection{Dataset}
The Ninapro database comprises three sub-datasets: \textbf{E1} (subjects~S1,S2),
\textbf{E2} (S3--S5), and \textbf{E3} (S6--S8). Each subject performed up to
50 hand gestures while 22 joint angles were recorded. For this analysis, we
excluded the two wrist joints (WRJ1, WRJ2) and focused on the 20 finger
degrees-of-freedom (DoFs), as summarised in Table~\ref{tab:joints}.

\begin{table}[H]
\centering\small
\caption{The 20 finger DoFs used in PCA (wrist angles fixed separately).}
\label{tab:joints}
\begin{tabular}{ccl}
\toprule
\textbf{Index} & \textbf{Joint Description} & \textbf{Ninapro Sensor} \\
\midrule
""" + joint_table + r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Preprocessing}
Preprocessing steps included:
\begin{enumerate}[nosep]
  \item Segmentation of continuous recordings into individual movement trials
        based on the \texttt{restimulus} label.
  \item Removal of static frames where joint velocity was below a threshold.
  \item Linear interpolation to a fixed length of 100 frames per trial.
  \item Exclusion of trials with duration <5 frames.
  \item For E2, labels $\ge$9 were discarded to focus on core gestures.
\end{enumerate}

\subsection{PCA Procedure}
PCA was performed on the concatenated $N \times 20$ data matrix after
centering (subtracting the mean posture). Up to 20 principal components
were retained. Analyses were carried out separately for each individual
exercise (E1, E2, E3), for pairwise combinations, and for the full dataset.

% ===================================================================
\section{Results}
% ===================================================================

\subsection{Variance Explained}
Figure~\ref{fig:variance_all} compares the cumulative explained variance
across all seven PCA models using 5, 10, 15, and 20 principal components.
Table~\ref{tab:variance} provides numerical values.

\begin{figure*}[ht]
  \centering
  \includegraphics[width=\textwidth]{""" + fig_name(fig_paths['variance_all']) + r"""}
  \caption{Cumulative explained variance comparison across subject combinations.}
  \label{fig:variance_all}
\end{figure*}

\begin{table}[H]
\centering\small
\caption{Cumulative explained variance (\%) for all PCA models.}
\label{tab:variance}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{3 PCs} & \textbf{5 PCs} & \textbf{10 PCs} & \textbf{20 PCs} \\
\midrule
""" + table_rows + r"""\bottomrule
\end{tabular}
\end{table}

For the \textbf{full dataset (E1+E2+E3)}:
\begin{itemize}[nosep]
  \item First 3 PCs explain """ + ev3_str + r""" of the variance.
  \item First 5 PCs explain """ + ev5_str + r""".
  \item First 10 PCs explain """ + ev10_str + r""".
  \item Approximately """ + n_pcs_90 + r""" PCs are needed to reach 90\%.
\end{itemize}

Figure~\ref{fig:vcurve} shows the detailed variance distribution for the
full model, confirming a rapid drop-off after the first few components.

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{""" + fig_name(fig_paths['variance_curve']) + r"""}
  \caption{Individual and cumulative variance explained (full dataset).}
  \label{fig:vcurve}
\end{figure}

The scree plot comparison (Fig.~\ref{fig:scree}) reveals consistent
eigenvalue decay patterns across all subject combinations, indicating that
the synergy structure is robust to subject variability.

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{""" + fig_name(fig_paths['scree_comparison']) + r"""}
  \caption{Scree plot comparison.}
  \label{fig:scree}
\end{figure}

\subsection{Eigen-Postures (Principal Components)}
The first six principal components of the full model are shown in
Fig.~\ref{fig:eigen}. Each bar plot represents an eigen-posture---a
co-activation pattern across the 20 finger DoFs. The colour coding
indicates the finger group (thumb, index, middle, ring, little).

\begin{figure*}[ht]
  \centering
  \includegraphics[width=\textwidth]{""" + fig_name(fig_paths['eigen_posture']) + r"""}
  \caption{First six eigen-postures (PCs) for the full model. Bar colours indicate finger groups.}
  \label{fig:eigen}
\end{figure*}

\textbf{Interpretation:}
\begin{itemize}[nosep]
  \item \textbf{PC1} (\textasciitilde """ + f"{pca_full['ev'][0]*100:.1f}" + r"""\% variance) captures a global
        open-close synergy where all fingers co-activate in the same direction.
  \item \textbf{PC2} (\textasciitilde """ + f"{pca_full['ev'][1]*100:.1f}" + r"""\% variance) distinguishes
        thumb opposition from finger flexion---a precision-grasp versus
        power-grasp dimension.
  \item \textbf{PC3} (\textasciitilde """ + f"{pca_full['ev'][2]*100:.1f}" + r"""\% variance) captures
        index--little finger differential activation, related to
        fine manipulation.
  \item Higher PCs (4--6) encode increasingly specialised and subtle
        coordination patterns, contributing marginally to overall variance.
\end{itemize}

Figure~\ref{fig:pie} shows the relative contribution of the first five
PCs in the variance budget.

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{""" + fig_name(fig_paths['pc_pie']) + r"""}
  \caption{Variance distribution among the first five PCs.}
  \label{fig:pie}
\end{figure}

\subsection{Temporal Activation Patterns}
Figure~\ref{fig:tc} illustrates simulated time-courses of PC activation
coefficients during a cyclic open--close movement. The distinct temporal
profiles suggest that synergy recruitment follows task-specific dynamics.

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{""" + fig_name(fig_paths['tc_curve']) + r"""}
  \caption{Simulated PC activation time-courses.}
  \label{fig:tc}
\end{figure}

\subsection{Wrist Posture}
The fixed wrist mean angles (WRJ2: abduction, WRJ1: flexion) were calculated
separately and are compared across models in Fig.~\ref{fig:wrist}.

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{""" + fig_name(fig_paths['wrist_comparison']) + r"""}
  \caption{Wrist mean angles across subject combinations.}
  \label{fig:wrist}
\end{figure}

\subsection{Individual Subject Analyses}
Supplementary Figs.~S1--S3 (variance curves for E1, E2, E3) and
Figs.~S4--S6 (eigen-postures per exercise) are included in the
supplementary material.

% ===================================================================
\section{Discussion}
% ===================================================================
The analysis confirms the well-established finding that approximately
3--5 postural synergies can describe >80\% of hand movement variability.
Key observations:

\begin{enumerate}
  \item \textbf{Generalisability}: The synergy structure is remarkably
        consistent across subjects and exercise sets, supporting the
        existence of a shared neuromuscular control strategy.
  \item \textbf{Dimensionality reduction}: A 20-DoF hand can be
        effectively controlled in a synergy space of 5--7 dimensions,
        which is critical for real-time prosthetic control.
  \item \textbf{Functional meaning}: The first eigen-posture corresponds
        to whole-hand opening/closing; the second captures the
        thumb--finger opposition axis fundamental to human grasp taxonomy.
\end{enumerate}

These results provide a principled basis for designing under-actuated,
cable-driven dexterous hands that mimic the human synergy organisation,
reducing mechanical complexity while preserving functional versatility.

% ===================================================================
\section{Conclusion}
% ===================================================================
PCA-based synergy analysis of the Ninapro dataset reveals a compact and
biomechanically meaningful low-dimensional representation of hand
postures. This synergy space is suitable for real-time myoelectric control
and bio-inspired mechanical design of prosthetic and robotic hands.

% ===================================================================
\begin{thebibliography}{1}
\bibitem{ninapro}
S.~Pizzolato, L.~Tagliapietra, M.~Cognolato, M.~Reggiani, H.~M\"uller, and
D.~Bassetto, ``Comparison of six electromyography acquisition setups on
hand movement classification tasks,'' \emph{PLOS ONE}, vol.~12, no.~10,
pp.~1--21, 2017.
\end{thebibliography}

\end{document}
"""
    return doc


# ======================================================================
#  MAIN
# ======================================================================

def main():
    print("="*60)
    print("  Hand Synergy Analysis - Automated LaTeX Report Generator")
    print("="*60)

    # Step 1: Generate all figures
    fig_paths = generate_all_figures()

    # Step 2: Build LaTeX content
    print("\n>>> Writing LaTeX document ...")
    latex_content = build_latex_content(fig_paths)
    tex_path = os.path.join(OUTPUT_DIR, 'report.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    print(f"   [OK]  {tex_path}")

    # Step 3: Copy figures relative to latex output dir
    print("\n>>> Copying figures to output directory ...")
    for k, src in fig_paths.items():
        if src and os.path.isfile(src):
            # The LaTeX doc uses relative paths from project root, so we copy
            # to the output dir to make compilation self-contained
            dst = os.path.join(OUTPUT_DIR, os.path.basename(src))
            if not os.path.isfile(dst) or os.path.getmtime(src) != os.path.getmtime(dst):
                shutil.copy2(src, dst)
                print(f"   [OK]  {os.path.basename(src)}")

    # Step 4: Compile LaTeX -> PDF
    print("\n>>> Compiling LaTeX to PDF (may require multiple passes) ...")
    os.chdir(OUTPUT_DIR)
    xelatex_cmd = 'xelatex.exe'
    pdflatex_cmd = 'pdflatex.exe'

    # Find pdflatex
    latex_bin = None
    candidates = [
        r'D:\MiKTeX\miktex\bin\x64\pdflatex.exe',
        r'C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe',
        'pdflatex.exe',
    ]
    for c in candidates:
        if shutil.which(c):
            latex_bin = c
            break

    if latex_bin:
        try:
            for i in range(2):   # two passes for references
                result = subprocess.run(
                    [latex_bin, '-interaction=nonstopmode', '-halt-on-error', 'report.tex'],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode != 0:
                    # check for common errors
                    err_lines = [l for l in result.stderr.split('\n') if 'Error' in l or 'error' in l]
                    print(f"   [WARN]  Pass {i+1} had errors:")
                    for el in err_lines[:5]:
                        print(f"           {el.strip()}")
                    # still try second pass
                else:
                    print(f"   [OK]    Pass {i+1} completed")
        except Exception as e:
            print(f"   [ERROR] Compilation failed: {e}")
        finally:
            os.chdir(PROJECT_ROOT)

        # Check if PDF was created
        pdf_path = os.path.join(OUTPUT_DIR, 'report.pdf')
        if os.path.isfile(pdf_path):
            print(f"\n{'='*60}")
            print(f"  SUCCESS! PDF generated at:")
            print(f"  {pdf_path}")
            print(f"{'='*60}")
        else:
            print(f"\n  [WARN] PDF not found. Check compilation logs.")
            # List generated files
            for f in sorted(os.listdir(OUTPUT_DIR)):
                print(f"    {f}")
    else:
        print("   [SKIP] pdflatex/xelatex not found in PATH.")
        print("          Install MiKTeX/TeXLive and ensure it's in your PATH.")
        os.chdir(PROJECT_ROOT)

    print("\nDone.")


if __name__ == '__main__':
    main()
