#!/usr/bin/env python
"""生成 MyoKi 抓取姿态协同课程论文 LaTeX 文稿。

论文理论背景以 Santello 等人的姿态协同为主，复杂度分析参考 West 等人
提出的 Variance-Accounted-For Complexity Index (VAF-CI)。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSE_DIR = PROJECT_ROOT / "course_project"
PAPER_DIR = COURSE_DIR / "paper"
PAPER_FIG_DIR = PAPER_DIR / "figures"
DATA_DIR = COURSE_DIR / "outputs" / "data"
PAYLOAD_PATH = DATA_DIR / "report_payload.json"

FAMILY_CN = {"Power": "力量型", "Precision": "精细型", "Complex": "复杂型"}


def fmt_p(p: float) -> str:
    if p == 0 or p < 1e-300:
        return "$<10^{-300}$"
    if p < 1e-4:
        mantissa, exponent = f"{p:.1e}".split("e")
        return rf"${mantissa}\times 10^{{{int(exponent)}}}$"
    return f"${p:.4f}$"


def fmt_p_text(p: float) -> str:
    if p == 0 or p < 1e-300:
        return f"p{fmt_p(p)}"
    return f"p={fmt_p(p)}"


def ensure_dirs() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_workflow() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.57, 0.17, 0.27, "MyoKi 数据\n23名受试者\n41个抓取任务"),
        (0.24, 0.57, 0.17, 0.27, "试次级特征\n15个关节\n活动范围"),
        (0.46, 0.57, 0.17, 0.27, "姿态协同表征\n任务均值\n主成分分析"),
        (0.68, 0.57, 0.18, 0.27, "统计推断\nbootstrap\nblocked ANOVA"),
        (0.46, 0.13, 0.17, 0.25, "复杂度分析\nVAF-CI\n方差解释复杂度"),
        (0.68, 0.13, 0.18, 0.25, "泛化验证\n留一受试者\n多项逻辑回归"),
    ]
    colors = ["#E8F1FA", "#EAF5EF", "#FFF4E5", "#FDECEC", "#F3ECFA", "#EEF0F2"]
    for i, (x, y, w, h, label) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            facecolor=colors[i],
            edgecolor="#333333",
            linewidth=0.9,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", linespacing=1.2)

    arrows = [
        ((0.19, 0.705), (0.24, 0.705)),
        ((0.41, 0.705), (0.46, 0.705)),
        ((0.63, 0.705), (0.68, 0.705)),
        ((0.545, 0.57), (0.545, 0.38)),
        ((0.63, 0.255), (0.68, 0.255)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#444444"))

    fig.savefig(PAPER_FIG_DIR / "graphical_workflow.png", dpi=450, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "graphical_workflow.pdf", bbox_inches="tight")
    plt.close(fig)


def write_tex(payload: dict) -> None:
    stats = payload["stats"]
    main = stats["main_pca"]
    boot = stats["bootstrap"]
    anova = stats["anova"]["overall"]
    prediction = stats["prediction"]
    sensitivity = stats["sensitivity"]
    complexity = stats["complexity_by_family"]
    pairwise = stats["anova"]["pairwise"]
    best = prediction["best_variant"]

    def fam(label: str) -> str:
        return FAMILY_CN.get(label, label)

    def feature_label(name: str) -> str:
        return {"PC2": "前2主成分", "PC3": "前3主成分", "PC5": "前5主成分", "Raw15": "原始15维"}.get(name, name)

    def pred_row(name: str) -> str:
        x = prediction["summary"][name]
        return (
            f"{feature_label(name)} & {x['balanced_accuracy_mean']:.3f} $\\pm$ {x['balanced_accuracy_sd']:.3f} "
            f"& {x['macro_f1_mean']:.3f} $\\pm$ {x['macro_f1_sd']:.3f} "
            f"& {x['accuracy_mean']:.3f} $\\pm$ {x['accuracy_sd']:.3f} \\\\"
        )

    complexity_rows = "\n".join(
        f"{fam(c['family'])} & {c['n_tasks']} & {c['pc1_pc2']*100:.1f} & {c['vaf_ci']:.3f} & {c['n90']} \\\\"
        for c in complexity
    )
    sens_rows = "\n".join(
        f"{s['label'].replace('All 15 joints', '全部15关节').replace('No abduction (12D)', '去除外展关节').replace('Flexion only (10D)', '仅屈曲关节')} "
        f"& {s['n_joints']} & {s['pc1_pc2']*100:.1f} & {s['vaf_ci']:.3f} & {s['n90']} \\\\"
        for s in sensitivity
    )
    pair_rows = "\n".join(
        f"{p['pc']} & {fam(p['family_a'])} vs {fam(p['family_b'])} & {p['difference_b_minus_a']:.2f} "
        f"& {p['F']:.1f} & {fmt_p(float(p['p']))} & {fmt_p(float(p['q_bh']))} \\\\"
        for p in pairwise
    )

    tex = rf"""
\documentclass[10pt,a4paper]{{ctexart}}
\usepackage[a4paper,margin=2.68cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath,amssymb}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{authblk}}
\usepackage{{titlesec}}
\usepackage{{setspace}}
\usepackage{{siunitx}}

\hypersetup{{colorlinks=true,linkcolor=blue!45!black,citecolor=blue!45!black,urlcolor=blue!45!black}}
\renewcommand\Authsep{{，}}
\renewcommand\Authand{{ 和 }}
\renewcommand\Authands{{ 和 }}
\setstretch{{1.16}}
\emergencystretch=2em
\setlength{{\parskip}}{{3pt}}
\setlength{{\parindent}}{{1.7em}}
\titleformat{{\section}}{{\large\bfseries}}{{\thesection}}{{0.6em}}{{}}
\titleformat{{\subsection}}{{\normalsize\bfseries}}{{\thesubsection}}{{0.6em}}{{}}
\captionsetup{{font=small,labelfont=bf}}
\pagestyle{{plain}}

\title{{\bfseries 基于 MyoKi 人手运动学数据的抓取姿态协同统计分析}}
\author{{王裕硕}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
\noindent
Santello、Flanders 和 Soechting 的经典研究表明，人手在准备抓取和使用物体时，多个关节角呈现少数“姿态协同”；前两个主成分即可解释大部分姿态方差，同时更高阶成分仍保留物体区分信息。本研究基于公开下载的 MyoKi 人手运动学数据，分析真实日常抓取任务中是否存在稳定、可解释且可预测的低维姿态协同结构。研究纳入 {payload['n_participants']} 名右利手健康受试者、{payload['n_tasks']} 个抓取任务和 {payload['n_trials']} 个试次；每个试次被重采样为 100 帧，并以 15 个 CyberGlove 手部关节角的活动范围表征。主成分分析在任务均值矩阵上拟合，随后采用受试者 bootstrap、方差解释复杂度指数（Variance-Accounted-For Complexity Index, VAF-CI）、受试者 blocked ANOVA、Benjamini-Hochberg 假发现率校正、留一受试者多项逻辑回归和关节集敏感性分析。前两主成分解释 {main['pc1_pc2']*100:.1f}\% 任务级方差，bootstrap 95\% 置信区间为 [{boot['pc1_pc2']['ci95'][0]*100:.1f}\%, {boot['pc1_pc2']['ci95'][1]*100:.1f}\%]；整体 VAF-CI 为 {main['vaf_ci']:.3f}，达到 90\% 方差解释率需要 {main['n90']} 个主成分。抓取家族对第一和第二主成分均有显著影响，第二主成分的偏 $\eta^2$ 为 {anova['PC2']['partial_eta2']:.3f}。留一受试者预测中，{feature_label(best)}的平衡准确率为 {prediction['summary'][best]['balanced_accuracy_mean']:.2f}$\pm${prediction['summary'][best]['balanced_accuracy_sd']:.2f}。这些结果表明，MyoKi 抓取运动学中存在稳健的低维姿态协同结构，高阶和局部关节信息对任务区分具有贡献。
\end{{abstract}}

\noindent\textbf{{关键词：}}MyoKi；手部运动学；姿态协同；主成分分析；bootstrap；blocked ANOVA；方差解释复杂度指数；多项逻辑回归

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\linewidth]{{figures/graphical_workflow.png}}
\caption{{研究流程。预处理后的 MyoKi 试次被转化为关节活动范围特征；任务均值主成分分析提供低维姿态协同表征；随后通过 bootstrap、VAF-CI、blocked ANOVA 和留一受试者预测验证其统计意义。}}
\label{{fig:workflow}}
\end{{figure}}

\section{{引言}}
人手能够在强力抓握、精细捏取和复杂工具操作之间灵活切换。若把每个手指关节都看作独立控制变量，运动控制和统计建模都会受到高维空间的限制。姿态协同理论提供了一个更合适的出发点：多个关节可能通过少数低维协调模式共同变化，使手的总体形状可以被少数变量概括。

Santello 等人在“Postural hand synergies for tool use”中要求受试者想象抓取并使用 57 个熟悉物体，记录 15 个手指与拇指关节角，并用主成分分析研究静态手姿态的协方差结构。他们发现前两个主成分解释了超过 80\% 的姿态方差，说明姿态空间维度远低于记录的自由度数；更高阶主成分提供额外的物体区分信息。更重要的是，姿态协同与传统抓握分类的对应关系有限，这提示“手的形状控制”与“接触力和抓握分类”需要分层讨论\cite{{santello1998}}。

MyoKi 数据集记录真实日常生活活动中的肌电、惯性测量和手部运动学信号，提供了把 Santello 式姿态协同思想扩展到更自然任务环境的机会\cite{{andreas2025}}。与此同时，West 等人提出方差解释复杂度指数（Variance-Accounted-For Complexity Index, VAF-CI），用累计方差解释曲线刻画一个操作任务需要多少独立自由度或协同成分来描述：当更多主成分才能解释同样比例的方差时，任务可被视为需要更强的关节个体化控制\cite{{west2025}}。

在此基础上，本研究将主成分分析作为后续统计推断的表征层，关注四个问题：第一，MyoKi 抓取任务中是否存在稳定的低维姿态协同；第二，按照 West 的方差解释思路，不同抓取家族是否表现出不同复杂度；第三，力量型、精细型和复杂型抓取是否在协同空间中存在显著差异；第四，低维协同得分是否保留跨受试者预测抓取家族的信息。

\section{{数据来源与研究问题}}
\subsection{{MyoKi 运动学数据}}
本研究使用公开发布的 MyoKi 数据集\cite{{andreas2025}}。数据从发布网站下载后，首先读取每名受试者的 CyberGlove II 手部关节角时间序列、试次边界、任务标签和受试者信息。为使不同时长的动作可以进入同一统计框架，每个试次按其起止边界截取后，通过线性插值重采样为 100 帧。原始手套信号中的掌弓通道和两个腕部通道被剔除，保留 15 个手指相关关节通道；随后根据受试者信息筛选右利手健康受试者。任务元数据用于识别日常活动的动作类型。为聚焦抓取姿态协同而非腕部校准、键盘鼠标输入或双手连续操作，本研究纳入 41 个抓取及工具操作任务，并将其归并为力量型、精细型和复杂型三个抓取家族。最终分析数据包括 {payload['n_participants']} 名受试者、{payload['n_tasks']} 个任务和 {payload['n_trials']} 个试次。

\subsection{{统计分析计划}}
本研究的统计分析围绕姿态协同的存在性、稳定性、任务复杂度、抓取家族差异和跨受试者泛化能力展开（表 \ref{{tab:analysisplan}}）。主成分分析用于估计低维姿态协同轴；受试者 bootstrap 用于评估协同结构对样本组成的敏感性；VAF-CI 用于描述累计方差解释曲线所反映的复杂度；受试者 blocked ANOVA 用于检验抓取家族在协同空间中的差异；假发现率校正用于控制两两比较中的多重检验风险；留一受试者多项逻辑回归用于评估协同表征是否具有跨个体预测信息。

\begin{{table}}[H]
\centering
\caption{{研究问题、统计方法与报告指标。}}
\label{{tab:analysisplan}}
\small
\begin{{tabular}}{{p{{0.22\linewidth}}p{{0.26\linewidth}}p{{0.38\linewidth}}}}
\toprule
研究问题 & 统计方法 & 主要报告指标 \\
\midrule
低维协同是否存在 & 任务均值主成分分析 & 单个与累计方差解释率、主成分载荷 \\
结果是否稳定 & 受试者 bootstrap & 95\% 置信区间、bootstrap 分布 \\
任务复杂度如何变化 & VAF-CI & VAF-CI、达到 90\% 方差所需主成分数 \\
抓取家族是否不同 & 受试者 blocked ANOVA & F 统计量、p 值、偏 $\eta^2$、假发现率 q 值 \\
表征是否可泛化 & 留一受试者多项逻辑回归 & 平衡准确率、宏平均 F1、混淆矩阵 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{研究方法}}
\subsection{{试次级特征提取}}
每个试次原始输入为 $100 \times 15$ 的关节角矩阵。考虑到不同抓取任务的关键差异既包含最终姿态，也包含动作幅度，本研究将每个关节的角度活动范围作为主要特征：
\begin{{equation}}
r_j^{{(t)}}=\max_f \theta_j^{{(t)}}(f)-\min_f \theta_j^{{(t)}}(f),
\end{{equation}}
其中 $t$ 表示试次，$j$ 表示关节，$f$ 表示重采样后的帧。范围特征具有明确物理含义，能描述某关节在任务中的募集幅度，同时降低完整时间序列建模的参数负担。其局限是丢失相位、速度和动作阶段信息，因此本研究将其作为稳健、可解释的运动学表征。

\subsection{{主成分表征与 VAF-CI}}
主成分分析在 $41 \times 15$ 的任务均值矩阵上拟合：对每个任务，将所有受试者和重复试次的关节活动范围取均值。主成分分析是一种以正交方向最大化方差的线性表征方法；本研究将主成分得分用于后续 ANOVA 和预测验证，并把神经控制模块解释保留在理论讨论层面\cite{{jolliffe2016}}。随后把所有试次投影到该主成分空间。参照 West 等人提出的 Variance-Accounted-For Complexity Index，本研究用累计方差解释曲线上方的归一化面积计算 VAF-CI：
\begin{{equation}}
\mathrm{{CI}} = \frac{{1}}{{m}}\sum_{{k=1}}^m \left(1-\sum_{{i=1}}^k \mathrm{{VAF}}_i\right),
\end{{equation}}
其中 $m$ 为主成分数，$\mathrm{{VAF}}_i$ 为第 $i$ 个主成分的方差解释率。VAF-CI 越高，说明需要更多协同成分才能描述数据方差；该指标用于描述本数据累计方差解释曲线的集中程度，并与 West 等人关于协同成分数量的思路对应。

\subsection{{统计推断与预测验证}}
不确定性通过受试者 bootstrap 估计。每次有放回抽取 23 名受试者，重新生成任务均值矩阵、拟合主成分分析，并计算前两主成分方差解释率、VAF-CI 和达到 90\% 方差解释率所需主成分数，重复 1000 次后报告中位数和 95\% 置信区间。

抓取家族差异使用受试者 blocked ANOVA。对第一和第二主成分得分分别建立固定效应模型：
\begin{{equation}}
\mathrm{{PC}}_{{it}} = \beta_0+\beta_1\mathrm{{Family}}_{{it}}+\gamma_i\mathrm{{Subject}}_i+\epsilon_{{it}}.
\end{{equation}}
受试者固定效应用于控制重复测量和个体基线差异。两两比较同样使用受试者阻塞模型，并采用 Benjamini-Hochberg 方法控制假发现率。除显著性检验外，本文报告偏 $\eta^2$ 作为效应量，以避免只用 p 值描述抓取家族差异\cite{{cohen1988}}。

预测验证使用多项逻辑回归，并采用留一受试者交叉验证。多项逻辑回归是多分类问题中可解释性较强的基线模型，适合在样本量有限时评估不同特征集的可预测信息\cite{{hastie2009}}。每次将一名受试者完全留出作为测试集，其余受试者训练模型。比较的特征集包括原始 15 维范围特征、前 2、前 3 和前 5 个主成分。主要指标为平衡准确率，辅助报告宏平均 F1 和总体准确率。

\section{{研究结果}}
\subsection{{低维姿态协同结构}}
任务均值主成分分析显示，第一主成分解释 {main['pc1']*100:.1f}\% 方差，第二主成分解释 {main['pc2']*100:.1f}\% 方差，前两主成分合计解释 {main['pc1_pc2']*100:.1f}\%。受试者 bootstrap 得到前两主成分方差解释率 95\% 置信区间为 [{boot['pc1_pc2']['ci95'][0]*100:.1f}\%, {boot['pc1_pc2']['ci95'][1]*100:.1f}\%]，提示低维结构在受试者重采样下保持稳定。第一主成分主要反映整体手部活动幅度，第二主成分则更明显地区分不同抓取家族。

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\linewidth]{{../outputs/figures/figure1_pca_overview.png}}
\caption{{主成分表征。A，三个抓取家族的试次数量。B，任务均值主成分分析的单个与累计方差解释率，浅色横带表示前两主成分方差解释率的 bootstrap 置信区间。C，前两种特征姿态的关节载荷。D，试次投影及抓取家族置信椭圆。}}
\label{{fig:pca}}
\end{{figure}}

\subsection{{VAF-CI 与抓取家族效应}}
整体数据的 VAF-CI 为 {main['vaf_ci']:.3f}，达到 90\% 方差解释率需要 {main['n90']} 个主成分。按抓取家族分层后，力量型、精细型和复杂型抓取的 VAF-CI 分别为 {complexity[0]['vaf_ci']:.3f}、{complexity[1]['vaf_ci']:.3f} 和 {complexity[2]['vaf_ci']:.3f}（表 \ref{{tab:complexity}}）。该结果说明 MyoKi 中不同抓取家族的方差集中度存在差异；这里的“复杂型”标签来自任务语义，解释时需与 West 定义的高个体化运动复杂度区分。

\begin{{table}}[H]
\centering
\caption{{按抓取家族计算的 VAF-CI。}}
\label{{tab:complexity}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
抓取家族 & 任务数 & 前两主成分方差解释率(\%) & VAF-CI & 90\% 方差所需主成分数 \\
\midrule
{complexity_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

受试者 blocked ANOVA 显示，抓取家族对第一主成分有显著影响，$F({anova['PC1']['df_num']},{anova['PC1']['df_den']})={anova['PC1']['F']:.1f}$，{fmt_p_text(float(anova['PC1']['p']))}，偏 $\eta^2={anova['PC1']['partial_eta2']:.3f}$；对第二主成分的影响更强，$F({anova['PC2']['df_num']},{anova['PC2']['df_den']})={anova['PC2']['F']:.1f}$，{fmt_p_text(float(anova['PC2']['p']))}，偏 $\eta^2={anova['PC2']['partial_eta2']:.3f}$。两两比较结果见表 \ref{{tab:pairwise}}。

\begin{{table}}[H]
\centering
\caption{{受试者阻塞模型的两两比较。差值定义为第二组均值减第一组均值。}}
\label{{tab:pairwise}}
\small
\begin{{tabular}}{{llrrrr}}
\toprule
主成分 & 比较 & 差值 & F & p & 假发现率 q \\
\midrule
{pair_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\linewidth]{{../outputs/figures/figure2_inference.png}}
\caption{{统计推断结果。A，前两主成分方差解释率的受试者 bootstrap 分布。B，按抓取家族计算的 VAF-CI。C，抓取家族对第一和第二主成分的效应量。D，两两比较的假发现率校正结果。}}
\label{{fig:inference}}
\end{{figure}}

\subsection{{预测验证与敏感性分析}}
留一受试者交叉验证结果见表 \ref{{tab:prediction}}。原始 15 维范围特征获得最高平衡准确率，为 {prediction['summary']['Raw15']['balanced_accuracy_mean']:.3f}$\pm${prediction['summary']['Raw15']['balanced_accuracy_sd']:.3f}。前 5 个主成分的平衡准确率为 {prediction['summary']['PC5']['balanced_accuracy_mean']:.3f}$\pm${prediction['summary']['PC5']['balanced_accuracy_sd']:.3f}。这说明低维姿态协同保留了相当一部分任务家族信息，但原始高维特征仍包含额外可预测的局部关节细节。

\begin{{table}}[H]
\centering
\caption{{留一受试者交叉验证的预测表现。}}
\label{{tab:prediction}}
\small
\begin{{tabular}}{{lccc}}
\toprule
特征集 & 平衡准确率 & 宏平均 F1 & 总体准确率 \\
\midrule
{pred_row('PC2')}
{pred_row('PC3')}
{pred_row('PC5')}
{pred_row('Raw15')}
\bottomrule
\end{{tabular}}
\end{{table}}

关节集敏感性分析显示，去除外展关节或仅保留屈曲关节后，低维结构仍然存在，但前两主成分方差解释率和 VAF-CI 会发生变化（表 \ref{{tab:sensitivity}}）。这提示关节通道选择本身是模型假设，应在报告中透明呈现。

\begin{{table}}[H]
\centering
\caption{{关节集敏感性分析。}}
\label{{tab:sensitivity}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
关节集 & 关节数 & 前两主成分方差解释率(\%) & VAF-CI & 90\% 方差所需主成分数 \\
\midrule
{sens_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\linewidth]{{../outputs/figures/figure3_prediction_sensitivity.png}}
\caption{{预测验证与敏感性分析。A，不同特征集在留一受试者交叉验证中的平衡准确率。B，最佳模型的归一化混淆矩阵。C，不同关节集对方差解释率和 VAF-CI 的影响。D，主要预测结论和统计保护策略。}}
\label{{fig:prediction}}
\end{{figure}}

\section{{研究结论}}
本研究表明，MyoKi 抓取运动学中存在稳定的低维姿态协同结构。任务均值主成分分析显示，前两主成分解释超过 90\% 的任务级方差；受试者 bootstrap 进一步显示这一结果在重采样下保持稳定。该结果支持 Santello 等人提出的基本观点：人手姿态变化具有低维协同组织，15 个关节自由度可由少数主要协同轴概括。

第二，抓取家族在协同空间中的分布存在统计差异。受试者 blocked ANOVA 显示，抓取家族对第一和第二主成分得分均有显著影响，说明力量型、精细型和复杂型任务在关节活动范围的低维表征中表现出系统差异。经 Benjamini-Hochberg 假发现率校正后，多数组间比较仍保持显著，提示该差异具有较强的统计稳健性。

第三，协同表征保留了一定的跨受试者预测信息。留一受试者交叉验证显示，前 2、3、5 个主成分均高于三分类随机水平，但原始 15 维特征的平衡准确率最高。这一结果与 Santello 等人关于高阶主成分仍携带物体区分信息的观察一致：低维协同适合解释总体姿态组织，但精细分类仍可能依赖更多局部关节信息。

第四，方差解释复杂度指数（Variance-Accounted-For Complexity Index, VAF-CI）为任务方差集中度提供了补充描述。整体数据达到 90\% 方差解释率只需 {main['n90']} 个主成分，说明抓取任务具有明显的低维组织；不同抓取家族的 VAF-CI 不同，提示任务语义和关节个体化控制需求之间可能存在对应关系。本文的 VAF-CI 基于任务均值关节活动范围，用于参照 West 等人的方法思想描述本数据的累计方差结构。

\section{{讨论与局限性}}
本研究支持 Santello 式姿态协同理论在 MyoKi 抓取运动学中的统计可见性。相较于 Santello 的静态想象抓取任务，本文使用真实日常任务数据，并以关节活动范围作为特征，因此结论可理解为对其理论框架的运动学扩展。该扩展把姿态协同从受控物体想象任务推进到更自然的日常操作数据，并在受试者层面给出不确定性估计。

VAF-CI 为本文提供了超出“前两主成分解释率”的描述维度。该指标把整个累计方差解释曲线纳入比较，用于描述任务是否需要更多协同成分才能达到同样的方差解释水平。VAF-CI 把累计曲线压缩为单个数值，便于比较，同时会掩盖具体哪些关节贡献了额外自由度；方差解释率高仍需结合任务设计和生理机制解释，神经控制模块数量不能由该指标单独确定。

本文的统计检验目标限定在人类运动学数据内部。MyoKi 数据来自人类日常活动，Santello 讨论的是人手姿态协方差结构，West 关注的是操作任务所需协同成分数量；三者共同支持从人类运动学本身提出问题。本文结论集中在三个可检验层面：低维结构是否稳定、抓取家族是否存在统计差异、协同表征是否保留可泛化的预测信息。

本研究仍有若干局限。第一，所有受试者均为健康人群，结果不能直接外推至神经损伤患者、假肢用户或康复场景。第二，主要特征为角度活动范围，忽略完整时间序列相位、速度和动作阶段信息。第三，主成分分析是线性模型，无法捕获非线性或阶段依赖的协同结构。第四，受试者固定效应控制了个体差异，但使年龄、性别等受试者层面协变量无法在同一模型中估计。第五，预测验证基于现有任务标签，不能替代真实机器人控制或临床诊断性能评估。后续工作可把完整时间序列、肌电信号和分层混合模型结合起来，进一步区分运动学共变、任务语义差异和个体运动策略。

\section{{AI 使用说明}}
本文在作者完成数据获取、预处理、Python 统计分析、图表生成和结果判读的基础上，使用 AI 辅助组织分析结果、整合图表叙事、润色中文科技写作并生成 LaTeX 报告。AI 未参与原始数据采集和统计结论判定；所有数值结果均来自作者运行的 Python 分析输出。

\begin{{thebibliography}}{{99}}
\bibitem{{santello1998}} Santello M, Flanders M, Soechting JF. Postural hand synergies for tool use. \textit{{Journal of Neuroscience}}. 1998;18(23):10105--10115.
\bibitem{{andreas2025}} Andreas D, Werner D, Hou Z, Dwivedi A, Castellini C, Beckerle P. A database of multimodal myography and hand kinematics during realistic daily life activities. \textit{{Scientific Data}}. 2025;12:1507.
\bibitem{{feix2016}} Feix T, Romero J, Schmiedmayer H-B, Dollar AM, Kragic D. The GRASP taxonomy of human grasp types. \textit{{IEEE Transactions on Human-Machine Systems}}. 2016;46(1):66--77.
\bibitem{{west2025}} West AM Jr, Tessari F, Wang M, Hogan N. The Study of Dexterous Hand Manipulation: A Synergy-Based Complexity Index. \textit{{IEEE Transactions on Medical Robotics and Bionics}}. 2025;7(1):156--163.
\bibitem{{jolliffe2016}} Jolliffe IT, Cadima J. Principal component analysis: a review and recent developments. \textit{{Philosophical Transactions of the Royal Society A}}. 2016;374:20150202.
\bibitem{{cohen1988}} Cohen J. \textit{{Statistical Power Analysis for the Behavioral Sciences}}. 2nd ed. Lawrence Erlbaum Associates; 1988.
\bibitem{{hastie2009}} Hastie T, Tibshirani R, Friedman J. \textit{{The Elements of Statistical Learning}}. 2nd ed. Springer; 2009.
\bibitem{{efron1993}} Efron B, Tibshirani RJ. \textit{{An Introduction to the Bootstrap}}. Chapman and Hall; 1993.
\bibitem{{benjamini1995}} Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. \textit{{Journal of the Royal Statistical Society Series B}}. 1995;57(1):289--300.
\end{{thebibliography}}

\end{{document}}
"""
    (PAPER_DIR / "course_paper.tex").write_text(tex.strip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    draw_workflow()
    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    write_tex(payload)
    print(PAPER_DIR / "course_paper.tex")


if __name__ == "__main__":
    main()
