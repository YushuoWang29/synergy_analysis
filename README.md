# synergy_analysis

**人手运动学协同分析**：围绕 MyoKi 与 Ninapro 数据，进行预处理、PCA、ShadowHand 姿态展示、实验比较和课程统计分析。

## 从哪里开始

- **课程研究主线**：[课程项目说明](course_project/README.md)、[论文 PDF](course_project/paper/course_paper.pdf)。
- **完整分析流程**：[详细参考](docs/analysis_reference.md)。
- **目录与结果索引**：[仓库导航](docs/REPOSITORY_GUIDE.md)。
- **数据库准备与上传边界**：[数据说明](docs/DATA.md)。
- **本次本地新增与远端整合**：[同步记录](docs/SYNC_2026-10-14.md)。

## 安装与运行

从仓库根目录执行。`environment.yml` 保存本地 Windows/Conda 环境版本，不包含个人安装路径。

```powershell
conda env create -f environment.yml
conda activate hand_synergy
python code/myoki/synergy_myoki.py
```

已有小型 PCA 结果与 ShadowHand 模型随仓库提供；重新预处理或运行完整统计流程需要另行准备本地数据库。

```powershell
# 先按 docs/DATA.md 准备数据
python code/myoki/preprocess_myoki.py --participants P01 P02 P03
python code/myoki/pca_myoki.py

# 课程统计、图表与论文流程；需已激活环境，PDF 编译还需 XeLaTeX
powershell -ExecutionPolicy Bypass -File course_project/run_all.ps1
```

## 目录职责

| 目录 | 内容 | 使用方式 |
|---|---|---|
| `code/myoki/`、`code/ninapro/` | 数据处理、PCA、可视化与截图 | 分析入口 |
| `experiments/` | 多代参数搜索与任务平衡实验 | 历史实验复现 |
| `analysis/` | 结果检查、比较和导出 | 分析已有实验 |
| `PCA_results/` | NPZ 模型、JSON 摘要与实验结果 | 版本化小型衍生成果 |
| `MyoKi_figures/`、`Ninapro_figures/`、`figure/` | 图表和姿态图 | 历史成果 |
| `course_project/` | 统计脚本、论文、结果表、图表和个人演示文稿 | 当前课程项目主线 |
| `latex_chapter/` | 历史论文章节与生成脚本 | 独立保留 |
| `model/mujoco_menagerie/shadow_hand/` | ShadowHand 模型、网格和许可证 | 原模型与截图模型分开 |
| `docs/` | 使用参考、实验记录、数据与同步说明 | 文档入口 |
| `reference/` | 论文与课程参考材料 | `course_materials/` 单独归档课程资料 |
| `MyoKi Dataset/`、`Ninapro Data/` | 原始及预处理数据库 | 大型数据仅留本地 |

## 结果的适用范围

课程结果以 [统计 JSON](course_project/outputs/data/course_project_stats.json) 与 [结果表](course_project/outputs/data/prediction_summary.csv) 为依据。历史参数搜索、15 维全关节、10 维屈曲子集及不同任务聚合方式分别保留，不混用其解释率或排名。

当前关节映射由 [myoki_config.py](code/myoki/myoki_config.py) 定义，使用 **15 个人手通道**，排除掌弓和两个腕通道。历史结果可能使用不同配置，应读取对应文件与摘要。

## 关联仓库

[synergy_hand_sim](https://github.com/YushuoWang29/synergy_hand_sim) 负责折纸手设计、协同模型和物理仿真；本仓库负责数据分析和统计研究。
