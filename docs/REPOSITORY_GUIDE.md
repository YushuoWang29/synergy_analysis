# 分析仓库导航

## 当前新增工作

| 工作 | 入口或位置 | 对应结果 |
|---|---|---|
| West 式任务均值 PCA | `code/myoki/save_west_all41_taskmean_all15_myoki.py` | `PCA_results/MyoKi/pca_myoki_west_all41_taskmean_all15_all23*` |
| RMS 起点、屈曲子集和差分变体 | `code/myoki/save_west_*.py`、`explore_west_variants_myoki.py` | 对应 NPZ 与 `_summary.json` |
| 时间序列 PCA | `code/myoki/pca_timeseries_west_myoki.py` | `pca_myoki_west_all_grasp_timeseries_all23*` |
| ShadowHand 截图 | `code/myoki/create_capture_shadow_hand.py`、`synergy_myoki_capture.py` | `right_hand_capture.xml`、`scene_right_capture.xml` |
| 课程统计研究 | `course_project/scripts/run_course_final_analysis.py` | `course_project/outputs/` |
| 论文生成 | `course_project/scripts/build_course_paper.py` | `course_project/paper/` |
| 个人演示文稿 | `course_project/presentation/` | 保留原文件名 |

## 历史工作

- `experiments/` 保留 V1–V5、任务平衡等实验脚本。
- `analysis/` 保留结果分析；`analysis/_scripts/` 保存内部与探索性脚本。
- `docs/note.md`、`docs/note_balance_summary.md` 及实验报告保留各自研究口径。
- `latex_chapter/` 与 `course_project/` 分开，避免把不同报告当成同一版本。
- `reference/course_materials/` 保存原 `course presentation/` 中的课程要求和授课资料；个人演示文稿归入课程项目。

## 模型与数据

- `model/mujoco_menagerie/shadow_hand/` 按普通文件随仓库提供，保留上游 `LICENSE`；本机嵌套 `.git` 不上传。
- 截图模型与原始模型分别命名，不覆盖原始 ShadowHand 模型。
- 小型 PCA 模型、统计表、研究图表和最终 PDF 保留；数据库与编译缓存见 [数据说明](DATA.md)。
- 所有运行命令以仓库根目录为工作目录；已有代码中部分旧的本机路径仅作为历史脚本保留。
