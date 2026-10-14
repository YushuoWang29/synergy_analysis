# 课程 Final Project：MyoKi 抓取姿态协同统计分析

本文件夹包含面向“生物统计和生物信息引论”课程的项目版本。当前版本把主成分分析作为运动学表征层，而不是把它当作全部分析；后续加入自助法、复杂度指数、重复测量统计检验、多重比较校正和跨受试者预测验证。

## 研究问题

1. MyoKi 抓取运动学中是否存在稳定的低维姿态协同结构？
2. 按 West 等人的方差解释思路，不同抓取家族是否具有不同的复杂度？
3. 力量型、精细型和复杂型抓取是否在协同空间中存在显著差异？
4. 低维协同得分是否保留跨受试者预测抓取家族的信息？

## 方法框架

- 任务均值主成分分析，用于提取低维姿态协同表征。
- 受试者自助法，用于估计方差解释率和复杂度指数的不确定性。
- West 式方差解释复杂度指数，用于描述累计方差曲线的集中程度。
- 受试者阻塞方差分析，用于检验重复测量条件下的抓取家族效应。
- Benjamini-Hochberg 假发现率校正，用于控制多重比较。
- 留一受试者多项逻辑回归，用于评估跨个体泛化能力。
- 关节集敏感性分析，用于检查结果是否依赖某一组关节定义。

## 主要输出

- 课程论文 PDF：`course_project/paper/course_paper.pdf`
- 课程论文 LaTeX：`course_project/paper/course_paper.tex`
- 主分析脚本：`course_project/scripts/run_course_final_analysis.py`
- 课程论文生成脚本：`course_project/scripts/build_course_paper.py`
- 自动复现入口：`course_project/run_all.ps1`
- 图表：`course_project/outputs/figures/`
- 统计结果表：`course_project/outputs/data/`

## 复现方式

从项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File course_project\run_all.ps1
```

也可以分步运行：

```powershell
conda activate hand_synergy
python course_project/scripts/run_course_final_analysis.py
python course_project/scripts/build_course_paper.py
xelatex -interaction=nonstopmode -halt-on-error -output-directory=course_project/paper course_project/paper/course_paper.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=course_project/paper course_project/paper/course_paper.tex
```

## 当前关键结果

- 纳入 23 名右利手健康受试者、41 个抓取任务和 5,656 个试次。
- 前两主成分解释 91.0% 的任务级方差，自助法 95% 置信区间为 89.1% 到 91.4%。
- 整体 West 式复杂度指数为 0.030，达到 90% 方差解释率需要 2 个主成分。
- 抓取家族对第一和第二主成分均有显著影响，第二主成分的偏 η² 为 0.288。
- 留一受试者预测中，原始 15 维特征平衡准确率为 0.70 ± 0.09，前 5 主成分为 0.63 ± 0.10。

## 理论边界

本课程论文的理论背景以 Santello 等人的姿态协同研究为主，复杂度指标参考 West 等人的方差解释曲线思想。论文不设置外部手型模板作为评价目标，也不把结果解释为对某一机器人手协同模式的拟合。

## 环境与数据前提

- 先激活 `hand_synergy` 环境；也可给 `run_all.ps1` 传入 `-Python "完整/python.exe/路径"`。
- PDF 编译需可用的 XeLaTeX；可用 `-XeLaTeX "完整/xelatex.exe/路径"` 指定。缺少编译器时保留 TeX 并明确提示 PDF 未重建。
- 需要本地 MyoKi 预处理数据，详见 [数据说明](../docs/DATA.md)。
- 个人演示文稿在 `presentation/`；课程资料在 `../reference/course_materials/`。
- 当前数值来自已保存的 `outputs/data/course_project_stats.json` 和统计表；本次目录整理不代表重新运行了全套统计研究。
