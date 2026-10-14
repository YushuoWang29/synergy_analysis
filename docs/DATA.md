# 数据准备与版本管理边界

## 仅存本地的大型数据

以下路径被 `.gitignore` 排除，整理过程不删除或移动其中的数据库文件：

```text
MyoKi Dataset/*.mat                 # 包括直接放在此处的 P01.mat、P26.mat
MyoKi Dataset/raw/                  # 原始参与者文件
MyoKi Dataset/preprocessed/         # 预处理参与者与合并文件
MyoKi Data/                        # 历史数据目录
Ninapro Data/                      # Ninapro 原始和预处理数据
```

2026-10-14 本地盘点：`MyoKi Dataset/` 约 8013 MiB，`Ninapro Data/` 约 6310 MiB。该体积是本机快照，不是数据集官方大小。

## 随仓库提供

- `MyoKi Dataset/Database_comparison.py` 与三个 XLSX 元数据/任务说明文件。
- `PCA_results/` 下的 NPZ、JSON 和结果说明；它们是衍生分析产物，不是原始数据库。
- 课程项目的统计 CSV、JSON、图表、论文及生成脚本。
- ShadowHand XML、网格、贴图和上游许可证。

## 本地复现

1. 将合法获取的 MyoKi 原始参与者数据放入 `MyoKi Dataset/raw/`；来源和数据结构参考 `reference/` 内已有数据集论文及元数据。
2. 在仓库根目录执行 `python code/myoki/preprocess_myoki.py --participants P01 P02 P03`，结果写入 `MyoKi Dataset/preprocessed/`。
3. 若原始数据直接放在 `MyoKi Dataset/`，可使用 `--raw_dir "MyoKi Dataset"` 显式指定位置，无需移动大型文件。
4. 完整课程分析需要脚本列出的全部参与者数据；只准备三个参与者不能复现全部课程结果。
5. 运行 `python code/myoki/pca_myoki.py` 或课程项目入口。MyoKi 原始文件很大，完整重算需要相应内存与时间。

## Git 历史说明

远端旧提交已包含少量预处理 MAT 文件。本次合并在新的仓库版本中取消跟踪这些数据，但保留旧提交历史，因此不保证克隆历史时完全不下载旧数据。新提交不再添加原始或预处理数据库。

上传前可用 `git ls-files -- "MyoKi Dataset/*.mat" "MyoKi Dataset/raw/*" "MyoKi Dataset/preprocessed/*" "Ninapro Data/*"` 检查，当前版本应没有输出。
