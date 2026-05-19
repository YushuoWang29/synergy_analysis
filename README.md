# Synergy Analysis of Hand Motion

This project explores human hand postures through **Principal Component Analysis (PCA)**. It uses public hand posture databases — **MyoKi** (primary) and **Ninapro** — visualizes motions on the **MuJoCo ShadowHand** model, performs PCA to extract synergies (principal components) from the **16-D hand posture space** (wrist excluded), and provides interactive exploration of the synergy space.

## Project Structure

```
synergy_analysis/
├── README.md                        # This file
├── environment.yml                  # Conda environment
│
├── code/                            # Source code
│   ├── visualization.py             # Unified visualization (Ninapro + MyoKi)
│   ├── pre_processing.py            # Ninapro preprocessing (legacy)
│   ├── run_pca.py                   # Ninapro PCA (legacy)
│   ├── synergy_demo.py              # Ninapro interactive demo (legacy)
│   ├── interactive_test.py          # Ninapro interaction test (legacy)
│   ├── generate_report.py           # Report generation (legacy)
│   └── myoki/                       # **MyoKi analysis package (recommended)**
│       ├── __init__.py
│       ├── myoki_config.py          # Joint mapping: MyoKi 16ch → ShadowHand 24dof
│       ├── preprocess_myoki.py      # Data preprocessing (segmentation, wrist removal)
│       ├── visualize_myoki.py       # Motion playback with OSD (participant, task, trial)
│       ├── pca_myoki.py             # PCA analysis on 16-D hand space
│       └── synergy_myoki.py         # Interactive synergy space exploration
│
├── MyoKi Dataset/
│   ├── raw/                         # Raw .mat files (P01.mat, P02.mat, ...)
│   ├── preprocessed/                # Preprocessed .mat files (16-D hand-only)
│   ├── Glove_calibration.xlsx
│   ├── Task_categorization.xlsx     # Task IDs 1-74 with descriptions
│   └── Participant_information.xlsx
│
├── PCA_results/
│   └── MyoKi/
│       └── pca_myoki_all.npz        # PCA model (16 components)
│
├── MyoKi_figures/
│   ├── myoki_variance.png
│   └── myoki_eigenpostures.png
│
├── model/mujoco_menagerie/shadow_hand/  # MuJoCo ShadowHand model
└── reference/                        # Reference papers
```

## Quick Start

### 1. Setup

```bash
conda env create -f environment.yml
conda activate hand_synergy
```

### 2. Preprocess MyoKi Data

Raw `.mat` files → segmented, wrist-stripped, resampled trials:

```bash
# Process all available participants
python code/myoki/preprocess_myoki.py

# Specific participants
python code/myoki/preprocess_myoki.py --participants P01 P02 P03
```

**Pipeline:**
1. Load raw CyberGlove data (18 channels @ 2000Hz)
2. **Strip wrist channels 16,17 → keep 16 hand-only channels**
3. Segment by task label (tasks 1-74 = daily activities, 0 = rest)
4. Remove static frames via velocity threshold (0.5°/frame)
5. Resample each trial to 100 frames
6. Save participant + combined `.mat` files (16-D)

### 3. Visualize Motion on ShadowHand

Play back recorded motions on MuJoCo ShadowHand with live status display:

```bash
# List available tasks with descriptions
python code/myoki/visualize_myoki.py --participant P01 --list_tasks

# Play a specific task (e.g., Task 5: "Take cup from 2, place on A")
python code/myoki/visualize_myoki.py --participant P01 --task 5 --subsample 500

# Play all tasks sequentially
python code/myoki/visualize_myoki.py --participant P01 --subsample 2000
```

During playback, the terminal shows: `[P01] Task 5: Take the cup from 2... | Frame 123/500 (25%)`.

**Note:** Playback uses raw data for full temporal resolution. The MuJoCo viewer window shows the hand; close it to stop.

### 4. PCA Analysis

```bash
python code/myoki/pca_myoki.py
```

Generates:
- `PCA_results/MyoKi/pca_myoki_all.npz` — PCA model
- `MyoKi_figures/myoki_variance.png` — Variance explained plot
- `MyoKi_figures/myoki_eigenpostures.png` — First 3 eigenpostures

### 5. Interactive Synergy Exploration

```bash
python code/myoki/synergy_myoki.py
```

- **2D Mode (default):** Click/drag on PC1-PC2 plane → real-time hand pose update
- **Slider Mode:** Press `m` to toggle sliders for first 5 PCs

## Joint Mapping: MyoKi → ShadowHand

**Wrist excluded.** MyoKi channels 16 (wrist flex/ext) and 17 (wrist rad/ulnar) are excluded from analysis.
The ShadowHand wrist joints (rh_WRJ1, rh_WRJ2) remain at 0.

| MyoKi # | MyoKi Joint | → | ShadowHand qpos | ShadowHand Joint |
|---------|-------------|---|-----------------|-----------------|
| 0 | Thumb Abduction | → | 21 | rh_THJ3 |
| 1 | Thumb MCP | → | 20 | rh_THJ4 |
| 2 | Thumb IP | → | 22 | rh_THJ2 |
| 3 | Thumb Opp/Abduction | → | 19 | rh_THJ5 |
| 4 | Index MCP | → | 3 | rh_FFJ3 |
| 5 | Index PIP | → | 4 | rh_FFJ2 |
| 6 | Middle MCP | → | 7 | rh_MFJ3 |
| 7 | Middle PIP | → | 8 | rh_MFJ2 |
| 8 | Index-Middle Abd | → | 6 | rh_MFJ4 |
| 9 | Ring MCP | → | 11 | rh_RFJ3 |
| 10 | Ring PIP | → | 12 | rh_RFJ2 |
| 11 | Middle-Ring Abd | → | 10 | rh_RFJ4 |
| 12 | Little MCP | → | 16 | rh_LFJ3 |
| 13 | Little PIP | → | 17 | rh_LFJ2 |
| 14 | Ring-Little Abd | → | 15 | rh_LFJ4 |
| 15 | Palmar Arch | → | 14 | rh_LFJ5 |

**DIP copies** (CyberGlove lacks DIP sensors; copied from PIP):
- FFJ1 ← Index PIP, MFJ1 ← Middle PIP, RFJ1 ← Ring PIP, LFJ1 ← Little PIP, THJ1 ← Thumb IP

**Unmapped:** FFJ4 (Index Abd) — no independent sensor.

## PCA Results (MyoKi, 3 participants, 16-D hand space)

| PC | Variance | Cumulative |
|----|----------|------------|
| PC1 | **43.25%** | 43.25% |
| PC2 | 21.74% | 64.99% |
| PC3 | 11.08% | 76.07% |
| PC4 | 5.90% | 81.97% |
| PC5 | 4.05% | 86.03% |
| PC6 | 3.54% | 89.57% |
| PC7 | 2.43% | **92.00%** |
| PC8 | 2.07% | 94.07% |
| PC9 | 1.97% | **96.04%** |

- **1 PC** → 43.3% variance
- **3 PCs** → 76.1%
- **7 PCs** → 92.0% (90% threshold)
- **9 PCs** → 96.0% (95% threshold)

Compared to the 18-D analysis (with wrist), removing wrist increased PC1 from 38.5% to 43.3%, showing hand postures have stronger low-dimensional structure when wrist motion is excluded.

## Unified Visualization

```bash
# MyoKi
python code/visualization.py --dataset myoki --participant P01 --task 5

# Ninapro
python code/visualization.py --dataset ninapro --path "Ninapro Data/E2/S5_E2_B.mat"
```

## Adding New Participants

1. Place `.mat` files in `MyoKi Dataset/raw/`
2. Run `python code/myoki/preprocess_myoki.py --participants P04 P05 ...`
3. Run `python code/myoki/pca_myoki.py` to recompute PCA

## Notes

- **Wrist excluded:** MyoKi channels 16 (wrist flex/ext) and 17 (wrist rad/ulnar) are removed from PCA. This prevents wrist variance from dominating the hand posture analysis.
- **Data format:** CyberGlove measures joint angles in degrees. Converted to radians for MuJoCo and clamped to joint limits.
- **Playback speed:** The terminal OSD shows frame progress. Default playback is ~60fps via `time.sleep(0.016)`.
- **Preprocessed vs Raw:** Preprocessed data has wrist removed, static frames filtered, and uniform trial length (100 frames). Raw data preserves full recording fidelity.
