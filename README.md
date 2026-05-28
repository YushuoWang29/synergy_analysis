# Synergy Analysis of Hand Motion

This project explores human hand postures through **Principal Component Analysis (PCA)**. It uses public hand posture databases — **MyoKi** (primary) and **Ninapro** — visualizes motions on the **MuJoCo ShadowHand** model, performs PCA to extract synergies (principal components) from the **16-D hand posture space** (wrist excluded), and provides interactive exploration of the synergy space.

The core research question: **Can human hand synergies be aligned with the Pisa/IIT SoftHand synergies (S1 = uniform close, S2 = finger gradient)?**

## Project Structure

```
synergy_analysis/
│
├── README.md                                    # This file
├── environment.yml                              # Conda environment
├── .gitignore                                   # Git ignore rules
├── check_v5_status.bat                          # V5 experiment status checker
│
├── code/                                        # Core library code
│   ├── pre_processing.py                        # Ninapro preprocessing (legacy)
│   ├── run_pca.py                               # Ninapro PCA (legacy)
│   ├── synergy_demo.py                          # Ninapro interactive demo (legacy)
│   ├── interactive_test.py                      # Ninapro interaction test (legacy)
│   ├── visualization.py                         # Unified visualization (Ninapro + MyoKi)
│   ├── generate_report.py                       # Report generation (legacy)
│   ├── generate_pc_comparison.py                # PC comparison figures
│   ├── generate_proposal_figures.py             # Proposal/research figures
│   ├── extract_pdf_figures.py                   # Extract figures from PDFs
│   ├── analyze_grasp_tasks.py                   # Grasp task analysis
│   ├── myoki/                                   # MyoKi analysis package (recommended)
│   │   ├── __init__.py
│   │   ├── myoki_config.py                      # Joint mapping: MyoKi → ShadowHand
│   │   ├── preprocess_myoki.py                  # Data preprocessing (segmentation, wrist removal)
│   │   ├── pca_myoki.py                         # PCA analysis on 16-D hand space
│   │   ├── visualize_myoki.py                   # Motion playback on ShadowHand
│   │   ├── visualize_pc_grid.py                 # PC grid visualization
│   │   ├── synergy_myoki.py                     # Interactive synergy space exploration
│   │   └── camera_calibrator.py                 # Camera calibration for visualization
│   └── ninapro/                                 # Ninapro analysis package
│       ├── __init__.py
│       ├── ninapro_config.py                    # Ninapro joint configuration
│       ├── preprocess_ninapro.py                # Ninapro preprocessing
│       └── pca_ninapro.py                       # Ninapro PCA analysis
│
├── experiments/                                 # Experiment runner scripts
│   ├── mega_experiment.py                       # v1: Exhaustive search for optimal configuration
│   ├── mega_experiment_v2.py                    # v2: Refined experiment set
│   ├── mega_experiment_v3.py                    # v3: Additional configurations
│   ├── mega_experiment_v4.py                    # v4: Final best-config experiments
│   ├── mega_experiment_v5.py                    # v5: Mega-scale combined search
│   └── mega_balance_search.py                   # Task-balancing experiments
│
├── analysis/                                    # Experiment result analysis scripts
│   ├── analyze_experiments.py                   # Rank and analyze experiment results
│   ├── analyze_glove.py                         # Glove data analysis
│   ├── check_best_pc.py                         # Verify best PC configurations
│   ├── check_pca_data.py                        # PCA data consistency checks
│   ├── experiment_deep_dive.py                  # Deep analysis of specific experiments
│   ├── experiment_search_pca.py                 # PCA search space explorer
│   ├── experiment_task_analysis.py              # Task-by-task PCA analysis
│   ├── final_report.py                          # Generate final comparison report
│   ├── quick_check.py                           # Quick overview of top results
│   ├── quick_ninapro.py                         # Quick Ninapro data exploration
│   ├── save_best_results.py                     # Export best results to .npz
│   ├── show_best_result.py                      # Display best result details
│   ├── vis_trialrange_grid.py                   # Visualize trial range grid
│   │
│   └── _scripts/                                # Internal/ad-hoc analysis scripts
│       ├── f.py                                 # Ninapro full PCA pipeline
│       ├── _analyze_balance_results.py          # Balance search analysis
│       ├── _analyze_pc1_pc2.py                  # PC1/PC2 pattern analysis
│       ├── _check_npz_exports.py                # NPZ export verification
│       ├── _check_v5_progress.py                # V5 progress monitor
│       ├── _compare_results.py                  # Cross-experiment comparison
│       ├── _current_status.py                   # Project status overview
│       ├── _deeper_search.py                    # Extended configuration search
│       ├── _exp_clean.py                        # Clean experiment artifacts
│       ├── _exp_wide.py                         # Wide experiment sweep
│       ├── _explore_data.py                     # Data exploration
│       ├── _explore_ninapro.py                  # Ninapro data exploration
│       ├── _explore_results.py                  # Results exploration
│       ├── _export_balance_best.py              # Export best balance configs
│       ├── _export_best.py                      # Export best results
│       ├── _export_top_results.py               # Export top-N results
│       ├── _final_export.py                     # Final export for publication
│       ├── _finalize_best.py                    # Finalize best configuration
│       ├── _finalize_v5.py                      # Finalize V5 experiment
│       ├── _inspect_npz.py                      # NPZ file contents inspector
│       ├── _inspect_pcs.py                      # PC weight inspector
│       ├── _run_ninapro_pca.py                  # Run Ninapro PCA pipeline
│       ├── _search_sweet_spot.py                # Sweet spot parameter search
│       ├── _summarize_results.py                # Results summarization
│       ├── _verify_best.py                      # Best result verification
│       └── _vis_exp90.py                        # 90% variance visualization
│
├── docs/                                        # Documentation and notes
│   ├── note.md                                  # Final results summary (best configs, insights)
│   ├── note_balance_summary.md                  # Balance search summary
│   ├── experiment_report_allgrasp_range_all15.md      # Experiment report (English)
│   └── experiment_report_allgrasp_range_all15_zh.md   # Experiment report (中文)
│
├── latex_chapter/                               # LaTeX chapter for thesis
│   ├── chapter_synergy_pca.tex                  # Main LaTeX source
│   ├── chapter_synergy_pca.pdf                  # Compiled PDF
│   ├── generate_chapter_figures.py              # Figure generation for chapter
│   ├── generate_pdf.py                          # PDF compilation script
│   ├── peek_pca_data.py                         # PCA data inspection
│   ├── task_progress.md                         # Writing progress tracker
│   └── figures/                                 # Chapter figures (7 pre-generated images)
│
├── MyoKi Dataset/                               # MyoKi motion capture data
│   ├── raw/                                     # Raw .mat files (P01.mat, P02.mat, ...)
│   ├── preprocessed/                            # Preprocessed .mat files (16-D hand-only)
│   ├── Glove_calibration.xlsx                   # Glove calibration data
│   ├── Task_categorization.xlsx                 # Task IDs 1-74 with descriptions/categories
│   ├── Participant_information.xlsx             # Participant demographics
│   └── Database_comparison.py                   # Database comparison script
│
├── Ninapro Data/                                # Ninapro EMG/kinematics database
│   ├── E1/ E2/ E3/                              # Exercise 1, 2, 3 data
│   ├── original/                                # Original unprocessed files
│   └── preprocessed/                            # Preprocessed .mat files
│
├── PCA_results/                                 # All PCA output files
│   ├── MyoKi/                                   # MyoKi PCA models (.npz)
│   ├── Ninapro/                                 # Ninapro PCA models (.npz + .png)
│   ├── experiments/                             # Experiment result files
│   │   ├── mega_experiment_results.json         # V1 results (ranked)
│   │   ├── mega_v2_results.json                # V2 results
│   │   ├── mega_v3_results.json                # V3 results
│   │   ├── mega_v4_results.json                # V4 results
│   │   ├── mega_v5_results.json                # V5 results (if completed)
│   │   ├── mega_balance_results.json           # Balance search results
│   │   ├── deep_search_results.json            # Deep search results
│   │   ├── experiment_results.json             # Round 1 experiments
│   │   ├── experiment_deep_results.json        # Deep experiment results
│   │   ├── clean_results.json                  # Cleaned/filtered results
│   │   ├── BEST_RESULTS.md                     # Best results summary
│   │   ├── balance_*.npz                       # Top balance configuration exports
│   │   ├── pca_balance_*.npz                   # Balance PCA models
│   │   ├── final_*.npz                         # Final configuration models
│   │   └── best_result_*.npz                   # Best per-participant results
│   └── figures/                                # Related figures (if any)
│
├── MyoKi_figures/                               # MyoKi visualization output
│   ├── myoki_variance.png                       # Variance explained plot
│   ├── myoki_eigenpostures.png                  # First 3 eigenpostures
│   ├── pc_grid_pca_myoki_*.png                  # PC grid visualizations
│   ├── pisa_rotation_*.png                      # Pisa/IIT Procrustes rotation
│   ├── variance_all_frames.png                  # Variance for all frames
│   ├── eigenpostures_*.png                      # Per-participant eigenpostures
│   ├── pc_comparison_all_participants.png        # Cross-participant comparison
│   ├── pca_grasp_vs_nongrasp.png                # Grasp vs non-grasp comparison
│   ├── pc_trialmeans_vs_allframes.png           # Trial means vs all frames
│   └── experiments/                             # Experiment figure output
│       ├── mega_top_experiments.png             # Top-N experiment comparison
│       ├── best_experiments.png                 # Best experiment summary
│       ├── best_result_detailed.png             # Best result detailed view
│       ├── best_result_summary.png              # Best result summary
│       └── final_comparison.png                 # Final config comparison
│
├── Ninapro_figures/                             # Ninapro visualization output
│   ├── variance_E1.png, variance_E2.png, ...    # Variance per exercise
│   ├── eigenpostures_E1.png, ...                # Eigenpostures per exercise
│   └── pisa_E1.png, ...                         # Pisa rotation figures
│
├── figure/                                      # Miscellaneous figures
│   ├── plot_forefinger_F.png                    # Forefinger force plot
│   └── plot_forefinger_F_A.png                  # Forefinger force-angle plot
│
├── model/mujoco_menagerie/                      # MuJoCo ShadowHand model
│
└── reference/                                   # Reference papers (PDFs)
    ├── A database of multimodal myography and hand kinematics ... .pdf
    ├── Electromyography data for non-invasive ... robotic hand prostheses.pdf
    ├── Feix 等 - 2016 - The GRASP Taxonomy of Human Grasp Types.pdf
    ├── Postural Hand Synergies for Tool Use.pdf
    ├── PisaIIT_SoftHand_0.pdf
    ├── Toward_Dexterous_Manipulation_With_Augmented_Adaptive_Synergies.pdf
    └── nihms922186.pdf
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
2. **Strip palmar arch (ch15) + wrist channels (16,17) → keep 15 hand-only channels**
3. Segment by task label (tasks 1-74 = daily activities, 0 = rest)
4. Remove static frames via velocity threshold (0.5°/frame)
5. Resample each trial to 100 frames
6. Save participant + combined `.mat` files (15-D)

### 3. Visualize Motion on ShadowHand

```bash
# List available tasks with descriptions
python code/myoki/visualize_myoki.py --participant P01 --list_tasks

# Play a specific task (e.g., Task 5: "Take cup from 2, place on A")
python code/myoki/visualize_myoki.py --participant P01 --task 5 --subsample 500
```

### 4. PCA Analysis

```bash
python code/myoki/pca_myoki.py
```

Generates:
- `PCA_results/MyoKi/pca_myoki_*.npz` — PCA model
- `MyoKi_figures/myoki_variance.png` — Variance explained plot
- `MyoKi_figures/myoki_eigenpostures.png` — First 3 eigenpostures
- `MyoKi_figures/pisa_rotation_*.png` — Procrustes rotation to Pisa targets

### 5. Interactive Synergy Exploration

```bash
python code/myoki/synergy_myoki.py
```

- **2D Mode (default):** Click/drag on PC1-PC2 plane → real-time hand pose update
- **Slider Mode:** Press `m` to toggle sliders for first 5 PCs

## Running Experiments

The project includes 5 generations of large-scale experiments searching for the optimal data configuration:

```bash
# Run from project root
python experiments/mega_experiment.py       # V1: broad sweep
python experiments/mega_experiment_v2.py    # V2: refined search
python experiments/mega_experiment_v4.py    # V4: focused best-config
python experiments/mega_balance_search.py   # Task-balancing search
```

Results are saved to `PCA_results/experiments/` as JSON and NPZ files.

## Analyzing Results

```bash
# Quick overview
python analysis/quick_check.py
python analysis/analyze_experiments.py

# Detailed analysis
python analysis/experiment_deep_dive.py
python analysis/final_report.py
python analysis/show_best_result.py
```

## Key Findings

| Configuration | PC1+PC2 | S1 (uniform) | S2 (gradient) |
|--------------|---------|-------------|--------------|
| **★ Best: power grasps + taskavg static5 + flexion10 joints** | **95.2%** | **0.9435** | **0.8100** |
| Flexion10 + allgrasp + taskavg | 92.7% | 0.9514 | 0.7929 |
| No abduction + power + taskavg | 93.3% | 0.8441 | 0.7750 |
| All 15 joints + power + taskavg | 92.8% | 0.7787 | 0.7166 |

**What works:**
- **Flexion-only joints** (10-joint subset: MCP+PIP of all fingers + thumb IP)
- **Task-averaging** each trial into a single posture
- **Static hold frames** (last 5 frames of each trial)
- **Power grasps only** (cylindrical/spherical/hook)

**What fails:**
- Including abduction joints adds noise
- Including non-grasp tasks corrupts synergy structure
- Per-subject PCA without averaging loses clean structure
- Ninapro S2 consistently poor (~0.2 vs target)

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

## PCA Results (MyoKi, 11 participants, 15-D hand space)

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

## Adding New Participants

1. Place `.mat` files in `MyoKi Dataset/raw/`
2. Run `python code/myoki/preprocess_myoki.py --participants P04 P05 ...`
3. Run `python code/myoki/pca_myoki.py` to recompute PCA

## Notes

- **Wrist excluded:** MyoKi channels 16 (wrist flex/ext) and 17 (wrist rad/ulnar) are removed from PCA. This prevents wrist variance from dominating the hand posture analysis.
- **Palmar arch also excluded:** Channel 15 (palmar arch) is removed as it had limited range and added noise.
- **Data format:** CyberGlove measures joint angles in degrees. Converted to radians for MuJoCo and clamped to joint limits.
- **Preprocessed vs Raw:** Preprocessed data has wrist removed, static frames filtered, and uniform trial length (100 frames). Raw data preserves full recording fidelity.
- `check_v5_status.bat` is a utility script to check if the long-running V5 experiment is still running, has completed, or crashed.
