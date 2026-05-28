# Hand Posture Analysis Using MyoKi Dataset: Task-Level Range-Based PCA (Expanded to 23 Participants)

## 1. Dataset

This analysis uses the **MyoKi (Myo Kinetic) dataset**, a comprehensive collection of hand kinematic data recorded from 23 healthy participants (P01–P15, P17–P24) using a CyberGlove II data glove. The glove captures 15 joint angles at 100 Hz covering all major degrees of freedom of the human hand:

| Joint Index | Joint Name       | Description                            |
|-------------|------------------|----------------------------------------|
| 0           | ThumbAbd         | Thumb abduction/adduction              |
| 1           | ThumbMCP         | Thumb metacarpophalangeal (flexion)    |
| 2           | ThumbIP          | Thumb interphalangeal (flexion)        |
| 3           | ThumbOpp         | Thumb opposition                       |
| 4           | IdxMCP           | Index finger MCP (flexion)             |
| 5           | IdxPIP           | Index finger PIP (flexion)             |
| 6           | MidMCP           | Middle finger MCP (flexion)            |
| 7           | MidPIP           | Middle finger PIP (flexion)            |
| 8           | IdxMidAbd        | Index–middle finger abduction          |
| 9           | RingMCP          | Ring finger MCP (flexion)              |
| 10          | RingPIP          | Ring finger PIP (flexion)              |
| 11          | MidRingAbd       | Middle–ring finger abduction           |
| 12          | LitMCP           | Little finger MCP (flexion)            |
| 13          | LitPIP           | Little finger PIP (flexion)            |
| 14          | RingLitAbd       | Ring–little finger abduction           |

All 15 glove channels are used without any dimensionality reduction or channel exclusion. The raw glove signals have been calibrated to joint angles in degrees using the manufacturer-provided calibration profiles stored in `Glove_calibration.xlsx`.

The original MyoKi dataset includes 35 participants (P01–P35). For this analysis:
- **Participants P01–P15 and P17–P24** (23 participants) are included, as they use the same 18-sensor CyberGlove II configuration with consistent sensor placement.
- **Participant P16** is excluded due to left-handedness.
- **Participants P25–P35** are excluded because they use a different sensor configuration (different glove or sensor setup).
- All included participants are right-handed with no known hand motor impairments.

## 2. Task Selection

The MyoKi protocol contains a wide variety of everyday manipulation tasks. For this analysis, we focus on **grasping tasks only**—tasks that involve actively grasping and manipulating an object. Non-grasping actions (e.g., pointing, resting posture, non-specific hand motions) are excluded.

The grasp categories included are:

| Grasp Category  | Description                                                        | Example Tasks                                                                 |
|-----------------|--------------------------------------------------------------------|-------------------------------------------------------------------------------|
| **Cylindrical** | Power grasp around a cylindrical object                            | Lifting cups, bottles of various weights                                     |
| **Spherical**   | Power grasp around a spherical or rounded object                   | Lifting balls of different sizes (d = 65, 85, 100 mm)                        |
| **Hook**        | Hook grip (tension-based, fingers acting as hooks)                 | Lifting laptop bags (0.75–3.0 kg), using saw                                  |
| **Tripod**      | Precision grip with three fingers (thumb, index, middle)            | Picking up tea cups by the rim                                                |
| **Pinch**       | Precision grip involving thumb–index opposition                    | Using a pen to write                                                         |
| **Complex**     | Multi-pattern grasps combining elements of multiple categories     | Using tools (hammer, screwdriver) with both power and precision components   |

In total, **41 distinct grasping tasks** are selected. Each task was performed by each participant in multiple trials (typically 3–5 repetitions per participant), resulting in a total of **5,656 trial recordings** across all 23 participants.

A detailed task list is provided in Appendix A.

## 3. Participants

All 23 included participants from the MyoKi dataset are:

| Participant | Age | Gender | Handedness |
|-------------|-----|--------|------------|
| P01         | 24  | M      | Right      |
| P02         | 24  | F      | Right      |
| P03         | 26  | M      | Right      |
| P04         | 21  | M      | Right      |
| P05         | 19  | F      | Right      |
| P06         | 56  | M      | Right      |
| P07         | 33  | F      | Right      |
| P08         | 30  | M      | Right      |
| P09         | 55  | F      | Right      |
| P10         | 63  | M      | Right      |
| P11         | 31  | M      | Right      |
| P12         | 55  | F      | Right      |
| P13         | 68  | F      | Right      |
| P14         | 73  | M      | Right      |
| P15         | 30  | F      | Right      |
| P17         | 47  | M      | Right      |
| P18         | 59  | M      | Right      |
| P19         | 53  | F      | Right      |
| P20         | 62  | M      | Right      |
| P21         | 76  | M      | Right      |
| P22         | 54  | F      | Right      |
| P23         | 22  | F      | Right      |
| P24         | 48  | F      | Right      |

All participants are right-handed, with no known hand motor impairments. The group includes 12 males and 11 females, spanning an age range of 19–76 years.

## 4. Preprocessing Pipeline

The preprocessing pipeline is designed to extract a compact yet representative feature vector for each grasping task. The rationale is to capture **how each joint modulates its angle range** during task execution, rather than the absolute posture at any single time point.

### Step 1: Channel Selection

Each raw recording contains 18 CyberGlove channels. The palmar arch (channel 15), wrist flexion/extension (channel 16), and wrist radial/ulnar deviation (channel 17) channels are removed, retaining **15 hand-related joint angles** for analysis.

### Step 2: Trial Segmentation

Each participant's recording session contains continuous time-series data with task markers. Trial boundaries are identified from the task labels, and individual trials are extracted as continuous segments. Trials shorter than 5 frames (50 ms) are discarded as unreliable.

### Step 3: Motion Frame Selection

To focus on active motion rather than static holds, frames with low velocity (gradient norm below a threshold of 0.5 degrees/frame) are removed. This eliminates static postures and transitional resting periods, ensuring that only dynamic motion segments contribute to the range computation.

### Step 4: Temporal Interpolation

To ensure consistent temporal sampling across trials of different durations, each trial is resampled to a uniform length of 100 frames using linear interpolation. This step preserves the full kinematic trajectory while removing duration as a confounding factor.

### Step 5: Range Feature Extraction

For each trial, we compute the **angular range of motion** for each of the 15 joints:

$$r_j^{(t)} = \max_{f \in \text{trial } t} \theta_j(f) - \min_{f \in \text{trial } t} \theta_j(f)$$

where $\theta_j(f)$ is the angle of joint $j$ at frame $f$.

This range feature encodes the **excursion amplitude** of each joint during task performance—a measure of how much that joint is recruited. Unlike static posture measures (which capture only one moment), or mean posture measures (which can cancel out opposing motions), the range captures the full dynamic involvement of each joint.

### Step 6: Task-Level Aggregation

Multiple trials of the same task (across all participants and repetitions) are aggregated by taking the **per-joint mean** across all trials belonging to that task:

$$X_{ij} = \frac{1}{N_i} \sum_{t \in \text{trials of task } i} r_j^{(t)}$$

where $N_i$ is the number of trials available for task $i$.

This yields a **41 × 15 data matrix** where each row corresponds to one manipulation task and each column to one joint. Each cell represents the average angular range (in degrees) with which that joint is recruited during that task.

No additional normalization (e.g., z-score or min-max scaling) is applied, preserving the natural scale differences between joints. This allows joints with intrinsically larger ranges (e.g., MCP joints) to contribute proportionally to the analysis.

### Step 7: Dimensionality Reduction via PCA

Principal Component Analysis (PCA) is performed on this 41 × 15 matrix. The covariance-based PCA preserves the natural variance structure since all variables are in the same physical units (degrees). The first 10 components are retained for analysis.

The mean pose vector (mean of all 41 task vectors) is stored alongside the component loadings for reconstruction purposes.

## 5. Results

### Variance Explained

The first two principal components account for **91.0% of the total variance** in the dataset:

| Component | Variance Explained | Cumulative |
|-----------|:-----------------:|:----------:|
| PC1       | 78.04%            | 78.04%     |
| PC2       | 12.98%            | 91.02%     |
| PC3       | 3.29%             | 94.31%     |
| PC4       | 2.19%             | 96.50%     |
| PC5       | 1.40%             | 97.90%     |
| PC6       | 0.66%             | 98.55%     |
| PC7       | 0.61%             | 99.16%     |
| PC8       | 0.36%             | 99.53%     |
| PC9       | 0.18%             | 99.71%     |
| PC10      | 0.14%             | 99.85%     |

The dominant contribution of PC1 and PC2 confirms that hand grasping kinematics can be effectively described by a two-dimensional synergy space. Compared to the previous analysis with 11 participants (PC1+PC2 = 90.9%), the variance explained by the first two PCs is virtually unchanged (91.0%), demonstrating the robustness of this low-dimensional structure.

### Joint Loading Patterns

**PC1** (78.04% of variance) shows **positive loadings across all 15 joints**, indicating that it represents a coordinated activation pattern where all joints move together. Joints with the highest loadings include the MCP joints of the middle (0.355), ring (0.365), and little (0.338) fingers, as well as the PIP joints of the ring (0.310) and little (0.314) fingers. This component captures the overall intensity of the grasp—stronger grasps drive all joints toward larger excursions.

**PC2** (12.98% of variance) reveals a **gradient pattern along the hand's mediolateral axis**:
- **Thumb** joints load strongly positively, especially opposition (ThumbOpp: +0.729) and abduction (ThumbAbd: +0.383)
- **Index finger** joints show moderate positive loadings (IdxMCP: +0.096, IdxPIP: +0.126)
- **Middle finger** joints load negatively (MidMCP: −0.132, MidPIP: −0.191)
- **Ring and little finger** joints load negatively, especially PIP joints (RingPIP: −0.290, LitPIP: −0.239)

This component captures the contrast between thumb-index driven precision actions and ulnar-side (ring–little finger) power stabilization. Tasks that rely more on the radial side of the hand (e.g., precision grips, tripod grasps) are separated from those that engage the full hand (e.g., power grasps on large objects).

### Comparison with Previous Analysis (11-Participant Version)

The key metrics are highly consistent between the two analyses:

| Metric                | 11 Participants | 23 Participants | Change |
|-----------------------|:---------:|:---------:|:------:|
| PC1 variance          | 76.95%    | 78.04%    | +1.09% |
| PC2 variance          | 13.97%    | 12.98%    | −0.99% |
| PC1+PC2 cumulative    | 90.92%    | 91.02%    | +0.10% |
| Number of trials      | ~2,700    | 5,656     | ×2.1   |

The remarkable stability of the results when more than doubling the sample size strongly validates the existence of consistent low-dimensional synergy structures in human hand grasping.

### Interpretation

The two-component model captures two fundamental axes of hand grasping:

1. **PC1 — Overall Grasp Intensity**: The coordinated recruitment of all joints, distinguishing light-touch manipulations from forceful grasps.
2. **PC2 — Radial–Ulnar Gradient**: The balance between thumb-index engagement and ring–little finger engagement, distinguishing precision-oriented tasks from power-oriented tasks requiring full-palmar contact.

Together, these two dimensions span a functional space that naturally organizes the 41 grasping tasks according to their biomechanical demands.

## 6. Storage

The PCA model is stored as a NumPy `.npz` file containing:
- `components`: (10 × 15) matrix of PC loadings
- `mean`: (15,) mean joint angle range vector
- `explained_variance`: (10,) variance ratios
- `explained_variance_cumsum`: (10,) cumulative variance ratios
- `n_tasks`: number of tasks used (41)
- `n_joints`: number of joints (15)
- `data_label`: descriptive string

This model can be loaded and used for reconstruction, visualization, or as a basis for synergy-based control.

---

## Appendix A: Task List

The following 41 grasping tasks were included in the analysis:

| Task ID | Description                           | Category        |
|---------|---------------------------------------|-----------------|
| 1       | Take cup from 2 to C                  | Cylindrical     |
| 2       | Take cup from C to 2                  | Cylindrical     |
| 3       | Take cup from 2 to B                  | Cylindrical     |
| 4       | Take cup from B to 2                  | Cylindrical     |
| 5       | Take cup from 2 to A                  | Cylindrical     |
| 6       | Take cup from A to 2                  | Cylindrical     |
| 9       | Take empty bottle from 2 to A         | Cylindrical     |
| 10      | Take empty bottle from A to 2         | Cylindrical     |
| 11      | Take half-full bottle from 2 to A     | Cylindrical     |
| 12      | Take half-full bottle from A to 2     | Cylindrical     |
| 13      | Take full bottle from 2 to A          | Cylindrical     |
| 14      | Take full bottle from A to 2          | Cylindrical     |
| 15      | Pour water from full bottle           | Cylindrical     |
| 16      | Take tea cup from 2 to C              | Tripod          |
| 17      | Take tea cup from C to 2              | Tripod          |
| 18      | Take tea cup from 2 to B              | Tripod          |
| 19      | Take tea cup from B to 2              | Tripod          |
| 20      | Take tea cup from 2 to A              | Tripod          |
| 21      | Take tea cup from A to 2              | Tripod          |
| 28      | Take ball d=65mm from 1 to A          | Spherical       |
| 29      | Take ball d=65mm from A to 1          | Spherical       |
| 30      | Take ball d=85mm from 2 to A          | Spherical       |
| 31      | Take ball d=85mm from A to 2          | Spherical       |
| 32      | Take ball d=100mm from 3 to A         | Spherical       |
| 33      | Take ball d=100mm from A to 3         | Spherical       |
| 40      | Lift 0.75kg laptop bag                | Hook            |
| 41      | Replace 0.75kg laptop bag             | Hook            |
| 42      | Lift 1.5kg laptop bag                 | Hook            |
| 43      | Replace 1.5kg laptop bag              | Hook            |
| 44      | Lift 3.0kg laptop bag                 | Hook            |
| 45      | Replace 3.0kg laptop bag              | Hook            |
| 46      | Wipe dust at position 1               | Complex         |
| 47      | Wipe dust at position 2               | Complex         |
| 48      | Wipe dust at position 3               | Complex         |
| 49      | Hammer (low force)                    | Complex         |
| 50      | Hammer (medium force)                 | Complex         |
| 51      | Hammer (high force)                   | Complex         |
| 54      | Tighten horizontal screw              | Complex         |
| 55      | Loosen horizontal screw               | Complex         |
| 56      | Saw wood                              | Complex         |
| 61      | Write with pen                        | Pinch           |
