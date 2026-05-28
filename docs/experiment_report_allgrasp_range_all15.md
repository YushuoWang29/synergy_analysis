# Hand Posture Analysis Using MyoKi Dataset: Task-Level Range-Based PCA

## 1. Dataset

This analysis uses the **MyoKi (Myo Kinetic) dataset**, a comprehensive collection of hand kinematic data recorded from 11 healthy participants (P01–P09, P11–P12) using a CyberGlove II data glove. The glove captures 15 joint angles at 100 Hz covering all major degrees of freedom of the human hand:

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
| **Lumbrical**   | Lumbrical grip (palm-oriented precision)                           | Pinch-like grasps with palmar support                                        |
| **Complex**     | Multi-pattern grasps combining elements of multiple categories     | Using tools (hammer, screwdriver) with both power and precision components   |

In total, **41 distinct grasping tasks** are selected. Each task was performed by each participant in multiple trials (typically 3–5 repetitions per participant), resulting in a total of approximately 2,700 trial recordings across all participants.

A detailed task list is provided in Appendix A.

## 3. Participants

All 11 participants from the MyoKi dataset are included:

| Participant | Age | Gender | Handedness |
|-------------|-----|--------|------------|
| P01         | 28  | M      | Right      |
| P02         | 26  | M      | Right      |
| P03         | 30  | M      | Right      |
| P04         | 24  | M      | Right      |
| P05         | 31  | M      | Right      |
| P06         | 22  | M      | Right      |
| P07         | 32  | F      | Right      |
| P08         | 25  | M      | Right      |
| P09         | 27  | M      | Right      |
| P11         | 26  | F      | Right      |
| P12         | 29  | F      | Right      |

All participants are right-handed, with no known hand motor impairments. The group includes 8 males and 3 females, spanning an age range of 22–32 years.

## 4. Preprocessing Pipeline

The preprocessing pipeline is designed to extract a compact yet representative feature vector for each grasping task. The rationale is to capture **how each joint modulates its angle range** during task execution, rather than the absolute posture at any single time point.

### Step 1: Trial Segmentation

Each participant's recording session contains continuous time-series data with task markers. Trial boundaries are identified from the task labels, and individual trials are extracted as continuous segments. Trials shorter than 5 frames (50 ms) are discarded as unreliable.

### Step 2: Temporal Interpolation

To ensure consistent temporal sampling across trials of different durations, each trial is resampled to a uniform length of 100 frames using linear interpolation. This step preserves the full kinematic trajectory while removing duration as a confounding factor.

### Step 3: Range Feature Extraction

For each trial, we compute the **angular range of motion** for each of the 15 joints:

$$r_j^{(t)} = \max_{f \in \text{trial } t} \theta_j(f) - \min_{f \in \text{trial } t} \theta_j(f)$$

where $\theta_j(f)$ is the angle of joint $j$ at frame $f$.

This range feature encodes the **excursion amplitude** of each joint during task performance—a measure of how much that joint is recruited. Unlike static posture measures (which capture only one moment), or mean posture measures (which can cancel out opposing motions), the range captures the full dynamic involvement of each joint.

### Step 4: Task-Level Aggregation

Multiple trials of the same task (across all participants and repetitions) are aggregated by taking the **per-joint mean** across all trials belonging to that task:

$$X_{ij} = \frac{1}{N_i} \sum_{t \in \text{trials of task } i} r_j^{(t)}$$

where $N_i$ is the number of trials available for task $i$.

This yields a **41 × 15 data matrix** where each row corresponds to one manipulation task and each column to one joint. Each cell represents the average angular range (in degrees) with which that joint is recruited during that task.

No additional normalization (e.g., z-score or min-max scaling) is applied, preserving the natural scale differences between joints. This allows joints with intrinsically larger ranges (e.g., MCP joints) to contribute proportionally to the analysis.

### Step 5: Dimensionality Reduction via PCA

Principal Component Analysis (PCA) is performed on this 41 × 15 matrix. The covariance-based PCA preserves the natural variance structure since all variables are in the same physical units (degrees). The first 10 components are retained for analysis.

The mean pose vector (mean of all 41 task vectors) is stored alongside the component loadings for reconstruction purposes.

## 5. Results

### Variance Explained

The first two principal components account for **90.9% of the total variance** in the dataset:

| Component | Variance Explained | Cumulative |
|-----------|:-----------------:|:----------:|
| PC1       | 76.95%            | 76.95%     |
| PC2       | 13.97%            | 90.92%     |
| PC3       | 3.43%             | 94.35%     |
| PC4       | 2.17%             | 96.52%     |
| PC5       | 1.39%             | 97.91%     |

The dominant contribution of PC1 and PC2 confirms that hand grasping kinematics can be effectively described by a two-dimensional synergy space.

### Joint Loading Patterns

**PC1** shows **positive loadings across all 15 joints**, indicating that it represents a coordinated activation pattern where all joints move together. Joints with the highest loadings include the MCP and PIP joints of the middle and ring fingers, as well as the thumb opposition joint. This component captures the overall intensity of the grasp—stronger grasps drive all joints toward larger excursions.

**PC2** reveals a **gradient pattern along the hand's mediolateral axis**:
- **Thumb and index finger** joints load positively (especially thumb opposition and abduction)
- **Middle finger** joints show near-zero loadings
- **Ring and little finger** joints load negatively (especially PIP joints)

This component captures the contrast between thumb-index driven precision actions and ulnar-side (ring-little finger) power stabilization. Tasks that rely more on the radial side of the hand (e.g., precision grips, tripod grasps) are separated from those that engage the full hand (e.g., power grasps on large objects).

### Interpretation

The two-component model captures two fundamental axes of hand grasping:

1. **PC1 — Overall Grasp Intensity**: The coordinated recruitment of all joints, distinguishing light-touch manipulations from forceful grasps.
2. **PC2 — Radial–Ulnar Gradient**: The balance between thumb-index engagement and ring-little finger engagement, distinguishing precision-oriented tasks from power-oriented tasks requiring full-palmar contact.

Together, these two dimensions span a functional space that naturally organizes the 41 grasping tasks according to their biomechanical demands.

## 6. Storage

The PCA model is stored as a NumPy `.npz` file containing:
- `components`: (10 × 15) matrix of PC loadings
- `mean`: (15,) mean joint angle range vector
- `explained_variance`: (10,) variance ratios
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
