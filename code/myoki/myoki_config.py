"""
MyoKi dataset configuration: joint definitions, mapping to ShadowHand, and sign corrections.

NOTE (2026-05-19): Palmar Arch (sensor 16) has been REMOVED because it is not a true
finger DOF. CyberGlove measures the transverse palmar arch which is related but not
equivalent to the little finger CMC joint. ShadowHand LFJ5 is now left at 0.
We now work with 15 hand joint angles (down from 16).

MyoKi uses an 18-sensor CyberGlove II; we retain only the 15 hand‑related channels
(channels 15=palmar arch, 16=wrist flex/ext, 17=wrist rad/ulnar are excluded).

  Joint# | Sensor Description
  -------|-------------------
  0      | Abduction Thumb to Palm
  1      | MCP Thumb
  2      | IP Thumb
  3      | Abduction Thumb-Index
  4      | MCP Index
  5      | PIP Index
  6      | MCP Middle
  7      | PIP Middle
  8      | Abduction Index-Middle
  9      | MCP Ring
  10     | PIP Ring
  11     | Abduction Middle-Ring
  12     | MCP Little
  13     | PIP Little
  14     | Abduction Ring-Little
  [palmar arch, wrist channels excluded]

ShadowHand (right hand) qpos indices (24 joints):
  0: rh_WRJ2  - Wrist Abduction (radial/ulnar)
  1: rh_WRJ1  - Wrist Flexion (flex/ext)
  2: rh_FFJ4  - Index Abduction
  3: rh_FFJ3  - Index MCP Flexion
  4: rh_FFJ2  - Index PIP Flexion
  5: rh_FFJ1  - Index DIP Flexion
  6: rh_MFJ4  - Middle Abduction
  7: rh_MFJ3  - Middle MCP Flexion
  8: rh_MFJ2  - Middle PIP Flexion
  9: rh_MFJ1  - Middle DIP Flexion
  10: rh_RFJ4 - Ring Abduction
  11: rh_RFJ3 - Ring MCP Flexion
  12: rh_RFJ2 - Ring PIP Flexion
  13: rh_RFJ1 - Ring DIP Flexion
  14: rh_LFJ5 - Little CMC Flexion (NOT DRIVEN - set to 0)
  15: rh_LFJ4 - Little Abduction
  16: rh_LFJ3 - Little MCP Flexion
  17: rh_LFJ2 - Little PIP Flexion
  18: rh_LFJ1 - Little DIP Flexion
  19: rh_THJ5 - Thumb Rotation (opposition)
  20: rh_THJ4 - Thumb MCP Flexion
  21: rh_THJ3 - Thumb Abduction
  22: rh_THJ2 - Thumb IP Flexion
  23: rh_THJ1 - Thumb Distal Flexion
"""

import numpy as np

# MyoKi 15 hand-only joint names (0-indexed, palmar arch + wrist excluded)
MYOKI_JOINT_NAMES = [
    "Thumb Abduction",        # 0:  sensor 1
    "Thumb MCP",              # 1:  sensor 2
    "Thumb IP",               # 2:  sensor 3
    "Thumb Opp/Abduction",    # 3:  sensor 4 (Abduction Thumb-Index)
    "Index MCP",              # 4:  sensor 5
    "Index PIP",              # 5:  sensor 6
    "Middle MCP",             # 6:  sensor 7
    "Middle PIP",             # 7:  sensor 8
    "Index-Middle Abd",       # 8:  sensor 9
    "Ring MCP",               # 9:  sensor 10
    "Ring PIP",               # 10: sensor 11
    "Middle-Ring Abd",        # 11: sensor 12
    "Little MCP",             # 12: sensor 13
    "Little PIP",             # 13: sensor 14
    "Ring-Little Abd",        # 14: sensor 15
]

MYOKI_N_JOINTS = len(MYOKI_JOINT_NAMES)  # 15

# ============================================================
# MyoKi 15-ch (hand only, no palmar arch) → ShadowHand 24-qpos Mapping
# ============================================================
# Key: MyoKi index (0-14), Value: ShadowHand qpos index (0-23)
# Wrist joints (SH indices 0,1) stay at 0.
# Little CMC (SH index 14) stays at 0 — not directly measurable by CyberGlove.
MYOKI_TO_SHADOW = {
    # --- Thumb (5 ShadowHand dofs) ---
    0: 21,   # MyoKi 0 (Thumb Abduction)     → THJ3 (Thumb Abduction)
    1: 20,   # MyoKi 1 (Thumb MCP)           → THJ4 (Thumb MCP Flexion)
    2: 22,   # MyoKi 2 (Thumb IP)            → THJ2 (Thumb IP Flexion)
    3: 19,   # MyoKi 3 (Thumb Opp/Abd)       → THJ5 (Thumb Rotation)
    # --- Index (4 ShadowHand dofs, but MyoKi has 2) ---
    4: 3,    # MyoKi 4 (Index MCP)           → FFJ3 (Index MCP)
    5: 4,    # MyoKi 5 (Index PIP)           → FFJ2 (Index PIP)
    # --- Middle (4 ShadowHand dofs, but MyoKi has 3 incl. abd) ---
    6: 7,    # MyoKi 6 (Middle MCP)          → MFJ3 (Middle MCP)
    7: 8,    # MyoKi 7 (Middle PIP)          → MFJ2 (Middle PIP)
    8: 6,    # MyoKi 8 (Index-Middle Abd)    → MFJ4 (Middle Abduction)
    # --- Ring (4 ShadowHand dofs, but MyoKi has 3 incl. abd) ---
    9: 11,   # MyoKi 9 (Ring MCP)            → RFJ3 (Ring MCP)
    10: 12,  # MyoKi 10 (Ring PIP)           → RFJ2 (Ring PIP)
    11: 10,  # MyoKi 11 (Middle-Ring Abd)    → RFJ4 (Ring Abduction)
    # --- Little (5 ShadowHand dofs, but MyoKi has 3 incl. abd) ---
    12: 16,  # MyoKi 12 (Little MCP)         → LFJ3 (Little MCP)
    13: 17,  # MyoKi 13 (Little PIP)         → LFJ2 (Little PIP)
    14: 15,  # MyoKi 14 (Ring-Little Abd)    → LFJ4 (Little Abduction)
    # --- NOTE: Little CMC (LFJ5, qpos 14) is NOT driven ---
}

# ============================================================
# DIP joints: CyberGlove has no DIP sensors, copy from PIP
# ============================================================
DIP_FROM_PIP_MAPPING = [
    (5,  5),   # FFJ1 (Index DIP)   ← MyoKi 5 (Index PIP)
    (9,  7),   # MFJ1 (Middle DIP)  ← MyoKi 7 (Middle PIP)
    (13, 10),  # RFJ1 (Ring DIP)    ← MyoKi 10 (Ring PIP)
    (18, 13),  # LFJ1 (Little DIP)  ← MyoKi 13 (Little PIP)
    (23, 2),   # THJ1 (Thumb Dist)  ← MyoKi 2 (Thumb IP)
]

# ============================================================
# Sign corrections (all 1 for now, adjust if needed)
# ============================================================
SIGN_CORRECTION = {i: 1 for i in range(24)}

# ============================================================
# ShadowHand joint names (for reference)
# ============================================================
SHADOW_JOINT_NAMES = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]

# ============================================================
# Helper: convert MyoKi 15-ch frame → ShadowHand 24-dof qpos
# ============================================================
def myoki_to_qpos(myoki_frame_deg, joint_ranges_rad=None):
    """
    Convert a MyoKi 15-channel frame (hand only, palmar arch + wrist excluded) to
    ShadowHand 24-dim qpos (in radians). Wrist joints stay at 0.
    Little CMC (LFJ5) stays at 0 — not directly measured by CyberGlove.

    Parameters
    ----------
    myoki_frame_deg : ndarray, shape (15,) or (N,15), in degrees
    joint_ranges_rad : list of (low, high) or None, length 24

    Returns
    -------
    qpos : ndarray, shape (24,) or (N,24), in radians
    """
    single_frame = myoki_frame_deg.ndim == 1
    if single_frame:
        myoki_frame_deg = myoki_frame_deg.reshape(1, -1)

    if myoki_frame_deg.shape[1] > MYOKI_N_JOINTS:
        # Slice off wrist + palmar arch channels if still 18-D or 16-D
        myoki_frame_deg = myoki_frame_deg[:, :MYOKI_N_JOINTS]

    n_frames = myoki_frame_deg.shape[0]
    qpos = np.zeros((n_frames, 24), dtype=np.float64)

    for myo_idx, sh_idx in MYOKI_TO_SHADOW.items():
        qpos[:, sh_idx] = np.deg2rad(myoki_frame_deg[:, myo_idx]) * SIGN_CORRECTION[sh_idx]

    for sh_idx, src_myo_idx in DIP_FROM_PIP_MAPPING:
        qpos[:, sh_idx] = np.deg2rad(myoki_frame_deg[:, src_myo_idx]) * SIGN_CORRECTION.get(sh_idx, 1)

    if joint_ranges_rad is not None:
        for sh_idx in range(24):
            if joint_ranges_rad[sh_idx] is not None:
                low, high = joint_ranges_rad[sh_idx]
                qpos[:, sh_idx] = np.clip(qpos[:, sh_idx], low, high)

    return qpos[0] if single_frame else qpos


def get_myoki_joint_order_string():
    """Return a human-readable description of the mapping."""
    lines = ["MyoKi (15-ch hand, no palmar arch) → ShadowHand (24-dof) Mapping:\n"]
    lines.append(f"{'MyoKi #':<8} {'MyoKi Joint':<25} {'→ SH qpos':<10} {'ShadowHand Joint'}")
    lines.append("-" * 70)
    for myo_idx, sh_idx in sorted(MYOKI_TO_SHADOW.items(), key=lambda x: x[1]):
        lines.append(f"{myo_idx:<8} {MYOKI_JOINT_NAMES[myo_idx]:<25} → {sh_idx:<8} {SHADOW_JOINT_NAMES[sh_idx]}")
    lines.append("\nNOTE: Little CMC (rh_LFJ5, qpos 14) is NOT driven (set to 0).")
    lines.append("Wrist joints (rh_WRJ1, rh_WRJ2) are also NOT driven.")
    lines.append("\nDIP joints copied from PIP:")
    for sh_idx, src_myo_idx in DIP_FROM_PIP_MAPPING:
        lines.append(f"  {SHADOW_JOINT_NAMES[sh_idx]} ← {MYOKI_JOINT_NAMES[src_myo_idx]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_myoki_joint_order_string())
