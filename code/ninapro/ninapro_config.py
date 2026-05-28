"""
Ninapro dataset configuration: joint definitions, mapping to ShadowHand.

Ninapro uses a 22-sensor CyberGlove II. The 22 sensors are:
(From order_of_angles field in original mat files)

 Joint# | Name       | Description
 -------|------------|---------------------------
  0     | CMC1_f     | Thumb CMC flexion/extension (ab/adduction in some conventions)
  1     | CMC1_a     | Thumb CMC abduction/adduction (opposition)
  2     | MCP1       | Thumb MCP flexion/extension
  3     | IP1        | Thumb IP flexion/extension
  4     | MCP2_f     | Index MCP flexion/extension
  5     | MCP2_a     | Index MCP abduction/adduction
  6     | PIP2       | Index PIP flexion/extension
  7     | MCP3_f     | Middle MCP flexion/extension
  8     | PIP3       | Middle PIP flexion/extension
  9     | MCP4_f     | Ring MCP flexion/extension
 10     | MCP4_a     | Ring MCP abduction/adduction
 11     | PIP4       | Ring PIP flexion/extension
 12     | CMC5       | Little CMC flexion/extension
 13     | MCP5_f     | Little MCP flexion/extension
 14     | MCP5_a     | Little MCP abduction/adduction
 15     | PIP5       | Little PIP flexion/extension
 16     | DIP2       | Index DIP flexion/extension
 17     | DIP3       | Middle DIP flexion/extension
 18     | DIP4       | Ring DIP flexion/extension
 19     | DIP5       | Little DIP flexion/extension
 20     | WRIST_F    | Wrist flexion/extension
 21     | WRIST_A    | Wrist abduction/adduction (radial/ulnar)

We use all 22 channels. DIPs are directly measured (unlike MyoKi which lacks them).
Wrist channels included for completeness (can be excluded for hand-only synergy).

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
 10: rh_RFJ4  - Ring Abduction
 11: rh_RFJ3  - Ring MCP Flexion
 12: rh_RFJ2  - Ring PIP Flexion
 13: rh_RFJ1  - Ring DIP Flexion
 14: rh_LFJ5  - Little CMC Flexion (NOT DRIVEN - set to 0)
 15: rh_LFJ4  - Little Abduction
 16: rh_LFJ3  - Little MCP Flexion
 17: rh_LFJ2  - Little PIP Flexion
 18: rh_LFJ1  - Little DIP Flexion
 19: rh_THJ5  - Thumb Rotation (opposition)
 20: rh_THJ4  - Thumb MCP Flexion
 21: rh_THJ3  - Thumb Abduction
 22: rh_THJ2  - Thumb IP Flexion
 23: rh_THJ1  - Thumb Distal Flexion
"""

import numpy as np

NINAPRO_N_JOINTS = 22  # All 22 CyberGlove sensors

NINAPRO_JOINT_NAMES = [
    "Thumb CMC flex/ext",      # 0
    "Thumb CMC abd/add",       # 1
    "Thumb MCP flex/ext",      # 2
    "Thumb IP flex/ext",       # 3
    "Index MCP flex/ext",      # 4
    "Index MCP abd/add",       # 5
    "Index PIP flex/ext",      # 6
    "Middle MCP flex/ext",     # 7
    "Middle PIP flex/ext",     # 8
    "Ring MCP flex/ext",       # 9
    "Ring MCP abd/add",        # 10
    "Ring PIP flex/ext",       # 11
    "Little CMC flex/ext",     # 12
    "Little MCP flex/ext",     # 13
    "Little MCP abd/add",      # 14
    "Little PIP flex/ext",     # 15
    "Index DIP flex/ext",      # 16
    "Middle DIP flex/ext",     # 17
    "Ring DIP flex/ext",       # 18
    "Little DIP flex/ext",     # 19
    "Wrist flex/ext",          # 20
    "Wrist abd/add",           # 21
]

# For hand-only PCA (exclude wrist joints 20, 21)
HAND_JOINT_INDICES = list(range(20))  # 0-19
N_HAND_JOINTS = 20

# ============================================================
# Ninapro (22-ch, raw CyberGlove) → ShadowHand (24-qpos) mapping
# ============================================================
# We map from Ninapro joint index (0-21) to ShadowHand qpos index (0-23).
# Wrist: Ninapro 20→SH0 (WRJ2), 21→SH1 (WRJ1)
# Little CMC (SH14) stays at 0 — not independently controlled in ShadowHand.

NINAPRO_TO_SHADOW = {
    # Thumb
    0: 19,   # CMC1_f → THJ5 (Thumb Rotation/Opposition)
    1: 21,   # CMC1_a → THJ3 (Thumb Abduction)
    2: 20,   # MCP1   → THJ4 (Thumb MCP Flexion)
    3: 22,   # IP1    → THJ2 (Thumb IP Flexion)
    # Index
    4: 3,    # MCP2_f → FFJ3 (Index MCP)
    5: 2,    # MCP2_a → FFJ4 (Index Abduction)
    6: 4,    # PIP2   → FFJ2 (Index PIP)
    16: 5,   # DIP2   → FFJ1 (Index DIP)
    # Middle
    7: 7,    # MCP3_f → MFJ3 (Middle MCP)
    8: 8,    # PIP3   → MFJ2 (Middle PIP)
    17: 9,   # DIP3   → MFJ1 (Middle DIP)
    # Ring
    9: 11,   # MCP4_f → RFJ3 (Ring MCP)
    10: 10,  # MCP4_a → RFJ4 (Ring Abduction)
    11: 12,  # PIP4   → RFJ2 (Ring PIP)
    18: 13,  # DIP4   → RFJ1 (Ring DIP)
    # Little
    12: 15,  # CMC5   → LFJ4 (Little Abduction) — no CMC in shadow hand
    13: 16,  # MCP5_f → LFJ3 (Little MCP)
    14: 14,  # MCP5_a → LFJ5 (Little CMC) — map ab/ad to CMC
    15: 17,  # PIP5   → LFJ2 (Little PIP)
    19: 18,  # DIP5   → LFJ1 (Little DIP)
    # Wrist
    20: 1,   # WRIST_F → WRJ1 (Wrist Flexion)
    21: 0,   # WRIST_A → WRJ2 (Wrist Abduction)
}

# Thumb distal joint (THJ1, qpos 23): copy from IP (Ninapro index 3)
# Use the same mapping pattern as the DIP for thumb
NINAPRO_THUMB_DISTAL_SRC = (23, 3)  # THJ1 ← Ninapro 3 (Thumb IP)

# Hand-only subset mapping (20 joints → ShadowHand)
NINAPRO_HAND_TO_SHADOW = {k: v for k, v in NINAPRO_TO_SHADOW.items() if k < 20}
# DIPs are included (16-19), little CMC mapped (14)

# ============================================================
# Hand-only joint names (20 joints, no wrist)
# ============================================================
HAND_ONLY_NAMES = NINAPRO_JOINT_NAMES[:20]

# ============================================================
# Sign corrections (all 1 for now)
# ============================================================
SIGN_CORRECTION = {i: 1 for i in range(24)}

# ShadowHand joint names
SHADOW_JOINT_NAMES = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]

def ninapro_to_qpos(ninapro_frame, hand_only=True):
    """
    Convert Ninapro 22-ch (or 20-ch) frame to ShadowHand 24-dim qpos (radians).
    
    Parameters
    ----------
    ninapro_frame : ndarray, shape (N, 22) or (N, 20) or (22,) or (20,)
        Raw CyberGlove II sensor values (8-bit, integers 0-255 typically)
        If hand_only=True, expects first 20 columns
    hand_only : bool
        If True, expect/pass only hand joints (0-19)
        
    Returns
    -------
    qpos : ndarray, shape (24,) or (N, 24)
    """
    single_frame = ninapro_frame.ndim == 1
    if single_frame:
        ninapro_frame = ninapro_frame.reshape(1, -1)
    
    n_frames = ninapro_frame.shape[0]
    n_cols = ninapro_frame.shape[1]
    
    # If 22 columns and hand_only=True, keep first 20
    if hand_only and n_cols == 22:
        ninapro_frame = ninapro_frame[:, :20]
    
    qpos = np.zeros((n_frames, 24), dtype=np.float64)
    
    mapping = NINAPRO_HAND_TO_SHADOW if hand_only else NINAPRO_TO_SHADOW
    for nin_idx, sh_idx in mapping.items():
        if nin_idx < n_cols:
            qpos[:, sh_idx] = ninapro_frame[:, nin_idx] * SIGN_CORRECTION.get(sh_idx, 1)
    
    # Thumb distal from IP
    if hand_only and 3 < n_cols:
        qpos[:, 23] = ninapro_frame[:, 3]  # THJ1 ← IP1
    
    return qpos[0] if single_frame else qpos


def get_ninapro_joint_order_string():
    lines = ["Ninapro (22-ch CyberGlove) → ShadowHand (24-dof) Mapping:\n"]
    lines.append(f"{'Ninap #':<8} {'Ninapro Joint':<30} {'→ SH':<6} {'ShadowHand Joint'}")
    lines.append("-" * 75)
    for nin_idx, sh_idx in sorted(NINAPRO_TO_SHADOW.items(), key=lambda x: x[1]):
        lines.append(f"{nin_idx:<8} {NINAPRO_JOINT_NAMES[nin_idx]:<30} → {sh_idx:<4} {SHADOW_JOINT_NAMES[sh_idx]}")
    lines.append("\n  Little CMC (rh_LFJ5, qpos 14) kept at 0.")
    lines.append(f"  Thumb distal (THJ1, qpos 23) ← Ninapro 3 (Thumb IP)")
    lines.append("\n  Hand-only PCA uses joints 0-19 (excludes WRIST_F, WRIST_A)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_ninapro_joint_order_string())
