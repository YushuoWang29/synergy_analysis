import mujoco
import mujoco_viewer
import numpy as np
import scipy.io
import time
import sys

# ================== 配置区域 ==================
DATASET_TYPE = "ninapro"  # 可选 "ninapro" 或 "myoki"

# 文件路径（请根据实际情况修改）
MODEL_PATH = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'
NINAPRO_PATH = 'Ninapro Data/E2/S5_E2_B.mat'
MYOKI_PATH = 'MyoKi Data/P26.mat'

# 播放帧数限制（None 表示全部播放，数字表示播放前 N 帧）
MAX_FRAMES = None  # 例如 20000
# =============================================

# 加载模型
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("=== MuJoCo Joint qpos Mapping ===")
for i in range(model.nq):
    if i < model.njnt:
        name = model.joint(i).name
    else:
        name = "free joint"
    print(f"qpos[{i}] : {name}")

# 预提取所有关节的限位（弧度），自由关节为 None
joint_ranges_rad = []
for i in range(model.nq):
    if i < model.njnt:
        joint_ranges_rad.append(model.jnt_range[i])  # (low, high) in radians
    else:
        joint_ranges_rad.append(None)

# 根据数据集类型加载数据并建立映射
if DATASET_TYPE.lower() == "ninapro":
    mat_path = NINAPRO_PATH
    mat = scipy.io.loadmat(mat_path)
    angles = mat['angles']  # 假设变量名为 'angles'
    angles = np.nan_to_num(angles)
    angles_rad = np.deg2rad(angles)

    # Ninapro 22维 -> MuJoCo qpos 索引
    mapping = {
        0: 20, 1: 19, 2: 22, 3: 23,          # 拇指
        4: 3,  5: 2,  6: 4,  16: 5,          # 食指
        7: 7,  8: 8,  17: 9,                 # 中指
        9: 11, 10: 10, 11: 12, 18: 13,       # 无名指
        12: 14, 13: 16, 14: 15, 15: 17, 19: 18,  # 小指
        20: 1, 21: 0                          # 手腕
    }
    # 符号修正
    sign_correction = {
        0: 0, 1: 0,        # 手腕暂时禁用（如需启用，改为 1 或 -1）
        2: 1, 3: 1, 4: 1, 5: 1,
        7: 1, 8: 1, 9: -1,
        10: 1, 11: 1, 12: 1, 13: -1,
        14: 1, 15: 1, 16: 1, 17: 1, 18: -1,
        19: 1, 20: 1, 21: 1, 22: 1, 23: 1
    }

elif DATASET_TYPE.lower() == "myoki":
    mat_path = MYOKI_PATH
    mat = scipy.io.loadmat(mat_path)
    angles = mat['glove_calibrated']
    angles = np.nan_to_num(angles)
    angles_rad = np.deg2rad(angles)

    # MyoKi 18维映射
    mapping = {
        0: 19, 1: 20, 2: 23, 3: 21,
        4: 3,  5: 4,  8: 2,
        6: 8,  7: 9,
        9: 12, 10: 13, 11: 11,
        12: 17, 13: 18, 14: 16,
    }
    sign_correction = {idx: 1 for idx in range(24)}

else:
    print("未知数据集类型，请设置为 'ninapro' 或 'myoki'")
    sys.exit(1)

# 限制播放帧数
if MAX_FRAMES is not None:
    angles_rad = angles_rad[:MAX_FRAMES]

# ================== 可视化 ==================
viewer = mujoco_viewer.MujocoViewer(model, data)

print(f"开始播放 {DATASET_TYPE} 数据集动作...")
for i in range(len(angles_rad)):
    if not viewer.is_alive:
        break

    data.qpos[:] = 0.0
    current_frame = angles_rad[i]

    for src_idx, mj_idx in mapping.items():
        if src_idx < current_frame.shape[0]:
            # 应用符号修正
            signed_angle = current_frame[src_idx] * sign_correction.get(mj_idx, 1)

            # 关节限位钳位
            range_rad = joint_ranges_rad[mj_idx]
            if range_rad is not None:
                low, high = range_rad
                if signed_angle < low:
                    signed_angle = low
                elif signed_angle > high:
                    signed_angle = high

            data.qpos[mj_idx] = signed_angle

    mujoco.mj_forward(model, data)
    viewer.render()
    time.sleep(0.01)

viewer.close()
print("播放结束")