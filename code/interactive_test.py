import mujoco
import mujoco_viewer
import numpy as np
import tkinter as tk
from tkinter import ttk

# ================== 配置 ==================
MODEL_PATH = "model/mujoco_menagerie/shadow_hand/scene_right.xml"
# ==========================================

# Ninapro 22个关节的名称（按您指定的顺序）
ninapro_joint_names = [
    "CMC1_f", "CMC1_a", "MCP1", "IP1",
    "MCP2_f", "MCP2_a", "PIP2", "MCP3_f", "PIP3",
    "MCP4_f", "MCP4_a", "PIP4", "CMC5", "MCP5_f", "MCP5_a", "PIP5",
    "DIP2", "DIP3", "DIP4", "DIP5",
    "WRIST_F", "WRIST_A"
]

# Ninapro 列索引 (0-based) 对应的 MuJoCo qpos 索引（基于之前打印的顺序）
mapping_ninapro = {
    0: 20,   # CMC1_f -> rh_THJ5
    1: 19,   # CMC1_a -> rh_THJ4
    2: 22,   # MCP1   -> rh_THJ2
    3: 23,   # IP1    -> rh_THJ1
    4: 3,    # MCP2_f -> rh_FFJ3
    5: 2,    # MCP2_a -> rh_FFJ4
    6: 4,    # PIP2   -> rh_FFJ2
    7: 7,    # MCP3_f -> rh_MFJ3
    8: 8,    # PIP3   -> rh_MFJ2
    9: 11,   # MCP4_f -> rh_RFJ3
    10: 10,  # MCP4_a -> rh_RFJ4
    11: 12,  # PIP4   -> rh_RFJ2
    12: 14,  # CMC5   -> rh_LFJ5
    13: 16,  # MCP5_f -> rh_LFJ3
    14: 15,  # MCP5_a -> rh_LFJ4
    15: 17,  # PIP5   -> rh_LFJ2
    16: 5,   # DIP2   -> rh_FFJ1
    17: 9,   # DIP3   -> rh_MFJ1
    18: 13,  # DIP4   -> rh_RFJ1
    19: 18,  # DIP5   -> rh_LFJ1
    20: 1,   # WRIST_F -> rh_WRJ1
    21: 0    # WRIST_A -> rh_WRJ2
}

# 符号修正字典（您之前调试的版本，可根据需要在此调整）
sign_correction = {
    0: 1,    # rh_WRJ2
    1: 1,    # rh_WRJ1
    2: 1,    # rh_FFJ4 (MCP2_a)
    3: 1,    # rh_FFJ3 (MCP2_f)
    4: 1,    # rh_FFJ2 (PIP2)
    5: 1,    # rh_FFJ1 (DIP2)
    6: 1,    # rh_MFJ4
    7: 1,    # rh_MFJ3 (MCP3_f)
    8: 1,    # rh_MFJ2 (PIP3)
    9: -1,   # rh_MFJ1 (DIP3)   # 远端指间反向
    10: 1,   # rh_RFJ4 (MCP4_a)
    11: 1,   # rh_RFJ3 (MCP4_f)
    12: 1,   # rh_RFJ2 (PIP4)
    13: -1,  # rh_RFJ1 (DIP4)   # 远端指间反向
    14: 1,   # rh_LFJ5 (CMC5)
    15: 1,   # rh_LFJ4 (MCP5_a)
    16: 1,   # rh_LFJ3 (MCP5_f)
    17: 1,   # rh_LFJ2 (PIP5)
    18: -1,  # rh_LFJ1 (DIP5)   # 远端指间反向
    19: 1,   # rh_THJ5
    20: 1,   # rh_THJ4 (CMC1_f)
    21: 1,   # rh_THJ3 (CMC1_a)
    22: 1,   # rh_THJ2 (MCP1)
    23: 1    # rh_THJ1 (IP1)
}

# 加载模型与数据
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
data.qpos[:] = 0.0

# 为每个 Ninapro 关节准备显示信息
class JointInfo:
    def __init__(self, name, mj_idx, sign):
        self.name = name
        self.mj_idx = mj_idx
        self.sign = sign
        self.low_rad, self.high_rad = model.jnt_range[mj_idx] if mj_idx < model.njnt else (-np.pi, np.pi)
        self.low_deg = np.rad2deg(self.low_rad)
        self.high_deg = np.rad2deg(self.high_rad)

# 按 Ninapro 顺序生成信息列表
joint_infos = []
for i in range(22):
    mj_idx = mapping_ninapro[i]
    sign = sign_correction[mj_idx]
    joint_infos.append(JointInfo(ninapro_joint_names[i], mj_idx, sign))

# 启动 MuJoCo 查看器
viewer = mujoco_viewer.MujocoViewer(model, data)

# ================== GUI 部分 ==================
class NinaproControlGUI:
    def __init__(self, root):
        self.root = root
        root.title("Ninapro 22 关节控制 (单位: 度) — 符号修正已应用")
        root.geometry("850x700")
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.sliders = []
        self.entry_vars = []
        self.entry_widgets = []

        # 带滚动条的画布
        canvas = tk.Canvas(root, borderwidth=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 表头
        ttk.Label(scrollable_frame, text="关节 (Ninapro)", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(scrollable_frame, text="滑块", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(scrollable_frame, text="角度值 (°)", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=2)

        # 为每个关节创建控件
        for i, info in enumerate(joint_infos):
            # 标签：显示 Ninapro 名称
            label = ttk.Label(scrollable_frame, text=info.name)
            label.grid(row=i+1, column=0, padx=5, pady=2, sticky="w")

            # 变量
            var = tk.DoubleVar(value=0.0)
            self.entry_vars.append(var)

            # 滑块
            slider = ttk.Scale(
                scrollable_frame,
                from_=info.low_deg, to=info.high_deg,
                orient="horizontal",
                length=400,
                variable=var,
                command=lambda val, idx=i: self.on_slider_change(idx, val)
            )
            slider.grid(row=i+1, column=1, padx=5, pady=2)
            self.sliders.append(slider)

            # 输入框
            entry = ttk.Entry(scrollable_frame, textvariable=var, width=10)
            entry.grid(row=i+1, column=2, padx=5, pady=2)
            entry.bind("<Return>", lambda e, idx=i: self.on_entry_change(idx))
            entry.bind("<FocusOut>", lambda e, idx=i: self.on_entry_change(idx))
            self.entry_widgets.append(entry)

            # 限位提示
            range_text = f"[{info.low_deg:.1f}, {info.high_deg:.1f}]"
            ttk.Label(scrollable_frame, text=range_text).grid(row=i+1, column=3, padx=5, pady=2)

        # 按钮
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.grid(row=23, column=0, columnspan=4, pady=10)
        ttk.Button(btn_frame, text="全部归零", command=self.reset_all).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="退出", command=self.on_closing).pack(side="left", padx=5)

        self.running = True
        self.update_render()

    def on_slider_change(self, idx, val):
        """滑块拖动：输入角度(度) -> 应用符号 -> 弧度 -> qpos"""
        deg = float(val)
        info = joint_infos[idx]
        # 钳位
        deg = max(info.low_deg, min(info.high_deg, deg))
        # 应用符号修正：界面上的正值对应握拳方向，内部乘以符号
        rad = np.deg2rad(deg) * info.sign
        data.qpos[info.mj_idx] = rad
        self.entry_vars[idx].set(deg)

    def on_entry_change(self, idx):
        """输入框修改"""
        try:
            deg = self.entry_vars[idx].get()
            info = joint_infos[idx]
            deg = max(info.low_deg, min(info.high_deg, deg))
            rad = np.deg2rad(deg) * info.sign
            data.qpos[info.mj_idx] = rad
            self.entry_vars[idx].set(deg)
        except tk.TclError:
            pass

    def reset_all(self):
        data.qpos[:] = 0.0
        for var in self.entry_vars:
            var.set(0.0)

    def update_render(self):
        if self.running and viewer.is_alive:
            mujoco.mj_forward(model, data)
            viewer.render()
            self.root.after(10, self.update_render)

    def on_closing(self):
        self.running = False
        viewer.close()
        self.root.destroy()

# ================== 启动 ==================
if __name__ == "__main__":
    root = tk.Tk()
    gui = NinaproControlGUI(root)
    root.mainloop()