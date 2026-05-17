import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
import time
from matplotlib.widgets import Slider

# --- 1. 加载 PCA 数据 ---
try:
    pca_data = np.load('PCA_results/pca_model_3.npz')
    pc_vectors = pca_data['components']  
    mean_pose = pca_data['mean']
    fixed_wrist = pca_data['fixed_wrist_mean'] 
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# --- 2. 映射与符号修正 ---
mapping = {
    0: 20, 1: 19, 2: 22, 3: 23,
    4: 3,  5: 2,  6: 4,  16: 5,
    7: 7,  8: 8,  17: 9,
    9: 11, 10: 10, 11: 12, 18: 13,
    12: 14, 13: 16, 14: 15, 15: 17, 19: 18,
    20: 1, 21: 0
}
sign_correction = {i: 1 for i in range(24)}
sign_correction.update({9: -1, 13: -1, 18: -1, 1: 0, 0: 0}) 

# --- 3. MuJoCo 初始化 ---
model_path = 'model/mujoco_menagerie/shadow_hand/scene_right.xml'
try:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"MuJoCo 模型加载失败: {e}")
    exit()

# 提取所有关节的限位（弧度），用于后续钳位
joint_ranges_rad = []
for i in range(model.nq):
    if i < model.njnt:
        joint_ranges_rad.append(model.jnt_range[i])  # (low, high) in radians
    else:
        joint_ranges_rad.append(None)

mode_flag = {"mode": "2d"}  # "2d" or "slider"
current_weights = [0.0, 0.0]
slider_weights = [0.0]*5

# --- 4. 交互式 UI ---
def start_ui():
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
        
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    # 扩大坐标系范围到 [-6, 6]
    ax.set_xlim([-6, 6]) 
    ax.set_ylim([-6, 6])
    ax.set_title("Press 'm' to switch mode")
    ax.grid(True)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    
    point, = ax.plot([0], [0], 'ro', markersize=12)

    slider_axes = []
    sliders = []
    # 滑块范围也相应扩大为 -6 到 6
    for i in range(5):
        ax_slider = plt.axes([0.15, 0.02 + i*0.04, 0.7, 0.02])
        slider = Slider(ax_slider, f'PC{i+1}', -6.0, 6.0, valinit=0.0)
        slider_axes.append(ax_slider)
        sliders.append(slider)

    for ax_s in slider_axes:
        ax_s.set_visible(False)

    def update_point(event):
        if mode_flag["mode"] != "2d":
            return
        if event.xdata is not None and event.ydata is not None:
            current_weights[0] = event.xdata
            current_weights[1] = event.ydata
            point.set_data([event.xdata], [event.ydata])
            fig.canvas.draw_idle()

    def update_sliders(val):
        if mode_flag["mode"] != "slider":
            return
        for i in range(5):
            slider_weights[i] = sliders[i].val

    for s in sliders:
        s.on_changed(update_sliders)

    def on_key(event):
        if event.key == 'm':
            if mode_flag["mode"] == "2d":
                mode_flag["mode"] = "slider"
                for ax_s in slider_axes:
                    ax_s.set_visible(True)
            else:
                mode_flag["mode"] = "2d"
                for ax_s in slider_axes:
                    ax_s.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', update_point)
    fig.canvas.mpl_connect('motion_notify_event', lambda ev: update_point(ev) if ev.button == 1 else None)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

# --- 5. 主程序入口 ---
if __name__ == "__main__":
    ui_thread = threading.Thread(target=start_ui, daemon=True)
    ui_thread.start()
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    print("Demo 已启动。按 m 切换控制模式")
    
    try:
        while viewer.is_alive:
            if mode_flag["mode"] == "2d":
                synergy_20d = mean_pose + current_weights[0]*pc_vectors[0]*50 + current_weights[1]*pc_vectors[1]*50
            else:
                synergy_20d = mean_pose.copy()
                for i in range(5):
                    synergy_20d += slider_weights[i]*pc_vectors[i]*50
            
            full_nina_angles = np.concatenate([synergy_20d, fixed_wrist])
            full_nina_rad = np.deg2rad(full_nina_angles)
            
            for nina_idx, mj_idx in mapping.items():
                # 应用符号修正
                signed_angle = full_nina_rad[nina_idx] * sign_correction.get(mj_idx, 1)
                
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
            
    except KeyboardInterrupt:
        pass
    
    viewer.close()