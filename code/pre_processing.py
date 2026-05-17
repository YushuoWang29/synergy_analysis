import os
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import glob

# --- 参数配置 ---
DATA_ROOT = 'Ninapro Data'  # 原始数据所在根目录
TARGET_FRAMES = 100         # 每段动作重采样后的固定帧数
VELOCITY_THRESHOLD = 0.05   # 梯度模长的阈值（用于静态筛除，需根据实际数据缩放调整）

def preprocess_files():
    # 1. 建立子文件夹 E1, E2, E3
    subfolders = ['E1', 'E2', 'E3']
    for sub in subfolders:
        os.makedirs(os.path.join(DATA_ROOT, sub), exist_ok=True)

    # 2. 遍历所有 .mat 文件 (假设文件可能在子目录中)
    mat_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.mat'), recursive=True)

    for file_path in mat_files:
        file_name = os.path.basename(file_path)
        
        # 排除已经是处理过的文件
        if '_B.mat' in file_name:
            continue

        # 判断数据归属 (根据文件名中的 E1/E2/E3)
        target_sub = None
        for sub in subfolders:
            if sub in file_name:
                target_sub = sub
                break
        
        if not target_sub:
            continue

        print(f"正在处理: {file_name} -> {target_sub}")

        # 3. 加载数据
        try:
            mat = scipy.io.loadmat(file_path)
            # 确保包含必要的 key
            if 'angles' not in mat or 'restimulus' not in mat:
                print(f"跳过: {file_name} (缺少 angles 或 restimulus)")
                continue
            
            angles = np.nan_to_num(mat['angles'])
            restimulus = mat['restimulus'].flatten()
            
            # --- 针对 E2 组的特殊处理：将 ≥9 的标签视为 0（忽略）---
            if target_sub == 'E2':
                restimulus = restimulus.copy()
                restimulus[restimulus >= 9] = 0
                
        except Exception as e:
            print(f"读取失败 {file_name}: {e}")
            continue

        # 4. 利用 restimulus 切分 angles (只保留非0部分)
        processed_data = []
        
        # 寻找 restimulus 非 0 的起止索引
        diff = np.diff(restimulus, prepend=0)
        starts = np.where(diff > 0)[0]
        ends = np.where(diff < 0)[0]
        
        # 如果长度不匹配，补齐末尾
        if len(starts) > len(ends):
            ends = np.append(ends, len(restimulus))

        for s, e in zip(starts, ends):
            trial = angles[s:e, :]  # 截取一段动作
            
            if len(trial) < 5:      # 忽略极短的干扰段
                continue

            # 5a. 静态筛除：计算 22 维梯度的模长
            grads = np.gradient(trial, axis=0)
            vel_norm = np.linalg.norm(grads, axis=1)
            
            # 保留模长大于阈值的帧（即处于运动状态的帧）
            moving_frames = trial[vel_norm > VELOCITY_THRESHOLD]
            
            # 如果筛除后剩太少，回退到原始片段防止插值失败
            if len(moving_frames) < 10:
                moving_frames = trial
                print(f"{file_name}回退")

            # 5b. 重采样：线性插值保留固定 100 帧
            x_old = np.linspace(0, 1, len(moving_frames))
            x_new = np.linspace(0, 1, TARGET_FRAMES)
            f_interp = interp1d(x_old, moving_frames, axis=0, kind='linear')
            resampled_trial = f_interp(x_new)
            
            processed_data.append(resampled_trial)

        # 6. 保存数据
        if processed_data:
            # 组合成一个新的矩阵 (动作段数量 * 100, 22)
            final_matrix = np.vstack(processed_data)
            
            # 构造新文件名 Sxx_Ex_B.mat
            parts = file_name.split('_')
            new_name = f"{parts[0]}_{parts[1]}_B.mat"
            save_path = os.path.join(DATA_ROOT, target_sub, new_name)
            
            scipy.io.savemat(save_path, {
                'angles': final_matrix,
                'num_trials': len(processed_data),
                'info': 'Processed: non-zero restimulus (E2: exclude labels >=9), static removed, resampled to 100 steps'
            })
            print(f"完成保存: {save_path}")

if __name__ == "__main__":
    preprocess_files()