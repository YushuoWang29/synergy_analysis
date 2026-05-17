import os
import numpy as np
import scipy.io
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import itertools

# --- 配置 ---
DATA_ROOT = 'Ninapro Data'
SAVE_PATH = 'PCA_results'
os.makedirs(SAVE_PATH, exist_ok=True)

def perform_pca_analysis(FOLDERS_TO_PROCESS, suffix):
    all_data_list = []
    wrist_data_list = []

    for sub in FOLDERS_TO_PROCESS:
        folder_path = os.path.join(DATA_ROOT, sub)
        files = glob.glob(os.path.join(folder_path, '*_B.mat'))
        
        for f in files:
            mat = scipy.io.loadmat(f)
            angles = mat['angles']
            all_data_list.append(angles[:, :20])
            wrist_data_list.append(angles[:, 20:22])

    if not all_data_list:
        print("未找到预处理后的数据文件。")
        return

    full_matrix = np.vstack(all_data_list)
    print(f"数据整合完成，形状为: {full_matrix.shape}")

    n_components = min(full_matrix.shape[1], 50) 
    pca = PCA(n_components=n_components)
    pca.fit(full_matrix)

    np.savez(os.path.join(SAVE_PATH, f'pca_model_{suffix}.npz'),
             components=pca.components_,
             mean=pca.mean_,
             explained_variance=pca.explained_variance_ratio_,
             fixed_wrist_mean=np.mean(np.vstack(wrist_data_list), axis=0)
    )

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual PC Variance')
    xticks = np.arange(2, len(pca.explained_variance_ratio_) + 1, 2)
    plt.xticks(xticks)

    plt.subplot(1, 2, 2)
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance')
    plt.xticks(xticks)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'variance_plot_{suffix}.png'))
    plt.close()
    
    print(f"PCA 完成。结果已保存至 {SAVE_PATH}")

if __name__ == "__main__":
    all_sets = ['E1', 'E2', 'E3']
    for r in range(1, 4):
        for combo in itertools.combinations(all_sets, r):
            folders = list(combo)
            suffix = ''.join([s[-1] for s in folders])
            perform_pca_analysis(folders, suffix)