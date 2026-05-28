"""Quick script to inspect PCA data files and extract numbers for the LaTeX document."""
import numpy as np
import os

DATA_DIR = 'PCA_results/MyoKi'

# Load combined PCA results
files_to_check = [
    'pca_myoki_all_grasp_static.npz',
    'pca_myoki_all_grasp_static_pisa_rotated.npz',
    'pca_myoki_all_tasks_static.npz',
]

for fname in files_to_check:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue
    d = np.load(path, allow_pickle=True)
    print(f"=== {fname} ===")
    for key in d.keys():
        val = d[key]
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            if val.ndim == 1 and len(val) <= 15:
                print(f"    values: {val}")
        else:
            print(f"  {key}: {val}")
    
    if 'explained_variance' in d:
        ev = d['explained_variance']
        cum = np.cumsum(ev)
        print(f"  Cumulative variance:")
        for i in range(min(10, len(ev))):
            print(f"    PC{i+1}: {ev[i]*100:.2f}%  cum: {cum[i]*100:.2f}%")
    print()

# Also check per-participant
print("=== Per-participant PCA (first 2 PCs) ===")
for pid in ['P01','P02','P03','P04','P05','P06','P07','P08','P09','P11','P12']:
    path = os.path.join(DATA_DIR, f'pca_myoki_{pid}_all_tasks_static.npz')
    if not os.path.exists(path):
        continue
    d = np.load(path, allow_pickle=True)
    ev = d['explained_variance']
    n = d['n_samples']
    print(f"  {pid}: PC1={ev[0]*100:.2f}%, PC2={ev[1]*100:.2f}%, PC1+PC2={(ev[0]+ev[1])*100:.2f}%, N={n}")
