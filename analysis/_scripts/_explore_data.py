"""Explore Ninapro data structure"""
import scipy.io as sio, numpy as np

# 1) Check original raw file
f = 'Ninapro Data/original/S1_E2_A1.mat'
mat = sio.loadmat(f)
print('=== Original raw S1_E2_A1.mat ===')
for k in sorted(mat.keys()):
    if not k.startswith('__'):
        v = mat[k]
        shape = v.shape if hasattr(v, 'shape') else 'scalar'
        dtype = v.dtype if hasattr(v, 'dtype') else type(v)
        print(f'  {k}: shape={shape}, dtype={dtype}')
        
print(f'\nglove: {mat["glove"].shape}, min={mat["glove"].min():.2f}, max={mat["glove"].max():.2f}')
print(f'restimulus: unique={sorted(np.unique(mat["restimulus"]))}')
print(f'repetition: unique={sorted(np.unique(mat["repetition"]))}')

# 2) Check preprocessed (_B.mat) file
f2 = 'Ninapro Data/E2/S1_E2_B.mat'
mat2 = sio.loadmat(f2)
print(f'\n=== Preprocessed E2/S1_E2_B.mat ===')
for k in sorted(mat2.keys()):
    if not k.startswith('__'):
        v = mat2[k]
        shape = v.shape if hasattr(v, 'shape') else 'scalar'
        dtype = v.dtype if hasattr(v, 'dtype') else type(v)
        print(f'  {k}: shape={shape}, dtype={dtype}')

angles = mat2['angles']
num_trials = mat2['num_trials'].item()
print(f'\nangles shape: {angles.shape}')
print(f'num_trials: {num_trials}')
print(f'Expected frames: {num_trials * 100}')
print(f'angles min={angles.min():.2f}, max={angles.max():.2f}')

# 3) Check E1 preprocessed
f3 = 'Ninapro Data/E1/S28_E1_B.mat'
mat3 = sio.loadmat(f3)
print(f'\n=== Preprocessed E1/S28_E1_B.mat ===')
for k in sorted(mat3.keys()):
    if not k.startswith('__'):
        v = mat3[k]
        shape = v.shape if hasattr(v, 'shape') else 'scalar'
        dtype = v.dtype if hasattr(v, 'dtype') else type(v)
        print(f'  {k}: shape={shape}, dtype={dtype}')

angles3 = mat3['angles']
num_trials3 = mat3['num_trials'].item()
print(f'\nangles shape: {angles3.shape}')
print(f'num_trials: {num_trials3}')
print(f'Expected frames: {num_trials3 * 100}')
print(f'angles min={angles3.min():.2f}, max={angles3.max():.2f}')

# 4) Check a finished PCA model from Ninapro
print(f'\n=== Existing PCA model ===')
import glob
models = glob.glob('PCA_results/pca_model_*.npz')
print(f'Found models: {models}')
for m in models:
    d = np.load(m)
    print(f'\n{m}:')
    for k in d.keys():
        v = d[k]
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={v.shape}')
        else:
            print(f'  {k}: scalar={v}')
