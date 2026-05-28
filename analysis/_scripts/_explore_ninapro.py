"""Explore Ninapro raw data structure and preprocessing."""
import scipy.io as sio
import numpy as np
import os, glob

for fname in ['S1_E2_A1.mat', 'S28_E1_A1.mat', 'S68_E3_A1.mat']:
    m = sio.loadmat(f'Ninapro Data/original/{fname}')
    glove = m['glove']
    angles = m['angles']
    restim = m['restimulus'].flatten()
    ord_angles = m['order_of_angles'].flatten().tolist()
    
    print(f'\n=== {fname} ===')
    print(f'  glove:  min={glove.min():.2f}, max={glove.max():.2f}, nan={np.isnan(glove).sum()}')
    print(f'  angles: min={angles.min():.2f}, max={angles.max():.2f}, nan={np.isnan(angles).sum()}')
    print(f'  order_of_angles: {ord_angles}')
    
    unique_labels = np.unique(restim)
    print(f'  restimulus labels: {unique_labels.tolist()}')
    counts = {int(l): int((restim==l).sum()) for l in unique_labels}
    print(f'  restimulus counts: {counts}')
    print(f'  duration: {len(glove)/100:.1f}s')
    print(f'  subject: {m["subject"].item()}, exercise: {m["exercise"].item()}')

# Also check E1 vs E2 vs E3 unique subjects
import glob
for ex in ['E1', 'E2', 'E3']:
    files = sorted(glob.glob(f'Ninapro Data/original/S*_{ex}_*.mat'))
    subs = sorted(set(f.split('_')[0] for f in files))
    print(f'\n{ex}: {len(files)} files, subjects: {subs}')

# Check existing preprocessed files
print('\n\n=== Existing preprocessed files ===')
for ex in ['E1', 'E2', 'E3']:
    files = sorted(glob.glob(f'Ninapro Data/{ex}/*.mat'))
    print(f'{ex}: {len(files)} files')
    for f in files[:3]:
        m = sio.loadmat(f)
        a = m['angles']
        print(f'  {os.path.basename(f)}: shape={a.shape}, nan={np.isnan(a).sum()}')
