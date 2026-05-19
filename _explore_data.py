"""
Temp script to explore MyoKi data structure and participant info.
"""
import scipy.io as sio
import numpy as np
import os, glob

# Load combined
data = sio.loadmat('MyoKi Dataset/preprocessed/all_participants_preprocessed.mat')
print('=== Combined Data ===')
print('Keys:', [k for k in data.keys() if not k.startswith('__')])
print('angles shape:', data['angles'].shape)
print('num_trials:', data['num_trials'])
print()

# Individual files
preproc_dir = 'MyoKi Dataset/preprocessed'
for f in sorted(glob.glob(os.path.join(preproc_dir, 'P*_preprocessed.mat'))):
    pdata = sio.loadmat(f)
    n = os.path.basename(f)
    a = pdata['angles']
    nt = pdata['num_trials']
    print(f'{n}: angles {a.shape}, num_trials={nt}')

# Check task distribution in combined data
print('\n=== Task distribution ===')
if 'trial_tasks' in data:
    tasks = data['trial_tasks'].flatten()
    unique, counts = np.unique(tasks, return_counts=True)
    for t, c in sorted(zip(unique, counts)):
        print(f'  Task {int(t)}: {c} trials')

# Check how many participants are in the combined
print('\n=== Participant-wise stats ===')
for f in sorted(glob.glob(os.path.join(preproc_dir, 'P*_preprocessed.mat'))):
    pdata = sio.loadmat(f)
    n = os.path.basename(f)
    angles = pdata['angles']
    print(f'{n}: {angles.shape[0]} frames, mean={np.mean(angles, axis=0)[:5]}...')
    
# Analyze: is the mean pose similar across participants?
print('\n=== Mean pose comparison across participants ===')
all_means = []
for f in sorted(glob.glob(os.path.join(preproc_dir, 'P*_preprocessed.mat'))):
    pdata = sio.loadmat(f)
    n = os.path.basename(f)
    angles = pdata['angles']
    m = np.mean(angles, axis=0)
    all_means.append(m)
    print(f'{n}: mean norm={np.linalg.norm(m):.2f}')

all_means = np.array(all_means)
print(f'\nBetween-participant mean angles STD: {np.std(all_means, axis=0).round(2)}')
print(f'Between-participant mean angles range: {np.std(all_means, axis=0).mean():.2f}')
