"""Compare Ninapro PCA results with MyoKi and Pisa/IIT."""
import numpy as np
import scipy.io as sio

# Load Ninapro E123 PCA
npz = np.load('PCA_results/Ninapro/pca_E123.npz')
ne = npz['explained_variance']
np_comp = npz['components']
print('=== Ninapro E123 (all data) ===')
print(f'PC1: {ne[0]:.2%}, PC2: {ne[1]:.2%}')
print(f'Components shape: {np_comp.shape}')

# Load Pisa rotated
rpz = np.load('PCA_results/Ninapro/pca_E123_pisa.npz')
rv = rpz['components']
print(f'Rotated components shape: {rv.shape}')

# Pisa targets
P1 = np.array([1]*20) / np.linalg.norm([1]*20)
P2_RAW = [2,2,1,1, 1,1,1,1, 1,1,1, -1,-1,-1,-1, -2,-2,-2,-2,-2]
P2 = np.array(P2_RAW, dtype=float) / np.linalg.norm(P2_RAW)

print(f'\n=== Pisa comparison ===')
print(f'RotPC1 vs Pisa Syn1: cos_sim = {np.dot(rv[0], P1):.4f}')
print(f'RotPC2 vs Pisa Syn2: cos_sim = {np.dot(rv[1], P2):.4f}')
print(f'RotPC1 vs Pisa Syn2: cos_sim = {np.dot(rv[0], P2):.4f}')
print(f'RotPC2 vs Pisa Syn1: cos_sim = {np.dot(rv[1], P1):.4f}')

# Check if rotation R is well-conditioned
R = rpz['rotation']
print(f'\nRotation R:\n{R}')
print(f'Det(R) = {np.linalg.det(R):.4f}')

# Per-exercise comparison
print(f'\n{"="*70}')
print(f'{"Dataset":<12} {"N":>6} {"PC1":>8} {"PC2":>8} {"Sum":>8} {"RotPC1→P1":>12} {"RotPC2→P2":>12}')
print(f'{"-"*70}')
for label in ['E1', 'E2', 'E3', 'E12', 'E13', 'E23', 'E123']:
    try:
        pz = np.load(f'PCA_results/Ninapro/pca_{label}.npz')
        rpz = np.load(f'PCA_results/Ninapro/pca_{label}_pisa.npz')
        e = pz['explained_variance']
        rv = rpz['components']
        s1 = np.dot(rv[0], P1)
        s2 = np.dot(rv[1], P2)
        print(f'{label:<12} {pz["n_samples"]:>6} {e[0]:>7.1%} {e[1]:>7.1%} {(e[0]+e[1]):>7.1%} {s1:>11.4f} {s2:>11.4f}')
    except:
        pass

# Now compare with MyoKi
print(f'\n{"="*70}')
print(f'{"="*70}')
print('COMPARISON WITH MyoKi:')
print(f'{"="*70}')

# MyoKi PCA load
try:
    mk = np.load('PCA_results/MyoKi/pca_myoki_all.npz')
    me = mk['explained_variance']
    mc = mk['components']
    print(f'\n=== MyoKi (all data) ===')
    print(f'PC1: {me[0]:.2%}, PC2: {me[1]:.2%}')
    print(f'N samples: {mk["n_samples"]}')
    
    # Compare with Ninapro E123 PC1
    print(f'\n=== Cross-dataset PC similarity ===')
    # Need to align joint spaces - both are 20D but different order
    # Ninapro joints are in their natural order
    # MyoKi joints need to be mapped from Ninapro order
    # Let me check MyoKi's joint order
    print(f'MyoKi components shape: {mc.shape}')
    print(f'MyoKi first PC:\n{mc[0]}')
    print(f'Ninapro first PC:\n{np_comp[0]}')
    
    # Cosine similarity between datasets' PC1
    p1_sim = np.dot(mc[0], np_comp[0]) / (np.linalg.norm(mc[0]) * np.linalg.norm(np_comp[0]))
    print(f'\nNinapro PC1 vs MyoKi PC1 (joint-by-joint): cos_sim = {p1_sim:.4f}')
    p2_sim = np.dot(mc[1], np_comp[1]) / (np.linalg.norm(mc[1]) * np.linalg.norm(np_comp[1]))
    print(f'Ninapro PC2 vs MyoKi PC2: cos_sim = {p2_sim:.4f}')
    
except Exception as ex:
    print(f'Could not load MyoKi PCA: {ex}')

# Check the DIP3 dominance in Ninapro
print(f'\n{"="*70}')
print('CHECKING DIP3 DOMINANCE IN NINAPRO PCA:')
print(f'{"="*70}')
for label in ['E1', 'E2', 'E3', 'E123']:
    pz = np.load(f'PCA_results/Ninapro/pca_{label}.npz')
    pc1 = pz['components'][0]
    print(f'\n{label} PC1 weights:')
    # DIP3 is index 10 in 20-joint set
    for i, w in enumerate(pc1):
        if abs(w) > 0.1:
            print(f'  Joint {i:2d} (DIP3={i==10:1d}): {w:+7.4f}')
    print(f'  DIP3 share of |PC1|: {abs(pc1[10])/np.sum(np.abs(pc1)):.1%}')

# Also check if the raw glove data has special properties
print(f'\n{"="*70}')
print('RAW DATA CHECK:')
for f in ['S1_E2_A1.mat', 'S28_E1_A1.mat', 'S68_E3_A1.mat']:
    m = sio.loadmat(f'Ninapro Data/original/{f}')
    g = m['glove']
    print(f'\n{f}:')
    print(f'  glove range: [{g.min():.2f}, {g.max():.2f}]')
    print(f'  per-joint std: {np.std(g, axis=0).round(2).tolist()}')
    print(f'  per-joint range: {[(g[:,i].min(), g[:,i].max()) for i in range(5)]}')
