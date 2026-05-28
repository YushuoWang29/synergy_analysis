"""Quick peek at PCA data for LaTeX document."""
import numpy as np
import os

d = np.load('../PCA_results/MyoKi/pca_myoki_all_grasp_static.npz', allow_pickle=True)
print('=== PCA Data Summary (all_grasp_static) ===')
print('n_samples:', d['n_samples'])
print('n_features:', d['n_features'])
print('data_label:', d['data_label'])
print('Components shape:', d['components'].shape)
print()
print('Explained variance:')
ev = d['explained_variance']
for i in range(len(ev)):
    print('  PC{}: {:.2f}%'.format(i+1, ev[i]*100))
print()
print('Cumulative:')
cum = d['explained_variance_cumsum']
for i in range(len(cum)):
    print('  {} PCs: {:.2f}%'.format(i+1, cum[i]*100))
print()
print('First 3 PC loadings:')
for i in range(3):
    arr = d['components'][i]
    s = ', '.join(['{:.3f}'.format(x) for x in arr])
    print('  PC{}: [{}]'.format(i+1, s))

# Pisa
rp = '../PCA_results/MyoKi/pca_myoki_all_grasp_static_pisa_rotated.npz'
if os.path.exists(rp):
    dr = np.load(rp, allow_pickle=True)
    print()
    print('Pisa Rotated components:')
    for i in range(2):
        arr = dr['components'][i]
        s = ', '.join(['{:.3f}'.format(x) for x in arr])
        print('   [{}]'.format(s))

# All tasks
d_all = np.load('../PCA_results/MyoKi/pca_myoki_all_tasks_static.npz', allow_pickle=True)
print()
print('=== All Tasks Static ===')
print('n_samples:', d_all['n_samples'])
for i, v in enumerate(d_all['explained_variance']):
    print('  PC{}: {:.2f}%  (cum: {:.2f}%)'.format(i+1, v*100, d_all['explained_variance_cumsum'][i]*100))
