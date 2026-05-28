"""Summarize all experiment results and find top candidates"""
import json, numpy as np

with open('PCA_results/experiments/wide_results.json') as f:
    data = json.load(f)

for r in data:
    r['source'] = 'myoki' if 'myoki' in r['label'].lower() else 'ninapro'
    r['pc1'] = np.array(r['pc1_raw'])
    r['pc2'] = np.array(r['pc2_raw'])

data.sort(key=lambda r: r['pc2_var'], reverse=True)

print(f"Total experiments: {len(data)}")
print()

# ============ MYOKI SECTION ============
print("=" * 90)
print("MYOKI RESULTS (sorted by PC2%)")
print("=" * 90)
myoki = [r for r in data if r['source'] == 'myoki']
myoki.sort(key=lambda r: r['pc2_var'], reverse=True)
print(f"Total MyoKi experiments: {len(myoki)}")
for i,r in enumerate(myoki[:30]):
    print(f"{i+1:2d}. PC2={r['pc2_var']*100:5.1f}% Sum={r['var_sum']*100:5.1f}% Uni={r['pc1_uniform_sim']:.3f} Grad={r['pc2_gradient_sim']:.3f} All+={r['pc1_all_positive']} | {r['label'][:70]}")

print()

# ============ NINAPRO SECTION ============
print("=" * 90)
print("NINAPRO RESULTS - TOP 20 (sorted by PC2%)")
print("=" * 90)
nina = [r for r in data if r['source'] == 'ninapro']
nina.sort(key=lambda r: r['pc2_var'], reverse=True)
print(f"Total Ninapro experiments: {len(nina)}")
for i,r in enumerate(nina[:20]):
    print(f"{i+1:2d}. PC2={r['pc2_var']*100:5.1f}% Sum={r['var_sum']*100:5.1f}% Uni={r['pc1_uniform_sim']:.3f} Grad={r['pc2_gradient_sim']:.3f} All+={r['pc1_all_positive']} | {r['label'][:70]}")

print()

# ============ BEST CANDIDATES DEEP DIVE ============
print("=" * 90)
print("BEST CANDIDATES - DEEP DIVE")
print("=" * 90)

# Filter: must have good pattern AND reasonable variance
candidates = [r for r in data if r['pc1_all_positive'] and r['pc1_uniform_sim'] > 0.85 and abs(r['pc2_gradient_sim']) > 0.45]
candidates.sort(key=lambda r: r['pc2_var'] * (abs(r['pc2_gradient_sim'])+r['pc1_uniform_sim'])/2, reverse=True)

for i,r in enumerate(candidates[:15]):
    src = 'M' if r['source'] == 'myoki' else 'N'
    print(f"\n{'='*60}")
    print(f"#{i+1} [{src}] {r['label']}")
    print(f"{'='*60}")
    print(f"  PC1: {r['pc1_var']*100:.1f}% variance")
    print(f"  PC2: {r['pc2_var']*100:.1f}% variance")
    print(f"  Sum: {r['var_sum']*100:.1f}%")
    print(f"  Samples: {r['n_samples']}")
    print(f"  Joints: {r['n_joints']}")
    print(f"  PC1 Uniformity: {r['pc1_uniform_sim']:.4f}")
    print(f"  PC2 Gradient: {r['pc2_gradient_sim']:.4f}")
    print(f"  PC1 all positive: {r['pc1_all_positive']}")
    print(f"  Quality: {r.get('quality', 'N/A')}")
    print(f"  Joint names: {r['joint_names']}")
    print(f"  PC1 weights: {np.round(r['pc1'], 4).tolist()}")
    print(f"  PC2 weights: {np.round(r['pc2'], 4).tolist()}")

print()
print("=" * 90)
print("SUMMARY: Top 15 by composite score (PC2% * pattern quality)")
print("=" * 90)
for i,r in enumerate(candidates[:15]):
    src = 'M' if r['source'] == 'myoki' else 'N'
    print(f"#{i+1:2d} [{src}] PC2={r['pc2_var']*100:.1f}% Sum={r['var_sum']*100:.1f}% Uni={r['pc1_uniform_sim']:.3f} Grad={r['pc2_gradient_sim']:.3f} | {r['label'][:60]}")
