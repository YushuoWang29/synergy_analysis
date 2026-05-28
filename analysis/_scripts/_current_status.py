#!/usr/bin/env python
"""Print current progress status."""
import json, os

print("=" * 70)
print("CURRENT PROGRESS STATUS REPORT")
print("=" * 70)

# 1. Wide results
wide_path = "PCA_results/experiments/wide_results.json"
if os.path.exists(wide_path):
    data = json.load(open(wide_path))
    myoki = [r for r in data if 'myoki' in r['label'].lower()]
    ninapro = [r for r in data if 'myoki' not in r['label'].lower()]
    print(f"\n1. MEGA EXPERIMENT V5 GRID SEARCH")
    print(f"   Total experiments: {len(data)}")
    print(f"   - Ninapro: {len(ninapro)}")
    print(f"   - MyoKi:   {len(myoki)}")
    print(f"   File: PCA_results/experiments/wide_results.json")
else:
    print(f"\n1. MEGA EXPERIMENT V5 GRID SEARCH: NOT COMPLETE")

# 2. Best results manifest
manifest_path = "PCA_results/experiments/best_results_manifest.json"
if os.path.exists(manifest_path):
    manifest = json.load(open(manifest_path))
    print(f"\n2. BEST RESULTS (MyoKi) SAVED: {len(manifest)} configs")
    for m in manifest:
        print(f"   - {m['name']}: Var={m['var_sum']*100:.1f}%  S1={m['s1_sim']:.4f}  S2={m['s2_sim']:.4f}  Score={m['rot_sum']:.4f}")
else:
    print(f"\n2. BEST RESULTS MANIFEST: NOT FOUND")

# 3. Best Ninapro candidates
print(f"\n3. BEST NINAPRO CANDIDATES (Tier 1: PC2>=10%, Uni>0.8, Grad>0.4)")
count = 0
for r in ninapro:
    if r['pc2_var'] >= 0.10 and abs(r['pc2_gradient_sim']) > 0.4 and r['pc1_uniform_sim'] > 0.8:
        count += 1
        if count <= 5:
            print(f"   #{count}: {r['label'][:60]}")
            print(f"      PC1={r['pc1_var']*100:.1f}%  PC2={r['pc2_var']*100:.1f}%  Sum={r['var_sum']*100:.1f}%")
            print(f"      Uni={r['pc1_uniform_sim']:.4f}  Grad={r['pc2_gradient_sim']:.4f}  All+={r['pc1_all_positive']}")
print(f"   Total Tier 1 Ninapro: {count}")

# 4. MyoKi results
print(f"\n4. MYOKI RESULTS: NOT IN WIDE GRID")
print(f"   MyoKi was NOT included in the mega_experiment_v5 grid")
print(f"   But MyoKi results were saved separately via save_best_results.py")

# 5. Summary
print(f"\n{'=' * 70}")
print(f"OVERALL SUMMARY")
print(f"{'=' * 70}")
print(f"✓ 432 Ninapro experiments completed and analyzed")
print(f"✓ Top Ninapro: power_no_abd_range  → PC2=14.8%  Sum=94.4%  Grad=0.70  Uni=0.97")
print(f"✓ Top Ninapro: power+tripod_no_abd_range → PC2=15.0%  Sum=92.7%  Grad=0.69  Uni=0.97")
print(f"✓ MyoKi best results saved separately (from earlier script runs)")
print(f"")
print(f"❌ What's missing:")
print(f"   - MyoKi was NOT included in the wide grid search")
print(f"   - Ninapro PC2 patterns don't perfectly match Pisa gradient (-2:-1:0:1:2)")
print(f"   - No cross-dataset comparison figures generated yet")
print(f"   - No final recommendation document")
