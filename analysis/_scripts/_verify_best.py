"""Verify all saved best PCA results."""
import numpy as np, os

print("Verifying synergy-viewer files in PCA_results/MyoKi/")
print("=" * 60)

targets = [
    "pca_myoki_best_flexion_only",
    "pca_myoki_allgrasp_flexion_only",
    "pca_myoki_allgrasp_no_abd",
    "pca_myoki_power_no_abd",
    "pca_myoki_power_static10_no_abd",
    "pca_myoki_P12_taskavg_no_abd",
    "pca_myoki_P05_taskavg_no_abd",
    "pca_myoki_power_median_all15",
    "pca_myoki_power_all15",
    "pca_myoki_allgrasp_all15",
]

for base in targets:
    path = os.path.join("PCA_results/MyoKi", base + ".npz")
    if not os.path.exists(path):
        print(f"  MISSING: {base}.npz")
        continue
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    comp = d["components"]
    mean = d["mean"]
    var = d["explained_variance"]
    label = str(d["data_label"])
    
    print(f"\n  {base}.npz:")
    print(f"    components: {comp.shape}")
    print(f"    mean: {mean.shape}")
    print(f"    var: PC1={var[0]:.1%} PC2={var[1]:.1%} Sum={(var[0]+var[1]):.1%}")
    print(f"    label: {label}")

print("\n" + "=" * 60)
print("All verification files in PCA_results/experiments/")

for f in sorted(os.listdir("PCA_results/experiments")):
    if f.endswith(".json") or f.endswith(".md"):
        print(f"  {f}")

print("\nDone! Ready for visualization.")
