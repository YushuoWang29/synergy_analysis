"""Check V5 results so far and display best balanced solutions."""
import json, os

path = 'PCA_results/experiments/mega_v5_results.json'
if not os.path.exists(path):
    print(f"File not found yet. V5 still running.")
    # Check if partial results exist
    for f in os.listdir('PCA_results/experiments'):
        if 'v5' in f:
            print(f"  Found: {f}")
    exit()

with open(path) as f:
    results = json.load(f)

print(f"V5 Results loaded: {len(results)} experiments\n")

# Filter for balanced variance + good similarity
print("=" * 100)
print("BEST BALANCED SOLUTIONS (PC2 >= 8%, S1+S2 >= 1.2)")
print("=" * 100)

filtered = [r for r in results if r['pc2_var'] >= 0.08 and (r['s1_sim'] + r['s2_sim']) >= 1.2]
filtered.sort(key=lambda r: r['balanced_score'], reverse=True)

print(f"{'Rank':<5} {'Label':<55} {'Var%':<8} {'PC2%':<8} {'S1':<8} {'S2':<8} {'BalSc':<8}")
print("-" * 100)

for i, r in enumerate(filtered[:30]):
    print(f"{i+1:<5} {r['label'][:54]:<55} {r['pc1_pc2_sum']*100:<7.1f}% "
          f"{r['pc2_var']*100:<7.1f}% {r['s1_sim']:<8.4f} {r['s2_sim']:<8.4f} "
          f"{r['balanced_score']:<8.4f}")

print(f"\n{'='*100}")
print("ALL SOLUTIONS WITH PC2 >= 10% (best variance balance)")
print(f"{'='*100}")

high_pc2 = [r for r in results if r['pc2_var'] >= 0.10]
high_pc2.sort(key=lambda r: r['rot_sum'], reverse=True)

for i, r in enumerate(high_pc2[:20]):
    print(f"{i+1}. {r['label'][:50]:<52} Var={r['pc1_pc2_sum']*100:.1f}% "
          f"PC2={r['pc2_var']*100:.1f}% S1={r['s1_sim']:.4f} S2={r['s2_sim']:.4f} "
          f"Bal={r['balanced_score']:.4f}")

print(f"\n{'='*100}")
print("TOP 15 BY PATTERN SIMILARITY (rot_sum) WITH PC2 >= 8%")
print(f"{'='*100}")

similar = [r for r in results if r['pc2_var'] >= 0.08]
similar.sort(key=lambda r: r['rot_sum'], reverse=True)

for i, r in enumerate(similar[:15]):
    print(f"{i+1}. {r['label'][:50]:<52} S1={r['s1_sim']:.4f} S2={r['s2_sim']:.4f} "
          f"Sum={r['rot_sum']:.4f} PC2={r['pc2_var']*100:.1f}% "
          f"Var={r['pc1_pc2_sum']*100:.1f}%")
