import json
with open('PCA_results/experiments/experiment_deep_results.json') as f:
    results = json.load(f)

r = results[0]
for k, v in r.items():
    print('{}: {} = {}'.format(k, type(v).__name__, str(v)[:80]))
print()

def score(r):
    return float(r.get('rot_s1_sim', 0)) + float(r.get('rot_s2_sim', 0))
results.sort(key=score, reverse=True)

for i, r in enumerate(results[:20]):
    s = score(r)
    p1 = float(r.get('pc1_var', 0)) * 100
    p2 = float(r.get('pc2_var', 0)) * 100
    sm = float(r.get('pc1_pc2_sum', 0)) * 100
    s1 = float(r.get('rot_s1_sim', 0))
    s2 = float(r.get('rot_s2_sim', 0))
    label = r['label'][:55]
    print('{:>3} {:<55} PC1={:>6.1f}% PC2={:>6.1f}% Sum={:>6.1f}% S1={:>+.3f} S2={:>+.3f} Score={:.3f}'.format(
        i+1, label, p1, p2, sm, s1, s2, s))
