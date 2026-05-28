"""Examine the best result's PC weights in detail."""
import json
import numpy as np

with open('PCA_results/experiments/experiment_deep_results.json') as f:
    results = json.load(f)

def score(r):
    return float(r.get('rot_s1_sim', 0)) + float(r.get('rot_s2_sim', 0))
results.sort(key=score, reverse=True)

top = results[0]
print("=" * 70)
print("BEST RESULT: {}".format(top['label']))
print("=" * 70)
print("S1 similarity: {:.4f}".format(float(top['rot_s1_sim'])))
print("S2 similarity: {:.4f}".format(float(top['rot_s2_sim'])))
print("PC1 var: {:.1f}%".format(float(top['pc1_var'])*100))
print("PC2 var: {:.1f}%".format(float(top['pc2_var'])*100))

names = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
         "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
         "RingMCP","RingPIP","MidRingAbd",
         "LitMCP","LitPIP","RingLitAbd"]

# Parse the arrays from string format
import ast
pc1 = np.array(ast.literal_eval(top['pc1']))
pc2 = np.array(ast.literal_eval(top['pc2']))
s1 = np.array(ast.literal_eval(top['rot_s1']))
s2 = np.array(ast.literal_eval(top['rot_s2']))

print("\n  Joint      |  Original PC1  |  Original PC2  |  Rotated S1   |  Rotated S2")
print("  " + "-"*72)
for i in range(len(pc1)):
    print("  {:12s} | {:>+10.4f}    {:>+10.4f}    {:>+10.4f}    {:>+10.4f}".format(
        names[i], pc1[i], pc2[i], s1[i], s2[i]))

print("\n  Pisa Target S1 (uniform):")
target_s1 = np.ones(len(pc1)) / np.linalg.norm(np.ones(len(pc1)))
for i in range(len(pc1)):
    print("  {:12s}: {:+.4f}".format(names[i], target_s1[i]))

print("\n  Pisa Target S2 (gradient):")
target_s2_raw = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
target_s2 = target_s2_raw / np.linalg.norm(target_s2_raw)
for i in range(len(pc1)):
    print("  {:12s}: {:+.4f}".format(names[i], target_s2[i]))

# Check correlations
print("\n\n  Without rotation:")
print("  PC1 · uniform target: {:.4f}".format(np.dot(pc1/np.linalg.norm(pc1), target_s1)))
print("  PC2 · gradient target: {:.4f}".format(np.dot(pc2/np.linalg.norm(pc2), target_s2)))

print("\n\n  WITH rotation:")
print("  S1 · uniform target: {:.4f}".format(np.dot(s1/np.linalg.norm(s1), target_s1)))
print("  S2 · gradient target: {:.4f}".format(np.dot(s2/np.linalg.norm(s2), target_s2)))

# Sign analysis
print("\n\n  PC1 sign analysis ({} joints):".format(len(pc1)))
pos_count = sum(1 for w in pc1 if float(w) > 0)
neg_count = sum(1 for w in pc1 if float(w) < 0)
print("  Positive: {}, Negative: {}".format(pos_count, neg_count))
print("  All same sign?: {}".format(pos_count == len(pc1) or neg_count == len(pc1)))

# Also compare with top 5 results
print("\n\n" + "=" * 70)
print("COMPARING TOP 5 RESULTS")
print("=" * 70)
for i, r in enumerate(results[:5]):
    pc1_i = np.array(ast.literal_eval(r['pc1']))
    if len(pc1_i) < 15:
        continue
    pos = sum(1 for w in pc1_i if float(w) > 0)
    neg = sum(1 for w in pc1_i if float(w) < 0)
    print("  #{}: {:<50} PC1 sign: {}+/{-}".format(
        i+1, r['label'][:50], pos, neg))
