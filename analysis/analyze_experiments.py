"""Analyze experiment results and find best configuration."""
import json
import numpy as np

with open('PCA_results/experiments/experiment_deep_results.json') as f:
    results = json.load(f)

def score(r):
    return r.get('rot_s1_sim', 0) + r.get('rot_s2_sim', 0)

results.sort(key=score, reverse=True)

print("=" * 110)
print("TOP 30 EXPERIMENTS - RANKED BY S1+S2 SIMILARITY")
print("=" * 110)
print(f"{'#':<3} {'Experiment':<55} {'N':>7} {'PC1':>8} {'PC2':>8} {'Sum':>8} {'R1-S1':>8} {'R2-S2':>8} {'Score':>8}")
print("-" * 110)

for i, r in enumerate(results[:30]):
    s = score(r)
    line = (f"{i+1:<3} {r['label']:<55} {r['n_samples']:>7} "
            f"{r['pc1_var']*100 if r['pc1_var'] > 0 else 0:>7.1f}% "
            f"{r['pc2_var']*100 if r['pc2_var'] > 0 else 0:>7.1f}% "
            f"{r['pc1_pc2_sum']*100 if r['pc1_pc2_sum'] > 0 else 0:>7.1f}% "
            f"{r['rot_s1_sim']:>+7.3f} {r['rot_s2_sim']:>+7.3f} {s:>7.3f}")
    prefix = "★" if i == 0 else " "
    print(f"{prefix} {line}")

print("\n\n" + "=" * 80)
print("BEST APPROACHES - GROUPED BY METHOD")
print("=" * 80)

# Power grasp approaches
print("\n--- POWER GRASP approaches ---")
for r in results:
    if 'power_grasp' in r['label'] and 'fa2' not in r['label']:
        s = score(r)
        print(f"  {r['label']:<55} R1→S1={r['rot_s1_sim']:+.3f} R2→S2={r['rot_s2_sim']:+.3f} Score={s:.3f}")

print("\n--- ALL GRASP approaches ---")
for r in results:
    if 'all_grasp' in r['label'] and 'fa2' not in r['label']:
        s = score(r)
        print(f"  {r['label']:<55} R1→S1={r['rot_s1_sim']:+.3f} R2→S2={r['rot_s2_sim']:+.3f} Score={s:.3f}")

print("\n--- PICKUP GRASP approaches ---")
for r in results:
    if 'pickup_grasp' in r['label'] and 'fa2' not in r['label']:
        s = score(r)
        print(f"  {r['label']:<55} R1→S1={r['rot_s1_sim']:+.3f} R2→S2={r['rot_s2_sim']:+.3f} Score={s:.3f}")

print("\n--- FACTOR ANALYSIS approaches ---")
for r in results:
    if 'fa2' in r['label']:
        s = score(r)
        print(f"  {r['label']:<55} R1→S1={r['rot_s1_sim']:+.3f} R2→S2={r['rot_s2_sim']:+.3f} Score={s:.3f}")

# Also check the Round 1 results
print("\n\n" + "=" * 80)
print("ROUND 1 RESULTS")
print("=" * 80)
with open('PCA_results/experiments/experiment_results.json') as f:
    r1 = json.load(f)

def score_r1(r):
    return r.get('rot_pc1_s1_sim', 0) + r.get('rot_pc2_s2_sim', 0)
r1.sort(key=score_r1, reverse=True)

print(f"{'Experiment':<55} {'N':>7} {'PC1':>8} {'PC2':>8} {'Sum':>8} {'R1-S1':>8} {'R2-S2':>8}")
print("-" * 110)
for r in r1[:15]:
    s = score_r1(r)
    print(f"  {r['label']:<55} {r['n_samples']:>7} "
          f"{r['pc1_var']*100:>7.1f}% {r['pc2_var']*100:>7.1f}% "
          f"{r['pc1_pc2_sum']*100:>7.1f}% "
          f"{r['rot_pc1_s1_sim']:>+7.3f} {r['rot_pc2_s2_sim']:>+7.3f}")

print("\n\n" + "=" * 80)
print("ANALYSIS: What makes PC1 look like uniform open/close?")
print("=" * 80)

# Check the top result's PC1 weights
top = results[0]
pc1 = top.get('pc1', [])
pc2 = top.get('pc2', [])
rot_s1 = top.get('rot_s1', [])
rot_s2 = top.get('rot_s2', [])

print(f"\nTop result: {top['label']}")
print(f"  Method: {'Factor Analysis' if 'fa2' in top['label'] else 'PCA'}")

if pc1:
    print(f"\n  PC1 weights:")
    names = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp",
             "IdxMCP","IdxPIP","MidMCP","MidPIP","IdxMidAbd",
             "RingMCP","RingPIP","MidRingAbd",
             "LitMCP","LitPIP","RingLitAbd"]
    for i, (n, w) in enumerate(zip(names[:len(pc1)], pc1)):
        print(f"    {n:12s}: {w:+7.4f}")
    
    # Compute gradient correlation
    gradient = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2])
    g = gradient[:len(pc1)] / np.linalg.norm(gradient[:len(pc1)])
    u = np.ones(len(pc1)) / np.linalg.norm(np.ones(len(pc1)))
    
    print(f"\n  PC1 vs uniform: {np.dot(np.array(pc1)/np.linalg.norm(pc1), u):+.4f}")
    print(f"  PC1 vs gradient: {np.dot(np.array(pc1)/np.linalg.norm(pc1), g[:len(pc1)]):+.4f}")

# Compare with other top results to find patterns
print("\n\n" + "=" * 80)
print("PATTERNS: What all top-5 results have in common")
print("=" * 80)

for i, r in enumerate(results[:5]):
    label = r['label']
    has_zscore = '_zTrue' in label or 'zscore' in label
    method = 'FA' if 'fa2' in label else 'PCA'
    task_type = 'power' if 'power' in label else ('all' if 'all' in label else 'pickup' if 'pickup' in label else 'other')
    rep_type = 'trial_mean' if 'trial_mean' in label else ('all_frames' if 'all_frames' in label else 
               'static_last25' if 'static_last25' in label else 'static_last50' if 'static_last50' in label else
               'reach_first25' if 'reach_first25' in label else 'reach_first50' if 'reach_first50' in label else
               'hold_last25' if 'hold_last25' in label else 'velocity' if 'velocity' in label else
               'full' if 'full' in label else 'mid_phase' if 'mid_phase' in label else 'other')
    joint_set = 'flexion10' if 'flexion' in label else ('no_thumbopp' if 'thumbopp' in label else
                'no_abduction' if 'abduction' in label else 'all15')
    print(f"  #{i+1}: {label}")
    print(f"    Task={task_type}, Rep={rep_type}, Joints={joint_set}, Zscore={has_zscore}, Method={method}")
    print(f"    S1={r['rot_s1_sim']:.3f}, S2={r['rot_s2_sim']:.3f}, PC1={r['pc1_var']*100:.1f}%")

print("\n\nRECOMMENDED CONFIGURATION:")
print("-" * 80)
top = results[0]
print(f"  Task set: Power grasps (cylindrical+spherical+hook)")
print(f"  Data rep: Trial means (static posture per trial)")
print(f"  Normalization: Z-score ({'Yes' if '_zTrue' in top['label'] else top['label']})")
print(f"  Achieves: S1={top['rot_s1_sim']:.3f}, S2={top['rot_s2_sim']:.3f}")
print(f"  PC1 variance: {top['pc1_var']*100:.1f}%")
print(f"  PC1+PC2: {top['pc1_pc2_sum']*100:.1f}%")
