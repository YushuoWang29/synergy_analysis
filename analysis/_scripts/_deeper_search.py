"""
DEEPER SEARCH: Focus on trial_range + variations around the sweet spot.
Try: different joints, normalization, task splits, robust aggregation.
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.linalg import orthogonal_procrustes
import openpyxl

SAVE_DIR = 'PCA_results/MyoKi'
EXP_DIR = 'PCA_results/experiments'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

N = 15
JN = ["ThumbAbd","ThumbMCP","ThumbIP","ThumbOpp","IdxMCP","IdxPIP",
      "MidMCP","MidPIP","IdxMidAbd","RingMCP","RingPIP",
      "MidRingAbd","LitMCP","LitPIP","RingLitAbd"]
PS1 = np.ones(N) / np.linalg.norm(np.ones(N))
PS2 = np.array([2,2,2,2,1,1,0,0,0,-1,-1,-1,-2,-2,-2], dtype=float)
PS2 = PS2 / np.linalg.norm(PS2)

FID = [1,2,4,5,6,7,9,10,12,13]
NAD = [0,1,2,3,4,5,6,7,9,10,12,13]
A15 = list(range(15))
# Joint sets for specific analysis
THUMB_PLUS_FLEX = [1,2,4,5,6,7,9,10,12,13]  # same as flex10 (no abd, no opp)
THUMB_BASE = [0,1,2]  # thumb + abduction
FINGER_FLEX = [4,5,6,7,9,10,12,13]  # all flexion without thumb
# All MCPs
MCP_JOINTS = [1,4,6,9,12]
# All PIPs  
PIP_JOINTS = [2,5,7,10,13]
# No thumb
NO_THUMB = [4,5,6,7,9,10,12,13,8,11,14]

wb = openpyxl.load_workbook('MyoKi Dataset/Task_categorization.xlsx')
ws = wb.active
cat = {}
ac = {'cylindrical':11,'spherical':12,'hook':13,'tripod':14,'pinch':15,'lumbrical':16,'complex':17,'none_action':18}
for row in ws.iter_rows(min_row=4, values_only=False):
    ti = row[0].value
    if ti is None: continue
    ti = int(ti)
    acts = []
    for name, ci in ac.items():
        if row[ci].value and str(row[ci].value).strip() == 'x':
            acts.append(name)
    cat[ti] = acts

POW = sorted([t for t,a in cat.items() if any(x in {'cylindrical','spherical','hook'} for x in a) and 'none_action' not in a])
PRE = sorted([t for t,a in cat.items() if any(x in {'tripod','pinch','lumbrical'} for x in a) and 'none_action' not in a and not any(x in {'cylindrical','spherical','hook'} for x in a)])
ALLG = sorted(set(POW)|set(PRE))

# Sub-categories
CYL = sorted([t for t,a in cat.items() if 'cylindrical' in a and 'none_action' not in a])
SPH = sorted([t for t,a in cat.items() if 'spherical' in a and 'none_action' not in a])
HOO = sorted([t for t,a in cat.items() if 'hook' in a and 'none_action' not in a])
TRI = sorted([t for t,a in cat.items() if 'tripod' in a and 'none_action' not in a])
PIN = sorted([t for t,a in cat.items() if 'pinch' in a and 'none_action' not in a])
LUM = sorted([t for t,a in cat.items() if 'lumbrical' in a and 'none_action' not in a])

PF = sorted(glob.glob('MyoKi Dataset/raw/P*.mat'))
VP = [f'P{p:02d}' for p in [1,2,3,4,5,6,7,8,9,11,12]]
FILES = [f for f in PF if any(p in os.path.basename(f) for p in VP)]

def load_trials(files, tids):
    out = []
    for f in files:
        m = sio.loadmat(f)
        g = np.nan_to_num(m['glove_calibrated'])
        ts = m['task'].flatten().astype(int)
        d = g[:, :15]
        for ti in tids:
            mk = (ts == ti)
            df = np.diff(mk.astype(int), prepend=0, append=0)
            st = np.where(df > 0)[0]
            en = np.where(df < 0)[0]
            for s, e in zip(st, en):
                tr = d[s:e, :]
                if len(tr) >= 5:
                    xo = np.linspace(0, 1, len(tr))
                    xn = np.linspace(0, 1, 100)
                    fint = interp1d(xo, tr, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
                    out.append((fint(xn), ti, os.path.basename(f)))
    return out

def pisa_sim(pc, jids):
    n = len(jids)
    s1 = np.ones(n) / np.linalg.norm(np.ones(n))
    s2r = PS2[jids]
    s2 = s2r / (np.linalg.norm(s2r) + 1e-10)
    tgt = np.column_stack([s1, s2])
    pc1 = pc[0] / (np.linalg.norm(pc[0]) + 1e-10)
    pc2 = pc[1] / (np.linalg.norm(pc[1]) + 1e-10)
    P = np.column_stack([pc1, pc2])
    R, _ = orthogonal_procrustes(P, tgt)
    rot = P @ R
    return float(np.dot(rot[:,0], s1)), float(np.dot(rot[:,1], s2)), R, rot

def export_full(name, X_full, X_src, jids, desc):
    """Run PCA on X_src (joint subset), export full 15-D model with padded zeros."""
    nc = min(len(jids), X_src.shape[0], 10)
    pca = PCA(n_components=nc)
    pca.fit(X_src)
    evr = pca.explained_variance_ratio_
    
    fc = np.zeros((nc, 15))
    for i in range(nc):
        fc[i, jids] = pca.components_[i]
    fm = np.mean(X_full, axis=0)
    
    # Sign-flip so thumb positive
    for i in range(nc):
        if fc[i, 1] < 0:
            fc[i] *= -1
    
    s1, s2, R, rot = pisa_sim(pca.components_, jids)
    p1 = evr[0]*100; p2 = evr[1]*100 if len(evr) > 1 else 0; tv = p1 + p2
    all_pos = bool(np.all(pca.components_[0] > 0))
    
    lbl = f"{desc} | PC1={p1:.1f}% PC2={p2:.1f}% S1={s1:.3f} S2={s2:.3f}"
    np.savez(os.path.join(SAVE_DIR, f'pca_myoki_{name}.npz'),
             components=fc, mean=fm, explained_variance=evr, data_label=lbl)
    np.savez(os.path.join(EXP_DIR, f'deep_{name}.npz'),
             components=fc, mean=fm, explained_variance=evr, data_label=lbl,
             s1_sim=s1, s2_sim=s2, all_pos=all_pos)
    
    return {'name':name, 'pc1':p1, 'pc2':p2, 'total':tv, 's1':s1, 's2':s2, 
            'nj':len(jids), 'all_pos':all_pos, 'n_samp':X_src.shape[0]}

def trial_range_per_task(trials, aggr_fn=np.mean):
    """For each task, compute trial range and aggregate."""
    by_t = {}
    for t, tid, _ in trials:
        if tid not in by_t: by_t[tid] = []
        by_t[tid].append(np.max(t, axis=0) - np.min(t, axis=0))
    Xf = np.array([aggr_fn(v, axis=0) for v in by_t.values()])
    return Xf, by_t

def trial_stats_per_task(trials, stats_fn):
    """Apply arbitrary stats function per task."""
    by_t = {}
    for t, tid, _ in trials:
        if tid not in by_t: by_t[tid] = []
        by_t[tid].append(stats_fn(t))
    return np.array([np.mean(v, axis=0) for v in by_t.values()])

def score(r):
    p1 = r['pc1']/100; p2 = r['pc2']/100; t = r['total']/100
    s1 = abs(r['s1']); s2 = abs(r['s2'])
    p1s = max(0, 1-abs(p1-0.75)*3); p2s = max(0, 1-abs(p2-0.15)*5)
    ts = max(0, 1-abs(t-0.90)*3); s1s = max(0, 1-abs(s1-0.95)*3)
    s2s = max(0, 1-abs(s2-0.75)*3); pb = min(p2*5, 0.5)
    allp_bonus = 0.1 if r['all_pos'] else 0.0
    return p1s*0.12 + p2s*0.20 + ts*0.12 + s1s*0.15 + s2s*0.20 + pb*0.10 + allp_bonus*0.11

def deep_search():
    results = []

    # ========== CONFIG A: trial_range + all joints, various norms ==========
    print("=== A: trial_range all15 various norms ===")
    for tn, ti in [('all_grasp', ALLG), ('power', POW), ('power_prec', sorted(set(POW)|set(PRE)))]:
        trials = load_trials(FILES, ti)
        for aggr_name, aggr_fn in [('mean', np.mean), ('median', np.median), ('trim20', lambda x, axis=0: np.mean(np.sort(x,axis=0)[int(len(x)*0.2):int(len(x)*0.8)],axis=0))]:
            try:
                Xf, _ = trial_range_per_task(trials, aggr_fn)
                for norm in ['none', 'zscore', 'range']:
                    if norm == 'zscore':
                        m=np.mean(Xf,0); s=np.std(Xf,0); s[s<1e-6]=1.0; Xn=(Xf-m)/s
                    elif norm == 'range':
                        mn=np.min(Xf,0); mx=np.max(Xf,0); rg=mx-mn; rg[rg<1e-6]=1.0; Xn=(Xf-mn)/rg
                    else:
                        Xn = Xf
                    nm = f"dA_{tn}_trialrange_{aggr_name}_{norm}"
                    r = export_full(nm, Xf, Xn, A15, f"A {tn} trial_range {aggr_name} {norm}")
                    results.append(r)
                    # Also try with no_abd
                    Xn_a = Xn[:, NAD]
                    nm2 = f"dA_{tn}_trialrange_{aggr_name}_{norm}_noabd"
                    r2 = export_full(nm, Xn_a, Xn_a, NAD, f"A {tn} trial_range {aggr_name} {norm} noabd")
                    results.append(r2)
            except Exception as e:
                print(f"  Error: {e}")

    # ========== CONFIG B: trial_range specific task splits ==========
    print("=== B: trial_range with specific grasp types ===")
    task_splits = {
        'cyl_sph': sorted(set(CYL)|set(SPH)),
        'cyl_prec': sorted(set(CYL)|set(PRE)),
        'sph_prec': sorted(set(SPH)|set(PRE)),
        'power_simple': sorted(set(CYL)|set(SPH)),
        'power_all': POW,
    }
    for sn, si in task_splits.items():
        trials = load_trials(FILES, si)
        for norm in ['none', 'zscore']:
            Xf, _ = trial_range_per_task(trials)
            if norm == 'zscore':
                m=np.mean(Xf,0); s=np.std(Xf,0); s[s<1e-6]=1.0; Xn=(Xf-m)/s
            else:
                Xn = Xf
            # all15
            nm = f"dB_{sn}_trialrange_{norm}"
            r = export_full(nm, Xf, Xn[:, A15], A15, f"B {sn} trial_range {norm}")
            results.append(r)
            # no_abd
            nm2 = f"dB_{sn}_trialrange_{norm}_noabd"
            r2 = export_full(nm, Xn[:, NAD], Xn[:, NAD], NAD, f"B {sn} trial_range {norm} noabd")
            results.append(r2)
            # flex10
            nm3 = f"dB_{sn}_trialrange_{norm}_flex10"
            r3 = export_full(nm, Xn[:, FID], Xn[:, FID], FID, f"B {sn} trial_range {norm} flex10")
            results.append(r3)

    # ========== CONFIG C: Log/sqrt transform of ranges ==========
    print("=== C: Transformed ranges ===")
    for tn, ti in [('all_grasp', ALLG), ('power', POW)]:
        trials = load_trials(FILES, ti)
        Xf, _ = trial_range_per_task(trials)
        # Log (with shift)
        X_log = np.log(Xf - np.min(Xf, axis=0) + 0.1)  # min-subtract + small offset
        for jn, ji in [('all15', A15), ('no_abd', NAD)]:
            nm = f"dC_{tn}_trialrange_log_{jn}"
            r = export_full(nm, Xf, X_log[:, ji], ji, f"C {tn} trial_range log {jn}")
            results.append(r)
        
        # Square root
        X_sqrt = np.sqrt(Xf - np.min(Xf, axis=0) + 0.01)
        for jn, ji in [('all15', A15), ('no_abd', NAD)]:
            nm = f"dC_{tn}_trialrange_sqrt_{jn}"
            r = export_full(nm, Xf, X_sqrt[:, ji], ji, f"C {tn} trial_range sqrt {jn}")
            results.append(r)
        
        # Robust zscore (median + IQR)
        med = np.median(Xf, axis=0); q75, q25 = np.percentile(Xf, [75,25], axis=0); iqr = q75 - q25; iqr[iqr<1e-6]=1.0
        X_rob = (Xf - med) / iqr
        for jn, ji in [('all15', A15), ('no_abd', NAD)]:
            nm = f"dC_{tn}_trialrange_robust_{jn}"
            r = export_full(nm, Xf, X_rob[:, ji], ji, f"C {tn} trial_range robust {jn}")
            results.append(r)

    # ========== CONFIG D: Mix of static5 + range (two features per task/joint) ==========
    print("=== D: Combined features ===")
    for tn, ti in [('all_grasp', ALLG), ('power', POW)]:
        trials = load_trials(FILES, ti)
        by_t_range = {}; by_t_static = {}
        for t, tid, _ in trials:
            if tid not in by_t_range: by_t_range[tid] = []; by_t_static[tid] = []
            by_t_range[tid].append(np.max(t,axis=0)-np.min(t,axis=0))
            by_t_static[tid].append(np.mean(t[-5:],axis=0))
        X_range = np.array([np.mean(v,axis=0) for v in by_t_range.values()])
        X_static = np.array([np.mean(v,axis=0) for v in by_t_static.values()])
        
        # Feature stack: [range, static5]
        X_comb = np.hstack([X_range, X_static])
        # Generate expanded joint names
        comb_jids_15 = list(range(15))  # range part
        comb_jids_15_2 = list(range(15, 30))  # static part
        
        # But for PCA, we only care about first 15-D component structure
        # Actually let's do PCA on combined, then extract first 15
        X_range_n = X_range; X_static_n = X_static
        for norm in ['none', 'zscore']:
            if norm == 'zscore':
                mr=np.mean(X_range,0); sr=np.std(X_range,0); sr[sr<1e-6]=1.0; X_range_n=(X_range-mr)/sr
                ms=np.mean(X_static,0); ss=np.std(X_static,0); ss[ss<1e-6]=1.0; X_static_n=(X_static-ms)/ss
            
            # PCA on each separately, then check
            for jn, ji in [('all15', A15), ('no_abd', NAD)]:
                nm = f"dD_{tn}_range_{norm}_{jn}"
                r = export_full(nm, X_range, X_range_n[:, ji], ji, f"D {tn} range {norm} {jn}")
                results.append(r)
                
                nm2 = f"dD_{tn}_static_{norm}_{jn}"
                r2 = export_full(nm, X_static, X_static_n[:, ji], ji, f"D {tn} static {norm} {jn}")
                results.append(r2)

    # ========== CONFIG E: per-subject range then aggregate ==========
    print("=== E: Per-subject range ===")
    for tn, ti in [('all_grasp', ALLG), ('power', POW)]:
        trials = load_trials(FILES, ti)
        by_subj_task = {}
        for t, tid, subj in trials:
            k = (subj, tid)
            if k not in by_subj_task: by_subj_task[k] = []
            by_subj_task[k].append(np.max(t,axis=0)-np.min(t,axis=0))
        
        # One (subj, task) sample
        X_subj_task = np.array([np.mean(v,axis=0) for v in by_subj_task.values()])
        print(f"  {tn}: {len(X_subj_task)} subj×task samples")
        
        for jn, ji in [('all15', A15), ('no_abd', NAD)]:
            for norm in ['none', 'zscore']:
                if norm == 'zscore':
                    m=np.mean(X_subj_task,0); s=np.std(X_subj_task,0); s[s<1e-6]=1.0
                    Xn = (X_subj_task - m) / s
                else:
                    Xn = X_subj_task
                nm = f"dE_{tn}_range_subjtask_{norm}_{jn}"
                r = export_full(nm, X_subj_task, Xn[:, ji], ji, f"E {tn} range subj×task {norm} {jn}")
                results.append(r)
    
    return results

def main():
    all_r = deep_search()
    for r in all_r:
        r['score'] = score(r)
    all_r.sort(key=lambda r: r['score'], reverse=True)
    
    print(f"\n{'='*140}")
    print(f"RANKED {len(all_r)} CONFIGURATIONS")
    print(f"{'='*140}")
    print(f"{'#':<3} {'Name':<55} {'PC1':>6} {'PC2':>6} {'Sum':>6} {'S1':>7} {'S2':>7} {'All+':<5} {'Sc':>6}")
    print("-"*140)
    for i, r in enumerate(all_r[:60]):
        st = "★" if r['pc2'] > 10 and abs(r['s1']) > 0.85 and abs(r['s2']) > 0.5 else " "
        ap = "✓" if r['all_pos'] else " "
        print(f"{st}{i+1:<2} {r['name'][:54]:<55} {r['pc1']:>5.1f}% {r['pc2']:>5.1f}% {r['total']:>5.1f}% {r['s1']:>+6.3f} {r['s2']:>+6.3f} {ap:<5} {r['score']:>6.4f}")
    
    clean = [{k:(float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v) for k,v in r.items()} for r in all_r]
    with open(os.path.join(EXP_DIR,'deep_search_results.json'),'w') as f:
        json.dump(clean,f,indent=2)
    
    print(f"\n{'='*80}")
    print("TOP 10 VISUALIZATION")
    print(f"{'='*80}")
    for i, r in enumerate(all_r[:10]):
        print(f"#{i+1}: python code/myoki/synergy_myoki.py --participant {r['name']}")
    print(f"\nDone! {len(all_r)} results.")

if __name__ == '__main__':
    main()
