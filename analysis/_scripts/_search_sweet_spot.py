"""
SWEET SPOT SEARCH: Find PCA configs with PC1 ~70-80%, PC2 ~10-20%.
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
PS2 = np.array([2,2,2,2, 1,1, 0,0,0, -1,-1,-1, -2,-2,-2], dtype=float)
PS2 = PS2 / np.linalg.norm(PS2)

FID = [1,2,4,5,6,7,9,10,12,13]
NAD = [0,1,2,3,4,5,6,7,9,10,12,13]
A15 = list(range(15))

# Load tasks
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
ALLT = sorted(set(cat.keys())-{0})

# Files
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
    return float(np.dot(rot[:,0], s1)), float(np.dot(rot[:,1], s2)), R

def export(name, X_full, X_j, jids, desc):
    nc = min(len(jids), X_j.shape[0], 10)
    pca = PCA(n_components=nc)
    pca.fit(X_j)
    evr = pca.explained_variance_ratio_
    fc = np.zeros((nc, 15))
    for i in range(nc):
        fc[i, jids] = pca.components_[i]
    fm = np.mean(X_full, axis=0)
    for i in range(nc):
        if fc[i, 1] < 0:
            fc[i] *= -1
    s1, s2, R = pisa_sim(pca.components_, jids)
    p1 = evr[0]*100
    p2 = evr[1]*100 if len(evr) > 1 else 0
    tv = p1 + p2
    lbl = f"{desc} | PC1={p1:.1f}% PC2={p2:.1f}% S1={s1:.3f} S2={s2:.3f}"
    np.savez(os.path.join(EXP_DIR, f'sweet_{name}.npz'),
             components=fc, mean=fm, explained_variance=evr, data_label=lbl,
             s1_sim=s1, s2_sim=s2)
    np.savez(os.path.join(SAVE_DIR, f'pca_myoki_{name}.npz'),
             components=fc, mean=fm, explained_variance=evr, data_label=lbl)
    return {'name':name, 'pc1':p1, 'pc2':p2, 'total':tv, 's1':s1, 's2':s2, 'nj':len(jids)}

def run_approach1():
    """Trial-level PCA - each trial is one sample."""
    print("\n=== APPROACH 1: TRIAL LEVEL ===")
    R = []
    TS = {'power':POW, 'all_grasp':ALLG, 'power_prec':sorted(set(POW)|set(PRE))}
    PREPROC = {'static5':lambda t: np.mean(t[-5:],axis=0),
               'trial_mean':lambda t: np.mean(t,axis=0),
               'trial_range':lambda t: np.max(t,axis=0)-np.min(t,axis=0)}
    JS = {'all15':A15, 'no_abd':NAD, 'flex10':FID}
    for tn, ti in TS.items():
        trials = load_trials(FILES, ti)
        print(f"  {tn}: {len(trials)} trials")
        for pn, pf in PREPROC.items():
            Xf = np.array([pf(t) for t,_,_ in trials if np.all(np.isfinite(pf(t)))])
            if len(Xf) < 5: continue
            for jn, ji in JS.items():
                for norm in ['none', 'zscore']:
                    if norm == 'zscore':
                        m = np.mean(Xf,0); s = np.std(Xf,0); s[s<1e-6]=1.0
                        Xn = (Xf - m) / s
                    else:
                        Xn = Xf
                    Xj = Xn[:, ji]
                    nm = f"s1_{tn}_{pn}_{jn}_{norm}"
                    r = export(nm, Xf, Xj, ji, f"Trial {tn} {pn} {jn} {norm}")
                    R.append(r)
    return R

def run_approach2():
    """Subject x Task approach."""
    print("\n=== APPROACH 2: SUBJ x TASK ===")
    R = []
    TS = {'power':POW, 'all_grasp':ALLG, 'power_prec':sorted(set(POW)|set(PRE))}
    PREPROC = {'static5':lambda t: np.mean(t[-5:],axis=0),
               'trial_mean':lambda t: np.mean(t,axis=0)}
    JS = {'all15':A15, 'no_abd':NAD, 'flex10':FID}
    for tn, ti in TS.items():
        trials = load_trials(FILES, ti)
        for pn, pf in PREPROC.items():
            by_sk = {}
            for t, tid, subj in trials:
                k = (subj, tid)
                if k not in by_sk: by_sk[k] = []
                by_sk[k].append(pf(t))
            Xf = np.array([np.mean(v,0) for v in by_sk.values()])
            print(f"  {tn} {pn}: {len(Xf)} samples")
            for jn, ji in JS.items():
                for norm in ['none', 'zscore']:
                    if norm == 'zscore':
                        m = np.mean(Xf,0); s = np.std(Xf,0); s[s<1e-6]=1.0
                        Xn = (Xf - m) / s
                    else:
                        Xn = Xf
                    Xj = Xn[:, ji]
                    nm = f"s2_{tn}_{pn}_{jn}_{norm}"
                    r = export(nm, Xf, Xj, ji, f"SubjxTask {tn} {pn} {jn} {norm}")
                    R.append(r)
    return R

def run_approach3():
    """One sample per task."""
    print("\n=== APPROACH 3: ONE PER TASK ===")
    R = []
    TS = {'power':POW, 'all_grasp':ALLG, 'power_prec':sorted(set(POW)|set(PRE))}
    PREPROC = {'static5':lambda t: np.mean(t[-5:],axis=0),
               'trial_mean':lambda t: np.mean(t,axis=0),
               'trial_range':lambda t: np.max(t,axis=0)-np.min(t,axis=0),
               'trial_std':lambda t: np.std(t,axis=0)}
    JS = {'all15':A15, 'no_abd':NAD, 'flex10':FID}
    for tn, ti in TS.items():
        trials = load_trials(FILES, ti)
        for pn, pf in PREPROC.items():
            by_t = {}
            for t, tid, _ in trials:
                if tid not in by_t: by_t[tid] = []
                by_t[tid].append(pf(t))
            Xf = np.array([np.mean(v,0) for v in by_t.values()])
            print(f"  {tn} {pn}: {len(Xf)} tasks")
            for jn, ji in JS.items():
                for norm in ['none', 'zscore', 'range']:
                    if norm == 'zscore':
                        m = np.mean(Xf,0); s = np.std(Xf,0); s[s<1e-6]=1.0
                        Xn = (Xf - m) / s
                    elif norm == 'range':
                        mn = np.min(Xf,0); mx = np.max(Xf,0); rg = mx-mn; rg[rg<1e-6]=1.0
                        Xn = (Xf - mn) / rg
                    else:
                        Xn = Xf
                    Xj = Xn[:, ji]
                    nm = f"s3_{tn}_{pn}_{jn}_{norm}"
                    r = export(nm, Xf, Xj, ji, f"OnePerTask {tn} {pn} {jn} {norm}")
                    R.append(r)
    return R

def score(r):
    p1 = r['pc1']/100; p2 = r['pc2']/100; t = r['total']/100
    s1 = abs(r['s1']); s2 = abs(r['s2'])
    p1s = max(0, 1 - abs(p1-0.75)*3)
    p2s = max(0, 1 - abs(p2-0.15)*5)
    ts = max(0, 1 - abs(t-0.90)*3)
    s1s = max(0, 1 - abs(s1-0.95)*3)
    s2s = max(0, 1 - abs(s2-0.75)*3)
    pb = min(p2*5, 0.5)
    return p1s*0.15 + p2s*0.25 + ts*0.15 + s1s*0.15 + s2s*0.20 + pb*0.10

def main():
    all_r = []
    all_r.extend(run_approach1())
    all_r.extend(run_approach2())
    all_r.extend(run_approach3())
    for r in all_r:
        r['score'] = score(r)
    all_r.sort(key=lambda r: r['score'], reverse=True)
    print(f"\n{'='*120}")
    print(f"RANKED ({len(all_r)} total)")
    print(f"{'='*120}")
    print(f"{'#':<3} {'Name':<50} {'PC1':>6} {'PC2':>6} {'Sum':>6} {'S1':>7} {'S2':>7} {'Sc':>6}")
    print("-"*120)
    for i, r in enumerate(all_r[:50]):
        st = "★" if r['pc2'] > 10 and abs(r['s1']) > 0.85 and abs(r['s2']) > 0.5 else " "
        print(f"{st}{i+1:<2} {r['name'][:49]:<50} {r['pc1']:>5.1f}% {r['pc2']:>5.1f}% {r['total']:>5.1f}% {r['s1']:>+6.3f} {r['s2']:>+6.3f} {r['score']:>6.4f}")
    clean = [{k:(float(v) if isinstance(v,(np.floating,)) else int(v) if isinstance(v,(np.integer,)) else v) for k,v in r.items()} for r in all_r]
    with open(os.path.join(EXP_DIR,'sweet_spot_results.json'),'w') as f:
        json.dump(clean,f,indent=2)
    print(f"\n{'='*70}")
    print("TOP 5 VISUALIZATION COMMANDS")
    print(f"{'='*70}")
    for r in all_r[:5]:
        print(f"  python code/myoki/synergy_myoki.py --participant {r['name']}")
    print(f"\nDone! {len(all_r)} results saved.")

if __name__ == '__main__':
    main()
