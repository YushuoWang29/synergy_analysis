"""
FINALIZE BEST RESULTS: Export properly formatted NPZs and generate summary.
Re-create the best configs with correct 15-D mean, handle all joint subsets.
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
CYL = sorted([t for t,a in cat.items() if 'cylindrical' in a and 'none_action' not in a])
SPH = sorted([t for t,a in cat.items() if 'spherical' in a and 'none_action' not in a])

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

def compute_best(name, task_ids, preproc_fn, joint_set, norm, desc):
    """Computes PCA with proper 15-D mean regardless of joint subset."""
    trials = load_trials(FILES, task_ids)
    by_t = {}
    for t, tid, _ in trials:
        if tid not in by_t: by_t[tid] = []
        by_t[tid].append(preproc_fn(t))
    
    # Full 15-D data
    X_full = np.array([np.mean(v, axis=0) for v in by_t.values()])
    
    # Normalize
    if norm == 'none':
        X_n = X_full.copy()
    elif norm == 'zscore':
        m = np.mean(X_full, 0); s = np.std(X_full, 0); s[s<1e-6] = 1.0
        X_n = (X_full - m) / s
    elif norm == 'robust':
        med = np.median(X_full, 0)
        q75, q25 = np.percentile(X_full, [75, 25], axis=0)
        iqr = q75 - q25; iqr[iqr<1e-6] = 1.0
        X_n = (X_full - med) / iqr
    elif norm == 'range':
        mn = np.min(X_full, 0); mx = np.max(X_full, 0); rg = mx - mn; rg[rg<1e-6] = 1.0
        X_n = (X_full - mn) / rg
    elif norm == 'log':
        X_n = np.log(X_full - np.min(X_full, 0) + 0.1)
    else:
        X_n = X_full.copy()
    
    # Select joints
    X_j = X_n[:, joint_set]
    
    # PCA
    nc = min(len(joint_set), X_j.shape[0], 10)
    pca = PCA(n_components=nc)
    pca.fit(X_j)
    evr = pca.explained_variance_ratio_
    
    # Build full 15-D components (padded with zeros for unused joints)
    fc = np.zeros((nc, 15))
    for i in range(nc):
        fc[i, joint_set] = pca.components_[i]
    
    # Mean is ALWAYS 15-D
    fm = np.mean(X_full, axis=0)
    
    # Sign-flip
    for i in range(nc):
        if fc[i, 1] < 0:
            fc[i] *= -1
    
    s1, s2, R, rot = pisa_sim(pca.components_, joint_set)
    p1 = evr[0]*100
    p2 = evr[1]*100 if len(evr) > 1 else 0
    tv = p1 + p2
    all_pos = bool(np.all(pca.components_[0] > 0))
    
    # Save
    np.savez(os.path.join(SAVE_DIR, f'pca_myoki_{name}.npz'),
             components=fc, mean=fm, explained_variance=evr, data_label=desc)
    
    return {'name':name, 'pc1':p1, 'pc2':p2, 'total':tv, 's1':s1, 's2':s2, 'all_pos':all_pos,
            'n_tasks':X_j.shape[0], 'n_joints':len(joint_set)}

def main():
    configs = []
    
    # 1. ALL GRASP - trial_range (best overall)
    # One sample per task: trial range (max-min), mean across trials per task
    for task_name, task_ids in [('all_grasp', ALLG), ('power_grasp', POW), ('power_prec', sorted(set(POW)|set(PRE)))]:
        for norm in ['none', 'zscore']:
            for jn, ji in [('all15', A15), ('no_abd', NAD)]:
                n = f"best_{task_name}_trialrange_{norm}_{jn}"
                desc = f"MyoKi {task_name} trial_range {norm} {jn}"
                r = compute_best(n, task_ids, lambda t: np.max(t,0)-np.min(t,0), ji, norm, desc)
                configs.append(r)
    
    # 2. ALL GRASP - static5
    for task_name, task_ids in [('all_grasp', ALLG), ('power_grasp', POW)]:
        for norm in ['none', 'zscore']:
            for jn, ji in [('all15', A15), ('no_abd', NAD)]:
                n = f"best_{task_name}_static5_{norm}_{jn}"
                desc = f"MyoKi {task_name} static5 {norm} {jn}"
                r = compute_best(n, task_ids, lambda t: np.mean(t[-5:],0), ji, norm, desc)
                configs.append(r)
    
    # 3. ALL GRASP - trial_mean
    for task_name, task_ids in [('all_grasp', ALLG), ('power_grasp', POW)]:
        for norm in ['none', 'zscore']:
            for jn, ji in [('all15', A15), ('no_abd', NAD)]:
                n = f"best_{task_name}_trialmean_{norm}_{jn}"
                desc = f"MyoKi {task_name} trial_mean {norm} {jn}"
                r = compute_best(n, task_ids, lambda t: np.mean(t,0), ji, norm, desc)
                configs.append(r)
    
    # 4. ALL GRASP - trial_std
    for task_name, task_ids in [('all_grasp', ALLG), ('power_grasp', POW)]:
        for norm in ['none', 'zscore']:
            for jn, ji in [('all15', A15), ('no_abd', NAD)]:
                n = f"best_{task_name}_trialstd_{norm}_{jn}"
                desc = f"MyoKi {task_name} trial_std {norm} {jn}"
                r = compute_best(n, task_ids, lambda t: np.std(t,0), ji, norm, desc)
                configs.append(r)
    
    # 5. Subj x Task - trial_range (one per subject-task combo)
    for task_name, task_ids in [('all_grasp', ALLG), ('power_grasp', POW)]:
        for norm in ['none', 'zscore']:
            for jn, ji in [('all15', A15), ('no_abd', NAD)]:
                n = f"best_{task_name}_range_subjtask_{norm}_{jn}"
                desc = f"MyoKi {task_name} range subj×task {norm} {jn}"
                # Different preproc: one per subj×task
                trials = load_trials(FILES, task_ids)
                by_sk = {}
                for t, tid, s in trials:
                    k = (s, tid)
                    if k not in by_sk: by_sk[k] = []
                    by_sk[k].append(np.max(t,0)-np.min(t,0))
                X_full_subj = np.array([np.mean(v,0) for v in by_sk.values()])
                # Normalize
                if norm == 'zscore':
                    m = np.mean(X_full_subj, 0); s = np.std(X_full_subj, 0); s[s<1e-6] = 1.0
                    X_n = (X_full_subj - m) / s
                else:
                    X_n = X_full_subj.copy()
                X_j = X_n[:, ji]
                nc = min(len(ji), X_j.shape[0], 10)
                pca = PCA(n_components=nc); pca.fit(X_j)
                evr = pca.explained_variance_ratio_
                fc = np.zeros((nc, 15))
                for i in range(nc): fc[i, ji] = pca.components_[i]
                fm = np.mean(X_full_subj, axis=0)
                for i in range(nc):
                    if fc[i, 1] < 0: fc[i] *= -1
                s1, s2, _, _ = pisa_sim(pca.components_, ji)
                p1 = evr[0]*100; p2 = evr[1]*100 if len(evr)>1 else 0
                np.savez(os.path.join(SAVE_DIR, f'pca_myoki_{n}.npz'),
                         components=fc, mean=fm, explained_variance=evr, data_label=desc)
                configs.append({'name':n, 'pc1':p1, 'pc2':p2, 'total':p1+p2, 's1':s1, 's2':s2,
                                'all_pos':bool(np.all(pca.components_[0]>0)), 'n_tasks':X_j.shape[0], 'n_joints':len(ji)})
    
    # Score and rank
    def score(r):
        p1=r['pc1']/100; p2=r['pc2']/100; t=r['total']/100; s1=abs(r['s1']); s2=abs(r['s2'])
        p1s=max(0,1-abs(p1-0.75)*3); p2s=max(0,1-abs(p2-0.15)*5)
        ts=max(0,1-abs(t-0.90)*3); s1s=max(0,1-abs(s1-0.95)*3)
        s2s=max(0,1-abs(s2-0.75)*3); pb=min(p2*5,0.5)
        ap = 0.1 if r.get('all_pos',False) else 0.0
        return p1s*0.12+p2s*0.20+ts*0.12+s1s*0.15+s2s*0.20+pb*0.10+ap*0.11
    
    for r in configs: r['score'] = score(r)
    configs.sort(key=lambda r: r['score'], reverse=True)
    
    # Save manifest
    clean = []
    for r in configs:
        cr = {}
        for k,v in r.items():
            if isinstance(v,np.floating): cr[k]=float(v)
            elif isinstance(v,np.integer): cr[k]=int(v)
            elif isinstance(v,np.bool_): cr[k]=bool(v)
            else: cr[k]=v
        clean.append(cr)
    
    with open(os.path.join(SAVE_DIR,'best_results_manifest.json'),'w') as f:
        json.dump(clean,f,indent=2)
    
    # Summary
    print(f"\n{'='*170}")
    print(f"BEST PCA RESULTS — MyoKi Dataset — All {len(FILES)} subjects")
    print(f"{'='*170}")
    print(f"{'#':<3} {'Name':<48} {'PC1':>6} {'PC2':>6} {'Sum':>6} {'S1_proc':>8} {'S2_proc':>8} {'All+':<5} {'Tasks':>6} {'Jts':>4}")
    print("-"*170)
    
    for i, r in enumerate(configs[:25]):
        st = "★" if r['pc2']>10 and abs(r['s1'])>0.85 and abs(r['s2'])>0.5 else " "
        ap = "✓" if r.get('all_pos',False) else " "
        print(f"{st}{i+1:<2} {r['name'][:47]:<48} {r['pc1']:>5.1f}% {r['pc2']:>5.1f}% {r['total']:>5.1f}% {abs(r['s1']):>+7.4f} {abs(r['s2']):>+7.4f} {ap:<5} {r.get('n_tasks',0):>6} {r.get('n_joints',0):>4}")
    
    print(f"\n{'='*80}")
    print("TOP 10 — VISUALIZATION COMMANDS")
    print(f"{'='*80}")
    for i, r in enumerate(configs[:10]):
        print(f"  {i+1}. python code/myoki/synergy_myoki.py --participant {r['name']}")
    
    print(f"\n✅ {len(configs)} configs exported to PCA_results/MyoKi/")
    print(f"✅ Manifest: PCA_results/MyoKi/best_results_manifest.json")

if __name__ == '__main__':
    main()
