#!/usr/bin/env python3
""" E3 (Figure 2, Table II, numbers of Sections I and III-C) from the frozen codecs.
Per dataset: eigenvalues lambda(u) (PCA codec variances, descending), rotated-basis variances, original-coordinate variances v(u);
CG(u) = AM(lambda)/GM(lambda), log2 CG, CV^2, Haar deficit delta = 2 CV^2/(d + 2); original-coordinate allocation gain
A = AM(v)/GM(v) and correlation factor GM(v)/GM(lambda) (share of log2 CG); Figure 2(b) terms at B = 3: sigma_i^4 4^{-b_i} for both
pairs (PCA alpha = 1, rotation uniform), normalized by lam_bar^2 4^{-3}; shape constants c0 at B = 3 of each pair from the measured
score-error run (score_error) when present. Output calc/<ds>.json and calc/table2.csv."""
import os, sys, json, csv
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair.config import PREPARED, FROZEN, OUT as OUT_ROOT, RESULTS  # noqa: E402
from pcafair.codec import FairCodec  # noqa: E402
PREP = str(PREPARED)
OUT = f'{OUT_ROOT}/calc'; os.makedirs(OUT, exist_ok=True)
ORDER = ['youtube', 'word2vec', 'openai1536', 'msmarco', 'openai3072', 'gist', 'fashion-mnist', 'audio', 'mnist', 'msong']
def am(x): return float(np.mean(x))
def gm(x): return float(np.exp(np.mean(np.log(x))))
rows = []
for ds in ORDER:
    paths = {p: f'{FROZEN}/codec_{ds}__{p}.npz' for p in ('pca', 'rot', 'orig')}
    if not all(os.path.exists(v) for v in paths.values()): print(ds, 'codecs missing'); continue
    cp, cr, co = (FairCodec.load(paths[p]) for p in ('pca', 'rot', 'orig'))
    lam = np.sort(np.maximum(np.asarray(cp.var, np.float64), 1e-300))[::-1]; d = len(lam); v = np.maximum(np.asarray(co.var, np.float64), 1e-300)
    CG = am(lam) / gm(lam); cv2 = float(np.mean(lam ** 2) / am(lam) ** 2 - 1); delta = 2 * cv2 / (d + 2)
    A = am(v) / gm(v); corr = gm(v) / gm(lam)
    bp = cp.alloc(3, 'alpha1'); br = cr.alloc(3, 'uniform'); lb = am(lam)
    tp = np.sort(np.asarray(cp.var, np.float64) ** 2 * 4.0 ** (-bp))[::-1] / (lb ** 2 * 4.0 ** -3)
    tr = np.sort(np.asarray(cr.var, np.float64) ** 2 * 4.0 ** (-br))[::-1] / (lb ** 2 * 4.0 ** -3)
    info = dict(dataset=ds, d=d, CG_u=round(CG, 4), log2CG_u=round(float(np.log2(CG)), 4), CV2=round(cv2, 4), delta_haar=round(delta, 6),
                A_orig=round(A, 4), corr_factor=round(corr, 4), corr_share_of_log2CG=round(float(np.log2(corr) / np.log2(CG)) if CG > 1 else 0.0, 4),
                fig2a_pca=(np.sort(np.asarray(cp.var))[::-1] / lb).tolist(), fig2a_rot=(np.sort(np.asarray(cr.var))[::-1] / lb).tolist(),
                fig2b_pca=tp.tolist(), fig2b_rot=tr.tolist(), active_pca_B3=int((bp > 0).sum()))
    se = (f'{OUT_ROOT}/score_error/{ds}.csv' if os.path.exists(f'{OUT_ROOT}/score_error/{ds}.csv') else f'{RESULTS}/score_error/{ds}.csv')
    if os.path.exists(se):                                         # shape factor S and the predicted ratio of Eq. (3)
        R = [r for r in csv.DictReader(open(se)) if r['mode'] == 'unified']
        for B in (1, 2, 3, 4, 6, 8):
            rp = [r for r in R if r['pair'] == 'pca' and r['B'] == str(B)]; rr = [r for r in R if r['pair'] == 'rot' and r['B'] == str(B)]
            if not rp or not rr: continue
            S = float(rr[0]['c0']) / float(rp[0]['c0']); meas = float(rr[0]['J_meas']) / float(rp[0]['J_meas'])
            info[f'c0_B{B}_pca'] = float(rp[0]['c0']); info[f'c0_B{B}_rot'] = float(rr[0]['c0']); info[f'S_B{B}'] = round(S, 4)
            info[f'ratio_meas_B{B}'] = round(meas, 4); info[f'ratio_pred_closed_B{B}'] = round(CG ** 2 * (1 + delta), 4)
            info[f'ratio_pred_S_B{B}'] = round(CG ** 2 * (1 + delta) * S, 4)
            info[f'ratio_pred_sum_B{B}'] = round(float(rr[0]['J_hr']) / float(rp[0]['J_hr']), 4)
    json.dump(info, open(f'{OUT}/{ds}.json', 'w'))
    rows.append({k: info[k] for k in ('dataset', 'd', 'CG_u', 'log2CG_u', 'CV2', 'delta_haar', 'A_orig', 'corr_factor', 'corr_share_of_log2CG', 'active_pca_B3')} |
                {k: info.get(k, '') for k in ('c0_B3_pca', 'c0_B3_rot', 'S_B3', 'ratio_meas_B3', 'ratio_pred_closed_B3', 'ratio_pred_S_B3', 'ratio_pred_sum_B3')})
    print(ds, rows[-1], flush=True)
with open(f'{OUT}/table2.csv', 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
