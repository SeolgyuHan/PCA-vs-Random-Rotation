#!/usr/bin/env python3
""" (Figure 2, Table III, checks): per large dataset, as results/calc_fair.py from the frozen codecs: CG(u) of the codec's PCA
variances, CV^2, Haar deficit delta, original-coordinate gain A and correlation factor, Figure 2(a) variances / lambda_bar and 2(b)
terms sigma_i^4 4^{-b_i} / (lambda_bar^2 4^{-3}) at B = 3, shape factor S_B = c0(rot)/c0(pca) of the score-error run and the
predicted ratios; plus the exact CG of pass A (Table III convention, experiments/calc/direction_stats.py), S from the training-sample
constants (c0_tr of the decomposition), the Table III row, and the ground-truth checks against the published lists
(sift100m: big-ann official top-100; msmarco113m: Cohere's cosine top-1000; unit vectors, so the same order up to ties).
Output results/calc/<ds>.json, results/table2_big.csv. Usage: python3 big_calc.py [ds,...] [--smoke]"""
import os, sys, csv, json
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
B.ensure_dirs()
OC = f'{B.OUT}/calc'; os.makedirs(OC, exist_ok=True)
def am(x): return float(np.mean(x))
def gm(x): return float(np.exp(np.mean(np.log(x))))


def gt_check(ds, tag):
    gt = np.load(f'{B.OUT}/{ds}-gt-l2{tag}.npy')[:, :10]; nq = len(gt); dist = np.load(f'{B.OUT}/{ds}-gt-l2-dist{tag}.npy')
    if ds == 'sift100m':
        a = np.fromfile(f'{B.SCAL}/SIFT100M/gt.official.100M.ibin', np.int32); n, k = int(a[0]), int(a[1])
        ref = a[2:2 + n * k].reshape(n, k)[:nq, :10]; src = 'big-ann official top-100 (exact L2 on uint8)'
    elif ds == 'msmarco113m':
        a = np.fromfile(f'{B.SCAL}/MSMARCO100M/gt.ivecs', np.int32); k = int(a[0]); ref = a.reshape(-1, k + 1)[:nq, 1:11]; src = 'Cohere top-1000 by cosine'
    else: return {}
    same = [len(set(gt[i].tolist()) & set(ref[i].tolist())) for i in range(nq)]
    tie10 = int((np.abs(dist[:, 9] - dist[:, 10]) <= 1e-9 * np.abs(dist[:, 9])).sum())
    return dict(gt_check_source=src, gt_check_mean_overlap_at10=round(float(np.mean(same)) / 10, 6), gt_check_identical_sets=int(np.sum(np.array(same) == 10)),
                gt_check_queries=nq, gt_ties_at_rank10=tie10)


def main(names, tag=''):
    rows = []
    for ds in names:
        s = B.spec(ds); name = ds + tag; FZ = B.FROZEN
        cp, cr, co = (B.BigCodec.load(f'{FZ}/codec_{name}__{p}.npz') for p in ('pca', 'rot', 'orig'))
        pa = np.load(f'{B.WORK}/{ds}-passA{tag}.npz'); lam_x = pa['lam']
        lam = np.sort(np.maximum(np.asarray(cp.var, np.float64), 1e-300))[::-1]; d = len(lam); v = np.maximum(np.asarray(co.var, np.float64), 1e-300)
        CG = am(lam) / gm(lam); cv2 = float(np.mean(lam ** 2) / am(lam) ** 2 - 1); delta = 2 * cv2 / (d + 2); A = am(v) / gm(v); corr = gm(v) / gm(lam)
        bp = cp.alloc(3, 'alpha1'); br = cr.alloc(3, 'uniform'); lb = am(lam)
        tp = np.sort(np.asarray(cp.var, np.float64) ** 2 * 4.0 ** (-bp))[::-1] / (lb ** 2 * 4.0 ** -3)
        tr = np.sort(np.asarray(cr.var, np.float64) ** 2 * 4.0 ** (-br))[::-1] / (lb ** 2 * 4.0 ** -3)
        info = dict(dataset=ds, N=int(pa['N']), d=d, nq=int(s['nq']), type=s['type'], CG_exact=round(float(pa['CG']), 4), CG_u=round(CG, 4),
                    log2CG_u=round(float(np.log2(CG)), 4), CG_x_campaign=round(float(np.mean(s['eig_x']) / np.exp(np.mean(np.log(s['eig_x'])))), 4),
                    CV2=round(cv2, 4), delta_haar=round(delta, 6), A_orig=round(A, 4), corr_factor=round(corr, 4),
                    corr_share_of_log2CG=round(float(np.log2(corr) / np.log2(CG)) if CG > 1 else 0.0, 4), ER2=float(pa['ER2']),
                    fig2a_pca=(np.sort(np.asarray(cp.var))[::-1] / lb).tolist(), fig2a_rot=(np.sort(np.asarray(cr.var))[::-1] / lb).tolist(),
                    fig2b_pca=tp.tolist(), fig2b_rot=tr.tolist(), active_pca_B3=int((bp > 0).sum()), lam_exact=lam_x.tolist())
        se = f'{B.OUT}/score_error/{name}.csv'
        if os.path.exists(se):
            R = [r for r in csv.DictReader(open(se)) if r['mode'] == 'unified']
            for Bb in (1, 2, 3, 4, 6, 8):
                rp = [r for r in R if r['pair'] == 'pca' and r['B'] == str(Bb)]; rr = [r for r in R if r['pair'] == 'rot' and r['B'] == str(Bb)]
                if not rp or not rr: continue
                Sv = float(rr[0]['c0']) / float(rp[0]['c0']); meas = float(rr[0]['J_meas']) / float(rp[0]['J_meas'])
                info[f'c0_B{Bb}_pca'] = float(rp[0]['c0']); info[f'c0_B{Bb}_rot'] = float(rr[0]['c0']); info[f'S_B{Bb}'] = round(Sv, 4)
                info[f'ratio_meas_B{Bb}'] = round(meas, 4); info[f'ratio_pred_closed_B{Bb}'] = round(CG ** 2 * (1 + delta), 4)
                info[f'ratio_pred_S_B{Bb}'] = round(CG ** 2 * (1 + delta) * Sv, 4); info[f'ratio_pred_sum_B{Bb}'] = round(float(rr[0]['J_hr']) / float(rp[0]['J_hr']), 4)
        ad = f'{B.OUT}/decomp/{name}.csv'
        if os.path.exists(ad):
            R2 = {(r['pair'], int(r['B'])): r for r in csv.DictReader(open(ad))}
            for Bb in (1, 2, 3, 4, 6, 8):
                if ('pca', Bb) in R2 and ('rot', Bb) in R2:
                    info[f'S_tr_B{Bb}'] = round(float(R2[('rot', Bb)]['c0_tr']) / float(R2[('pca', Bb)]['c0_tr']), 4)
                    info[f'ratio_pred_strict_B{Bb}'] = round(float(R2[('rot', Bb)]['J_A2tr']) / float(R2[('pca', Bb)]['J_A2tr']), 4)
                    info[f'pred_err_bits_B{Bb}'] = round(0.5 * float(np.log2(info.get(f'ratio_meas_B{Bb}', np.nan) / info[f'ratio_pred_strict_B{Bb}'])), 4)
            info['a2_exponent'] = json.load(open(f'{B.OUT}/decomp/{name}.a2.json')).get('exponent')
        fsn = f'{B.OUT}/fullscan/{name}.csv'
        if os.path.exists(fsn):
            for r in csv.DictReader(open(fsn)): info[f'recall_B3_{r["piece"]}'] = float(r['recall_l2']); info[f'recall_tie_B3_{r["piece"]}'] = float(r['recall_l2_tie'])
        info.update(gt_check(ds, tag))
        json.dump(info, open(f'{OC}/{name}.json', 'w'))
        rows.append({k: info.get(k, '') for k in ('dataset', 'N', 'd', 'nq', 'type', 'CG_exact', 'CG_u', 'CG_x_campaign', 'S_tr_B3', 'S_B3', 'ratio_meas_B3',
                                                    'ratio_pred_strict_B3', 'pred_err_bits_B3', 'recall_B3_orig', 'recall_B3_rot', 'recall_B3_pca',
                                                    'gt_check_mean_overlap_at10', 'a2_exponent')})
        B.log(f'calc {name}: ' + ', '.join(f'{k} {v}' for k, v in rows[-1].items() if k not in ('dataset',)))
    if rows:
        with open(f'{B.OUT}/table2_big{tag}.csv', 'w', newline='') as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0].split(',') if args else ['sift100m', 'msmarco10m', 'msmarco113m'], '.smoke' if '--smoke' in sys.argv else '')
