#!/usr/bin/env python3
"""Where does the prediction of the score-error ratio miss? Exact telescoping of the measured within-query variance
J_meas (E2 protocol: frozen codecs, unified quantizer with dither seed 0, every query x 10,000 database rows, rng 123) into the
model assumptions, per dataset, pair (pca alpha=1 / rot uniform) and budget B:
  J_meas = J_A2 * (J_hr/J_A2) [A2: query weights] * (J_sep/J_hr) [A3': one c0 per basis] * (J_full/J_sep) [A1 kappa: cross terms]
           * (J_meas/J_full) [A0: norm independent of the error]
  J_full = 4 E[R^2] mean_q q~' Cov(e) q~ (covariance model, R independent of e)
  J_sep = 4 E[R^2] mean_q sum_i q~_i^2 D_i (diagonal of the same covariance; = E2's J_sep)
  J_hr = 4 E[R^2] c0 mean_q sum_i q~_i^2 s_i 4^{-b_i} (one constant per basis; b_i = 0: s_i, constant 1; = E2's J_hr)
  J_A2 = 4 E[R^2] c0 k sum_i s_i^2 4^{-b_i} (A2 weights k s_i, k = sum_i E q~_i^2 / sum_i s_i; b_i = 0: s_i^2)
Strict "before encoding" variants: c0 from the codec's training sample (rows rng 7, min(200k, 1.2e8/d)) instead of the evaluated
sample -> J_A2tr; query-sample variant J_qh (weights from the first half of the queries, c0 from the training sample) compared with
J_meas on the second half -> Jm_h2. Also the A2 exponent (slope of log E q~_i^2 on log lambda_i in the PCA basis).
Output decomp/<ds>.csv and <ds>.a2.json. Usage: python3 decomp.py [ds,...]"""
import os, sys, csv, json, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair.config import PREPARED, FROZEN, OUT as OUT_ROOT, RESULTS  # noqa: E402
from pcafair.codec import FairCodec  # noqa: E402
PREP = str(PREPARED)
OUT = f'{OUT_ROOT}/decomp'; os.makedirs(OUT, exist_ok=True); NS = 10_000; BUDGETS = (1, 2, 3, 4, 6, 8)
ORDER = ['audio', 'fashion-mnist', 'mnist', 'msong', 'word2vec', 'gist', 'youtube', 'msmarco', 'openai1536', 'openai3072']
F = ['dataset', 'pair', 'B', 'J_meas', 'J_full', 'J_sep', 'J_hr', 'J_A2', 'J_A2tr', 'J_qh', 'Jm_h2', 'c0', 'c0_tr', 'k_A2', 'E2_J_meas', 'E2_J_hr']


def c0_of(D, bits, s2):
    on = bits >= 1
    return float(np.exp(np.mean(np.log(np.maximum(D[on] * 4.0 ** bits[on] / s2[on], 1e-300))))) if on.any() else 1.0


def model(c0, w, s2, bits): return float(w @ np.where(bits >= 1, c0 * s2 * 4.0 ** (-bits.astype(np.float64)), s2))


def run(ds):
    t0 = time.time(); db = np.load(f'{PREP}/{ds}-db.npy', mmap_mode='r'); N, d = db.shape
    idx = np.sort(np.random.default_rng(123).choice(N, min(NS, N), replace=False)); X = np.asarray(db[idx]).astype(np.float64)
    R = np.sqrt((X ** 2).sum(1)); ER2 = float(np.mean(R ** 2))
    gt = np.load(f'{PREP}/{ds}-gt-l2.npy'); nq = gt.shape[0]; Q = np.load(f'{PREP}/{ds}-query.npy')[:nq].astype(np.float64)
    e2 = {(r['pair'], r['mode'], int(r['B'])): r for r in csv.DictReader(open((f'{OUT_ROOT}/score_error/{ds}.csv' if os.path.exists(f'{OUT_ROOT}/score_error/{ds}.csv') else f'{RESULTS}/score_error/{ds}.csv')))}
    h = nq // 2; rows = []; a2 = {}
    for pair, rule in (('pca', 'alpha1'), ('rot', 'uniform')):
        c = FairCodec.load(f'{FROZEN}/codec_{ds}__{pair}.npz'); Y = c.coords(X, R); Qt = c.qcoords(Q); Y32 = Y.astype(np.float32)
        s2 = np.maximum(np.asarray(c.var, np.float64), 1e-300); q2 = (Qt ** 2).mean(0); q2a = (Qt[:h] ** 2).mean(0); k = float(q2.sum() / s2.sum())
        if pair == 'pca':                                    # A2 exponent on the PCA basis (components above 1e-8 of the largest)
            keep = s2 > 1e-8 * s2.max(); sl = np.polyfit(np.log(s2[keep]), np.log(np.maximum(q2[keep], 1e-300)), 1)[0]
            a2 = dict(dataset=ds, exponent=round(float(sl), 4), n_comp=int(keep.sum()), nq=nq)
        St = c.ensure_sample().astype(np.float32)               # the codec's training sample (rng 7) in coding coordinates
        for B in BUDGETS:
            bits = c.alloc(B, rule); pl = c.plan(bits); z = c.dither(bits, 0); bf = bits.astype(np.float64)
            E = c.decode_unified(Y32, pl, z).astype(np.float64) - Y
            Jq = np.zeros(nq)
            for q0 in range(0, nq, 500):
                P = -2.0 * R[None, :] * (Qt[q0:q0 + 500] @ E.T); Jq[q0:q0 + 500] = P.var(axis=1)
            Ec = E - E.mean(0); Sig = Ec.T @ Ec / len(Ec); D = np.diag(Sig).copy()
            Jfull = 4 * ER2 * float(np.mean(np.einsum('qi,qi->q', Qt @ Sig, Qt)))
            Jsep = 4 * ER2 * float(q2 @ D); c0 = c0_of(D, bf, s2)
            Jhr = 4 * ER2 * model(c0, q2, s2, bf); JA2 = 4 * ER2 * model(c0, k * s2, s2, bf)
            Et = c.decode_unified(St, pl, z).astype(np.float64) - St.astype(np.float64); c0t = c0_of(Et.var(0), bf, s2)
            JA2t = 4 * ER2 * model(c0t, k * s2, s2, bf); Jqh = 4 * ER2 * model(c0t, q2a, s2, bf)
            r2 = e2.get((pair, 'unified', B), {})
            rows.append(dict(dataset=ds, pair=pair, B=B, J_meas=f'{Jq.mean():.6e}', J_full=f'{Jfull:.6e}', J_sep=f'{Jsep:.6e}', J_hr=f'{Jhr:.6e}',
                             J_A2=f'{JA2:.6e}', J_A2tr=f'{JA2t:.6e}', J_qh=f'{Jqh:.6e}', Jm_h2=f'{Jq[h:].mean():.6e}', c0=f'{c0:.4f}', c0_tr=f'{c0t:.4f}',
                             k_A2=f'{k:.6e}', E2_J_meas=r2.get('J_meas', ''), E2_J_hr=r2.get('J_hr', '')))
            print(f'  {ds} {pair} B{B}: meas/E2 {Jq.mean()/float(r2.get("J_meas", "nan")):.4f} | A0 {Jq.mean()/Jfull:.3f} kappa {Jfull/Jsep:.3f} '
                  f'A3p {Jsep/Jhr:.3f} A2 {Jhr/JA2:.3f} | c0 {c0:.3f} c0_tr {c0t:.3f} [{time.time()-t0:.0f}s]', flush=True)
        del St
    with open(f'{OUT}/{ds}.csv.tmp', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=F); w.writeheader(); w.writerows(rows)
    os.replace(f'{OUT}/{ds}.csv.tmp', f'{OUT}/{ds}.csv'); json.dump(a2, open(f'{OUT}/{ds}.a2.json', 'w'))


if __name__ == '__main__':
    for ds in (sys.argv[1].split(',') if len(sys.argv) > 1 else ORDER):
        if os.path.exists(f'{OUT}/{ds}.csv'): continue
        while int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 1048576 < 15: time.sleep(60)
        run(ds)
