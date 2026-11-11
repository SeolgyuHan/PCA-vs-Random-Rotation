#!/usr/bin/env python3
""" E2 (Figure 3): score error within each query with the frozen codecs, protocol of the repository's
experiments/fullscan/score_error.py: every query x 10,000 database vectors sampled with numpy default_rng(123).
L2 error of the estimate: e = d_hat - d = -2 R <q~, u_hat - u> (the stored norm R is exact).
  J_meas mean over queries of Var_x(e) (the within-query variance: the shift common to a query is removed)
  bq mean over queries of (mean_x e)^2
  J_sep 4 E[R^2] mean_q sum_i q~_i^2 D_i, D_i = measured per-component error variance on the same sample (A1: no cross terms)
  J_hr the high-rate model of Eq. (1): 4 E[R^2] c0 mean_q sum_i q~_i^2 sigma_i^2 4^{-b_i} (b_i = 0: sigma_i^2 with c0 = 1),
           c0 = geometric mean over components with b_i >= 1 of D_i 4^{b_i} / sigma_i^2 (one constant per pair, dataset and B)
  J_noise 4 E[R^2] mean_q sum_{b_i >= 2} q~_i^2 W_i^2 / (12 g_i^2) (the dither part, exact)
Pairs: pca (alpha = 1), rot (uniform); B in {1, 2, 3, 4, 6, 8}; modes unified (dither seed 0) and det.
Output score_error/<ds>.csv. Usage: python3 score_error.py [ds,...] (lock files)"""
import os, sys, csv, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair.config import PREPARED, FROZEN, OUT as OUT_ROOT, RESULTS  # noqa: E402
from pcafair.codec import FairCodec  # noqa: E402
PREP = str(PREPARED)
OUT = f'{OUT_ROOT}/score_error'; os.makedirs(OUT, exist_ok=True); NS = 10_000
ORDER = ['audio', 'fashion-mnist', 'mnist', 'msong', 'word2vec', 'gist', 'youtube', 'msmarco', 'openai1536', 'openai3072']
FIELDS = ['dataset', 'pair', 'mode', 'B', 'J_meas', 'bq', 'J_sep', 'J_hr', 'J_noise', 'c0', 'ER2', 'lam_mean_u', 'd', 'n_active']
def avail(): return int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 1048576


def run(ds):
    t0 = time.time(); db = np.load(f'{PREP}/{ds}-db.npy', mmap_mode='r'); N, d = db.shape
    idx = np.sort(np.random.default_rng(123).choice(N, min(NS, N), replace=False)); X = np.asarray(db[idx]).astype(np.float64)
    R = np.sqrt((X ** 2).sum(1)); ER2 = float(np.mean(R ** 2))
    gt = np.load(f'{PREP}/{ds}-gt-l2.npy'); nq = gt.shape[0]; Q = np.load(f'{PREP}/{ds}-query.npy')[:nq].astype(np.float64)
    rows = []
    for pair, rule in (('pca', 'alpha1'), ('rot', 'uniform')):
        c = FairCodec.load(f'{FROZEN}/codec_{ds}__{pair}.npz'); Y = c.coords(X, R); Qt = c.qcoords(Q); Y32 = Y.astype(np.float32)
        sig2 = np.maximum(np.asarray(c.var, np.float64), 1e-300); q2 = (Qt ** 2).mean(0)
        for B in (1, 2, 3, 4, 6, 8):
            bits = c.alloc(B, rule); pl = c.plan(bits); act = pl['act']; z = c.dither(bits, 0)
            for mode in ('unified', 'det'):
                Yh = (c.decode_unified(Y32, pl, z) if mode == 'unified' else c.decode_fast(Y32, pl, 'det')).astype(np.float64)
                E = Yh - Y; Jq, mq = np.zeros(nq), np.zeros(nq)
                for q0 in range(0, nq, 500):
                    P = -2.0 * R[None, :] * (Qt[q0:q0 + 500] @ E.T); Jq[q0:q0 + 500] = P.var(axis=1); mq[q0:q0 + 500] = P.mean(axis=1)
                Di = E.var(0); Jsep = 4 * ER2 * float(q2 @ Di)
                on = bits >= 1; c0 = float(np.exp(np.mean(np.log(np.maximum(Di[on] * 4.0 ** bits[on] / sig2[on], 1e-300))))) if on.any() else 1.0
                hr = np.where(bits >= 1, c0 * sig2 * 4.0 ** (-bits.astype(np.float64)), sig2); Jhr = 4 * ER2 * float(q2 @ hr)
                m2 = pl['b'] >= 2; gfx = c.g_fix[act[m2], pl['b'][m2]]
                Jn = 4 * ER2 * float(q2[act[m2]] @ (pl['W'][m2] ** 2 / (12 * gfx ** 2))) if m2.any() else 0.0
                rows.append(dict(dataset=ds, pair=pair, mode=mode, B=B, J_meas=f'{Jq.mean():.6e}', bq=f'{(mq ** 2).mean():.6e}', J_sep=f'{Jsep:.6e}',
                                 J_hr=f'{Jhr:.6e}', J_noise=f'{Jn:.6e}', c0=f'{c0:.4f}', ER2=f'{ER2:.6e}', lam_mean_u=f'{sig2.mean():.6e}', d=d, n_active=int(on.sum())))
                print(f'  {ds} {pair} {mode:7s} B{B}: J_meas/J_sep {Jq.mean()/Jsep:.3f} J_meas/J_hr {Jq.mean()/Jhr:.3f} c0 {c0:.3f} [{time.time()-t0:.0f}s]', flush=True)
    with open(f'{OUT}/{ds}.csv.tmp', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    os.replace(f'{OUT}/{ds}.csv.tmp', f'{OUT}/{ds}.csv')


if __name__ == '__main__':
    for ds in (sys.argv[1].split(',') if len(sys.argv) > 1 else ORDER):
        if os.path.exists(f'{OUT}/{ds}.csv'): continue
        while not all(os.path.exists(f'{FROZEN}/codec_{ds}__{p}.json') for p in ('pca', 'rot')): time.sleep(60)
        try: os.close(os.open(f'{OUT}/{ds}.lock', os.O_CREAT | os.O_EXCL))
        except FileExistsError: continue
        while avail() < 8: time.sleep(120)
        try: run(ds)
        finally: os.remove(f'{OUT}/{ds}.lock')
