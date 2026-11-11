#!/usr/bin/env python3
"""score error of the two designs at budgets 5 and 7 (unified mode) for the large datasets, with exactly the first part of
big_score.py (score_error protocol: every query x 10,000 database rows sampled with numpy default_rng(123)). Writes only
results/score_error_b57/<ds>.csv; never touches score_error/ or decomp/. Usage: python3 big_score_b57.py ds[,ds]"""
import os, sys, csv, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
NS = 10_000; BUDGETS = (5, 7); O1 = f'{B.OUT}/score_error_b57'; os.makedirs(O1, exist_ok=True)
F1 = ['dataset', 'pair', 'mode', 'B', 'J_meas', 'bq', 'ER2', 'lam_mean_u', 'd', 'n_active']


def main(ds):
    B.wait_mem(8, f'score b57 {ds}'); t0 = time.time(); s = B.spec(ds); N = int(np.load(f'{B.WORK}/{ds}-passA.npz')['N']); d = s['d']
    idx = np.sort(np.random.default_rng(123).choice(N, min(NS, N), replace=False))
    raw = B.read_rows(s['base'], d, idx); xr = raw.astype(np.float64) - s['mu']
    X = (xr @ s['W']).astype(np.float32).astype(np.float64); del xr
    R = np.sqrt((X ** 2).sum(1)); ER2 = float(np.mean(R ** 2))
    Q = B.prepared_queries(s, B.read_queries(s))[0]; nq = len(Q); rows = []
    for pair, rule in (('pca', 'alpha1'), ('rot', 'uniform')):
        c = B.BigCodec.load(f'{B.FROZEN}/codec_{ds}__{pair}.npz'); Y = c.coords(X, R); Qt = c.qcoords(Q); Y32 = Y.astype(np.float32)
        sig2 = np.maximum(np.asarray(c.var, np.float64), 1e-300)
        for Bb in BUDGETS:
            bits = c.alloc(Bb, rule); pl = c.plan(bits); z = c.dither(bits, 0)
            E = c.decode_unified(Y32, pl, z).astype(np.float64) - Y; Jq, mq = np.zeros(nq), np.zeros(nq)
            for q0 in range(0, nq, 500):
                P = -2.0 * R[None, :] * (Qt[q0:q0 + 500] @ E.T); Jq[q0:q0 + 500] = P.var(axis=1); mq[q0:q0 + 500] = P.mean(axis=1)
            rows.append(dict(dataset=ds, pair=pair, mode='unified', B=Bb, J_meas=f'{Jq.mean():.6e}', bq=f'{(mq ** 2).mean():.6e}', ER2=f'{ER2:.6e}',
                             lam_mean_u=f'{sig2.mean():.6e}', d=d, n_active=int((bits >= 1).sum())))
            print(f'  {ds} {pair} B{Bb}: J_meas {Jq.mean():.4e} [{time.time()-t0:.0f}s]', flush=True)
    with open(f'{O1}/{ds}.csv.tmp', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=F1); w.writeheader(); w.writerows(rows)
    os.replace(f'{O1}/{ds}.csv.tmp', f'{O1}/{ds}.csv')


if __name__ == '__main__':
    for ds in sys.argv[1].split(','): main(ds)
