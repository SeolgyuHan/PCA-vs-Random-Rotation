#!/usr/bin/env python3
"""Exact statistics of the direction u = x / ||x|| over all database rows of a prepared dataset (x = the prepared row, centered on
the database mean): the eigenvalues of Cov(u) in float64, the coding gain CG = AM / GM of those eigenvalues (Table II), and
||E[u]||^2 (the term that Assumption 5 drops). Every component is kept, near-null ones included (mnist keeps 712 components;
without its six components below 1e-6 of the mean its CG would be 28.8 instead of 31.6), and no floor is applied.
Writes $PCARR_OUT/calc/direction_stats.csv and $PCARR_OUT/calc/eig_u/<dataset>.npy (eigenvalues, descending).
The published values are results/calc/exact_cg.json and results/calc/eig_u/.
Usage: python3 experiments/calc/direction_stats.py [--datasets audio,mnist]"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import numpy as np  # noqa: E402
from pcafair.config import PREPARED, TEN  # noqa: E402
from pcafair.runio import parse_args, write_csv  # noqa: E402

FIELDS = ['dataset', 'N', 'd', 'CG_u', 'log2_CG_u', 'eu_norm2', 'n_below_1e6_of_mean', 'CG_u_without_them']


def am_gm(lam):
    lam = np.asarray(lam, np.float64)
    return float(lam.mean() / np.exp(np.mean(np.log(lam))))


if __name__ == '__main__':
    a = parse_args(__doc__.splitlines()[0], TEN)
    rows = []
    for ds in a.datasets:
        db = np.load(PREPARED / f'{ds}-db.npy', mmap_mode='r'); N, d = db.shape; ch = max(1000, min(100_000, int(2e8 // d)))
        su = np.zeros(d); suu = np.zeros((d, d))
        for r in range(0, N, ch):
            X = np.asarray(db[r:r + ch], np.float64); U = X / np.sqrt((X ** 2).sum(1))[:, None]
            su += U.sum(0); suu += U.T @ U
        m = su / N; lam = np.sort(np.linalg.eigvalsh(suu / N - np.outer(m, m)))[::-1]
        assert lam.min() > 0, (ds, lam.min())
        small = lam < lam.mean() * 1e-6
        os.makedirs(a.out / 'calc' / 'eig_u', exist_ok=True); np.save(a.out / 'calc' / 'eig_u' / f'{ds}.npy', lam)
        rows.append(dict(dataset=ds, N=N, d=d, CG_u=f'{am_gm(lam):.6f}', log2_CG_u=f'{np.log2(am_gm(lam)):.6f}', eu_norm2=f'{float(m @ m):.6f}',
                         n_below_1e6_of_mean=int(small.sum()), CG_u_without_them=f'{am_gm(lam[~small]):.4f}'))
        print(rows[-1], flush=True)
        write_csv(a.out / 'calc' / 'direction_stats.csv', FIELDS, rows, crlf=False)
