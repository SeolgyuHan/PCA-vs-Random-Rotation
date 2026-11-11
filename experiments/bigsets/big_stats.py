#!/usr/bin/env python3
"""Statistics of a large dataset or of a prefix of its stream (step 0 of the large-dataset pipeline): the mean of the raw rows and the
PCA axes W of Cov(x) over all N rows (float64 accumulation, eigenvectors in descending order), written to $PCARR_BIG_WORK/<name>-stats.npz
(keys N, d, nq, mean, eig, W, CG) for pcafair.bigsets.spec. Names: sift100m, msmarco10m, msmarco113m, and the prefixes pfx_sift1m,
pfx_sift10m, pfx_msmarco1m (Section V-A). Usage: python3 experiments/bigsets/big_stats.py name [name ...]"""
import os, sys, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

for name in sys.argv[1:]:
    out = f'{B.WORK}/{name}-stats.npz'
    if os.path.exists(out): continue
    s = B.SETS[name]; N, d = s['N'], s['d']; t0 = time.time(); B.gate(f'stats {name}')
    s1 = np.zeros(d); S2 = np.zeros((d, d))
    with threadpool_limits(limits=8):
        for row0, X in B.iter_chunks(s['base'], d, 0, N, B.chunk_rows(d)):
            X64 = X.astype(np.float64); s1 += X64.sum(0); S2 += X64.T @ X64
    mu = s1 / N; C = S2 / N - np.outer(mu, mu); ev, V = np.linalg.eigh(C); order = np.argsort(ev)[::-1]
    eig = np.maximum(ev[order], 0); W = V[:, order].astype(np.float32); pos = eig[eig > 0]
    cg = float(pos.mean() / np.exp(np.mean(np.log(pos))))
    np.savez(out, N=N, d=d, nq=s['nq'], mean=mu, eig=eig, W=W, CG=cg)
    B.log(f'stats {name}: N {N:,}, CG_x {cg:.4f} [{time.time()-t0:.0f}s]')
