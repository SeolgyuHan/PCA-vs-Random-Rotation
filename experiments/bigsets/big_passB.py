#!/usr/bin/env python3
""" pass B (Figure 1): one streaming pass per group, full scan of every query over the whole database with the frozen codecs
at B = 3, unified quantizer (dither seed 0), as results/fullscan/fair_fullscan_v2.py: arms pca (alpha = 1), rot (uniform),
orig (original coordinates, alpha = 1). Score R^2 - 2 R <q~, u_hat> (float32, query blocks of 256), per-query top-10.
Recall@10 strict and tie-aware against pass A's ground truth (tie check by the exact float64 L2 of the returned rows).
Checkpoints about every 10 minutes; guards of pcafair.bigsets between chunks. Thread count: 10 while E5 runs, 20 after E7 is done.
Output results/fullscan/<ds>__<arm>.csv + ids_<ds>.npz. Usage: python3 big_passB.py <group> [--rows-max R]"""
import os, sys, csv, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
import nu_taskq as NU  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
f = NU.f; K = 10; QB = 64; ARMS = (('pca', 'alpha1', 1.0), ('rot', 'uniform', 0.0), ('orig', 'alpha1', 1.0))
FS = f'{B.OUT}/fullscan'; os.makedirs(FS, exist_ok=True)
FIELDS = ['dataset', 'piece', 'mode', 'B', 'alpha', 'recall_l2', 'recall_l2_tie', 'n_active', 'max_bits', 'realized_bits', 'N', 'nq']


def threads_now(): return 20 if B.e7_done() else 10


def main(grp, rmax=None):
    tag = '.smoke' if rmax else ''; names = B.GROUPS[grp]; S = {n: B.spec(n) for n in names}
    if rmax:
        for n in names: S[n]['N'] = min(S[n]['N'], rmax if n != 'msmarco10m' else rmax // 2)
    d = S[names[0]]['d']; base = S[names[0]]['base']; Nmax = max(S[n]['N'] for n in names); CH = B.chunk_rows(d) // (2 if d <= 256 else 1)
    Q = B.read_queries(S[names[0]]); nq = len(Q)
    V = {}
    for n in names:
        s = S[n]; Qp, Qr = B.prepared_queries(s, Q)
        for arm, rule, a in ARMS:
            c = B.BigCodec.load(f'{B.FROZEN}/codec_{n}{tag}__{arm}.npz'); bits = c.alloc(3, rule); pl = c.plan(bits)
            c.M32 = c.M.astype(np.float32) if c.M is not None else None
            Qa = c.qcoords(Qr if arm == 'orig' else Qp).astype(np.float32)
            V[(n, arm)] = dict(c=c, bits=bits, pl=pl, z=c.dither(bits, 0), Qa=Qa, top=f.Top(nq, False), a=a)
    ck = f'{B.WORK}/{grp}-passB{tag}.ckpt.npz'; z = B.load_ckpt(ck); pos = 0
    if z is not None:
        pos = int(z['pos'])
        for (n, arm), v in V.items(): v['top'].s[:] = z[f'{n}|{arm}|s']; v['top'].i[:] = z[f'{n}|{arm}|i']
    cuts = sorted(set(S[n]['N'] for n in names)); t0 = time.time(); last = time.time(); nthr = threads_now()
    B.log(f'pass B {grp}{tag}: start at row {pos:,} of {Nmax:,} ({len(V)} arms, {nq} queries, threads {nthr})')
    pool = ThreadPoolExecutor(max_workers=16); threadpool_limits(limits=nthr)
    for row0, X in B.iter_chunks(base, d, pos, Nmax, CH, cuts):
        B.gate(f'pass B {grp}')
        if threads_now() != nthr: nthr = threads_now(); threadpool_limits(limits=nthr); B.log(f'pass B {grp}: threads {nthr}')
        n = X.shape[0]
        for nm in names:
            if row0 >= S[nm]['N']: continue
            Up, Rp, Ur, Rr = B.prepared(S[nm], X)
            for arm, rule, a in ARMS:
                v = V[(nm, arm)]; c = v['c']
                if arm == 'orig': Y32, R = Ur, Rr
                else: Y32, R = Up @ c.M32, Rp
                Rs = R.astype(np.float32); nxs = (R.astype(np.float64) ** 2).astype(np.float32)
                Yh = B.decode_par(c, Y32, v['pl'], v['z'], pool, min(nthr, 16))
                Sf = v['Qa'] @ Yh.T; del Yh                                          # (nq, n) float32, then per query block as fullscan
                T = v['top']
                list(pool.map(lambda q0: T.update(q0, nxs[None, :] - np.float32(2.0) * Rs[None, :] * Sf[q0:q0 + QB], row0), range(0, nq, QB))); del Sf
            del Up, Ur
        pos = row0 + n
        if time.time() - last > 600 or pos >= Nmax:
            arr = dict(pos=np.array(pos))
            for (nm, arm), v in V.items(): arr[f'{nm}|{arm}|s'] = v['top'].s; arr[f'{nm}|{arm}|i'] = v['top'].i
            B.save_ckpt(ck, **arr); last = time.time(); print(f'  pass B {grp}{tag}: {pos:,}/{Nmax:,} rows [{time.time()-t0:.0f}s]', flush=True)
    pool.shutdown()
    for nm in names:
        s = S[nm]; gt = np.load(f'{B.OUT}/{nm}-gt-l2{tag}.npy')[:, :K]; thr = np.load(f'{B.OUT}/{nm}-gt-l2-dist{tag}.npy')[:, K - 1]
        Q64 = Q.astype(np.float64); rows, ids_all = [], {}
        for arm, rule, a in ARMS:
            v = V[(nm, arm)]; ids = v['top'].i.astype(np.int64); ids_all[arm] = ids
            strict = float(np.mean([len(set(ids[i].tolist()) & set(gt[i].tolist())) for i in range(nq)])) / K
            Xc = B.read_rows(s['base'], d, ids.ravel()).astype(np.float64).reshape(nq, K, d); dist = ((Xc - Q64[:, None, :]) ** 2).sum(2)
            tie = float(np.mean(np.minimum(K, (dist <= (thr + 1e-9 * np.abs(thr))[:, None]).sum(1)))) / K
            b = v['bits']; rows.append(dict(dataset=nm, piece=arm, mode='unified', B=3, alpha=a, recall_l2=round(strict, 6), recall_l2_tie=round(tie, 6),
                                            n_active=int((b > 0).sum()), max_bits=int(b.max()), realized_bits=int(b.sum()), N=s['N'], nq=nq))
            B.log(f'pass B {nm}{tag} {arm}: Recall@10 {strict:.4f} (tie-aware {tie:.4f}) | active {rows[-1]["n_active"]}/{d}, max bits {rows[-1]["max_bits"]}')
        np.savez_compressed(f'{FS}/ids_{nm}{tag}.npz', **ids_all)
        with open(f'{FS}/{nm}{tag}.csv', 'w', newline='') as fo:
            w = csv.DictWriter(fo, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    B.log(f'pass B {grp}{tag} done [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[sys.argv.index('--rows-max') + 1]) if '--rows-max' in sys.argv else None)
