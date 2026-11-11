#!/usr/bin/env python3
""" S7: slopes other than 1 on the datasets of 10M rows or more. Same streaming full scan as big_passB.py (B = 3, unified
quantizer, dither seed 0, score R^2 - 2 R <q~, u_hat>, per-query top-10, strict and tie-aware Recall@10 against pass A's ground
truth), but every arm is the frozen PCA codec of the dataset with the allocation of Eq. (2) at another slope: 'alpha1' for 1.0
(the pca arm of big_passB.py, used to validate), 'alpha<a>' otherwise. One codec and one projection per dataset feed all arms.
Checkpoint {WORK}/<group>-passB-slope-<alphas>.ckpt.npz; output results/fullscan_slope/<ds>__<alphas>.csv + ids_<ds>__<alphas>.npz.
Usage: python3 big_passB_slope.py <group> [--alphas 0.0,0.5,1.5]"""
import os, sys, csv, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
from pcafair.topk import Top  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
K = 10; QB = 64
FS = f'{B.OUT}/fullscan_slope'; os.makedirs(FS, exist_ok=True)
FIELDS = ['dataset', 'piece', 'mode', 'B', 'alpha', 'recall_l2', 'recall_l2_tie', 'n_active', 'max_bits', 'realized_bits', 'N', 'nq']


def threads_now():
    h = os.environ.get('S7_THREADS_FILE')   # per-process thread hint (a number in a file), re-read every chunk; results do not depend on it
    if h:
        try: return max(1, int(open(h).read().strip()))
        except (OSError, ValueError): pass
    return 20 if B.e7_done() else 10


def rule_of(a): return 'alpha1' if a == 1.0 else f'alpha{a}'


def main(grp, alphas):
    names = B.GROUPS[grp]; S = {n: B.spec(n) for n in names}; atag = '_'.join(f'{a:g}' for a in alphas)
    d = S[names[0]]['d']; base = S[names[0]]['base']; Nmax = max(S[n]['N'] for n in names); CH = B.chunk_rows(d) // (2 if d <= 256 else 1)
    Q = B.read_queries(S[names[0]]); nq = len(Q)
    C, V = {}, {}
    for n in names:
        s = S[n]; Qp, Qr = B.prepared_queries(s, Q)
        c = B.BigCodec.load(f'{B.FROZEN}/codec_{n}__pca.npz'); c.M32 = c.M.astype(np.float32); C[n] = c
        Qa = c.qcoords(Qp).astype(np.float32)
        for a in alphas:
            bits = c.alloc(3, rule_of(a)); pl = c.plan(bits)
            V[(n, a)] = dict(bits=bits, pl=pl, z=c.dither(bits, 0), Qa=Qa, top=Top(nq, False, K))
    ck = f'{B.WORK}/{grp}-passB-slope-{atag}.ckpt.npz'; z = B.load_ckpt(ck); pos = 0
    if z is not None:
        pos = int(z['pos'])
        for (n, a), v in V.items(): v['top'].s[:] = z[f'{n}|{a:g}|s']; v['top'].i[:] = z[f'{n}|{a:g}|i']
    cuts = sorted(set(S[n]['N'] for n in names)); t0 = time.time(); last = time.time(); nthr = threads_now()
    B.log(f'S7 pass B slope {grp}: start at row {pos:,} of {Nmax:,} (slopes {atag}, {len(V)} arms, {nq} queries, threads {nthr})')
    pool = ThreadPoolExecutor(max_workers=16); threadpool_limits(limits=nthr)
    for row0, X in B.iter_chunks(base, d, pos, Nmax, CH, cuts):
        B.gate(f'S7 pass B slope {grp}')
        if threads_now() != nthr: nthr = threads_now(); threadpool_limits(limits=nthr); B.log(f'S7 pass B slope {grp}: threads {nthr}')
        n = X.shape[0]
        for nm in names:
            if row0 >= S[nm]['N']: continue
            Up, Rp, Ur, Rr = B.prepared(S[nm], X); del Ur
            c = C[nm]; Y32 = Up @ c.M32; del Up
            Rs = Rp.astype(np.float32); nxs = (Rp.astype(np.float64) ** 2).astype(np.float32)
            for a in alphas:
                v = V[(nm, a)]
                Yh = B.decode_par(c, Y32, v['pl'], v['z'], pool, min(nthr, 16))
                Sf = v['Qa'] @ Yh.T; del Yh
                T = v['top']
                list(pool.map(lambda q0: T.update(q0, nxs[None, :] - np.float32(2.0) * Rs[None, :] * Sf[q0:q0 + QB], row0), range(0, nq, QB))); del Sf
            del Y32
        pos = row0 + n
        if time.time() - last > 600 or pos >= Nmax:
            arr = dict(pos=np.array(pos))
            for (nm, a), v in V.items(): arr[f'{nm}|{a:g}|s'] = v['top'].s; arr[f'{nm}|{a:g}|i'] = v['top'].i
            B.save_ckpt(ck, **arr); last = time.time(); print(f'  S7 pass B slope {grp}: {pos:,}/{Nmax:,} rows [{time.time()-t0:.0f}s]', flush=True)
    pool.shutdown()
    for nm in names:
        s = S[nm]; gt = np.load(f'{B.OUT}/{nm}-gt-l2.npy')[:, :K]; thr = np.load(f'{B.OUT}/{nm}-gt-l2-dist.npy')[:, K - 1]
        Q64 = Q.astype(np.float64); rows, ids_all = [], {}
        for a in alphas:
            v = V[(nm, a)]; ids = v['top'].i.astype(np.int64); ids_all[f'a{a:g}'] = ids
            strict = float(np.mean([len(set(ids[i].tolist()) & set(gt[i].tolist())) for i in range(nq)])) / K
            Xc = B.read_rows(s['base'], d, ids.ravel()).astype(np.float64).reshape(nq, K, d); dist = ((Xc - Q64[:, None, :]) ** 2).sum(2)
            tie = float(np.mean(np.minimum(K, (dist <= (thr + 1e-9 * np.abs(thr))[:, None]).sum(1)))) / K
            b = v['bits']; rows.append(dict(dataset=nm, piece='pcaslope', mode='unified', B=3, alpha=a, recall_l2=round(strict, 6), recall_l2_tie=round(tie, 6),
                                            n_active=int((b > 0).sum()), max_bits=int(b.max()), realized_bits=int(b.sum()), N=s['N'], nq=nq))
            B.log(f'S7 pass B slope {nm} alpha {a:g}: Recall@10 {strict:.4f} (tie-aware {tie:.4f}) | active {rows[-1]["n_active"]}/{d}, max bits {rows[-1]["max_bits"]}, sum {rows[-1]["realized_bits"]}')
        np.savez_compressed(f'{FS}/ids_{nm}__{atag}.npz', **ids_all)
        with open(f'{FS}/{nm}__{atag}.csv', 'w', newline='') as fo:
            w = csv.DictWriter(fo, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    B.log(f'S7 pass B slope {grp} done [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    al = sys.argv[sys.argv.index('--alphas') + 1] if '--alphas' in sys.argv else '0.0,0.5,1.5'
    main(sys.argv[1], [float(x) for x in al.split(',')])
