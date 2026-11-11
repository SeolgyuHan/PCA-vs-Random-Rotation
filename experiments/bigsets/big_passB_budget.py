#!/usr/bin/env python3
"""pass B at other bit budgets for one large dataset (Figure 1 as small multiples). Same streaming full scan as big_passB.py
(frozen codecs, unified quantizer, dither seed 0, score R^2 - 2 R <q~, u_hat>, per-query top-10, strict and tie-aware Recall@10
against pass A's ground truth), but for the arms and budgets given on the command line; one pass over the rows feeds all
(arm, B) pairs (the projection Y32 of a chunk is decoded once per budget). Checkpoint {WORK}/<tag>-passB-budget.ckpt.npz;
output results/fullscan_budget/<tag>.csv + ids_<tag>.npz, tag = <ds>__<arms>__B<budgets>.
Usage: python3 big_passB_budget.py <ds> --arms pca,rot,orig --B 1,2,4,6,8 [--threads 6] [--rows-max R] (smoke: --rows-max 300000)"""
import os, sys, csv, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
import nu_taskq as NU  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
f = NU.f; K = 10; QB = 64; RULE = {'pca': ('alpha1', 1.0), 'rot': ('uniform', 0.0), 'orig': ('alpha1', 1.0)}
FS = f'{B.OUT}/fullscan_budget'; os.makedirs(FS, exist_ok=True)
FIELDS = ['dataset', 'piece', 'mode', 'B', 'alpha', 'recall_l2', 'recall_l2_tie', 'n_active', 'max_bits', 'realized_bits', 'N', 'nq']


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main(ds, arms, budgets, nthr, rmax):
    s = B.spec(ds); d = s['d']; base = s['base']; N = min(s['N'], rmax) if rmax else s['N']; CH = B.chunk_rows(d) // (2 if d <= 256 else 1)
    tag = f"{ds}__{'-'.join(arms)}__B{'-'.join(map(str, budgets))}" + ('.smoke' if rmax else '')
    Q = B.read_queries(s); nq = len(Q); Qp, Qr = B.prepared_queries(s, Q); V = {}
    for arm in arms:
        rule, a = RULE[arm]; c = B.BigCodec.load(f'{B.FROZEN}/codec_{ds}__{arm}.npz'); c.M32 = c.M.astype(np.float32) if c.M is not None else None
        Qa = c.qcoords(Qr if arm == 'orig' else Qp).astype(np.float32)
        for b in budgets:
            bits = c.alloc(b, rule); pl = c.plan(bits); V[(arm, b)] = dict(c=c, bits=bits, pl=pl, z=c.dither(bits, 0), Qa=Qa, top=f.Top(nq, False), a=a)
    ck = f'{B.WORK}/{tag}-passB-budget.ckpt.npz'; z = B.load_ckpt(ck); pos = 0
    if z is not None:
        pos = int(z['pos'])
        for (arm, b), v in V.items(): v['top'].s[:] = z[f'{arm}|{b}|s']; v['top'].i[:] = z[f'{arm}|{b}|i']
    t0 = time.time(); last = time.time(); B.log(f'pass B budget {tag}: start at row {pos:,} of {N:,} ({len(V)} arms, {nq} queries, threads {nthr})')
    pool = ThreadPoolExecutor(max_workers=nthr); threadpool_limits(limits=nthr)
    for row0, X in B.iter_chunks(base, d, pos, N, CH, ()):
        B.gate(f'pass B budget {tag}'); n = X.shape[0]; Up, Rp, Ur, Rr = B.prepared(s, X)
        for arm in arms:
            c = V[(arm, budgets[0])]['c']
            if arm == 'orig': Y32, R = Ur, Rr
            else: Y32, R = Up @ c.M32, Rp
            Rs = R.astype(np.float32); nxs = (R.astype(np.float64) ** 2).astype(np.float32)
            for b in budgets:
                v = V[(arm, b)]; Yh = B.decode_par(c, Y32, v['pl'], v['z'], pool, nthr); Sf = v['Qa'] @ Yh.T; del Yh; T = v['top']
                list(pool.map(lambda q0: T.update(q0, nxs[None, :] - np.float32(2.0) * Rs[None, :] * Sf[q0:q0 + QB], row0), range(0, nq, QB))); del Sf
        del Up, Ur; pos = row0 + n
        if time.time() - last > 600 or pos >= N:
            arr = dict(pos=np.array(pos))
            for (arm, b), v in V.items(): arr[f'{arm}|{b}|s'] = v['top'].s; arr[f'{arm}|{b}|i'] = v['top'].i
            B.save_ckpt(ck, **arr); last = time.time(); print(f'  pass B budget {tag}: {pos:,}/{N:,} rows [{time.time()-t0:.0f}s]', flush=True)
    pool.shutdown()
    gt = np.load(f'{B.OUT}/{ds}-gt-l2.npy')[:, :K]; thr = np.load(f'{B.OUT}/{ds}-gt-l2-dist.npy')[:, K - 1]; Q64 = Q.astype(np.float64); rows, ids_all = [], {}
    for (arm, b), v in V.items():
        ids = v['top'].i.astype(np.int64); ids_all[f'{arm}_B{b}'] = ids
        strict = float(np.mean([len(set(ids[i].tolist()) & set(gt[i].tolist())) for i in range(nq)])) / K
        Xc = B.read_rows(base, d, ids.ravel()).astype(np.float64).reshape(nq, K, d); dist = ((Xc - Q64[:, None, :]) ** 2).sum(2)
        tie = float(np.mean(np.minimum(K, (dist <= (thr + 1e-9 * np.abs(thr))[:, None]).sum(1)))) / K
        bits = v['bits']; rows.append(dict(dataset=ds, piece=arm, mode='unified', B=b, alpha=v['a'], recall_l2=round(strict, 6), recall_l2_tie=round(tie, 6),
                                           n_active=int((bits > 0).sum()), max_bits=int(bits.max()), realized_bits=int(bits.sum()), N=N, nq=nq))
        B.log(f'pass B budget {ds}{" smoke" if rmax else ""} {arm} B={b}: Recall@10 {strict:.4f} (tie-aware {tie:.4f}) | active {rows[-1]["n_active"]}/{d}, max bits {rows[-1]["max_bits"]}')
    np.savez_compressed(f'{FS}/ids_{tag}.npz', **ids_all)
    with open(f'{FS}/{tag}.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    B.log(f'pass B budget {tag} done [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main(sys.argv[1], arg('--arms', 'pca,rot,orig').split(','), [int(x) for x in arg('--B', '1,2,4,6,8').split(',')], int(arg('--threads', '6')), int(arg('--rows-max', '0')) or None)
