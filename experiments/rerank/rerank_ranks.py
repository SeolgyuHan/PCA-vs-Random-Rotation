#!/usr/bin/env python3
""" the re-ranking experiment: one full scan per (dataset, design) with the frozen codecs at
B = 1..8 (integers: Random Rotation+alpha=0 uniform bits, PCA+alpha=1 Eq. (2) exact budget), the same scoring as E1 (fullscan.py):
R^2 - 2R<q~, u_hat> with the stored norm. For every query and each of its 10 true neighbours (gt-l2) the rank of the neighbour in the
code ranking is counted: the number of other database vectors whose approximate score is strictly smaller (chunk scores as E1, the
neighbour's own score computed once from its row; the self comparison is removed). From the ranks:
  recall without re-ranking: Recall@10 = mean #(rank < 10)/10 (E1's value for B in {1,2,3,4,6,8}, up to rounding at near ties)
  re-ranking: Recall@10 after exact re-scoring of the top c with the originals = mean #(rank < c)/10 for every c (a true neighbour inside the
        top c is always among the 10 nearest of those c); the exact depth for a target is the smallest c >= 10 that reaches it.
Output rerank/<ds>__<design>.npz (ranks per B, int32, nq x 10) and rerank/em_summary.csv
(dataset, design, B, nq, recall_norerank, depth90, depth95, depth99; depth 10 = nothing re-scored).
Usage: python3 rerank_ranks.py [ds,...] [design,...] (lock files; several workers may share one list)"""
import os, sys, csv, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair.config import PREPARED, FROZEN, OUT as OUT_ROOT, RESULTS  # noqa: E402
from pcafair.codec import FairCodec  # noqa: E402
PREP = str(PREPARED)
OUT = f'{OUT_ROOT}/rerank'; os.makedirs(OUT, exist_ok=True)
K = 10; QB = 256; BS = (1, 2, 3, 4, 5, 6, 7, 8); TARGETS = (0.90, 0.95, 0.99)
ORDER = ['audio', 'fashion-mnist', 'mnist', 'msong', 'word2vec', 'gist', 'youtube', 'msmarco', 'openai1536', 'openai3072']
FIELDS = ['dataset', 'design', 'B', 'nq', 'recall_norerank', 'depth90', 'depth95', 'depth99']
def avail(): return int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 1048576


def depth_for(ranks, tt):
    """Smallest depth c >= 10 such that the mean over queries of #(rank < c)/10 reaches tt."""
    r = np.sort(ranks.ravel()); need = int(np.ceil(tt * r.size - 1e-9))
    return K if need <= 0 else max(K, int(r[need - 1]) + 1)


def run(ds, design):
    t0 = time.time(); c = FairCodec.load(f'{FROZEN}/codec_{ds}__{design}.npz'); rule = 'alpha1' if design == 'pca' else 'uniform'
    db = np.load(f'{PREP}/{ds}-db.npy', mmap_mode='r'); N = db.shape[0]; d = c.d; CH = min(100_000, 40_000_000 // max(d, 1))
    gt = np.load(f'{PREP}/{ds}-gt-l2.npy')[:, :K].astype(np.int64); nq = gt.shape[0]
    Qp = np.load(f'{PREP}/{ds}-query.npy')[:nq].astype(np.float64); Qa = c.qcoords(Qp).astype(np.float32)
    Xg = np.asarray(db[gt.ravel()]).astype(np.float64); Rg = np.sqrt((Xg ** 2).sum(1)); Yg = c.coords(Xg, Rg).astype(np.float32)
    Rgs = Rg.astype(np.float32); nxg = (Rg ** 2).astype(np.float32); del Xg
    V = []
    for B in BS:
        bits = c.alloc(B, rule); pl = c.plan(bits); z = c.dither(bits, 0)
        Yhg = c.decode_unified(Yg, pl, z).reshape(nq, K, d)                 # the neighbours' own scores, in the scan's precision
        dot = np.einsum('qkd,qd->qk', Yhg, Qa.astype(Yhg.dtype))
        sg = nxg.reshape(nq, K) - np.float32(2.0) * Rgs.reshape(nq, K) * dot
        V.append(dict(B=B, pl=pl, z=z, sg=sg, cnt=np.zeros((nq, K), np.int64), realized=int(bits.sum())))
    print(f'  {ds} {design}: N={N} d={d} nq={nq} B={list(BS)} [{time.time()-t0:.0f}s]', flush=True)
    for r0 in range(0, N, CH):
        r1 = min(N, r0 + CH); X = c.rows(r0, r1); R = np.sqrt((X ** 2).sum(1)); Y = c.coords(X, R).astype(np.float32); Rs = R.astype(np.float32)
        nxs = (R.astype(np.float64) ** 2).astype(np.float32); del X
        qi, ki = np.nonzero((gt >= r0) & (gt < r1)); li = gt[qi, ki] - r0
        for v in V:
            Yh = c.decode_unified(Y, v['pl'], v['z'])
            for q0 in range(0, nq, QB):
                q1 = min(nq, q0 + QB); S = nxs[None, :] - np.float32(2.0) * Rs[None, :] * (Qa[q0:q1] @ Yh.T)
                sg = v['sg'][q0:q1].astype(S.dtype)
                for k in range(K): v['cnt'][q0:q1, k] += np.count_nonzero(S < sg[:, k:k + 1], axis=1)
                m = (qi >= q0) & (qi < q1)
                if m.any():                                                  # remove the neighbour's comparison with itself
                    own = S[qi[m] - q0, li[m]]; v['cnt'][qi[m], ki[m]] -= (own < v['sg'][qi[m], ki[m]].astype(S.dtype)).astype(np.int64)
                del S
            del Yh
        if (r0 // CH) % 5 == 0: print(f'  {ds} {design} rows {r1}/{N} [{time.time()-t0:.0f}s]', flush=True)
    rows, arrs = [], {}
    for v in V:
        rk = v['cnt'].astype(np.int32); arrs[f'B{v["B"]}'] = rk
        row = dict(dataset=ds, design=design, B=v['B'], nq=nq, recall_norerank=round(float((rk < K).mean()), 6),
                   **{f'depth{round(t * 100)}': depth_for(rk, t) for t in TARGETS}); rows.append(row)
        print(f"  {ds} {design} B={v['B']}: recall {row['recall_norerank']:.4f} | depth90 {row['depth90']} depth95 {row['depth95']} depth99 {row['depth99']} (bits {v['realized']})", flush=True)
    np.savez_compressed(f'{OUT}/{ds}__{design}.npz', **arrs)
    p = f'{OUT}/em_summary.csv'; new = not os.path.exists(p)
    with open(p, 'a', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS); w.writeheader() if new else None; w.writerows(rows)
    print(f'--- {ds} {design} done [{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    sets = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] else ORDER
    designs = sys.argv[2].split(',') if len(sys.argv) > 2 else ['pca', 'rot']
    for ds in sets:
        for design in designs:
            tag = f'{ds}__{design}'
            if os.path.exists(f'{OUT}/{tag}.npz'): continue
            try: os.close(os.open(f'{OUT}/{tag}.lock', os.O_CREAT | os.O_EXCL))
            except FileExistsError: continue
            while avail() < 10: print(f'  [{tag}] waiting for 10 GB', flush=True); time.sleep(120)
            try: run(ds, design)
            finally: os.remove(f'{OUT}/{tag}.lock')
