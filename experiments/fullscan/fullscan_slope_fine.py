#!/usr/bin/env python3
""" fine slope sweep (copy of fullscan.py; only the piece list differs): piece pcafine = PCA basis, B = 3, unified,
slopes 0.5-1.5 in steps of 0.05 that the E1 pieces pca/pcaslope did not already measure (0.5, 0.75, 1.0, 1.25, 1.5 are there).
Original docstring follows.
 E1: full scan (whole database, all queries, L2 with the stored norm) with the frozen codecs.
Pieces <ds>__<piece>:
  pca PCA basis of u, alpha = 1, B in {1, 2, 3, 4, 6, 8}, modes 'unified' (dither seed 0: the measured = theory quantizer) and
             'det' (deterministic rounding + gain: the cost of the dither, Section V-D)
  rot haar_fullscan(d, 0), uniform bits, the same B and modes
  orig original coordinates (u), alpha = 1, B = 3, both modes (Figure 1)
  pcaslope PCA basis, B = 3, slopes alpha in {0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2}, unified (Section V-A; alpha = 1 is in 'pca')
Score: R^2 - 2 R <q~, u_hat> (smaller = nearer). Recall@10 strict and tie-aware against $PCARR_PREPARED/<ds>-gt-l2.npy (tie check by
exact L2 in the prepared coordinates). Per-query top-10 ids saved. Output fullscan/<ds>__<piece>.csv + ids_<...>.npz.
Usage: python3 fullscan.py [ds,...] [piece,...] (lock files; several workers may share one list)"""
import os, sys, csv, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair.config import PREPARED, FROZEN, OUT as OUT_ROOT, RESULTS  # noqa: E402
from pcafair.codec import FairCodec  # noqa: E402
PREP = str(PREPARED)
OUT = f'{OUT_ROOT}/fullscan'; os.makedirs(OUT, exist_ok=True)
from pcafair import nu_taskq as NU  # noqa: E402
from pcafair.topk import Top  # noqa: E402
K = 10; QB = 256
ORDER = ['audio', 'fashion-mnist', 'mnist', 'msong', 'word2vec', 'gist', 'youtube', 'msmarco', 'openai1536', 'openai3072']
PIECES = ['pcafine']
FIELDS = ['dataset', 'piece', 'mode', 'B', 'alpha', 'recall_l2', 'recall_l2_tie', 'n_active', 'max_bits', 'realized_bits']
def avail(): return int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 1048576


def variants(piece):
    if piece in ('pca', 'rot'):
        rule = 'alpha1' if piece == 'pca' else 'uniform'; a = 1.0 if piece == 'pca' else 0.0
        return [(m, B, a, rule) for B in (1, 2, 3, 4, 6, 8) for m in ('unified', 'det')]
    if piece == 'orig': return [(m, 3, 1.0, 'alpha1') for m in ('unified', 'det')]
    if piece == 'pcafine':
        done = {0.5, 0.75, 1.0, 1.25, 1.5}
        return [('unified', 3, a, f'alpha{a}') for a in (round(0.5 + 0.05 * k, 2) for k in range(21)) if a not in done]
    return [('unified', 3, a, f'alpha{a}') for a in (0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0)]


def run(ds, piece):
    t0 = time.time(); basis = {'pca': 'pca', 'rot': 'rot', 'orig': 'orig', 'pcaslope': 'pca', 'pcafine': 'pca'}[piece]
    c = FairCodec.load(f'{FROZEN}/codec_{ds}__{basis}.npz')
    db = np.load(f'{PREP}/{ds}-db.npy', mmap_mode='r'); N = db.shape[0]; d = c.d; CH = min(100_000, 40_000_000 // max(d, 1))
    gt = np.load(f'{PREP}/{ds}-gt-l2.npy')[:, :K]; nq = gt.shape[0]
    Qp = np.load(f'{PREP}/{ds}-query.npy')[:nq].astype(np.float64)                     # prepared coordinates (tie check)
    Qa = c.qcoords(c.queries(nq) if basis == 'orig' else Qp).astype(np.float32)
    nxall = np.empty(N, np.float64)
    for r0 in range(0, N, CH):
        X = np.asarray(db[r0:r0 + CH]).astype(np.float64); nxall[r0:r0 + CH] = np.einsum('ij,ij->i', X, X)
    def exact_l2(qi, ids):
        rows = np.asarray(db[np.sort(ids)], np.float64)[np.argsort(np.argsort(ids))]; return nxall[ids] - 2.0 * (rows @ Qp[qi])
    thr = np.array([exact_l2(i, gt[i]).max() for i in range(nq)])
    V = []
    for (mode, B, a, rule) in variants(piece):
        bits = c.alloc(B, 'uniform' if rule == 'uniform' else ('alpha1' if a == 1.0 else rule)); pl = c.plan(bits)
        V.append(dict(mode=mode, B=B, a=a, bits=bits, pl=pl, z=c.dither(bits, 0), top=Top(nq, False, K)))
    print(f'  {ds} {piece}: {len(V)} variants, N={N} d={d} nq={nq} [{time.time()-t0:.0f}s]', flush=True)
    for r0 in range(0, N, CH):
        X = c.rows(r0, min(N, r0 + CH)); R = np.sqrt((X ** 2).sum(1)); Y = c.coords(X, R).astype(np.float32); Rs = R.astype(np.float32)
        nxs = (R.astype(np.float64) ** 2).astype(np.float32); del X
        for v in V:
            Yh = c.decode_unified(Y, v['pl'], v['z']) if v['mode'] == 'unified' else c.decode_fast(Y, v['pl'], 'det')
            for q0 in range(0, nq, QB):
                S = Qa[q0:q0 + QB] @ Yh.T; v['top'].update(q0, nxs[None, :] - np.float32(2.0) * Rs[None, :] * S, r0); del S
            del Yh
        if (r0 // CH) % 5 == 0: print(f'  {ds} {piece} rows {min(N, r0 + CH)}/{N} [{time.time()-t0:.0f}s]', flush=True)
    rows, ids_all = [], {}
    for v in V:
        ids = v['top'].i; key = f"{v['mode']}_B{v['B']}_a{v['a']}"; ids_all[key] = ids.astype(np.int32)
        strict = float(np.mean([len(set(ids[i].tolist()) & set(gt[i].tolist())) for i in range(nq)])) / K
        tie = float(np.mean([min(K, int((exact_l2(i, ids[i]) <= thr[i] + 1e-9 * abs(thr[i])).sum())) for i in range(nq)])) / K
        b = v['bits']; rows.append(dict(dataset=ds, piece=piece, mode=v['mode'], B=v['B'], alpha=v['a'], recall_l2=round(strict, 6), recall_l2_tie=round(tie, 6),
                                        n_active=int((b > 0).sum()), max_bits=int(b.max()), realized_bits=int(b.sum())))
        print(f"  {ds} {piece} {v['mode']:7s} B{v['B']} a{v['a']}: L2 {strict:.4f} tie {tie:.4f} | active {rows[-1]['n_active']}/{d} max {rows[-1]['max_bits']}", flush=True)
    np.savez_compressed(f'{OUT}/ids_{ds}__{piece}.npz', **ids_all)
    with open(f'{OUT}/{ds}__{piece}.csv.tmp', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    os.replace(f'{OUT}/{ds}__{piece}.csv.tmp', f'{OUT}/{ds}__{piece}.csv'); print(f'--- {ds} {piece} done [{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    sets = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] else ORDER
    pieces = sys.argv[2].split(',') if len(sys.argv) > 2 else PIECES
    for ds in sets:
        for piece in pieces:
            tag = f'{ds}__{piece}'; basis = {'pca': 'pca', 'rot': 'rot', 'orig': 'orig', 'pcaslope': 'pca', 'pcafine': 'pca'}[piece]
            if os.path.exists(f'{OUT}/{tag}.csv'): continue
            while not os.path.exists(f'{FROZEN}/codec_{ds}__{basis}.json'): time.sleep(60)
            try: os.close(os.open(f'{OUT}/{tag}.lock', os.O_CREAT | os.O_EXCL))
            except FileExistsError: continue
            while avail() < 10: print(f'  [{tag}] waiting for 10 GB', flush=True); time.sleep(120)
            try: run(ds, piece)
            finally: os.remove(f'{OUT}/{tag}.lock')
