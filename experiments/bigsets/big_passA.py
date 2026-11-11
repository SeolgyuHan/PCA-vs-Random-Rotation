#!/usr/bin/env python3
""" pass A: one streaming pass per group (sift100m | msmarco = msmarco10m prefix + msmarco113m) that computes
  (1) the exact L2 ground truth: top-100 per query by 2<q, x> - ||x||^2 over the raw rows (float32 scan), the 100 candidates then
      re-sorted by the float64 distance (ties by id); prefix snapshots at the cuts (sift100m 1M, 10M; msmarco113m 1M, 10M);
  (2) the exact Cov(u) over all rows (float64), u = p / ||p||, p = the prepared row ((x - mu) W rounded to float32, as the 1M
      datasets' $PCARR_PREPARED/<ds>-db.npy) -> the eigenvalues and CG of Table II (as data/coding_gain_direction.py);
  (3) min, max, sum, sum of squares over all rows of the coding coordinates of the three codecs, the database statistics of
      FairCodec.fit: pca u M_pca (M_pca = eigenvectors of Cov(u) on the training sample), rot u H (H = haar_fullscan(d, 0)),
      orig (x - mu)/||x - mu|| in the raw coordinates.
Stage 1 before the pass (cached): the training sample rows (numpy default_rng(7), min(200k, 1.2e8/d) of N) and M_pca.
Checkpoints about every 10 minutes (resume from them). Guards of pcafair.bigsets (memory; E7 window) between chunks.
Usage: python3 big_passA.py <group> [--rows-max R] (R: smoke test on the first R rows only, outputs tagged.smoke)"""
import os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
sys.path.insert(0, B.CAMP)
from pcafair import bigkern as C  # noqa: E402 (top-k filtering kernels; make -C csrc)
from pcafair.engine import haar_fullscan  # noqa: E402
from scipy.linalg.blas import dsyrk  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
KGT = 100; BASES = ('pca', 'rot', 'orig')


def stage1(s, tag):
    p = f'{B.WORK}/{s["name"]}-stage1{tag}.npz'
    if os.path.exists(p): z = np.load(p); return {k: z[k] for k in z.files}
    B.wait_mem(10, f'pass A stage 1 {s["name"]}{tag}'); t0 = time.time(); N, d = s['N'], s['d']; TS = B.TS_RULE(d)
    ti = np.sort(np.random.default_rng(7).choice(N, min(TS, N), replace=False))
    raw = B.read_rows(s['base'], d, ti); Up, Rp, Ur, Rr = B.prepared(s, raw)
    ev, V = np.linalg.eigh(np.cov(Up, rowvar=False)); Mp = np.ascontiguousarray(V[:, ::-1])
    H = haar_fullscan(d, 0).astype(np.float64)
    out = dict(ti=ti, raw=raw, Rp=Rp, Rr=Rr, Mp=Mp, H=H); B.save_ckpt(p, **out)
    B.log(f'pass A stage 1 {s["name"]}{tag}: training sample {len(ti):,} rows, M_pca fitted [{time.time()-t0:.0f}s]')
    return out


class Acc:
    def __init__(self, d): self.mn = np.full(d, np.inf); self.mx = np.full(d, -np.inf); self.s1 = np.zeros(d); self.s2 = np.zeros(d)
    def add(self, Y):                                                             # float32 coordinates, float64 sums
        np.minimum(self.mn, Y.min(0), out=self.mn); np.maximum(self.mx, Y.max(0), out=self.mx)
        self.s1 += Y.sum(0, dtype=np.float64); self.s2 += np.einsum('ij,ij->j', Y, Y, dtype=np.float64)


def main(grp, rmax=None):
    tag = '.smoke' if rmax else ''; names = B.GROUPS[grp]; S = {n: B.spec(n) for n in names}
    if rmax:
        for n in names: S[n]['N'] = min(S[n]['N'], rmax if n != 'msmarco10m' else rmax // 2)
    d = S[names[0]]['d']; base = S[names[0]]['base']; Nmax = max(S[n]['N'] for n in names); CH = B.chunk_rows(d)
    Q = B.read_queries(S[names[0]]); nq = len(Q)
    st1 = {n: stage1(S[n], tag) for n in names}; Q2 = (Q * np.float32(2.0)).astype(np.float32)
    for n in names: st1[n]['Mp32'] = st1[n]['Mp'].astype(np.float32); st1[n]['H32'] = st1[n]['H'].astype(np.float32)
    ck = f'{B.WORK}/{grp}-passA{tag}.ckpt.npz'; z = B.load_ckpt(ck)
    D = {}
    for n in names:
        s = S[n]; D[n] = dict(top=C.TopK(nq, KGT), su=np.zeros(d), suu=np.zeros((d, d)), sR2=0.0, acc={b: Acc(d) for b in BASES}, snaps={})
        if z is not None:
            T = D[n]['top']; T.s[:] = z[f'{n}|top_s']; T.i[:] = z[f'{n}|top_i']; T.theta[:] = z[f'{n}|top_theta']; T.filled = bool(z[f'{n}|top_filled'])
            D[n]['su'] = z[f'{n}|su']; D[n]['suu'] = z[f'{n}|suu']; D[n]['sR2'] = float(z[f'{n}|sR2'])
            for b in BASES:
                a = D[n]['acc'][b]; a.mn, a.mx, a.s1, a.s2 = (z[f'{n}|{b}|{k}'] for k in ('mn', 'mx', 's1', 's2'))
            for k in z.files:
                if k.startswith(f'{n}|snap'): D[n]['snaps'][k.split('|')[1]] = z[k]
    pos = int(z['pos']) if z is not None else 0
    flt = C.Filter(); t0 = time.time(); last = time.time(); nthr = 10; pool = ThreadPoolExecutor(max_workers=3)
    cuts = sorted(set(c for n in names for c in S[n]['cuts']) | set(S[n]['N'] for n in names))
    B.log(f'pass A {grp}{tag}: start at row {pos:,} of {Nmax:,} (d {d}, {nq} queries, chunk {CH:,})')
    with threadpool_limits(limits=nthr):
        for row0, X in B.iter_chunks(base, d, pos, Nmax, CH, cuts):
            B.gate(f'pass A {grp}'); n = X.shape[0]
            nx = np.einsum('ij,ij->i', X, X); Sq = Q2 @ X.T; Sq -= nx[None, :]                     # 2<q, x> - ||x||^2 (2q exact)
            for nm in names:
                if row0 < S[nm]['N']: D[nm]['top'].update(Sq, row0, flt)
            del Sq
            for nm in names:
                s = S[nm]
                if row0 >= s['N']: continue
                Up, Rp, Ur, Rr = B.prepared(s, X); R_ = D[nm]
                U64 = Up.astype(np.float64)                                          # exact Cov(u) in float64 (Table II CG)
                R_['sR2'] += float(np.dot(Rp, Rp)); R_['su'] += U64.sum(0); R_['suu'] = dsyrk(1.0, U64.T, beta=1.0, c=R_['suu'], trans=0, lower=0, overwrite_c=1); del U64
                Yp = Up @ st1[nm]['Mp32']; Yr = Up @ st1[nm]['H32']; del Up
                list(pool.map(lambda bY: R_['acc'][bY[0]].add(bY[1]), (('orig', Ur), ('pca', Yp), ('rot', Yr)))); del Ur, Yp, Yr
                end = row0 + n
                for c in s['cuts']:
                    if end == c: R_['snaps'][f'snap{c}'] = np.concatenate([R_['top'].i, R_['top'].s.astype(np.float64)], 1)
            pos = row0 + n
            if time.time() - last > 600 or pos >= Nmax:
                arr = dict(pos=np.array(pos))
                for nm in names:
                    R_ = D[nm]; T = R_['top']
                    arr.update({f'{nm}|top_s': T.s, f'{nm}|top_i': T.i, f'{nm}|top_theta': T.theta, f'{nm}|top_filled': np.array(T.filled),
                                f'{nm}|su': R_['su'], f'{nm}|suu': R_['suu'], f'{nm}|sR2': np.array(R_['sR2'])})
                    for b in BASES:
                        a = R_['acc'][b]; arr.update({f'{nm}|{b}|mn': a.mn, f'{nm}|{b}|mx': a.mx, f'{nm}|{b}|s1': a.s1, f'{nm}|{b}|s2': a.s2})
                    for k, v in R_['snaps'].items(): arr[f'{nm}|{k}'] = v
                B.save_ckpt(ck, **arr); last = time.time()
                el = time.time() - t0; print(f'  pass A {grp}{tag}: {pos:,}/{Nmax:,} rows [{el:.0f}s]', flush=True)
    # ---------------- post: eigenvalues, exact re-sort of the ground truth, statistics
    for nm in names:
        s = S[nm]; R_ = D[nm]; N = s['N']
        suu = np.triu(R_['suu']) + np.triu(R_['suu'], 1).T; m = R_['su'] / N
        lam = np.sort(np.linalg.eigvalsh(suu / N - np.outer(m, m)))[::-1]
        cg = float(lam.mean() / np.exp(np.mean(np.log(lam)))) if lam.min() > 0 else float('nan')
        gts = {'': R_['top'].i}; gts.update({'.prefix' + k[4:]: v[:, :KGT].astype(np.int64) for k, v in R_['snaps'].items()})
        for suf, ids in gts.items():
            Xc = B.read_rows(s['base'], d, ids.ravel()).astype(np.float64).reshape(nq, KGT, d)
            dist = ((Xc - Q.astype(np.float64)[:, None, :]) ** 2).sum(2)
            o = np.lexsort((ids, dist), axis=1); gid = np.take_along_axis(ids, o, 1); gd = np.take_along_axis(dist, o, 1)
            np.save(f'{B.OUT}/{nm}-gt-l2{suf}{tag}.npy', gid.astype(np.int64)); np.save(f'{B.OUT}/{nm}-gt-l2-dist{suf}{tag}.npy', gd)
        arr = dict(N=np.array(N), ER2=np.array(R_['sR2'] / N), lam=lam, CG=np.array(cg))
        for b in BASES:
            a = R_['acc'][b]; arr.update({f'{b}|mn': a.mn, f'{b}|mx': a.mx, f'{b}|s1': a.s1, f'{b}|s2': a.s2})
        B.save_ckpt(f'{B.WORK}/{nm}-passA{tag}.npz', **arr)
        B.log(f'pass A {nm}{tag} done: CG_u(exact) {cg:.4f}, E[R^2] {R_["sR2"]/N:.6g}, GT top-{KGT} for {nq} queries (+{len(gts)-1} prefix) [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[sys.argv.index('--rows-max') + 1]) if '--rows-max' in sys.argv else None)
