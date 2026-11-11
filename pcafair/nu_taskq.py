#!/usr/bin/env python3
"""non-uniform per-component scalar quantizers (Bernardo-Zhu-Eldar-Evans 2023, Lloyd-Max levels, 'distortion factors')
inside our exact allocation, on the L2 task with the stored norm. Subset protocol of nd_mech.py: first 200k vectors (100k if
d > 1024), first 200 queries, exact L2 ground truth on the subset, Recall@10.
Quantizer families (per component, curves D_i(b) for the dynamic program, weights = variance (A2)):
  mm uniform cells over the database [min, max], midpoint levels (the design of the paper)
  ro uniform cells over clipped ranges, clipping chosen per bit count at database quantiles
  lm4 Lloyd-Max (centroid levels, nearest-level thresholds) for b <= 4, 'ro' for b > 4 -> LUT kernel unchanged (nibble tables)
  lm8 Lloyd-Max for b <= 8, 'ro' above (reference)
Their identical-bit design 'id' (System E with optimized P): top-P PCA components, the same b_u bits each (Lloyd-Max, ro above 8),
P = floor(B d / b_u), b_u chosen by the model J = sum_i lam_i D_i(b_i) (their eq. (11) for Gamma = Sigma_q^{1/2}, Sigma_q = Sigma_x).
Bases: raw PCA (x) and norm division (u = x/||x||, PCA basis of u from the training sample; score ||x||^2 - 2||x|| <q, u_hat>).
Lloyd-Max levels, quantiles and ranges come from a random training sample of the database (TS rows, seed 7); [min, max] from the
whole database. J_meas = within-query variance of the score error over the subset (mean over queries); for x the model J is
sum_i E[q_i^2] D_i(b_i) with E[q_i^2] over the 200 queries (their Theorem 1 / our Lemma 2: exact when Sigma_q is diagonal)."""
import os, sys, json, time, importlib.util
import numpy as np
from .config import PREPARED, WORK
OUT = str(WORK / 'nu_taskq')
from. import ea_lib as E  # noqa: E402
PG = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]; NSUB, NQ, K, TS, LMAX = 200_000, 200, 10, 200_000, 8


def lloyd(s, cs, cs2, L, init):
    """1-D Lloyd-Max on the sorted sample s (prefix sums cs, cs2 with a leading 0). Returns thresholds (L-1), levels (L), mse."""
    n = len(s); lev = init.copy()
    for _ in range(80):
        t = (lev[:-1] + lev[1:]) / 2; cut = np.concatenate([[0], np.searchsorted(s, t), [n]])
        cnt = np.diff(cut); sm = cs[cut[1:]] - cs[cut[:-1]]
        new = np.where(cnt > 0, sm / np.maximum(cnt, 1), lev)
        if np.max(np.abs(new - lev)) <= 1e-7 * (s[-1] - s[0] + 1e-30): lev = new; break
        lev = new
    t = (lev[:-1] + lev[1:]) / 2; cut = np.concatenate([[0], np.searchsorted(s, t), [n]]); cnt = np.diff(cut)
    sm = cs[cut[1:]] - cs[cut[:-1]]; sq = cs2[cut[1:]] - cs2[cut[:-1]]
    mse = float((sq - np.where(cnt > 0, sm ** 2 / np.maximum(cnt, 1), 0)).sum() / n)
    return t, lev, mse


def design(S, mn, mx, var, N):
    """Per-component curves and quantizer parameters from the training sample S (n_s x d) and the database [min, max]."""
    d = S.shape[1]; n = S.shape[0]; BM = E.BMAX
    vmm = E.curves_dexact(var, (mx - mn).astype(np.float64))
    vro = np.zeros((d, BM + 1)); vro[:, 0] = var; rlo = np.zeros((d, BM + 1)); rhi = np.zeros((d, BM + 1))
    vlm = np.full((d, LMAX + 1), np.nan); vlm[:, 0] = var; LM = {}
    for i in range(d):
        s = np.sort(S[:, i].astype(np.float64)); cs = np.concatenate([[0], np.cumsum(s)]); cs2 = np.concatenate([[0], np.cumsum(s * s)])
        cand = []
        for p in PG:
            k = int(np.floor(p * n)); lo, hi = s[k], s[n - 1 - k]
            clip = ((lo - s[:k]) ** 2).sum() / n + ((s[n - k:] - hi) ** 2).sum() / n if k > 0 else 0.0
            if p == 0.0: lo, hi = mn[i], mx[i]                       # p = 0 means the database range
            cand.append((lo, hi, clip))
        for b in range(1, BM + 1):
            vals = [(hi - lo) ** 2 * 4.0 ** (-b) / 12 + clip for lo, hi, clip in cand]; j = int(np.argmin(vals))
            vro[i, b] = vals[j]; rlo[i, b], rhi[i, b] = cand[j][0], cand[j][1]
        if var[i] <= 1e-12 * var.max(): continue
        lev = np.array([s[n // 4], s[(3 * n) // 4]], np.float64)
        for b in range(1, LMAX + 1):
            L = 2 ** b
            if b > 1: lev = np.sort(np.concatenate([lev - 1e-3 * (s[-1] - s[0]) / L, lev + 1e-3 * (s[-1] - s[0]) / L]))
            else: lev = np.array([np.mean(s[s < np.median(s)]), np.mean(s[s >= np.median(s)])])
            t, lev, mse = lloyd(s, cs, cs2, L, lev); vlm[i, b] = mse; LM[(i, b)] = (t.astype(np.float32), lev.astype(np.float32))
    return vmm, vro, rlo, rhi, vlm, LM


def quantize(X, bits, fam, mn, mx, rlo, rhi, LM, lmcap):
    d = X.shape[1]; Xq = np.zeros_like(X)
    for i in range(d):
        b = int(bits[i])
        if b == 0: continue
        if fam != 'mm' and b <= lmcap and (i, b) in LM:
            t, lev = LM[(i, b)]; Xq[:, i] = lev[np.searchsorted(t, X[:, i])]
        else:
            lo, hi = (mn[i], mx[i]) if fam == 'mm' else (rlo[i, b], rhi[i, b])
            if hi <= lo: continue
            Lc = 2.0 ** b; W = (hi - lo) / Lc; k = np.clip(np.floor((X[:, i] - lo) / W), 0, Lc - 1); Xq[:, i] = lo + (k + 0.5) * W
    return Xq


def run(ds, Bs):
    t0 = time.time(); db = np.load(f'{PREPARED}/{ds}-db.npy', mmap_mode='r'); N, d = db.shape; CH = min(100_000, 40_000_000 // d)
    R = np.empty(N); mn = np.full(d, np.inf); mx = np.full(d, -np.inf); s1 = np.zeros(d); s2 = np.zeros(d)
    for r0 in range(0, N, CH):
        X = np.asarray(db[r0:r0 + CH]).astype(np.float64); R[r0:r0 + CH] = np.sqrt((X ** 2).sum(1))
        mn = np.minimum(mn, X.min(0)); mx = np.maximum(mx, X.max(0)); s1 += X.sum(0); s2 += (X ** 2).sum(0)
    var = np.maximum(s2 / N - (s1 / N) ** 2, 1e-30)
    ti = np.sort(np.random.default_rng(7).choice(N, min(TS, N), replace=False)); S = np.asarray(db[ti]).astype(np.float64)
    U = S / np.maximum(R[ti], 1e-30)[:, None]; C = np.cov(U, rowvar=False); ev, V = np.linalg.eigh(C); V = np.ascontiguousarray(V[:, ::-1])
    SV = U @ V; mnv = np.full(d, np.inf); mxv = np.full(d, -np.inf); a1 = np.zeros(d); a2 = np.zeros(d)
    for r0 in range(0, N, CH):
        Y = (np.asarray(db[r0:r0 + CH]).astype(np.float64) / np.maximum(R[r0:r0 + CH], 1e-30)[:, None]) @ V
        mnv = np.minimum(mnv, Y.min(0)); mxv = np.maximum(mxv, Y.max(0)); a1 += Y.sum(0); a2 += (Y ** 2).sum(0)
    varv = np.maximum(a2 / N - (a1 / N) ** 2, 1e-30)
    print(f'  {ds} stats [{time.time()-t0:.0f}s]', flush=True)
    Dx = design(S, mn, mx, var, N); Dv = design(SV, mnv, mxv, varv, N); del S, U, SV
    print(f'  {ds} Lloyd-Max designs [{time.time()-t0:.0f}s]', flush=True)
    n = min(N, NSUB if d <= 1024 else NSUB // 2); Xs = np.asarray(db[:n]).astype(np.float32); Rs = R[:n]; nx = Rs ** 2
    Q = np.load(f'{PREPARED}/{ds}-query.npy').astype(np.float32)[:NQ]; nq = Q.shape[0]
    true = nx[None, :] - 2 * (Q.astype(np.float64) @ Xs.T.astype(np.float64)); T = np.argpartition(true, K, axis=1)[:, :K]
    Vs = ((Xs / np.maximum(Rs, 1e-30)[:, None].astype(np.float32)) @ V.astype(np.float32)); QV = Q @ V.astype(np.float32)
    wq = (Q.astype(np.float64) ** 2).mean(0)
    def ev_(Yh, scale, Qb):
        P_ = Qb.astype(np.float64) @ Yh.T.astype(np.float64)
        if scale is not None: P_ *= scale[None, :]
        est = nx[None, :] - 2 * P_; Pk = np.argpartition(est, K, axis=1)[:, :K]
        return float(np.mean([len(set(Pk[i]) & set(T[i])) / K for i in range(nq)])), float((est - true).var(axis=1).mean()) / 4.0
    res = {}
    for B in Bs:
        tot = B * d; row = {}
        for tag, D_, X_, w_, mn_, mx_, sc, Qb in (('x', Dx, Xs, var, mn, mx, None, Q), ('u', Dv, Vs, varv, mnv, mxv, Rs, QV)):
            vmm, vro, rlo, rhi, vlm, LM = D_
            for fam, cap in (('mm', 0), ('ro', 0), ('lm4', 4), ('lm8', 8)):
                if fam == 'mm': cur = vmm
                elif fam == 'ro': cur = vro
                else:
                    cur = vro.copy(); m = ~np.isnan(vlm[:, 1:cap + 1]); cur[:, 1:cap + 1] = np.where(m, vlm[:, 1:cap + 1], vro[:, 1:cap + 1])
                bits = E.dp_alloc(w_[:, None] * cur, tot)
                rec, J = ev_(quantize(X_, bits, fam, mn_, mx_, rlo, rhi, LM, cap), sc, Qb)
                Jm = float((wq * cur[np.arange(d), bits]).sum()) if tag == 'x' else float('nan')
                row[f'{tag}_{fam}'] = dict(recall=rec, J=J, Jmodel=Jm, active=int((bits > 0).sum()), maxb=int(bits.max()))
            if tag == 'x':                                                # their identical-bit design with optimized P
                order = np.argsort(-w_); best = None
                cur = vro.copy(); m = ~np.isnan(vlm[:, 1:LMAX + 1]); cur[:, 1:LMAX + 1] = np.where(m, vlm[:, 1:LMAX + 1], vro[:, 1:LMAX + 1])
                for bu in range(1, 13):
                    P = min(d, tot // bu)
                    if P < 1: break
                    bits = np.zeros(d, np.int64); bits[order[:P]] = bu; Jm = float((w_ * cur[np.arange(d), bits]).sum())
                    if best is None or Jm < best[0]: best = (Jm, bu, P, bits)
                rec, J = ev_(quantize(X_, best[3], 'lm8', mn_, mx_, rlo, rhi, LM, LMAX), sc, Qb)
                row['x_id'] = dict(recall=rec, J=J, Jmodel=float((wq * cur[np.arange(d), best[3]]).sum()), active=best[2], maxb=best[1])
        res[B] = row
        print(f'  {ds} B={B}: ' + ' '.join(f"{k} {v['recall']:.3f}" for k, v in row.items()) + f' [{time.time()-t0:.0f}s]', flush=True)
        print(f'  {ds} B={B} J_meas/J_model (x): ' + ' '.join(f"{k} {v['J']/v['Jmodel']:.3f}" for k, v in row.items() if k.startswith('x_')), flush=True)
    json.dump(res, open(f'{OUT}/{ds}.json', 'w'), indent=1)


if __name__ == '__main__':
    Bs = [int(b) for b in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1, 2, 3, 4]
    for ds in sys.argv[1].split(','):
        if not os.path.exists(f'{OUT}/{ds}.json'): run(ds, Bs)
