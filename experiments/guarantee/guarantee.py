#!/usr/bin/env python3
""" E4 (Section V-D): the data-independent interval with a stored per-vector residual size.
Inner-product error of the interval estimate over ALL components (dithered reconstruction WITHOUT gain correction for b >= 2, split
level for b = 1, the stored mean for b = 0): eps = sum_{b>=2} q~_i nu_i + sum_{i in D} q~_i r_i, where nu_i | y ~ U(-W_i/2, W_i/2)
(independent) and r(x) is the deterministic residual (edge bias beta_i(y_i) of b >= 2 components, the split-level error of 1-bit
components, and the deviation m_i - u_i of the components without bits from their stored mean m_i).
  |eps| <= tau_delta(q) + ||q~_D||_2 * S_hat(x) with probability >= 1 - delta over the shared dither, for every database vector:
  S(x) = ||r(x)||_2 computed at encoding; S_hat = its k-bit log-scale code rounded UP (k = 2, 3, 4; code 0 = no residual, codes
  1..2^k-1 = geometric levels from the 10th percentile of the nonzero S over the database to its maximum); D = the components whose
  code is an edge code (K = 0 or L-1, b >= 2), that have 1 bit, or that have no bits ('mask' = code), or all components ('mask' = all).
  Also reported: opt1 = the worst case from the stored per-component extremes (no stored S), and k = inf (S stored exactly).
Protocol: first 20,000 database vectors, queries 0-199, dither seeds 0-4, nominal 95/99/99.9%; S levels from the whole database.
Output guarantee/<ds>.csv (coverage, mean half-width / tau, S bits). Usage: python3 guarantee.py [ds,...] (locks)"""
import os, sys, csv, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair.config import PREPARED, FROZEN, OUT as OUT_ROOT, RESULTS  # noqa: E402
from pcafair.codec import FairCodec  # noqa: E402
PREP = str(PREPARED)
OUT = f'{OUT_ROOT}/guarantee'; os.makedirs(OUT, exist_ok=True); NSUB, NQ = 20_000, 200; DELTAS = (0.05, 0.01, 0.001); KS = (2, 3, 4)
ORDER = ['audio', 'fashion-mnist', 'mnist', 'msong', 'word2vec', 'gist', 'youtube', 'msmarco', 'openai1536', 'openai3072']
FIELDS = ['dataset', 'pair', 'B', 'seed', 'nominal', 'option', 'mask', 's_bits', 'coverage', 'width_over_tau', 'frac_S_zero']
def avail(): return int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 1048576
def tau_of(V, M, delta):
    a = np.log(2.0 / delta); return (2.0 / 3.0) * M * a / 2 + np.sqrt(((2.0 / 3.0) * M * a / 2) ** 2 + 2 * V * a)


def residual(Ya, pl):
    """Deterministic residual r (n x active) of the interval estimate and the 1-bit split levels."""
    bb = pl['b']; m2 = bb >= 2; m1 = bb == 1; lo, W, L = pl['lo'], pl['W'], pl['L']; hi = lo + L * W; r = np.zeros_like(Ya)
    beta = np.maximum(W / 2 - (Ya - lo), 0) - np.maximum(W / 2 - (hi - Ya), 0); r[:, m2] = beta[:, m2]
    if m1.any():
        Kd = np.clip(np.floor((Ya - lo) / W), 0, L - 1); r[:, m1] = (pl['a'] + pl['w'] * Kd - Ya)[:, m1]
    return r


def levels(Sdb, k):
    nz = Sdb[Sdb > 0]
    if not len(nz): return np.array([0.0])
    t1 = np.quantile(nz, 0.10); smax = nz.max(); m = 2 ** k - 1
    lv = t1 * (smax / t1) ** (np.arange(m) / max(m - 1, 1)) if smax > t1 else np.full(m, smax)
    lv[-1] = smax; return np.concatenate([[0.0], lv])


def code_up(S, lv):
    j = np.searchsorted(lv, S - 1e-15 * np.maximum(S, 1e-300), side='left'); j = np.minimum(j, len(lv) - 1); return lv[j]


def run(ds):
    t0 = time.time(); db = np.load(f'{PREP}/{ds}-db.npy', mmap_mode='r'); N, d = db.shape; n = min(NSUB, N); CH = min(100_000, 40_000_000 // d)
    X = np.asarray(db[:n]).astype(np.float64); R = np.sqrt((X ** 2).sum(1)); Q = np.load(f'{PREP}/{ds}-query.npy')[:NQ].astype(np.float64); rows = []
    for pair, rule in (('pca', 'alpha1'), ('rot', 'uniform')):
        c = FairCodec.load(f'{FROZEN}/codec_{ds}__{pair}.npz'); Y = c.coords(X, R); Qt = c.qcoords(Q)
        for B in (1, 2, 3, 4, 6, 8):
            bits = c.alloc(B, rule); pl = c.plan(bits); act = pl['act']; bb = pl['b']; m2 = bb >= 2; m1 = bb == 1
            lo, W, L = pl['lo'], pl['W'], pl['L']; hi = lo + L * W; mn, mx = c.mn[act], c.mx[act]; qa = Qt[:, act]
            Z = np.setdiff1d(np.arange(d), act); qz = Qt[:, Z]; c0z = np.asarray(c.c0, np.float64)[Z]; Yz = Y[:, Z]   # components without bits (act is an index array): reconstructed as their mean
            dz = c0z[None, :] - Yz                                                                             # their deterministic error (n x |Z|)
            rho_z = np.maximum(np.abs(c0z - c.mn[Z]), np.abs(c.mx[Z] - c0z))                                 # per-component extreme of that error (opt1)
            Sdb = np.empty(N)                                            # S over the whole database (levels must bound every stored vector)
            for r0 in range(0, N, CH):
                Xb = np.asarray(db[r0:r0 + CH]).astype(np.float64); Rb = np.sqrt((Xb ** 2).sum(1)); Yb = c.coords(Xb, Rb)
                Sdb[r0:r0 + CH] = np.sqrt((residual(Yb[:, act], pl) ** 2).sum(1) + ((c0z[None, :] - Yb[:, Z]) ** 2).sum(1))
            Ya = Y[:, act]; r = residual(Ya, pl); S = np.sqrt((r ** 2).sum(1) + (dz ** 2).sum(1)); Shat = {'inf': S} | {str(k): code_up(S, levels(Sdb, k)) for k in KS}
            V = (qa[:, m2] ** 2 * (W[m2] ** 2 / 12)[None, :]).sum(1); M = (np.abs(qa[:, m2]) * (W[m2] / 2)[None, :]).max(1) if m2.any() else np.zeros(len(Q))
            qall = np.sqrt((Qt ** 2).sum(1))[:, None]; qz2 = (qz ** 2).sum(1)[:, None]; w1z = (np.abs(qz) @ rho_z)[:, None]
            for s in range(5):
                z = c.dither(bits, s); za = z[act]; Kz = np.clip(np.floor((Ya + za - lo) / W), 0, L - 1); yt = lo + (Kz + 0.5) * W - za
                err = np.zeros_like(Ya); err[:, m2] = (yt - Ya)[:, m2]
                if m1.any(): err[:, m1] = r[:, m1]
                eps = np.abs(qa @ err.T + qz @ dz.T); edge = ((Kz == 0) | (Kz == L - 1)) & m2[None, :]; Dm = edge | m1[None, :]
                qD = np.sqrt((qa ** 2) @ Dm.T.astype(np.float64) + qz2)                        # (nq, n); the zero-bit components are always in D
                rho = np.zeros_like(Ya); rho = np.where((Kz == 0) & m2[None, :], (W / 2 + np.maximum(lo - mn, 0))[None, :], rho)
                rho = np.where((Kz == L - 1) & m2[None, :], (W / 2 + np.maximum(mx - hi, 0))[None, :], rho)
                if m1.any():
                    Kd = np.clip(np.floor((Ya - lo) / W), 0, L - 1); lev = pl['a'] + pl['w'] * Kd; split = lo + W
                    rho[:, m1] = np.where(Kd == 0, np.maximum(np.abs(lev - mn), np.abs(lev - split)), np.maximum(np.abs(lev - split), np.abs(lev - mx)))[:, m1]
                w1 = np.abs(qa) @ rho.T + w1z
                for dl in DELTAS:
                    tau = tau_of(V, M, dl)[:, None]; tden = np.maximum(tau, 1e-300)
                    def add(opt, mask, sb, wd):
                        hw = tau + wd; rows.append(dict(dataset=ds, pair=pair, B=B, seed=s, nominal=1 - dl, option=opt, mask=mask, s_bits=sb,
                                                        coverage=round(float((eps <= hw * (1 + 1e-12)).mean()), 6),
                                                        width_over_tau=round(float((hw / tden).mean()), 4) if V.max() > 0 else float('inf'),
                                                        frac_S_zero=round(float((S == 0).mean()), 4)))
                    add('opt1', 'code', 0, w1)
                    for key, Sv in Shat.items():
                        add('opt2', 'code', key, qD * Sv[None, :]); add('opt2', 'all', key, qall * Sv[None, :])
            sel = [x for x in rows[-(5 * 3 * 9):] if x['nominal'] == 0.99 and x['seed'] == 0]
            print(f'  {ds} {pair} B{B}: ' + ' | '.join(f"{x['option']}/{x['mask']}/{x['s_bits']} x{x['width_over_tau']:.2f} cov {x['coverage']:.4f}" for x in sel if x['mask'] == 'code' or x['s_bits'] == '4') + f' [{time.time()-t0:.0f}s]', flush=True)
    with open(f'{OUT}/{ds}.csv.tmp', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    os.replace(f'{OUT}/{ds}.csv.tmp', f'{OUT}/{ds}.csv')


if __name__ == '__main__':
    for ds in (sys.argv[1].split(',') if len(sys.argv) > 1 else ORDER):
        if os.path.exists(f'{OUT}/{ds}.csv'): continue
        while not all(os.path.exists(f'{FROZEN}/codec_{ds}__{p}.json') for p in ('pca', 'rot')): time.sleep(60)
        try: os.close(os.open(f'{OUT}/{ds}.lock', os.O_CREAT | os.O_EXCL))
        except FileExistsError: continue
        while avail() < 8: time.sleep(120)
        try: run(ds)
        finally: os.remove(f'{OUT}/{ds}.lock')
