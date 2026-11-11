#!/usr/bin/env python3
"""original coordinates + alpha = 1 (frozen 'orig' codec) for the large datasets, exactly big_score.py's two parts in one run
(E2 score error, every query x 10,000 rows sampled with rng 123, B 1-8 unified (+ det at 1, 2, 3, 4, 6, 8); decomp with c0_tr from
the codec's training sample, B 1, 2, 3, 4, 6, 8), with the rows and queries of the orig arm of big_passB_budget.py (Ur, Rr of pcafair.bigsets.prepared;
centred raw queries). Writes results/score_error_orig/<ds>.csv and results/decomp_orig/<ds>.csv only.
Usage: python3 big_score_orig.py <ds> [--threads T]"""
import os, sys, csv, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
NS = 10_000; B_ALL = (1, 2, 3, 4, 5, 6, 7, 8); B_DEC = (1, 2, 3, 4, 6, 8)
O1 = f'{B.OUT}/score_error_orig'; O2 = f'{B.OUT}/decomp_orig'
for _d in (O1, O2): os.makedirs(_d, exist_ok=True)
F1 = ['dataset', 'pair', 'mode', 'B', 'J_meas', 'bq', 'J_sep', 'J_hr', 'J_noise', 'c0', 'ER2', 'lam_mean_u', 'd', 'n_active']
F2 = ['dataset', 'pair', 'B', 'J_meas', 'J_full', 'J_sep', 'J_hr', 'J_A2', 'J_A2tr', 'J_qh', 'Jm_h2', 'c0', 'c0_tr', 'k_A2', 'E2_J_meas', 'E2_J_hr']


def c0_of(D, bits, s2):
    on = bits >= 1
    return float(np.exp(np.mean(np.log(np.maximum(D[on] * 4.0 ** bits[on] / s2[on], 1e-300))))) if on.any() else 1.0


def model(c0, w, s2, bits): return float(w @ np.where(bits >= 1, c0 * s2 * 4.0 ** (-bits.astype(np.float64)), s2))


def main(ds):
    B.wait_mem(8, f'score orig {ds}'); t0 = time.time(); s = B.spec(ds); pair, rule = 'orig', 'alpha1'
    N = int(np.load(f'{B.WORK}/{ds}-passA.npz')['N']); d = s['d']
    idx = np.sort(np.random.default_rng(123).choice(N, min(NS, N), replace=False))
    raw = B.read_rows(s['base'], d, idx); Up, Rp, Ur, Rr = B.prepared(s, raw.astype(np.float32)); del raw, Up, Rp
    Y = Ur.astype(np.float64); R = np.asarray(Rr, np.float64); ER2 = float(np.mean(R ** 2)); Y32 = Ur.astype(np.float32); del Ur
    Qp, Qr = B.prepared_queries(s, B.read_queries(s)); nq = len(Qr)
    c = B.BigCodec.load(f'{B.FROZEN}/codec_{ds}__{pair}.npz'); Qt = c.qcoords(Qr)
    sig2 = np.maximum(np.asarray(c.var, np.float64), 1e-300); q2 = (Qt ** 2).mean(0)
    print(f'  {ds} orig: N {N} d {d} nq {nq} sample {len(idx)} [{time.time()-t0:.0f}s]', flush=True)
    rows1 = []
    for Bb in B_ALL:
        bits = c.alloc(Bb, rule); pl = c.plan(bits); act = pl['act']; z = c.dither(bits, 0)
        for mode in (('unified', 'det') if Bb in B_DEC else ('unified',)):
            Yh = (c.decode_unified(Y32, pl, z) if mode == 'unified' else c.decode_fast(Y32, pl, 'det')).astype(np.float64)
            E = Yh - Y; Jq, mq = np.zeros(nq), np.zeros(nq)
            for q0 in range(0, nq, 500):
                P = -2.0 * R[None, :] * (Qt[q0:q0 + 500] @ E.T); Jq[q0:q0 + 500] = P.var(axis=1); mq[q0:q0 + 500] = P.mean(axis=1)
            Di = E.var(0); Jsep = 4 * ER2 * float(q2 @ Di)
            on = bits >= 1; c0 = c0_of(Di, bits, sig2)
            hr = np.where(bits >= 1, c0 * sig2 * 4.0 ** (-bits.astype(np.float64)), sig2); Jhr = 4 * ER2 * float(q2 @ hr)
            m2 = pl['b'] >= 2; gfx = c.g_fix[act[m2], pl['b'][m2]]
            Jn = 4 * ER2 * float(q2[act[m2]] @ (pl['W'][m2] ** 2 / (12 * gfx ** 2))) if m2.any() else 0.0
            rows1.append(dict(dataset=ds, pair=pair, mode=mode, B=Bb, J_meas=f'{Jq.mean():.6e}', bq=f'{(mq ** 2).mean():.6e}', J_sep=f'{Jsep:.6e}',
                              J_hr=f'{Jhr:.6e}', J_noise=f'{Jn:.6e}', c0=f'{c0:.4f}', ER2=f'{ER2:.6e}', lam_mean_u=f'{sig2.mean():.6e}', d=d, n_active=int(on.sum())))
            print(f'  {ds} {pair} {mode:7s} B{Bb}: J_meas {Jq.mean():.4e} J_meas/J_sep {Jq.mean()/Jsep:.3f} J_meas/J_hr {Jq.mean()/Jhr:.3f} c0 {c0:.3f} [{time.time()-t0:.0f}s]', flush=True)
    with open(f'{O1}/{ds}.csv.tmp', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=F1); w.writeheader(); w.writerows(rows1)
    os.replace(f'{O1}/{ds}.csv.tmp', f'{O1}/{ds}.csv')
    e2 = {(r['pair'], r['mode'], int(r['B'])): r for r in rows1}
    h = nq // 2; rows2 = []; s2 = sig2; q2a = (Qt[:h] ** 2).mean(0); k = float(q2.sum() / s2.sum())
    St = c.ensure_sample().astype(np.float32)
    for Bb in B_DEC:
        bits = c.alloc(Bb, rule); pl = c.plan(bits); z = c.dither(bits, 0); bf = bits.astype(np.float64)
        E = c.decode_unified(Y32, pl, z).astype(np.float64) - Y
        Jq = np.zeros(nq)
        for q0 in range(0, nq, 500):
            P = -2.0 * R[None, :] * (Qt[q0:q0 + 500] @ E.T); Jq[q0:q0 + 500] = P.var(axis=1)
        Ec = E - E.mean(0); Sig = Ec.T @ Ec / len(Ec); D = np.diag(Sig).copy()
        Jfull = 4 * ER2 * float(np.mean(np.einsum('qi,qi->q', Qt @ Sig, Qt)))
        Jsep = 4 * ER2 * float(q2 @ D); c0 = c0_of(D, bf, s2)
        Jhr = 4 * ER2 * model(c0, q2, s2, bf); JA2 = 4 * ER2 * model(c0, k * s2, s2, bf)
        Et = c.decode_unified(St, pl, z).astype(np.float64) - St.astype(np.float64); c0t = c0_of(Et.var(0), bf, s2)
        JA2t = 4 * ER2 * model(c0t, k * s2, s2, bf); Jqh = 4 * ER2 * model(c0t, q2a, s2, bf)
        r2 = e2.get((pair, 'unified', Bb), {})
        rows2.append(dict(dataset=ds, pair=pair, B=Bb, J_meas=f'{Jq.mean():.6e}', J_full=f'{Jfull:.6e}', J_sep=f'{Jsep:.6e}', J_hr=f'{Jhr:.6e}',
                          J_A2=f'{JA2:.6e}', J_A2tr=f'{JA2t:.6e}', J_qh=f'{Jqh:.6e}', Jm_h2=f'{Jq[h:].mean():.6e}', c0=f'{c0:.4f}', c0_tr=f'{c0t:.4f}',
                          k_A2=f'{k:.6e}', E2_J_meas=r2.get('J_meas', ''), E2_J_hr=r2.get('J_hr', '')))
        print(f'  {ds} {pair} decomp B{Bb}: A0 {Jq.mean()/Jfull:.3f} kappa {Jfull/Jsep:.3f} A3p {Jsep/Jhr:.3f} A2 {Jhr/JA2:.3f} | c0 {c0:.3f} c0_tr {c0t:.3f} [{time.time()-t0:.0f}s]', flush=True)
    del St
    with open(f'{O2}/{ds}.csv.tmp', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=F2); w.writeheader(); w.writerows(rows2)
    os.replace(f'{O2}/{ds}.csv.tmp', f'{O2}/{ds}.csv'); B.log(f'score error orig {ds}: done, B=3 J_meas {float(e2[("orig","unified",3)]["J_meas"]):.4e} [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    a = sys.argv; T = int(a[a.index('--threads') + 1]) if '--threads' in a else 4
    with threadpool_limits(limits=T):
        main(a[1])
