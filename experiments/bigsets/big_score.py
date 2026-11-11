#!/usr/bin/env python3
""" (Figure 3 and the S of Table III): the score error within each query with the frozen codecs, exactly the protocols of
experiments/score_error/score_error.py (E2: every query x 10,000 database rows sampled with numpy default_rng(123); J_meas, bq, J_sep, J_hr,
J_noise, c0; pairs pca alpha = 1 / rot uniform; B in {1, 2, 3, 4, 6, 8}; modes unified and det) and results/decomp_G1045.py
(the telescoping J_meas -> J_full -> J_sep -> J_hr -> J_A2 and the strict before-encoding prediction J_A2tr with c0 from the
codec's training sample; A2 exponent). The rows are read from the raw file and put in the prepared coordinates (pcafair.bigsets.prepared).
Output results/score_error/<ds>.csv, results/decomp/<ds>.csv and <ds>.a2.json. Usage: python3 big_score.py <ds> [--smoke]"""
import os, sys, csv, json, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
B.ensure_dirs()
from threadpoolctl import threadpool_limits  # noqa: E402
NS = 10_000; BUDGETS = (1, 2, 3, 4, 6, 8)
O1 = f'{B.OUT}/score_error'; O2 = f'{B.OUT}/decomp'
for _d in (O1, O2): os.makedirs(_d, exist_ok=True)
F1 = ['dataset', 'pair', 'mode', 'B', 'J_meas', 'bq', 'J_sep', 'J_hr', 'J_noise', 'c0', 'ER2', 'lam_mean_u', 'd', 'n_active']
F2 = ['dataset', 'pair', 'B', 'J_meas', 'J_full', 'J_sep', 'J_hr', 'J_A2', 'J_A2tr', 'J_qh', 'Jm_h2', 'c0', 'c0_tr', 'k_A2', 'E2_J_meas', 'E2_J_hr']


def c0_of(D, bits, s2):
    on = bits >= 1
    return float(np.exp(np.mean(np.log(np.maximum(D[on] * 4.0 ** bits[on] / s2[on], 1e-300))))) if on.any() else 1.0


def model(c0, w, s2, bits): return float(w @ np.where(bits >= 1, c0 * s2 * 4.0 ** (-bits.astype(np.float64)), s2))


def main(ds, tag=''):
    B.wait_mem(8, f'score {ds}{tag}'); t0 = time.time(); s = B.spec(ds); name = ds + tag; N = int(np.load(f'{B.WORK}/{ds}-passA{tag}.npz')['N']); d = s['d']
    idx = np.sort(np.random.default_rng(123).choice(N, min(NS, N), replace=False))
    raw = B.read_rows(s['base'], d, idx); xr = raw.astype(np.float64) - s['mu']
    X = (xr @ s['W']).astype(np.float32).astype(np.float64); del xr                       # prepared rows, as $PCARR_PREPARED/<ds>-db.npy
    R = np.sqrt((X ** 2).sum(1)); ER2 = float(np.mean(R ** 2))
    Q = B.prepared_queries(s, B.read_queries(s))[0]; nq = len(Q)
    rows1 = []
    for pair, rule in (('pca', 'alpha1'), ('rot', 'uniform')):                              # ---- score_error.run
        c = B.BigCodec.load(f'{B.FROZEN}/codec_{name}__{pair}.npz'); Y = c.coords(X, R); Qt = c.qcoords(Q); Y32 = Y.astype(np.float32)
        sig2 = np.maximum(np.asarray(c.var, np.float64), 1e-300); q2 = (Qt ** 2).mean(0)
        for Bb in BUDGETS:
            bits = c.alloc(Bb, rule); pl = c.plan(bits); act = pl['act']; z = c.dither(bits, 0)
            for mode in ('unified', 'det'):
                Yh = (c.decode_unified(Y32, pl, z) if mode == 'unified' else c.decode_fast(Y32, pl, 'det')).astype(np.float64)
                E = Yh - Y; Jq, mq = np.zeros(nq), np.zeros(nq)
                for q0 in range(0, nq, 500):
                    P = -2.0 * R[None, :] * (Qt[q0:q0 + 500] @ E.T); Jq[q0:q0 + 500] = P.var(axis=1); mq[q0:q0 + 500] = P.mean(axis=1)
                Di = E.var(0); Jsep = 4 * ER2 * float(q2 @ Di)
                on = bits >= 1; c0 = float(np.exp(np.mean(np.log(np.maximum(Di[on] * 4.0 ** bits[on] / sig2[on], 1e-300))))) if on.any() else 1.0
                hr = np.where(bits >= 1, c0 * sig2 * 4.0 ** (-bits.astype(np.float64)), sig2); Jhr = 4 * ER2 * float(q2 @ hr)
                m2 = pl['b'] >= 2; gfx = c.g_fix[act[m2], pl['b'][m2]]
                Jn = 4 * ER2 * float(q2[act[m2]] @ (pl['W'][m2] ** 2 / (12 * gfx ** 2))) if m2.any() else 0.0
                rows1.append(dict(dataset=ds, pair=pair, mode=mode, B=Bb, J_meas=f'{Jq.mean():.6e}', bq=f'{(mq ** 2).mean():.6e}', J_sep=f'{Jsep:.6e}',
                                  J_hr=f'{Jhr:.6e}', J_noise=f'{Jn:.6e}', c0=f'{c0:.4f}', ER2=f'{ER2:.6e}', lam_mean_u=f'{sig2.mean():.6e}', d=d, n_active=int(on.sum())))
        print(f'  score error {name} {pair} done [{time.time()-t0:.0f}s]', flush=True)
    with open(f'{O1}/{name}.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=F1); w.writeheader(); w.writerows(rows1)
    e2 = {(r['pair'], r['mode'], int(r['B'])): r for r in rows1}
    h = nq // 2; rows2 = []; a2 = {}
    for pair, rule in (('pca', 'alpha1'), ('rot', 'uniform')):                              # ---- decomp.run
        c = B.BigCodec.load(f'{B.FROZEN}/codec_{name}__{pair}.npz'); Y = c.coords(X, R); Qt = c.qcoords(Q); Y32 = Y.astype(np.float32)
        s2 = np.maximum(np.asarray(c.var, np.float64), 1e-300); q2 = (Qt ** 2).mean(0); q2a = (Qt[:h] ** 2).mean(0); k = float(q2.sum() / s2.sum())
        if pair == 'pca':
            keep = s2 > 1e-8 * s2.max(); sl = np.polyfit(np.log(s2[keep]), np.log(np.maximum(q2[keep], 1e-300)), 1)[0]
            a2 = dict(dataset=ds, exponent=round(float(sl), 4), n_comp=int(keep.sum()), nq=nq)
        St = c.ensure_sample().astype(np.float32)
        for Bb in BUDGETS:
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
        del St
    with open(f'{O2}/{name}.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=F2); w.writeheader(); w.writerows(rows2)
    json.dump(a2, open(f'{O2}/{name}.a2.json', 'w'))
    m = {(r['pair'], int(r['B'])): r for r in rows2}
    B.log(f'score error {name}: B=3 J_meas rot/pca {float(m[("rot",3)]["J_meas"])/float(m[("pca",3)]["J_meas"]):.3f}, strict prediction J_A2tr rot/pca '
          f'{float(m[("rot",3)]["J_A2tr"])/float(m[("pca",3)]["J_A2tr"]):.3f}, A2 exponent {a2["exponent"]} [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main(sys.argv[1], '.smoke' if '--smoke' in sys.argv else '')
