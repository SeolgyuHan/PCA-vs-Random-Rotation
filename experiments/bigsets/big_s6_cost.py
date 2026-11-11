#!/usr/bin/env python3
""" S6: the cost check of Section IV on the datasets of 10M rows or more.
Run single-threaded on one pinned core in a quiet window (the chain: taskset -c 2, one BLAS thread, after E8). Per dataset:
  pca_fit covariance of the frozen PCA codec's training sample of u (its stored rows ti: seed 7, min(200k, 1.2e8/d)) + eigh
  rot_draw haar_fullscan(d, 0), what Random Rotation+alpha=0 needs instead
  encode PCA design at B = 3: norms, u = x/||x||, one projection by the stored basis (W M, folded before the clock), dithered
            codes of the frozen codec; 3 chunks of 50,000 rows (start, middle, end; median), scaled to N (linear in N).
            Reading the rows and subtracting the mean are not timed (as in A3, whose database file is already centered).
Output results/s6_cost.csv; with --smoke one chunk of 5,000 rows and 5,000 sample rows to results/s6_cost.smoke.csv (times not
meaningful). Usage: python3 big_s6_cost.py [--smoke] ds [ds...]"""
import os, sys, csv, time, statistics
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402

SMOKE = '--smoke' in sys.argv; CH = 5_000 if SMOKE else 50_000
OUT = f'{B.OUT}/s6_cost{".smoke" if SMOKE else ""}.csv'


def centered_rows(s, ids):
    return B.read_rows(s['base'], s['d'], np.asarray(ids, np.int64)).astype(np.float64) - s['mu']


def chunk_time(s, P, r0, n, pl, z):
    X = centered_rows(s, np.arange(r0, r0 + n))                              # read and centered before the clock
    t = time.perf_counter()
    R = np.sqrt((X ** 2).sum(1)); Y = (X / R[:, None]) @ P
    act = pl['act']; bb = pl['b']; zz = np.where(bb >= 2, z[act], 0.0)
    K = np.clip(np.floor((Y[:, act] + zz - pl['lo']) / pl['W']), 0, pl['L'] - 1).astype(np.int64)
    return time.perf_counter() - t, int(K.size)


if __name__ == '__main__':
    rows = []
    for ds in [a for a in sys.argv[1:] if not a.startswith('--')]:
        s = B.spec(ds); N, d = s['N'], s['d']
        c = B.BigCodec.load(f'{B.FROZEN}/codec_{ds}__pca.npz'); bits = c.alloc(3, 'alpha1'); pl = c.plan(bits); z = c.dither(bits, 0)
        ti = np.sort(np.load(f'{B.WORK}/{ds}__pca-sample.npz')['ti']); ti = ti[:CH] if SMOKE else ti
        X = centered_rows(s, ti); U0 = (X / np.sqrt((X ** 2).sum(1))[:, None]) @ s['W']; del X       # u of the sample, not timed
        t = time.perf_counter(); ev, V = np.linalg.eigh(np.cov(U0, rowvar=False)); t_pca = time.perf_counter() - t
        del U0, V
        t = time.perf_counter(); Hm = B.FC.haar_fullscan(d, 0); t_rot = time.perf_counter() - t
        del Hm
        P = s['W'] @ np.asarray(c.M, np.float64)                                # folded basis, before the clock
        starts = [0] if SMOKE else [0, max(0, N // 2 - CH // 2), max(0, N - CH)]
        ts = [chunk_time(s, P, r0, CH, pl, z)[0] for r0 in starts]; t_enc = statistics.median(ts) * N / CH
        rows.append(dict(dataset=ds, N=N, d=d, train_rows=len(ti), pca_fit_s=f'{t_pca:.2f}', rot_draw_s=f'{t_rot:.2f}', encode_s=f'{t_enc:.1f}',
                         pca_fit_over_encode=f'{t_pca / t_enc:.5f}', extra_over_encode=f'{(t_pca - t_rot) / t_enc:.5f}'))
        B.log(f'S6 cost {ds}{" (smoke)" if SMOKE else ""}: pca_fit {t_pca:.2f}s, rot_draw {t_rot:.2f}s, encode {t_enc:.0f}s (scaled to N), fit/encode {t_pca / t_enc:.5f}')
        with open(OUT, 'w', newline='') as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
