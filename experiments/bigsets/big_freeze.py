#!/usr/bin/env python3
"""freeze the three codecs (pca alpha = 1 / rot uniform / orig alpha = 1) of a large dataset exactly as experiments/codec/freeze_codecs.py
(norm division, range-selected cells, unified quantizer: b >= 2 shared subtractive dither + fixed closed-form gain, b = 1
deterministic split + gain, b = 0 the component mean), from the training sample of pass A's stage 1 and pass A's full-database
statistics. Gate per codec: save -> load -> the unified decode of the first 2,000 database rows at B = 3 (dither seed 0) is identical.
Output results/frozen/codec_<ds>__<basis>.npz +.json. Usage: python3 big_freeze.py <ds> [--smoke]"""
import os, sys, json, time
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
FZ = B.FROZEN; os.makedirs(FZ, exist_ok=True)


def main(ds, tag=''):
    s = B.spec(ds); name = ds + tag
    st1 = dict(np.load(f'{B.WORK}/{ds}-stage1{tag}.npz')); pa = dict(np.load(f'{B.WORK}/{ds}-passA{tag}.npz')); N = int(pa['N'])
    Up, Rp, Ur, Rr = B.prepared(s, st1['raw'])
    X0 = next(B.iter_chunks(s['base'], s['d'], 0, 2000, 2000))[1]; U0p, R0p, U0r, R0r = B.prepared(s, X0)
    for basis in ('pca', 'rot', 'orig'):
        tg = f'{name}__{basis}'
        if os.path.exists(f'{FZ}/codec_{tg}.json'): continue
        B.gate(f'freeze {tg}'); B.wait_mem(12, f'freeze {tg}'); t0 = time.time()
        if basis == 'orig': M, U0, Rt, Y0 = None, Ur, Rr, U0r
        else:
            M = st1['Mp'] if basis == 'pca' else st1['H']; U0, Rt, Y0 = Up, Rp, U0p @ M
        Str = (U0 @ M if M is not None else U0).astype(np.float32)
        np.savez(f'{B.WORK}/{tg}-sample.npz', Str=Str, ti=st1['ti'], Rt=Rt)
        c = B.freeze_big(name, basis, Str, st1['ti'], Rt, M, pa[f'{basis}|mn'], pa[f'{basis}|mx'], pa[f'{basis}|s1'], pa[f'{basis}|s2'], N)
        tf = time.time() - t0
        c.save(f'{FZ}/codec_{tg}.npz'); c2 = B.BigCodec.load(f'{FZ}/codec_{tg}.npz')
        rule = 'uniform' if basis == 'rot' else 'alpha1'; bits = c.alloc(3, rule); pl = c.plan(bits); z = c.dither(bits, 0)
        Y = Y0.astype(np.float32); b2 = c2.alloc(3, rule)
        same = bool(np.array_equal(c.decode_unified(Y, pl, z), c2.decode_unified(Y, c2.plan(b2), c2.dither(b2, 0))))
        info = dict(dataset=name, basis=basis, design=' (as experiments/codec/freeze_codecs.py): norm division + range selection + unified (fixed closed-form gain) + zero-bit mean + alpha1/uniform',
                    d=int(c.d), N=int(c.N), training_rows=int(len(c.Str)), seed_sample=7, rotation='haar_fullscan(d, 0) in the prepared coordinates' if basis == 'rot' else None,
                    fit_seconds=round(tf, 1), gate_save_load_identical=same,
                    active_bits={f'B{Bb}': [int((c.alloc(Bb, rule) > 0).sum()), int(c.alloc(Bb, rule).max()), int(c.alloc(Bb, rule).sum())] for Bb in (1, 2, 3, 4, 6, 8)},
                    median_fixed_gain={f'b{b}': round(float(np.median(c.g_fix[:, b])), 4) for b in (2, 3, 4, 8)})
        json.dump(info, open(f'{FZ}/codec_{tg}.json', 'w'), indent=1)
        B.log(f'freeze {tg}: fitted {tf:.0f}s, gate {same}, B=3 active/max/sum {info["active_bits"]["B3"]}, gains b2/b3 {info["median_fixed_gain"]["b2"]}/{info["median_fixed_gain"]["b3"]}')
        if not same: raise SystemExit(f'gate failed for {tg}')
        del c, c2


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main(sys.argv[1], '.smoke' if '--smoke' in sys.argv else '')
