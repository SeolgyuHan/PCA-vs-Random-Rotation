#!/usr/bin/env python3
"""fit and freeze the codecs of the confirmed design for every experiment of the re-measurement.
Design: norm division (u = x/||x||), basis 'pca' (eigenvectors of Cov(u)) | 'rot' (haar_fullscan(d, 0)) | 'orig' (original
coordinates, Figure 1); range-selected cells; unified quantizer (b >= 2: shared subtractive dither + fixed closed-form gain;
b = 1: deterministic split + gain; b = 0: component mean); no norm weighting; allocation alpha = 1 (pca, orig) / uniform (rot).
Training sample: database rows, numpy default_rng(7), min(200k, 1.2e8/d) rows. Output frozen/codec_<ds>__<basis>.npz +.json.
Gate per codec: save -> load -> the unified decode of 2,000 database rows at B = 3 (dither seed 0) is identical.
Usage: python3 freeze_codecs.py [ds,ds,...] (lock files; several workers may share one list)"""
import os, sys, json, time, numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair.config import PREPARED, FROZEN, OUT as OUT_ROOT, RESULTS  # noqa: E402
from pcafair.codec import FairCodec  # noqa: E402
PREP = str(PREPARED)
OUT = str(FROZEN); os.makedirs(OUT, exist_ok=True)
SETS = ['audio', 'fashion-mnist', 'mnist', 'msong', 'word2vec', 'gist', 'youtube', 'msmarco', 'openai1536', 'openai3072']
def avail(): return int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 1048576
todo = sys.argv[1].split(',') if len(sys.argv) > 1 else SETS
for ds in todo:
    for basis in ('pca', 'rot', 'orig'):
        tag = f'{ds}__{basis}'
        if os.path.exists(f'{OUT}/codec_{tag}.json'): continue
        while avail() < 14: print(f'  [{tag}] waiting for 14 GB available', flush=True); time.sleep(120)
        try: os.close(os.open(f'{OUT}/{tag}.lock', os.O_CREAT | os.O_EXCL))
        except FileExistsError: continue
        t0 = time.time()
        c = FairCodec(ds, basis, True, keep_sample=True, zero='mean', gains='closed').fit(); tf = time.time() - t0
        c.save(f'{OUT}/codec_{tag}.npz'); c2 = FairCodec.load(f'{OUT}/codec_{tag}.npz')
        rule = 'uniform' if basis == 'rot' else 'alpha1'; bits = c.alloc(3, rule); pl = c.plan(bits); z = c.dither(bits, 0)
        X = c.rows(0, 2000); r = np.sqrt((X ** 2).sum(1)); Y = c.coords(X, r).astype(np.float32)
        same = bool(np.array_equal(c.decode_unified(Y, pl, z), c2.decode_unified(Y, c2.plan(c2.alloc(3, rule)), c2.dither(c2.alloc(3, rule), 0))))
        info = dict(dataset=ds, basis=basis, design='norm division + range selection + unified (fixed closed-form gain) + zero-bit mean + alpha1/uniform',
                    d=int(c.d), N=int(c.N), training_rows=int(len(c.Str)), seed_sample=7, rotation='haar_fullscan(d, 0)' if basis == 'rot' else None,
                    fit_seconds=round(tf, 1), gate_save_load_identical=same,
                    active_bits={f'B{B}': [int((c.alloc(B, rule) > 0).sum()), int(c.alloc(B, rule).max()), int(c.alloc(B, rule).sum())] for B in (1, 2, 3, 4, 6, 8)},
                    median_fixed_gain={f'b{b}': round(float(np.median(c.g_fix[:, b])), 4) for b in (2, 3, 4, 8)})
        json.dump(info, open(f'{OUT}/codec_{tag}.json', 'w'), indent=1); os.remove(f'{OUT}/{tag}.lock')
        print(f'{tag}: fitted {tf:.0f}s, gate {same}, alpha/uniform B=3 active {info["active_bits"]["B3"]}, gains b2/b3 {info["median_fixed_gain"]["b2"]}/{info["median_fixed_gain"]["b3"]} [{time.time()-t0:.0f}s]', flush=True)
        del c, c2
