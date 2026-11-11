#!/usr/bin/env python3
"""Per-query hit counts of the slope sweep at B = 3, the input of the out-of-sample check of Section V-C. For every slope alpha of
the sweep (0 to 2 in steps of 0.25, and 0.5 to 1.5 in steps of 0.05) and every query, the number of true top-10 neighbors among
the top 10 of the full scan of the PCA basis with slope alpha (unified quantizer, dither seed 0). Reads the top-10 ids saved by
fullscan.py and fullscan_slope_fine.py (ids_<dataset>__{pca,pcaslope,pcafine}.npz) and the ground truth <dataset>-gt-l2.npy.
Writes $PCARR_OUT/fullscan/slope_hits/<dataset>.npz with 'alphas' (float64) and 'hits' (uint8, n_alpha x n_queries).
Usage: python3 experiments/fullscan/slope_hits.py [--datasets...] [--ids DIR] (DIR holds the ids files; default $PCARR_OUT/fullscan)"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import numpy as np  # noqa: E402
from pcafair.config import PREPARED, TEN  # noqa: E402
from pcafair.runio import parse_args  # noqa: E402

COARSE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SWEEP = sorted(set(COARSE) | {round(0.5 + 0.05 * k, 2) for k in range(21)})
K = 10


def piece(alpha): return 'pca' if alpha == 1.0 else ('pcaslope' if alpha in COARSE else 'pcafine')


if __name__ == '__main__':
    a = parse_args(__doc__.splitlines()[0], TEN, lambda ap: ap.add_argument('--ids', default=None))
    ids_dir = a.ids or str(a.out / 'fullscan')
    for ds in a.datasets:
        gt = np.load(PREPARED / f'{ds}-gt-l2.npy')[:, :K]; nq = gt.shape[0]; H = np.zeros((len(SWEEP), nq), np.uint8); cache = {}
        for i, alpha in enumerate(SWEEP):
            p = piece(alpha)
            if p not in cache: cache[p] = np.load(f'{ids_dir}/ids_{ds}__{p}.npz')
            ids = cache[p][f'unified_B3_a{alpha}']
            H[i] = [len(set(ids[q].tolist()) & set(gt[q].tolist())) for q in range(nq)]
        os.makedirs(a.out / 'fullscan' / 'slope_hits', exist_ok=True)
        np.savez_compressed(a.out / 'fullscan' / 'slope_hits' / f'{ds}.npz', alphas=np.array(SWEEP), hits=H)
        print(ds, 'recall at slope 1: %.2f%%' % (100 * H[SWEEP.index(1.0)].mean() / K), flush=True)
