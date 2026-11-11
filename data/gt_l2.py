#!/usr/bin/env python3
"""Exact L2 ground truth of a prepared dataset: for every query, the ids of its 100 nearest database rows by squared L2 distance in
the prepared coordinates (centering and the orthogonal projection preserve L2 distances, and the constant coordinates that
prepare.py drops contribute nothing). Writes $PCARR_PREPARED/<dataset>-gt-l2.npy (int32, n_queries x 100, nearest first).
The paper's runs used the official ground-truth files of the collections that provide one (GQR, BIGANN) and this computation
otherwise; the two agree up to the order of exact ties. Usage: python3 data/gt_l2.py <dataset> [<dataset>...]"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import numpy as np  # noqa: E402
from pcafair.config import PREPARED  # noqa: E402

KGT = 100

if __name__ == '__main__':
    for ds in sys.argv[1:]:
        db = np.load(PREPARED / f'{ds}-db.npy', mmap_mode='r'); Q = np.load(PREPARED / f'{ds}-query.npy').astype(np.float64)
        N, d = db.shape; nq = Q.shape[0]; ch = max(1000, min(200_000, int(4e8 // d))); q2 = (Q ** 2).sum(1)
        best_d = np.full((nq, KGT), np.inf); best_i = np.zeros((nq, KGT), np.int64)
        for r0 in range(0, N, ch):
            X = np.asarray(db[r0:r0 + ch], np.float64); D = (X ** 2).sum(1)[None, :] - 2.0 * (Q @ X.T) + q2[:, None]
            cand_d = np.concatenate([best_d, D], 1); cand_i = np.concatenate([best_i, np.arange(r0, r0 + X.shape[0])[None, :].repeat(nq, 0)], 1)
            part = np.argpartition(cand_d, KGT - 1, axis=1)[:, :KGT]
            best_d = np.take_along_axis(cand_d, part, 1); best_i = np.take_along_axis(cand_i, part, 1)
        order = np.lexsort((best_i, best_d), axis=1) if False else np.argsort(best_d, axis=1, kind='stable')
        gt = np.take_along_axis(best_i, order, 1).astype(np.int32)
        np.save(PREPARED / f'{ds}-gt-l2.npy', gt); print(ds, gt.shape, 'written', flush=True)
