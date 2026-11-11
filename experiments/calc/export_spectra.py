#!/usr/bin/env python3
"""The inputs of Fig. 2 without the codecs: for every dataset and each of its three frozen codecs (basis pca, rot, orig), the
variance of every coding coordinate over the whole database and the bits the design gives it at B = 3 (PCA and the original
coordinates: slope 1; Random Rotation: uniform). Writes $PCARR_OUT/spectra/<dataset>.csv with columns basis, index, var, b3.
Usage: python3 experiments/calc/export_spectra.py [--datasets audio,sift100m] (needs the frozen codecs in $PCARR_FROZEN)"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import numpy as np  # noqa: E402
from pcafair.config import FROZEN, TEN, LARGE  # noqa: E402
from pcafair.runio import parse_args, write_csv  # noqa: E402
from pcafair.codec import FairCodec  # noqa: E402

RULE = {'pca': 'alpha1', 'rot': 'uniform', 'orig': 'alpha1'}

if __name__ == '__main__':
    a = parse_args(__doc__.splitlines()[0], TEN + LARGE)
    for ds in a.datasets:
        rows = []
        for basis in ('pca', 'rot', 'orig'):
            if ds in LARGE:
                from pcafair.bigsets import BigCodec
                c = BigCodec.load(f'{FROZEN}/codec_{ds}__{basis}.npz')
            else:
                c = FairCodec.load(f'{FROZEN}/codec_{ds}__{basis}.npz')
            var = np.asarray(c.var, np.float64); b3 = c.alloc(3, RULE[basis])
            assert int(b3.sum()) == 3 * c.d, (ds, basis, int(b3.sum()))
            rows += [dict(basis=basis, index=i, var=repr(float(var[i])), b3=int(b3[i])) for i in range(c.d)]
        write_csv(a.out / 'spectra' / f'{ds}.csv', ['basis', 'index', 'var', 'b3'], rows, crlf=False); print(ds, len(rows), 'rows', flush=True)
