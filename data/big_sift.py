#!/usr/bin/env python3
"""sift100m: the first 100,000,000 vectors of BIGANN (ANN_SIFT1B) and its 10,000 public queries, as .fvecs for the large-dataset
passes (experiments/bigsets). Reads the big-ann-benchmarks files
    base.1B.u8bin         (uint8, header n:int32 d:int32; only the first 100M rows are needed: 12.8 GB of the 132 GB file)
    query.public.10K.u8bin
and writes $PCARR_BIG_RAW/SIFT100M/base.100M.fvecs and query.10K.fvecs (float32; the experiments use the first 1,000 queries).
The official 100M-subset ground truth is not needed: experiments/bigsets/big_passA.py computes the exact top-100 by L2.
Usage: python3 data/big_sift.py <base.1B.u8bin or its first 12.8 GB> <query.public.10K.u8bin>"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import numpy as np  # noqa: E402
from pcafair.config import BIG_RAW  # noqa: E402

N = 100_000_000; CH = 1_000_000


def convert(src, dst, n_rows):
    with open(src, 'rb') as f:
        n, d = np.fromfile(f, np.int32, 2); n = min(int(n), n_rows); d = int(d)
        with open(dst, 'wb') as g:
            for r0 in range(0, n, CH):
                m = min(CH, n - r0); x = np.fromfile(f, np.uint8, m * d).reshape(m, d)
                out = np.empty((m, d + 1), np.float32); out[:, 0] = np.array([d], np.int32).view(np.float32)[0]; out[:, 1:] = x
                out.tofile(g)
    print(dst, n, 'x', d, flush=True)


if __name__ == '__main__':
    out = BIG_RAW / 'SIFT100M'; out.mkdir(parents=True, exist_ok=True)
    convert(sys.argv[1], out / 'base.100M.fvecs', N); convert(sys.argv[2], out / 'query.10K.fvecs', 10_000)
