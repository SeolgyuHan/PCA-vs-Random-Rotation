#!/usr/bin/env python3
"""mnist and fashion-mnist from ANN-Benchmarks (https://github.com/erikbern/ann-benchmarks):
convert <name>-784-euclidean.hdf5 to $PCARR_RAW/<name>/<name>_{base,query}.fvecs (train -> base, test -> query).
Usage: python3 data/convert_hdf5.py mnist|fashion-mnist <path to the.hdf5 file>"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcafair.config import RAW  # noqa: E402


def write_fvecs(filename, data):
    data = np.asarray(data, dtype=np.float32)
    n, d = data.shape
    with open(filename, 'wb') as f:
        for vec in data:
            f.write(np.int32(d).tobytes())
            f.write(vec.tobytes())


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('mnist', 'fashion-mnist'):
        sys.exit(__doc__)
    name, src = sys.argv[1], sys.argv[2]
    f = h5py.File(src, 'r')               # fails before anything is written when the file is missing
    out = RAW / name
    os.makedirs(out, exist_ok=True)
    write_fvecs(out / f'{name}_base.fvecs', f['train'])     # (60000, 784)
    write_fvecs(out / f'{name}_query.fvecs', f['test'])     # (10000, 784)
    print(f"{name}: base {f['train'].shape}, query {f['test'].shape} -> {out}")
