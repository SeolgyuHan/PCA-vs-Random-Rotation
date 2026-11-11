#!/usr/bin/env python3
"""youtube: YouTube-8M video-level features (mean_rgb, 1024 dimensions; https://research.google.com/youtube8m/download.html).
Reads the train*.tfrecord shards listed in data/youtube8m_shards.json (file name and md5; download exactly these shards
into one directory), takes the first 1,000,000 records in reading order, and writes the first 1,000 as queries and the
other 999,000 as the base: $PCARR_RAW/youtube/youtube_{base,query}.fvecs.
Usage: python3 data/youtube8m_extract.py <directory with the shards>
Requires tensorflow (record parsing only)."""
import glob
import os
import struct
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcafair.config import RAW  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
SRC = sys.argv[1]
OUT = RAW / 'youtube'
N_TOTAL, N_QUERY, DIM = 1_000_000, 1_000, 1024


def parse(rec):
    feat = {"mean_rgb": tf.io.FixedLenFeature([DIM], tf.float32)}
    return tf.io.parse_single_example(rec, feat)["mean_rgb"]


files = sorted(glob.glob(f"{SRC}/train*.tfrecord"))
ds = tf.data.TFRecordDataset(files, num_parallel_reads=8).map(parse).prefetch(2048)

buf = np.empty((N_TOTAL, DIM), dtype=np.float32)
i = 0
for v in ds:
    buf[i] = v.numpy()
    i += 1
    if i % 100_000 == 0:
        print(f"{i}/{N_TOTAL}")
    if i == N_TOTAL:
        break
assert i == N_TOTAL, f"only {i} records in the shards, fewer than 1,000,000"
os.makedirs(OUT, exist_ok=True)   # only once the input is complete

query, base = buf[:N_QUERY], buf[N_QUERY:]


def write_fvecs(path, x):
    with open(path, "wb") as f:
        for row in x:
            f.write(struct.pack("<i", DIM)); f.write(row.tobytes())


write_fvecs(f"{OUT}/youtube_base.fvecs", base)
write_fvecs(f"{OUT}/youtube_query.fvecs", query)
print(f"base {base.shape}, query {query.shape} -> {OUT}")
