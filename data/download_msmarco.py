#!/usr/bin/env python3
"""msmarco: MS MARCO v2.1 passages embedded with Cohere embed-english-v3 (Hugging Face Cohere/msmarco-v2.1-embed-english-v3,
now CohereLabs/msmarco-v2.1-embed-english-v3). Streams the first 10,001,000 rows of the parquet snapshot, draws 1,000
queries among them with a fixed seed (42), and writes the other 10,000,000 rows as the base; data/prepare.py keeps the
first 1,000,000 base rows. Writes $PCARR_RAW/msmarco/msmarco_{base,query}.fvecs and msmarco_query_ids.npy.
Fetch the snapshot first (about 246 GB):
  huggingface-cli download Cohere/msmarco-v2.1-embed-english-v3 --repo-type dataset \
      --revision e78737fe92ac1b783211b705c12207ca75fcc9b7 --include "passages_parquet/*"
and set MSMARCO_PARQUET_DIR to its passages_parquet directory.
Usage: MSMARCO_PARQUET_DIR=... python3 data/download_msmarco.py"""
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcafair.config import RAW  # noqa: E402

name = "msmarco"
data_dir = os.path.join(str(RAW), name) + "/"

N_BASE = 10_000_000              # base vectors
NUM_QUERIES = 1_000              # query vectors (separate from base)
N_TOTAL = N_BASE + NUM_QUERIES   # rows pulled from parquet (10,001,000)
d = 1024
SEED = 42
BATCH_SIZE = 65_536  # rows per pyarrow batch (~256 MB float32 working set)

PARQUET_DIR = os.environ.get("MSMARCO_PARQUET_DIR", "")

if not os.path.isdir(PARQUET_DIR):
    sys.exit(f"Cached parquet dir not found: {PARQUET_DIR}")

files = sorted(
    os.path.join(PARQUET_DIR, f)
    for f in os.listdir(PARQUET_DIR)
    if f.startswith("msmarco_v2.1_doc_segmented_") and f.endswith(".parquet")
)
if not files:
    sys.exit(f"No parquet files in {PARQUET_DIR}")
os.makedirs(data_dir, exist_ok=True)   # only once the input is there

# Reproducible query selection over the full N_TOTAL row range
rng = np.random.default_rng(SEED)
query_ids_sorted = np.sort(rng.permutation(N_TOTAL)[:NUM_QUERIES]).astype(np.int64)
np.save(data_dir + f"{name}_query_ids.npy", query_ids_sorted)

dim_header = struct.pack("i", d)
dim_header_arr = np.frombuffer(dim_header, dtype=np.uint8)
ROW_BYTES = 4 + 4 * d  # int32 dim + d * float32

base_path = data_dir + f"{name}_base.fvecs"
query_path = data_dir + f"{name}_query.fvecs"
base_tmp = base_path + ".tmp"
query_tmp = query_path + ".tmp"

i = 0
n_base = n_query = 0
t0 = time.time()
last_print = 0

try:
    with open(base_tmp, "wb") as fb, open(query_tmp, "wb") as fq:
        print(
            f"Reading {N_TOTAL:,} rows = {N_BASE:,} base + {NUM_QUERIES:,} queries "
            f"(seed={SEED})",
            flush=True,
        )
        print(f"Source: {PARQUET_DIR}", flush=True)
        done = False
        for path in files:
            if done:
                break
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["emb"]):
                bsz = batch.num_rows
                if i + bsz > N_TOTAL:
                    bsz = N_TOTAL - i
                    batch = batch.slice(0, bsz)

                emb_col = batch.column("emb")
                # NOTE: use flatten (not.values) —.values ignores the slice/offsets
                # and returns the full underlying chunk, which breaks on the last
                # truncated batch (where batch.slice has been applied).
                flat = emb_col.flatten().to_numpy(zero_copy_only=False)
                if flat.dtype != np.float32:
                    flat = flat.astype(np.float32, copy=False)
                if flat.size != bsz * d:
                    raise ValueError(
                        f"dim mismatch at i={i}: {flat.size} floats for {bsz} rows, "
                        f"expected {bsz * d} (d={d})"
                    )
                arr = flat.reshape(bsz, d)

                lo = np.searchsorted(query_ids_sorted, i)
                hi = np.searchsorted(query_ids_sorted, i + bsz)
                qrows_local = (query_ids_sorted[lo:hi] - i).astype(np.intp)
                base_mask = np.ones(bsz, dtype=bool)
                base_mask[qrows_local] = False

                base_rows = arr[base_mask]
                if base_rows.size:
                    nb = base_rows.shape[0]
                    buf = np.empty((nb, ROW_BYTES), dtype=np.uint8)
                    buf[:, :4] = dim_header_arr
                    buf[:, 4:] = base_rows.view(np.uint8).reshape(nb, 4 * d)
                    fb.write(buf.tobytes())
                    n_base += nb

                if qrows_local.size:
                    q_rows = arr[qrows_local]
                    nq = q_rows.shape[0]
                    qbuf = np.empty((nq, ROW_BYTES), dtype=np.uint8)
                    qbuf[:, :4] = dim_header_arr
                    qbuf[:, 4:] = q_rows.view(np.uint8).reshape(nq, 4 * d)
                    fq.write(qbuf.tobytes())
                    n_query += nq

                i += bsz
                if i - last_print >= 500_000:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (N_TOTAL - i) / rate
                    print(
                        f"  {i:,} / {N_TOTAL:,}  base={n_base:,} query={n_query:,}  "
                        f"{rate/1000:.1f}K rows/s  ETA {eta/60:.1f}m",
                        flush=True,
                    )
                    last_print = i

                if i >= N_TOTAL:
                    done = True
                    break

        fb.flush()
        fq.flush()
        os.fsync(fb.fileno())
        os.fsync(fq.fileno())

except KeyboardInterrupt:
    print(f"\nInterrupted at i={i}. Partial files left at *.tmp.", flush=True)
    sys.exit(130)
except Exception as e:
    print(f"\nError at i={i}: {type(e).__name__}: {e}", flush=True)
    raise

assert i == N_TOTAL, f"Expected i={N_TOTAL}, got {i}"
assert n_query == NUM_QUERIES, f"Expected {NUM_QUERIES} queries, got {n_query}"
assert n_base == N_BASE, f"Expected {N_BASE} base, got {n_base}"
assert n_base + n_query == N_TOTAL, f"base+query mismatch: {n_base}+{n_query} != {N_TOTAL}"
assert os.path.getsize(base_tmp) == n_base * ROW_BYTES
assert os.path.getsize(query_tmp) == n_query * ROW_BYTES

os.replace(base_tmp, base_path)
os.replace(query_tmp, query_path)

elapsed = time.time() - t0
print(
    f"Done! base={n_base:,}, query={n_query:,}  "
    f"({elapsed:.1f}s, {N_TOTAL/elapsed/1000:.1f}K rows/s avg)"
)
