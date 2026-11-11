#!/usr/bin/env python3
"""Download openai1536 or openai3072 (Qdrant dbpedia entities, OpenAI text-embedding-3-large, 1M rows) and split it into
1,000 queries and 999,000 base vectors with a fixed seed (42; the same rows for both sizes).
Writes $PCARR_RAW/<name>/<name>_{base,query}.fvecs and <name>_query_ids.npy (the query rows).
Usage: python3 data/download_openai.py 1536|3072
The snapshot revisions are those downloaded for the paper."""
import glob
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcafair.config import RAW  # noqa: E402

SIZES = {
    '1536': dict(repo='Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M', revision='4a9731217921bc476a0f03544f11f22ae4903fa5', batch=16_384),
    '3072': dict(repo='Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M', revision='b2f301648eda4c5dbe659235589c86ffec702581', batch=8_192),
}
if len(sys.argv) != 2 or sys.argv[1] not in SIZES:
    sys.exit('usage: python3 data/download_openai.py 1536|3072')
_S = SIZES[sys.argv[1]]
REPO_ID = _S['repo']
COL = f"text-embedding-3-large-{sys.argv[1]}-embedding"
name = f"openai{sys.argv[1]}"
d = int(sys.argv[1])
NUM_QUERIES = 1_000
SEED = 42
BATCH_SIZE = _S['batch']  # rows per pyarrow batch (~100 MB float32 working set)

data_dir = os.path.join(str(RAW), name) + "/"

# 1) Fetch the parquet files only.
print(f"Downloading parquet snapshot for {REPO_ID} ...", flush=True)
snap_dir = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    revision=_S['revision'],
    allow_patterns=["*.parquet"],
)
files = sorted(glob.glob(os.path.join(snap_dir, "**", "*.parquet"), recursive=True))
if not files:
    sys.exit(f"No parquet files found under {snap_dir}")
os.makedirs(data_dir, exist_ok=True)   # only once the input is there
print(f"Snapshot: {snap_dir}", flush=True)
print(f"Found {len(files)} parquet file(s):", flush=True)
for f in files:
    print(f"  {os.path.basename(f)}", flush=True)

# 2) Total rows -> base/query counts.
N_TOTAL = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
N_BASE = N_TOTAL - NUM_QUERIES
print(f"N_TOTAL={N_TOTAL:,}  ->  base={N_BASE:,} + query={NUM_QUERIES:,}  (seed={SEED})", flush=True)

# 3) Reproducible query selection over the full row range.
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
norm_samples = []
t0 = time.time()
last_print = 0

try:
    with open(base_tmp, "wb") as fb, open(query_tmp, "wb") as fq:
        done = False
        for path in files:
            if done:
                break
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[COL]):
                bsz = batch.num_rows
                if i + bsz > N_TOTAL:
                    bsz = N_TOTAL - i
                    batch = batch.slice(0, bsz)

                emb_col = batch.column(COL)
                # use flatten (not.values) so list offsets / slices are honored
                flat = emb_col.flatten().to_numpy(zero_copy_only=False)
                if flat.dtype != np.float32:
                    flat = flat.astype(np.float32, copy=False)
                if flat.size != bsz * d:
                    raise ValueError(
                        f"dim mismatch at i={i}: {flat.size} floats for {bsz} rows, "
                        f"expected {bsz * d} (d={d})"
                    )
                arr = flat.reshape(bsz, d)

                if len(norm_samples) < 20:
                    norms = np.linalg.norm(arr[: min(bsz, 20 - len(norm_samples))], axis=1)
                    norm_samples.extend(norms.tolist())

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
                if i - last_print >= 100_000:
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
assert os.path.getsize(base_tmp) == n_base * ROW_BYTES
assert os.path.getsize(query_tmp) == n_query * ROW_BYTES

os.replace(base_tmp, base_path)
os.replace(query_tmp, query_path)

ns = np.array(norm_samples)
elapsed = time.time() - t0
print(
    f"Done! base={n_base:,}, query={n_query:,}  "
    f"({elapsed:.1f}s, {N_TOTAL/elapsed/1000:.1f}K rows/s avg)",
    flush=True,
)
print(
    f"L2-norm of first {len(ns)} vectors: min={ns.min():.6f} max={ns.max():.6f} "
    f"mean={ns.mean():.6f}  (==1.0 => unit-normalized)",
    flush=True,
)
