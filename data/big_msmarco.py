#!/usr/bin/env python3
"""msmarco10m / msmarco113m: the MS MARCO v2.1 passage embeddings of Cohere embed-english-v3 (Hugging Face dataset
CohereLabs/msmarco-v2.1-embed-english-v3, revision e78737fe92ac1b783211b705c12207ca75fcc9b7), as .fvecs for the large-dataset passes.
base.fvecs holds all 113,520,750 passage embeddings in the order of the passages_parquet shards 00-59 (msmarco113m); msmarco10m is
its first 10,000,000 rows. query.fvecs holds the 1,677 TREC-DL 2021-2023 query embeddings of queries_parquet/queries.parquet.
Writes $PCARR_BIG_RAW/MSMARCO100M/{base.fvecs, query.fvecs} (433.5 GB and 6.9 MB).
Usage: python3 data/big_msmarco.py <snapshot dir containing passages_parquet/ and queries_parquet/>   (pyarrow needed)"""
import os, sys, glob, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from pcafair.config import BIG_RAW  # noqa: E402

D = 1024; BATCH = 65536
HDR = np.array([D], np.int32).view(np.float32)[0]      # the .fvecs row header, bit for bit


def write_rows(fout, arr):
    out = np.empty((arr.shape[0], D + 1), np.float32); out[:, 0] = HDR; out[:, 1:] = arr; out.tofile(fout)


if __name__ == '__main__':
    snap = sys.argv[1]; out = BIG_RAW / 'MSMARCO100M'; out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f'{snap}/passages_parquet/*.parquet')); assert len(files) == 60, len(files)
    n = 0; t0 = time.time()
    with open(out / 'base.fvecs.part', 'wb', buffering=8 << 20) as fout:
        for i, fp in enumerate(files):
            for batch in pq.ParquetFile(fp).iter_batches(batch_size=BATCH, columns=['emb'], use_threads=True):
                flat = batch.column(0).flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                write_rows(fout, flat.reshape(len(batch), D)); n += len(batch)
            print(f'shard {i + 1}/60 rows {n:,} [{(time.time() - t0) / 60:.1f} min]', flush=True)
    assert n == 113_520_750, n; os.replace(out / 'base.fvecs.part', out / 'base.fvecs')
    q = pq.read_table(f'{snap}/queries_parquet/queries.parquet', columns=['emb']).column('emb').to_pylist()
    with open(out / 'query.fvecs', 'wb') as fout: write_rows(fout, np.asarray(q, np.float32))
    print('base.fvecs', n, 'rows; query.fvecs', len(q), 'rows')
