#!/usr/bin/env python3
"""Prepare one dataset: center on the database mean, fit PCA on the database, project database and queries,
and (optionally, with faiss) compute the exact top-10 by centered inner product. The L2 ground truth
that the experiments use is written by data/gt_l2.py.

Usage:
  python3 data/prepare.py <dataset_name>

Reads $PCARR_RAW/<name>/<name>_{base,query}.fvecs and writes to $PCARR_PREPARED:
  {name}-db.npy database in PCA coordinates (float32, N x d)
  {name}-query.npy queries in PCA coordinates (float32)
  {name}-gt-ip.npy exact top-10 by centered inner product (int32, faiss IndexFlatIP)
  {name}-eigenvalues.npy eigenvalues, descending (float32)
  {name}-pca-basis.npy PCA basis W (float32, d_raw x d), db = (x - mean) @ W [written by this script; the published runs did not keep it]
  {name}-mean.npy database mean (float64) [written by this script; the published runs did not keep it]

Pipeline:
  1. Read fvecs (base + query); msmarco keeps the first 1,000,000 base rows; queries capped at 1,000 (mnist: 10,000).
  2. Mean-center on the database, covariance (float64), eigh, descending order; mnist drops the 72 components with
     eigenvalue < 1e-10 * max (constant border pixels).
  3. Project database and queries; exact top-10 with faiss IndexFlatIP when faiss is installed.
"""

import os
import sys
import struct
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcafair.config import PREPARED, RAW  # noqa: E402

SRC_BASES = [RAW]                            # <RAW>/<name>/<name>_base.fvecs (or <RAW>/<name>/<name>/...)
OUT_DIR = PREPARED
TOPK = 10
SEED = 42
DEFAULT_NQ = 1000
# Per-dataset NQ override; falls back to NQ_MAX otherwise.
# Set to match RaBitQ paper Table 3 (Query Size column):
# Msong 200, SIFT 10000, DEEP 1000, Word2Vec 1000, GIST 1000, Image 200.
# Anything not listed uses NQ_MAX (1000).
NQ_PER_DATASET = {
    "sift": 10_000,       # RaBitQ paper uses full 10K query set
    "mnist": 10_000,      # use the full 10K query set
}
NQ_MAX = 1000  # default cap for datasets not in NQ_PER_DATASET

# Per-dataset N cap for database vectors. Used to subsample very large
# datasets to fit in memory (e.g. msmarco has 10M × 1024 = 40 GB which
# does not fit even before the PCA projection). The subsample uses the
# first N_PER_DATASET[name] rows deterministically.
N_PER_DATASET = {
    "msmarco": 1_000_000,  # subsample to 1M for consistency with other datasets
}

# Per-dataset row-wise unit-norm normalization (applied BEFORE mean-centering).
# DISABLED: the registry is empty so no dataset is normalized. The branch code
# is kept for reference; all datasets are used as-is (must NOT be normalized).
NORMALIZE_PER_DATASET = {}

# Per-dataset numerical-rank truncation (relative eigenvalue threshold τ). When
# set for a dataset, PCA keeps only components with λ_i >= τ·λ_max and drops the
# rest BEFORE projection — removing degenerate (zero/near-zero variance) dims so
# CG and the bit budget (B·d) are meaningful. The IP GT is INVARIANT to this
# (dropped dims contribute exactly 0 to the centered-PCA inner product). Default
# (not listed) = no truncation → all other datasets stay byte-identical.
# mnist: 28×28 has constant border pixels (dead) → 72 degenerate dims; rank 712,
# CG 367(floor artifact)→30.7.
RANK_TRUNCATE_PER_DATASET = {
    "mnist": 1e-10,
}


def find_dataset_dir(name):
    """Return the directory containing {name}_base.fvecs, etc.

    Supports two layouts under any of SRC_BASES:
      - {base}/{name}/{name}_*.fvecs (flat, e.g. sift, openai1536)
      - {base}/{name}/{name}/{name}_*.fvecs (nested, e.g. deep1M)
    """
    candidates = []
    for base in SRC_BASES:
        candidates.append(base / name / name)   # nested
        candidates.append(base / name)          # flat
    for d in candidates:
        if (d / f"{name}_base.fvecs").exists():
            return d
    raise FileNotFoundError(
        f"No {name}_base.fvecs under any of: {[str(b/name) for b in SRC_BASES]}"
    )


def read_fvecs(path, max_n=None):
    """Read a.fvecs file → (N, d) float32.

    If `max_n` is given and smaller than the file's N, only the first `max_n`
    rows are loaded via memmap (avoids loading the whole file into RAM).
    """
    sz = os.path.getsize(path)
    with open(path, "rb") as f:
        d = struct.unpack("<i", f.read(4))[0]
    record_floats = 1 + d
    n_total = sz // (4 * record_floats)
    if max_n is not None and max_n < n_total:
        mm = np.memmap(path, dtype=np.float32, mode="r",
                       shape=(n_total, record_floats))
        return np.ascontiguousarray(mm[:max_n, 1:])
    data = np.fromfile(path, dtype=np.float32).reshape(n_total, record_floats)
    return np.ascontiguousarray(data[:, 1:])


def compute_cg(eigvals, clip=1e-10):
    e = np.maximum(eigvals.astype(np.float64), clip)
    return float(e.mean() / np.exp(np.log(e).mean()))


def brute_force_topk_ip(q_pca, db_pca, k):
    """Centered-PCA IP top-k via faiss IndexFlatIP."""
    import faiss
    db_c = np.ascontiguousarray(db_pca, dtype=np.float32)
    q_c = np.ascontiguousarray(q_pca, dtype=np.float32)
    index = faiss.IndexFlatIP(db_c.shape[1])
    index.add(db_c)
    _, I = index.search(q_c, k)
    return I.astype(np.int32, copy=False)


def prep_dataset(name, source_dir, output_dir, n_query=DEFAULT_NQ, topk=TOPK, seed=SEED):
    """Convert an.fvecs dataset into the prepared npy files."""
    print("=" * 70)
    print(f"  prepare: {name}")
    print("=" * 70)

    src_dir = Path(source_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = src_dir / f"{name}_base.fvecs"
    query_path = src_dir / f"{name}_query.fvecs"

    if not base_path.exists():
        raise FileNotFoundError(f"base file not found: {base_path}")

    t0 = time.time()
    print(f"\n[1/5] reading base: {base_path}")
    n_cap = N_PER_DATASET.get(name)
    if n_cap is not None:
        print(f"      subsample cap: first {n_cap:,} rows (via memmap)")
    base = read_fvecs(base_path, max_n=n_cap)
    print(f"      base shape: {base.shape}  dtype: {base.dtype}  ({time.time()-t0:.1f}s)")

    t1 = time.time()
    if query_path.exists():
        print(f"\n[2/5] reading query: {query_path}")
        query = read_fvecs(query_path)
        # Per-dataset Nq cap (RaBitQ paper-aligned via NQ_PER_DATASET).
        nq_cap = NQ_PER_DATASET.get(name, NQ_MAX)
        if query.shape[0] > nq_cap:
            print(f"      subsampling {nq_cap} of {query.shape[0]} queries (first-N)")
            query = query[:nq_cap]
        db = base
        print(f"      query shape: {query.shape}  ({time.time()-t1:.1f}s)")
    else:
        print(f"\n[2/5] no query file; subsampling {n_query} from base (seed={seed})")
        rng = np.random.RandomState(seed)
        idx = rng.permutation(base.shape[0])[:n_query]
        mask = np.ones(base.shape[0], dtype=bool)
        mask[idx] = False
        query = base[idx]
        db = base[mask]
        print(f"      db: {db.shape}  query: {query.shape}  ({time.time()-t1:.1f}s)")

    # Optional per-dataset row-wise unit-norm BEFORE mean-centering.
    # Guarded by NORMALIZE_PER_DATASET.get(name, False) so the other 7
    # datasets take exactly the same code path as before (bit-identical).
    if NORMALIZE_PER_DATASET.get(name, False):
        orig_db = float(np.linalg.norm(db, axis=1).mean())
        orig_q  = float(np.linalg.norm(query, axis=1).mean())
        db_norms = np.linalg.norm(db, axis=1, keepdims=True)
        q_norms  = np.linalg.norm(query, axis=1, keepdims=True)
        n_zero_db = int((db_norms < 1e-12).sum())
        n_zero_q  = int((q_norms  < 1e-12).sum())
        # 0-norm rows divide by 1.0 (vector stays at 0, no NaN).
        db    = db    / np.where(db_norms < 1e-12, 1.0, db_norms)
        query = query / np.where(q_norms  < 1e-12, 1.0, q_norms)
        print(
            f"      [normalize] {name}: base mean‖v‖ {orig_db:.4f}->1.0, "
            f"query mean‖v‖ {orig_q:.4f}->1.0  "
            f"(0-norm: db={n_zero_db}, q={n_zero_q})"
        )

    t2 = time.time()
    print(f"\n[3/5] PCA fit on db (centering + cov + eigh, descending)")
    mu = db.mean(axis=0)
    db_c = db - mu
    q_c = query - mu
    # np.cov upcasts to float64 and materializes ~2x the array internally; for
    # large (N, d) this OOMs (openai3072: 999K x 3072 -> ~74 GB peak on 62 GB RAM).
    # When the float64 copy would exceed 16 GB, accumulate the covariance in
    # row-chunks instead. Result is the same (1/(N-1)) * Σ (x_i-mu)(x_i-mu)^T to
    # float precision; peak memory drops to ~one chunk. Datasets below the
    # threshold (all current 7 incl. openai1536) keep the exact np.cov path.
    n_rows, n_dim = db_c.shape
    if n_rows * n_dim * 8 > 12 * 1024**3:
        print(f"      [cov] chunked accumulation (N={n_rows:,}, d={n_dim}) to bound memory")
        cov = np.zeros((n_dim, n_dim), dtype=np.float64)
        CHUNK = 50_000
        for i in range(0, n_rows, CHUNK):
            b = db_c[i:i + CHUNK].astype(np.float64)
            cov += b.T @ b
            del b
        cov /= (n_rows - 1)
    else:
        cov = np.cov(db_c.astype(np.float64), rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    print(f"      eigvals[0]={eigvals[0]:.4e}  eigvals[-1]={eigvals[-1]:.4e}  "
          f"({time.time()-t2:.1f}s)")

    # Per-dataset numerical-rank truncation (drops degenerate dims; IP GT invariant).
    _tau = RANK_TRUNCATE_PER_DATASET.get(name)
    if _tau is not None:
        _r = int((eigvals >= _tau * eigvals[0]).sum())
        if _r < len(eigvals):
            print(f"      [rank-truncate] {name}: τ={_tau:g} → rank {_r}/{len(eigvals)} "
                  f"(dropped {len(eigvals)-_r} degenerate dims; kept λ ≥ {_tau*eigvals[0]:.3e})")
            eigvals = eigvals[:_r]
            eigvecs = eigvecs[:, :_r]

    t3 = time.time()
    print(f"\n[4/5] projecting db and query")
    # For large (N, d), `db_c @ eigvecs` materializes a full (N, d) float64
    # intermediate (openai3072: ~24.6 GB) before the float32 cast -> OOM.
    # Project in row-chunks instead (per-chunk float64 matmul then cast, so the
    # numerical result matches the non-chunked path). Threshold matches the cov
    # guard above, so all current 7 datasets keep the exact original line.
    if n_rows * n_dim * 8 > 12 * 1024**3:
        print(f"      [proj] chunked projection (N={n_rows:,}, d={n_dim}) to bound memory")
        db_pca = np.empty((n_rows, eigvecs.shape[1]), dtype=np.float32)
        CHUNK = 50_000
        for i in range(0, n_rows, CHUNK):
            db_pca[i:i + CHUNK] = (db_c[i:i + CHUNK] @ eigvecs).astype(np.float32)
    else:
        db_pca = (db_c @ eigvecs).astype(np.float32, copy=False)
    q_pca = (q_c @ eigvecs).astype(np.float32, copy=False)
    del db, query, db_c, q_c
    print(f"      db_pca: {db_pca.shape}  q_pca: {q_pca.shape}  ({time.time()-t3:.1f}s)")

    t4 = time.time()
    try:
        import faiss  # noqa: F401
        print(f"\n[5/5] computing top-{topk} GT by centered inner product (faiss IndexFlatIP; not used by the paper's L2 experiments)")
        gt = brute_force_topk_ip(q_pca, db_pca, topk)
        print(f"      GT shape: {gt.shape}  ({time.time()-t4:.1f}s)")
    except ImportError:
        gt = None; print("\n[5/5] faiss not installed: skipping the inner-product ground truth (the paper uses data/gt_l2.py)")
    cg = compute_cg(eigvals)
    print(f"      CG of Cov(x) = {cg:.6f}   log2(CG) = {np.log2(cg):.6f}   (Table III uses Cov(u): experiments/calc/direction_stats.py)")

    np.save(out_dir / f"{name}-db.npy", db_pca)
    np.save(out_dir / f"{name}-query.npy", q_pca)
    if gt is not None: np.save(out_dir / f"{name}-gt-ip.npy", gt)
    np.save(out_dir / f"{name}-eigenvalues.npy", eigvals.astype(np.float32))
    np.save(out_dir / f"{name}-pca-basis.npy", eigvecs.astype(np.float32))
    np.save(out_dir / f"{name}-mean.npy", np.asarray(mu, dtype=np.float64))

    print(f"\n[DONE] saved 6 files under {out_dir}  (total {time.time()-t0:.1f}s)")
    return {
        "name": name,
        "d": db_pca.shape[1],
        "N": db_pca.shape[0],
        "N_q": q_pca.shape[0],
        "CG": cg,
        "log2_CG": float(np.log2(cg)),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 data/prepare.py <dataset_name>")
        sys.exit(1)
    name = sys.argv[1]
    src = find_dataset_dir(name)
    info = prep_dataset(name, src, OUT_DIR)
    print()
    print("Dataset info:", info)


if __name__ == "__main__":
    main()
