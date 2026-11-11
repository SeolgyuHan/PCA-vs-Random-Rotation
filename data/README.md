# Datasets

The experiments read *prepared* datasets: each dataset centered on its database mean and projected on its database PCA (an
orthogonal change of coordinates that preserves L2 distances), with the exact L2 top-100 of every query. This folder turns the
public datasets into those files. Nothing here redistributes data; every dataset keeps its own license.

Paths (environment variables, read by `pcafair/config.py`):

| Variable | Holds | Default |
|---|---|---|
| `PCARR_RAW` | raw vectors, `<name>/<name>_base.fvecs` and `<name>/<name>_query.fvecs` | `data/raw` |
| `PCARR_PREPARED` | prepared files, `<name>-{db,query,gt-l2,eigenvalues}.npy` | `data/prepared` |
| `PCARR_BIG_RAW` | the large datasets, `SIFT100M/` and `MSMARCO100M/` | `$PCARR_RAW` |

## 1. The ten datasets: raw vectors

| Dataset | Source | How | N (database) | d | Queries | Raw size |
|---|---|---|---|---|---|---|
| youtube | YouTube-8M video-level features (`mean_rgb`), https://research.google.com/youtube8m/download.html | download the train shards listed in `youtube8m_shards.json` (file name and md5), then `python3 data/youtube8m_extract.py <shard dir>` | 999,000 | 1024 | 1,000: the first 1,000 records in reading order | 4.1 GB |
| word2vec | GQR collection, https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html | download, place `word2vec_base.fvecs`, `word2vec_query.fvecs` | 1,000,000 | 300 | its query file (1,000) | 1.2 GB |
| openai1536 | https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | `python3 data/download_openai.py 1536` | 999,000 | 1536 | 1,000 random rows (seed 42) | 6.2 GB |
| msmarco | MS MARCO v2.1 with Cohere embed-english-v3, https://huggingface.co/datasets/CohereLabs/msmarco-v2.1-embed-english-v3 | fetch the parquet snapshot (see the script header), then `MSMARCO_PARQUET_DIR=... python3 data/download_msmarco.py` | 1,000,000 (first rows of 10,000,000) | 1024 | 1,000 random rows (seed 42) | 41 GB |
| openai3072 | https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M | `python3 data/download_openai.py 3072` | 999,000 | 3072 | the same rows as openai1536 | 12.3 GB |
| gist | GQR collection | download, place `gist_base.fvecs`, `gist_query.fvecs` | 1,000,000 | 960 | its query file (1,000) | 3.8 GB |
| fashion-mnist | ANN-Benchmarks, http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5 | `python3 data/convert_hdf5.py fashion-mnist <file>` | 60,000 | 784 | the first 1,000 of the test set | 0.2 GB |
| audio | GQR collection | download, place `audio_base.fvecs`, `audio_query.fvecs` | 53,387 | 192 | its query file (200) | 0.04 GB |
| msong | GQR collection (from the Million Song Dataset) | download, place `msong_base.fvecs`, `msong_query.fvecs` | 992,272 | 420 | its query file (200) | 1.7 GB |
| mnist | ANN-Benchmarks, http://ann-benchmarks.com/mnist-784-euclidean.hdf5 | `python3 data/convert_hdf5.py mnist <file>` | 60,000 | 784 raw, 712 prepared | the test set (10,000) | 0.2 GB |

The download scripts pin the Hugging Face snapshot revisions used for the paper; the query rows are drawn with
`numpy.random.default_rng(42)`. `checksums/raw.sha256` lists the SHA-256 of the raw files we used, so a new download can be checked.

## 2. The ten datasets: prepared files

```bash
for ds in youtube word2vec openai1536 msmarco openai3072 gist fashion-mnist audio msong mnist; do
  python3 data/prepare.py $ds          # <ds>-db.npy, <ds>-query.npy, <ds>-eigenvalues.npy (+ pca-basis, mean)
  python3 data/gt_l2.py $ds            # <ds>-gt-l2.npy: exact L2 top-100 of every query (int32, nearest first)
done
```

`prepare.py` centers on the database mean, fits PCA on the database (float64 covariance, `numpy.linalg.eigh`, descending), and
projects database and queries. msmarco keeps the first 1,000,000 base rows, and mnist drops its 72 components with eigenvalue below
1e-10 of the largest (constant border pixels), leaving 712. With faiss installed it also writes `<ds>-gt-ip.npy`, the inner-product
ground truth of an earlier version of this work, which no experiment here uses.

`gt_l2.py` computes the exact squared-L2 top-100 in the prepared coordinates. The paper's runs used the official ground-truth files
of the collections that provide one (GQR, BIGANN) and this computation otherwise; the two agree up to the order of exact ties.

Re-preparing audio, fashion-mnist, and mnist from the raw files reproduces our prepared files byte for byte
(`checksums/prepared.sha256`). For the large datasets the projections can differ in the last bits with the BLAS build and thread
count; the eigenvalues and the ground truth are the quantities to check first. Prepared sizes: 4.1 GB (youtube), 1.2 GB (word2vec),
6.1 GB (openai1536), 4.1 GB (msmarco), 12.3 GB (openai3072), 3.8 GB (gist), 0.2 GB each (fashion-mnist, mnist), 0.04 GB (audio),
1.7 GB (msong); 34 GB in all.

## 3. The large datasets (sift100m, msmarco10m, msmarco113m)

The passes of `experiments/bigsets/` stream the raw `.fvecs` files; nothing is prepared in advance. `big_stats.py` computes the mean
and the PCA axes of the stream, `big_passA.py` the ground truth and the codec statistics, and `big_freeze.py` the codecs.

| Dataset | Source | How | N | d | Queries |
|---|---|---|---|---|---|
| sift100m | BIGANN (ANN_SIFT1B), the big-ann-benchmarks mirror `dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/` | fetch the first 12.8 GB of `base.1B.u8bin` (an HTTP range request) and `query.public.10K.u8bin`, then `python3 data/big_sift.py <base> <query>` | 100,000,000 (the first 100M of the billion) | 128 | the first 1,000 of the 10,000 public queries |
| msmarco113m | Cohere embed-english-v3 of MS MARCO v2.1, Hugging Face `CohereLabs/msmarco-v2.1-embed-english-v3` (revision `e78737fe92ac1b783211b705c12207ca75fcc9b7`) | download `passages_parquet/` (60 shards) and `queries_parquet/queries.parquet`, then `python3 data/big_msmarco.py <snapshot dir>` | 113,520,750 (every passage, shard order) | 1024 | the 1,677 TREC-DL 2021–2023 queries of the dataset |
| msmarco10m | the first 10,000,000 rows of the same stream | nothing more | 10,000,000 | 1024 | the same 1,677 |

Files: `$PCARR_BIG_RAW/SIFT100M/{base.100M.fvecs, query.10K.fvecs}` (48 GB) and `$PCARR_BIG_RAW/MSMARCO100M/{base.fvecs, query.fvecs}`
(434 GB). The ground truth (exact L2 top-100, re-sorted in float64) is computed by `big_passA.py` itself; the Cohere embeddings have
unit norm, so their L2, cosine, and inner-product rankings coincide.

## 4. Coding gain (Table III)

CG = AM(λ)/GM(λ) over the exact eigenvalues λ of Cov(u), the covariance of the direction u = x/||x|| of every prepared database row,
float64, all components kept (mnist keeps 712; without its six components below 1e-6 of the mean its CG would be 28.8 instead of
31.6) and no floor. `python3 experiments/calc/direction_stats.py` recomputes it (`results/calc/direction_stats.csv`,
`results/calc/eig_u/`); the paper's values are `results/calc/exact_cg.json`. For the large datasets `big_passA.py` computes the same
quantity in its streaming pass (`results/bigsets/calc/<ds>.json`, `CG_exact`). The frozen codecs fit their own PCA basis on a
training sample, so the variances they store (`results/spectra/`) differ slightly from these exact eigenvalues (at most 0.9%, except
openai3072: 3.8% with a 39,062-row sample in 3072 dimensions).
