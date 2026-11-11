# Experiments

Every script writes under `$PCARR_OUT` (default `work/results`), in the layout of `results/`, so a rerun never overwrites the
published files; `tests/compare_dir.py <dir>` compares the two trees. The frozen codecs are read from `$PCARR_FROZEN` (default
`work/frozen`). Scripts skip outputs that already exist and use lock files, so several workers may share one dataset list.

## The ten datasets (prepared files, `data/README.md`)

| Step | Script | What it measures | Output under `$PCARR_OUT` |
|---|---|---|---|
| 0 | `codec/freeze_codecs.py [ds,...]` | fits and saves the three codecs of a dataset (bases pca, rot, orig; `docs/quantizer.md`) | `$PCARR_FROZEN/codec_<ds>__<basis>.{npz,json}` |
| 1 | `fullscan/fullscan.py [ds,...] [piece,...]` | Recall@10 of a full scan (every query, every row) at B = 1, 2, 3, 4, 6, 8; pieces `pca` (PCA+α=1), `rot` (Random Rotation+α=0), `orig` (original coordinates, slope 1), `pcaslope` (PCA, B = 3, slopes 0–2 in steps of 0.25); modes `unified` (the dithered quantizer, seed 0) and `det` (deterministic rounding) | `fullscan/<ds>__<piece>.csv`, per-query top-10 ids `fullscan/ids_<ds>__<piece>.npz` |
| 1b | `fullscan/fullscan_slope_fine.py [ds,...]` | the same at slopes 0.5–1.5 in steps of 0.05 (piece `pcafine`) | `fullscan/<ds>__pcafine.csv` |
| 1c | `fullscan/slope_hits.py` | per-query hit counts over the 25 slopes (the input of the out-of-sample check of Section V-C) | `fullscan/slope_hits/<ds>.npz` |
| 2 | `score_error/score_error.py [ds,...]` | the score error J: within-query variance of the L2 estimate's error over every query × 10,000 database rows sampled with `default_rng(123)`; also the model terms J_sep, J_hr, J_noise and c0 | `score_error/<ds>.csv` |
| 2b | `score_error/score_error_b57.py`, `score_error_orig.py` | the same at B = 5 and 7; the same for the original coordinates at B = 1–8 | `score_error_b57/`, `score_error_orig/`, `decomp_orig/` |
| 2c | `score_error/decomp.py [ds,...]` | the telescoping of the measured J into the model's assumptions (J_full, J_sep, J_hr, J_A2, J_A2tr; c0 on the training sample) | `decomp/<ds>.csv` |
| 3 | `calc/calc.py` | per-dataset constants from the codecs: CG of the codec's PCA variances, CV², Haar deficit δ, allocation gain A and correlation factor of the original coordinates, the Fig. 2 terms, the shape factor S | `calc/<ds>.json`, `calc/table2.csv` |
| 3b | `calc/direction_stats.py`, `calc/export_spectra.py` | exact Cov(u) eigenvalues, CG and ‖E[u]‖² from the prepared data; the per-component variances and B = 3 allocations of the codecs (the inputs of Fig. 2) | `calc/direction_stats.csv`, `calc/eig_u/`, `spectra/<ds>.csv` |
| 4 | `rerank/rerank_ranks.py [ds,...] [pca,rot]` | the rank of every true neighbor in the code ranking of a full scan at B = 1–8 (from which Recall@10 after re-scoring the top W is exact for every W ≥ 10) | `rerank/<ds>__<design>.npz`, `rerank/em_summary.csv` |
| 5 | `guarantee/guarantee.py [ds,...]` | coverage of the data-independent interval (Proposition 2) at nominal 95, 99, 99.9%: first 20,000 database rows × first 200 queries, dither seeds 0–4, B = 1–8, residual size in 2–4 bits | `guarantee/<ds>.csv` |

`bash tests/parity_audio.sh` runs steps 0–5 on audio (the smallest dataset) into `work/parity/` and compares every file with
`results/`.

## The large datasets (`bigsets/`, raw `.fvecs` streams, `data/README.md` §3)

`bash experiments/bigsets/run_big.sh` runs, per group (sift100m; msmarco = msmarco10m prefix + msmarco113m): `big_stats.py` (the
mean and the PCA axes of the raw stream, one pass) → `big_passA.py` (one
streaming pass: exact L2 top-100 ground truth, exact Cov(u), and the codec statistics; checkpoints every ten minutes) →
`big_freeze.py` (the three codecs) → `big_passB.py` (Recall@10 at B = 3) → `big_score.py` (score error and its decomposition) →
`big_calc.py` (constants, Table III row). Other budgets: `big_passB_budget.py <ds> --arms pca,rot,orig --B 1,2,4,6,8`; other slopes:
`big_passB_slope.py <group> --alphas ...` and `big_s7_summary.py`; prefixes of the streams as datasets of their own (Section V-A):
`big_stats.py pfx_sift1m pfx_sift10m pfx_msmarco1m` and the same passes, then `big_s3_summary.py`; encoding cost: `big_s6_cost.py`. The passes use the C kernels of `csrc/` (`make -C csrc`)
through `pcafair/bigkern.py`. `big_validate.py` runs the large-dataset pipeline on audio and checks it against the ten-dataset
results (`results/bigsets/validation/validate_audio.json`: the codecs agree to 1e-7, Recall@10 to 0.05 pp, the score error to 1e-4). Expect hours per pass on sift100m and a day or more on msmarco113m; `$PCARR_BIG_WORK` holds the checkpoints and the
cached training samples.

## Conventions shared by all scripts

- Random Rotation+α=0: dense Haar rotation `haar_fullscan(d, 0)` (`pcafair/engine.py`), B bits in every component.
  PCA+α=1: PCA of u = x/‖x‖ on the training sample, bits `max(0, log2 σ_i² + c)` summing to B·d (cap 24 bits).
- The same quantizer for both designs (`docs/quantizer.md`): shared subtractive dither (seed 0) with fixed gains for components with
  at least 2 bits, a gain-corrected split for 1-bit components, the mean for components without bits.
- Score: ‖x‖² − 2‖x‖⟨q̃, û⟩ with the stored norm; Recall@10 against the exact L2 top-10; queries are not quantized.
- Thread counts: the published runs fixed `OMP_NUM_THREADS` (8–24); floating-point reductions, and so the last digits of some
  values, can depend on it.
