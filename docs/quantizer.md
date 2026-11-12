# The quantizer shared by both designs

Both designs (Random Rotation+α=0 and PCA+α=1, and the original-coordinate arm of Fig. 1) are encoded by one codec, `pcafair/codec.py`
(`FairCodec`; `pcafair/bigsets.py` fits the same codec from streaming statistics for the datasets of 10M rows or more). The designs differ
only in the basis and the bit allocation. This page lists what the paper's Section IV refers to the repository for.

## Coordinates

- `x`: a prepared database row (centered on the database mean; see `data/README.md`). The code keeps its norm `R = ||x||` unquantized (float64 here;
  a 32-bit float in the paper's design) and quantizes the direction `u = x / R`.
- Basis `M` of the coding coordinates `y = M^T u`: `pca`, the eigenvectors of Cov(u) on the training sample (descending); `rot`, a dense
  Haar rotation (`pcafair.engine.haar_fullscan(d, 0)`: QR of a Gaussian matrix with seed 1000, signs of the diagonal of R folded into Q);
  `orig`, the identity (the original coordinates, the raw rows centered by the raw mean, constant coordinates removed).
- Score of a candidate: `R^2 - 2 R <q~, u_hat>` with `q~ = M^T q`, smaller is nearer (L2 search; queries are not quantized).

## Training sample

A random sample of database rows, `numpy.random.default_rng(7).choice(N, min(TS, N), replace=False)` with `TS = min(200,000, 1.2e8 / d)`
(all rows when N is smaller). Queries are not used. The sample fixes the PCA basis, the range candidates, and the gains; the
component variances `var` and the database `[min, max]` of every coordinate come from a pass over the whole database.

## Cells (range selection)

For every component `i` and bit count `b = 1..24`, the cells are `2^b` uniform cells on a range `[lo, hi]`. The range is chosen among
the quantile candidates of the training sample: with `p` in

    PG = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

the candidate is `[s_(k), s_(n-1-k)]`, `k = floor(p n)` (`p = 0` means the database `[min, max]`), and the one with the smallest
`(hi - lo)^2 4^{-b} / 12 + clipping MSE` on the sample is kept (`pcafair/nu_taskq.py`, `design`). The ranges are stored per
component and bit count (`rlo`, `rhi` in the codec file).

## Reconstruction ("unified" quantizer, the measured = analyzed quantizer)

- `b >= 2`: shared subtractive dither, one draw `z_i ~ U(-W_i/2, W_i/2)` per component (`numpy.random.default_rng(seed)`, seed 0 in
  every experiment; seeds 0–4 in the guarantee experiment), the same for all vectors, so nothing is stored per vector; the code is
  `k = clip(floor((y_i + z_i - lo) / W), 0, 2^b - 1)` and the reconstruction `lo + (k + 1/2) W - z_i`, followed by a fixed closed-form
  affine gain correction (Bussgang gain fitted on the training sample under the same draw; `closed_gains`, `dither_gains`).
- `b = 1`: deterministic split at the middle of the clipped range, reconstruction to the two cell centers, gain-corrected.
- `b = 0`: the component is set to its mean (`zero='mean'` in the frozen codecs).
- "det" mode (used only to measure the effect of the dither, Sections IV and V-D): deterministic rounding to cell centers with the same
  gain correction; above 8 bits the midpoint rule.

The reconstruction is affine in the integer code, `u_hat_i = a_i + w_i k_i`, which is what Proposition 3 (same lookups) needs.

## Bit allocation

- PCA+α=1 and the original-coordinate arm: `b_i = max(0, log2(var_i) + c)` with `sum_i b_i = B d` (Lemma 1, slope 1), solved by
  water-filling with a cap of 24 bits per component, then integerized: floor, then the missing bits go one each to the active components
  with the largest fractional parts (`alpha_alloc`). Other slopes α (Section V-C) use `alpha * log2(var_i) + c` in the same way.
- Random Rotation+α=0: `B` bits in every component (`uniform`).
- The `exact` rule (dynamic program on measured per-component error curves, `pcafair/ea_lib.py`, `dp_alloc`) is implemented for
  reference and is not used by the paper.

## What a frozen codec stores

`codec_<dataset>__<basis>.npz`: `M` (basis), `mn`, `mx` (database range), `var` (database variances of the coding coordinates), `rlo`,
`rhi` (selected ranges per bit count), `a`, `w` (affine reconstruction), `curve` (error curves), `R` (norms of all rows), `N`, `c0`,
`A_fix`, `g_fix` (fixed gains), `vro` (the design's per-bit round-off error curves), `wq` (per-component query weights, a copy of `var` in the
published codecs), `meta` (fitting metadata), and for the original coordinates `mu_raw`, `act_raw`. `experiments/codec/freeze_codecs.py` fits and
saves them (deterministic given the prepared data); `results/spectra/<dataset>.csv` exports the variances and the B = 3 allocations
that Fig. 2 draws.

## The guarantee's residual code (Section III-D)

For each vector, the size of the residual `r(x)` (end-cell remainders of dithered components, the errors of 1-bit components, and the errors of components without bits, which are reconstructed as their mean) is
stored rounded up on a geometric scale with 2, 3, or 4 bits (code 0 = no residual; levels from the 10th percentile of the nonzero sizes
over the database to the maximum). `experiments/guarantee/guarantee.py` measures the coverage of the resulting intervals.
