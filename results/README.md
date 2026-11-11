# Published result files

Everything the paper prints is computed from these files by `numbers/make_numbers.py` (numbers) and `figures/*.py` (figures);
`bash reproduce.sh` regenerates and checks both. Produced by the scripts of `experiments/` (see its README for the protocols).

| Path | Content | Columns / arrays |
|---|---|---|
| `fullscan/<ds>__<piece>.csv` | Recall@10 of the full scan (ten datasets); pieces pca, rot, orig, pcaslope, pcafine; modes unified, det | `dataset, piece, mode, B, alpha, recall_l2, recall_l2_tie, n_active, max_bits, realized_bits` |
| `fullscan/slope_hits/<ds>.npz` | per-query hit counts of the slope sweep at B = 3 | `alphas` (25), `hits` (25 × n_queries, uint8) |
| `score_error/<ds>.csv`, `score_error_b57/`, `score_error_orig/` | score error J (within-query variance, mean over queries) and model terms | `dataset, pair, mode, B, J_meas, bq, J_sep, J_hr, J_noise, c0, ER2, lam_mean_u, d, n_active` |
| `decomp/<ds>.csv`, `decomp_orig/` | decomposition of J into the model's assumptions; `c0_tr` and `J_A2tr` use the training sample | `dataset, pair, B, J_meas, J_full, J_sep, J_hr, J_A2, J_A2tr, J_qh, Jm_h2, c0, c0_tr, k_A2, E2_J_meas, E2_J_hr` |
| `rerank/<ds>__<design>.npz` | rank of each true top-10 neighbor in the code ranking, per budget (`B1`…`B8`, int32, n_queries × 10) | re-ranking the top W restores a neighbor iff its rank < W |
| `rerank/em_summary.csv` | Recall@10 without re-ranking and the candidates needed for 90/95/99% | `dataset, design, B, nq, recall_norerank, depth90, depth95, depth99` |
| `guarantee/<ds>.csv` | coverage of the interval of Proposition 2 | `dataset, pair, B, seed, nominal, option, mask, s_bits, coverage, width_over_tau, frac_S_zero` |
| `calc/<ds>.json`, `calc/table2.csv` | constants from the codecs (CG of the codec's PCA variances, CV², δ, A, correlation factor, Fig. 2 terms, S, predicted ratios) | see `experiments/calc/calc.py` |
| `calc/exact_cg.json`, `calc/eig_u/<ds>.npy`, `calc/direction_stats.csv` | exact CG of Cov(u) over all rows (Table III), its eigenvalues, ‖E[u]‖² | |
| `spectra/<ds>.csv` | per-component variances and B = 3 bit allocations of the three codecs (Fig. 2) | `basis, index, var, b3` |
| `bigsets/fullscan/<ds>.csv`, `fullscan_budget/`, `fullscan_slope/` | the large datasets: Recall@10 at B = 3 (three arms), at other budgets, and at other slopes | as `fullscan/` plus `N, nq` |
| `bigsets/score_error*/`, `bigsets/decomp*/`, `bigsets/calc/` | as above for the large datasets (`decomp/<ds>.a2.json`: the fitted A2 exponent) | |
| `bigsets/s3_nseries.csv` | Recall@10 at B = 3 against N on the same stream and queries | `family, dataset, N, pca, rot, orig, gap_pp` |
| `bigsets/s6_cost.csv` | PCA fitting and encoding time at B = 3, single thread | `dataset, N, d, train_rows, pca_fit_s, rot_draw_s, encode_s, ...` |
| `bigsets/s7_slope.csv` | Recall@10 at slopes 0, 0.5, 1, 1.5 | `dataset, alpha, recall_l2, recall_l2_tie, n_active, max_bits` |

Designs: `pca` = PCA+α=1, `rot` = Random Rotation+α=0, `orig` = slope-1 allocation over the original coordinates. B is the bit
budget per dimension. Recall values are fractions (0–1); the paper prints percentages.
