#!/usr/bin/env python3
"""Every number printed in the paper, recomputed from results/.

    python3 numbers/make_numbers.py            # recompute, compare with numbers/numbers.json, exit 1 on any difference
    python3 numbers/make_numbers.py --write    # recompute and rewrite numbers/numbers.json (value and rule per key)

Keys: cg_<ds> (Table II), b3_* / rot_below_orig_count / orig_even_pca_gain_count / rec_b3_hicg_dpp_* (Section V-A, Fig. 1),
jr_* and jratio_<ds>_b<B> (Fig. 3 and its checks), det_* / rot_det_* / adv_<ds>_drecall (dither, Sections IV and V-D, Table IV),
pca_inactive_b3_count, corr_share_* / openai1536_* (Section V-A), eu_norm2_max (Assumption 5), big_* / s3_* (large datasets),
slope_* (Section V-C, Fig. 5), rr_* (Section V-B, Fig. 4), guar_miss_<ds>_<level> (Table IV)."""
import os, sys, csv, json, math
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, ROOT)
from pcafair.config import RESULTS, NUMBERS, TEN, LARGE  # noqa: E402

R = str(RESULTS); BIG = f'{R}/bigsets'; BUD = (1, 2, 3, 4, 6, 8); K = 10; MINUS = '−'
OUT = {}


def rows(p): return list(csv.DictReader(open(p)))
def half_up(x, nd): return float(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))   # away from zero
def fmt(x, nd): return f'{half_up(x, nd):.{nd}f}'
def sgn(x, nd=1):
    v = half_up(x, nd); s = f'{abs(v):.{nd}f}'
    return s if v == 0 else ('+' if v > 0 else MINUS) + s
def ratio_fmt(v): return f'{v:.2f}' if v < 10 else f'{v:.0f}'
def put(k, v, rule): OUT[k] = {'v': str(v), 'src': rule}


# ------------------------------------------------------------------ inputs
REC, NACT = {}, {}                                   # recall of the ten datasets: (ds, arm, mode, B) -> Recall@10 ; n_active of pca
for d in TEN:
    for arm in ('pca', 'rot', 'orig'):
        for r in rows(f'{R}/fullscan/{d}__{arm}.csv'):
            REC[(d, arm, r['mode'], int(r['B']))] = float(r['recall_l2'])
            if arm == 'pca' and r['mode'] == 'unified': NACT[(d, int(r['B']))] = int(r['n_active'])
SLOPE = {}                                           # (ds, alpha) -> Recall@10 at B = 3 from the slope sweep (csv values)
for d in TEN:
    for piece in ('pcaslope', 'pcafine'):
        for r in rows(f'{R}/fullscan/{d}__{piece}.csv'):
            if r['mode'] == 'unified' and r['B'] == '3': SLOPE[(d, round(float(r['alpha']), 3))] = float(r['recall_l2'])
    SLOPE[(d, 1.0)] = REC[(d, 'pca', 'unified', 3)]
J, DIM = {}, {}                                      # score error: (ds, pair, mode, B) -> J_meas
for d in TEN + LARGE:
    base = R if d in TEN else BIG
    for sub in ('score_error', 'score_error_b57', 'score_error_orig'):
        for r in rows(f'{base}/{sub}/{d}.csv'): J[(d, r['pair'], r['mode'], int(r['B']))] = float(r['J_meas']); DIM[d] = int(r['d'])
EXACT = json.load(open(f'{R}/calc/exact_cg.json'))  # exact Cov(u) statistics of the ten datasets
CG = {d: EXACT[d]['CG'] for d in TEN}
for d in LARGE: CG[d] = json.load(open(f'{BIG}/calc/{d}.json'))['CG_exact']
HI = [d for d in TEN if CG[d] > 10]                  # the datasets with CG above 10
EVEN = [d for d in TEN if EXACT[d]['A_orig'] < 1.2]  # original-coordinate variances nearly even: allocation gain of those coordinates below 1.2
assert sorted(EVEN) == sorted(['youtube', 'word2vec', 'openai1536', 'msmarco', 'openai3072', 'gist']), EVEN

# ------------------------------------------------------------------ Table II, Section V-A
for d in TEN + LARGE: put(f'cg_{d}', f'{CG[d]:.2f}', 'coding gain, exact eigenvalues of Cov(u) over all database rows (results/calc/exact_cg.json; large datasets results/bigsets/calc)')
put('cg_min', f'{min(CG[d] for d in TEN):.2f}', 'smallest CG of the ten datasets'); put('cg_max', f'{max(CG[d] for d in TEN):.1f}', 'largest CG of the ten datasets')
rec3 = [d for d in TEN if REC[(d, 'pca', 'unified', 3)] > REC[(d, 'rot', 'unified', 3)]]
sc3 = [d for d in TEN if J[(d, 'pca', 'unified', 3)] < J[(d, 'rot', 'unified', 3)]]
put('b3_recall_lead_count', len(rec3), 'datasets where PCA+alpha=1 has higher Recall@10 than Random Rotation+alpha=0 at B = 3 (results/fullscan, unified)')
put('b3_score_lead_count', len(sc3), 'datasets where PCA+alpha=1 has lower score error at B = 3 (results/score_error, unified)')
put('b3_lead_count', len([d for d in rec3 if d in sc3]), 'datasets where both hold at B = 3')
put('rot_below_orig_count', len([d for d in TEN if REC[(d, 'rot', 'unified', 3)] < REC[(d, 'orig', 'unified', 3)]]), 'datasets where Random Rotation+alpha=0 has lower Recall@10 at B = 3 than slope-1 allocation over the original coordinates')
put('orig_even_pca_gain_count', len([d for d in EVEN if REC[(d, 'pca', 'unified', 3)] > REC[(d, 'orig', 'unified', 3)]]), 'of the six datasets with nearly even original-coordinate variances (A_orig < 1.2), those where PCA+alpha=1 has higher Recall@10 than the original coordinates at B = 3')
dpp = [round(100 * (REC[(d, 'pca', 'unified', 3)] - REC[(d, 'rot', 'unified', 3)])) for d in HI]
put('rec_b3_hicg_dpp_min', min(dpp), 'Recall@10 of PCA+alpha=1 minus Random Rotation+alpha=0 at B = 3 (pp, rounded), datasets with CG > 10, min'); put('rec_b3_hicg_dpp_max', max(dpp), 'same, max')
put('pca_inactive_b3_count', len([d for d in TEN if NACT[(d, 3)] < DIM[d]]), 'datasets where PCA+alpha=1 leaves components without bits at B = 3 (n_active < d)')
corr = [round(100 * EXACT[d]['corr_factor']) for d in EVEN]
put('corr_share_min', min(corr), 'share of log2 CG in the correlation factor, six even-variance datasets, min (%)'); put('corr_share_max', max(corr), 'same, max')
put('openai1536_A', f"{EXACT['openai1536']['A_orig']:.2f}", 'allocation gain of the original coordinates, openai1536 (exact)'); put('openai1536_CG', f"{CG['openai1536']:.2f}", 'CG of openai1536')
ds_path = f'{R}/calc/direction_stats.csv'
if os.path.exists(ds_path):
    eu = {r['dataset']: float(r['eu_norm2']) for r in rows(ds_path)}
    if all(d in eu for d in TEN): put('eu_norm2_max', f'{math.ceil(max(eu[d] for d in TEN) * 100) / 100:.2f}', 'largest ||E[u]||^2 over the ten datasets (results/calc/direction_stats.csv), rounded up to two decimals')

# ------------------------------------------------------------------ score-error ratios (Fig. 3)
JR = {(d, B): J[(d, 'rot', 'unified', B)] / J[(d, 'pca', 'unified', B)] for d in TEN + LARGE for B in BUD}
for (d, B), v in JR.items(): put(f'jratio_{d}_b{B}', ratio_fmt(v), 'score error of Random Rotation+alpha=0 divided by that of PCA+alpha=1 (results/score_error, unified)')
low = [JR[(d, 3)] for d in ('youtube', 'word2vec')]; hi = [JR[(d, 3)] for d in HI]
put('jr_b3_lowcg_min', ratio_fmt(min(low)), 'B = 3 ratio, youtube and word2vec, min'); put('jr_b3_lowcg_max', ratio_fmt(max(low)), 'same, max')
put('jr_b3_hicg_min', ratio_fmt(min(hi)), 'B = 3 ratio, datasets with CG > 10, min'); put('jr_b3_hicg_max', ratio_fmt(max(hi)), 'same, max')
dis = [(d, B) for d in TEN for B in BUD if (JR[(d, B)] > 1) != (REC[(d, 'pca', 'unified', B)] > REC[(d, 'rot', 'unified', B)])]
put('jr_cells', len(TEN) * len(BUD), 'ten datasets x six budgets'); put('jr_agree', len(TEN) * len(BUD) - len(dis), 'cells where score error and recall favor the same design')

# ------------------------------------------------------------------ dither (Sections IV, V-D; Table IV)
dd = {d: 100 * (REC[(d, 'pca', 'unified', 3)] - REC[(d, 'pca', 'det', 3)]) for d in TEN}
for d in TEN: put(f'adv_{d}_drecall', sgn(dd[d]), 'Recall@10 of PCA+alpha=1 at B = 3 with dither minus deterministic rounding (pp)')
put('det_drecall_min', sgn(min(dd.values())), 'same, min over the ten'); put('det_drecall_max', sgn(max(dd.values())), 'same, max over the ten')
put('det_j_pct', math.ceil(100 * max(abs(J[(d, 'pca', 'unified', 3)] / J[(d, 'pca', 'det', 3)] - 1) for d in TEN)), 'largest change of the score error of PCA+alpha=1 at B = 3 from the dither (%, rounded up)')
rd_ = {d: 100 * (REC[(d, 'rot', 'det', 3)] - REC[(d, 'rot', 'unified', 3)]) for d in TEN}
put('rot_det_drecall_min', sgn(min(rd_.values())), 'Recall@10 of Random Rotation+alpha=0 at B = 3 without dither minus with dither (pp), min'); put('rot_det_drecall_max', sgn(max(rd_.values())), 'same, max')

# ------------------------------------------------------------------ large datasets
for d, key in (('sift100m', 'big_sift_pp'), ('msmarco10m', 'big_m10_pp'), ('msmarco113m', 'big_m113_pp')):
    rr = {r['piece']: float(r['recall_l2']) for r in rows(f'{BIG}/fullscan/{d}.csv') if r['mode'] == 'unified' and r['B'] == '3'}
    put(key, fmt(100 * (rr['pca'] - rr['rot']), 1), f'Recall@10 of PCA+alpha=1 minus Random Rotation+alpha=0 at B = 3 on {d} (pp)')
s3 = {r['dataset']: float(r['gap_pp']) for r in rows(f'{BIG}/s3_nseries.csv')}
put('s3_sift_1m', fmt(s3['pfx_sift1m'], 1), 'same difference on the first million rows of sift100m, fitted on their own (results/bigsets/s3_nseries.csv)')
put('s3_msm_1m', fmt(s3['pfx_msmarco1m'], 1), 'same difference on the first million rows of msmarco')

# ------------------------------------------------------------------ slope (Section V-C, Fig. 5)
COARSE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]; SWEEP = sorted(set(COARSE) | {round(0.5 + 0.05 * k, 2) for k in range(21)})
def pick(scores): m = max(scores); return min((i for i, s in enumerate(scores) if s >= m - 1e-12), key=lambda i: abs(SWEEP[i] - 1.0))
rng = np.random.default_rng(2026); A_STAR, GAIN, OOS = {}, {}, {}
for d in TEN:
    z = np.load(f'{R}/fullscan/slope_hits/{d}.npz'); assert np.allclose(z['alphas'], SWEEP); H = z['hits'].astype(np.float64); nq = H.shape[1]
    rec = H.mean(1) * 100 / K; i1 = SWEEP.index(1.0); ib = pick(list(rec))
    for a in SWEEP: assert abs(rec[SWEEP.index(a)] - 100 * SLOPE[(d, a)]) < 0.05 + 1e-9, (d, a)   # the csv recalls are the same curve, rounded
    oos = []
    for _ in range(2000):
        p = rng.permutation(nq); Ah, Bh = p[: nq // 2], p[nq // 2:]; ah = pick(list(H[:, Ah].mean(1))); oos.append((H[ah, Bh].mean() - H[i1, Bh].mean()) * 100 / K)
    ra, r1 = half_up(rec[ib], 1), half_up(rec[i1], 1); A_STAR[d] = SWEEP[ib]; GAIN[d] = round(ra - r1, 1); OOS[d] = float(np.mean(oos))
    put(f'slope_{d}_astar', f'{SWEEP[ib]:.2f}', 'slope with the highest Recall@10 at B = 3 (PCA basis, 25 slopes; ties to the slope closest to 1)')
    put(f'slope_{d}_r_astar', f'{ra:.1f}', 'Recall@10 at that slope (%)'); put(f'slope_{d}_r1', f'{r1:.1f}', 'Recall@10 at slope 1 (%)'); put(f'slope_{d}_gain', f'{GAIN[d]:.1f}', 'difference of the two printed recalls (pp)')
rest = [d for d in TEN if d != 'youtube']
put('slope_astar_min', f'{min(A_STAR.values()):.2f}', 'smallest best slope over the ten'); put('slope_astar_max', f'{max(A_STAR.values()):.2f}', 'largest best slope over the ten')
put('slope_gain_max', f"{GAIN['youtube']:.1f}", 'largest gain of the best slope over slope 1 (youtube)'); put('slope_gain_max_rest', f'{max(GAIN[d] for d in rest):.1f}', 'largest gain over slope 1 on the nine datasets other than youtube (pp)')
put('slope_oos_min_rest', sgn(min(OOS[d] for d in rest)), 'out-of-sample gain over slope 1: slope chosen on one half of the queries, gain measured on the other half, mean over 2,000 random splits (rng 2026); nine datasets other than youtube, min (pp)')
put('slope_oos_max_rest', sgn(max(OOS[d] for d in rest)), 'same, max'); put('slope_oos_youtube', f"{half_up(OOS['youtube'], 1):.1f}", 'same on youtube')
put('slope0_gap_median', fmt(float(np.median([100 * (SLOPE[(d, 1.0)] - SLOPE[(d, 0.0)]) for d in TEN])), 1), 'median over the ten of Recall@10 at slope 1 minus slope 0 (uniform bits on the PCA basis), B = 3 (pp)')
BIG_N = {'sift100m': 100_000_000, 'msmarco10m': 10_000_000, 'msmarco113m': 113_520_750}
for d in ('sift100m', 'msmarco113m'):
    Rb = {}
    for r in rows(f'{BIG}/s7_slope.csv'):
        if r['dataset'] == d: Rb[round(float(r['alpha']), 3)] = 100 * float(r['recall_l2'])
    import glob
    for f in sorted(glob.glob(f'{BIG}/fullscan_slope/{d}__*.csv')):
        for r in rows(f):
            if r['dataset'] != d or r['mode'] != 'unified' or r['B'] != '3': continue
            assert int(r['N']) == BIG_N[d], (f, r['N']); Rb[round(float(r['alpha']), 3)] = 100 * float(r['recall_l2'])
    a = np.array(sorted(Rb)); rc = np.array([Rb[x] for x in a]); best = rc.max(); astar = float(min(a[rc == best], key=lambda x: abs(x - 1))); r1 = float(rc[a == 1.0][0])
    put(f'slope_{d}_astar', f'{astar:.2f}', 'best slope at B = 3 on the full scan of the large dataset (results/bigsets/s7_slope.csv and fullscan_slope; ties to the slope closest to 1)')
    put(f'slope_{d}_gain', f'{best - r1 + 1e-9:.1f}', 'its Recall@10 minus that of slope 1 (pp)'); put(f'slope_{d}_r1', f'{r1 + 1e-9:.1f}', 'Recall@10 at slope 1 (%)'); put(f'slope_{d}_r_astar', f'{best + 1e-9:.1f}', 'Recall@10 at the best slope (%)')

# ------------------------------------------------------------------ re-ranking (Section V-B, Fig. 4)
T = (0.90, 0.95, 0.99); B13 = (1, 2, 3)
RK = {(d, z, B): np.load(f'{R}/rerank/{d}__{z}.npz')[f'B{B}'] for d in TEN for z in ('rot', 'pca') for B in (1, 2, 3, 4)}
def need(d, z, B, t): s = np.sort(RK[(d, z, B)].ravel()); return max(10, int(s[int(np.ceil(t * s.size)) - 1]) + 1)
n = {(d, z, B, t): need(d, z, B, t) for d in TEN for z in ('rot', 'pca') for B in (1, 2, 3, 4) for t in T}
src_rr = 'smallest number of re-ranking candidates W >= 10 whose Recall@10 reaches the target (results/rerank, code ranks of the true neighbors): '
fac = {t: max(((n[(d, 'rot', B, t)] / n[(d, 'pca', B, t)], d, B) for d in TEN for B in B13)) for t in T}
alone = {t: [(d, B) for d in TEN for B in B13 if n[(d, 'pca', B, t)] == 10 < n[(d, 'rot', B, t)]] for t in T}
b4 = {t: max(n[(d, z, 4, t)] for d in TEN for z in ('rot', 'pca') if not (d == 'msong' and z == 'rot')) for t in T}
five = [d for d in TEN if CG[d] >= CG['gist']]; four = [d for d in five if d != 'msong']
f_ = lambda x: f'{x:,}'
put('rr_pairs', len(TEN) * len(B13), 'dataset x budget pairs, B = 1-3')
put('rr_msong_rot_95', f_(n[('msong', 'rot', 2, 0.95)]), src_rr + 'msong, Random Rotation+alpha=0, B = 2, 95%'); put('rr_msong_pca_95', f_(n[('msong', 'pca', 2, 0.95)]), src_rr + 'msong, PCA+alpha=1, B = 2, 95%')
for t, key in ((0.90, '90'), (0.95, '95'), (0.99, '99')):
    put(f'rr_max_{key}', round(fac[t][0]), src_rr + f'largest factor Random Rotation / PCA at {key}% over datasets and B = 1-3 ({fac[t][1]}, B = {fac[t][2]})')
    put(f'rr_max_{key}_b', fac[t][2], f'budget of that largest factor at {key}%')
put('rr_alone_90', len(alone[0.90]), 'pairs where only PCA+alpha=1 reaches 90% without re-ranking'); put('rr_alone_95', len(alone[0.95]), 'pairs where only PCA+alpha=1 reaches 95% without re-ranking')
put('rr_yt_pca_95', n[('youtube', 'pca', 1, 0.95)], src_rr + 'youtube, PCA+alpha=1, B = 1, 95%'); put('rr_yt_rot_95', n[('youtube', 'rot', 1, 0.95)], src_rr + 'youtube, Random Rotation+alpha=0, B = 1, 95%')
put('rr_b4_max_95', b4[0.95], 'B = 4: largest candidates for 95% over both designs and ten datasets, Random Rotation+alpha=0 on msong excluded'); put('rr_b4_max_99', b4[0.99], 'same for 99%')
put('rr_b4_msong_rot_95', f_(n[('msong', 'rot', 4, 0.95)]), 'B = 4: msong, Random Rotation+alpha=0, candidates for 95%'); put('rr_b4_msong_rot_99', f_(n[('msong', 'rot', 4, 0.99)]), 'B = 4: msong, Random Rotation+alpha=0, candidates for 99%')
put('rr_b1b3_msong_pca', n[('msong', 'pca', 1, 0.95)], 'msong, PCA+alpha=1 at B = 1, candidates for 95%'); put('rr_b1b3_msong_rot', f_(n[('msong', 'rot', 3, 0.95)]), 'msong, Random Rotation+alpha=0 at B = 3, candidates for 95%')
put('rr_b1b3_pca_min', min(n[(d, 'pca', 1, 0.95)] for d in four), 'gist, fashion-mnist, audio, mnist: PCA+alpha=1 at B = 1, candidates for 95%, min'); put('rr_b1b3_pca_max', max(n[(d, 'pca', 1, 0.95)] for d in four), 'same, max')
put('rr_b1b3_rot_min', min(n[(d, 'rot', 3, 0.95)] for d in four), 'same four datasets: Random Rotation+alpha=0 at B = 3, candidates for 95%, min'); put('rr_b1b3_rot_max', max(n[(d, 'rot', 3, 0.95)] for d in four), 'same, max')
cells = [(d, B, t) for d in TEN if d != 'youtube' for B in B13 for t in T]
fewer = [c for c in cells if n[(c[0], 'pca', c[1], c[2])] < n[(c[0], 'rot', c[1], c[2])]]; tie = [c for c in cells if n[(c[0], 'pca', c[1], c[2])] == n[(c[0], 'rot', c[1], c[2])]]
assert len(fewer) + len(tie) == len(cells) and all(n[(c[0], 'pca', c[1], c[2])] == 10 for c in tie), 'the paper states no cell where PCA+alpha=1 needs more, and ties only at W = 10'
put('rr_cells_rest', len(cells), 'cells outside youtube: nine datasets x B = 1-3 x targets 90/95/99%'); put('rr_fewer', len(fewer), 'cells where PCA+alpha=1 needs fewer candidates'); put('rr_tie', len(tie), 'cells where both need the same number (all at W = 10)')

# ------------------------------------------------------------------ guarantee (Table IV)
for d in TEN:
    gr = [r for r in rows(f'{R}/guarantee/{d}.csv') if r['pair'] == 'pca' and r['option'] == 'opt2' and r['mask'] == 'code' and r['s_bits'] in ('2', '3', '4')]
    for lv, nom, nd in (('95', '0.95', 2), ('99', '0.99', 3), ('999', '0.999', 4)):
        c = [r for r in gr if r['nominal'] == nom]; assert len(c) == 90, (d, lv, len(c))
        miss = 1.0 - min(float(r['coverage']) for r in c); assert miss <= 1.0 - float(nom) + 1e-12, (d, lv, miss)
        put(f'guar_miss_{d}_{lv}', f'{100.0 * miss:.{nd}f}', f'worst miss rate of the interval at nominal {nom} over B = 1/2/3/4/6/8, residual size in 2-4 bits, dither seeds 0-4 (results/guarantee; %)')

# ------------------------------------------------------------------ compare / write
if __name__ == '__main__':
    N = json.load(open(NUMBERS, encoding='utf-8')) if os.path.exists(NUMBERS) else {}
    diff = {k: (N[k]['v'], OUT[k]['v']) for k in OUT if k in N and N[k]['v'] != OUT[k]['v']}
    missing = [k for k in N if k not in OUT]
    print(f'{len(OUT)} keys recomputed; {len(diff)} differ from numbers.json; {len(missing)} keys of numbers.json not recomputed')
    for k, (a, b) in sorted(diff.items()): print(f'  {k}: paper {a} | recomputed {b}')
    if '--write' in sys.argv:
        json.dump(dict(sorted(OUT.items())), open(NUMBERS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); open(NUMBERS, 'a').write('\n'); print('numbers.json written')
    elif diff: sys.exit(1)
