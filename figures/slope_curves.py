#!/usr/bin/env python3
"""The slope sweep as a figure: Recall@10 at B = 3 against the slope alpha of the PCA allocation,
one panel per dataset, the ten datasets in CG order and then the three datasets of ten million rows or more (as in Table III).
Ten datasets: results/fullscan/<ds>__pcaslope.csv (0-2, step 0.25) + <ds>__pcafine.csv (0.5-1.5, step 0.05), unified quantizer, strict
recall_l2, all queries (the source of the slope_* keys of numbers.json). Large datasets: results/bigsets/s7_slope.csv (slopes 0, 0.5, 1, 1.5, B = 3), plus
results/bigsets/fullscan_slope/<ds>__*.csv once a full scan of the finer slopes is done.
Line = PCA at B = 3 (alpha = 1 from <ds>__pca.csv where the sweeps lack it, as in the slope_* keys; markers on the 0.25 grid; the large datasets at their four slopes), grey dotted line = alpha = 1 (the theory),
star = alpha* (slope_<ds>_astar of numbers.json; for the large datasets the best of the four slopes). Per-panel y range: from the
multiple of 10 below the minimum to the multiple of 5 above the maximum (at most 100); titles carry the CG as in Figs. 1 and 3 (cg, one decimal). Fonts as in Fig. 3. Writes out/fig_slope_curves_l2.png/.pdf."""
import csv, glob, numpy as np
from common import order, cg, style, save, plt, L2, BIG, NUM, TEN
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
DS = order(big=True); C = plt.get_cmap('Oranges')(0.95)
BIG_N = {'sift100m': 100_000_000, 'msmarco10m': 10_000_000, 'msmarco113m': 113_520_750}   # rows of each large dataset
def curve(d):
    R = {}
    if d in TEN:
        for piece in ('pcaslope', 'pcafine'):
            for r in csv.DictReader(open(f'{L2}/fullscan/{d}__{piece}.csv')):
                if r['mode'] == 'unified' and r['B'] == '3': R[round(float(r['alpha']), 3)] = 100 * float(r['recall_l2'])
        for r in csv.DictReader(open(f'{L2}/fullscan/{d}__pca.csv')):          # alpha = 1 from the main run, as in the slope_* keys
            if r['mode'] == 'unified' and r['B'] == '3': R.setdefault(1.0, 100 * float(r['recall_l2']))
        assert abs(R[1.0] - float(NUM[f'slope_{d}_r1']['v'])) <= 0.05 + 1e-9, (d, R[1.0])  # same R(1) as numbers.json (half-up rounding there)
    else:
        for r in csv.DictReader(open(f'{BIG}/s7_slope.csv')):
            if r['dataset'] == d: R[round(float(r['alpha']), 3)] = 100 * float(r['recall_l2'])
        for f in sorted(glob.glob(f'{BIG}/fullscan_slope/{d}__*.csv')):     # the finer slopes of (0.05 steps in 0.5-1.5), merged when
            for r in csv.DictReader(open(f)):                                # a full-scan file exists; the coarse files repeat s7_slope.csv exactly
                if r['dataset'] != d or r['mode'] != 'unified' or r['B'] != '3': continue
                assert int(r['N']) == BIG_N[d], (f, r['N'])                     # only results over every database row
                al = round(float(r['alpha']), 3); v = 100 * float(r['recall_l2'])
                if al in R: assert abs(R[al] - v) < 1e-9, (d, al, R[al], v)
                R[al] = v
    a = np.array(sorted(R)); return a, np.array([R[x] for x in a])
if __name__ == '__main__':
    style(); fig, axs = plt.subplots(2, 7, figsize=(7.16, 3.55)); axs = axs.ravel(); CHK = {}
    for ax, d in zip(axs, DS):
        a, r = curve(d); best = r.max(); astar = min(a[r == best], key=lambda x: abs(x - 1))
        if d in TEN: assert abs(astar - float(NUM[f'slope_{d}_astar']['v'])) < 1e-9, (d, astar)   # same alpha* as numbers.json
        CHK[d] = dict(n=len(a), astar=float(astar), r_astar=round(float(best), 1), r1=round(float(r[a == 1.0][0]), 1), r0=round(float(r[a == 0.0][0]), 1))
        ax.axvline(1.0, color='#999999', lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
        ax.plot(a, r, color=C, lw=1.1, zorder=3)
        mk = np.isclose(a / 0.25, np.round(a / 0.25)) if d in TEN else np.ones(len(a), bool)
        ax.plot(a[mk], r[mk], 'o', color=C, ms=2.6, mew=0.4, mec='white', zorder=4)
        ax.plot([astar], [best], '*', color=C, ms=8.5, mew=0.5, mec='white', zorder=5)
        lo = 10 * np.floor((r.min() - 2) / 10); hi = min(100, 5 * np.ceil((r.max() + 0.5) / 5)); ax.set_ylim(lo, hi + 0.5)
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10], integer=True))
        ax.set_xlim(-0.05, 2.05); ax.set_xticks([0, 1, 2])
        ax.set_title(f'{d}\n(CG {cg(d):.1f})', fontsize=7.5, pad=2, linespacing=1.0); ax.tick_params(labelsize=7.5, length=2)
        for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    for i in (0, 7): axs[i].set_ylabel('Recall@10 (%)', fontsize=8.5)
    lg = axs[len(DS)]; lg.axis('off')
    H = [Line2D([], [], color=C, lw=1.1, marker='o', ms=3.6, mec='white', mew=0.4, label='PCA, B=3'),
         Line2D([], [], color=C, lw=0, marker='*', ms=9, mec='white', mew=0.5, label='α*'),
         Line2D([], [], color='#999999', lw=0.8, ls=(0, (1.5, 1.5)), label='α = 1')]
    lg.legend(handles=H, loc='center', fontsize=8.0, frameon=False, handlelength=2.0)
    fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.35, h_pad=0.9); fig.supxlabel('Slope α', fontsize=8.5, y=0.005); save(fig, 'slope_curves')
    for d in DS: print(f'  {d:13s} n={CHK[d]["n"]:2d} a*={CHK[d]["astar"]:.2f} R(a*)={CHK[d]["r_astar"]:5.1f} R(1)={CHK[d]["r1"]:5.1f} R(0)={CHK[d]["r0"]:5.1f}')
