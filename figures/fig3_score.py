#!/usr/bin/env python3
"""third line, original coordinates+α=1, from score_error_orig/.: the score error of the two designs against the bit
budget, which replaces Table III (ratio of score errors) in V-A. Twelve panels in two rows of six, as Fig. 5 (slope): the ten datasets in CG
order, then sift100m and msmarco113m. Each panel
plots the measured score error J (E2 protocol: within-query variance over 10,000 sampled database rows, averaged over queries; unified
quantizer; one seed) of Random Rotation+alpha=0 and PCA+alpha=1 at B = 1-8 on a log axis.
Sources: <L2>/score_error/<ds>.csv (B 1, 2, 3, 4, 6, 8) + <L2>/score_error_b57/<ds>.csv, and the same two
folders under the large-dataset results.
each panel's log axis is fitted to its own data (0.3-decade margins); before every panel spanned the same number of decades so that the gap and
slopes compare across panels; the caption says so. Major ticks every second decade, a minor white grid line every decade.
Checks (assert): eight budgets per dataset and design; J falls with every bit; the B = 3 ratio equals Table III's jratio_<ds>_b3; youtube is the
only dataset where Random Rotation+alpha=0 is lower, at B >= 3; the four datasets with CG > 10 have a B = 3 ratio above 100.
Writes out/fig_score_budget_l2.png/.pdf."""
import csv, os
import numpy as np
from common import style, save, plt, cg, order, L2, BIG, NUM, TEN, C_ROT, C_PCA, C_RAW
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

BG = '#EAEAF2' # pastel panels and white grid, as Figs. 2, 4, 5
DS = [d for d in order(big=True) if d != 'msmarco10m']; assert len(DS) == 12, DS
ALL = DS + ['msmarco10m']                                             # checked, not drawn
BUDGETS = tuple(range(1, 9))


def load_all():
    J = {}
    for d in ALL:
        base = BIG if d not in TEN else L2
        for sub in ('score_error', 'score_error_b57', 'score_error_orig'):
            for r in csv.DictReader(open(f'{base}/{sub}/{d}.csv')):
                if r['mode'] == 'unified': J[(d, r['pair'], int(r['B']))] = float(r['J_meas'])
    return J


def checks(J):
    for d in ALL:
        for z in ('rot', 'pca'):
            assert all((d, z, b) in J for b in BUDGETS), (d, z)
            assert all(J[(d, z, b)] > J[(d, z, b + 1)] for b in BUDGETS[:-1]), (d, z, 'not monotone')
        assert all((d, 'orig', b) in J for b in BUDGETS), (d, 'orig')
        ratio3 = J[(d, 'rot', 3)] / J[(d, 'pca', 3)]; key = float(NUM[f'jratio_{d}_b3']['v'])
        assert abs(ratio3 - key) <= (0.005 if key < 10 else 0.5) + 1e-9, (d, ratio3, key)      # same as Table III's cell
        lower = [b for b in BUDGETS if J[(d, 'pca', b)] < J[(d, 'rot', b)]]
        assert lower == ([1, 2] if d == 'youtube' else list(BUDGETS)), (d, lower)               # the caption's first sentence
        if d in TEN and cg(d) > 10: assert ratio3 > 100, (d, ratio3)                            # "less than a hundredth"
    return True


if __name__ == '__main__':
    J = load_all(); checks(J)
    spans = {d: (np.log10(min(J[(d, z, b)] for z in ('orig', 'rot', 'pca') for b in BUDGETS)), np.log10(max(J[(d, z, b)] for z in ('orig', 'rot', 'pca') for b in BUDGETS))) for d in DS}
    M = 0.3    # each panel is fitted to its own data, 0.3 decades of margin above and below (the common-span rule of 07D was dropped by the author)
    style(); fig = plt.figure(figsize=(7.16, 3.4))   # 10% shorter # -> taller, narrower gaps (panel ~0.78 x 0.91 -> 0.84 x 1.10 -> ~0.88 x 1.33 in)
    gs = fig.add_gridspec(2, 6, left=0.07, right=0.995, top=0.86, bottom=0.10, wspace=0.30, hspace=0.42)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(6)]
    for ax, d in zip(axs, DS):
        for z, c, m in (('orig', C_RAW, 'D'), ('rot', C_ROT, 'o'), ('pca', C_PCA, 's')):
            ax.plot(BUDGETS, [J[(d, z, b)] for b in BUDGETS], '-' + m, color=c, ms=2.8, lw=1.1, mec='white', mew=0.4, zorder=3 if z == 'orig' else 4)
        lo, hi = spans[d]; ax.set_yscale('log'); ax.set_ylim(10 ** (lo - M), 10 ** (hi + M))
        ax.yaxis.set_major_locator(LogLocator(base=100.0)); ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=20)); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(0.6, 8.4); ax.set_xticks(BUDGETS); ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_facecolor(BG); ax.set_axisbelow(True)
        ax.grid(True, which='major', color='white', lw=0.9, zorder=0); ax.grid(True, which='minor', axis='y', color='white', lw=0.45, zorder=0)
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.tick_params(which='both', length=0, pad=1.5); ax.tick_params(axis='y', labelsize=6.5); ax.tick_params(axis='x', labelsize=6.0)
        ax.set_title(f'{d}\n(CG {cg(d):.1f})', fontsize=7.5, pad=2, linespacing=1.0)
    fig.legend(handles=[Line2D([], [], color=C_RAW, marker='D', ms=3.6, mec='white', mew=0.4, lw=1.1, label='original coordinates+α=1'),
                        Line2D([], [], color=C_ROT, marker='o', ms=3.6, mec='white', mew=0.4, lw=1.1, label='Random Rotation+α=0'),
                        Line2D([], [], color=C_PCA, marker='s', ms=3.6, mec='white', mew=0.4, lw=1.1, label='PCA+α=1')],
               loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=3, fontsize=8.0, frameon=False, handlelength=2.0, columnspacing=1.6)
    fig.supxlabel('Bits per dimension B', fontsize=8.5, y=0.005); fig.supylabel('Score error J (log scale)', fontsize=8.5, x=0.005)
    save(fig, 'fig3_score')
    print('span per panel (decades):', {d: round(hi - lo + 2 * M, 1) for d, (lo, hi) in spans.items()}, '| checks passed for', len(ALL), 'datasets')
