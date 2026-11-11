#!/usr/bin/env python3
"""Variant of slope_curves.py: the same 13 panels, but the
y-axis is the recall gap to each dataset's best slope, R(alpha) - R(alpha*), in percentage points, on one shared range (-8 to +0.6),
with a dotted line at 0 (the best, as the previous version's horizontal 'best' line) and at alpha = 1. Points below the range
leave the axes. Legend: 'oracle alpha*' (best slope chosen after scoring), 'optimal alpha = 1'
(the theory's optimum for the score error); the 0 line is the oracle baseline.
pastel panels and white grid as Figures 2 and 3, one y label for both rows, written as the difference R(alpha) - R(alpha*).
msmarco10m is left out of this figure so that the twelve panels fill two rows of six.
the curve and its points are black.
the oracle star is seaborn deep red (#C44E52, the palette made for this #EAEAF2 background) with a darker edge of the same hue
(#7a1f23) instead of a white halo, slightly smaller (8 pt); red is not a design colour in the other figures (blue = rotation, orange = PCA).
the dotted line's label 'optimal alpha = 1' became "theory's optimum alpha = 1": alpha = 1 is the optimum the theory derives for
the score error, not the measured best slope (that is the oracle alpha*).
the large datasets take the finer slopes once their full scans are done, with
markers on the same 0.25 grid as the ten datasets.
"theory's optimum" could be read as an optimum proved earlier, so the label is "our theoretical optimum alpha = 1" (Lemma 1 of this paper).
the curve's label 'PCA, B=3' became 'PCA+α, B=3': the family of PCA designs as the slope varies, in the paper's design-name form (user choice A).
Writes out/fig_slope_gap_l2.png/.pdf."""
import numpy as np
from slope_curves import curve, DS, C, plt, style, save, cg, NUM, TEN   # reuses the data loading and checks
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
LINE, STAR, STAR_EDGE = 'black', '#C44E52', '#7a1f23' # black curve; deep-red oracle star with a same-hue dark edge
BG, REF = '#EAEAF2', '#8c8c99' # the panel background and reference-line colour of Figures 2 and 3
LO, HI = -8.0, 0.6
import sys
GRID025 = '--grid025' in sys.argv   # one grid for line, markers and oracle (0.25 for the ten datasets; the large datasets keep their 0, 0.5, 1, 1.5)
DS4 = [d for d in DS if d != 'msmarco10m']; assert len(DS4) == 12, DS4    # two rows of six
style(); fig = plt.figure(figsize=(7.16, 2.8))   # 10% shorter # legend on top
gs = fig.add_gridspec(2, 6, left=0.075, right=0.995, top=0.82, bottom=0.12, wspace=0.22, hspace=0.55)
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(6)]
for ax in axs[1:]: ax.sharey(axs[0])
for ax, d in zip(axs, DS4):
    a, r = curve(d)
    if GRID025 and d in TEN: keep = np.isclose(a / 0.25, np.round(a / 0.25)); a, r = a[keep], r[keep]
    best = r.max(); astar = min(a[r == best], key=lambda x: abs(x - 1)); g = r - best
    ax.axhline(0, color=REF, lw=0.7, zorder=2); ax.axvline(1.0, color=REF, lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)   # 0 = oracle baseline (no legend), dotted = optimal alpha = 1; REF colour over the white grid
    ax.plot(a, g, color=LINE, lw=1.1, zorder=3, clip_on=True)   # true values; the axes clip the line below the range
    mk = np.ones(len(a), bool) if GRID025 else np.isclose(a / 0.25, np.round(a / 0.25))   # markers on the 0.25 grid for every dataset (the large ones now carry 0.05 steps too)
    inr = mk & (g >= LO)
    ax.plot(a[inr], g[inr], 'o', color=LINE, ms=2.6, mew=0.4, mec='white', zorder=4)
    ax.plot([astar], [0], '*', color=STAR, ms=6.5, mew=0.45, mec=STAR_EDGE, zorder=5)
    ax.set_ylim(LO, HI); ax.set_xlim(-0.05, 2.05); ax.set_xticks([0, 1, 2]); ax.set_yticks([-8, -6, -4, -2, 0])
    ax.set_title(f'{d}\n(CG {cg(d):.1f})', fontsize=7.5, pad=2, linespacing=1.0); ax.tick_params(labelsize=7.5, length=2)
    if ax not in (axs[0], axs[6]): ax.tick_params(labelleft=False)          # y labels on the first panel of each row
    ax.set_facecolor(BG); ax.set_axisbelow(True); ax.xaxis.set_minor_locator(FixedLocator([0.5, 1.5]))   # as Figures 2 and 3
    ax.grid(True, which='major', color='white', lw=0.9, zorder=0); ax.grid(True, which='minor', axis='x', color='white', lw=0.45, zorder=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(which='both', length=0, pad=2)
fig.supylabel(r'Recall@10($\alpha$) $-$ Recall@10($\alpha^*$) (pp)', fontsize=8.5, x=0.005)   # one label for both rows, the gap written as a difference
H = [Line2D([], [], color=LINE, lw=1.1, marker='o', ms=3.6, mec='white', mew=0.4, label='PCA+α, B=3'),
     Line2D([], [], color=STAR, lw=0, marker='*', ms=7.0, mec=STAR_EDGE, mew=0.45, label='oracle α*'),
     Line2D([], [], color=REF, lw=0.9, ls=(0, (1.5, 1.5)), label='our theoretical optimum α = 1')]   # → the optimum this paper's theory gives (Lemma 1), not the measured best and not an earlier result
fig.legend(handles=H, loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=3, fontsize=8.0, frameon=False, handlelength=2.0, columnspacing=1.6)   # legend top centre
fig.supxlabel('Slope α', fontsize=8.5, y=0.005); save(fig, 'fig5_slope' + ('_grid025' if GRID025 else ''))
