#!/usr/bin/env python3
"""Figure 4: Recall@10 against the number of candidates for re-ranking, ten datasets in CG order (2 x 5), both designs at
B = 1-3. Source: the code ranks (results/rerank/<ds>__<design>.npz; frozen codecs, unified quantizer, dither seed 0, all
queries): for every query and each of its 10 true neighbours, the number of database vectors with a smaller code score. Recall@10
after re-scoring the top W of the code ranking with the original vectors = mean #(rank < W)/10, exact for every W >= 10 (W = 10 is
the scan alone); the earlier re-ranking grid agrees within 0.11 pp at all 520 grid points. Colour family = design (Blues: Random
Rotation+alpha=0, Oranges: PCA+alpha=1), shade + line style + marker = B (light/dotted/circle 1, medium/dashed/square 2,
dark/solid/triangle 3). y = 80-100% (shared); x = log, 10 to 1.2 x the depth at which all six curves reach 99% (per panel, end value
labelled); markers at 10 log-even integer depths per panel; grey dotted lines at 90/95/99%.
pastel panel background (#EAEAF2) with a white grid (major ticks; log-minor ticks on x), no spines, as in Extended RaBitQ Fig. 4.
Writes out/fig4_rerank_l2.png/.pdf and out/fig4_rerank_curves_checks.json (x-range per dataset)."""
import numpy as np
from common import order, cg, style, save, plt, L2, OUT, LBL_ROT, LBL_PCA
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, LogLocator
DS = order(); BS = (1, 2, 3); GRID = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
from common import C_ROT, C_PCA
CMAP = {'rot': plt.get_cmap('Blues'), 'pca': plt.get_cmap('Oranges')}; SHADE = {1: 0.55, 2: 0.75, 3: 0.95}
COLB = lambda dz, B: CMAP[dz](SHADE[B]); MK = {1: 'o', 2: 's', 3: '^'}; LS = {1: (0, (1, 1.2)), 2: (0, (4, 1.6)), 3: '-'}; NM = 10
BG = '#EAEAF2' # panel background (seaborn darkgrid tone, the one in Extended RaBitQ Fig. 4)
style(); fig, axs = plt.subplots(2, 5, figsize=(7.16, 3.1)); axs = axs.ravel(); NEED = {}; XM = {}   # taller for the larger fonts
for ax, d in zip(axs, DS):
    Z = {dz: np.load(f'{L2}/rerank/{d}__{dz}.npz') for dz in ('rot', 'pca')}
    need99 = max(int(np.sort(Z[dz][f'B{B}'].ravel())[int(np.ceil(0.99 * Z[dz][f'B{B}'].size)) - 1]) + 1 for dz in Z for B in BS)   # smallest W with recall >= 99%
    wmax = int(np.ceil(1.2 * max(need99, 10))); XM[d] = (need99, wmax)                  # end right after the 99% point, no rounding up
    W = np.unique(np.concatenate([np.arange(10, 201), np.round(np.geomspace(200, wmax, 400)).astype(int)])); ymin = 100
    for dz in ('rot', 'pca'):
        for B in BS:
            r = np.sort(Z[dz][f'B{B}'].ravel()); rec = 100 * np.searchsorted(r, W, side='left') / r.size; c = COLB(dz, B); ymin = min(ymin, rec[0])
            ax.plot(W, rec, color=c, lw=1.0, ls=LS[B], zorder=3)
            g = np.unique(np.round(np.geomspace(10, wmax, NM)).astype(int)); ax.plot(g, 100 * np.searchsorted(r, g, side='left') / r.size, MK[B], color=c, ms=2.6, mew=0.4, mec='white', zorder=4)
            for t in (90, 95, 99):
                ok = np.nonzero(rec >= t - 1e-9)[0]; NEED[(d, dz, B, t)] = int(W[ok[0]]) if len(ok) else None
    for t in (90, 95, 99): ax.axhline(t, color='#8c8c99', lw=0.6, ls=(0, (3, 2)), zorder=2) # darker grey so the targets stay visible over the white grid
    ax.set_xscale('log'); ax.set_xlim(10, wmax); tk = [v for v in ((10, 15, 20, 30) if wmax <= 40 else (10, 20, 50, 100, 200) if wmax <= 200 else (10, 100, 1000, 10000)) if v <= wmax]
    span = np.log10(wmax / 10); tk = [v for v in tk if np.log10(wmax / v) >= 0.22 * span] + [wmax]; ax.set_xticks(tk)   # wider gap before the end label at the larger font
    ax.set_xticklabels([f'{v // 1000}k' if v >= 1000 and v % 1000 == 0 else f'{v:,}' for v in tk]); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_ylim(80, 100.5)
    ax.set_title(f'{d} (CG {cg(d):.1f})', fontsize=7.5, pad=2); ax.tick_params(labelsize=7.5, length=2)   # 6.5/5.5 -> 7.5
    ax.set_facecolor(BG); ax.set_axisbelow(True)                                                     # pastel panel and white grid (as Extended RaBitQ Fig. 4)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10))); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', color='white', lw=0.9, zorder=0); ax.grid(True, which='minor', axis='x', color='white', lw=0.45, zorder=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(which='both', length=0, pad=2)
for i in (0, 5): axs[i].set_ylabel('Recall@10 (%)', fontsize=8.5)   # 7 -> 8.5

H = [Line2D([], [], color=COLB(dz, B), lw=1.0, ls=LS[B], marker=MK[B], ms=4.6, mec='white', mew=0.4, label=f'{LBL_ROT if dz == "rot" else LBL_PCA}, B={B}') for B in BS for dz in ('rot', 'pca')]   # column-major fill: rows = designs
fig.legend(handles=H, loc='upper center', fontsize=8.0, frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=2.4)   # 6 -> 8
fig.tight_layout(rect=[0, 0.055, 1, 0.89], w_pad=0.6, h_pad=0.7); fig.supxlabel('Candidates for re-ranking', fontsize=8.5, y=0.005); save(fig, 'fig4_rerank'); print('  x-range (largest depth for 99%, x max):', XM)   # 7 -> 8.5
import json; json.dump({'xrange': XM, 'need': {f'{k[0]}|{k[1]}|B{k[2]}|{k[3]}': v for k, v in NEED.items()}}, open(f'{OUT}/fig4_rerank_curves_checks.json', 'w'), indent=1)
for t in (90, 95, 99):
    print(f'  {t}%: ' + ' | '.join(f"{d} " + ' '.join(f"{dz}{B}:{NEED[(d,dz,B,t)]}" for dz in ('rot', 'pca') for B in BS) for d in DS))
