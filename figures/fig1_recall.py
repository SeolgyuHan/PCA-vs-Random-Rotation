#!/usr/bin/env python3
""" preview: Figure 1 as small multiples like Figs. 3 and 4. Twelve panels: the ten datasets in CG order, then sift100m
and msmarco10m (msmarco113m is not drawn; it stays in Figs. 3 and 5 and Table III). x = bits per dimension B, y = Recall@10 (%).
Three arms, legend with base names only: original coordinates (slope-1 allocation on the original
coordinates), Random Rotation (uniform bits), PCA (slope 1). Unified quantizer, dither seed 0, full scans, strict recall_l2.
Data: ten datasets <L2>/fullscan/<ds>__{rot,pca,orig}.csv; large datasets <BIG>/fullscan/<ds>.csv (B = 3) and
<BIG>/fullscan_budget/<ds>__<arm>__B1-2-4-6-8.csv when present. The budgets present are drawn as a line; a single budget is a marker.
Each panel's y range is fitted to its own data. Height 3.5 in (the bar Fig. 1 is 3.4 in). Output out/fig1_recall_budget_l2.png/.pdf."""
import csv, glob, os
from common import style, save, plt, cg, order, L2, BIG, TEN, C_ROT, C_PCA, C_RAW
from matplotlib.lines import Line2D
BG = '#EAEAF2'
DS = [d for d in order(big=True) if d != 'msmarco113m']; assert len(DS) == 12, DS
BUD = (1, 2, 3, 4, 6, 8); ARMS = (('orig', C_RAW, 'D', 'original coordinates'), ('rot', C_ROT, 'o', 'Random Rotation'), ('pca', C_PCA, 's', 'PCA'))


def load(d):
    out = {}
    files = [f'{L2}/fullscan/{d}__{a}.csv' for a, *_ in ARMS] if d in TEN else [f'{BIG}/fullscan/{d}.csv'] + glob.glob(f'{BIG}/fullscan_budget/{d}__*__B*.csv')
    for p in files:
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p)):
            if r['mode'] == 'unified' and r['piece'] in ('orig', 'rot', 'pca'): out[(r['piece'], int(r['B']))] = 100 * float(r['recall_l2'])
    return out


R = {d: load(d) for d in DS}
style(); fig = plt.figure(figsize=(7.16, 3.5))
gs = fig.add_gridspec(2, 6, left=0.065, right=0.995, top=0.855, bottom=0.115, wspace=0.32, hspace=0.50)
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(6)]
for ax, d in zip(axs, DS):
    vals = []
    for z, c, m, _ in ARMS:
        bs = [b for b in BUD if (z, b) in R[d]]; ys = [R[d][(z, b)] for b in bs]; vals += ys
        if not bs: continue
        ax.plot(bs, ys, ('-' if len(bs) > 1 else '') + m, color=c, ms=3.0, lw=1.1, mec='white', mew=0.4, zorder=3 if z == 'orig' else 4)
    lo, hi = min(vals), max(vals); pad = max(2.0, 0.08 * (hi - lo)); ax.set_ylim(max(0, lo - pad), min(100.5, hi + pad))
    ax.set_xlim(0.6, 8.4); ax.set_xticks(BUD); ax.set_xticklabels([str(b) for b in BUD])
    ax.set_facecolor(BG); ax.set_axisbelow(True); ax.grid(True, color='white', lw=0.9, zorder=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(which='both', length=0, pad=1.5); ax.tick_params(axis='y', labelsize=6.5); ax.tick_params(axis='x', labelsize=6.0)
    ax.set_title(f'{d}  (CG {cg(d):.1f})', fontsize=7.0, pad=4)   # a little more room between the panel and its title
H = [Line2D([], [], color=c, marker=m, ms=3.6, mec='white', mew=0.4, lw=1.1, label=lab) for _, c, m, lab in ARMS]
fig.legend(handles=H, loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=3, fontsize=8.0, frameon=False, handlelength=2.0, columnspacing=1.6)
fig.supxlabel('Bits per dimension B', fontsize=8.5, y=0.005); fig.supylabel('Recall@10 (%)', fontsize=8.5, x=0.005)
save(fig, 'fig1_recall')
for d in DS:
    print(f'  {d:13s} ' + ' | '.join(f'B{b}: ' + ' '.join(f'{z} {R[d][(z, b)]:5.1f}' for z, *_ in ARMS if (z, b) in R[d]) for b in BUD if any((z, b) in R[d] for z, *_ in ARMS)))
