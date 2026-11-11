#!/usr/bin/env python3
"""(b) also draws original coordinates+alpha=1 (per-dimension terms with c0_tr(orig, B=3) from decomp_orig/, ); the (b)
legend has a fixed order and sits right of the subtitle. Figure 2: (a) the per-dimension variance sigma_i^2(U) / lambda_bar of the original coordinates, Random Rotation
and PCA, largest first; (b) at B = 3, the per-dimension terms of Eq. (1) computed before encoding, c0(U) sigma_i^4 4^{-b_i}
(b_i = 0: sigma_i^4), divided by lambda_bar^2 4^{-3} (4E[R^2] cancels), largest first, for Random Rotation+alpha=0 (uniform 3 bits)
and PCA+alpha=1. Inputs: the frozen codecs (variances of each basis on the whole database, the allocations) and c0 of each basis
from the codec's training sample (c0_tr of decomp, B = 3). Check: the ratio of the (b) sums equals the strict prediction
J_A2tr(rot)/J_A2tr(pca). pastel panel background (#EAEAF2), white grid (labelled ticks, every decade on y, quarters on x), no spines, as Figure 3.
x-axis label per block (each curve sorts its d values from largest to smallest and spreads them over 0-1).
Usage: python3 fig2_spectrum.py [--big]"""
import os, sys
import numpy as np
from matplotlib.ticker import LogLocator, FixedLocator, NullFormatter
BG = '#EAEAF2' # the panel background of Figure 3 (seaborn darkgrid tone, as in Extended RaBitQ Fig. 4)
from common import order, cg, rd, src, style, add_grid, save, plt, C_ROT, C_PCA, C_RAW, LBL_ROT, LBL_PCA
LBL_ORIG = 'original coordinates+α=1'


def data(ds):
    """Per-dimension variances (a) and the per-dimension terms of Eq. (1) at B = 3 (b) from results/spectra (the codecs' variances and
    allocations) and the c0 of each basis on its training sample (c0_tr of results/decomp at B = 3)."""
    sp = rd(src(ds, 'spectra'))
    var = {k: np.array([float(r['var']) for r in sp if r['basis'] == k]) for k in ('orig', 'rot', 'pca')}
    b3 = {k: np.array([int(r['b3']) for r in sp if r['basis'] == k], np.float64) for k in ('orig', 'rot', 'pca')}
    lam = np.maximum(var['pca'], 1e-300); lb = lam.mean()
    dec = {(r['pair'], r['B']): r for r in rd(src(ds, 'decomp'))}
    for r in rd(src(ds, 'decomp_orig')): dec[('orig', r['B'])] = r
    out = dict(a={k: np.sort(var[k])[::-1] / lb for k in ('orig', 'rot', 'pca')})
    terms = {}
    for k in ('orig', 'rot', 'pca'):
        b = b3[k]; s2 = np.maximum(var[k], 1e-300); c0 = float(dec[(k, '3')]['c0_tr'])
        t = np.where(b >= 1, c0 * s2 ** 2 * 4.0 ** (-b), s2 ** 2) / (lb ** 2 * 4.0 ** -3); terms[k] = np.sort(t)[::-1]
    pred = float(dec[('rot', '3')]['J_A2tr']) / float(dec[('pca', '3')]['J_A2tr']); got = terms['rot'].sum() / terms['pca'].sum()
    assert abs(got / pred - 1) < 1e-3, (ds, got, pred)                     # c0_tr is stored with 4 decimals
    out['b'] = terms; out['ratio'] = got; return out


def main(big):
    style(); O = order(big); nc = 5 if len(O) <= 10 else 7; nr = int(np.ceil(len(O) / nc))
    D = {d: data(d) for d in O}
    fig = plt.figure(figsize=(7.16, 0.9 * 2 * nr + 0.9)); sf = fig.subfigures(2, 1, hspace=0.04)   # 6.3 -> 4.9 in (row 1.3 -> 1.0 in)
    for part, sub in enumerate(sf):
        axs = sub.subplots(nr, nc, sharex=True, sharey=True, squeeze=False, gridspec_kw=dict(hspace=0.5, wspace=0.12))
        for i in range(nr * nc):
            r, c = divmod(i, nc); ax = axs[r, c]
            if i >= len(O): ax.axis('off'); continue
            d = O[i]
            if part == 0:
                for k, col, lab in (('orig', C_RAW, 'original coordinates'), ('rot', C_ROT, 'Random Rotation'), ('pca', C_PCA, 'PCA')):
                    y = D[d]['a'][k]; ax.plot(np.linspace(0, 1, len(y)), y, color=col, linewidth=1.0, label=lab, ls='--' if k == 'orig' else '-', zorder=5 if k == 'orig' else 3)   # G1544: original coordinates dashed on top, so the line shows where it coincides with the rotation line (word2vec, gist)
                ax.set_yscale('log'); ax.set_ylim(1e-3, 1e2); ax.set_yticks([1e-3, 1e-1, 1e1])
            else:
                for k, col, lab in (('orig', C_RAW, LBL_ORIG), ('rot', C_ROT, LBL_ROT), ('pca', C_PCA, LBL_PCA)):
                    y = D[d]['b'][k]; ax.plot(np.linspace(0, 1, len(y)), y, color=col, linewidth=1.0, label=lab, ls='--' if k == 'orig' else '-', zorder=5 if k == 'orig' else 3)
                ax.set_yscale('log'); ax.set_ylim(1e-6, 1e2); ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
            ax.set_title(f'{d}  (CG {cg(d):.1f})', fontsize=7.2, pad=2); ax.tick_params(length=2, labelsize=6.5)
            ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '.5', '1'])
            ax.set_facecolor(BG); ax.set_axisbelow(True)                                            # pastel panel, white grid, no spines (as Figure 3)
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=20)); ax.yaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_minor_locator(FixedLocator([0.25, 0.75]))
            ax.grid(True, which='major', color='white', lw=0.9, zorder=0); ax.grid(True, which='minor', color='white', lw=0.45, zorder=0)
            for s_ in ax.spines.values(): s_.set_visible(False)
            ax.tick_params(which='both', length=0, pad=2)
        h, l = axs[0, 0].get_legend_handles_labels()
        sub.legend(h, l, loc='upper center', bbox_to_anchor=(0.56 if part == 0 else 0.66, 1.0), ncol=len(l), frameon=False, fontsize=7.5, columnspacing=1.2, handlelength=1.6)
        sub.text(0.005, 0.995, '(a) Variance per dimension' if part == 0 else '(b) Score error per dimension, B=3', fontsize=8.5, va='top', ha='left')
        sub.supxlabel('dimension rank / d, largest first', fontsize=7.5, y=0.0)   # what x is (each curve sorted on its own)
        sub.subplots_adjust(top=0.80, bottom=0.13, left=0.07, right=0.99)
    save(fig, 'fig2_spectrum' + ('_big' if big else ''))
    for d in O: print(f'  {d:13s} CG {cg(d):6.2f} | sum of (b) terms rot/pca {D[d]["ratio"]:9.2f} (= strict prediction)')


if __name__ == '__main__':
    main('--big' in sys.argv)
