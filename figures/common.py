"""Shared inputs and style of the figures: result files under results/, numbers from numbers/numbers.json.
The large datasets (10M rows or more) are appended after the ten when their results are present (big=True)."""
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, ROOT)
from pcafair.config import RESULTS, FROZEN, NUMBERS  # noqa: E402
L2 = str(RESULTS); BIG = f'{RESULTS}/bigsets'                     # result files of the ten datasets / of the large datasets
OUT = f'{ROOT}/figures/out'; os.makedirs(OUT, exist_ok=True)
NUM = json.load(open(NUMBERS, encoding='utf-8'))
TEN = ['youtube', 'word2vec', 'openai1536', 'msmarco', 'openai3072', 'gist', 'fashion-mnist', 'audio', 'msong', 'mnist']
LARGE = ['sift100m', 'msmarco10m', 'msmarco113m']
C_ROT, C_PCA, C_RAW = '#1f77b4', '#ff7f0e', '#333333'
LBL_ROT, LBL_PCA = 'Random Rotation+α=0', 'PCA+α=1'


def cg(ds):
    if ds in TEN: return float(NUM[f'cg_{ds}']['v'])
    return float(json.load(open(f'{BIG}/calc/{ds}.json'))['CG_exact'])


def order(big=False):
    o = sorted(TEN, key=cg)
    assert o == TEN, o
    if big: o = o + [d for d in LARGE if have_big(d)]
    return o


def have_big(ds): return os.path.exists(f'{BIG}/calc/{ds}.json') and os.path.exists(f'{BIG}/fullscan/{ds}.csv')


def rd(p): return list(csv.DictReader(open(p)))


def src(ds, kind):
    """Result file of one dataset: kind in fullscan_<piece>, score, score_orig, decomp, decomp_orig, spectra."""
    big = ds in LARGE
    if kind.startswith('fullscan_'):
        piece = kind.split('_', 1)[1]
        return f'{BIG}/fullscan/{ds}.csv' if big else f'{L2}/fullscan/{ds}__{piece}.csv'
    base = BIG if big else L2
    if kind in ('score', 'decomp', 'decomp_orig', 'spectra', 'score_orig'):
        return f'{base}/{dict(score="score_error", decomp="decomp", decomp_orig="decomp_orig", spectra="spectra", score_orig="score_error_orig")[kind]}/{ds}.csv'
    raise KeyError(kind)



def style():
    plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'DejaVu Sans'})


def add_grid(ax, axis='y'):
    ax.grid(True, axis=axis, color='#d0d0d0', linewidth=0.6, zorder=0); ax.set_axisbelow(True)


def save(fig, name):
    for ext, dpi in (('png', 600), ('pdf', None)):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig); print('wrote', f'{OUT}/{name}.png / .pdf')
