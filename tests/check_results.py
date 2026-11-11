#!/usr/bin/env python3
"""Checks that the published result files reproduce the paper: every number in numbers/numbers.json (numbers/make_numbers.py)
and the five figures (figures/fig*.py compared pixel by pixel with figures/published/; small rendering differences between
matplotlib or freetype versions are reported but only a structural difference fails the check).
Usage: python3 tests/check_results.py          (about one minute; needs numpy and matplotlib)"""
import os, sys, subprocess
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); os.chdir(ROOT)
FIGS = ('fig1_recall', 'fig2_spectrum', 'fig3_score', 'fig4_rerank', 'fig5_slope')

ok = True
r = subprocess.run([sys.executable, 'numbers/make_numbers.py'], capture_output=True, text=True); print(r.stdout.strip().splitlines()[0])
if r.returncode != 0: ok = False; print(r.stdout, r.stderr)
env = dict(os.environ, MPLBACKEND='Agg')
for f in FIGS:
    r = subprocess.run([sys.executable, f'figures/{f}.py'], capture_output=True, text=True, env=env)
    if r.returncode != 0: ok = False; print(f, 'FAILED'); print(r.stderr[-2000:]); continue
    import matplotlib.image as mpimg
    a = mpimg.imread(f'figures/out/{f}.png'); b = mpimg.imread(f'figures/published/{f}.png')
    if a.shape != b.shape: ok = False; print(f'{f}: image size differs {a.shape} vs {b.shape}'); continue
    diff = np.abs(a - b).max(axis=2); frac_any = float((diff > 2 / 255).mean()); frac_big = float((diff > 16 / 255).mean())
    verdict = 'identical' if frac_any == 0 else (f'{frac_any * 100:.2f}% of the pixels differ slightly (rendering), {frac_big * 100:.3f}% by more than 16/255' + ('' if frac_big < 0.01 else '  <-- structural difference'))
    print(f'{f}: {verdict}'); ok = ok and frac_big < 0.01
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'); sys.exit(0 if ok else 1)
