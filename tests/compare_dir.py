#!/usr/bin/env python3
"""Compares a new run with the published results: every csv / json / npz under NEW that also exists under results/ is compared value by
value (numbers within a relative tolerance, strings exactly). Usage: python3 tests/compare_dir.py <NEW> [--rtol 1e-6] [--pattern audio]"""
import os, sys, csv, json, glob
import numpy as np
new = sys.argv[1]; rtol = float(sys.argv[sys.argv.index('--rtol') + 1]) if '--rtol' in sys.argv else 1e-6
pat = sys.argv[sys.argv.index('--pattern') + 1] if '--pattern' in sys.argv else ''
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); PUB = os.path.join(ROOT, 'results')


def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def same(a, b):
    fa, fb = num(a), num(b)
    if fa is None or fb is None: return str(a) == str(b)
    return abs(fa - fb) <= rtol * max(abs(fa), abs(fb), 1e-300) or (fa == fb)


bad = 0; n = 0
for f in sorted(glob.glob(f'{new}/**/*', recursive=True)):
    rel = os.path.relpath(f, new); pub = os.path.join(PUB, rel)
    if os.path.isdir(f) or pat not in rel or not os.path.exists(pub) or rel.endswith(('.lock', '.log')): continue
    n += 1
    if rel.endswith('.csv'):
        A, B = list(csv.DictReader(open(f))), list(csv.DictReader(open(pub)))
        diffs = [(i, k, ra[k], rb.get(k)) for i, (ra, rb) in enumerate(zip(A, B)) for k in ra if not same(ra[k], rb.get(k))]
        if len(A) != len(B) or diffs: bad += 1; print(f'DIFF {rel}: rows {len(A)} vs {len(B)}; first differences {diffs[:3]}')
        else: print(f'same {rel} ({len(A)} rows)')
    elif rel.endswith('.json'):
        A, B = json.load(open(f)), json.load(open(pub)); ks = [k for k in A if k in B and not same(A[k], B[k]) and not isinstance(A[k], (list, dict))]
        if ks: bad += 1; print(f'DIFF {rel}: {[(k, A[k], B[k]) for k in ks[:3]]}')
        else: print(f'same {rel}')
    elif rel.endswith('.npz'):
        A, B = np.load(f), np.load(pub); ks = [k for k in A.files if k in B.files and not (A[k].shape == B[k].shape and np.allclose(A[k], B[k], rtol=rtol, atol=0))]
        if ks: bad += 1; print(f'DIFF {rel}: arrays {ks[:4]}')
        else: print(f'same {rel} ({len(A.files)} arrays)')
print(f'{n} files compared, {bad} differ'); sys.exit(1 if bad else 0)
