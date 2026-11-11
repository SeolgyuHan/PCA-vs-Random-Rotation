#!/usr/bin/env python3
""" S7 summary: Recall@10 at B = 3 of the PCA codec at slopes 0, 0.5, 1, 1.5 on the datasets of 10M rows or more
(slope 1 from big_passB.py's pca arm, the others from big_passB_slope.py). Output $PCARR_OUT/bigsets/s7_slope.csv."""
import os, sys, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pcafair.config import OUT as OUT_ROOT  # noqa: E402
OUT = str(OUT_ROOT / 'bigsets'); os.makedirs(OUT, exist_ok=True); rows = []
for ds in ('sift100m', 'msmarco10m', 'msmarco113m'):
    R = {}
    p1 = f'{OUT}/fullscan/{ds}.csv'
    if os.path.exists(p1): R.update({1.0: r for r in csv.DictReader(open(p1)) if r['piece'] == 'pca'})
    ps = f'{OUT}/fullscan_slope/{ds}__0_0.5_1.5.csv'
    if os.path.exists(ps): R.update({float(r['alpha']): r for r in csv.DictReader(open(ps))})
    for a in sorted(R):
        r = R[a]; rows.append(dict(dataset=ds, alpha=a, recall_l2=r['recall_l2'], recall_l2_tie=r['recall_l2_tie'], n_active=r['n_active'], max_bits=r['max_bits']))
if rows:
    with open(f'{OUT}/s7_slope.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
