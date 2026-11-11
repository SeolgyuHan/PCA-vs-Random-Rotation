#!/usr/bin/env python3
""" S3 summary: Recall@10 at B = 3 of the three arms against N on the same stream and queries (each N fitted as its own dataset):
sift 1M -> 10M -> 100M (BIGANN prefix, 1,000 queries), msmarco 1M -> 10M -> 113.5M (Cohere stream prefix, TREC-DL 1,677 queries).
Output $PCARR_OUT/bigsets/s3_nseries.csv."""
import os, sys, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pcafair.config import OUT as OUT_ROOT  # noqa: E402
OUT = str(OUT_ROOT / 'bigsets')
SER = [('sift', ['pfx_sift1m', 'pfx_sift10m', 'sift100m']), ('msmarco', ['pfx_msmarco1m', 'msmarco10m', 'msmarco113m'])]
rows = []
for fam, names in SER:
    for n in names:
        p = f'{OUT}/fullscan/{n}.csv'
        if not os.path.exists(p): continue
        R = {r['piece']: r for r in csv.DictReader(open(p))}
        rows.append(dict(family=fam, dataset=n, N=R['pca']['N'], pca=R['pca']['recall_l2'], rot=R['rot']['recall_l2'], orig=R['orig']['recall_l2'],
                         gap_pp=round(100 * (float(R['pca']['recall_l2']) - float(R['rot']['recall_l2'])), 2)))
with open(f'{OUT}/s3_nseries.csv', 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
