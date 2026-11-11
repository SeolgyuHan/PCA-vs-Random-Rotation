"""Argument parsing and result-file helpers of the experiment scripts.

Every experiment writes under $PCARR_OUT (default: $PCARR_WORK/results), in the same layout as results/, so a new run
never overwrites the published files; Long runs write one part file
per dataset (resumable: a dataset whose part exists is skipped) and merge the parts into the published file layout
once every dataset is done.
"""
import argparse
import csv
from pathlib import Path

from .config import OUT


def parse_args(description, default_sets, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--datasets', default=','.join(default_sets), help='comma-separated dataset names (default: all)')
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.datasets = [s for s in a.datasets.split(',') if s]
    a.out = OUT                     # root of the new result files ($PCARR_OUT)
    return a


def write_csv(path, fields, rows, crlf=True):
    """csv.DictWriter output; crlf=True is the csv module's default line ending (the published files keep it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\r\n' if crlf else '\n')
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def part_path(out, rel, ds):
    """Per-dataset part of the result file `rel` (e.g. 'fullscan/recall_cells.csv')."""
    p = Path(out) / (rel + '.parts') / f'{ds}.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def merge_parts(out, rel, order, fields, crlf=True):
    """Concatenate the per-dataset parts in `order` into out/rel once all exist; returns the missing datasets."""
    missing = [ds for ds in order if not part_path(out, rel, ds).exists()]
    if missing:
        print(f'{rel}: parts missing for {",".join(missing)}; the merged file is written once every dataset is done')
        return missing
    rows = []
    for ds in order:
        rows += read_csv(part_path(out, rel, ds))
    write_csv(Path(out) / rel, fields, rows, crlf)
    print(f'wrote {Path(out) / rel} ({len(rows)} rows)')
    return []
