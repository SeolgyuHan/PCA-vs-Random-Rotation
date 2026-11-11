#!/usr/bin/env bash
# Results layer: regenerate the paper's numbers and figures from results/ and check them (about one minute).
# bash reproduce.sh
# Experiment layer (prepared datasets and the frozen codecs needed): see README.md, "Rerunning the experiments".
set -euo pipefail
cd "$(dirname "$0")"
python3 numbers/make_numbers.py                 # recompute every number of the paper from results/ and compare with numbers/numbers.json
for f in fig1_recall fig2_spectrum fig3_score fig4_rerank fig5_slope; do
  MPLBACKEND=Agg python3 figures/$f.py > /dev/null  # -> figures/out/<name>.png and.pdf
done
python3 tests/check_results.py                  # pixel comparison with figures/published/
