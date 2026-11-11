#!/usr/bin/env bash
# End-to-end check of the experiment layer on the smallest dataset (audio: 53,387 rows, 192 dimensions, 200 queries):
# fit and freeze the three codecs, then run every ten-dataset experiment, into a fresh work directory, and compare the files with
# results/. Needs the prepared audio files ($PCARR_PREPARED/audio-{db,query,gt-l2}.npy) and the raw audio .fvecs ($PCARR_RAW/audio/)
# for the original-coordinate arm. About 10-20 minutes on one machine.
#   PCARR_PREPARED=... PCARR_RAW=... bash tests/parity_audio.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PCARR_WORK=${PCARR_WORK:-$PWD/work/parity}
export PCARR_OUT=$PCARR_WORK/results PCARR_FROZEN=$PCARR_WORK/frozen
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
mkdir -p "$PCARR_OUT" "$PCARR_FROZEN"
python3 experiments/codec/freeze_codecs.py audio
python3 experiments/fullscan/fullscan.py audio pca,rot,orig,pcaslope
python3 experiments/fullscan/fullscan_slope_fine.py audio
python3 experiments/score_error/score_error.py audio
python3 experiments/score_error/score_error_b57.py audio
python3 experiments/score_error/score_error_orig.py audio
python3 experiments/score_error/decomp.py audio
python3 experiments/calc/calc.py
python3 experiments/rerank/rerank_ranks.py audio pca,rot
python3 experiments/guarantee/guarantee.py audio
python3 experiments/calc/export_spectra.py --datasets audio
python3 tests/compare_dir.py "$PCARR_OUT" --pattern audio
