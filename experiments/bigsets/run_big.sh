#!/usr/bin/env bash
# Runner for the large datasets: sift100m, then the msmarco group (msmarco10m prefix + msmarco113m). Per group: statistics (mean, PCA axes) -> pass A (ground truth,
# exact Cov(u), codec statistics) -> freeze (3 codecs per dataset) -> pass B (Recall@10 at B = 3) -> score (score error, decomposition)
# -> calc (constants, Table III row). Every step skips finished outputs and the passes resume from their checkpoints, so rerunning this
# script resumes. The python steps wait by themselves while memory is short (pcafair.bigsets.gate). Stop: touch $PCARR_BIG_WORK/STOP.
# Needs: the raw files ($PCARR_BIG_RAW, data/README.md section 3), the C kernels (make -C csrc), scipy and threadpoolctl.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PCARR_WORK=${PCARR_WORK:-$PWD/work}; export PCARR_OUT=${PCARR_OUT:-$PCARR_WORK/results}; export PCARR_BIG_WORK=${PCARR_BIG_WORK:-$PCARR_WORK/bigsets}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-24} MKL_NUM_THREADS=${MKL_NUM_THREADS:-24}
OUT=$PCARR_OUT/bigsets; LOGS=$PCARR_BIG_WORK/logs; mkdir -p "$OUT" "$LOGS"
say() { echo "- $(date '+%F %H:%M') $*" | tee -a "$PCARR_BIG_WORK/progress.md"; }
run() { local s=$1; shift; nice -n 19 python3 experiments/bigsets/$s "$@" >> "$LOGS/${s%.py}_$1.log" 2>&1 || { say "FAILED $s $* (see $LOGS/${s%.py}_$1.log)"; exit 1; }; }
say "runner start (pid $$): sift100m -> msmarco group"
for grp in sift100m msmarco; do
  case $grp in sift100m) sets="sift100m";; msmarco) sets="msmarco10m msmarco113m";; esac
  for ds in $sets; do run big_stats.py $ds; done
  [ -f "$OUT/.passA_$grp.done" ] || { run big_passA.py $grp && touch "$OUT/.passA_$grp.done"; }
  for ds in $sets; do run big_freeze.py $ds; done
  done_b=1; for ds in $sets; do [ -f "$OUT/fullscan/$ds.csv" ] || done_b=0; done
  [ $done_b = 1 ] || run big_passB.py $grp
  for ds in $sets; do [ -f "$OUT/decomp/$ds.a2.json" ] || run big_score.py $ds; done
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 run big_calc.py ${sets// /,}
  say "group $grp complete"
done
say "all groups complete"
