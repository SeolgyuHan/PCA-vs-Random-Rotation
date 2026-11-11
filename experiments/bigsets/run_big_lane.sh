#!/usr/bin/env bash
# One group of run_big.sh (sift100m | msmarco) so that two groups can run side by side on one machine; same steps, same resumption.
# Usage: bash experiments/bigsets/run_big_lane.sh <group>
set -uo pipefail
cd "$(dirname "$0")/../.."
grp=$1; case $grp in sift100m) sets="sift100m";; msmarco) sets="msmarco10m msmarco113m";; *) echo "unknown group $grp"; exit 2;; esac
export PCARR_WORK=${PCARR_WORK:-$PWD/work}; export PCARR_OUT=${PCARR_OUT:-$PCARR_WORK/results}; export PCARR_BIG_WORK=${PCARR_BIG_WORK:-$PCARR_WORK/bigsets}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-24} MKL_NUM_THREADS=${MKL_NUM_THREADS:-24}
OUT=$PCARR_OUT/bigsets; LOGS=$PCARR_BIG_WORK/logs; mkdir -p "$OUT" "$LOGS"
say() { echo "- $(date '+%F %H:%M') lane $grp: $*" | tee -a "$PCARR_BIG_WORK/progress.md"; }
run() { local s=$1; shift; nice -n 19 python3 experiments/bigsets/$s "$@" >> "$LOGS/${s%.py}_$1.log" 2>&1 || { say "FAILED $s $* (see $LOGS/${s%.py}_$1.log)"; exit 1; }; }
say "start (pid $$)"
for ds in $sets; do run big_stats.py $ds; done
done_a=1; for ds in $sets; do [ -f "$PCARR_BIG_WORK/$ds-passA.npz" ] || done_a=0; done
[ $done_a = 1 ] || run big_passA.py $grp
for ds in $sets; do run big_freeze.py $ds; done
done_b=1; for ds in $sets; do [ -f "$OUT/fullscan/$ds.csv" ] || done_b=0; done
[ $done_b = 1 ] || run big_passB.py $grp
for ds in $sets; do [ -f "$OUT/decomp/$ds.a2.json" ] || run big_score.py $ds; done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 run big_calc.py ${sets// /,}
say "complete"
