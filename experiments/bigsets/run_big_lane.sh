#!/bin/bash
# lane runner for one group (sift100m | msmarco): pass A -> freeze -> pass B -> score -> calc, resumable (finished outputs are
# skipped, passes resume from checkpoints). Two lanes run side by side; the python steps wait by themselves for memory and for the
# E7 timing window (pcafair.bigsets.gate / wait_mem). Stop: touch STOP; rerun the lane to resume. Usage: bash run_big_lane.sh <group>
cd "$(dirname "$0")"
grp=$1; case $grp in sift100m) sets="sift100m";; msmarco) sets="msmarco10m msmarco113m";; *) echo "unknown group"; exit 2;; esac
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
LOG=BL2_PROGRESS_LOG.md; say() { echo "- $(date '+%F %H:%M') $*" >> $LOG; }
run() { local s=$1; shift; nice -n 19 python3 $s "$@" >> logs/${s%.py}_$1.log 2>&1; local rc=$?
        if [ $rc -ne 0 ]; then say "lane $grp: FAILED $s $* (exit $rc), see logs/${s%.py}_$1.log"; exit $rc; fi; }
say "lane $grp start (pid $$)"
done_a=1; for ds in $sets; do [ -f ${PCARR_BIG_WORK:-work/bigsets}/$ds-passA.npz ] || done_a=0; done
[ $done_a = 1 ] || run big_passA.py $grp
for ds in $sets; do run big_freeze.py $ds; done
done_b=1; for ds in $sets; do [ -f results/fullscan/$ds.csv ] || done_b=0; done
[ $done_b = 1 ] || run big_passB.py $grp
for ds in $sets; do [ -f results/decomp/$ds.a2.json ] || run big_score.py $ds; done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 run big_calc.py ${sets// /,}
say "lane $grp complete"
