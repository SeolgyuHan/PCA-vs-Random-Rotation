#!/bin/bash
# runner: sift100m, then the msmarco group (msmarco10m prefix + msmarco113m); per group: pass A (ground truth, exact Cov(u),
# codec statistics) -> freeze (3 codecs per dataset) -> pass B (Figure 1 recall at B = 3) -> score (Figure 3, S) -> calc (Figure 2,
# Table II). Every step skips finished outputs and the two passes resume from their checkpoints, so rerunning this script resumes.
# The python steps wait by themselves while memory is short or the E7 timing window is pending (pcafair.bigsets.gate).
# Stop: touch STOP (the python steps exit at the next chunk; rerun this script to resume).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
LOG=BL2_PROGRESS_LOG.md; say() { echo "- $(date '+%F %H:%M') $*" >> $LOG; }
run() { local s=$1; shift; nice -n 19 python3 $s "$@" >> logs/${s%.py}_$1.log 2>&1; local rc=$?
        if [ $rc -ne 0 ]; then say "runner: FAILED $s $* (exit $rc), see logs/${s%.py}_$1.log"; exit $rc; fi; }
say "runner start (pid $$): sift100m -> msmarco group"
for grp in sift100m msmarco; do
  case $grp in sift100m) sets="sift100m";; msmarco) sets="msmarco10m msmarco113m";; esac
  [ -f results/.passA_$grp.done ] || { run big_passA.py $grp && touch results/.passA_$grp.done; }
  for ds in $sets; do run big_freeze.py $ds; done
  done_b=1; for ds in $sets; do [ -f results/fullscan/$ds.csv ] || done_b=0; done
  [ $done_b = 1 ] || run big_passB.py $grp
  for ds in $sets; do [ -f results/decomp/$ds.a2.json ] || run big_score.py $ds; done
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 run big_calc.py ${sets// /,}
  say "runner: group $grp complete"
done
say "runner: all groups complete"
