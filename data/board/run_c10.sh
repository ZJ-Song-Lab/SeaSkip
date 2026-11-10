#!/bin/bash
# C10: v4 board repeated runs for confidence intervals (5x full + 5x skip)
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
cd ~/seaskip
rm -f c10_done.flag
for i in 1 2 3 4 5; do
  ./seaskip_eval_v4 gate_tiny_v4.om det_hrsid.om patches/hrsid_test results_v4_c10_full_$i.csv full > eval_v4_c10_full_$i.log 2>&1
done
for i in 1 2 3 4 5; do
  ./seaskip_eval_v4 gate_tiny_v4.om det_hrsid.om patches/hrsid_test results_v4_c10_skip_$i.csv skip > eval_v4_c10_skip_$i.log 2>&1
done
touch c10_done.flag