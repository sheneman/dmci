#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_real_islands.sh: Launch 3 OpenEvolve battery islands on the REAL Severson target (lifefrac SOH, 117 cells).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

# Launch 3 OpenEvolve battery islands on the REAL Severson target (lifefrac SOH, 117 cells).
# Mirrors the synthetic run (slurm_battery_node.sh, 200 iters, 24 workers, qwen3.6 real preset)
# but points BAT_TARGET at target_real.pt and outputs to bat_real_island_{0,1,2}, leaving the
# synthetic target.pt + bat_island_* intact. Search split = KSPLIT_LATE=70 (cleaner gradient
# signal; rescore_real.py re-scores winners at BOTH 70 and 45). Nodes overridable as args 1-3.
set -euo pipefail
cd /mnt/ceph/sheneman/src/nncompile
TGT=experiments/exp_battery/results/target_real.pt
KS=${KS:-70}
SL=experiments/exp_battery/openevolve/slurm_battery_node.sh
N0=${1:-n121}; N1=${2:-n122}; N2=${3:-n002}
P=${PART:-gpu-8}
for i in 0 1 2; do
  eval "NODE=\$N$i"
  sbatch --partition="$P" --nodelist="$NODE" --job-name="bat_real$i" \
    --export=ALL,BAT_TARGET="$TGT",BAT_KSPLIT="$KS" \
    "$SL" "$i" "experiments/exp_battery/results/bat_real_island_$i" 24
done
echo "launched 3 real-data islands (ksplit=$KS) on $N0 $N1 $N2"
squeue -u "$USER" | tail -6
