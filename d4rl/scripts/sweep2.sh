#!/bin/bash
# usage: sweep2.sh <script> <env> <seed> <gpu> ; N in {5,50,500} x {argmax, soft0.5}
source $WORKDIR/env.sh
cd $WORKDIR/PG
S=$1; E=$2; SD=$3; G=$4
mkdir -p sweep_logs
for N in 5 50 500; do echo "$N argmax 1.0"; echo "$N soft 0.5"; done | \
  xargs -P 6 -L 1 bash -c "CUDA_VISIBLE_DEVICES=$G python $S --env_name $E --seed $SD --mode eval --planner_num_candidates \$0 --select_rule \$1 --soft_temp \$2 --eval_seed 10000 --out_json sweep2_${E}.jsonl > sweep_logs/${E}_s${SD}_N\$0_\$1_\$2.log 2>&1"
