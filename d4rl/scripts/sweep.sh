#!/bin/bash
# usage: sweep.sh <script> <env> <gpu>
source $WORKDIR/env.sh
cd $WORKDIR/PG
S=$1; E=$2; G=$3
mkdir -p sweep_logs
for N in 5 10 50 200 500; do
  echo "$N argmax 1.0"; echo "$N soft 1.0"; echo "$N soft 0.5"
done | xargs -P 5 -L 1 bash -c "CUDA_VISIBLE_DEVICES=$G python $S --env_name $E --seed 0 --mode eval --planner_num_candidates \$0 --select_rule \$1 --soft_temp \$2 --eval_seed 10000 --out_json sweep_${E}.jsonl > sweep_logs/${E}_N\$0_\$1_\$2.log 2>&1"
