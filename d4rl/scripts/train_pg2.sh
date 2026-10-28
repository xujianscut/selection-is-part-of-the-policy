#!/bin/bash
# usage: train_pg2.sh "<seeds>" "<domain env alpha>;..."  round-robin GPUs 7 8 9
source $WORKDIR/env.sh
cd $WORKDIR/PG
gpus=(7 8 9); i=0
IFS=";" read -ra SPECS <<< "$2"
for s in $1; do for spec in "${SPECS[@]}"; do
  set -- $spec
  g=${gpus[$((i % 3))]}; i=$((i+1))
  test -f checkpoint/DV/$2/$s/model_1000000.pkl || { echo "skip $2 s$s (no DV ckpt)"; continue; }
  CUDA_VISIBLE_DEVICES=$g nohup python pg_d4rl_$1.py --env_name $2 --seed $s --alpha $3 > logs/pg_${2}_s${s}.log 2>&1 < /dev/null &
done; done
