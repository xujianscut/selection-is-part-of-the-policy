#!/bin/bash
# usage: train_seeds.sh "<seeds>" ; round-robin over GPUs 7 8 9
source $WORKDIR/env.sh
cd $WORKDIR/PG
gpus=(7 8 9); i=0
for s in $1; do
  for spec in "kitchen kitchen-partial-v0" "kitchen kitchen-mixed-v0" "antmaze antmaze-medium-play-v2" "antmaze antmaze-large-play-v2" "maze2d maze2d-large-v1"; do
    set -- $spec
    g=${gpus[$((i % 3))]}; i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g nohup python DV/dv_d4rl_$1.py --env_name $2 --seed $s > logs/dv_${2}_s${s}.log 2>&1 < /dev/null &
  done
done
