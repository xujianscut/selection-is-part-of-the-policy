#!/bin/bash
# usage: train_pg.sh <seed>   (alpha from PG Table 6, KL divergence)
source $WORKDIR/env.sh
cd $WORKDIR/PG
s=$1; gpus=(7 8 9); i=0
for spec in "kitchen kitchen-partial-v0 1.0" "kitchen kitchen-mixed-v0 1.0" "antmaze antmaze-medium-play-v2 1.0" "antmaze antmaze-large-play-v2 50.0" "maze2d maze2d-large-v1 0.1"; do
  set -- $spec
  g=${gpus[$((i % 3))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g nohup python pg_d4rl_$1.py --env_name $2 --seed $s --alpha $3 > logs/pg_${2}_s${s}.log 2>&1 < /dev/null &
done
