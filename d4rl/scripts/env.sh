source ~/miniconda3/etc/profile.d/conda.sh && conda activate pgdv
export TMPDIR=$WORKDIR/tmp LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$CONDA_PREFIX/lib:/usr/lib/nvidia D4RL_DATASET_DIR=$WORKDIR/d4rl_data XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled MUJOCO_GL=egl
