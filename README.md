# Selection is part of the policy

Code, logs and results for the paper *Selection is part of the policy: candidate reranking as an uncontrolled variable in generative-policy offline reinforcement learning* (submitted to Neurocomputing).

Generative policies and diffusion planners are often deployed with a selection step: sample `N` candidates, let a critic pick one. This repository measures how much `N` and the selection rule change performance, and contains everything needed to regenerate every table and figure in the paper.

## Layout

| Path | Contents |
|---|---|
| `ogbench/bonsel.py` | OGBench study (Section 4): flow behavior-cloning generator, 10-head IQL critic with episode-split heads, evaluation of all selection rules over `N` |
| `ogbench/logs/` | Training and evaluation logs of the OGBench runs (5 tasks, seeds 0 to 4 for the three main tasks) |
| `ogbench/diag.py` | Trajectory diagnostics: empirical calibration curve by exact branching rollouts, and distance of visited states to the data |
| `ogbench/logs_diag_emaq/` | Logs of the diagnostics and of the EMaQ-critic runs |
| `d4rl/pg_9526eb9.patch` | Patch for [ku-dmlab/PG](https://github.com/ku-dmlab/PG) at commit `9526eb9` (MIT license) used in Sections 5 and 6 |
| `d4rl/scripts/` | Launch scripts for DV* training, PG training and the `N` x rule sweeps |
| `d4rl/logs/`, `d4rl/sweep_logs/` | Training and evaluation logs for DV* (5 seeds) and PG (3 seeds) on D4RL |
| `results/` | Raw results: OGBench evaluation and offline-diagnostic JSON, D4RL final evaluations and sweeps, numbers parsed from the PG paper |
| `analysis/` | Scripts that turn `results/` into the paper's tables and figures (`paper_assets/`), plus a Monte Carlo check of the theory |

## Regenerating the tables and figures

```bash
pip install numpy matplotlib
python analysis/make_tables.py        # Tables 2, 4, 5 -> paper_assets/
python analysis/budget_table.py       # Table 3 (budget-matched comparison) -> paper_assets/
python analysis/make_figs.py          # Figures 1, 2 -> paper_assets/
python analysis/make_diag.py          # Figures 2-3 and the EMaQ table (diagnostics, second critic family)
python analysis/verify_theory.py      # Monte Carlo check of Theorem 1, Propositions 2 and 4
```

All scripts assert that their inputs are complete (for example, five seeds per configuration) and fail otherwise.

`results/pg_published/pg_table3.json` and `pg_alpha.json` contain the numbers we parsed from the appendix of the Prior Guidance paper (its Table 3 and Tables 11 to 14). They are committed, so the scripts above do not need the paper. To re-derive them, extract the paper text with `pdftotext` into `results/pg_published/pg_paper.txt` and run `analysis/parse_pg_table3.py` and `analysis/parse_pg_alpha.py`; both check several values against the PDF and check that PG's main-table score equals the maximum of its alpha ablation for all 18 tasks.

## OGBench experiments

Environment used: Python 3.10/3.12, PyTorch 2.5, `pip install ogbench` (MuJoCo 3.x).

```bash
python ogbench/bonsel.py --env cube-double-play-singletask-v0 --seed 0 --split --qclip \
    --rules mean,lcb2,lcb4,min,xfit,soft1,soft0.3,mean_t05 --data <dataset_dir> --out <run_dir>
```

Rule names: `mean` = argmax of the mean critic, `lcbR` = mean minus `R` standard deviations, `min` = minimum over heads, `xfit` = argmax of the five heads trained on one half of the episodes, `softT` = sample proportional to `exp(z/T)` with `z` the critic value standardized across candidates, `mean_t05` = argmax with base noise scaled by 0.5. Each run trains for 5e5 steps and evaluates 50 episodes per configuration with shared environment seeds, writing `*_eval_*.json` and `*_offline.json`.

Second critic family (EMaQ-style backup over 8 behavior samples): add `--critic emaq --steps 300000`.

Diagnostics on a trained IQL checkpoint:

```bash
python ogbench/diag.py --env cube-single-play-singletask-v0 --seed 0 --Nc 16 --k 20 \
    --ckpt <run_dir>/cube-single-play-singletask-v0_s0_K10_e0.9_split_clip.pt --data <dataset_dir> --out <out_dir>
```

## D4RL experiments (DV* and Prior Guidance)

Environment used: MuJoCo 2.1.0 binaries with `mujoco-py<2.2`, `gym==0.23.1`, `numpy==1.26.4`, `jax[cuda12]==0.4.35`, `flax==0.10.2`, `mujoco==2.3.7`, `dm_control==1.0.14` (newer versions break the D4RL kitchen environments), D4RL from the Farama repository. With recent `nvidia-cuda-nvcc-cu12` wheels JAX needs an empty `__init__.py` in `site-packages/nvidia/cuda_nvcc/`.

```bash
git clone https://github.com/ku-dmlab/PG.git && cd PG && git checkout 9526eb9
git apply ../d4rl/pg_9526eb9.patch
export WORKDIR=<directory containing PG/ and env.sh>   # see d4rl/scripts/env.sh
python DV/dv_d4rl_kitchen.py --env_name kitchen-partial-v0 --seed 0                     # train DV*
python pg_d4rl_kitchen.py --env_name kitchen-partial-v0 --seed 0 --alpha 1.0              # train PG on top of DV*
python DV/dv_d4rl_kitchen.py --env_name kitchen-partial-v0 --seed 0 --mode eval \
    --planner_num_candidates 500 --select_rule soft --soft_temp 0.5 --eval_seed 10000 --out_json sweep.jsonl
```

The patch:

* adds the four flags that the released PG training scripts read but do not define (`guidance_type`, `pipeline_type`, `save_interval`, `planner_num_candidates`), with the defaults of the corresponding DV* scripts; without them the scripts fail at start-up;
* adds an `eval` mode to the DV* scripts that reloads a trained checkpoint (and defines the critic object, which the released scripts only create inside the training loop);
* adds soft selection (`--select_rule soft --soft_temp T`), per-episode environment seeding (`--eval_seed`) and JSON output (`--out_json`) to the evaluation loop. The default behavior (`argmax`, unseeded) is unchanged.

`d4rl/scripts/` contains the exact launch scripts (`train_seeds.sh`, `train_pg.sh`, `train_pg2.sh`, `sweep.sh`, `sweep2.sh`, `rerun_missing.py`).

## License

MIT for the code in this repository. The patch modifies code from ku-dmlab/PG, which is also MIT-licensed.
