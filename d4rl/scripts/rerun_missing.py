"""After sweep2 finishes: rerun (env, seed, N, rule) configs missing from sweep2_*.jsonl, 2 at a time per GPU."""
import json, os, subprocess, itertools, glob
need = []
for env, script in [("antmaze-medium-play-v2", "DV/dv_d4rl_antmaze.py"), ("kitchen-partial-v0", "DV/dv_d4rl_kitchen.py")]:
    have = set()
    f = f"sweep2_{env}.jsonl"
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l); have.add((r["seed"], r["N"], r["rule"]))
    for sd, N, (rule, T) in itertools.product([1, 2, 3, 4], [5, 50, 500], [("argmax", 1.0), ("soft", 0.5)]):
        if (sd, N, rule) not in have:
            need.append((script, env, sd, N, rule, T))
print("missing:", len(need), flush=True)
gpus = [7, 8, 9]
procs = []
for i, (script, env, sd, N, rule, T) in enumerate(need):
    while len([p for p in procs if p.poll() is None]) >= 6:
        import time; time.sleep(30)
    cmd = (f"source $WORKDIR/env.sh && CUDA_VISIBLE_DEVICES={gpus[i % 3]} python {script} --env_name {env} --seed {sd} --mode eval "
           f"--planner_num_candidates {N} --select_rule {rule} --soft_temp {T} --eval_seed 10000 --out_json sweep2_{env}.jsonl "
           f"> sweep_logs/{env}_s{sd}_N{N}_{rule}_{T}_rerun.log 2>&1")
    procs.append(subprocess.Popen(["bash", "-c", cmd]))
for p in procs:
    p.wait()
print("done", flush=True)
