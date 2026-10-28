from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = str(ROOT / 'results') + '/'
OUT = str(ROOT / 'paper_assets') + '/'
(ROOT / 'paper_assets').mkdir(exist_ok=True)
import json, glob, re, numpy as np
from collections import defaultdict
D = RES
pm = lambda v: f"{np.mean(v):.2f}{{\\scriptsize$\\pm${np.std(v, ddof=1)/np.sqrt(len(v)):.2f}}}"
pm1 = lambda v: f"{np.mean(v):.1f}{{\\scriptsize$\\pm${np.std(v, ddof=1)/np.sqrt(len(v)):.1f}}}"
NUMS = {}
# ---------------- OGBench ----------------
files = glob.glob(D + 'ogbench/*_eval_*.json'); assert files
A = defaultdict(dict)
for f in files:
    for r in json.load(open(f)):
        A[(r['env'], r['rule'], r['N'])][r['seed']] = r['success']
tasks = [('cube-single-play-singletask-v0', 'cube-single'), ('cube-double-play-singletask-v0', 'cube-double'), ('scene-play-singletask-v0', 'scene')]
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
rules = [('mean', 'argmax $\\bar Q$'), ('lcb2', 'LCB, $\\rho{=}2$'), ('lcb4', 'LCB, $\\rho{=}4$'), ('min', 'min over heads'),
         ('xfit', 'cross-fit heads'), ('mean_t05', 'argmax, noise $\\times 0.5$'), ('soft0.3', 'soft, $T{=}0.3$'), ('soft1', 'soft, $T{=}1$')]
def curve(env, rule):
    out = []
    for N in NS:
        d = A[(env, rule if N > 1 else 'mean', N)]; assert sorted(d) == [0, 1, 2, 3, 4], (env, rule, N)
        out.append(np.array([d[s] for s in range(5)]))
    return out
L = ["\\begin{tabular}{l ccc ccc ccc}", "\\toprule",
     " & \\multicolumn{3}{c}{cube-single} & \\multicolumn{3}{c}{cube-double} & \\multicolumn{3}{c}{scene} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "Selection rule & peak & $N{=}256$ & worst$_{N\\ge 8}$ & peak & $N{=}256$ & worst$_{N\\ge 8}$ & peak & $N{=}256$ & worst$_{N\\ge 8}$ \\\\", "\\midrule"]
for rule, lab in rules:
    cells = []
    for env, short in tasks:
        c = curve(env, rule); means = [x.mean() for x in c]
        k = int(np.argmax(means)); w = 3 + int(np.argmin(means[3:]))
        cells += [f"{means[k]:.2f}", f"{means[-1]:.2f}", f"{means[w]:.2f}"]
        NUMS[f"og_{short}_{rule}"] = dict(peak=means[k], peakN=NS[k], n256=means[-1], worst8=means[w])
    L.append(lab + " & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(OUT + 'tab_ogbench.tex', 'w').write("\n".join(L) + "\n")
# full OGBench table with standard errors (appendix)
NSF = NS[1:]
L = ["\\begin{tabular}{ll " + "c" * len(NSF) + " c}", "\\toprule",
     "Task & Rule & " + " & ".join(f"$N{{=}}{n}$" for n in NSF) + " & peak ($N^\\ast$) \\\\", "\\midrule"]
for ti, (env, short) in enumerate(tasks):
    for ri, (rule, lab) in enumerate(rules):
        c = curve(env, rule)[1:]
        means = [x.mean() for x in c]; k = int(np.argmax(means))
        cells = [pm(x) for x in c]
        L.append((short if ri == 0 else "") + f" & {lab} & " + " & ".join(cells) + f" & {pm(c[k])}\\,({NSF[k]}) \\\\")
    if ti < len(tasks) - 1:
        L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
open(OUT + 'tab_ogbench_full.tex', 'w').write("\n".join(L) + "\n")
# offline diagnostic (appendix)
offs = glob.glob(D + 'ogbench/*_offline.json'); assert offs
O = defaultdict(lambda: defaultdict(list))
for f in offs:
    env = re.match(r'(.+?)_s\d+_', f.split('/')[-1]).group(1)
    for r in json.load(open(f)):
        O[env][r['N']].append((r['q_select_heads'], r['q_heldout_heads']))
L = ["\\begin{tabular}{l rrr rrr}", "\\toprule", " & \\multicolumn{3}{c}{selecting heads} & \\multicolumn{3}{c}{held-out heads} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Task & $N{=}1$ & $N{=}16$ & $N{=}256$ & $N{=}1$ & $N{=}16$ & $N{=}256$ \\\\", "\\midrule"]
for env, short in tasks + [('puzzle-3x3-play-singletask-v0', 'puzzle-3x3'), ('antmaze-large-navigate-singletask-v0', 'antmaze-large')]:
    if env not in O: continue
    s = lambda N, j: np.mean([x[j] for x in O[env][N]])
    L.append(f"{short} & " + " & ".join(f"{s(N,0):.1f}" for N in (1, 16, 256)) + " & " + " & ".join(f"{s(N,1):.1f}" for N in (1, 16, 256)) + " \\\\")
    NUMS[f"off_{short}"] = dict(sel1=s(1, 0), sel256=s(256, 0), ho1=s(1, 1), ho256=s(256, 1), nseeds=len(O[env][1]))
L += ["\\bottomrule", "\\end{tabular}"]
open(OUT + 'tab_offline.tex', 'w').write("\n".join(L) + "\n")
# ---------------- D4RL reproduction ----------------
fin = {}
for line in open(D + 'd4rl/final_evals.txt'):
    p = line.split()
    if len(p) < 2: continue
    m = re.match(r'(dv|pg)_(.+?)(?:-v\d)?_s(\d)$', p[0]); assert m, p[0]
    fin.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = float(p[1])
t3 = json.load(open(D + 'pg_published/pg_table3.json'))
envs = [('kitchen-mixed', 'kitchen-mixed-v0'), ('kitchen-partial', 'kitchen-partial-v0'), ('antmaze-medium-play', 'antmaze-medium-play-v2'),
        ('antmaze-large-play', 'antmaze-large-play-v2'), ('maze2d-large', 'maze2d-large-v1')]
L = ["\\begin{tabular}{l cc c cc c}", "\\toprule", " & \\multicolumn{2}{c}{DV$^\\star$, $N{=}50$, argmax} & & \\multicolumn{2}{c}{PG (released code)} & \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}", "Task & reported & ours (5 seeds) & & reported & ours (3 seeds) & PG$-$DV$^\\star$ (ours, paired) \\\\", "\\midrule"]
for short, full in envs:
    dv = fin[('dv', short)]; pg = fin[('pg', short)]
    assert sorted(dv) == [0, 1, 2, 3, 4] and sorted(pg) == [0, 1, 2], (short, sorted(dv), sorted(pg))
    dvv = np.array([dv[s] for s in range(5)]); pgv = np.array([pg[s] for s in range(3)])
    diff = pgv - dvv[:3]
    rep_dv, rep_pg = t3[full]['DV*_N50'], t3[full]['PG']
    L.append(f"{short} & {rep_dv[0]:.1f} & {pm1(dvv)} & & {rep_pg[0]:.1f} & {pm1(pgv)} & " + "/".join(f"{d:+.0f}" for d in diff) + " \\\\")
    NUMS[f"repro_{short}"] = dict(dv=dvv.tolist(), pg=pgv.tolist(), rep_dv=rep_dv, rep_pg=rep_pg)
L += ["\\bottomrule", "\\end{tabular}"]
open(OUT + 'tab_repro.tex', 'w').write("\n".join(L) + "\n")
# ---------------- D4RL N x rule (5 seeds) ----------------
rows = []
for f in glob.glob(D + 'd4rl/sweep*.jsonl'): rows += [json.loads(l) for l in open(f)]
assert rows
S = defaultdict(dict)
for r in rows:
    if r['rule'] == 'argmax' or (r['rule'] == 'soft' and r['T'] == 0.5): S[(r['env'], r['rule'], r['N'])][r['seed']] = r['ret']
for env in ['antmaze-medium-play-v2', 'kitchen-partial-v0']:
    for rule in ['argmax', 'soft']:
        for N in [5, 50, 500]:
            d = S[(env, rule, N)]; assert sorted(d) == [0, 1, 2, 3, 4]
            v = np.array([d[s] for s in range(5)])
            NUMS[f"sw_{env}_{rule}_{N}"] = dict(mean=v.mean(), se=v.std(ddof=1) / np.sqrt(5))
json.dump(NUMS, open(D + 'paper_numbers.json', 'w'), indent=1, default=float)
print('tables written;', len(NUMS), 'numbers')
for k in ['og_cube-double_mean', 'og_cube-double_soft1', 'og_scene_mean', 'og_scene_soft1', 'og_cube-single_mean', 'og_cube-single_soft1', 'og_cube-double_lcb2', 'og_cube-double_xfit']:
    print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in NUMS[k].items()})
for k in NUMS:
    if k.startswith('sw_') or k.startswith('off_'): print(k, {a: round(b, 2) if isinstance(b, float) else b for a, b in NUMS[k].items()})
