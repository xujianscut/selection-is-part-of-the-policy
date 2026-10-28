from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = str(ROOT / 'results') + '/'
OUT = str(ROOT / 'paper_assets') + '/'
(ROOT / 'paper_assets').mkdir(exist_ok=True)
import json, glob, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
D = RES
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})
C = dict(argmax='#2a78d6', lcb='#eb6834', xfit='#1baf7a', soft='#eda100', ref='#8a8a85')
MK = dict(argmax='o', lcb='s', xfit='^', soft='D')
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# ---------- Figure 1: OGBench ----------
files = glob.glob(D + 'ogbench/*_eval_*.json'); assert files
A = defaultdict(dict)
for f in files:
    for r in json.load(open(f)):
        A[(r['env'], r['rule'], r['N'])][r['seed']] = r['success']
tasks = [('cube-single-play-singletask-v0', 'cube-single'), ('cube-double-play-singletask-v0', 'cube-double'),
         ('scene-play-singletask-v0', 'scene')]
rules = [('mean', 'argmax', 'argmax $\\bar Q$'), ('lcb2', 'lcb', 'LCB ($\\rho{=}2$)'),
         ('xfit', 'xfit', 'cross-fit heads'), ('soft1', 'soft', 'soft ($T{=}1$)')]
fig, axs = plt.subplots(1, 3, figsize=(6.75, 1.9))
for ax, (env, title) in zip(axs, tasks):
    for rule, key, lab in rules:
        m, s = [], []
        for N in NS:
            d = A[(env, rule if N > 1 else 'mean', N)]
            assert sorted(d) == [0, 1, 2, 3, 4], (env, rule, N, sorted(d))
            v = np.array(list(d.values())); m.append(v.mean()); s.append(v.std(ddof=1) / np.sqrt(len(v)))
        m, s = np.array(m), np.array(s)
        ax.plot(NS, m, color=C[key], lw=1.6, marker=MK[key], ms=3.2, label=lab)
        ax.fill_between(NS, m - s, m + s, color=C[key], alpha=0.15, lw=0)
    ax.set_xscale('log', base=2); ax.set_xticks(NS[::2]); ax.set_xticklabels([str(n) for n in NS[::2]])
    ax.set_title(title, fontsize=8.5); ax.set_xlabel('candidates $N$'); ax.grid(axis='y', lw=0.4, color='#e4e3df')
axs[0].set_ylabel('success rate')
axs[0].legend(loc='lower right', fontsize=6.5, handlelength=1.6)
fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig_ogbench.pdf'); fig.savefig(OUT + 'fig_ogbench.png', dpi=200)

# ---------- Figure 2: D4RL ----------
t3 = json.load(open(D + 'pg_published/pg_table3.json'))
rows = []
for f in glob.glob(D + 'd4rl/sweep*.jsonl'):
    rows += [json.loads(l) for l in open(f)]
assert rows
S = defaultdict(dict)
for r in rows:
    if r['rule'] == 'argmax' or (r['rule'] == 'soft' and r['T'] == 0.5):
        S[(r['env'], r['rule'], r['N'])][r['seed']] = r['ret']
fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.95))
for ax, (env, title) in zip(axs, [('antmaze-medium-play-v2', 'antmaze-medium-play'), ('kitchen-partial-v0', 'kitchen-partial')]):
    pn = [1, 5, 10, 50, 500]; pv = np.array([t3[env][f'DV*_N{n}'] for n in pn])
    ax.errorbar(pn, pv[:, 0], yerr=pv[:, 1], color=C['ref'], lw=1.2, ls='--', marker='x', ms=3.5, capsize=1.5,
                label='DV$^\\star$ argmax (PG paper)')
    if env in ('antmaze-medium-play-v2',):
        ax.axhline(t3[env]['PG'][0], color=C['ref'], lw=0.8, ls=':')
        ax.text(1.1, t3[env]['PG'][0] + 1.0, 'PG (reported)', color='#5f5e5a', fontsize=6)
    else:
        ax.axhline(t3[env]['PG'][0], color=C['ref'], lw=0.8, ls=':')
        ax.text(1.1, t3[env]['PG'][0] + 1.0, 'PG (reported)', color='#5f5e5a', fontsize=6)
    for rule, key, lab in [('argmax', 'argmax', 'ours, argmax'), ('soft', 'soft', 'ours, soft ($T{=}0.5$)')]:
        m, s = [], []
        for N in [5, 50, 500]:
            d = S[(env, rule, N)]; assert sorted(d) == [0, 1, 2, 3, 4], (env, rule, N, sorted(d))
            v = np.array(list(d.values())); m.append(v.mean()); s.append(v.std(ddof=1) / np.sqrt(len(v)))
        ax.errorbar([5, 50, 500], m, yerr=s, color=C[key], lw=1.6, marker=MK[key], ms=3.5, capsize=1.5, label=lab)
    ax.set_xscale('log'); ax.set_xticks(pn); ax.set_xticklabels([str(n) for n in pn])
    ax.set_title(title, fontsize=8.5); ax.set_xlabel('candidates $N$'); ax.grid(axis='y', lw=0.4, color='#e4e3df')
axs[0].set_ylabel('normalized return')
axs[1].legend(loc='lower left', fontsize=6, handlelength=1.6)
fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig_d4rl.pdf'); fig.savefig(OUT + 'fig_d4rl.png', dpi=200)
print('figures written')
