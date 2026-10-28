"""Figures and tables for the diagnostics (calibration curve, state shift) and the EMaQ critic study."""
import json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'; OUT = ROOT / 'paper_assets'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'legend.frameon': False, 'pdf.fonttype': 42})
C = dict(argmax='#2a78d6', lcb='#eb6834', xfit='#1baf7a', soft='#eda100', ref='#8a8a85')
MK = dict(argmax='o', lcb='s', xfit='^', soft='D')
TASKS = [('cube-single', 'cube-single-play-singletask-v0'), ('cube-double', 'cube-double-play-singletask-v0'),
         ('scene', 'scene-play-singletask-v0')]
se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
NUM = {}

# ---------------- diagnostics ----------------
fig, axs = plt.subplots(2, 3, figsize=(6.75, 3.6))
for j, (short, env) in enumerate(TASKS):
    fs = sorted(glob.glob(str(RES / 'ogbench_diag' / f'diag_{env}_s*.json')))
    assert len(fs) == 3, (env, fs)
    D = [json.load(open(f)) for f in fs]
    us = [float(u) for u in D[0]['calib']]
    rows = [r for d in D for r in d['rows']]
    m, s = [], []
    for u in us:
        v = np.array([r['success'] for r in rows if r['u'] == u]); m.append(v.mean()); s.append(se(v))
    # paired top-1 minus median, over anchors
    by = defaultdict(dict)
    for si, d in enumerate(D):
        for r in d['rows']:
            by[(si, r['episode'], r['anchor'])][r['u']] = r['success']
    diff = np.array([v[1.0] - v[0.5] for v in by.values()])
    NUM[f'calib_{short}'] = dict(top1_minus_median=diff.mean(), se=se(diff), anchors=len(diff), Nc=D[0]['Nc'], k=D[0]['k'],
                                 success=dict(zip(map(str, us), m)),
                                 qpred={u: np.mean([d['calib'][u]['qpred'] for d in D]) for u in D[0]['calib']})
    ax = axs[0, j]; x = np.arange(len(us))
    ax.errorbar(x, m, yerr=s, color=C['argmax'], lw=1.6, marker='o', ms=3.2, capsize=1.5)
    ax.set_xticks(x); ax.set_xticklabels([f'{u:g}' for u in us], fontsize=6.5)
    ax.set_title(f'{short}', fontsize=8.5); ax.grid(axis='y', lw=0.4, color='#e4e3df')
    ax.set_xlabel('followed critic quantile $u$')
    # state shift
    ax = axs[1, j]
    for rule, key, lab in [('argmax', 'argmax', 'argmax'), ('soft1', 'soft', 'soft ($T{=}1$)')]:
        Ns = [1, 4, 16, 64, 256] if rule == 'argmax' else [4, 16, 64, 256]
        mm, ss = [], []
        for N in Ns:
            k = f'{rule}_{N}'; v = [d['shift'][k]['nn_mean'] for d in D]; mm.append(np.mean(v)); ss.append(se(v))
            NUM[f'shift_{short}_{k}'] = dict(nn=np.mean(v), se=se(v), success=np.mean([d['shift'][k]['success'] for d in D]))
        ax.errorbar(Ns, mm, yerr=ss, color=C[key], lw=1.6, marker=MK[key], ms=3.2, capsize=1.5, label=lab)
    ax.set_xscale('log', base=2); ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(['1', '4', '16', '64', '256'])
    ax.set_xlabel('candidates $N$'); ax.grid(axis='y', lw=0.4, color='#e4e3df')
axs[0, 0].set_ylabel('success rate'); axs[1, 0].set_ylabel('distance to nearest\ndataset state')
axs[1, 0].legend(loc='upper left', fontsize=6.5)
fig.tight_layout(pad=0.3); fig.savefig(OUT / 'fig_diag.pdf', bbox_inches='tight'); fig.savefig(OUT / 'fig_diag.png', dpi=200, bbox_inches='tight')

# ---------------- EMaQ critic ----------------
A = defaultdict(dict)
files = glob.glob(str(RES / 'ogbench_emaq' / '*_eval_*.json')); assert files
for f in files:
    for r in json.load(open(f)):
        A[(r['env'], r['rule'], r['N'])][r['seed']] = r['success']
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
rules = [('mean', 'argmax', 'argmax $\\bar Q$'), ('lcb2', 'lcb', 'LCB ($\\rho{=}2$)'), ('xfit', 'xfit', 'cross-fit heads'),
         ('soft1', 'soft', 'soft ($T{=}1$)')]
def curve(env, rule):
    out = []
    for N in NS:
        d = A[(env, rule if N > 1 else 'mean', N)]; assert sorted(d) == [0, 1, 2], (env, rule, N, sorted(d))
        out.append(np.array([d[s] for s in range(3)]))
    return out
fig, axs = plt.subplots(1, 3, figsize=(6.75, 1.9))
for ax, (short, env) in zip(axs, TASKS):
    for rule, key, lab in rules:
        c = curve(env, rule); m = np.array([x.mean() for x in c]); s = np.array([se(x) for x in c])
        ax.plot(NS, m, color=C[key], lw=1.6, marker=MK[key], ms=3.2, label=lab)
        ax.fill_between(NS, m - s, m + s, color=C[key], alpha=0.15, lw=0)
    ax.set_xscale('log', base=2); ax.set_xticks(NS[::2]); ax.set_xticklabels([str(n) for n in NS[::2]])
    ax.set_title(short, fontsize=8.5); ax.set_xlabel('candidates $N$'); ax.grid(axis='y', lw=0.4, color='#e4e3df')
axs[0].set_ylabel('success rate'); axs[0].legend(loc='lower left', fontsize=6.3, handlelength=1.6)
fig.tight_layout(pad=0.3); fig.savefig(OUT / 'fig_emaq.pdf', bbox_inches='tight'); fig.savefig(OUT / 'fig_emaq.png', dpi=200, bbox_inches='tight')
pm = lambda v: f"{np.mean(v):.2f}{{\\scriptsize$\\pm${se(v):.2f}}}"
rules_t = [('mean', 'argmax $\\bar Q$'), ('lcb2', 'LCB, $\\rho{=}2$'), ('min', 'min over heads'), ('xfit', 'cross-fit heads'), ('soft1', 'soft, $T{=}1$')]
L = ["\\begin{tabular}{l ccc ccc ccc}", "\\toprule",
     " & \\multicolumn{3}{c}{cube-single} & \\multicolumn{3}{c}{cube-double} & \\multicolumn{3}{c}{scene} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "Selection rule & peak ($N^\\ast$) & $N{=}256$ & worst$_{N\\ge 8}$ & peak ($N^\\ast$) & $N{=}256$ & worst$_{N\\ge 8}$ & peak ($N^\\ast$) & $N{=}256$ & worst$_{N\\ge 8}$ \\\\", "\\midrule"]
for rule, lab in rules_t:
    cells = []
    for short, env in TASKS:
        c = curve(env, rule)[1:]; means = [x.mean() for x in c]; k = int(np.argmax(means)); w = 2 + int(np.argmin(means[2:]))
        cells += [f"{pm(c[k])}\\,({NS[1:][k]})", pm(c[-1]), pm(c[w])]
        NUM[f'emaq_{short}_{rule}'] = dict(peak=means[k], peak_se=se(c[k]), peakN=NS[1:][k], n256=means[-1], worst8=means[w])
    L.append(lab + " & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(OUT / 'tab_emaq.tex', 'w').write("\n".join(L) + "\n")
json.dump(NUM, open(RES / 'diag_emaq_numbers.json', 'w'), indent=1, default=float)
for k, v in NUM.items():
    if k.startswith('calib') or k.startswith('emaq_') and k.endswith(('_mean', '_soft1')):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in ('success', 'qpred')})
for k, v in NUM.items():
    if k.startswith('calib'):
        print(k, 'qpred', {u: round(q, 2) for u, q in v['qpred'].items()})
