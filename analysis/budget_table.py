"""Budget-matched comparison of PG vs DV* using only numbers published in the PG paper (Tables 3, 11-14)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = str(ROOT / 'results') + '/'
OUT = str(ROOT / 'paper_assets') + '/'
(ROOT / 'paper_assets').mkdir(exist_ok=True)
import json, numpy as np
D = RES
t3 = json.load(open(D + 'pg_published/pg_table3.json')); al = json.load(open(D + 'pg_published/pg_alpha.json'))
assert set(t3) == set(al) and len(t3) == 18
NS = ['DV*_N1', 'DV*_N5', 'DV*_N10', 'DV*_N50', 'DV*_N500']
dom = lambda t: t.split('-')[0] if not t.startswith(('walker', 'hopper', 'halfcheetah')) else 'mujoco'
rows, summ = [], {}
for t in t3:
    pg_best = t3[t]['PG']                      # = max over alpha (asserted in parse_pg_alpha.py)
    pg_a1 = al[t]['1.0']                       # single global alpha available in every domain
    dv50 = t3[t]['DV*_N50']
    k = max(NS, key=lambda c: t3[t][c][0]); dvbest = t3[t][k]
    rows.append(dict(task=t, dom=dom(t), dv50=dv50, dvbest=dvbest, dvbestN=k.split('_N')[1], pg_best=pg_best, pg_a1=pg_a1))
def gap(a, b):  # difference and whether it exceeds 2 combined SEs
    d = a[0] - b[0]; se = np.hypot(a[1], b[1]); return d, abs(d) > 2 * se
out = {}
for name, pa, pb in [('reported', 'pg_best', 'dv50'), ('matched', 'pg_best', 'dvbest'), ('untuned', 'pg_a1', 'dv50')]:
    ds = [gap(r[pa], r[pb]) for r in rows]
    out[name] = dict(mean_gap=float(np.mean([d for d, _ in ds])), pg_sig_wins=int(sum(bool(d > 0 and s) for d, s in ds)),
                     dv_sig_wins=int(sum(bool(d < 0 and s) for d, s in ds)))
json.dump(dict(rows=rows, summary=out), open(D + 'pg_published/budget_summary.json', 'w'), indent=1)
for k, v in out.items():
    print(k, {a: round(b, 2) if isinstance(b, float) else b for a, b in v.items()})
# LaTeX table
short = lambda t: (t.replace('-v2', '').replace('-v1', '').replace('-v0', '').replace('medium-replay', 'm-r')
                   .replace('medium-expert', 'm-e').replace('medium', 'm').replace('large', 'l').replace('umaze', 'u')
                   .replace('halfcheetah', 'hc').replace('walker2d', 'w2d').replace('play', 'p').replace('diverse', 'd')
                   .replace('partial', 'p').replace('mixed', 'mx'))
f = lambda v: f"{v[0]:.1f}{{\\scriptsize$\\pm${v[1]:.1f}}}"
L = ["\\begin{tabular}{l rr r rr}", "\\toprule",
     " & \\multicolumn{2}{c}{DV$^\\star$ (published)} & & \\multicolumn{2}{c}{PG (published)} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}",
     "Task & $N{=}50$ & best $N$ & & $\\alpha{=}1$ & best $\\alpha$ \\\\", "\\midrule"]
for r in rows:
    L.append(f"{short(r['task'])} & {f(r['dv50'])} & {f(r['dvbest'])}\\,{{\\scriptsize({r['dvbestN']})}} & & {f(r['pg_a1'])} & {f(r['pg_best'])} \\\\")
avg = lambda key: np.mean([r[key][0] for r in rows])
L += ["\\midrule", f"Mean & {avg('dv50'):.1f} & {avg('dvbest'):.1f} & & {avg('pg_a1'):.1f} & {avg('pg_best'):.1f} \\\\",
      "\\bottomrule", "\\end{tabular}"]
open(OUT + 'tab_budget.tex', 'w').write('\n'.join(L) + '\n')
print('means: dv50 %.1f dvbest %.1f pg_a1 %.1f pg_best %.1f' % (avg('dv50'), avg('dvbest'), avg('pg_a1'), avg('pg_best')))
