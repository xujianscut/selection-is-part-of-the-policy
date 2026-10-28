"""Parse PG appendix Tables 11-14 (alpha ablation, 5 seeds). Values are column-major per alpha."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = str(ROOT / 'results') + '/'
OUT = str(ROOT / 'paper_assets') + '/'
(ROOT / 'paper_assets').mkdir(exist_ok=True)
import re, json
txt = open(RES + 'pg_published/pg_paper.txt').read().replace('\n', ' ')
specs = [
    ('Table 11: Ablation study of α in MuJoCo', [0.001, 0.01, 0.1, 1.0, 10.0],
     ['walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2', 'hopper-medium-v2',
      'hopper-medium-replay-v2', 'hopper-medium-expert-v2', 'halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2',
      'halfcheetah-medium-expert-v2']),
    ('Table 12: Ablation study of α in Kitchen', [0.1, 1.0, 10.0], ['kitchen-mixed-v0', 'kitchen-partial-v0']),
    ('Table 13: Ablation study of α in Antmaze', [0.1, 1.0, 5.0, 10.0, 50.0],
     ['antmaze-medium-play-v2', 'antmaze-medium-diverse-v2', 'antmaze-large-play-v2', 'antmaze-large-diverse-v2']),
    ('Table 14: Ablation study of α in Maze2D', [0.001, 0.01, 0.1, 1.0, 10.0],
     ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']),
]
out = {}
for head, alphas, tasks in specs:
    i = txt.index(head)
    j = txt.index(tasks[-1], i) + len(tasks[-1])
    vals = re.findall(r'(\d+\.\d)\s*±\s*(\d+\.\d)', txt[j:j + 3000])[: len(alphas) * len(tasks)]
    assert len(vals) == len(alphas) * len(tasks), head
    for a_i, a in enumerate(alphas):
        for t_i, t in enumerate(tasks):
            out.setdefault(t, {})[str(a)] = [float(x) for x in vals[a_i * len(tasks) + t_i]]
pg3 = json.load(open(RES + 'pg_published/pg_table3.json'))
# the main-table PG number must equal the best alpha in the ablation for every task
for t in out:
    best = max(v[0] for v in out[t].values())
    assert abs(best - pg3[t]['PG'][0]) < 1e-9, (t, best, pg3[t]['PG'][0])
assert out['antmaze-large-play-v2']['50.0'] == [82.4, 2.1] and out['antmaze-large-play-v2']['1.0'] == [74.6, 2.5]
json.dump(out, open(RES + 'pg_published/pg_alpha.json', 'w'), indent=1)
print('ok: PG main-table score == max over alpha for all', len(out), 'tasks')
