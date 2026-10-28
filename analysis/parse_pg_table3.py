"""Parse PG (NeurIPS 2025) appendix Table 3: DV* at N in {1,5,10,50,500}, original DV (N=50), PG.
Columns are printed column-major in the PDF text: for each column, one value per task of the block."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = str(ROOT / 'results') + '/'
OUT = str(ROOT / 'paper_assets') + '/'
(ROOT / 'paper_assets').mkdir(exist_ok=True)
import re, json
txt = open(RES + 'pg_published/pg_paper.txt').read()
i = txt.index('Table 3: Normalized scores on the D4RL benchmark')
seg = txt[i:i + 6000].replace('\n', ' ')
COLS = ['DV*_N1', 'DV*_N5', 'DV*_N10', 'DV*_N50', 'DV*_N500', 'DV_N50', 'PG']
blocks = {
    'mujoco': ['walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2', 'hopper-medium-v2',
               'hopper-medium-replay-v2', 'hopper-medium-expert-v2', 'halfcheetah-medium-v2',
               'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2'],
    'kitchen': ['kitchen-mixed-v0', 'kitchen-partial-v0'],
    'antmaze': ['antmaze-medium-play-v2', 'antmaze-medium-diverse-v2', 'antmaze-large-play-v2', 'antmaze-large-diverse-v2'],
    'maze2d': ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1'],
}
out = {}
for name, tasks in blocks.items():
    j = seg.index(tasks[-1]) + len(tasks[-1])
    vals = re.findall(r'(\d+\.\d)\s*±\s*(\d+\.\d)', seg[j:])[: len(tasks) * len(COLS)]
    assert len(vals) == len(tasks) * len(COLS), (name, len(vals))
    for c, col in enumerate(COLS):
        for t, task in enumerate(tasks):
            m, s = vals[c * len(tasks) + t]
            out.setdefault(task, {})[col] = [float(m), float(s)]
# spot checks against values read by hand from the PDF
assert out['kitchen-partial-v0']['DV*_N10'] == [80.6, 1.8]
assert out['kitchen-partial-v0']['DV_N50'] == [94.0, 0.3]
assert out['kitchen-partial-v0']['PG'] == [88.0, 3.1]
assert out['antmaze-medium-play-v2']['DV*_N500'] == [62.2, 3.2]
assert out['antmaze-large-play-v2']['PG'] == [82.4, 2.1]
assert out['maze2d-large-v1']['DV*_N500'] == [201.8, 5.5]
assert out['maze2d-medium-v1']['PG'] == [159.5, 0.8]
assert out['hopper-medium-v2']['PG'] == [97.5, 0.6]
json.dump(out, open(RES + 'pg_published/pg_table3.json', 'w'), indent=1)
for t, d in out.items():
    print(f"{t:32s}", ' '.join(f"{c}={d[c][0]}" for c in COLS))
