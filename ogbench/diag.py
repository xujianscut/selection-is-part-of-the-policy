"""Diagnostics linking the single-step analysis to the multi-step experiments (Section 4).

1. Empirical calibration curve. From states visited by a continuation policy pi_c (argmax over N_c
   candidates), draw N_b candidates, rank them by the mean critic, and follow the candidate at quantile
   rank u for the next k steps; then continue with pi_c until the episode ends. The success rate as a
   function of u estimates the k-step calibration curve m_k(u); the critic's mean score of the chosen
   candidate gives its predicted counterpart. Branches start from exactly reproduced states (same reset
   seed and replayed action prefix; MuJoCo is deterministic), which is asserted.
2. State shift. For several N and rules, the distance of visited states to their nearest dataset state.

Uses a checkpoint written by bonsel.py (IQL critic, --split --qclip).
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import ogbench

p = argparse.ArgumentParser()
p.add_argument('--env', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--ckpt', required=True)
p.add_argument('--data', required=True)
p.add_argument('--out', required=True)
p.add_argument('--Nc', type=int, default=4, help='candidates of the continuation policy')
p.add_argument('--Nb', type=int, default=256, help='candidates ranked in a branch')
p.add_argument('--k', type=int, default=10, help='steps that follow the quantile-u candidate')
p.add_argument('--quantiles', default='0.5,0.75,0.9,0.97,0.99,1.0')
p.add_argument('--episodes', type=int, default=20)
p.add_argument('--anchor_every', type=int, default=20)
p.add_argument('--max_anchors', type=int, default=8)
p.add_argument('--shift_Ns', default='1,4,16,64,256')
p.add_argument('--shift_episodes', type=int, default=20)
p.add_argument('--flow_steps', type=int, default=10)
args = p.parse_args()
dev = 'cuda'
torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
tag = f"diag_{args.env}_s{args.seed}"

_, ds, _ = ogbench.make_env_and_datasets(args.env, dataset_dir=args.data)
obs_all = torch.as_tensor(ds['observations'], dtype=torch.float32, device=dev)
obs_dim, act_dim = obs_all.shape[1], ds['actions'].shape[1]
assert len(obs_all) > 0


class ELinear(nn.Module):
    def __init__(self, K, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(K, i, o)); self.b = nn.Parameter(torch.zeros(K, 1, o))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class EMLP(nn.Module):
    def __init__(self, K, i, o, h=512, L=4):
        super().__init__()
        dims = [i] + [h] * L
        self.layers = nn.ModuleList([ELinear(K, a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.norms = nn.ModuleList([nn.LayerNorm(h, elementwise_affine=False) for _ in range(L)])
        self.gains = nn.ParameterList([nn.Parameter(torch.ones(K, 1, h)) for _ in range(L)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(K, 1, h)) for _ in range(L)])
        self.head = ELinear(K, h, o); self.K = K

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.K, -1, -1)
        for l, nm, ga, be in zip(self.layers, self.norms, self.gains, self.biases):
            x = torch.nn.functional.gelu(nm(l(x)) * ga + be)
        return self.head(x)


st = torch.load(args.ckpt, map_location=dev)
K = st['args']['K']
policy = EMLP(1, obs_dim + act_dim + 1, act_dim).to(dev); policy.load_state_dict(st['policy'])
qnet = EMLP(K, obs_dim + act_dim, 1).to(dev); qnet.load_state_dict(st['q'])
gen = torch.Generator(device=dev); gen.manual_seed(1234 + args.seed)


@torch.no_grad()
def candidates(s, N):
    """s: (B, obs) -> actions (B, N, act), mean critic (B, N)."""
    B = s.shape[0]
    ss = s.repeat_interleave(N, 0)
    a = torch.randn(B * N, act_dim, device=dev, generator=gen)
    h = 1.0 / args.flow_steps
    for i in range(args.flow_steps):
        a = a + h * policy(torch.cat([ss, a, torch.full((B * N, 1), i * h, device=dev)], -1))[0]
    a = a.clamp(-1, 1)
    q = qnet(torch.cat([ss, a], -1)).view(K, B, N).mean(0)
    return a.view(B, N, act_dim), q


def make_env():
    return ogbench.make_env_and_datasets(args.env, env_only=True)


# ---------- 1. calibration curve via branching ----------
qs = [float(x) for x in args.quantiles.split(',')]
rows = []
for e in range(args.episodes):
    env = make_env(); seed = 30000 + e
    o, _ = env.reset(seed=seed); acts, obs_hist, done, t = [], [o], False, 0
    while not done:
        A, Q = candidates(torch.as_tensor(o[None], dtype=torch.float32, device=dev), args.Nc)
        a = A[0, Q[0].argmax()].cpu().numpy()
        o, r, term, trunc, info = env.step(a); acts.append(a); obs_hist.append(o); t += 1
        done = term or trunc
    T = len(acts)
    anchors = [t for t in range(args.anchor_every, T - args.k, args.anchor_every)][: args.max_anchors]
    # build branches: (anchor, quantile)
    br = []
    for ta in anchors:
        for u in qs:
            env_b = make_env(); ob, _ = env_b.reset(seed=seed)
            for a in acts[:ta]:
                ob, *_ = env_b.step(a)
            assert np.allclose(ob, obs_hist[ta], atol=1e-6), 'branch state does not match the recorded state'
            br.append(dict(env=env_b, obs=ob, ta=ta, u=u, step=0, done=False, succ=0.0, qpred=None))
    while not all(b['done'] for b in br):
        live = [b for b in br if not b['done']]
        branch = [b for b in live if b['step'] < args.k]; cont = [b for b in live if b['step'] >= args.k]
        acts_live = {}
        if branch:
            A, Q = candidates(torch.as_tensor(np.stack([b['obs'] for b in branch]), dtype=torch.float32, device=dev), args.Nb)
            order = Q.argsort(-1)
            for j, b in enumerate(branch):
                r_ = int(round(b['u'] * (args.Nb - 1)))
                idx = order[j, r_]
                if b['step'] == 0:
                    b['qpred'] = Q[j, idx].item()
                acts_live[id(b)] = A[j, idx].cpu().numpy()
        if cont:
            A, Q = candidates(torch.as_tensor(np.stack([b['obs'] for b in cont]), dtype=torch.float32, device=dev), args.Nc)
            for j, b in enumerate(cont):
                acts_live[id(b)] = A[j, Q[j].argmax()].cpu().numpy()
        for b in live:
            ob, r, term, trunc, info = b['env'].step(acts_live[id(b)])
            b['obs'] = ob; b['step'] += 1
            if term or trunc:
                b['done'] = True; b['succ'] = float(info.get('success', 0.0))
    for b in br:
        rows.append(dict(episode=e, anchor=b['ta'], u=b['u'], success=b['succ'], qpred=b['qpred']))
    print(f"episode {e}: T={T} anchors={len(anchors)} branches={len(br)}", flush=True)
calib = {}
for u in qs:
    rr = [r for r in rows if r['u'] == u]
    sv = np.array([r['success'] for r in rr]); qv = np.array([r['qpred'] for r in rr])
    calib[str(u)] = dict(success=sv.mean(), success_se=sv.std(ddof=1) / np.sqrt(len(sv)), qpred=qv.mean(), n=len(sv))
    print('calib', u, calib[str(u)], flush=True)

# ---------- 2. state shift ----------
mu, sd = obs_all.mean(0), obs_all.std(0) + 1e-6
ref = ((obs_all - mu) / sd)[torch.randperm(len(obs_all), generator=torch.Generator().manual_seed(0))[:100_000].to(dev)]


@torch.no_grad()
def nn_dist(X):
    X = (torch.as_tensor(np.stack(X), dtype=torch.float32, device=dev) - mu) / sd
    out = []
    for xb in X.split(1024):
        out.append(torch.cdist(xb, ref).min(1).values)
    return torch.cat(out).cpu().numpy()


shift = {}
for rule in ['argmax', 'soft1']:
    for N in [int(x) for x in args.shift_Ns.split(',')]:
        if N == 1 and rule != 'argmax':
            continue
        visited, succ = [], []
        for e in range(args.shift_episodes):
            env = make_env(); o, _ = env.reset(seed=40000 + e); done = False
            while not done:
                A, Q = candidates(torch.as_tensor(o[None], dtype=torch.float32, device=dev), N)
                if rule == 'argmax':
                    j = int(Q[0].argmax())
                else:
                    z = (Q[0] - Q[0].mean()) / (Q[0].std() + 1e-6) if N > 1 else Q[0] * 0
                    j = int(torch.multinomial(torch.softmax(z, -1), 1, generator=gen))
                o, r, term, trunc, info = env.step(A[0, j].cpu().numpy()); visited.append(o)
                done = term or trunc
            succ.append(float(info.get('success', 0.0)))
        d = nn_dist(visited)
        shift[f"{rule}_{N}"] = dict(nn_mean=float(d.mean()), nn_p90=float(np.percentile(d, 90)), success=float(np.mean(succ)), steps=len(d))
        print('shift', rule, N, shift[f"{rule}_{N}"], flush=True)

json.dump(dict(env=args.env, seed=args.seed, Nc=args.Nc, Nb=args.Nb, k=args.k, calib=calib, rows=rows, shift=shift),
          open(os.path.join(args.out, tag + '.json'), 'w'), indent=1, default=float)
