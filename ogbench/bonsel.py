"""Kill test for idea #1: does best-of-N reranking of a flow BC policy by a learned critic
show a true-return inflection in N, and does an ensemble LCB remove it?

Stage 1: train flow BC policy + K-head IQL critic ensemble on an OGBench singletask dataset.
Stage 2: evaluate success rate vs N for several selection rules with common random numbers.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import ogbench

p = argparse.ArgumentParser()
p.add_argument('--env', default='cube-single-play-singletask-v0')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=500_000)
p.add_argument('--batch', type=int, default=256)
p.add_argument('--K', type=int, default=10)
p.add_argument('--expectile', type=float, default=0.9)
p.add_argument('--discount', type=float, default=None)
p.add_argument('--flow_steps', type=int, default=10)
p.add_argument('--episodes', type=int, default=50)
p.add_argument('--Ns', default='1,2,4,8,16,32,64,128,256')
p.add_argument('--rules', default='mean,lcb1,lcb2,min')
p.add_argument('--out', default='runs')
p.add_argument('--eval_only', action='store_true')
p.add_argument('--compile', action='store_true')
p.add_argument('--data', default='data')
p.add_argument('--split', action='store_true', help='heads 0..K/2-1 train on episode half A, rest on half B')
p.add_argument('--qclip', action='store_true', help='clip TD targets to [rmin, rmax]/(1-gamma)')
p.add_argument('--critic', default='iql', choices=['iql', 'emaq'], help='iql: in-sample expectile; emaq: max over BC samples at s\'')
p.add_argument('--emaq_n', type=int, default=8)
args = p.parse_args()
assert args.K >= 2
if args.discount is None:
    args.discount = 0.995 if 'antmaze' in args.env else 0.99
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = 'cuda'
tag = f"{args.env}_s{args.seed}_K{args.K}_e{args.expectile}" + ("_split" if args.split else "") + ("_clip" if args.qclip else "") + ("_emaq" if args.critic == "emaq" else "")
os.makedirs(args.out, exist_ok=True)
ckpt = os.path.join(args.out, tag + '.pt')

env, ds, _ = ogbench.make_env_and_datasets(args.env, dataset_dir=args.data)
D = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in ds.items()}
n, obs_dim = D['observations'].shape
act_dim = D['actions'].shape[1]
assert n > 0 and act_dim == env.action_space.shape[0]
rmin, rmax = D['rewards'].min().item(), D['rewards'].max().item()
vmin, vmax = min(rmin, 0) / (1 - args.discount), max(rmax, 0) / (1 - args.discount)
# episode-level split so adjacent transitions never leak across halves
ep = torch.cumsum(torch.cat([torch.zeros(1, device=dev), D['terminals'][:-1]]), 0).long()
g = torch.Generator(device='cpu'); g.manual_seed(args.seed)
half = (torch.randperm(int(ep.max()) + 1, generator=g) % 2).to(dev)[ep]
idxA, idxB = torch.where(half == 0)[0], torch.where(half == 1)[0]
assert len(idxA) > 0 and len(idxB) > 0
KA = args.K // 2


class ELinear(nn.Module):
    def __init__(self, K, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(K, i, o)); self.b = nn.Parameter(torch.zeros(K, 1, o))
        for k in range(K):
            nn.init.kaiming_uniform_(self.w.data[k].T, a=5 ** 0.5)

    def forward(self, x):  # x: (K, B, i)
        return torch.baddbmm(self.b, x, self.w)


class EMLP(nn.Module):
    """K independent MLPs with LayerNorm, evaluated in one batched call."""
    def __init__(self, K, i, o, h=512, L=4):
        super().__init__()
        dims = [i] + [h] * L
        self.layers = nn.ModuleList([ELinear(K, a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.norms = nn.ModuleList([nn.LayerNorm(h, elementwise_affine=False) for _ in range(L)])
        self.gains = nn.ParameterList([nn.Parameter(torch.ones(K, 1, h)) for _ in range(L)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(K, 1, h)) for _ in range(L)])
        self.head = ELinear(K, h, o)
        self.K = K

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.K, -1, -1)
        for l, nm, ga, be in zip(self.layers, self.norms, self.gains, self.biases):
            x = torch.nn.functional.gelu(nm(l(x)) * ga + be)
        return self.head(x)


policy = EMLP(1, obs_dim + act_dim + 1, act_dim).to(dev)
qnet = EMLP(args.K, obs_dim + act_dim, 1).to(dev)
qtarg = EMLP(args.K, obs_dim + act_dim, 1).to(dev); qtarg.load_state_dict(qnet.state_dict())
vnet = EMLP(args.K, obs_dim, 1).to(dev)


def vel(s, a, t):
    return policy(torch.cat([s, a, t], -1))[0]


@torch.no_grad()
def sample_actions(s, N, gen, scale=1.0):
    """s: (B, obs) -> (B, N, act) via Euler integration of the flow from N(0, I)."""
    B = s.shape[0]
    ss = s.repeat_interleave(N, 0)
    a = scale * torch.randn(B * N, act_dim, device=dev, generator=gen)
    h = 1.0 / args.flow_steps
    for i in range(args.flow_steps):
        t = torch.full((B * N, 1), i * h, device=dev)
        a = a + h * vel(ss, a, t)
    return a.clamp(-1, 1).view(B, N, act_dim)


def losses():
    # flow BC
    idx = torch.randint(0, n, (args.batch,), device=dev)
    s, a = D['observations'][idx], D['actions'][idx]
    x0 = torch.randn_like(a); t = torch.rand(args.batch, 1, device=dev)
    xt = (1 - t) * x0 + t * a
    lpi = ((vel(s, xt, t) - (a - x0)) ** 2).mean()
    # IQL ensemble: each head sees its own minibatch
    if args.split:
        iA = idxA[torch.randint(0, len(idxA), (KA, args.batch), device=dev)]
        iB = idxB[torch.randint(0, len(idxB), (args.K - KA, args.batch), device=dev)]
        idx = torch.cat([iA, iB], 0)
    else:
        idx = torch.randint(0, n, (args.K, args.batch), device=dev)
    s, a, r = D['observations'][idx], D['actions'][idx], D['rewards'][idx]
    s2, m = D['next_observations'][idx], D['masks'][idx]
    with torch.no_grad():
        qt = qtarg(torch.cat([s, a], -1)).squeeze(-1)
        if args.critic == 'emaq':
            # EMaQ-style backup: each head maximizes its own target over n behavior samples at s'
            n_ = args.emaq_n; Kh, Bh = s2.shape[0], s2.shape[1]
            a2 = sample_actions(s2.reshape(Kh * Bh, -1), n_, None)            # (K*B, n, act)
            s2r = s2.reshape(Kh * Bh, 1, -1).expand(-1, n_, -1)
            q2 = qtarg(torch.cat([s2r, a2], -1).reshape(Kh, Bh * n_, -1)).reshape(Kh, Bh, n_)
            y = r + args.discount * m * q2.max(-1).values
        else:
            y = r + args.discount * m * vnet(s2).squeeze(-1)
        if args.qclip:
            y = y.clamp(vmin, vmax)
    u = qt - vnet(s).squeeze(-1)
    w = torch.where(u > 0, args.expectile, 1 - args.expectile)
    lv = (w * u ** 2).mean() if args.critic == 'iql' else (vnet(s).squeeze(-1) * 0).mean()
    lq = ((qnet(torch.cat([s, a], -1)).squeeze(-1) - y) ** 2).mean()
    # parameter sets are disjoint, so one backward on the sum = three separate updates
    return lpi + lv + lq, lpi.detach(), lv.detach(), lq.detach()


def train():
    params = list(policy.parameters()) + list(qnet.parameters()) + list(vnet.parameters())
    opt = torch.optim.Adam(params, 3e-4, fused=True)
    loss_fn = torch.compile(losses) if args.compile else losses
    qp, tp = list(qnet.parameters()), list(qtarg.parameters())
    t0 = time.time()
    for it in range(args.steps):
        tot, lpi, lv, lq = loss_fn()
        opt.zero_grad(set_to_none=True); tot.backward(); opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(tp, qp, 0.005)
        if it % 50_000 == 0:
            print(f"it {it} lpi {lpi.item():.4f} lv {lv.item():.4f} lq {lq.item():.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    torch.save({'policy': policy.state_dict(), 'q': qnet.state_dict(), 'args': vars(args)}, ckpt)


def select(q, rule, gen):
    """q: (K, B, N) -> chosen index (B,).
    mean/lcbR/min: argmax of a pessimistic score; xfit: argmax of heads A only (heads B held out);
    softT: sample prop. to exp(z/T), z = per-state standardized mean Q; mean_t05: mean, noise scale 0.5."""
    mu, sd = q.mean(0), q.std(0)
    if rule in ('mean', 'mean_t05'):
        score = mu
    elif rule.startswith('lcb'):
        score = mu - float(rule[3:]) * sd
    elif rule == 'min':
        score = q.min(0).values
    elif rule == 'xfit':
        score = q[:KA].mean(0)
    elif rule.startswith('soft'):
        z = (mu - mu.mean(-1, keepdim=True)) / (mu.std(-1, keepdim=True) + 1e-6)
        return torch.multinomial(torch.softmax(z / float(rule[4:]), -1), 1, generator=gen).squeeze(-1)
    else:
        raise ValueError(rule)
    return score.argmax(-1)


@torch.no_grad()
def offline_diag(Ns, M=4096):
    """For dataset states: Q of the BoN-selected action under (a) the selecting heads, (b) held-out heads."""
    gen = torch.Generator(device=dev); gen.manual_seed(7)
    src = idxB if args.split else torch.arange(n, device=dev)
    s = D['observations'][src[torch.randint(0, len(src), (M,), device=dev, generator=gen)]]
    rows = []
    for N in Ns:
        qcs = []
        for s_ in s.split(max(1, 8192 // N)):  # chunk states to bound memory at large N
            m_ = s_.shape[0]
            A = sample_actions(s_, N, gen)
            sa = torch.cat([s_.unsqueeze(1).expand(-1, N, -1), A], -1).view(m_ * N, -1)
            q = qnet(sa).view(args.K, m_, N)
            j = q[:KA].mean(0).argmax(-1)
            qcs.append(q[:, torch.arange(m_), j])
        qc = torch.cat(qcs, 1)
        assert qc.shape == (args.K, M)
        rows.append(dict(N=N, q_select_heads=qc[:KA].mean().item(), q_heldout_heads=qc[KA:].mean().item()))
        print('offline', json.dumps(rows[-1]), flush=True)
    return rows


@torch.no_grad()
def evaluate():
    Ns = [int(x) for x in args.Ns.split(',')]
    rules = args.rules.split(',')
    with open(os.path.join(args.out, tag + '_offline.json'), 'w') as f:
        json.dump(offline_diag(Ns), f, indent=1)
    E = args.episodes
    envs = [ogbench.make_env_and_datasets(args.env, env_only=True) for _ in range(E)]
    results = []
    for N in Ns:
        for rule in (rules if N > 1 else ['mean']):
            gen = torch.Generator(device=dev); gen.manual_seed(1000 + args.seed)
            obs = np.stack([e.reset(seed=10_000 + j)[0] for j, e in enumerate(envs)])
            done = np.zeros(E, bool); succ = np.zeros(E); ret = np.zeros(E)
            qsel, qsd, qB, steps = 0.0, 0.0, 0.0, 0
            while not done.all():
                act_idx = np.where(~done)[0]
                s = torch.as_tensor(obs[act_idx], dtype=torch.float32, device=dev)
                A = sample_actions(s, N, gen, 0.5 if rule == 'mean_t05' else 1.0)  # (B, N, act)
                B = A.shape[0]
                sa = torch.cat([s.unsqueeze(1).expand(-1, N, -1), A], -1).view(B * N, -1)
                q = qnet(sa).view(args.K, B, N)
                j = select(q, rule, gen)
                a = A[torch.arange(B), j].cpu().numpy()
                qc = q[:, torch.arange(B), j]
                qB += qc[KA:].mean().item() * B
                qsel += qc.mean().item() * B; qsd += qc.std(0).mean().item() * B; steps += B
                for k, ei in enumerate(act_idx):
                    o, rew, term, trunc, info = envs[ei].step(a[k])
                    obs[ei] = o; ret[ei] += rew
                    if term or trunc:
                        done[ei] = True; succ[ei] = float(info.get('success', 0.0))
            row = dict(env=args.env, seed=args.seed, N=N, rule=rule, success=succ.mean(),
                       success_se=succ.std(ddof=1) / np.sqrt(E), ret=ret.mean(),
                       q_selected=qsel / steps, q_sd_selected=qsd / steps, q_heldout=qB / steps)
            print(json.dumps(row), flush=True)
            results.append(row)
    with open(os.path.join(args.out, tag + f'_eval_{args.rules.replace(",", "-")}.json'), 'w') as f:
        json.dump(results, f, indent=1)


if not args.eval_only:
    train()
else:
    st = torch.load(ckpt); policy.load_state_dict(st['policy']); qnet.load_state_dict(st['q'])
evaluate()
