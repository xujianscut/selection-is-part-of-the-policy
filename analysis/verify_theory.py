"""Monte Carlo check of Theorem 1, Proposition 2 and Proposition 4 on the closed-form example
m(u) = u - c u^K (Example 1 in the paper), with uniform critic scores S = U."""
import numpy as np

rng = np.random.default_rng(0)
c, K = 0.9, 20
m = lambda u: u - c * u ** K
M = 400_000
print("argmax selection: Monte Carlo vs. Theorem 1, and the critic's own estimate (Prop. 2)")
for N in [1, 2, 5, 10, 50, 200]:
    U = rng.random((M // N, N))
    q = m(U) + rng.normal(0, 0.3, U.shape)          # true values with calibration curve m
    sel = q[np.arange(len(U)), U.argmax(1)]
    exact = N / (N + 1) - c * N / (N + K)
    assert abs(sel.mean() - exact) < 0.01, (N, sel.mean(), exact)
    print(f"  N={N:4d}  MC={sel.mean():.4f}  Thm1={exact:.4f}  critic self-estimate={U.max(1).mean():.4f}")
print("peak N* =", (K - np.sqrt(c * K)) / (np.sqrt(c * K) - 1))
T = 1.0
u = np.linspace(0, 1, 200_001); z = (u - 0.5) * np.sqrt(12); w = np.exp(z / T)
lim = (m(u) * w).sum() / w.sum()
print(f"soft selection, T={T}: limit (Prop. 4) = {lim:.4f}")
for N in [50, 500, 5000]:
    R = 4000 if N <= 500 else 400
    U = rng.random((R, N)); q = m(U) + rng.normal(0, 0.3, U.shape)
    zz = (U - U.mean(1, keepdims=True)) / U.std(1, keepdims=True)
    W = np.exp(zz / T); W /= W.sum(1, keepdims=True)
    v = (W * q).sum(1).mean()
    assert abs(v - lim) < 0.01, (N, v, lim)
    print(f"  N={N:5d}  MC={v:.4f}")
