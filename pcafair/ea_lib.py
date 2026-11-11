"""Error curves and the exact integer allocation used by the codec (pcafair/codec.py).

`curves_dexact` gives the error variance of a component with b bits over a range r under shared subtractive dither, r^2 4^{-b} / 12
(b = 0: the component's variance). `dp_alloc` minimizes sum_i cost[i, b_i] over integer bit counts with a fixed total by dynamic
programming (the 'exact' allocation rule; the paper's designs use the closed-form slope-1 rule and uniform bits)."""
import numpy as np
BMAX = 24


def curves_dexact(var, rng):
    v = np.zeros((len(var), BMAX + 1)); v[:, 0] = var
    for b in range(1, BMAX + 1): v[:, b] = rng.astype(np.float64) ** 2 * 4.0 ** (-b) / 12
    return v


def dp_alloc(cost, total):
    """Exact integer minimizer of sum_i cost[i, b_i] subject to sum_i b_i = total, b_i in [0, cost.shape[1]-1] (dynamic programming)."""
    d, nb = cost.shape; INF = np.inf
    f = np.full(total + 1, INF); f[0] = 0.0; arg = np.zeros((d, total + 1), np.int8)
    for i in range(d):
        g = np.full(total + 1, INF); a = np.zeros(total + 1, np.int8)
        for b in range(nb):
            if b > total: break
            cand = np.full(total + 1, INF); cand[b:] = f[:total + 1 - b] + cost[i, b]
            m = cand < g; g[m] = cand[m]; a[m] = b
        f = g; arg[i] = a
    bits = np.zeros(d, np.int64); s = total
    for i in range(d - 1, -1, -1):
        bits[i] = arg[i, s]; s -= bits[i]
    assert s == 0
    return bits
