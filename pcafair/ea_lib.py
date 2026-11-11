"""exact allocation for the fixed-range uniform scalar quantizer (shared library).
Error model (shared subtractive dither, no overload): component i with b >= 1 bits over the database range r_i has error variance
exactly r_i^2 4^-b / 12 = c0_i sigma_i^2 4^-b with c0_i = r_i^2 / (12 sigma_i^2); with b = 0 the component is dropped (error variance
sigma_i^2). Rule 'dexact' minimizes sum_i w_i v_i(b_i) (w_i = sigma_i^2, A2) over integer b_i in [0, BMAX], sum b_i = B d, by a
Lagrange multiplier on each component's lower convex hull, then fills the remainder by marginal return. Rule 'op' does the same with
v_i(b) measured on a database sample under deterministic rounding."""
import numpy as np
BMAX = 24


def lagrange_alloc(v, w, total):
    """v: (d, BMAX+1) error variance per bit count, w: (d) weights; integer allocation with sum = total."""
    d = len(w); ar = np.arange(d); steps = np.arange(v.shape[1])[None, :]
    pick = lambda l: (w[:, None] * v + l * steps).argmin(1)
    lo, hi = 0.0, float((w[:, None] * v).max()) * 10 + 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid).sum() > total: lo = mid
        else: hi = mid
    b = pick(hi); top = v.shape[1] - 1
    while b.sum() < total:
        g = np.where(b < top, w * (v[ar, b] - v[ar, np.minimum(b + 1, top)]), -np.inf); b[np.argmax(g)] += 1
    while b.sum() > total:
        l = np.where(b > 0, w * (v[ar, np.maximum(b - 1, 0)] - v[ar, b]), np.inf); b[np.argmin(l)] -= 1
    return b.astype(np.int64)


def curves_dexact(var, rng):
    v = np.zeros((len(var), BMAX + 1)); v[:, 0] = var
    for b in range(1, BMAX + 1): v[:, b] = rng.astype(np.float64) ** 2 * 4.0 ** (-b) / 12
    return v


def quantize(X, bits, mn, mx, z=None):
    """The paper's quantizer, vectorized: cells (mx - mn) / 2^b from mn, centre reconstruction, b = 0 -> 0.
    z: optional shared subtractive dither (one value per component, in [-W/2, W/2)), subtracted after reconstruction."""
    bits = np.asarray(bits); act = (bits > 0) & (mx > mn)
    L = np.where(act, 2.0 ** bits, 1.0).astype(np.float32); W = np.where(act, (mx - mn) / L, 1.0).astype(np.float32)
    Y = X if z is None else X + z[None, :]
    k = np.clip(np.floor((Y - mn[None, :]) / W[None, :]), 0, L[None, :] - 1)
    Xq = mn[None, :] + (k + 0.5) * W[None, :]
    if z is not None: Xq = Xq - z[None, :]
    Xq[:, ~act] = 0.0
    return Xq.astype(np.float32)


def shared_dither(bits, mn, mx, seed):
    bits = np.asarray(bits); act = (bits > 0) & (mx > mn); W = np.where(act, (mx - mn) / np.where(act, 2.0 ** bits, 1.0), 0.0)
    return (np.random.default_rng(seed).uniform(-0.5, 0.5, size=len(bits)) * W).astype(np.float32)


def edge_excess(X, mn, mx, bmax=BMAX):
    """Exact overload terms of the dithered quantizer (cells (mx - mn)/2^b from mn, index clipped to the range):
    for a value at distance t from the nearer range end and half-cell h = (mx - mn)/2^(b+1),
      E_zeta[e^2 | x] = Delta^2/12 + (h - t)_+^2, E_zeta[e | x] = +(h - t)_+ at the lower end, -(h - t)_+ at the upper end.
    Returns eta (d, bmax+1): the mean over the rows of X of (h_b - t)_+^2 (eta[:, 0] = 0)."""
    X = np.asarray(X, np.float64); t = np.minimum(X - mn[None, :], mx[None, :] - X); r = (mx - mn).astype(np.float64)
    eta = np.zeros((X.shape[1], bmax + 1))
    for b in range(1, bmax + 1):
        h = r / 2.0 ** (b + 1); eta[:, b] = (np.maximum(h[None, :] - t, 0.0) ** 2).mean(0)
    return eta


def curves_exact(var, rng, eta):
    """Exact per-component error variance of the dithered quantizer: D(0) = var, D(b) = (rng/2^b)^2/12 + eta(b)."""
    return curves_dexact(var, rng) + eta


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
