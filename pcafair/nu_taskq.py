"""Range selection of the quantizer (pcafair/codec.py calls `design`).

For every component and bit count b = 1..BMAX, the cells are 2^b uniform cells on a range [lo, hi] chosen among quantile candidates of
the training sample: with p in PG the candidate is [s_(k), s_(n-1-k)], k = floor(p n) (p = 0 is the database [min, max]), and the one
with the smallest (hi - lo)^2 4^{-b} / 12 + clipping MSE on the sample is kept. `design` also returns the uniform-cell curves over the
database range (vmm) and Lloyd-Max levels for b <= LMAX (vlm, LM); the codec sets LMAX = 1 and uses only vro, rlo and rhi."""
import numpy as np
from . import ea_lib as E

PG = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]      # quantile candidates (p = 0: the database range)
LMAX = 8                                                         # largest bit count with Lloyd-Max levels (the codec sets 1)


def lloyd(s, cs, cs2, L, init):
    """1-D Lloyd-Max on the sorted sample s (prefix sums cs, cs2 with a leading 0). Returns thresholds (L-1), levels (L), mse."""
    n = len(s); lev = init.copy()
    for _ in range(80):
        t = (lev[:-1] + lev[1:]) / 2; cut = np.concatenate([[0], np.searchsorted(s, t), [n]])
        cnt = np.diff(cut); sm = cs[cut[1:]] - cs[cut[:-1]]
        new = np.where(cnt > 0, sm / np.maximum(cnt, 1), lev)
        if np.max(np.abs(new - lev)) <= 1e-7 * (s[-1] - s[0] + 1e-30): lev = new; break
        lev = new
    t = (lev[:-1] + lev[1:]) / 2; cut = np.concatenate([[0], np.searchsorted(s, t), [n]]); cnt = np.diff(cut)
    sm = cs[cut[1:]] - cs[cut[:-1]]; sq = cs2[cut[1:]] - cs2[cut[:-1]]
    mse = float((sq - np.where(cnt > 0, sm ** 2 / np.maximum(cnt, 1), 0)).sum() / n)
    return t, lev, mse


def design(S, mn, mx, var, N):
    """Per-component curves and quantizer parameters from the training sample S (n_s x d) and the database [min, max]."""
    d = S.shape[1]; n = S.shape[0]; BM = E.BMAX
    vmm = E.curves_dexact(var, (mx - mn).astype(np.float64))
    vro = np.zeros((d, BM + 1)); vro[:, 0] = var; rlo = np.zeros((d, BM + 1)); rhi = np.zeros((d, BM + 1))
    vlm = np.full((d, LMAX + 1), np.nan); vlm[:, 0] = var; LM = {}
    for i in range(d):
        s = np.sort(S[:, i].astype(np.float64)); cs = np.concatenate([[0], np.cumsum(s)]); cs2 = np.concatenate([[0], np.cumsum(s * s)])
        cand = []
        for p in PG:
            k = int(np.floor(p * n)); lo, hi = s[k], s[n - 1 - k]
            clip = ((lo - s[:k]) ** 2).sum() / n + ((s[n - k:] - hi) ** 2).sum() / n if k > 0 else 0.0
            if p == 0.0: lo, hi = mn[i], mx[i]                       # p = 0 means the database range
            cand.append((lo, hi, clip))
        for b in range(1, BM + 1):
            vals = [(hi - lo) ** 2 * 4.0 ** (-b) / 12 + clip for lo, hi, clip in cand]; j = int(np.argmin(vals))
            vro[i, b] = vals[j]; rlo[i, b], rhi[i, b] = cand[j][0], cand[j][1]
        if var[i] <= 1e-12 * var.max(): continue
        lev = np.array([s[n // 4], s[(3 * n) // 4]], np.float64)
        for b in range(1, LMAX + 1):
            L = 2 ** b
            if b > 1: lev = np.sort(np.concatenate([lev - 1e-3 * (s[-1] - s[0]) / L, lev + 1e-3 * (s[-1] - s[0]) / L]))
            else: lev = np.array([np.mean(s[s < np.median(s)]), np.mean(s[s >= np.median(s)])])
            t, lev, mse = lloyd(s, cs, cs2, L, lev); vlm[i, b] = mse; LM[(i, b)] = (t.astype(np.float32), lev.astype(np.float32))
    return vmm, vro, rlo, rhi, vlm, LM
