#!/usr/bin/env python3
"""The common codec of both designs. Every experiment reads this module.

Design (both pairs):
  normalization u = x / ||x|| (x = centred database vector in the prepared PCA coordinates; ||x|| is the norm the L2 estimate stores)
  basis 'pca': eigenvectors of Cov(u) on the training sample (descending); 'rot': haar_fullscan(d, 0); 'orig': the original
                 coordinates (raw rows centred by the raw mean, constant coordinates removed)
  cells range selection: per component and bit count b, uniform 2^b cells on [lo, hi], where (lo, hi) is the clip at
                 training-sample quantiles p in PG (p = 0 -> the database [min, max]) that minimizes (hi-lo)^2 4^-b / 12 + clipMSE
  reconstruction 'det': deterministic rounding with gain correction: x_hat = a_i + w_i * k, a_i = mu_i + (lo + W/2 - m_i)/g_i,
                 w_i = W/g_i (Bussgang gain 1 on the training sample; linear in the code k);
                 above 8 bits the midpoint (a_i = lo + W/2, w_i = W).
                 'dither': shared subtractive dither (one z_i ~ U(-W/2, W/2) per component, seed), x_hat = lo + (k + 1/2) W - z_i
                 'unified': components with >= 2 bits are dithered (seed) and then gain
                 corrected (affine per component, Bussgang gain 1 on the training sample under the same draw); 1-bit components use
                 'det' (sign of the clipped range's midpoint split + gain correction)
  allocation 'exact': dynamic program on the measured MSE curves of the 'det' reconstruction (b <= 8; above: the model), weights =
                 variance of the coordinate (A2); 'alpha1': Eq. (2) slope 1 on the variances; 'uniform': B bits everywhere
  score ||x||^2 - 2 ||x|| <q, u_hat> (norm=True) or ||x||^2 - 2 <q, x_hat> (norm=False, ablation)
Training sample: database rows, numpy default_rng(7), min(200k, 1.2e8/d) rows.
Usage: codec = FairCodec(ds, basis, norm=True).fit(); bits = codec.alloc(B, 'exact'); pl = codec.plan(bits)
       z, gd = codec.unified_draw(bits, pl, seed=0); Yh = codec.decode_mode(codec.coords(X, R), pl, 'unified', z, gd)
       codec.save(path); codec = FairCodec.load(path) (the training sample is rebuilt on demand for the gain fits)"""
import os, sys, importlib.util
import numpy as np
from. import ea_lib as E  # noqa: E402
from. import nu_taskq as NU  # noqa: E402
from.config import PREPARED, RAW as RAW_DIR  # noqa: E402
from.engine import haar_fullscan, compute_allocation, integerize_budget  # noqa: E402
PREP = str(PREPARED); BM = E.BMAX; LM8 = 8


class FairCodec:
    def __init__(self, ds, basis='pca', norm=True, seed=7, cells='range', keep_sample=False, curve='det', weight='none', zero='zero', gains='draw'):
        self.ds, self.basis, self.norm, self.seed, self.cells, self.keep_sample = ds, basis, norm, seed, cells, keep_sample
        # step 0 options: curve 'det'|'unified' (DP curves of deterministic rounding or of the
        # unified quantizer), weight 'none'|'r2' (norm weighting of gains, curves and DP weights), zero 'zero'|'mean'|'cstar' (value of
        # components without bits), gains 'draw'|'fixed' (gain correction fitted for the dither draw, or its expectation over draws)
        self.curve_mode, self.weight, self.zero, self.gains = curve, weight, zero, gains
        self.raw = None
        if basis == 'orig':                                       # original coordinates (Figure 1): raw rows centred by the raw mean
            RAW = str(RAW_DIR); rp = f'{RAW}/{ds}/{ds}_base.fvecs'
            if not os.path.exists(rp): rp = f'{RAW}/{ds}/{ds}_base_1M.fvecs'
            dr = int(np.fromfile(rp, np.int32, 1)[0]); N = np.load(f'{PREP}/{ds}-db.npy', mmap_mode='r').shape[0]
            self.raw = np.memmap(rp, np.float32, 'r', shape=(os.path.getsize(rp) // (4 * (dr + 1)), dr + 1))[:N, 1:]
            self.rawq = f'{RAW}/{ds}/{ds}_query.fvecs'

    def rows(self, r0, r1):
        """Source rows of the codec: prepared PCA coordinates, or raw rows centred (active dims) for basis 'orig'."""
        if self.raw is None:
            return np.asarray(np.load(f'{PREP}/{self.ds}-db.npy', mmap_mode='r')[r0:r1]).astype(np.float64)
        return (np.asarray(self.raw[r0:r1], np.float64) - self.mu_raw)[:, self.act_raw]

    def queries(self, nq=None):
        if self.raw is None:
            Q = np.load(f'{PREP}/{self.ds}-query.npy').astype(np.float64)
        else:
            from .engine import read_fvecs
            Q = (read_fvecs(self.rawq).astype(np.float64) - self.mu_raw)[:, self.act_raw]
        return Q if nq is None else Q[:nq]

    # ---------- fitting ----------
    def fit(self, verbose=False):
        db = np.load(f'{PREP}/{self.ds}-db.npy', mmap_mode='r'); N, d = db.shape; CH = min(100_000, 40_000_000 // d)
        if self.raw is not None:                                  # raw mean and active (non-constant) dims, as fig1_l2_v2.py
            dr = self.raw.shape[1]; s1 = np.zeros(dr)
            for r0 in range(0, N, CH): s1 += np.asarray(self.raw[r0:r0 + CH], np.float64).sum(0)
            self.mu_raw = s1 / N; s2 = np.zeros(dr)
            for r0 in range(0, N, CH): s2 += ((np.asarray(self.raw[r0:r0 + CH], np.float64) - self.mu_raw) ** 2).sum(0)
            v = s2 / N; self.act_raw = v > v.max() * 1e-10; d = int(self.act_raw.sum()); CH = min(100_000, 40_000_000 // d)
        self.N, self.d = N, d; R = np.empty(N)
        for r0 in range(0, N, CH):
            X = self.rows(r0, r0 + CH); R[r0:r0 + CH] = np.sqrt((X ** 2).sum(1))
        self.R = R
        TS = int(min(200_000, 1.2e8 / d)); ti = np.sort(np.random.default_rng(self.seed).choice(N, min(TS, N), replace=False))
        U0 = np.concatenate([self.rows(int(t), int(t) + 1) for t in ti]) if self.raw is not None else np.asarray(db[ti]).astype(np.float64)
        if self.norm: U0 = U0 / np.maximum(R[ti], 1e-30)[:, None]
        if self.basis == 'pca' and self.norm:
            ev, V = np.linalg.eigh(np.cov(U0, rowvar=False)); M = np.ascontiguousarray(V[:, ::-1])
        elif self.basis == 'pca':
            M = None                                              # prepared coordinates are already the PCA of x
        elif self.basis == 'rot':
            M = haar_fullscan(d, 0).astype(np.float64)
        else:
            M = None
        self.M = M
        mn = np.full(d, np.inf); mx = np.full(d, -np.inf); s1 = np.zeros(d); s2 = np.zeros(d)
        for r0 in range(0, N, CH):
            Y = self.coords(self.rows(r0, r0 + CH), R[r0:r0 + CH])
            mn = np.minimum(mn, Y.min(0)); mx = np.maximum(mx, Y.max(0)); s1 += Y.sum(0); s2 += (Y ** 2).sum(0)
        self.mn, self.mx = mn, mx; self.var = np.maximum(s2 / N - (s1 / N) ** 2, 1e-30)
        Str = (U0 @ M if M is not None else U0).astype(np.float32)
        NU.LMAX = 1                                               # Lloyd-Max curves are not needed here (keep design light)
        vmm, vro, rlo, rhi, vlm, LM = NU.design(Str, mn, mx, self.var, N)
        if self.cells == 'minmax':                                # previous design: database [min, max] at every bit count
            rlo = np.repeat(mn[:, None], BM + 1, axis=1); rhi = np.repeat(mx[:, None], BM + 1, axis=1)
            vro = np.zeros((d, BM + 1)); vro[:, 0] = self.var
            for b in range(1, BM + 1): vro[:, b] = (mx - mn) ** 2 * 4.0 ** (-b) / 12
        self.rlo, self.rhi, self.vro = rlo, rhi, vro
        if self.keep_sample: self.Str = Str
        self.ti, self.Rt = ti, R[ti]
        # per component and b <= 8: gain-corrected affine reconstruction and its MSE on the training sample
        self.a = np.zeros((d, BM + 1)); self.w = np.zeros((d, BM + 1)); self.curve = np.zeros((d, BM + 1)); self.curve[:, 0] = self.var
        for i in range(d):
            s = Str[:, i].astype(np.float64); vx = s.var() + 1e-300; mu = s.mean()
            for b in range(1, BM + 1):
                lo, hi = rlo[i, b], rhi[i, b]
                if hi <= lo: hi = lo + 1e-12
                L = 2 ** b; W = (hi - lo) / L
                if b <= LM8:
                    kk = np.clip(np.floor((s - lo) / W), 0, L - 1); mid = lo + (kk + 0.5) * W
                    g = np.mean((mid - mid.mean()) * (s - mu)) / vx; g = g if abs(g) > 1e-9 else 1.0
                    self.a[i, b] = mu + (lo + 0.5 * W - mid.mean()) / g; self.w[i, b] = W / g
                    self.curve[i, b] = np.mean((mu + (mid - mid.mean()) / g - s) ** 2)
                else:
                    self.a[i, b] = lo + 0.5 * W; self.w[i, b] = W; self.curve[i, b] = vro[i, b]
        self.curve_det = self.curve.copy(); self._cache = {}; self._a_none = self.a.copy(); self._w_none = self.w.copy()
        if (self.curve_mode, self.weight, self.zero, self.gains) != ('det', 'none', 'zero', 'draw'):
            self.ensure_sample(); self._apply_options()
        if self.gains == 'closed': self.closed_gains()
        if verbose: print(f'  [{self.ds} {self.basis} norm={self.norm}] codec fitted', flush=True)
        return self

    # ---------- step 0: options of the unified quantizer ----------
    @staticmethod
    def _gain_fit(ut, y, Rw=None):
        """Affine y_hat = A + ut / g per column. Rw None: Bussgang gain 1 and mean match on y. Rw (norms): the same on x = R y
        (E[R y_hat] = E[R y], Cov(R y_hat, R y) = Var(R y)), i.e. B = I of the effective-query lemma in the stored-norm score."""
        ut = ut.astype(np.float64); y = y.astype(np.float64)
        if Rw is None:
            my, mu = y.mean(0), ut.mean(0); g = ((ut - mu) * (y - my)).mean(0) / np.maximum(y.var(0), 1e-300)
            g = np.where(np.abs(g) > 1e-9, g, 1.0); return my - mu / g, g
        R = Rw[:, None].astype(np.float64); X = R * y; RU = R * ut
        a1 = R.mean(); a2 = RU.mean(0); b1 = X.mean(0); c1 = ((R - a1) * (X - b1)).mean(0); c2 = ((RU - a2) * (X - b1)).mean(0); v = X.var(0)
        num = c2 - a2 * c1 / a1; den = v - b1 * c1 / a1; ok = (np.abs(den) > 1e-300) & (np.abs(num) > 1e-12 * np.maximum(np.abs(den), 1e-300))
        g = np.where(ok, num / np.where(ok, den, 1.0), 1.0); A = (b1 - a2 / g) / a1
        return A, g

    def _apply_options(self):
        """Zero-bit values, DP weights, deterministic affine levels and DP curves for the current options (training sample)."""
        St = self.Str; Rt = self.Rt; d = self.d; Y64 = St.astype(np.float64)
        self.mu_t = Y64.mean(0)
        if self.zero == 'zero': self.c0 = np.zeros(d)
        elif self.zero == 'mean': self.c0 = self.mu_t.copy()
        else:
            R = Rt.astype(np.float64); X = R[:, None] * Y64; self.c0 = ((R - R.mean())[:, None] * (X - X.mean(0))).mean(0) / max(R.var(), 1e-300)
        self.wq = self.var.copy() if self.weight == 'none' else ((Rt.astype(np.float64) ** 2)[:, None] * Y64 ** 2).mean(0)
        Rw = None if self.weight == 'none' else Rt
        if self.weight == 'r2':                                    # deterministic levels with the norm-weighted gain
            for b in range(1, min(BM, LM8) + 1):
                lo = self.rlo[:, b]; hi = np.maximum(self.rhi[:, b], lo + 1e-12); L = 2.0 ** b; W = (hi - lo) / L
                mid = lo + (np.clip(np.floor((St - lo) / W), 0, L - 1) + 0.5) * W; A, g = self._gain_fit(mid, St, Rw)
                self.a[:, b] = A + (lo + 0.5 * W) / g; self.w[:, b] = W / g
        if self.curve_mode == 'det': self.curve = self.curve_det.copy(); return
        key = (self.weight, self.gains, self.zero)
        if key not in self._cache: self._cache[key] = self._curves_unified(Rw)
        self.curve = self._cache[key]

    def _curves_unified(self, Rw, G=16, nmax=50_000):
        """Per-component DP cost of the unified quantizer: expected within-variance of the error (times R if norm weighted) over
        dither draws (grid of G values; per-draw gain fit or the fixed gain), on at most nmax training rows."""
        St = self.Str[:nmax]; Rt = self.Rt[:nmax].astype(np.float64); d = self.d; cur = np.zeros((d, BM + 1)); Y64 = St.astype(np.float64)
        wv = (lambda E: E.var(0)) if Rw is None else (lambda E: (Rt[:, None] * E).var(0))
        Rsub = None if Rw is None else Rt
        cur[:, 0] = wv(self.c0[None, :] - Y64)
        for b in range(1, BM + 1):
            lo = self.rlo[:, b]; hi = np.maximum(self.rhi[:, b], lo + 1e-12); L = 2.0 ** b; W = (hi - lo) / L
            if b > LM8: cur[:, b] = (W ** 2 / 12) * (1.0 if Rw is None else float(np.mean(Rt ** 2))); continue
            if b == 1:
                K = np.clip(np.floor((St - lo) / W), 0, L - 1); cur[:, b] = wv(self.a[:, b] + self.w[:, b] * K - Y64); continue
            zg = (np.arange(G) + 0.5) / G - 0.5; acc = np.zeros(d); Eut = np.zeros_like(Y64) if self.gains == 'fixed' else None
            for zz in zg:
                zw = zz * W; ut = lo + (np.clip(np.floor((St + zw - lo) / W), 0, L - 1) + 0.5) * W - zw
                if self.gains == 'fixed': Eut += ut; continue
                A, g = self._gain_fit(ut, St, Rsub); acc += wv(A + ut / g - Y64)
            if self.gains == 'fixed':
                A, g = self._gain_fit(Eut / G, St, Rsub)
                for zz in zg:
                    zw = zz * W; ut = lo + (np.clip(np.floor((St + zw - lo) / W), 0, L - 1) + 0.5) * W - zw; acc += wv(A + ut / g - Y64)
                self._cache[('fg', b, Rw is not None)] = (A, g)      # shared by all variants of this fit
            cur[:, b] = acc / G
        return cur

    def variant(self, **opts):
        """A copy of this fitted codec with other step-0 options (cells, sample and basis shared; curves recomputed and cached)."""
        import copy
        c = copy.copy(self); c.a = self.a.copy(); c.w = self.w.copy(); c._cache = self._cache
        for k, v in opts.items(): setattr(c, {'curve': 'curve_mode'}.get(k, k), v)
        if c.weight == 'none':                                    # restore the unweighted deterministic levels
            c.a = self._a_none.copy() if hasattr(self, '_a_none') else self.a.copy(); c.w = self._w_none.copy() if hasattr(self, '_w_none') else self.w.copy()
        c.ensure_sample(); c._apply_options(); return c

    def coords(self, X, r):
        Y = X / np.maximum(r, 1e-30)[:, None] if self.norm else X
        return Y @ self.M if self.M is not None else Y

    def qcoords(self, Q):
        return Q @ self.M if self.M is not None else Q

    # ---------- allocation ----------
    def alloc(self, B, rule='exact'):
        d = self.d
        if rule == 'uniform': return np.full(d, int(B), np.int64)
        if rule == 'alpha1': return self.alpha_alloc(B, 1.0)
        if rule.startswith('alpha'): return self.alpha_alloc(B, float(rule[5:]))
        wq = getattr(self, 'wq', None); wq = self.var if wq is None else wq
        return E.dp_alloc(wq[:, None] * self.curve, int(round(B * d)))

    def alpha_alloc(self, B, alpha, cap=BM):
        """Eq. (2) with slope alpha on the component variances. Components above `cap` bits are
        fixed at the cap and the rest re-solved (water-filling); then integerized to exactly B*d bits as before (floor, the deficit one
        bit each to the active components with the largest fractional parts), never above the cap."""
        d = self.d; total = int(round(B * d)); lam = np.maximum(np.asarray(self.var, np.float64), 1e-300); order = np.argsort(-lam)
        fixed = np.zeros(d, bool); b = np.zeros(d)
        for _ in range(d + 1):
            idx = order[~fixed[order]]; budget = total - cap * int(fixed.sum()); b[:] = 0.0; b[fixed] = cap
            if len(idx) and budget > 0: b[idx] = compute_allocation(lam[idx], alpha, budget / len(idx))
            over = b > cap + 1e-9
            if not over.any(): break
            fixed |= over
        base = np.minimum(np.floor(b + 1e-9).astype(np.int64), cap); deficit = total - int(base.sum())
        if deficit > 0:
            frac = np.where((b > 0) & (base < cap), b - np.floor(b + 1e-9), -1.0)
            pick = np.argsort(-frac, kind='stable')[:deficit]; base[pick] += 1
        deficit = total - int(base.sum())
        while deficit > 0:                                        # rare: not enough active components below the cap
            j = int(np.argmax(np.where(base < cap, lam, -1.0))); base[j] += 1; deficit -= 1
        return base

    # ---------- frozen quantizer: fixed closed-form gains ----------
    def closed_gains(self):
        """Fixed gain of the dithered reconstruction (expectation over draws), closed form per component and b >= 2:
        g = 1 + Cov(beta, y)/Var(y), A = E[y] - (E[y] + E[beta])/g, beta(y) = (W/2 - (y - lo))_+ - (W/2 - (hi - y))_+."""
        self.ensure_sample(); St = self.Str; mu = St.astype(np.float64).mean(0); vy = np.maximum(St.astype(np.float64).var(0), 1e-300)
        self.A_fix = np.zeros((self.d, BM + 1)); self.g_fix = np.ones((self.d, BM + 1))
        for b in range(2, BM + 1):
            lo = self.rlo[:, b]; hi = np.maximum(self.rhi[:, b], lo + 1e-12); W = (hi - lo) / 2.0 ** b; acc = np.zeros(self.d); sb = np.zeros(self.d)
            for r0 in range(0, len(St), 50_000):
                y = St[r0:r0 + 50_000].astype(np.float64); beta = np.maximum(W / 2 - (y - lo), 0) - np.maximum(W / 2 - (hi - y), 0)
                acc += (beta * (y - mu)).sum(0); sb += beta.sum(0)
            n = len(St); bbar = sb / n; g = 1 + (acc / n) / vy; g = np.where(np.abs(g) > 1e-9, g, 1.0)
            self.g_fix[:, b] = g; self.A_fix[:, b] = mu - (mu + bbar) / g
        return self

    def decode_unified(self, Y, pl, z):
        """The frozen measurement = theory quantizer: b >= 2 shared subtractive dither + fixed gain, b = 1 deterministic split +
        gain, b = 0 the constant c0 (the component mean)."""
        act = pl['act']; bb = pl['b']; m2 = bb >= 2; out = self.decode_fast(Y, pl, 'dither', z)
        if m2.any():
            A = self.A_fix[act[m2], bb[m2]]; g = self.g_fix[act[m2], bb[m2]]; out[:, act[m2]] = A[None, :] + out[:, act[m2]] / g[None, :]
        if (~m2).any():
            Yd = self.decode_fast(Y, pl, 'det'); one = act[~m2]; out[:, one] = Yd[:, one]
        return out

    def affine_unified(self, pl, z):
        """Levels of the frozen quantizer as y_hat_i = a_i + w_i K_i on the active components (dither and gain folded into a_i, w_i)."""
        act = pl['act']; bb = pl['b']; m2 = bb >= 2; A = self.A_fix[act, bb]; g = self.g_fix[act, bb]
        a2 = A + (pl['lo'] + 0.5 * pl['W'] - z[act]) / g; w2 = pl['W'] / g
        return np.where(m2, a2, pl['a']), np.where(m2, w2, pl['w'])

    # ---------- vectorized coding (full scans) ----------
    def plan(self, bits):
        """Per-allocation vectors over the active components: lo, W, L, a, w."""
        act = np.nonzero(bits > 0)[0]; b = bits[act].astype(np.int64)
        lo = self.rlo[act, b]; hi = np.maximum(self.rhi[act, b], lo + 1e-12); L = 2.0 ** b; W = (hi - lo) / L
        c0 = getattr(self, 'c0', None); c0 = np.zeros(len(bits)) if c0 is None else c0
        return dict(act=act, lo=lo, W=W, L=L, a=self.a[act, b], w=self.w[act, b], d=len(bits), c0=np.where(bits > 0, 0.0, c0).astype(np.float32), b=b)

    def decode_fast(self, Y, pl, mode='det', z=None):
        act = pl['act']; Ya = Y[:, act]
        if mode == 'dither': Ya = Ya + z[act][None, :]
        K = np.clip(np.floor((Ya - pl['lo'][None, :]) / pl['W'][None, :]), 0, pl['L'][None, :] - 1)
        out = np.zeros((Y.shape[0], pl['d']), np.float32)
        if 'c0' in pl and pl['c0'].any(): out[:, :] = pl['c0'][None, :]
        if mode == 'det': out[:, act] = pl['a'][None, :] + pl['w'][None, :] * K
        else: out[:, act] = pl['lo'][None, :] + (K + 0.5) * pl['W'][None, :] - z[act][None, :]
        return out

    def ensure_sample(self):
        """Rebuild the training sample (same rows, seed) after load; needed by dither_gains."""
        if getattr(self, 'Str', None) is not None: return self.Str
        N, d = self.N, self.d; TS = int(min(200_000, 1.2e8 / d)); ti = np.sort(np.random.default_rng(self.seed).choice(N, min(TS, N), replace=False))
        if self.raw is not None: U0 = np.concatenate([self.rows(int(t), int(t) + 1) for t in ti])
        else: U0 = np.asarray(np.load(f'{PREP}/{self.ds}-db.npy', mmap_mode='r')[ti]).astype(np.float64)
        if self.norm: U0 = U0 / np.maximum(self.R[ti], 1e-30)[:, None]
        self.Str = (U0 @ self.M if self.M is not None else U0).astype(np.float32); self.ti, self.Rt = ti, self.R[ti]; return self.Str

    def unified_draw(self, bits, pl, seed=0):
        """Dither draw and its gain correction for the unified quantizer."""
        self.ensure_sample(); z = self.dither(bits, seed); return z, self.dither_gains(pl, z)

    def dither_gains(self, pl, z):
        """Per-component gain correction of the dithered reconstruction under the draw z, fitted on the training sample."""
        Yt = self.Str; act = pl['act']; Yq = self.decode_fast(Yt, pl, 'dither', z)
        ya = Yt[:, act].astype(np.float64); yq = Yq[:, act].astype(np.float64)
        if self.gains == 'fixed':                                  # expected gain over draws, per component at its bit count
            A = np.zeros(len(act)); g = np.ones(len(act)); w2 = self.weight == 'r2'
            for b in np.unique(pl['b']):
                sel = pl['b'] == b
                fg = getattr(self, '_cache', {}).get(('fg', int(b), w2))
                if b >= 2 and fg is not None: A0, g0 = fg; A[sel] = A0[act[sel]]; g[sel] = g0[act[sel]]
                else: A[sel], g[sel] = self._gain_fit(yq[:, sel], ya[:, sel], self.Rt if w2 else None)
            return dict(A=A, g=g, mu=ya.mean(0), m=yq.mean(0))
        if self.weight == 'r2':
            A, g = self._gain_fit(yq, ya, self.Rt); return dict(A=A, g=g, mu=ya.mean(0), m=yq.mean(0))
        g = ((yq - yq.mean(0)) * (ya - ya.mean(0))).mean(0) / np.maximum(ya.var(0), 1e-300); g = np.where(np.abs(g) > 1e-9, g, 1.0)
        return dict(mu=ya.mean(0), m=yq.mean(0), g=g)

    def decode_mode(self, Y, pl, mode, z=None, gd=None):
        """mode: 'det' (gain-corrected rounding), 'mid' (plain rounding), 'dither', 'dither_gain', 'unified' (1-bit det, >= 2 bits dither_gain)."""
        if mode == 'det': return self.decode_fast(Y, pl, 'det')
        if mode == 'mid':
            q = dict(pl); q['a'] = pl['lo'] + 0.5 * pl['W']; q['w'] = pl['W']; return self.decode_fast(Y, q, 'det')
        Yz = self.decode_fast(Y, pl, 'dither', z)
        if mode == 'dither': return Yz
        act = pl['act']; Yg = Yz.copy()
        if 'A' in gd: Yg[:, act] = gd['A'][None, :] + Yz[:, act] / gd['g'][None, :]
        else: Yg[:, act] = gd['mu'][None, :] + (Yz[:, act] - gd['m'][None, :]) / gd['g'][None, :]
        if mode == 'dither_gain': return Yg
        one = act[pl['L'] == 2]                                   # unified: 1-bit components deterministic + gain
        if len(one):
            Yd = self.decode_fast(Y, pl, 'det'); Yg[:, one] = Yd[:, one]
        return Yg

    # ---------- coding ----------
    def codes(self, Y, bits, z=None):
        d = Y.shape[1]; K = np.zeros(Y.shape, np.int64)
        for i in range(d):
            b = int(bits[i])
            if b == 0: continue
            lo, hi = self.rlo[i, b], self.rhi[i, b]
            if hi <= lo: hi = lo + 1e-12
            L = 2 ** b; W = (hi - lo) / L; y = Y[:, i] if z is None else Y[:, i] + z[i]
            K[:, i] = np.clip(np.floor((y - lo) / W), 0, L - 1)
        return K

    def dither(self, bits, seed=0):
        Wv = np.array([(max(self.rhi[i, b], self.rlo[i, b] + 1e-12) - self.rlo[i, b]) / 2 ** b if b > 0 else 0.0 for i, b in enumerate(bits)])
        return (np.random.default_rng(seed).uniform(-0.5, 0.5, size=len(bits)) * Wv)

    def decode(self, Y, bits, mode='det', z=None):
        K = self.codes(Y, bits, z if mode == 'dither' else None); Yh = np.zeros(Y.shape, np.float32)
        for i in range(Y.shape[1]):
            b = int(bits[i])
            if b == 0: continue
            if mode == 'det':
                Yh[:, i] = self.a[i, b] + self.w[i, b] * K[:, i]
            else:
                lo = self.rlo[i, b]; W = (max(self.rhi[i, b], lo + 1e-12) - lo) / 2 ** b
                Yh[:, i] = lo + (K[:, i] + 0.5) * W - z[i]
        return Yh

    def score(self, Q, Yh, r):
        """L2 estimate up to ||q||^2 (smaller is nearer): ||x||^2 - 2 s(x) <q, y_hat>."""
        S = self.qcoords(Q).astype(np.float64) @ Yh.T.astype(np.float64)
        if self.norm: S = S * r[None, :]
        return (r ** 2)[None, :] - 2 * S

    # ---------- persistence ----------
    def save(self, path):
        extra = dict(mu_raw=self.mu_raw, act_raw=self.act_raw) if self.raw is not None else {}
        np.savez_compressed(path, M=self.M if self.M is not None else np.zeros(0), mn=self.mn, mx=self.mx, var=self.var, rlo=self.rlo, rhi=self.rhi,
                            vro=self.vro, a=self.a, w=self.w, curve=self.curve, R=self.R, N=np.array(self.N),
                            meta=np.array([self.ds, self.basis, str(self.norm), str(self.seed), self.cells, self.curve_mode, self.weight, self.zero, self.gains]),
                            c0=getattr(self, 'c0', np.zeros(self.d)), wq=getattr(self, 'wq', self.var),
                            A_fix=getattr(self, 'A_fix', np.zeros(0)), g_fix=getattr(self, 'g_fix', np.zeros(0)), **extra)

    @classmethod
    def load(cls, path, R=None):
        z = np.load(path, allow_pickle=False); meta = [str(v) for v in z['meta']]; ds, basis, norm, seed = meta[:4]
        opt = dict(zip(('curve', 'weight', 'zero', 'gains'), meta[5:9])) if len(meta) > 8 else {}
        c = cls(ds, basis, norm == 'True', int(seed), cells=meta[4] if len(meta) > 4 else 'range', **opt); c.M = z['M'] if z['M'].size else None
        if 'c0' in z.files: c.c0 = z['c0']; c.wq = z['wq']
        if 'A_fix' in z.files and z['A_fix'].size: c.A_fix = z['A_fix']; c.g_fix = z['g_fix']
        for k in ('mn', 'mx', 'var', 'rlo', 'rhi', 'vro', 'a', 'w', 'curve'): setattr(c, k, z[k])
        c.d = len(c.var)
        if 'R' in z.files: c.R = z['R']; c.N = int(z['N'])
        if 'mu_raw' in z.files: c.mu_raw = z['mu_raw']; c.act_raw = z['act_raw']
        if R is not None: c.R = R
        return c
