"""ctypes wrapper of csrc/libbigkern.so, the streaming kernels of the large-dataset passes (experiments/bigsets/big_passA.py):
score-block filtering against per-query thresholds and a per-query top-k kept over chunks. Build the library first: make -C csrc."""
import os, ctypes
import numpy as np

_SO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csrc', 'libbigkern.so')
if not os.path.exists(_SO):
    raise ImportError(f'{_SO} not found: build it with  make -C csrc  (g++ with OpenMP)')
_L = ctypes.CDLL(_SO)
P = ctypes.c_void_p; I64 = ctypes.c_int64; I32 = ctypes.c_int32
_L.quant_idx.argtypes = [P, I64, I32, P, P, P, P]
_L.filter_gt.restype = I64; _L.filter_gt.argtypes = [P, I32, I64, P, P, P, P, I64]
_L.minmax_cols.argtypes = [P, I64, I32, P, P]
_L.pair_dots.argtypes = [P, P, I32, P, P, I64, P]
_L.score_filter.restype = I64; _L.score_filter.argtypes = [P, I64, I32, P, I32, P, P, P, P, I64]


def quant_idx(Z, mn, iw, Lm1, out=None):
    """Cell indices clip(floor((Z - mn) * iw), 0, Lm1) per column, as float32."""
    n, d = Z.shape; out = np.empty_like(Z) if out is None else out
    _L.quant_idx(Z.ctypes.data, n, d, mn.ctypes.data, iw.ctypes.data, Lm1.ctypes.data, out.ctypes.data); return out


class Filter:
    """Survivors S[q, j] > theta[q] of a score block, in a growable buffer; fused(X, Q, theta) scores and filters without the score matrix."""
    def __init__(self, cap=4_000_000):
        self.cap = cap; self.q = np.empty(cap, np.int32); self.j = np.empty(cap, np.int32); self.s = np.empty(cap, np.float32)

    def __call__(self, S, theta):
        nq, n = S.shape
        while True:
            c = _L.filter_gt(S.ctypes.data, nq, n, theta.ctypes.data, self.q.ctypes.data, self.j.ctypes.data, self.s.ctypes.data, self.cap)
            if c >= 0: return self.q[:c], self.j[:c], self.s[:c]
            self.__init__(int(-c * 1.2) + 1024)

    def fused(self, X, Q, theta):
        n, d = X.shape; nq = Q.shape[0]
        while True:
            c = _L.score_filter(X.ctypes.data, n, d, Q.ctypes.data, nq, theta.ctypes.data, self.q.ctypes.data, self.j.ctypes.data, self.s.ctypes.data, self.cap)
            if c >= 0: return self.q[:c], self.j[:c], self.s[:c]
            self.__init__(int(-c * 1.2) + 1024)


def minmax_cols(X, mn, mx):
    _L.minmax_cols(X.ctypes.data, X.shape[0], X.shape[1], mn.ctypes.data, mx.ctypes.data)


def pair_dots(A, B, ia, ib):
    out = np.empty(len(ia), np.float32); ia = np.ascontiguousarray(ia, np.int64); ib = np.ascontiguousarray(ib, np.int64)
    _L.pair_dots(A.ctypes.data, B.ctypes.data, A.shape[1], ia.ctypes.data, ib.ctypes.data, len(ia), out.ctypes.data); return out


class TopK:
    """Per-query top-k (scores, global ids) maintained over chunks; theta = the current k-th score of each query."""
    def __init__(self, nq, k):
        self.k = k; self.s = np.full((nq, k), -np.inf, np.float32); self.i = np.full((nq, k), -1, np.int64); self.theta = np.full(nq, -np.inf, np.float32); self.filled = False

    def first(self, S, row0):
        """Initialise from a full score block (argpartition per query)."""
        k = self.k; nq, n = S.shape
        if n <= k:
            self.s[:, :n] = S; self.i[:, :n] = row0 + np.arange(n)[None, :]
        else:
            part = np.argpartition(-S, k - 1, axis=1)[:, :k]
            self.s[:] = np.take_along_axis(S, part, 1); self.i[:] = part + row0
        self.theta[:] = self.s.min(1); self.filled = n >= k

    def merge(self, q, j, s, row0):
        if q.size == 0: return
        o = np.argsort(q, kind='stable'); q, j, s = q[o], j[o].astype(np.int64) + row0, s[o]
        cuts = np.flatnonzero(np.diff(q)) + 1; starts = np.r_[0, cuts]; ends = np.r_[cuts, q.size]; k = self.k
        for a, b in zip(starts, ends):
            qq = int(q[a]); cs = np.concatenate([self.s[qq], s[a:b]]); ci = np.concatenate([self.i[qq], j[a:b]])
            if cs.size > k: sel = np.argpartition(-cs, k - 1)[:k]; cs, ci = cs[sel], ci[sel]
            self.s[qq] = cs; self.i[qq] = ci; self.theta[qq] = cs.min()

    def update(self, S, row0, flt, m_first=65536):
        if not self.filled:                                   # initialise on the first m_first columns (argpartition), filter the rest
            n = S.shape[1]; m = min(n, m_first); self.first(S[:, :m], row0)
            if n > m and self.filled:
                q, j, s = flt(np.ascontiguousarray(S[:, m:]), self.theta); self.merge(q, j, s, row0 + m)
            elif n > m: self.first(S, row0)
        else: q, j, s = flt(S, self.theta); self.merge(q, j, s, row0)

    def ids(self): return self.i


def hits_per_query(ids, gt):
    return np.array([len(set(ids[i].tolist()) & set(gt[i].tolist())) for i in range(len(gt))], np.int8)
