"""Per-query top-k over score chunks, kept in numpy (the full scans of experiments/fullscan)."""
import numpy as np


class Top:
    """Per-query top-k over chunks: the first chunk by argpartition, later chunks only through the entries that beat the current k-th
    value (strictly), merged by one stable lexsort. `largest=False` keeps the k smallest scores (L2 distances)."""
    def __init__(self, nq, largest, k=10):
        self.k = k; self.s = np.full((nq, k), -np.inf if largest else np.inf); self.i = np.full((nq, k), -1, np.int64); self.lg = largest

    def update(self, q0, S, row0):
        K = self.k; q1 = q0 + S.shape[0]; cur_s, cur_i = self.s[q0:q1], self.i[q0:q1]
        if not np.isfinite(cur_s).all():
            part = np.argpartition(-S if self.lg else S, K, axis=1)[:, :K] if S.shape[1] > K else np.tile(np.arange(S.shape[1]), (S.shape[0], 1))
            cs = np.take_along_axis(S, part, 1).astype(np.float64); ci = part + row0
            allS = np.concatenate([cur_s, cs], 1); allI = np.concatenate([cur_i, ci], 1)
            keep = np.argpartition(-allS if self.lg else allS, K - 1, axis=1)[:, :K]
            cur_s[:] = np.take_along_axis(allS, keep, 1); cur_i[:] = np.take_along_axis(allI, keep, 1); return
        thr = (cur_s.min(1) if self.lg else cur_s.max(1)).astype(S.dtype)
        r, c = np.nonzero(S > thr[:, None] if self.lg else S < thr[:, None])
        if r.size == 0: return
        U = np.unique(r)
        rows = np.concatenate([r, np.repeat(U, K)]); vals = np.concatenate([S[r, c].astype(np.float64), cur_s[U].ravel()])
        idx = np.concatenate([c + row0, cur_i[U].ravel()])
        o = np.lexsort((-vals if self.lg else vals, rows)); rows, vals, idx = rows[o], vals[o], idx[o]
        take = (np.searchsorted(rows, U)[:, None] + np.arange(K)[None, :]).ravel()
        cur_s[U] = vals[take].reshape(-1, K); cur_i[U] = idx[take].reshape(-1, K)

    def recall(self, gt):
        return float(np.mean([len(set(self.i[q].tolist()) & set(gt[q].tolist())) / self.k for q in range(len(gt))]))
