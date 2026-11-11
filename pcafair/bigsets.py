#!/usr/bin/env python3
"""Figures 1-3 and Table II on the kept datasets of 10M rows or more (sift100m, msmarco10m, msmarco113m) with L2 ground
truth and L2 scoring, under the frozen-codec protocol of the ten datasets (experiments/) unchanged.
Shared pieces:
  registry raw.fvecs, N, d, queries; the dataset mean mu and the PCA axes W of Cov(x) from the campaign's full pass
             (bigsets_work/stats/<ds>-stats.npz), W re-orthonormalized in float64. Prepared coordinates = (x_raw - mu) W, as
             $PCARR_PREPARED/<ds>-db.npy holds them for the 1M datasets.
  reader contiguous chunks of the raw.fvecs by pread; each chunk's pages are dropped from the page cache afterwards
             (posix_fadvise DONTNEED), so the 433 GB stream does not evict the running experiments' cached files.
  guards memory: below 6 GB available, wait until 9 GB; E7 window: while the chain is about to time the SIMD kernels
             (E5 done, E7 not done, chain alive) or E7 runs, wait (at most 4 h after E5 finished), so nothing of ours runs then.
  BigCodec a FairCodec fitted from the training sample (same rows: numpy default_rng(7), min(200k, 1.2e8/d) rows) and the
             full-database statistics of its coordinates (one streaming pass), then frozen exactly as freeze_codecs.py does."""
import os, sys, time, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from.codec import FairCodec, BM, LM8  # noqa: E402
from. import codec as FC  # noqa: E402
from.config import BIG_RAW, BIG_WORK, OUT as OUT_ROOT, FROZEN as FROZEN_DIR  # noqa: E402
WORK = str(BIG_WORK); OUT = str(OUT_ROOT / 'bigsets'); LOG = str(BIG_WORK / 'progress.md')
STATS = str(BIG_WORK / 'stats'); SCAL = str(BIG_RAW); FROZEN = str(FROZEN_DIR)
SETS = {
    'sift100m':    dict(base=f'{SCAL}/SIFT100M/base.100M.fvecs', d=128, N=100_000_000, query=f'{SCAL}/SIFT100M/query.10K.fvecs', nq=1000,
                        type='image', group='sift100m', cuts=(1_000_000, 10_000_000)),
    'msmarco10m':  dict(base=f'{SCAL}/MSMARCO100M/base.fvecs', d=1024, N=10_000_000, query=f'{SCAL}/MSMARCO100M/query.fvecs', nq=1677,
                        type='text', group='msmarco', cuts=(1_000_000,)),
    'msmarco113m': dict(base=f'{SCAL}/MSMARCO100M/base.fvecs', d=1024, N=113_520_750, query=f'{SCAL}/MSMARCO100M/query.fvecs', nq=1677,
                        type='text', group='msmarco', cuts=(1_000_000, 10_000_000)),
}
# S3: prefixes of the same streams as datasets of their own (own mean, PCA axes, codecs), same queries
SETS['pfx_sift1m'] = dict(SETS['sift100m'], N=1_000_000, cuts=(), group='pfx_sift')
SETS['pfx_sift10m'] = dict(SETS['sift100m'], N=10_000_000, cuts=(1_000_000,), group='pfx_sift')
SETS['pfx_msmarco1m'] = dict(SETS['msmarco113m'], N=1_000_000, cuts=(), group='pfx_msmarco')
GROUPS = {'sift100m': ['sift100m'], 'msmarco': ['msmarco10m', 'msmarco113m'], 'pfx_sift': ['pfx_sift1m', 'pfx_sift10m'], 'pfx_msmarco': ['pfx_msmarco1m']}
TS_RULE = lambda d: int(min(200_000, 1.2e8 / d))                                    # the codec's training-sample size (pcafair/codec.py)


def log(msg):
    line = f'- {time.strftime("%Y-%m-%d %H:%M")} {msg}'
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a') as f: f.write(line + '\n')


def spec(ds):
    """Registry entry plus mu (float64), W (float64, re-orthonormalized: polar factor of the campaign's float32 axes).
    Validation entries (big_validate.py) keep their own stats file in WORK."""
    s = dict(SETS[ds]); sp = f'{WORK}/{ds}-stats.npz' if ds.startswith(('val_', 'pfx_')) else f'{STATS}/{ds}-stats.npz'; z = np.load(sp)
    assert int(z['N']) == s['N'] and int(z['d']) == s['d'], ds
    s['mu'] = z['mean'].astype(np.float64)
    U_, _, Vt = np.linalg.svd(z['W'].astype(np.float64)); s['W'] = U_ @ Vt
    s['eig_x'] = z['eig'].astype(np.float64); s['name'] = ds
    return s


def prepared(s, raw):
    """raw rows (float32) -> Up, Rp, Ur, Rr, all float32 except the norms (float64): the prepared row p = (x - mu) W in float32 (as
    $PCARR_PREPARED/<ds>-db.npy holds it), u = p / ||p||, and u in the raw coordinates (the codec's 'orig' rows). speed choice:
    the 1M code builds u in float64 from the float32 prepared row and then rounds the coordinates to float32 before quantizing;
    here the transform runs in float32 (differences ~1e-7 relative, checked on audio: results/validation)."""
    if 'mu32' not in s: s['mu32'] = s['mu'].astype(np.float32); s['W32'] = s['W'].astype(np.float32)
    xr = raw - s['mu32']
    p = xr @ s['W32']
    Rp = np.sqrt(np.einsum('ij,ij->i', p, p, dtype=np.float64)); p /= np.maximum(Rp, 1e-30).astype(np.float32)[:, None]
    Rr = np.sqrt(np.einsum('ij,ij->i', xr, xr, dtype=np.float64)); xr /= np.maximum(Rr, 1e-30).astype(np.float32)[:, None]
    return p, Rp, xr, Rr


def prepared_queries(s, Q):
    """Queries in the prepared coordinates (float32 rounding as $PCARR_PREPARED/<ds>-query.npy) and centred raw (orig)."""
    Q64 = Q.astype(np.float64) - s['mu']
    if 'W32' not in s: s['W32'] = s['W'].astype(np.float32)
    return (Q64.astype(np.float32) @ s['W32']).astype(np.float64), Q64


def read_queries(s):
    with open(s['query'], 'rb') as f: d = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    assert d == s['d']
    a = np.fromfile(s['query'], dtype=np.float32).reshape(-1, d + 1)[:, 1:]
    return np.ascontiguousarray(a[:s['nq']])


def chunk_rows(d): return 524_288 if d <= 256 else 65_536


def iter_chunks(path, d, r0, r1, rows, cuts=()):
    """(row0, X float32 (n, d) contiguous) over rows [r0, r1) of an.fvecs file; chunks never cross a cut; pages dropped after use."""
    rec = (d + 1) * 4; fd = os.open(path, os.O_RDONLY)
    try:
        hd = np.frombuffer(os.pread(fd, 4, 0), np.int32)[0]; assert hd == d, (hd, d)
        bounds = sorted(set([c for c in cuts if r0 < c < r1]))
        pos = r0
        while pos < r1:
            nxt = min([r1, pos + rows] + [c for c in bounds if c > pos])
            n = nxt - pos; off = pos * rec; buf = bytearray(n * rec); mv = memoryview(buf); got = 0
            while got < n * rec:
                k = os.preadv(fd, [mv[got:]], off + got)
                if k <= 0: raise IOError('short read')
                got += k
            X = np.frombuffer(buf, np.float32).reshape(n, d + 1)[:, 1:].copy()
            os.posix_fadvise(fd, off, n * rec, os.POSIX_FADV_DONTNEED)
            yield pos, X
            pos = nxt
    finally:
        os.close(fd)


def read_rows(path, d, ids):
    """Rows `ids` (any order) of an.fvecs file as float32 (len(ids), d), by sorted pread runs; pages dropped afterwards."""
    ids = np.asarray(ids, np.int64); order = np.argsort(ids, kind='stable'); sid = ids[order]; rec = (d + 1) * 4
    out = np.empty((len(ids), d), np.float32); fd = os.open(path, os.O_RDONLY)
    try:
        for j, r in enumerate(sid):
            b = os.pread(fd, rec, int(r) * rec); out[order[j]] = np.frombuffer(b, np.float32)[1:]
        if len(sid):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return out


# ------------------------------------------------------------------ guards
def mem_avail_gb(): return int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 1048576


def e7_done(): return True        # the published runs shared the machine with a timing experiment; nothing to wait for here


def gate(tag=''):
    """Block while less than 6 GB of memory is available (the streaming passes are rerun from their checkpoints). A file named STOP
    next to the work directory stops the run at the next chunk."""
    said = False
    while True:
        if os.path.exists(f'{WORK}/STOP'): log(f'{tag}: STOP file found, exiting'); raise SystemExit(3)
        av = mem_avail_gb()
        if av >= (9 if said else 6):
            if said: log(f'{tag}: resumed (available {av:.1f} GB)')
            return
        if not said: log(f'{tag}: waiting (memory, available {av:.1f} GB)'); said = True
        time.sleep(30)


def wait_mem(gb, tag):
    """Before a step with a large one-off footprint (codec fit, training sample): wait until `gb` GB are available."""
    said = False
    while mem_avail_gb() < gb:
        if os.path.exists(f'{WORK}/STOP'): raise SystemExit(3)
        if not said: log(f'{tag}: waiting for {gb} GB available (now {mem_avail_gb():.1f} GB)'); said = True
        time.sleep(30)
    if said: log(f'{tag}: {mem_avail_gb():.1f} GB available, starting')


# ------------------------------------------------------------------ checkpoints
def save_ckpt(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path + '.tmp.npz'; np.savez(tmp, **arrs); os.replace(tmp, path)


def load_ckpt(path):
    if not os.path.exists(path): return None
    z = np.load(path, allow_pickle=False); return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ codec
class BigCodec(FairCodec):
    """FairCodec without the 1M data paths: built by freeze_big from the training sample and full-database statistics; the
    training sample (coding coordinates) and its norms are cached in WORK so that ensure_sample needs no database pass."""
    def __init__(self, ds, basis='pca', norm=True, seed=7, cells='range', keep_sample=False, curve='det', weight='none', zero='zero', gains='draw'):
        super().__init__(ds, 'pca' if basis == 'orig' else basis, norm, seed, cells, keep_sample, curve, weight, zero, gains)
        self.basis = basis

    def ensure_sample(self):
        if getattr(self, 'Str', None) is not None: return self.Str
        z = np.load(f'{WORK}/{self.ds}__{self.basis}-sample.npz')
        self.Str = z['Str']; self.ti = z['ti']; self.Rt = z['Rt']; return self.Str


def freeze_big(ds, basis, Str, ti, Rt, M, mn, mx, s1, s2, N, zero='mean', gains='closed'):
    """The tail of FairCodec.fit (pcafair/codec.py, after the database pass) on given statistics; options as freeze_codecs.py."""
    c = BigCodec(ds, basis, True, keep_sample=True, zero=zero, gains=gains)
    d = len(mn); c.N, c.d = int(N), d; c.M = M; c.R = np.asarray(Rt, np.float64)
    c.mn, c.mx = mn, mx; c.var = np.maximum(s2 / N - (s1 / N) ** 2, 1e-30)
    FC.NU.LMAX = 1
    vmm, vro, rlo, rhi, vlm, LM = FC.NU.design(Str, mn, mx, c.var, N)
    c.rlo, c.rhi, c.vro = rlo, rhi, vro
    c.Str = Str; c.ti, c.Rt = ti, np.asarray(Rt, np.float64)
    c.a = np.zeros((d, BM + 1)); c.w = np.zeros((d, BM + 1)); c.curve = np.zeros((d, BM + 1)); c.curve[:, 0] = c.var
    for i in range(d):
        s = Str[:, i].astype(np.float64); vx = s.var() + 1e-300; mu = s.mean()
        for b in range(1, BM + 1):
            lo, hi = rlo[i, b], rhi[i, b]
            if hi <= lo: hi = lo + 1e-12
            L = 2 ** b; W = (hi - lo) / L
            if b <= LM8:
                kk = np.clip(np.floor((s - lo) / W), 0, L - 1); mid = lo + (kk + 0.5) * W
                g = np.mean((mid - mid.mean()) * (s - mu)) / vx; g = g if abs(g) > 1e-9 else 1.0
                c.a[i, b] = mu + (lo + 0.5 * W - mid.mean()) / g; c.w[i, b] = W / g
                c.curve[i, b] = np.mean((mu + (mid - mid.mean()) / g - s) ** 2)
            else:
                c.a[i, b] = lo + 0.5 * W; c.w[i, b] = W; c.curve[i, b] = vro[i, b]
    c.curve_det = c.curve.copy(); c._cache = {}; c._a_none = c.a.copy(); c._w_none = c.w.copy()
    if (c.curve_mode, c.weight, c.zero, c.gains) != ('det', 'none', 'zero', 'draw'):
        c.ensure_sample(); c._apply_options()
    if c.gains == 'closed': c.closed_gains()
    return c


def decode_par(c, Y32, pl, z, pool, parts):
    """c.decode_unified on row blocks in threads (numpy releases the GIL; rows are independent, so the result is identical)."""
    n = Y32.shape[0]
    if parts <= 1 or n < 4096: return c.decode_unified(Y32, pl, z)
    cuts = np.linspace(0, n, parts + 1).astype(int)
    outs = list(pool.map(lambda ab: c.decode_unified(Y32[ab[0]:ab[1]], pl, z), zip(cuts[:-1], cuts[1:])))
    return np.concatenate(outs, axis=0)
