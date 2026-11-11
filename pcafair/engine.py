"""Allocation, coding gain, rotations, and.fvecs reading shared by the experiments.

Allocation: b_i = max(0, alpha * log2(sigma_i^2) + c) with sum b_i = B * d (continuous), integerized with the fixed-budget
rule (floor, then one extra bit each to the active components with the largest fractional parts).
"""
import numpy as np


def compute_allocation(eigvals, alpha, B):
    """Continuous allocation of Eq. (2) for eigenvalues sorted in descending order; sum = B * d."""
    d = len(eigvals)
    budget = B * d
    if alpha < 1e-10:
        return np.full(d, float(B))
    log_lam = np.log2(eigvals)
    for n_active in range(d, 0, -1):
        c = (budget - alpha * log_lam[:n_active].sum()) / n_active
        if alpha * log_lam[n_active - 1] + c >= 0:
            break
    else:
        n_active = 1
        c = budget - alpha * log_lam[0]
    b = np.zeros(d)
    b[:n_active] = alpha * log_lam[:n_active] + c
    b = np.maximum(b, 0)
    return b


def integerize_budget(b_cont, total_bits):
    """Integer allocation with exactly total_bits: floor, then the deficit goes one bit each to the active components
    with the largest fractional parts (components with b = 0 are never bumped)."""
    base = np.floor(b_cont).astype(np.int64)
    deficit = int(round(total_bits)) - int(base.sum())
    if deficit > 0:
        frac = np.where(b_cont > 0, b_cont - np.floor(b_cont), -1.0)
        base[np.argsort(-frac, kind="stable")[:deficit]] += 1
    return base


def compute_cg(eigvals, clip=1e-10):
    """Coding gain CG = AM / GM of the eigenvalues (floored at clip)."""
    e = np.maximum(eigvals.astype(np.float64), clip)
    return e.mean() / np.exp(np.log(e).mean())


def dense_haar(d, rng_seed, dtype=np.float64):
    """Dense Haar rotation: QR of a Gaussian matrix with the signs of R's diagonal folded into Q."""
    rng = np.random.default_rng(rng_seed)
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return (q * np.sign(np.diag(r))[None, :]).astype(dtype)


def haar_fullscan(d, seed):
    """Rotation of the full-scan and re-ranking experiments (seeds 0-4)."""
    return dense_haar(d, 1000 + seed, np.float64)


def read_fvecs(path, n=None):
    """First n rows (all if None) of an.fvecs file as float32."""
    with open(path, "rb") as f:
        d = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    a = np.fromfile(path, dtype=np.float32, count=((n * (d + 1)) if n else -1))
    return a.reshape(-1, d + 1)[:, 1:]
