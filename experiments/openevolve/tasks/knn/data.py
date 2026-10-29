"""Data for the kNN task: dataset generators shared by the host and the sandbox driver.

``GENERATOR_SRC`` is executed here and embedded verbatim in the sandbox driver, so the
host and the container build bit-identical datasets from the same seeds (numpy's
default_rng is stable within one numpy version; host and image both pin 2.5.4).
Only the development sets live here; sealed evaluation sets belong to the re-check.
"""
from __future__ import annotations

import numpy as np

GENERATOR_SRC = '''
import numpy as np

QUERY_CHUNK = 100


def make_set(seed, n, d, k, centres, sigma, q=500, copies=0):
    """Gaussian blobs (centres > 0) or uniform [0,1)^d data; the first ``copies``
    queries are exact copies of indexed vectors. Returns (x, queries, k, copied_ids)."""
    rng = np.random.default_rng(seed)
    if centres:
        c = rng.random((centres, d))
        x = c[rng.integers(centres, size=n)] + rng.normal(0.0, sigma, (n, d))
        queries = c[rng.integers(centres, size=q)] + rng.normal(0.0, sigma, (q, d))
    else:
        x = rng.random((n, d))
        queries = rng.random((q, d))
    copied = rng.choice(n, copies, replace=False) if copies else np.empty(0, dtype=np.int64)
    queries[:copies] = x[copied]
    return x, queries, int(k), copied


def brute_knn(x, queries, k):
    """Exact kNN by chunked matrix products; distances recomputed exactly."""
    norms = (x * x).sum(axis=1)
    ids = np.empty((len(queries), k), dtype=np.int64)
    dists = np.empty((len(queries), k))
    for start in range(0, len(queries), QUERY_CHUNK):
        q = queries[start:start + QUERY_CHUNK]
        approx = (q * q).sum(axis=1)[:, None] + norms[None, :] - 2.0 * q @ x.T
        part = np.argpartition(approx, k - 1, axis=1)[:, :k]
        true = ((x[part] - q[:, None, :]) ** 2).sum(axis=2)
        order = np.lexsort((part, true), axis=1)
        ids[start:start + len(q)] = np.take_along_axis(part, order, axis=1)
        dists[start:start + len(q)] = np.take_along_axis(true, order, axis=1)
    return ids, dists
'''

exec(GENERATOR_SRC)  # defines make_set, brute_knn, QUERY_CHUNK in this module

DEV_SETS = [
    dict(seed=11, n=200_000, d=32, k=10, centres=64, sigma=0.05),
    dict(seed=12, n=100_000, d=16, k=10, centres=100, sigma=0.03),
    dict(seed=13, n=150_000, d=24, k=5, centres=50, sigma=0.06),
]


def fresh_specs(seed: int, count: int = 6) -> list[dict]:
    """Small uniform validity challenges with 5 exact-copy queries each."""
    rng = np.random.default_rng(seed)
    return [
        dict(seed=int(rng.integers(2**31)), n=int(rng.integers(2000, 20001)), d=int(rng.integers(2, 65)),
             k=int(rng.integers(1, 11)), centres=0, sigma=0.0, q=50, copies=5)
        for _ in range(count)
    ]
