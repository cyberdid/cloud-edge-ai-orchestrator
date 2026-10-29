"""Approximate nearest-neighbour search (ORION retrieval contract).

Both functions take `vectors` (`(N, d)` float array), `queries` (`(Q, d)` float
array) and an integer `k` (`1 <= k <= N`), and return `(indices, distances)`, each of
shape `(Q, k)`: row `q` holds `k` distinct vector indices and their squared Euclidean
distances to query `q`, in ascending order of distance.

- `exact_knn(vectors, queries, k)`: the exact `k` nearest neighbours.
- `approximate_knn(vectors, queries, k)`: an approximate nearest-neighbour search (it
  may miss some true neighbours). The reported distances must be the true squared
  distances of the returned indices, and when a query is identical to an indexed
  vector, that vector (distance 0) must be returned first.

Goal: make `approximate_knn` much faster than exact search while keeping recall@k
high. It is scored by recall@k against exact search, penalised when it takes more
than half the time of exact search on the same data. Available libraries: Python 3
standard library, numpy, scipy, scikit-learn.
"""
import numpy as np


def exact_knn(vectors, queries, k):
    x = np.asarray(vectors, dtype=float)
    q_all = np.asarray(queries, dtype=float)
    norms = (x * x).sum(axis=1)
    ids = np.empty((len(q_all), k), dtype=np.int64)
    dists = np.empty((len(q_all), k))
    for start in range(0, len(q_all), 100):
        q = q_all[start:start + 100]
        approx = (q * q).sum(axis=1)[:, None] + norms[None, :] - 2.0 * q @ x.T
        part = np.argpartition(approx, k - 1, axis=1)[:, :k]
        true = ((x[part] - q[:, None, :]) ** 2).sum(axis=2)
        order = np.lexsort((part, true), axis=1)
        ids[start:start + len(q)] = np.take_along_axis(part, order, axis=1)
        dists[start:start + len(q)] = np.take_along_axis(true, order, axis=1)
    return ids, dists


# EVOLVE-BLOCK-START
def approximate_knn(vectors, queries, k):
    """Baseline: exact search (recall 1.0, no speed-up)."""
    return exact_knn(vectors, queries, k)
# EVOLVE-BLOCK-END
