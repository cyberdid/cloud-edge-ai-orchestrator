"""Independent structural and distance checks for approximate kNN outputs."""
from __future__ import annotations

import numpy as np


def validate_knn_output(vectors, queries, output, k: int, *, atol: float = 1e-6,
                        rtol: float = 1e-5) -> list[str]:
    """Return correctness failures for ``(neighbor_ids, squared_distances)``.

    This validates that every reported distance is the squared Euclidean distance
    from its query to the reported vector. It deliberately does not require the
    neighbors to be the exact nearest neighbors; recall is a separate metric.
    """
    errors: list[str] = []
    try:
        x = np.asarray(vectors, dtype=np.float64)
        q = np.asarray(queries, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return [f"inputs must be numeric: {type(exc).__name__}"]

    if x.ndim != 2 or q.ndim != 2 or x.shape[1:] != q.shape[1:]:
        return ["input shape mismatch"]
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        return ["inputs must be finite"]
    if not isinstance(k, (int, np.integer)) or k < 1 or k > len(x):
        return ["k must be an integer in [1, number of vectors]"]
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        return ["output must contain neighbor IDs and distances"]

    raw_ids, raw_distances = output
    try:
        ids = np.asarray(raw_ids)
        distances = np.asarray(raw_distances, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return [f"output must be numeric: {type(exc).__name__}"]

    expected_shape = (len(q), k)
    if ids.shape != expected_shape:
        errors.append(f"neighbor ID shape {ids.shape}, expected {expected_shape}")
    if distances.shape != expected_shape:
        errors.append(f"distance shape {distances.shape}, expected {expected_shape}")
    if errors:
        return errors
    if not np.issubdtype(ids.dtype, np.integer):
        return ["neighbor IDs must have an integer dtype"]

    if np.any(ids < 0) or np.any(ids >= len(x)):
        errors.append("neighbor IDs out of range")
    if any(len(set(row.tolist())) != k for row in ids):
        errors.append("duplicate neighbor IDs")
    if errors:
        return errors
    if not np.isfinite(distances).all():
        errors.append("distances must be finite")
        return errors
    if np.any(distances < 0):
        errors.append("distances must be non-negative")
        return errors

    expected = ((x[ids] - q[:, None, :]) ** 2).sum(axis=2)
    mismatches = ~np.isclose(distances, expected, atol=atol, rtol=rtol)
    if mismatches.any():
        row, column = np.argwhere(mismatches)[0]
        errors.append(
            "distance mismatch at query "
            f"{int(row)}, neighbor {int(column)}: reported {distances[row, column]:.9g}, "
            f"expected {expected[row, column]:.9g}"
        )
    return errors
