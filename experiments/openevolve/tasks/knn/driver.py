"""Sandbox driver for the kNN task. Datasets are generated before the candidate is
imported, and the clock is bound to a local name first, so module-level code in the
candidate cannot change the data or the timer the driver uses."""
from experiments.openevolve.tasks.knn.data import GENERATOR_SRC

DRIVER = GENERATOR_SRC + '''
import json
import time

_clock = time.perf_counter
payload = json.load(open("payload.json"))
mode = payload["mode"]
sets = [make_set(**spec) for spec in payload.get("sets", [])]
fresh = [make_set(**spec) for spec in payload.get("fresh", [])]


def _timed(fn, *args):
    start = _clock()
    out = fn(*args)
    return _clock() - start, out


def _rows(out):
    ids, dists = out
    return np.asarray(ids).tolist(), np.asarray(dists, dtype=float).tolist()


def _batches(queries):
    return np.array_split(np.arange(len(queries)), payload.get("batches", 3))


result = {}
if mode == "reference":
    result["sets"] = [sum(_timed(brute_knn, x, qs[b], k)[0] for b in _batches(qs)) for x, qs, k, _ in sets]
else:
    import candidate
    if mode == "working":
        rows = []
        for x, qs, k, _ in sets:
            try:
                ref_t, (ref_ids, _) = min((_timed(brute_knn, x, qs, k) for _ in range(3)), key=lambda r: r[0])
                runs = [_timed(candidate.approximate_knn, x, qs, k) for _ in range(3)]
                cand_t, cand_out = min(runs, key=lambda r: r[0])
                ids = np.asarray(cand_out[0])
                if ids.shape != (len(qs), k):
                    raise ValueError(f"indices shape {ids.shape}, expected {(len(qs), k)}")
                hits = sum(len(set(map(int, a)) & set(map(int, b))) for a, b in zip(ids, ref_ids))
                rows.append({"recall": hits / (k * len(qs)), "ratio": cand_t / ref_t})
            except Exception as exc:
                rows.append({"error": type(exc).__name__ + ": " + str(exc)[:200]})
        result["sets"] = rows
    elif mode == "recheck":
        out_fresh, out_sets = [], []
        for x, qs, k, _ in fresh:
            try:
                out_fresh.append(_rows(candidate.approximate_knn(x, qs, k)))
            except Exception as exc:
                out_fresh.append({"error": type(exc).__name__ + ": " + str(exc)[:200]})
        for x, qs, k, _ in sets:
            try:
                total, ids = 0.0, []
                for b in _batches(qs):
                    t, out = _timed(candidate.approximate_knn, x, qs[b], k)
                    total += t
                    ids += _rows(out)[0]
                out_sets.append({"time": total, "ids": ids})
            except Exception as exc:
                out_sets.append({"error": type(exc).__name__ + ": " + str(exc)[:200]})
        result = {"fresh": out_fresh, "sets": out_sets}
json.dump(result, open("/output/result.json", "w"))
'''
