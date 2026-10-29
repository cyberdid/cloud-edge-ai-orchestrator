"""OpenEvolve pilot T2: approximate kNN working evaluator and RIE-based re-check."""
import os

import numpy as np
import pytest

from experiments.openevolve.static_check import forbidden_primitives
from experiments.openevolve.tasks.knn import data, working_evaluator
from experiments.openevolve.tasks.knn import recheck
from experiments.openevolve.tasks.knn.recheck import recheck_knn

real_docker = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1"
)

SEED = "experiments/openevolve/tasks/knn/initial_program.py"


def with_block(project_root, block):
    """The seed program with its EVOLVE-BLOCK replaced by ``block``."""
    code = (project_root / SEED).read_text()
    head, rest = code.split("# EVOLVE-BLOCK-START\n")
    _, tail = rest.split("# EVOLVE-BLOCK-END\n")
    return head + "# EVOLVE-BLOCK-START\n" + block + "# EVOLVE-BLOCK-END\n" + tail


def write(tmp_path, code):
    path = tmp_path / "program.py"
    path.write_text(code)
    return str(path)


EUCLIDEAN = '''
def approximate_knn(vectors, queries, k):
    ids, d = exact_knn(vectors, queries, k)
    return ids, np.sqrt(d)
'''

QUERY_INDEPENDENT = '''
def approximate_knn(vectors, queries, k):
    q = np.asarray(queries)
    return np.tile(np.arange(k), (len(q), 1)), np.zeros((len(q), k))
'''

IVF = '''
def approximate_knn(vectors, queries, k, cells=128, probe=3):
    x = np.asarray(vectors, dtype=float)
    q = np.asarray(queries, dtype=float)
    rng = np.random.default_rng(0)
    centroids = x[rng.choice(len(x), min(cells, len(x)), replace=False)]
    assign = exact_knn(centroids, x, 1)[0][:, 0]
    order = np.argsort(assign, kind="stable")
    starts = np.searchsorted(assign[order], np.arange(len(centroids) + 1))
    probes = exact_knn(centroids, q, min(probe, len(centroids)))[0]
    ids = np.empty((len(q), k), dtype=np.int64)
    dists = np.empty((len(q), k))
    for row in range(len(q)):
        cand = np.concatenate([order[starts[p]:starts[p + 1]] for p in probes[row]])
        if len(cand) < k:
            cand = np.arange(len(x))
        d = ((x[cand] - q[row]) ** 2).sum(axis=1)
        best = np.argsort(d, kind="stable")[:k]
        ids[row], dists[row] = cand[best], d[best]
    return ids, dists
'''

CACHING = '''
_cache = {}


def approximate_knn(vectors, queries, k):
    key = (len(vectors), len(queries), k, float(np.asarray(queries)[0, 0]))
    if key not in _cache:
        _cache[key] = exact_knn(vectors, queries, k)
    return _cache[key]
'''


def test_gc_is_forbidden():
    assert forbidden_primitives("import gc\n") != []


def test_holdout_not_reachable_from_working_path(project_root):
    for relative in ("experiments/openevolve/tasks/knn/working_evaluator.py",
                     "experiments/openevolve/tasks/knn/data.py", SEED):
        text = (project_root / relative).read_text()
        assert "holdout" not in text and "9001" not in text, relative


def test_dev_and_holdout_parameters_differ():
    dev = {(s["n"], s["d"], s["centres"]) for s in data.DEV_SETS}
    hold = {(s["n"], s["d"], s["centres"]) for s in recheck.HOLDOUT_SETS}
    assert len(data.DEV_SETS) == 3 and len(recheck.HOLDOUT_SETS) == 3
    assert not dev & hold


@real_docker
def test_seed_is_valid_but_slow(project_root):
    metrics = working_evaluator.evaluate(str(project_root / SEED))
    assert metrics["recall"] == pytest.approx(1.0)
    assert 0.12 < metrics["combined_score"] < 0.45
    result = recheck_knn((project_root / SEED).read_text())
    assert result["valid"] is True, result["reasons"]
    assert result["holdout_recall"] == pytest.approx(1.0)
    assert result["score"] < 0.45


@real_docker
def test_euclidean_distances_pass_working_fail_recheck(project_root, tmp_path):
    code = with_block(project_root, EUCLIDEAN)
    assert working_evaluator.evaluate(write(tmp_path, code))["recall"] == pytest.approx(1.0)
    result = recheck_knn(code)
    assert result["valid"] is False
    assert any("distance not derived" in r for r in result["reasons"])


@real_docker
def test_query_independent_result_scores_low(project_root, tmp_path):
    code = with_block(project_root, QUERY_INDEPENDENT)
    assert working_evaluator.evaluate(write(tmp_path, code))["recall"] < 0.05


@real_docker
def test_ivf_reference_passes_both(project_root, tmp_path):
    code = with_block(project_root, IVF)
    metrics = working_evaluator.evaluate(write(tmp_path, code))
    assert metrics["recall"] > 0.8
    result = recheck_knn(code)
    assert result["valid"] is True, result["reasons"]
    assert result["holdout_recall"] > 0.5


@real_docker
def test_caching_inflates_working_but_not_recheck(project_root, tmp_path):
    code = with_block(project_root, CACHING)
    metrics = working_evaluator.evaluate(write(tmp_path, code))
    assert metrics["speed_ratio"] < 0.2
    result = recheck_knn(code)
    assert result["holdout_speed_ratio"] > 0.5


@real_docker
def test_numpy_int_ids_accepted(project_root):
    # the seed returns np.int64 arrays; re-check must convert before check_neighbors
    assert recheck_knn((project_root / SEED).read_text())["valid"] is True


def test_generators_are_deterministic():
    a = data.make_set(**data.DEV_SETS[0], q=20)
    b = data.make_set(**data.DEV_SETS[0], q=20)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


MATMUL_DISTANCES = '''
def approximate_knn(vectors, queries, k):
    x = np.asarray(vectors, dtype=float)
    q = np.asarray(queries, dtype=float)
    ids, _ = exact_knn(x, q, k)
    d = (q * q).sum(axis=1)[:, None] + (x[ids] ** 2).sum(axis=2) - 2.0 * np.einsum("qd,qkd->qk", q, x[ids])
    return ids, np.maximum(d, 0.0) + 1e-15
'''


@real_docker
def test_exact_copy_rule_uses_rie_tolerance(project_root):
    result = recheck_knn(with_block(project_root, MATMUL_DISTANCES))
    assert not any("indexed copy" in r for r in result["reasons"]), result["reasons"]
