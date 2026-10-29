"""OpenEvolve pilot: candidate code runs only inside the Docker sandbox."""
import os

import pytest

from experiments.openevolve.sandbox_exec import run_candidate

real_docker = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1"
)

DRIVER = """
import json, candidate
json.dump(candidate.main(), open('/output/result.json', 'w'))
"""


@real_docker
def test_returns_driver_output():
    driver = "import json, candidate\njson.dump({'v': candidate.f() + 1}, open('/output/result.json', 'w'))"
    result = run_candidate("def f():\n    return 41\n", driver, {}, timeout_s=30)
    assert result.ok, result.error
    assert result.output == {"v": 42}


@real_docker
def test_payload_is_readable():
    program = "import json\ndef main():\n    return json.load(open('payload.json'))['x'] * 2\n"
    result = run_candidate(program, DRIVER, {"x": 21}, timeout_s=30)
    assert result.ok and result.output == 42


@real_docker
def test_timeout_is_failure():
    result = run_candidate("def main():\n    while True:\n        pass\n", DRIVER, {}, timeout_s=3)
    assert not result.ok
    assert result.error == "timeout"
    assert result.elapsed_s < 6


@real_docker
def test_no_network_and_no_host_files():
    program = '''
def main():
    out = {}
    try:
        import urllib.request
        urllib.request.urlopen("http://example.com", timeout=3)
        out["net"] = "open"
    except Exception as e:
        out["net"] = type(e).__name__
    try:
        open("/home/cyberdid/.bashrc").read()
        out["home"] = "read"
    except Exception as e:
        out["home"] = type(e).__name__
    return out
'''
    result = run_candidate(program, DRIVER, {}, timeout_s=30)
    assert result.ok, result.error
    assert result.output["net"] != "open"
    assert result.output["home"] != "read"


@real_docker
def test_crash_reports_stderr_tail():
    result = run_candidate("def main():\n    raise ValueError('boom')\n", DRIVER, {}, timeout_s=30)
    assert not result.ok
    assert "boom" in result.error


@real_docker
def test_output_too_large():
    program = "def main():\n    return 'x' * (10 * 1024 * 1024)\n"
    result = run_candidate(program, DRIVER, {}, timeout_s=60)
    assert not result.ok
    assert result.error == "output_too_large"
