"""OpenEvolve pilot: seeded, logging LLM router (fake backends, no network)."""
import json
import urllib.request

from experiments.openevolve.router import Router, choose_model, serve


def fake(text="stub", tokens=(3, 5)):
    calls = []

    def backend(messages):
        calls.append(messages)
        return text, tokens[0], tokens[1]

    backend.calls = calls
    return backend


def boom(messages):
    raise RuntimeError("quota exhausted")


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_choose_model_is_deterministic_and_near_share():
    first = [choose_model(101, i, 0.1) for i in range(1000)]
    assert first == [choose_model(101, i, 0.1) for i in range(1000)]
    assert 70 <= first.count("argon") <= 130


def test_share_zero_never_calls_argon(tmp_path):
    router = Router(101, 0.0, tmp_path / "calls.jsonl", gemma=fake(), argon=boom)
    for _ in range(100):
        assert router.complete([{"role": "user", "content": "hi"}]) == "stub"
    assert {r["model"] for r in read_log(tmp_path / "calls.jsonl")} == {"gemma"}


def test_backend_error_returns_empty_and_logs(tmp_path):
    router = Router(101, 1.0, tmp_path / "calls.jsonl", gemma=fake(), argon=boom)
    assert router.complete([{"role": "user", "content": "hi"}]) == ""
    assert router.complete([{"role": "user", "content": "hi"}]) == ""
    log = read_log(tmp_path / "calls.jsonl")
    assert [r["call_index"] for r in log] == [0, 1]
    assert log[0]["status"] == "error:RuntimeError"
    assert "quota exhausted" in log[0]["error"]


def test_log_record_fields(tmp_path):
    router = Router(7, 0.0, tmp_path / "calls.jsonl", gemma=fake("abc", (11, 22)), argon=boom)
    router.complete([{"role": "user", "content": "hi"}])
    record = read_log(tmp_path / "calls.jsonl")[0]
    assert record["prompt_tokens"] == 11 and record["completion_tokens"] == 22
    assert record["response"] == "abc" and record["status"] == "ok"
    assert len(record["response_sha256"]) == 64
    assert record["latency_s"] >= 0 and record["ts_start"] > 0


def test_empty_reply_status(tmp_path):
    router = Router(7, 0.0, tmp_path / "calls.jsonl", gemma=fake(""), argon=boom)
    router.complete([{"role": "user", "content": "hi"}])
    assert read_log(tmp_path / "calls.jsonl")[0]["status"] == "empty"


def test_http_roundtrip_openai_shape(tmp_path):
    router = Router(101, 0.0, tmp_path / "calls.jsonl", gemma=fake(), argon=boom)
    server = serve(router, 0)
    try:
        port = server.server_address[1]
        body = json.dumps({"model": "pilot-router", "messages": [{"role": "user", "content": "hi"}]}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(request, timeout=10))
    finally:
        server.shutdown()
    assert reply["choices"][0]["message"]["content"] == "stub"
    assert reply["usage"]["completion_tokens"] == 5


def test_resume_start_index(tmp_path):
    router = Router(101, 0.5, tmp_path / "calls.jsonl", gemma=fake(), argon=fake("cloud"), start_index=37)
    router.complete([{"role": "user", "content": "hi"}])
    record = read_log(tmp_path / "calls.jsonl")[0]
    assert record["call_index"] == 37
    assert record["model"] == choose_model(101, 37, 0.5)
