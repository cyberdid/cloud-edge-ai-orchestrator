"""Colab scale study: VM driver (fake runner and sync; no network, no GPU)."""
import json

from experiments.openevolve.colab import study
from experiments.openevolve.colab.study import RunSpec, batches, study_plan


class FakeProc:
    def __init__(self, code=0):
        self.returncode = code

    def poll(self):
        return self.returncode


class FakeSync:
    def __init__(self):
        self.states, self.pushed, self.restored = [], [], 0

    def restore(self):
        self.restored += 1

    def push(self, path):
        self.pushed.append(path)

    def write_state(self, state):
        self.states.append(state)

    def push_file(self, path):
        self.pushed.append(path)


def complete(root, spec):
    path = spec.path(root)
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({"status": "complete"}))


def test_plan_order_and_counts():
    plan = study_plan()
    assert len(plan) == 156
    assert {r.model for r in plan[:40]} == {"gemma4:26b"} and {r.arm for r in plan[:40]} == {"L", "B"}
    assert [r.model for r in (plan[40], plan[80])] == ["gemma4:12b", "gemma4:e4b"]
    # Deviation 2026-11-04: MoE long runs and MoE LC move before the 31B block.
    assert all(r.long and r.model == "gemma4:26b" for r in plan[120:126])
    assert all(r.arm == "LC" and r.model == "gemma4:26b" for r in plan[126:136])
    assert all(r.model == "gemma4:31b" for r in plan[136:])
    long = [r for r in plan if r.long]
    assert len(long) == 6 and {(r.task, r.max_iter) for r in long} == {("circle_packing", 1000), ("knn", 500)}
    assert {r.seed for r in plan if r.arm == "LC"} == {101, 202, 303, 404, 505}
    assert plan[0].dirname == "circle_packing_L_s101" and long[0].dirname.endswith("_long")
    assert plan[0].slug == "gemma4-26b"


def test_31b_runs_one_at_a_time(tmp_path):
    big = [r for r in study_plan() if r.model == "gemma4:31b"]
    out = list(batches(big, tmp_path))
    assert len(out) == 20 and all(len(b) == 1 for b in out)
    assert study.SERVING_PARALLEL["gemma4:31b"] == 1


def test_lc_deferred_until_agy_ready_then_runs_next(tmp_path):
    lc = [r for r in study_plan() if r.arm == "LC"][:2]
    big = [r for r in study_plan() if r.model == "gemma4:31b"][:3]
    answers = iter([False, False, True])  # owner logs in while the first 31B run is going
    launched = []
    result = study.main(tmp_path, FakeSync(), plan=lc + big,
                        runner=lambda spec, root: launched.append(spec) or FakeProc(),
                        prepare_model=lambda model: None, agy_check=lambda: next(answers, True), sleep=lambda s: None)
    assert result == "done"
    assert launched == [big[0], big[1], lc[0], lc[1], big[2]]


def test_batches_never_mix_models(tmp_path):
    plan = [r for r in study_plan() if r.model in ("gemma4:26b", "gemma4:12b") and not r.long and r.arm != "LC"]
    for r in plan[:39]:
        complete(tmp_path, r)
    out = list(batches(plan, tmp_path))
    assert out[0] == [plan[39]]  # the last 26b run alone, not paired with a 12b run
    assert all(len({r.model for r in b}) == 1 and len(b) <= 2 for b in out)


def test_restore_then_resume_skips_complete(tmp_path):
    plan = study_plan()[:3]
    complete(tmp_path, plan[0])
    plan[1].path(tmp_path).mkdir(parents=True)
    (plan[1].path(tmp_path) / "manifest.json").write_text(json.dumps({"status": "incomplete"}))
    launched, sync = [], FakeSync()
    result = study.main(tmp_path, sync, plan=plan, runner=lambda spec, root: launched.append(spec) or FakeProc(),
                        prepare_model=lambda model: None, agy_check=lambda: True, sleep=lambda s: None)
    assert result == "done" and sync.restored == 1
    assert launched == plan[1:]
    assert study.launch_args(plan[1], tmp_path)[-1] == "--resume"
    assert "--resume" not in study.launch_args(plan[2], tmp_path)


def test_lc_blocked_on_auth_writes_state(tmp_path):
    plan = [r for r in study_plan() if r.arm == "LC"][:2]
    launched, sync = [], FakeSync()
    result = study.main(tmp_path, sync, plan=plan, runner=lambda spec, root: launched.append(spec) or FakeProc(),
                        prepare_model=lambda model: None, agy_check=lambda: False, sleep=lambda s: None)
    assert result == "blocked_agy" and launched == []
    assert sync.states[-1]["blocked"] == "agy_login"


def test_quota_exit_blocks_lc(tmp_path):
    plan = [r for r in study_plan() if r.arm == "LC"][:4]
    sync = FakeSync()
    result = study.main(tmp_path, sync, plan=plan, runner=lambda spec, root: FakeProc(3),
                        prepare_model=lambda model: None, agy_check=lambda: True, sleep=lambda s: None)
    assert result == "blocked_quota" and sync.states[-1]["blocked"] == "argon_quota"


def test_model_prepared_once_per_switch(tmp_path):
    plan = [r for r in study_plan() if not r.long and r.arm != "LC"]
    plan = plan[38:42]  # two 26b runs, then two 12b runs
    prepared = []
    study.main(tmp_path, FakeSync(), plan=plan, runner=lambda spec, root: FakeProc(),
               prepare_model=prepared.append, agy_check=lambda: True, sleep=lambda s: None)
    assert prepared == ["gemma4:26b", "gemma4:12b"]


def test_finished_runs_are_pushed(tmp_path):
    plan = study_plan()[:2]
    sync = FakeSync()
    study.main(tmp_path, sync, plan=plan, runner=lambda spec, root: FakeProc(),
               prepare_model=lambda model: None, agy_check=lambda: True, sleep=lambda s: None)
    assert [p.name for p in sync.pushed] == [r.dirname for r in plan]
    assert sync.states[-1]["phase"] == "done"


def test_31b_runs_five_seeds_and_plan_total():
    plan = study_plan()
    big = [r for r in plan if r.model == "gemma4:31b"]
    assert len(big) == 20 and {r.seed for r in big} == {101, 202, 303, 404, 505}
    assert len(plan) == 156


def test_ensure_ollama_starts_server_when_down():
    started = []
    study.ensure_ollama(is_up=lambda: False, start=lambda: started.append(1), wait=lambda: True)
    assert started == [1]
    study.ensure_ollama(is_up=lambda: True, start=lambda: started.append(2), wait=lambda: True)
    assert started == [1]


def test_running_state_is_written_before_restore(tmp_path):
    events = []

    class OrderedSync(FakeSync):
        def restore(self):
            events.append("restore")
            super().restore()

        def write_state(self, state):
            events.append(("write_state", state["phase"]))
            super().write_state(state)

    sync = OrderedSync()
    study.main(tmp_path, sync, plan=study_plan()[:1], runner=lambda spec, root: FakeProc(),
               prepare_model=lambda model: None, agy_check=lambda: True, sleep=lambda s: None)
    assert events[:3] == [("write_state", "running"), "restore", ("write_state", "running")]



def test_model_digest_recorded_per_pull(tmp_path):
    plan = [r for r in study_plan() if not r.long and r.arm != "LC"][38:42]  # 26b, 26b, 12b, 12b
    sync = FakeSync()
    digests = {"gemma4:26b": "001e5dafc3c7", "gemma4:12b": "6114515d63c1"}
    study.main(tmp_path, sync, plan=plan, runner=lambda spec, root: FakeProc(),
               prepare_model=digests.get, agy_check=lambda: True, sleep=lambda s: None)
    assert sync.states[-1]["digests"] == digests
    record = json.loads((tmp_path / "gemma4-12b" / "model_digest.json").read_text())
    assert [e["digest"] for e in record] == ["6114515d63c1"]
    assert tmp_path / "gemma4-12b" / "model_digest.json" in sync.pushed


def test_digest_history_appends_only_on_change(tmp_path):
    plan = study_plan()[:1]
    for digest in ("aaa", "aaa", "bbb"):  # three VMs; the registry tag changes before the third
        study.main(tmp_path, FakeSync(), plan=plan, runner=lambda spec, root: FakeProc(),
                   prepare_model=lambda model: digest, agy_check=lambda: True, sleep=lambda s: None)
    record = json.loads((tmp_path / "gemma4-26b" / "model_digest.json").read_text())
    assert [e["digest"] for e in record] == ["aaa", "bbb"]


def test_login_window_waits_for_agy_then_runs_lc_first(tmp_path):
    lc = [r for r in study_plan() if r.arm == "LC"][:2]
    big = [r for r in study_plan() if r.model == "gemma4:31b"][:1]
    answers = iter([False, False, True])  # the owner logs in during the window
    launched, sync = [], FakeSync()
    result = study.main(tmp_path, sync, plan=lc + big,
                        runner=lambda spec, root: launched.append(spec) or FakeProc(),
                        prepare_model=lambda model: None, agy_check=lambda: next(answers, True),
                        sleep=lambda s: None, login_window=lambda: True)
    assert result == "done" and launched == [lc[0], lc[1], big[0]]
    assert any(s.get("waiting") == "login_window" for s in sync.states)


def test_login_window_expiry_defers_lc(tmp_path):
    lc = [r for r in study_plan() if r.arm == "LC"][:2]
    big = [r for r in study_plan() if r.model == "gemma4:31b"][:1]
    launched, sync = [], FakeSync()
    result = study.main(tmp_path, sync, plan=lc + big,
                        runner=lambda spec, root: launched.append(spec) or FakeProc(),
                        prepare_model=lambda model: None, agy_check=lambda: False,
                        sleep=lambda s: None, login_window=lambda: True, login_wait_s=120)
    assert result == "blocked_agy" and launched == [big[0]]
