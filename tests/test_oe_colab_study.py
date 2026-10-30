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


def complete(root, spec):
    path = spec.path(root)
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({"status": "complete"}))


def test_plan_order_and_counts():
    plan = study_plan()
    assert len(plan) == 156
    assert {r.model for r in plan[:40]} == {"gemma4:26b"} and {r.arm for r in plan[:40]} == {"L", "B"}
    assert [r.model for r in (plan[40], plan[80], plan[120])] == ["gemma4:12b", "gemma4:e4b", "gemma4:31b"]
    long = [r for r in plan if r.long]
    assert len(long) == 6 and {(r.task, r.max_iter) for r in long} == {("circle_packing", 1000), ("knn", 500)}
    assert all(r.arm == "LC" and r.model == "gemma4:26b" for r in plan[-10:])
    assert {r.seed for r in plan if r.arm == "LC"} == {101, 202, 303, 404, 505}
    assert plan[0].dirname == "circle_packing_L_s101" and long[0].dirname.endswith("_long")
    assert plan[0].slug == "gemma4-26b"


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
