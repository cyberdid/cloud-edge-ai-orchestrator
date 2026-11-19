from experiments.openevolve.colab import paper2_pilot


def test_ensure_system_package_installs_missing_dependency(monkeypatch):
    calls = []
    monkeypatch.setattr(paper2_pilot.shutil, "which", lambda _package: None)
    monkeypatch.setattr(
        paper2_pilot,
        "run_checked",
        lambda args, **kwargs: calls.append(args),
    )

    paper2_pilot.ensure_system_package("zstd")

    assert calls == [["apt-get", "update", "-qq"], ["apt-get", "install", "-y", "zstd"]]


def test_ensure_system_package_skips_installed_dependency(monkeypatch):
    calls = []
    monkeypatch.setattr(paper2_pilot.shutil, "which", lambda _package: "/usr/bin/zstd")
    monkeypatch.setattr(
        paper2_pilot,
        "run_checked",
        lambda *args, **kwargs: calls.append(args),
    )

    paper2_pilot.ensure_system_package("zstd")

    assert calls == []
