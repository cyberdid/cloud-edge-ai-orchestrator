import tarfile

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


def test_project_site_packages_returns_uv_environment(tmp_path):
    expected = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages"
    expected.mkdir(parents=True)

    assert paper2_pilot.project_site_packages(tmp_path) == expected


def test_sync_source_archive_refreshes_revision_and_preserves_uv_environment(tmp_path):
    repo = tmp_path / "oe"
    repo.mkdir()
    (repo / "stale.py").write_text("old")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "marker").write_text("keep")
    archive_path = tmp_path / "oe.tar.gz"
    source = tmp_path / "source"
    source.mkdir()
    (source / "fresh.py").write_text("new")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source / "fresh.py", arcname="fresh.py")

    paper2_pilot.sync_source_archive(repo, archive_path, "a" * 40)

    assert (repo / "fresh.py").read_text() == "new"
    assert not (repo / "stale.py").exists()
    assert (repo / ".venv" / "marker").read_text() == "keep"
    assert (repo / ".paper2-source-sha").read_text() == "a" * 40
