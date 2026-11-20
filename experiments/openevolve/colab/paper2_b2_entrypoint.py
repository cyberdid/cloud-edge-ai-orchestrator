"""Tiny Colab entrypoint: extract the pinned checkout before importing B2 code."""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

REPO = Path("/content/oe")
ARCHIVE = Path("/content/oe.tar.gz")


def main() -> None:
    stamp = REPO / ".paper2-source-sha"
    revision = __import__("os").environ.get("PAPER2_GIT_SHA", "")
    if not revision:
        raise SystemExit("PAPER2_GIT_SHA must be set by the B2 watchdog")
    if not (stamp.exists() and stamp.read_text().strip() == revision and
            (REPO / "pyproject.toml").exists()):
        REPO.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ARCHIVE, "r:gz") as bundle:
            bundle.extractall(REPO, filter="data")
        stamp.write_text(revision)
    sys.path.insert(0, str(REPO))
    sys.argv = [sys.argv[0], "--protocol", str(
        REPO / "experiments/openevolve/configs/b2_estimation_2026-11-20.json")]
    from experiments.openevolve.colab.paper2_b2_estimation import main as run_b2
    run_b2()


if __name__ == "__main__":
    main()
