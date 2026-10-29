"""OpenEvolve pilot: pinned dependency and vendored upstream example files."""
import hashlib
import importlib.metadata as md
import re

import pytest

pytest.importorskip("openevolve")

UPSTREAM = "experiments/openevolve/upstream"


def test_openevolve_pinned_and_api_present():
    assert md.version("openevolve") == "0.4.0"
    from openevolve import OpenEvolve  # noqa: F401
    from openevolve.config import load_config  # noqa: F401
    from openevolve.utils.code_utils import apply_diff, extract_diffs  # noqa: F401


def test_vendored_files_match_notice(project_root):
    notice = (project_root / UPSTREAM / "NOTICE.md").read_text()
    entries = re.findall(r"^`?([\w/.]+\.(?:py|yaml))`?\s+`?([0-9a-f]{64})`?\s*$", notice, re.M)
    assert len(entries) == 3
    for relative, digest in entries:
        data = (project_root / UPSTREAM / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, relative
