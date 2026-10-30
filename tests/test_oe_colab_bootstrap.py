"""Colab scale study: static checks of the VM bootstrap script."""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/openevolve/colab/bootstrap.sh"


def text():
    return SCRIPT.read_text()


def test_bootstrap_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_bootstrap_pins_versions():
    for needle in ("--locked", "requirements.lock", "OLLAMA_VERSION=0.40.13", "OLLAMA_NUM_PARALLEL=2",
                   "OLLAMA_KEEP_ALIVE=-1", "chmod 700 /root", "/opt/oe-cand", "UV_PYTHON_INSTALL_DIR=/opt/uv-python",
                   "BOOTSTRAP OK"):
        assert needle in text(), needle


def test_bootstrap_never_echoes_key():
    assert "cat /root/.oe/sa-key.json" not in text()
    assert "activate-service-account --key-file /root/.oe/sa-key.json" in text()
    assert "CLOUDSDK_CONFIG=/root/.oe/gcloud" in text()
