"""OpenEvolve pilot: campaign script order, resume and quota pause (fake runner)."""
import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_openevolve_pilot.sh"


def run_script(tmp_path, runner, *args, dry=False):
    env = dict(os.environ, OE_PILOT_ROOT=str(tmp_path / "runs"), OE_PILOT_RUNNER=runner,
               OE_PILOT_SKIP_TIMERS="1")
    if dry:
        env["OE_PILOT_DRY_RUN"] = "1"
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


def test_run_order_L_B_before_LC(tmp_path):
    out = run_script(tmp_path, "echo", "run", dry=True)
    planned = [line for line in out.stdout.splitlines() if line.startswith("PLAN ")]
    assert len(planned) == 18
    arms = [line.split("--arm ")[1].split()[0] for line in planned]
    assert set(arms[:12]) == {"L", "B"} and arms[12:] == ["LC"] * 6


def test_quota_pause_stops_campaign(tmp_path):
    fake = tmp_path / "fake_runner.sh"
    log = tmp_path / "invocations.txt"
    fake.write_text(f"""#!/usr/bin/env bash
echo "$*" >> {log}
out=""; arm=""
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift;; --arm) arm="$2"; shift;; esac; shift; done
mkdir -p "$out"
if [ "$arm" = "LC" ]; then echo '{{"status": "paused_quota"}}' > "$out/manifest.json"; exit 3; fi
echo '{{"status": "complete"}}' > "$out/manifest.json"
""")
    fake.chmod(0o755)
    result = run_script(tmp_path, str(fake), "run")
    calls = log.read_text().splitlines()
    assert len(calls) == 13  # 12 L/B runs, then the first LC run pauses the campaign
    assert "paused" in result.stdout + result.stderr
    again = run_script(tmp_path, str(fake), "run")
    resumed = log.read_text().splitlines()[13:]
    assert len(resumed) == 1 and "--resume" in resumed[0]  # complete runs are skipped
    assert again.returncode == 3
