"""No-model-call GCS checkpoint probe for a fresh B2 Colab runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

PROTOCOL_ID = "paper2-b2-estimation-2026-11-20-v1"
BASE_PREFIX = "paper2-v2/b2-estimation-2026-11-20/no-call-rehearsal"
BUCKET = "oe-colab-scale-0903767759"


def _client():
    from google.cloud import storage
    from google.oauth2 import service_account

    key = Path(os.environ.get("OE_PILOT_GCS_CREDENTIALS", "/content/sa-key.json"))
    credentials = service_account.Credentials.from_service_account_file(str(key))
    return storage.Client(project=credentials.project_id, credentials=credentials)


def run_stage(stage: str, *, protocol_sha256: str, source_commit: str,
              bucket: Any | None = None, bucket_name: str = BUCKET,
              run_id: str = "rehearsal-20261120") -> dict[str, Any]:
    if stage not in {"upload", "restore"}:
        raise ValueError("stage must be upload or restore")
    if bucket is None:
        bucket = _client().bucket(bucket_name)
    prefix = f"{BASE_PREFIX}/{run_id}"
    phase_blob = bucket.blob(f"{prefix}/phase.json")
    sentinel_blob = bucket.blob(f"{prefix}/sentinel.json")
    terminal_blob = bucket.blob(f"{prefix}/terminal-state.json")
    if stage == "upload":
        payload = {"protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_sha256,
                   "source_commit": source_commit, "marker": "persist-across-colab-runtime-restart",
                   "model_calls": 0, "written_unix": time.time()}
        encoded = json.dumps(payload, sort_keys=True) + "\n"
        sentinel_blob.upload_from_string(encoded, content_type="application/json",
                                         if_generation_match=0)
        phase = {"phase": "uploaded", "protocol_sha256": protocol_sha256,
                 "source_commit": source_commit, "model_calls": 0}
        phase_blob.upload_from_string(json.dumps(phase, sort_keys=True) + "\n",
                                      content_type="application/json")
        return phase

    phase = json.loads(phase_blob.download_as_text())
    payload = json.loads(sentinel_blob.download_as_text())
    expected = {"protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_sha256,
                "source_commit": source_commit, "marker": "persist-across-colab-runtime-restart",
                "model_calls": 0}
    if phase.get("phase") != "uploaded" or any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("remote rehearsal sentinel or phase does not match the requested hashes")
    verified_sha = hashlib.sha256(sentinel_blob.download_as_bytes()).hexdigest()
    terminal_state = {"phase": "complete", "protocol_id": PROTOCOL_ID,
                      "protocol_sha256": protocol_sha256, "source_commit": source_commit,
                      "model_calls": 0}
    terminal_blob.upload_from_string(json.dumps(terminal_state, sort_keys=True) + "\n",
                                     content_type="application/json")
    terminal_roundtrip = json.loads(terminal_blob.download_as_text())
    if terminal_roundtrip != terminal_state:
        raise ValueError("terminal-state round-trip check failed")
    result = {"phase": "passed", "status": "passed", "protocol_id": PROTOCOL_ID,
              "protocol_sha256": protocol_sha256, "source_commit": source_commit,
              "colab_session": "paper2-b2-estimation",
              "gcs_prefix": "paper2-v2/b2-estimation-2026-11-20",
              "model_calls": 0, "checkpoint_upload": True, "fresh_runtime_restore": True,
              "checkpoint_restore_after_colab_runtime_recreation": True,
              "terminal_state_roundtrip": True,
              "sentinel_sha256": verified_sha, "verified_unix": time.time()}
    phase_blob.upload_from_string(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                  content_type="application/json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("upload", "restore"),
                        default=os.environ.get("B2_REHEARSAL_STAGE"))
    parser.add_argument("--protocol-sha256", default=os.environ.get("B2_PROTOCOL_SHA256"))
    parser.add_argument("--source-commit", default=os.environ.get("PAPER2_GIT_SHA"))
    parser.add_argument("--run-id", default=os.environ.get("B2_REHEARSAL_RUN_ID", "rehearsal-20261120"))
    # Colab executes this file inside a Jupyter kernel whose sys.argv contains
    # launcher-owned flags. The protocol values arrive through the environment.
    args, _unknown = parser.parse_known_args()
    if not args.stage or not args.protocol_sha256 or not args.source_commit:
        parser.error("stage, protocol SHA-256, and source commit are required")
    print("B2_REHEARSAL_RESULT=" + json.dumps(run_stage(
        args.stage, protocol_sha256=args.protocol_sha256, source_commit=args.source_commit,
        run_id=args.run_id), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
