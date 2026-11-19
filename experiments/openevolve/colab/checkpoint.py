"""Resumable file mirroring for long Colab experiments."""
from __future__ import annotations

import fcntl
import hashlib
from pathlib import Path, PurePosixPath


class GcsMirror:
    """Mirror a local directory below one fixed object prefix in Cloud Storage."""

    def __init__(self, bucket_name: str, prefix: str, local_root: Path, *,
                 credentials_path: Path | None = None, client=None):
        self.prefix = prefix.strip("/")
        if not self.prefix:
            raise ValueError("a non-empty GCS prefix is required")
        self.prefix += "/"
        self.local_root = Path(local_root).resolve()
        lock_id = hashlib.sha256(str(self.local_root).encode()).hexdigest()[:16]
        self.lock_path = Path("/tmp") / f"oe-gcs-sync-{lock_id}.lock"
        if client is None:
            from google.cloud import storage

            if credentials_path is None:
                client = storage.Client()
            else:
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(str(credentials_path))
                client = storage.Client(project=credentials.project_id, credentials=credentials)
        self.bucket = client.bucket(bucket_name)

    def sync(self) -> int:
        """Upload all regular files under the root and return the file count."""
        if not self.local_root.exists():
            return 0
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            count = 0
            for path in sorted(self.local_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.local_root).as_posix()
                if relative.endswith(".json"):
                    try:
                        import json
                        json.loads(path.read_text())
                    except (OSError, ValueError):
                        continue
                if relative.endswith(".jsonl"):
                    try:
                        data = path.read_bytes()
                    except OSError:
                        continue
                    if data and not data.endswith(b"\n"):
                        data = data[:data.rfind(b"\n") + 1] if b"\n" in data else b""
                    blob = self.bucket.blob(self.prefix + relative)
                    blob.upload_from_string(data)
                else:
                    self.bucket.blob(self.prefix + relative).upload_from_filename(str(path))
                count += 1
            fcntl.flock(lock, fcntl.LOCK_UN)
            return count

    def restore(self) -> int:
        """Download objects under this prefix, refusing paths outside the root."""
        self.local_root.mkdir(parents=True, exist_ok=True)
        count = 0
        for blob in self.bucket.list_blobs(prefix=self.prefix):
            relative_name = blob.name[len(self.prefix):]
            relative = PurePosixPath(relative_name)
            if not relative_name or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"GCS object path escapes mirror root: {blob.name}")
            destination = (self.local_root / Path(*relative.parts)).resolve()
            if not destination.is_relative_to(self.local_root):
                raise ValueError(f"GCS object path escapes mirror root: {blob.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination))
            count += 1
        return count
