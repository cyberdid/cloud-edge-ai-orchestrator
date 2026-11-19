from pathlib import Path

import pytest

from experiments.openevolve.colab.checkpoint import GcsMirror


class MemoryBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def upload_from_filename(self, filename):
        self.bucket.objects[self.name] = Path(filename).read_bytes()

    def upload_from_string(self, data):
        self.bucket.objects[self.name] = data.encode() if isinstance(data, str) else data

    def download_to_filename(self, filename):
        Path(filename).write_bytes(self.bucket.objects[self.name])


class MemoryBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return MemoryBlob(self, name)

    def list_blobs(self, prefix):
        return [MemoryBlob(self, name) for name in self.objects if name.startswith(prefix)]


class MemoryClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        return self._bucket


def test_mirror_restores_files_from_its_prefix(tmp_path):
    bucket = MemoryBucket()
    source = tmp_path / "source"
    source.mkdir()
    (source / "calls.jsonl").write_text('{"call_index":0}\n')
    (source / "nested").mkdir()
    (source / "nested" / "manifest.json").write_text('{"status":"running"}')
    mirror = GcsMirror("bucket", "paper2-v2/b1/runs", source, client=MemoryClient(bucket))

    mirror.sync()
    target = tmp_path / "restored"
    GcsMirror("bucket", "paper2-v2/b1/runs", target, client=MemoryClient(bucket)).restore()

    assert (target / "calls.jsonl").read_text() == '{"call_index":0}\n'
    assert (target / "nested" / "manifest.json").read_text() == '{"status":"running"}'
    assert set(bucket.objects) == {
        "paper2-v2/b1/runs/calls.jsonl",
        "paper2-v2/b1/runs/nested/manifest.json",
    }


def test_mirror_rejects_object_names_that_escape_local_root(tmp_path):
    bucket = MemoryBucket()
    bucket.objects["paper2-v2/b1/runs/../../escape.txt"] = b"no"
    mirror = GcsMirror("bucket", "paper2-v2/b1/runs", tmp_path / "restored", client=MemoryClient(bucket))

    with pytest.raises(ValueError, match="escapes mirror root"):
        mirror.restore()


def test_mirror_skips_incomplete_json_and_trims_partial_jsonl(tmp_path):
    bucket = MemoryBucket()
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text('{"status":"running"')
    (source / "calls.jsonl").write_text('{"call_index":0}\n{"call_index":1')
    mirror = GcsMirror("bucket", "paper2-v2/b1/runs", source, client=MemoryClient(bucket))

    mirror.sync()

    assert "paper2-v2/b1/runs/manifest.json" not in bucket.objects
    assert bucket.objects["paper2-v2/b1/runs/calls.jsonl"] == b'{"call_index":0}\n'
