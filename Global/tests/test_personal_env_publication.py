"""Published environment records must retain the bytes covered by their manifest."""

import hashlib
import json
from pathlib import Path

from sixletters import experiment, personal_env


def test_publication_keeps_manifest_hashes_valid_with_windows_text_newlines(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    staged = tmp_path / "staged"
    (staged / personal_env.ENV_DIR).mkdir(parents=True)
    content = "# 연구자\npersonal-package==1.0.0\n"
    lock = "personal-package==1.0.0\n"
    baseline = {"python": "3.12", "packages": {"personal-package": "1.0.0"}, "fingerprint": "test"}
    manifest = personal_env._manifest(content, lock, baseline)
    contents = {
        personal_env.INPUT_FILE: content,
        personal_env.LOCK_FILE: lock,
        personal_env.MANIFEST_FILE: json.dumps(manifest) + "\n",
    }
    write_text = Path.write_text

    def windows_write_text(self, data, encoding=None, errors=None, newline=None):
        return write_text(
            self,
            data,
            encoding=encoding,
            errors=errors,
            newline="\r\n" if newline is None else newline,
        )

    monkeypatch.setattr(Path, "write_text", windows_write_text)
    personal_env._publish(target, staged, contents)
    resources = {name: (target / name).read_bytes() for name in contents}
    for name, digest in (
        (personal_env.INPUT_FILE, "requirements_sha256"),
        (personal_env.LOCK_FILE, "lock_sha256"),
    ):
        assert hashlib.sha256(resources[name]).hexdigest() == manifest[digest]
    experiment._validate_environment_runtime(resources, {"packages": baseline["packages"]})
