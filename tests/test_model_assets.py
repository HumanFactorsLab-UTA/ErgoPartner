import hashlib

import pytest

from core.model_assets import ModelIntegrityError, verify_sha256


def test_model_checksum_verification(tmp_path):
    artifact = tmp_path / "model.task"
    artifact.write_bytes(b"pinned model")
    expected = hashlib.sha256(b"pinned model").hexdigest()
    verify_sha256(artifact, expected)


@pytest.mark.parametrize("expected", [None, "not-a-hash", "0" * 64])
def test_missing_invalid_or_mismatched_model_checksum_fails(tmp_path, expected):
    artifact = tmp_path / "model.task"
    artifact.write_bytes(b"model")
    with pytest.raises(ModelIntegrityError):
        verify_sha256(artifact, expected)
