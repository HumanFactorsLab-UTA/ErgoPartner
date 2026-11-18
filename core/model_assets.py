"""Model artifact integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


class ModelIntegrityError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        raise ModelIntegrityError("A pinned model SHA-256 is required")
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ModelIntegrityError("Model SHA-256 must be 64 hexadecimal characters")
    actual = sha256_file(path)
    if actual != expected:
        raise ModelIntegrityError(
            f"Model checksum mismatch for {Path(path).resolve()}: expected {expected}, got {actual}"
        )
