from __future__ import annotations

import pytest

from moraweave.local_teacher import LocalTeacherClient, validate_endpoint


def test_loopback_endpoint_normalization() -> None:
    assert validate_endpoint("http://127.0.0.1:11434") == "http://127.0.0.1:11434/api/chat"
    assert validate_endpoint("http://localhost:11434/api/chat") == "http://localhost:11434/api/chat"


def test_remote_and_redirect_capable_paths_are_rejected() -> None:
    with pytest.raises(ValueError):
        validate_endpoint("https://example.com/api/chat")
    with pytest.raises(ValueError):
        validate_endpoint("http://127.0.0.1:11434/other")
    with pytest.raises(ValueError):
        validate_endpoint("http://user:pass@127.0.0.1:11434/api/chat")


def test_cloud_routed_model_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        LocalTeacherClient(model="qwen:cloud")
    with pytest.raises(ValueError):
        LocalTeacherClient(model="cloud/qwen")
