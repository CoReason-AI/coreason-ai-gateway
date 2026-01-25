# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import pytest
from pydantic import ValidationError

from coreason_ai_gateway.config import get_settings


@pytest.fixture
def valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    monkeypatch.setenv("GATEWAY_ACCESS_TOKEN", "s3cr3t")


def test_settings_valid_config(valid_env: None) -> None:
    settings = get_settings()
    assert str(settings.VAULT_ADDR) == "http://vault:8200/"
    assert settings.VAULT_ROLE_ID == "role-id"
    assert settings.VAULT_SECRET_ID.get_secret_value() == "secret-id"
    assert str(settings.REDIS_URL) == "redis://redis:6379"
    assert settings.GATEWAY_ACCESS_TOKEN.get_secret_value() == "s3cr3t"


def test_settings_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    monkeypatch.delenv("VAULT_ROLE_ID", raising=False)
    monkeypatch.delenv("VAULT_SECRET_ID", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("GATEWAY_ACCESS_TOKEN", raising=False)

    with pytest.raises(ValidationError):
        get_settings()


def test_settings_forbidden_keys(valid_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-malicious-key")

    with pytest.raises(ValueError, match="Security Violation"):
        get_settings()

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key")

    with pytest.raises(ValueError, match="Security Violation"):
        get_settings()
