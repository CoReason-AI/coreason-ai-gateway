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
from coreason_ai_gateway.routing import resolve_provider_path
from fastapi import HTTPException


def test_resolve_provider_path_openai() -> None:
    assert resolve_provider_path("gpt-4o") == "infrastructure/openai"
    assert resolve_provider_path("gpt-3.5-turbo") == "infrastructure/openai"
    assert resolve_provider_path("o1-preview") == "infrastructure/openai"
    assert resolve_provider_path("o1-mini") == "infrastructure/openai"


def test_resolve_provider_path_anthropic() -> None:
    assert resolve_provider_path("claude-3-opus") == "infrastructure/anthropic"
    assert resolve_provider_path("claude-3.5-sonnet") == "infrastructure/anthropic"


def test_resolve_provider_path_invalid() -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_provider_path("llama-3-70b")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unsupported model architecture"

    with pytest.raises(HTTPException) as exc_info:
        resolve_provider_path("gemini-pro")

    assert exc_info.value.status_code == 400
