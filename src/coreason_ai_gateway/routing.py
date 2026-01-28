# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from fastapi import HTTPException, status


def resolve_provider_path(model: str) -> str:
    """
    Resolves the Vault secret path based on the requested model name.

    Args:
        model (str): The model identifier (e.g., 'gpt-4o', 'claude-3-opus').

    Returns:
        str: The path suffix for the secret in Vault (e.g., 'infrastructure/openai').

    Raises:
        HTTPException: If the model is not supported (400 Bad Request).
    """
    if model.startswith(("gpt-", "o1-")):
        return "infrastructure/openai"
    elif model.startswith("claude-"):
        return "infrastructure/anthropic"

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported model architecture")
