# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from coreason_ai_gateway.dependencies import get_redis_client, get_service, get_vault_client


@pytest.mark.anyio
async def test_dependencies_coverage() -> None:
    # Test error case where state is missing
    request = MagicMock(spec=Request)
    request.app.state = MagicMock()
    del request.app.state.redis
    del request.app.state.vault
    # Ensure service is also missing (MagicMock might auto-create attributes)
    if hasattr(request.app.state, "service"):
        del request.app.state.service

    with pytest.raises(RuntimeError, match="Redis client is not initialized"):
        get_redis_client(request)

    with pytest.raises(RuntimeError, match="Vault client is not initialized"):
        get_vault_client(request)

    with pytest.raises(RuntimeError, match="Service is not initialized"):
        get_service(request)

    # Success case
    request.app.state.redis = AsyncMock()
    request.app.state.vault = AsyncMock()
    request.app.state.service = AsyncMock()

    assert get_redis_client(request) is request.app.state.redis
    assert get_vault_client(request) is request.app.state.vault
    assert get_service(request) is request.app.state.service
