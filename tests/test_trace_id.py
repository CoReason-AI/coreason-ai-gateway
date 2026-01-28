import json
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from coreason_ai_gateway.server import app
from coreason_ai_gateway.utils.logger import logger


@pytest.fixture(autouse=True)
def setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
    monkeypatch.setenv("VAULT_ROLE_ID", "dummy-role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "dummy-secret-id")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    monkeypatch.setenv("GATEWAY_ACCESS_TOKEN", "valid-token")


@pytest.fixture
def mock_dependencies() -> Generator[dict[str, Any], None, None]:
    with (
        patch("coreason_ai_gateway.server.redis.from_url") as mock_redis,
        patch("coreason_ai_gateway.server.VaultManagerAsync") as mock_vault,
        patch("coreason_ai_gateway.server.CoreasonVaultConfig"),
        patch("coreason_ai_gateway.dependencies.AsyncOpenAI") as mock_openai,
    ):
        redis_instance = AsyncMock()
        mock_redis.return_value = redis_instance
        redis_instance.get.return_value = "1000"

        redis_instance.pipeline = MagicMock()
        pipeline_mock = MagicMock()
        redis_instance.pipeline.return_value = pipeline_mock

        async def aenter(*args: Any, **kwargs: Any) -> MagicMock:
            return pipeline_mock

        pipeline_mock.__aenter__ = AsyncMock(side_effect=aenter)
        pipeline_mock.__aexit__ = AsyncMock()
        pipeline_mock.execute = AsyncMock()

        vault_instance = AsyncMock()
        mock_vault.return_value = vault_instance
        vault_instance.authenticate = AsyncMock()
        vault_instance.get_secret.return_value = {"api_key": "sk-test"}

        openai_client = AsyncMock()
        mock_openai.return_value = openai_client

        yield {"redis": redis_instance, "vault": vault_instance, "openai": mock_openai, "client": openai_client}


@pytest.fixture
def log_capture() -> Generator[list[str], None, None]:
    logs: list[str] = []
    # Capture raw record dict to avoid serialization issues
    handler_id = logger.add(lambda msg: logs.append(msg), format="{message}", level="INFO", serialize=True)
    yield logs
    logger.remove(handler_id)


def test_trace_id_in_logs(mock_dependencies: dict[str, Any], log_capture: list[str]) -> None:
    mock_response = MagicMock()
    mock_response.usage.total_tokens = 10
    mock_response.model_dump.return_value = {"id": "123", "choices": []}
    mock_dependencies["client"].chat.completions.create.return_value = mock_response

    trace_id = "test-trace-id-12345"

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            headers={
                "Authorization": "Bearer valid-token",
                "X-Coreason-Project-ID": "proj-1",
                "X-Coreason-Trace-ID": trace_id,
            },
        )

    found_trace = False
    debug_logs = []
    for log_msg in log_capture:
        try:
            log_entry = json.loads(log_msg)
            extra = log_entry.get("record", {}).get("extra", {})

            debug_logs.append(log_entry)

            if extra.get("trace_id") == trace_id:
                found_trace = True
        except Exception:
            pass

    assert found_trace, f"Trace ID {trace_id} not found in logs. Captured logs: {json.dumps(debug_logs, indent=2)}"
