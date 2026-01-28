# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import os
from typing import Any, Literal

from pydantic import AnyHttpUrl, AnyUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Strictly forbids static API keys for LLM providers to enforce security policies.

    Attributes:
        ENV (str): The deployment environment (development, testing, production).
        LOG_LEVEL (str): The logging level (default: INFO).
        VAULT_ADDR (AnyHttpUrl): The address of the HashiCorp Vault instance.
        VAULT_ROLE_ID (str): The AppRole ID for Vault authentication.
        VAULT_SECRET_ID (SecretStr): The AppRole Secret ID for Vault authentication.
        REDIS_URL (AnyUrl): The connection string for the Redis budget store.
        GATEWAY_ACCESS_TOKEN (SecretStr): The shared secret token for internal service authentication.
        RETRY_STOP_AFTER_ATTEMPT (int): Max retry attempts for upstream calls.
        RETRY_STOP_AFTER_DELAY (int): Max time to wait for retries.
        RETRY_WAIT_MIN (int): Minimum wait time between retries.
        RETRY_WAIT_MAX (int): Maximum wait time between retries.
    """

    # Core
    ENV: Literal["development", "testing", "production"] = "production"
    LOG_LEVEL: str = "INFO"

    # Infrastructure
    VAULT_ADDR: AnyHttpUrl
    VAULT_ROLE_ID: str
    VAULT_SECRET_ID: SecretStr
    REDIS_URL: AnyUrl

    # Security
    GATEWAY_ACCESS_TOKEN: SecretStr

    # Resilience
    RETRY_STOP_AFTER_ATTEMPT: int = 3
    RETRY_STOP_AFTER_DELAY: int = 10
    RETRY_WAIT_MIN: int = 2
    RETRY_WAIT_MAX: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def check_forbidden_keys(cls, data: Any) -> Any:
        """
        Ensures that no static API keys are present in the environment.
        This enforces the 'Shared Nothing' policy and 'No Static Secrets' rule.

        Args:
            data (Any): The raw environment data to validate.

        Returns:
            Any: The validated data if no forbidden keys are found.

        Raises:
            ValueError: If any forbidden keys (e.g., OPENAI_API_KEY) are present.
        """
        forbidden_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        for key in forbidden_keys:
            if key in os.environ:
                raise ValueError(
                    f"Security Violation: '{key}' found in environment variables. "
                    "Static secrets are strictly forbidden. Use Vault."
                )
        return data


# Instantiate settings
# This will raise validation error if required env vars are missing.
# We defer instantiation to runtime or testing setup if needed,
# but for a simple app pattern, global instantiation is common.
# However, to avoid import errors during test collection if envs are missing,
# we can use a lru_cache or just let it fail if running the app.
# For now, we will expose the class primarily.


def get_settings() -> Settings:
    return Settings()
