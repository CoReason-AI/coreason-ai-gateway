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
    Strictly forbids static API keys for LLM providers.
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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def check_forbidden_keys(cls, data: Any) -> Any:
        """
        Ensures that no static API keys are present in the environment.
        This enforces the 'Shared Nothing' policy and 'No Static Secrets' rule.
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
