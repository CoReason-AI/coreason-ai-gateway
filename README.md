# coreason-ai-gateway

**The Central GenAI Security & Egress Proxy**

[![License](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason_ai_gateway/blob/main/LICENSE)
[![CI/CD](https://github.com/CoReason-AI/coreason_ai_gateway/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/CoReason-AI/coreason_ai_gateway/actions/workflows/ci-cd.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs](https://img.shields.io/badge/docs-Product%20Requirements-informational)](docs/product_requirements.md)

`coreason-ai-gateway` is a high-performance, asynchronous FastAPI microservice acting as the **sole egress point** for all Generative AI traffic within the CoReason ecosystem. It implements a "Hollow Proxy" architecture to enforce security, budget, and resilience policies.

## Features

-   **Shared Nothing Policy:** Designed with zero circular dependencies and strict isolation.
-   **Security Model:** No static secrets. API keys are injected Just-In-Time (JIT) via `coreason-vault`.
-   **Budget Gate:** Real-time financial authorization using Redis to prevent "Token Burn".
-   **Accounting:** Asynchronous usage tracking and reporting.
-   **Resilience:** Built-in retries (Exponential Backoff) and circuit breaking using `tenacity`.
-   **Standard Interface:** Fully compatible with the OpenAI Chat Completion API (v1).

## Installation

### Standard Installation
```bash
pip install coreason-ai-gateway
```

### Development Setup (Poetry)
```bash
git clone https://github.com/CoReason-AI/coreason_ai_gateway.git
cd coreason_ai_gateway
poetry install
```

## Usage

The gateway is designed to be a drop-in replacement for direct provider calls. Configure your OpenAI client to point to the gateway.

```python
import os
from openai import OpenAI

# Initialize the client pointing to the Gateway
# Ensure GATEWAY_ACCESS_TOKEN is available
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="gate_..." # Your GATEWAY_ACCESS_TOKEN
)

# Make a request (Gateway handles Auth, Budget, and Provider Routing)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello world"}],
    extra_headers={
        "X-Coreason-Project-ID": "proj_123",
        "X-Coreason-Trace-ID": "abc-123"
    }
)

print(response.choices[0].message.content)
```
