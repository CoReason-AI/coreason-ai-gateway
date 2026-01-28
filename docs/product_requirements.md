# Business Requirement Document (BRD)

## **Project:** `coreason-ai-gateway`

### **Role:** The Central GenAI Security & Egress Proxy


---

## 1. Executive Summary

**Objective:** Build `coreason-ai-gateway`, a high-performance, asynchronous FastAPI microservice acting as the **sole egress point** for all Generative AI traffic within the CoReason ecosystem.

**The Problem:** Internal services (`coreason-cortex`, `coreason-prism`) must never hold sensitive provider API keys (e.g., OpenAI, Anthropic). Furthermore, unconstrained agents can cause runaway costs ("Token Burn").

**The Solution:** A "Hollow Proxy" architecture that:

1. **Authenticates** internal microservices.
2. **Authorizes** requests against a dynamic monetary budget (via Redis).
3. **Injects** provider credentials Just-In-Time (via `coreason-vault`).
4. **Proxies** the request to the upstream provider (OpenAI, vLLM, etc.).

---

## 2. Architectural Constraints (Strict)

### 2.1 "Shared Nothing" Policy

* **Dependencies:** This package **MUST NOT** import `coreason-api`, `coreason-cortex`, or `coreason-manifest`. It must have **zero** circular dependencies.
* **Allowed Internal Dependency:** It is explicitly authorized to import `coreason-vault` for secret retrieval. Coreason-vault is pip installable and version controlled.

### 2.2 Security Model

* **No Static Secrets:** `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` must **NEVER** appear in `os.environ`, `config.py`, or `Dockerfile`.
* **Ephemeral Memory:** Secrets are fetched from Vault, used to instantiate a client for *one request*, and immediately garbage collected.
* **Privileged Access:** This is the only service in the cluster running with the `VAULT_ROLE_ID` capable of reading root provider secrets.

### 2.3 Interface Standard

* **Input/Output:** Must strictly adhere to the **OpenAI Chat Completion API** (v1) specification.
* **Client Compatibility:** Internal services must be able to use the standard Python `openai` library by changing `base_url` to this gateway.

---

## 3. Functional Requirements

### FR-01: The Gateway Endpoint

* **Path:** `POST /v1/chat/completions`
* **Input Schema:** Standard OpenAI Request Body (messages, model, temperature, tools).
* **Headers Required:**
* `Authorization`: `Bearer <GATEWAY_ACCESS_TOKEN>` (Shared internal secret).
* `X-Coreason-Project-ID`: String (Used for budget tracking).
* `X-Coreason-Trace-ID`: UUID (For distributed tracing).



### FR-02: The "Budget Gate" (Middleware Phase 1)

Before processing any logic, the system must check financial authorization.

1. **Token Estimation:** Perform a fast heuristic estimation of input tokens (e.g., `len(json_body) / 4`).
2. **Redis Check:** Query Redis key `budget:{project_id}`.
* *Logic:* `IF (current_spend + estimated_cost) > hard_limit THEN Reject`.


3. **Rejection:** Return HTTP `402 Payment Required` immediately if budget is insufficient.

### FR-03: Just-In-Time Secret Injection (Middleware Phase 2)

1. **Model Routing:** Inspect the `model` field in the JSON body.
* *Map:* `gpt-*` -> `infrastructure/openai`
* *Map:* `claude-*` -> `infrastructure/anthropic` (Future scope, strictly structured for now).


2. **Vault Retrieval:** Use `app.state.vault_client` to fetch the specific API key for that route.
3. **Client Instantiation:** Create an **ephemeral** `AsyncOpenAI` client using this specific key.

### FR-04: Upstream Execution & Resilience (Middleware Phase 3)

1. **Forwarding:** Send the request to the upstream provider.
2. **Retry Logic:** Use `tenacity` library.
* *Retry on:* HTTP 429 (Rate Limit), HTTP 5xx (Server Error).
* *Stop after:* 3 attempts or 10 seconds.
* *Backoff:* Exponential.


3. **Streaming:** Support Server-Sent Events (SSE) if `stream=True` in request.

### FR-05: Accounting (Post-Response)

1. **Usage Extraction:** Extract `usage.total_tokens` (or stream equivalent) from the upstream response.
2. **Async Decoupling:** Fire-and-forget a background task to update Redis:
* `INCRBY usage:{project_id} <token_count>`



---

## 4. Technical Specifications

### 4.1 Dependency Stack (`pyproject.toml`)

* `fastapi`, `uvicorn[standard]` (Server)
* `openai` (For typed schemas and client construction)
* `tenacity` (Resilience)
* `redis` (Budgeting state)
* `coreason-vault` (Secret Management)
* `pydantic-settings` (Config)

### 4.2 Configuration (`config.py`)

| Variable | Description | Required | Source |
| --- | --- | --- | --- |
| `GATEWAY_ACCESS_TOKEN` | Secret token internal apps must present | Yes | Env |
| `REDIS_URL` | Connection string for budget store | Yes | Env |
| `VAULT_ADDR` | HashiCorp Vault address | Yes | Env |
| `VAULT_ROLE_ID` | AppRole ID for Auth | Yes | Env |
| `VAULT_SECRET_ID` | AppRole Secret for Auth | Yes | Env |

### 4.3 Directory Structure

```text
src/coreason_ai_gateway/
├── server.py           # Main FastAPI app & Lifecycle logic
├── routing.py          # Logic to map "model name" to "vault path"
├── middleware/
│   ├── auth.py         # Validates GATEWAY_ACCESS_TOKEN
│   ├── budget.py       # Checks Redis limits
│   └── accounting.py   # Updates Redis usage after success
└── config.py           # Pydantic settings

```

---

## 5. Implementation Logic (Pseudocode for LLM)

**Class: GatewayServer**

```python
@app.post("/v1/chat/completions")
async def proxy_request(
    request: Request,
    body: ChatCompletionRequest,
    background_tasks: BackgroundTasks
):
    # 1. Validation
    token = request.headers.get("Authorization")
    validate_internal_token(token)

    # 2. Budget Check (Fail Fast)
    project_id = request.headers.get("X-Coreason-Project-ID")
    if not budget_manager.check_availability(project_id):
        raise HTTPException(status_code=402, detail="Budget Exceeded")

    # 3. Secret Retrieval (JIT)
    provider_path = router.resolve_provider(body.model) # e.g., "openai"
    secret = await vault.get_secret(f"infrastructure/{provider_path}")

    # 4. Ephemeral Client
    # NOTE: Client is created AND destroyed inside this scope
    async with AsyncOpenAI(api_key=secret['value']) as client:
        try:
            # 5. Upstream Call (with Retry)
            response = await client.chat.completions.create(**body.model_dump())

            # 6. Accounting (Background)
            background_tasks.add_task(
                budget_manager.record_usage,
                project_id,
                response.usage
            )

            return response

        except RateLimitError:
            raise HTTPException(status_code=429)
        except APIError as e:
            raise HTTPException(status_code=502, detail=str(e))

```

---

## 6. Definition of Done (DoD)

1. **Strict Isolation:** The codebase has imports *only* from `fastapi`, `pydantic`, `openai`, `redis`, `tenacity`, and `coreason-vault`.
2. **Security Verification:** A grep of the codebase confirms NO static API keys are defined.
3. **Output Verification:** The API response is perfectly indistinguishable from a direct OpenAI response (same JSON structure).
4. **Error Handling:** Sending a request with an exhausted Project ID returns `402`. Sending a request to a down provider returns `5xx` (not crashing the gateway).
# Functional Requirement Document (FRD)

## **Component:** `coreason-ai-gateway`

**Target Audience:** Autonomous Coding Agent / Lead Engineer
**Version:** 1.0.0

---

## 1. System Overview

The `coreason-ai-gateway` is a stateless, asynchronous FastAPI microservice. It serves as the **secure proxy** for all outbound Large Language Model (LLM) requests. It decouples internal services from external provider credentials and enforces financial governance.

**Core Responsibility:**
Receive an OpenAI-compatible request  Validate Budget  Fetch Secret (Just-In-Time)  Execute Upstream  Record Usage.

---

## 2. Interface Specifications

### 2.1 API Endpoint Definition

The service must expose a **single primary endpoint** that mirrors the OpenAI Chat Completion API.

* **Method:** `POST`
* **Path:** `/v1/chat/completions`
* **Content-Type:** `application/json`

**Request Headers:**
| Header | Requirement | Validation Logic |
| :--- | :--- | :--- |
| `Authorization` | Mandatory | Must match `Bearer <GATEWAY_ACCESS_TOKEN>` from config. |
| `X-Coreason-Project-ID` | Mandatory | Non-empty string. Used as the Redis key suffix for budgeting. |
| `X-Coreason-Trace-ID` | Optional | UUID. Logged for distributed tracing. |

**Request Body Schema:**

* Must utilize `openai.types.chat.ChatCompletionRequest` (or equivalent Pydantic model) to validate standard fields: `model`, `messages`, `temperature`, `stream`, `tools`.

**Response Schema:**

* Must return standard `openai.types.chat.ChatCompletion` JSON.
* If `stream=True`, must return a `StreamingResponse` (Server-Sent Events).

---

## 3. Detailed Logic Specifications

### 3.1 Configuration (`src/coreason_ai_gateway/config.py`)

Implement a `Settings` class using `pydantic_settings`.

**Environment Variables:**

1. `VAULT_ADDR` (Url): Address of the HashiCorp Vault.
2. `VAULT_ROLE_ID` (Str): AppRole ID for Vault authentication.
3. `VAULT_SECRET_ID` (Str): AppRole Secret for Vault authentication.
4. `REDIS_URL` (Url): Connection string for the Redis budget store.
5. `GATEWAY_ACCESS_TOKEN` (Str): The shared secret for internal service authentication.

**Constraint:** The `Settings` class must **strictly exclude** any fields named `OPENAI_API_KEY`, `ANTHROPIC_KEY`, or similar.

### 3.2 Application Lifespan (`src/coreason_ai_gateway/server.py`)

Use FastAPI's `@asynccontextmanager`.

**Startup Sequence:**

1. **Initialize Vault:** Instantiate `VaultManagerAsync` (from `coreason-vault`). Call `.authenticate()` immediately using the Role/Secret IDs from config.
2. **Initialize Redis:** Create an asynchronous `redis.Redis` client using `REDIS_URL`.
3. **State Storage:** Store both clients in `app.state.vault` and `app.state.redis`.

**Shutdown Sequence:**

1. Close Redis connection.
2. Close Vault client session.

### 3.3 Request Processing Pipeline (Middleware Logic)

The endpoint implementation must follow this **strict execution order**:

#### Phase 1: Authorization & Validation

* **Action:** Check `Authorization` header.
* **Failure:** If invalid, return HTTP `401 Unauthorized`.

#### Phase 2: Budget Gate (The "Check")

* **Input:** `X-Coreason-Project-ID` header, Request JSON body.
* **Heuristic:** Calculate `estimated_tokens = len(json.dumps(messages)) / 4`.
* **Redis Check:**
* Key: `budget:{project_id}:remaining`
* Logic: `GET` the key. If `value < estimated_tokens`, return HTTP `402 Payment Required`.
* *Note:* If key does not exist, assume default budget (fail-open or fail-closed based on policy—default to **fail-closed** for security).



#### Phase 3: Secret Retrieval (The "Fetch")

* **Input:** `model` field from Request JSON (e.g., `"gpt-4o"`).
* **Routing Logic:**
* If `model` starts with `"gpt"`, target path: `secret/infrastructure/openai`.
* If `model` starts with `"claude"`, target path: `secret/infrastructure/anthropic` (Future proofing).


* **Vault Action:** `await app.state.vault.get_secret(path)`.
* **Extraction:** Get the raw API key string from the secret dictionary.

#### Phase 4: Upstream Execution (The "Proxy")

* **Client Setup:** Instantiate `AsyncOpenAI` client.
* `api_key`: The value fetched in Phase 3.
* `base_url`: Default (for OpenAI) or custom (if internal vLLM).


* **Execution:** Call `client.chat.completions.create(**request_body)`.
* **Resilience:** Wrap this call in a `tenacity` retry decorator.
* Retry on: `openai.RateLimitError`, `openai.APIConnectionError`, `openai.InternalServerError`.
* Max Attempts: 3.


* **Scope:** The client and API key variable must go out of scope immediately after this block to ensure garbage collection.

#### Phase 5: Accounting (The "Record")

* **Mechanism:** FastAPI `BackgroundTasks`.
* **Action:** Trigger `record_usage(project_id, usage_object)` after response is sent.
* **Redis Update:**
* `DECRBY budget:{project_id}:remaining <total_tokens_used>`
* `INCRBY usage:{project_id}:total <total_tokens_used>`



---

## 4. Error Handling Specifications

The API must map internal failures to standard HTTP codes to ensure the client (`coreason-cortex`) can handle them gracefully.

| Scenario | HTTP Code | Error Message Structure |
| --- | --- | --- |
| Invalid Auth Token | 401 | `{"detail": "Invalid Gateway Access Token"}` |
| Insufficient Budget | 402 | `{"detail": "Budget exceeded for Project ID <ID>"}` |
| Unknown Model | 400 | `{"detail": "Unsupported model architecture"}` |
| Vault Auth Failure | 503 | `{"detail": "Security subsystem unavailable"}` |
| Upstream Rate Limit | 429 | `{"detail": "Upstream provider rate limit exceeded"}` |
| Upstream Server Error | 502 | `{"detail": "Upstream provider error: <msg>"}` |

---

## 5. Security & Compliance Checklist for Implementation

1. **Zero Persistence:** The code must **never** write the fetched API key to logs (stdout/stderr) or disk.
2. **Sanitization:** If logging the request body, sensitive PII must be scrubbed (though for this initial version, standard debug logging is disabled in production).
3. **Dependencies:**
* `coreason-vault` is the **only** permitted internal dependency.
* No direct imports from `coreason-api` or `coreason-manifest`.



## 6. Testing Strategy

* **Unit Tests:** Mock `app.state.vault` and `app.state.redis`. Verify that the gateway correctly routes requests and rejects low budget.
* **Integration Tests:** Use `respx` to mock the OpenAI API. Verify that headers are passed correctly and retries trigger on 500 errors.


# Technical Requirement Document (TRD)

## **Component:** `coreason-ai-gateway`

**Role:** GenAI Egress Proxy & Security Enforcement Point
**Stack:** Python 3.12, FastAPI, Redis, HashiCorp Vault
**Version:** 1.0.0

---

## 1. Component Architecture

### 1.1 Dependency Graph

The service acts as a **leaf node** in the dependency tree to prevent circular references.

* **Upstream (Imports):**
* `fastapi` (Web Framework)
* `uvicorn` (ASGI Server)
* `openai` (SDK & Types)
* `redis` (Async Client)
* `tenacity` (Retry Logic)
* `coreason-vault` (Internal: Secret Management)


* **Downstream (Used By):**
* `coreason-cortex` (via HTTP)
* `coreason-prism` (via HTTP)


* **Forbidden Imports:** `coreason-api`, `coreason-manifest`, `coreason-ledger`.

### 1.2 Data Flow

1. **Inbound:** HTTP POST `/v1/chat/completions` (OpenAI Schema).
2. **Gate 1 (Auth):** Validate Bearer Token (Stateless).
3. **Gate 2 (Budget):** Read Redis `budget:{project_id}` (Async).
4. **Fetch (Secret):** Read Vault `secret/infrastructure/{provider}` (Async/Cached).
5. **Outbound:** HTTP POST to Provider (e.g., `api.openai.com`).
6. **Accounting:** Write Redis `usage:{project_id}` (Background Task).

---

## 2. Implementation Specifications

### 2.1 Configuration Module (`src/coreason_ai_gateway/config.py`)

Implement `Settings` inheriting from `pydantic_settings.BaseSettings`.

| Variable | Type | Default | Constraint |
| --- | --- | --- | --- |
| `ENV` | Str | `production` | - |
| `LOG_LEVEL` | Str | `INFO` | - |
| `VAULT_ADDR` | AnyHttpUrl | - | Required |
| `VAULT_ROLE_ID` | Str | - | Required |
| `VAULT_SECRET_ID` | SecretStr | - | Required |
| `REDIS_URL` | AnyUrl | - | Required |
| `GATEWAY_ACCESS_TOKEN` | SecretStr | - | Required (Shared Secret) |

**Validation Rule:** Raise `ValueError` immediately if `OPENAI_API_KEY` is detected in environment variables.

### 2.2 Server Entrypoint (`src/coreason_ai_gateway/server.py`)

#### A. Lifespan Manager (`@asynccontextmanager`)

Must manage connection pools to avoid resource exhaustion.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Setup Redis
    app.state.redis = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)

    # 2. Setup Vault
    app.state.vault = VaultManagerAsync(url=settings.VAULT_ADDR)
    await app.state.vault.authenticate(role_id=..., secret_id=...)

    yield

    # 3. Teardown
    await app.state.redis.close()
    await app.state.vault.close()

```

#### B. The Proxy Endpoint

**Signature:**

```python
@app.post("/v1/chat/completions", status_code=200)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest, # Pydantic model from openai or custom mirror
    background_tasks: BackgroundTasks
) -> ChatCompletionResponse:

```

### 2.3 Logic Pipeline (Sequential Execution)

#### Step 1: Security Validation

* Extract `Authorization` header.
* Compare constant-time: `secrets.compare_digest(token, settings.GATEWAY_ACCESS_TOKEN.get_secret_value())`.
* **Error:** 401 Unauthorized.

#### Step 2: Budget Authorization

* **Heuristic:** `estimated_cost = len(str(body.messages)) // 4`.
* **Redis Key:** `budget:{header.x_coreason_project_id}:remaining`.
* **Operation:** `await app.state.redis.get(key)`.
* **Logic:**
* If `key` is None: Assume 0 (Fail Secure).
* If `int(value) < estimated_cost`: Raise 402 Payment Required.



#### Step 3: Secret Routing

* **Map:**
* `gpt-*`, `o1-*`  `secret/infrastructure/openai`
* `claude-*`  `secret/infrastructure/anthropic`


* **Fetch:** `secret_data = await app.state.vault.get_secret(path)`.
* **Extract:** `api_key = secret_data["api_key"]`.

#### Step 4: Upstream Execution (The Critical Path)

* **Isolation:** Instantiate `AsyncOpenAI` client *inside* the route handler.
* **Retry Policy (`tenacity`):**
* `stop=stop_after_attempt(3)`
* `wait=wait_exponential(multiplier=1, min=2, max=10)`
* `retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError))`


* **Call:** `response = await client.chat.completions.create(...)`
* **Streaming:** If `body.stream` is True, return `StreamingResponse`.

#### Step 5: Asynchronous Accounting

* **Trigger:** Add to `background_tasks`.
* **Function:** `update_usage(project_id: str, tokens: int)`.
* **Redis Operations (Pipeline):**
```python
async with redis.pipeline() as pipe:
    pipe.decrby(f"budget:{pid}:remaining", tokens)
    pipe.incrby(f"usage:{pid}:total", tokens)
    await pipe.execute()

```



---

## 3. Data Schema & Redis Layout

### 3.1 Redis Keys

| Key Pattern | Type | Purpose | TTL |
| --- | --- | --- | --- |
| `budget:{project_id}:remaining` | String (Int) | Hard token limit countdown | None |
| `usage:{project_id}:total` | String (Int) | Accumulator for auditing | None |

### 3.2 Vault Paths

| Path | Key | Value Format |
| --- | --- | --- |
| `secret/infrastructure/openai` | `api_key` | `sk-...` |
| `secret/infrastructure/anthropic` | `api_key` | `sk-ant...` |

---

## 4. Docker & Deployment Requirements

### 4.1 Dockerfile

* **Base:** `python:3.12-slim-bookworm` (Minimizes CVEs).
* **User:** Create non-root `appuser` (UID 1000).
* **Env:** `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`.
* **Healthcheck:**
* `CMD curl -f http://localhost:8000/health || exit 1`


* **Security:**
* Do not copy `.env` files.
* Use `pip install --no-cache-dir`.



### 4.2 Logging

* Format: JSON (using `loguru` or `python-json-logger`).
* **Redaction:** Never log `request.headers["Authorization"]` or `body.messages` (contains proprietary prompts).

---

## 5. Testing Strategy

### 5.1 Mocking

* **Vault:** Mock `VaultManagerAsync.get_secret` to return dummy keys `{"api_key": "sk-dummy"}`.
* **Redis:** Use `fakeredis` or mock `app.state.redis`.
* **Upstream:** Use `respx` to intercept `https://api.openai.com/*`.

### 5.2 Scenarios

1. **Happy Path:** Valid Token + Sufficient Budget  200 OK.
2. **Budget Fail:** Valid Token + Zero Budget  402 Payment Required.
3. **Auth Fail:** Invalid Token  401 Unauthorized.
4. **Resilience:** Upstream returns 500  Gateway retries 3x  Returns 502.
