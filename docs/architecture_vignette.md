# The Architecture and Utility of coreason-ai-gateway

### 1. The Philosophy (The Why)

In the rapidly evolving landscape of Generative AI, the "token" has become a new unit of currency, and the API key a set of keys to the kingdom. A standard distributed microservices architecture typically scatters these credentials across various services—`coreason-cortex`, `coreason-prism`—creating a sprawling surface area for security leaks and financial "token burn."

The `coreason-ai-gateway` was born from a singular insight: **Internal services should not hold the keys.**

Instead, we implement a "Hollow Proxy" architecture. This service is designed to be stateless and ephemeral. It holds no data, stores no static secrets, and trusts no internal actor implicitly. By centralizing the egress point for all LLM traffic, we achieve two critical goals:
1.  **Just-In-Time Security:** Credentials are injected from a secure vault only for the milliseconds they are needed to initiate a connection, then immediately discarded.
2.  **Financial Gravity:** By placing a strict "Budget Gate" before the first byte is ever sent upstream, we prevent runaway processes from draining resources, enforcing hard fiscal limits in real-time.

This package is not just a proxy; it is the financial and security conscience of the CoReason AI ecosystem.

### 2. Under the Hood (The Dependencies & logic)

The `coreason-ai-gateway` is built on a high-performance, asynchronous Python stack designed for I/O-bound workloads—specifically, long-lived LLM streaming connections.

*   **FastAPI & Uvicorn:** The backbone of the service. Its native support for Python's `async`/`await` pattern allows the gateway to handle thousands of concurrent streaming connections with minimal overhead, a necessity when proxying slow upstream providers.
*   **Redis:** Serves as the high-speed state store for our budgeting logic. We utilize atomic operations (like `DECRBY`) to ensure that even in a highly concurrent environment, two requests cannot simultaneously spend the last dollar of a budget.
*   **Coreason-Vault:** This internal dependency is the key to our "Shared Nothing" security model. It allows the application to authenticate via AppRole and fetch provider credentials (OpenAI, Anthropic) dynamically at runtime, ensuring no static API keys ever exist in environment variables or code.
*   **Tenacity:** LLM providers are notoriously flaky. We wrap our upstream calls in sophisticated retry logic (exponential backoff) to absorb transient failures (HTTP 429, 502) before they propagate to the calling service.
*   **OpenAI SDK:** We leverage the standard library not just for connectivity, but for schema validation, ensuring our "Hollow Proxy" remains fully compliant with the industry-standard interface.

**The Logic Flow:**
The request lifecycle is a study in defensive programming. An incoming request must first pass the **Auth Gate** (verifying the internal JWT), then the **Budget Gate** (checking Redis). Only then does the **Router** determine the model destination, fetch the specific credential from Vault, and instantiate an ephemeral client. This client executes the request, streams the response, and—in a decoupled background task—asynchronously updates the accounting ledger in Redis.

### 3. In Practice (The How)

The true utility of `coreason-ai-gateway` is its transparency. To an internal developer, it looks exactly like the standard OpenAI API, just with a different address.

#### Example 1: The Standard "Happy Path"

Here, a service like `coreason-cortex` initiates a chat completion. Note the custom `base_url` and the injection of the `X-Coreason-Project-ID` header, which ties this request to a specific financial budget.

```python
import os
from openai import AsyncOpenAI

# The gateway acts as a drop-in replacement
client = AsyncOpenAI(
    base_url="http://ai-gateway:8000/v1",
    api_key=os.getenv("GATEWAY_ACCESS_TOKEN"),  # Internal shared secret
)

async def generate_thought():
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Explain quantum entanglement."}],
        # Critical: Tie this request to a specific project budget
        extra_headers={
            "X-Coreason-Project-ID": "research-division-alpha",
            "X-Coreason-Trace-ID": "uuid-1234-5678"
        }
    )
    print(response.choices[0].message.content)
```

#### Example 2: Streaming with Automatic Accounting

The gateway fully supports Server-Sent Events (SSE). Even when streaming, the gateway tracks token usage. It accumulates the usage stats from the stream chunks and fires the accounting background task only after the stream concludes.

```python
async def stream_thought():
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a long poem about rust."}],
        stream=True,
        extra_headers={"X-Coreason-Project-ID": "creative-writing-beta"}
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    # Behind the scenes, the gateway has now asynchronously decremented
    # the 'creative-writing-beta' budget in Redis.
```
