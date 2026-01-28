from typing import Any

from redis.asyncio import Redis

try:
    print("Attempting Redis[Any]")
    x = Redis[Any]
    print(f"Success: {x}")
except Exception as e:
    print(f"Failed: {e}")
