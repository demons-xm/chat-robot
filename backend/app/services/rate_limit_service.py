import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

from app.config import config

_lock = Lock()
_request_times: dict[str, deque[float]] = defaultdict(deque)


def allow_request(key: str) -> bool:
    now = time.monotonic()
    with _lock:
        calls = _request_times[key]
        while calls and now - calls[0] >= 60:
            calls.popleft()
        if len(calls) >= config.RATE_LIMIT_PER_MINUTE:
            allowed = False
        else:
            calls.append(now)
            allowed = True
        if len(_request_times) > 5000:
            expired_keys = [
                item_key
                for item_key, timestamps in _request_times.items()
                if not timestamps or now - timestamps[-1] >= 60
            ]
            for item_key in expired_keys:
                _request_times.pop(item_key, None)
        return allowed


def enforce_tenant_rate_limit(request: Request, tenant_id: str) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not allow_request(f"tenant:{tenant_id}:{client_ip}"):
        raise HTTPException(status_code=429, detail="租户请求过于频繁，请稍后重试")
