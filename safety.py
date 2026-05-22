"""Overload-resilient wrapper around `LLMClient.complete`.

Catches transient errors from EITHER provider (anthropic or openai) and
retries with exponential backoff + jitter, capped at 60s. Retry count comes
from SETTINGS.txt (`max_api_retries`); the default is 6.

Retried:
  - connection / timeout errors,
  - rate limits (429),
  - any 5xx (including Anthropic's 529 "Overloaded").
"""

from __future__ import annotations

import random
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm import LLMClient


_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 529}


def _provider_exception_classes() -> dict[str, dict]:
    """Lazy-import each provider's exception classes — neither package is
    required to be installed if you're using the other one."""
    out: dict[str, dict] = {"anthropic": {}, "openai": {}}
    try:
        import anthropic
        out["anthropic"] = {
            "ConnectionError": anthropic.APIConnectionError,
            "TimeoutError": anthropic.APITimeoutError,
            "RateLimitError": anthropic.RateLimitError,
            "InternalServerError": anthropic.InternalServerError,
            "APIStatusError": anthropic.APIStatusError,
        }
    except Exception:
        pass
    try:
        import openai
        out["openai"] = {
            "ConnectionError": openai.APIConnectionError,
            "TimeoutError": openai.APITimeoutError,
            "RateLimitError": openai.RateLimitError,
            "InternalServerError": openai.InternalServerError,
            "APIStatusError": openai.APIStatusError,
        }
    except Exception:
        pass
    return out


_EXC = _provider_exception_classes()


def _is_retryable(err: BaseException) -> bool:
    for provider in ("anthropic", "openai"):
        classes = _EXC.get(provider, {})
        if not classes:
            continue
        conn = classes.get("ConnectionError")
        tmo = classes.get("TimeoutError")
        rate = classes.get("RateLimitError")
        ise = classes.get("InternalServerError")
        stat = classes.get("APIStatusError")
        if conn and isinstance(err, conn):
            return True
        if tmo and isinstance(err, tmo):
            return True
        if rate and isinstance(err, rate):
            return True
        if ise and isinstance(err, ise):
            return True
        if stat and isinstance(err, stat):
            status = getattr(err, "status_code", None)
            if status in _RETRYABLE_STATUS:
                return True
    return False


def _max_retries_from_settings() -> int:
    try:
        from settings import load_settings
        return int(load_settings()["max_api_retries"])
    except Exception:
        return 6


def safe_complete(client: "LLMClient", **kwargs: Any) -> str:
    """Retry-wrapped `LLMClient.complete`. Same signature, returns text."""
    max_retries = _max_retries_from_settings()
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.complete(**kwargs)
        except Exception as e:  # noqa: BLE001
            if not _is_retryable(e) or attempt == max_retries:
                raise
            last = e
            wait = min(60.0, (2 ** attempt) + random.uniform(0.0, 1.0))
            label = type(e).__name__
            status = getattr(e, "status_code", None)
            extra = f" status={status}" if status is not None else ""
            print(
                f"  ⚠ api {label}{extra}; retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    if last is not None:
        raise last
    raise RuntimeError("safe_complete exhausted retries without an exception")
