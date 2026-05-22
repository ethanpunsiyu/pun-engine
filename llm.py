"""Provider-agnostic LLM client.

Wraps Anthropic and OpenAI behind one `complete(...)` call so the rest of
the codebase doesn't care which provider is in use. Provider, generator
model and judge model come from SETTINGS.txt.

Why this exists: previous iterations hard-coded `anthropic.Anthropic()` and
`resp.content[0].text` everywhere, which made it impossible to swap in an
OpenAI key without rewriting every call site.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from settings import load_settings


_PROVIDER_DEFAULTS = {
    "anthropic": {
        "generator_model": "claude-opus-4-7",
        "judge_model": "claude-sonnet-4-6",
    },
    # gpt-4o / gpt-4o-mini: non-reasoning, fast, well-suited to short JSON
    # tasks. gpt-5 / o-series CAN be used (just set them in SETTINGS.txt)
    # but they spend tokens on internal reasoning and easily take 30-120s
    # per call — switch to them deliberately, not as a default.
    "openai": {
        "generator_model": "gpt-4o",
        "judge_model": "gpt-4o-mini",
    },
}


# Per-request HTTP timeout (seconds). On timeout, safe_complete's retry loop
# kicks in. 300s is generous for reasoning models without letting a true
# network hang stall forever.
REQUEST_TIMEOUT_SECONDS = 300


def _resolve_models(settings: dict) -> tuple[str, str, str]:
    provider = settings.get("provider", "anthropic")
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"unknown provider {provider!r} — expected 'anthropic' or 'openai'"
        )
    defaults = _PROVIDER_DEFAULTS[provider]
    gen = settings.get("generator_model") or defaults["generator_model"]
    judge = settings.get("judge_model") or defaults["judge_model"]
    return provider, gen, judge


class LLMClient:
    """Single entry point for LLM calls regardless of provider."""

    def __init__(self, provider: Optional[str] = None) -> None:
        s = load_settings()
        if provider is not None:
            s = {**s, "provider": provider}
        self.provider, self.generator_model, self.judge_model = _resolve_models(s)

        if self.provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "provider=anthropic but the `anthropic` package is not "
                    "installed. Run `pip install anthropic`."
                ) from e
            import anthropic
            self._anthropic = anthropic.Anthropic()
            self._openai = None
        elif self.provider == "openai":
            try:
                import openai  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "provider=openai but the `openai` package is not "
                    "installed. Run `pip install openai`."
                ) from e
            import openai
            self._openai = openai.OpenAI()
            self._anthropic = None

    # -----------------------------------------------------------------------
    # Unified completion call
    # -----------------------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system: list[dict],
        user: str,
        max_tokens: int,
        json_mode: bool = True,
    ) -> str:
        """Run a single-turn completion.

        `system` is a list of `{"text": "...", "cache_control": {...}}`
        dicts (the shape Anthropic uses). For OpenAI the blocks are
        concatenated into one system message and cache_control is dropped
        — newer OpenAI models do prompt caching transparently.

        `json_mode=True` asks the model to return JSON. Anthropic relies on
        the prompt to say so; OpenAI sets `response_format`.
        """
        if self.provider == "anthropic":
            resp = self._anthropic.with_options(
                timeout=REQUEST_TIMEOUT_SECONDS,
            ).messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        # OpenAI
        system_text = "\n\n".join(b["text"] for b in system if b.get("text"))

        # Reasoning-class OpenAI models (gpt-5, o1, o3, …) burn tokens on
        # INTERNAL reasoning that count against max_completion_tokens, so
        # we want headroom above the caller's output budget. We only pay
        # for tokens actually used, so inflating the ceiling is safe.
        # 16384 is the cap on gpt-4o / gpt-4o-mini; reasoning models accept
        # higher values, which we'll discover and bump to lazily.
        oai_token_budget = max(max_tokens, 16384)

        base_kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            base_kwargs["response_format"] = {"type": "json_object"}

        oai = self._openai.with_options(timeout=REQUEST_TIMEOUT_SECONDS)

        def _call(budget: int, param: str):
            return oai.chat.completions.create(
                **base_kwargs, **{param: budget},
            )

        # Try the modern parameter first; the various per-model quirks get
        # papered over in the except branches.
        try:
            resp = _call(oai_token_budget, "max_completion_tokens")
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            # Some older chat models reject `max_completion_tokens` entirely.
            if "max_completion_tokens" in msg and (
                "unsupported" in msg or "unknown" in msg or "not supported" in msg
            ):
                resp = _call(oai_token_budget, "max_tokens")
            # Some models cap the budget below ours — parse the limit out of
            # the error message and retry at the cap.
            elif "too large" in msg:
                m = re.search(r"at most (\d+)", str(e))
                if not m:
                    raise
                cap = int(m.group(1))
                try:
                    resp = _call(cap, "max_completion_tokens")
                except Exception as e2:  # noqa: BLE001
                    msg2 = str(e2).lower()
                    if "max_completion_tokens" in msg2 and (
                        "unsupported" in msg2 or "unknown" in msg2
                        or "not supported" in msg2
                    ):
                        resp = _call(cap, "max_tokens")
                    else:
                        raise
            else:
                raise

        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            reason = getattr(choice, "finish_reason", "unknown")
            # Pull reasoning-token usage if available — helps the user see
            # where the budget went.
            usage = getattr(resp, "usage", None)
            usage_str = ""
            if usage is not None:
                details = getattr(usage, "completion_tokens_details", None)
                rtok = getattr(details, "reasoning_tokens", None) if details else None
                ctok = getattr(usage, "completion_tokens", None)
                if rtok is not None or ctok is not None:
                    usage_str = (
                        f"  completion_tokens={ctok}, reasoning_tokens={rtok}"
                    )
            raise RuntimeError(
                f"OpenAI returned empty content from {model!r} "
                f"(finish_reason={reason}).{usage_str}\n"
                "Most common cause: reasoning models consume "
                "max_completion_tokens internally before producing output. "
                "Either raise the per-call budget, switch to a "
                "non-reasoning model (e.g. gpt-4o-mini), or set "
                "judge_model / generator_model in SETTINGS.txt to something "
                "lighter."
            )
        return content


def get_client(provider: Optional[str] = None) -> LLMClient:
    """Module-level convenience constructor."""
    return LLMClient(provider=provider)
