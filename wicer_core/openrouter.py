"""OpenRouter backend, shared by the compiler, the diagnoser and the judge.

The Anthropic path uses four features OpenRouter's OpenAI-compatible endpoint
does not have in the same shape: `thinking`, `output_config.effort`, an
`ephemeral` system block with a 1-hour TTL, and `cache_read_input_tokens`.
This module owns the whole translation so the three call sites stay readable
and the mapping is stated once:

    thinking={"type": "disabled"}  ->  no `reasoning` block
    output_config={"effort": e}    ->  reasoning={"effort": e}
    cache_control ttl 1h           ->  same, passed through; anthropic/* only
    usage.cache_read_input_tokens  ->  usage.prompt_tokens_details.cached_tokens

Cost comes from OpenRouter itself (`usage.include`), not a price table that
goes stale the moment you switch slugs.  See DEVIATIONS.md Section 11.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_VAR = "OPENROUTER_API_KEY"

# Only Anthropic-served slugs accept a cache_control breakpoint through
# OpenRouter.  Sending one to any other provider is at best ignored.
_CACHE_CAPABLE_PREFIXES = ("anthropic/",)


class OpenRouterError(RuntimeError):
    """Anything that makes an OpenRouter response unusable."""


@dataclass(frozen=True)
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    served_model: str
    cache_read_tokens: int = 0
    finish_reason: str = ""
    # Reasoning tokens bill as completion tokens and count against max_tokens.
    # At a high effort with a tight cap they can consume the whole budget and
    # leave no text at all -- see the empty-completion guard in `chat`.
    reasoning_tokens: int = 0


def supports_cache_control(model: str) -> bool:
    return model.startswith(_CACHE_CAPABLE_PREFIXES)


def build_payload(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    effort: str | None = None,
    cache_system: bool = False,
    cache_ttl: str | None = None,
) -> dict[str, Any]:
    """Shape one chat-completions request.

    `effort="disabled"` is accepted and means "no reasoning block", mirroring
    the Anthropic path's thinking={"type": "disabled"}.

    `cache_ttl="1h"` is passed through to Anthropic.  A 1-hour write costs 2x
    base input against 1.25x for the default window, so it only pays when the
    prefix is genuinely reused -- the compile stage, not the judge.
    """
    if cache_system and supports_cache_control(model):
        cache_control: dict[str, Any] = {"type": "ephemeral"}
        if cache_ttl:
            cache_control["ttl"] = cache_ttl
        system_content: Any = [
            {
                "type": "text",
                "text": system,
                "cache_control": cache_control,
            }
        ]
    else:
        system_content = system

    # `usage` and `reasoning` are OpenRouter extensions, not OpenAI SDK
    # parameters.  Completions.create() has no **kwargs, so passing them at the
    # top level is a TypeError -- they must ride in extra_body.
    extra_body: dict[str, Any] = {"usage": {"include": True}}
    if effort and effort != "disabled":
        extra_body["reasoning"] = {"effort": effort}

    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user},
        ],
        "extra_body": extra_body,
    }


def parse_usage(usage: Any) -> dict[str, Any]:
    """Read an OpenRouter usage object defensively.

    Providers vary in which fields they populate, and a judge sweep is 54K
    calls long -- a missing field must cost accuracy, never the run.
    """
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "cache_read_tokens": 0,
            "reasoning_tokens": 0,
        }
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "cost_usd": float(getattr(usage, "cost", 0.0) or 0.0),
        "cache_read_tokens": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
        "reasoning_tokens": (
            getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        ),
    }


class OpenRouterClient:
    """Thin wrapper over the OpenAI SDK pointed at OpenRouter."""

    def __init__(self, client: Any = None, api_key: str | None = None):
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            # Key first: an unset key is the common mistake, and its message
            # must not be masked by the optional dependency being absent.
            key = self._api_key or os.environ.get(API_KEY_VAR)
            if not key:
                raise OpenRouterError(
                    f"{API_KEY_VAR} is not set -- put it in .env or export it"
                )

            try:
                import openai
            except ImportError as e:  # pragma: no cover - install-time guidance
                raise OpenRouterError(
                    "the openrouter backend needs the openai package: "
                    "uv pip install -e '.[openrouter]'"
                ) from e

            self._client = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        return self._client

    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        effort: str | None = None,
        cache_system: bool = False,
        cache_ttl: str | None = None,
    ) -> Reply:
        payload = build_payload(
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            effort=effort,
            cache_system=cache_system,
            cache_ttl=cache_ttl,
        )
        resp = self.client.chat.completions.create(**payload)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise OpenRouterError(f"{model} returned no choices")
        u = parse_usage(getattr(resp, "usage", None))
        text = (choices[0].message.content or "").strip()
        finish_reason = getattr(choices[0], "finish_reason", "") or ""

        if not text:
            # An empty completion is unusable to every caller, and writing it
            # out produces a 0-byte wiki that reads as "the method does not
            # work".  Fail loudly and name the cause.
            raise OpenRouterError(
                f"{model} returned no text "
                f"(finish_reason={finish_reason!r}, "
                f"max_tokens={max_tokens}, "
                f"completion_tokens={u['output_tokens']}, "
                f"reasoning_tokens={u['reasoning_tokens']}). "
                f"Reasoning counts against max_tokens -- raise max_tokens or "
                f"lower effort (currently {effort!r})."
            )

        return Reply(
            text=text,
            served_model=getattr(resp, "model", model),
            finish_reason=finish_reason,
            **u,
        )
