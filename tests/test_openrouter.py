"""OpenRouter backend: payload shaping, usage parsing, dispatch.

No network.  A stub standing in for the OpenAI SDK client records the payload
it was handed, so every assertion here is about what we would send and how we
read what comes back.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wicer_core.openrouter import (
    OPENROUTER_BASE_URL,
    OpenRouterClient,
    OpenRouterError,
    Reply,
    build_payload,
    parse_usage,
    supports_cache_control,
)

ANTHROPIC_SLUG = "anthropic/claude-sonnet-4.5"
OTHER_SLUG = "google/gemini-2.5-pro"


# -- cache-control eligibility ------------------------------------------------


def test_anthropic_slugs_support_cache_control():
    assert supports_cache_control(ANTHROPIC_SLUG) is True


def test_non_anthropic_slugs_do_not_support_cache_control():
    assert supports_cache_control(OTHER_SLUG) is False


# -- payload shaping ----------------------------------------------------------


def test_payload_always_asks_for_usage_accounting():
    """Cost reporting is the whole reason we trust OpenRouter's numbers."""
    p = build_payload(model=OTHER_SLUG, system="s", user="u", max_tokens=10)
    assert p["extra_body"]["usage"] == {"include": True}


def test_openrouter_extensions_travel_in_extra_body():
    """`usage` and `reasoning` are not OpenAI SDK parameters -- create() would
    reject them as unexpected keyword arguments."""
    p = build_payload(model=OTHER_SLUG, system="s", user="u", max_tokens=10, effort="high")
    assert "usage" not in p
    assert "reasoning" not in p
    assert set(p["extra_body"]) == {"usage", "reasoning"}


def test_payload_carries_model_and_max_tokens():
    p = build_payload(model=OTHER_SLUG, system="s", user="u", max_tokens=321)
    assert p["model"] == OTHER_SLUG
    assert p["max_tokens"] == 321


def test_payload_puts_system_first_then_user():
    p = build_payload(model=OTHER_SLUG, system="sys", user="usr", max_tokens=10)
    assert [m["role"] for m in p["messages"]] == ["system", "user"]
    assert p["messages"][1]["content"] == "usr"


def test_effort_becomes_a_reasoning_block():
    p = build_payload(model=OTHER_SLUG, system="s", user="u", max_tokens=10, effort="high")
    assert p["extra_body"]["reasoning"] == {"effort": "high"}


def test_no_reasoning_block_when_effort_is_unset():
    p = build_payload(model=OTHER_SLUG, system="s", user="u", max_tokens=10)
    assert "reasoning" not in p["extra_body"]


def test_disabled_effort_omits_the_reasoning_block():
    """The Anthropic path uses thinking={'type':'disabled'}; this is its analogue."""
    p = build_payload(model=OTHER_SLUG, system="s", user="u", max_tokens=10, effort="disabled")
    assert "reasoning" not in p["extra_body"]


def test_cache_system_marks_the_system_block_on_anthropic_slugs():
    p = build_payload(
        model=ANTHROPIC_SLUG, system="big source text", user="u", max_tokens=10, cache_system=True
    )
    block = p["messages"][0]["content"][0]
    assert block["type"] == "text"
    assert block["text"] == "big source text"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_cache_system_is_ignored_on_slugs_that_cannot_cache():
    """OpenRouter would pass the key through to a provider that rejects it."""
    p = build_payload(
        model=OTHER_SLUG, system="big source text", user="u", max_tokens=10, cache_system=True
    )
    assert p["messages"][0]["content"] == "big source text"


def test_system_is_a_plain_string_when_caching_is_not_requested():
    p = build_payload(model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=10)
    assert p["messages"][0]["content"] == "s"


# -- usage parsing ------------------------------------------------------------


def _usage(**kw):
    base = {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0123}
    base.update(kw)
    return SimpleNamespace(**base)


def test_parse_usage_reads_tokens_and_cost():
    u = parse_usage(_usage())
    assert u["input_tokens"] == 100
    assert u["output_tokens"] == 20
    assert u["cost_usd"] == pytest.approx(0.0123)


def test_parse_usage_reads_cached_tokens_when_present():
    u = parse_usage(_usage(prompt_tokens_details=SimpleNamespace(cached_tokens=64)))
    assert u["cache_read_tokens"] == 64


def test_parse_usage_defaults_cached_tokens_to_zero():
    assert parse_usage(_usage())["cache_read_tokens"] == 0


def test_parse_usage_survives_a_missing_usage_object():
    """A provider that reports nothing must not crash a 6800-question sweep."""
    u = parse_usage(None)
    assert u == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "cache_read_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_parse_usage_treats_a_missing_cost_as_zero():
    u = parse_usage(SimpleNamespace(prompt_tokens=1, completion_tokens=2))
    assert u["cost_usd"] == 0.0


# -- the client ---------------------------------------------------------------


class _StubCompletions:
    """Mirrors openai.resources.chat.completions.Completions.create.

    That method has no **kwargs: an OpenRouter extension passed as a top-level
    keyword is a TypeError, not a request.  The stub must be as strict.
    """

    def __init__(self, response):
        self.response = response
        self.payload = None

    def create(self, *, model, messages, max_tokens=None, extra_body=None, **unsupported):
        if unsupported:
            raise TypeError(
                f"create() got an unexpected keyword argument "
                f"{min(unsupported)!r}"
            )
        self.payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body or {},
        }
        return self.response


class _StubClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_StubCompletions(response))


def _response(text="hello", model="anthropic/claude-sonnet-4.5", usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        model=model,
        usage=usage if usage is not None else _usage(),
    )


def test_chat_returns_the_text_and_the_served_model():
    stub = _StubClient(_response(text=" spaced ", model="anthropic/claude-sonnet-4.5-20991231"))
    c = OpenRouterClient(client=stub)

    reply = c.chat(model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=10)

    assert isinstance(reply, Reply)
    assert reply.text == "spaced"
    assert reply.served_model == "anthropic/claude-sonnet-4.5-20991231"


def test_chat_reports_usage_and_cost():
    stub = _StubClient(_response())
    reply = OpenRouterClient(client=stub).chat(
        model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=10
    )
    assert reply.input_tokens == 100
    assert reply.output_tokens == 20
    assert reply.cost_usd == pytest.approx(0.0123)


def test_chat_sends_the_payload_build_payload_produced():
    stub = _StubClient(_response())
    OpenRouterClient(client=stub).chat(
        model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=77, effort="low"
    )
    sent = stub.chat.completions.payload
    assert sent["max_tokens"] == 77
    assert sent["extra_body"]["reasoning"] == {"effort": "low"}
    assert sent["extra_body"]["usage"] == {"include": True}


def test_chat_raises_a_clear_error_on_an_empty_choice_list():
    stub = _StubClient(SimpleNamespace(choices=[], model=ANTHROPIC_SLUG, usage=_usage()))
    with pytest.raises(OpenRouterError, match="no choices"):
        OpenRouterClient(client=stub).chat(
            model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=10
        )


def test_missing_api_key_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        _ = OpenRouterClient().client


def test_base_url_points_at_openrouter():
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


# -- cache TTL ---------------------------------------------------------------


def test_cache_ttl_is_included_when_requested():
    """OpenRouter passes Anthropic's 1h TTL through; the compile stage needs it."""
    p = build_payload(
        model=ANTHROPIC_SLUG, system="src", user="u", max_tokens=10,
        cache_system=True, cache_ttl="1h",
    )
    block = p["messages"][0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_cache_control_defaults_to_the_short_window():
    """No ttl means the provider default -- right for prompts never reused."""
    p = build_payload(model=ANTHROPIC_SLUG, system="src", user="u", max_tokens=10,
                      cache_system=True)
    assert p["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_ttl_is_dropped_on_a_slug_that_cannot_cache():
    p = build_payload(model=OTHER_SLUG, system="src", user="u", max_tokens=10,
                      cache_system=True, cache_ttl="1h")
    assert p["messages"][0]["content"] == "src"


def test_chat_forwards_the_ttl():
    stub = _StubClient(_response())
    OpenRouterClient(client=stub).chat(
        model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=10,
        cache_system=True, cache_ttl="1h",
    )
    block = stub.chat.completions.payload["messages"][0]["content"][0]
    assert block["cache_control"]["ttl"] == "1h"


# -- empty and truncated completions -----------------------------------------


def _response_with(finish_reason="stop", text="hi", reasoning_tokens=0):
    usage = _usage()
    usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                 finish_reason=finish_reason)],
        model=ANTHROPIC_SLUG,
        usage=usage,
    )


def test_an_empty_completion_raises_rather_than_returning_nothing():
    """A 0-byte wiki silently becomes a null experimental result."""
    stub = _StubClient(_response_with(finish_reason="length", text="", reasoning_tokens=4000))
    with pytest.raises(OpenRouterError, match="no text"):
        OpenRouterClient(client=stub).chat(
            model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=4000
        )


def test_the_empty_completion_error_names_the_cause():
    """reasoning_tokens == max_tokens is the actual failure; say so."""
    stub = _StubClient(_response_with(finish_reason="length", text="", reasoning_tokens=4000))
    with pytest.raises(OpenRouterError) as e:
        OpenRouterClient(client=stub).chat(
            model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=4000, effort="high"
        )
    msg = str(e.value)
    assert "length" in msg
    assert "4000" in msg
    assert "effort" in msg or "max_tokens" in msg


def test_whitespace_only_counts_as_empty():
    stub = _StubClient(_response_with(text="   \n  "))
    with pytest.raises(OpenRouterError, match="no text"):
        OpenRouterClient(client=stub).chat(
            model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=10
        )


def test_reply_carries_the_finish_reason_and_reasoning_tokens():
    stub = _StubClient(_response_with(text="real output", reasoning_tokens=120))
    r = OpenRouterClient(client=stub).chat(
        model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=10
    )
    assert r.finish_reason == "stop"
    assert r.reasoning_tokens == 120


def test_a_truncated_but_non_empty_completion_is_returned():
    """Losing a long-but-cut-off wiki to an exception would be worse."""
    stub = _StubClient(_response_with(finish_reason="length", text="a partial wiki"))
    r = OpenRouterClient(client=stub).chat(
        model=ANTHROPIC_SLUG, system="s", user="u", max_tokens=10
    )
    assert r.text == "a partial wiki"
    assert r.finish_reason == "length"
