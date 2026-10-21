"""Reader client: SSE timing extraction, KV-cache reuse, context-fit checks."""

from __future__ import annotations

import pytest

from wicer_core.reader import (
    ContextOverflow,
    ReaderClient,
    _iter_sse,
    build_user_prompt,
)


class _FakeResponse:
    """Minimal stand-in for a streamed requests.Response."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    def iter_lines(self, decode_unicode: bool = True):
        return iter(self._lines)


def test_sse_parser_skips_noise_and_stops_at_done():
    events = list(_iter_sse(_FakeResponse([
        "",
        ": keep-alive comment",
        'data: {"a": 1}',
        "not-an-sse-line",
        'data: {"b": 2}',
        "data: {malformed",
        "data: [DONE]",
        'data: {"never": true}',
    ])))
    assert events == [{"a": 1}, {"b": 2}]


def test_health_and_props(profile):
    reader = ReaderClient(profile.reader)
    assert reader.health() is True
    assert reader.server_n_ctx() == 8192


def test_health_is_false_when_nothing_is_listening(profile):
    from dataclasses import replace

    reader = ReaderClient(replace(profile.reader, port=1))
    assert reader.health() is False


def test_tokenize_counts_grow_with_text(profile):
    reader = ReaderClient(profile.reader)
    assert reader.count_tokens("") == 0
    assert reader.count_tokens("one two three") == 3
    assert reader.count_tokens("a " * 100) > reader.count_tokens("a " * 10)


def test_context_limit_reserves_generation_and_template_slack(profile):
    reader = ReaderClient(profile.reader)
    limit = reader.context_limit()
    assert limit == 8192 - profile.reader.max_tokens - profile.reader.ctx_margin_tokens
    assert reader.check_fits("short context") < limit


def test_oversized_context_raises_rather_than_truncating(profile):
    """The paper's two excluded topics fall out here, by measurement."""
    reader = ReaderClient(profile.reader)
    with pytest.raises(ContextOverflow) as exc:
        reader.check_fits("word " * 20_000)
    assert exc.value.n_tokens > exc.value.limit


def test_generation_records_ttft_total_and_throughput(profile):
    reader = ReaderClient(profile.reader)
    system = "The capital of Ruritania is Strelsau. " * 20
    gen = reader.chat(system, build_user_prompt("What is the capital of Ruritania?"))
    assert "Strelsau" in gen.text
    assert 0 < gen.ttft_s <= gen.total_s
    assert gen.n_generated_tokens > 0
    assert gen.throughput_tok_s > 0
    assert gen.finish_reason == "stop"
    assert gen.n_prompt_tokens and gen.n_prompt_tokens > 0


def test_non_ascii_answers_survive_the_stream(profile):
    """llama-server streams UTF-8 under a bare ``text/event-stream`` header.

    requests decodes any charset-less text/* response as ISO-8859-1, which put
    "GarcÃ\xada" for "García" into 1-2% of the pilots' answers.
    """
    reader = ReaderClient(profile.reader)
    system = "VirtualMarket was founded by Ana García in Málaga in late 2023. " * 5
    gen = reader.chat(system, build_user_prompt("Who founded VirtualMarket?"))
    assert "Ana García" in gen.text
    assert "Málaga" in gen.text


def test_priming_enables_prefix_reuse_on_the_next_query(profile):
    """Paper Section 3.4: the preflight request is what makes later TTFT warm."""
    reader = ReaderClient(profile.reader)
    system = "Fact: the Suden River flows east. " * 50
    cold = reader.chat(system + " (unique suffix)", build_user_prompt("Which way?"))
    assert cold.cache_n == 0

    reader.prime(system)
    warm = reader.chat(system, build_user_prompt("Which way does the Suden River flow?"))
    assert warm.cache_n and warm.cache_n > 0


def test_user_prompt_is_a_short_question_suffix():
    prompt = build_user_prompt("Who founded the town?")
    assert "Who founded the town?" in prompt
    # Paper Section 3.4 puts this at ~30-50 tokens; keep it in that neighbourhood.
    assert len(prompt.split()) < 40
