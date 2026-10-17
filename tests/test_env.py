"""`.env` loading: parsing rules and the shell-wins precedence guarantee."""

from __future__ import annotations

import os

from wicer_core.env import load_dotenv, parse_dotenv


def test_parses_simple_assignments():
    assert parse_dotenv("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_ignores_comments_and_blank_lines():
    text = "# a comment\n\nA=1\n   \n#B=2\n"
    assert parse_dotenv(text) == {"A": "1"}


def test_strips_surrounding_quotes():
    text = "A='single'\nB=\"double\"\nC=bare\n"
    assert parse_dotenv(text) == {"A": "single", "B": "double", "C": "bare"}


def test_keeps_inner_equals_signs():
    # Base64 and JWT-ish secrets routinely end in '='.
    assert parse_dotenv("K=abc=def==\n") == {"K": "abc=def=="}


def test_strips_export_prefix():
    assert parse_dotenv("export A=1\n") == {"A": "1"}


def test_trims_whitespace_around_key_and_value():
    assert parse_dotenv("  A = 1  \n") == {"A": "1"}


def test_ignores_lines_without_an_equals_sign():
    assert parse_dotenv("A=1\ngarbage line\nB=2\n") == {"A": "1", "B": "2"}


def test_empty_value_is_kept_as_empty_string():
    assert parse_dotenv("A=\n") == {"A": ""}


def test_load_dotenv_sets_absent_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("WICER_TEST_FRESH=from_file\n")
    monkeypatch.delenv("WICER_TEST_FRESH", raising=False)

    loaded = load_dotenv(env_file)

    assert os.environ["WICER_TEST_FRESH"] == "from_file"
    assert loaded == {"WICER_TEST_FRESH": "from_file"}


def test_load_dotenv_never_overrides_the_shell(tmp_path, monkeypatch):
    """The shell is authoritative; a stale .env must not silently win."""
    env_file = tmp_path / ".env"
    env_file.write_text("WICER_TEST_SET=from_file\n")
    monkeypatch.setenv("WICER_TEST_SET", "from_shell")

    loaded = load_dotenv(env_file)

    assert os.environ["WICER_TEST_SET"] == "from_shell"
    assert loaded == {}


def test_load_dotenv_is_a_noop_when_the_file_is_missing(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_load_dotenv_tolerates_an_unreadable_file(tmp_path, monkeypatch):
    """A broken .env must never take down a 40-GPU-hour run at import time."""
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"\xff\xfe\x00bad")
    assert load_dotenv(env_file) == {}


# -- wiring into the config module -------------------------------------------


def test_config_loads_dotenv_at_import(monkeypatch):
    """Every entry point imports wicer_core.config; that is where .env lands."""
    import importlib

    import wicer_core.config
    import wicer_core.env

    called: list[object] = []
    monkeypatch.setattr(wicer_core.env, "load_dotenv", lambda *a, **k: called.append(1) or {})
    importlib.reload(wicer_core.config)

    assert called, "wicer_core.config must call load_dotenv() at import"


def test_openrouter_model_default_replaces_an_anthropic_native_id():
    from wicer_core.config import OPENROUTER_DEFAULT_MODEL, resolve_openrouter_model

    assert resolve_openrouter_model("claude-sonnet-5") == OPENROUTER_DEFAULT_MODEL


def test_openrouter_model_default_leaves_a_real_slug_alone():
    from wicer_core.config import resolve_openrouter_model

    assert resolve_openrouter_model("google/gemini-2.5-pro") == "google/gemini-2.5-pro"


def test_openrouter_default_matches_the_configured_judge_model():
    """Resolving must not silently swap in an older model than config pins."""
    from wicer_core.config import OPENROUTER_DEFAULT_MODEL, get_profile

    pinned = get_profile("paper").judge.model          # e.g. "claude-sonnet-5"
    assert OPENROUTER_DEFAULT_MODEL.split("/", 1)[1] == pinned


def test_openrouter_default_model_is_a_slug():
    from wicer_core.config import OPENROUTER_DEFAULT_MODEL

    assert "/" in OPENROUTER_DEFAULT_MODEL


def test_compiler_effort_is_overridable(monkeypatch):
    """Effort competes with output for max_tokens on the openrouter path, so
    it has to be tunable without editing config.py."""
    from wicer_core.config import get_profile

    monkeypatch.setenv("WICER_COMPILER_EFFORT", "low")
    assert get_profile("paper").compiler.effort == "low"


def test_judge_effort_is_overridable(monkeypatch):
    from wicer_core.config import get_profile

    monkeypatch.setenv("WICER_JUDGE_EFFORT", "medium")
    assert get_profile("paper").judge.effort == "medium"


def test_compiler_max_tokens_is_overridable(monkeypatch):
    from wicer_core.config import get_profile

    monkeypatch.setenv("WICER_COMPILER_MAX_TOKENS", "16000")
    assert get_profile("paper").compiler.max_tokens == 16_000


def test_efforts_default_to_the_configured_values(monkeypatch):
    from wicer_core.config import get_profile

    monkeypatch.delenv("WICER_COMPILER_EFFORT", raising=False)
    monkeypatch.delenv("WICER_JUDGE_EFFORT", raising=False)
    p = get_profile("paper")
    assert p.compiler.effort == "high"
    assert p.judge.effort == "low"
