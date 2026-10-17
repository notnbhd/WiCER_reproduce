"""Provenance and result-file plumbing."""

from __future__ import annotations

import json

from wicer_core.provenance import build_run_meta, file_sha256, write_run_meta
from wicer_core.records import (
    CONDITIONS,
    JsonlWriter,
    completed_question_ids,
    read_jsonl,
    write_skip,
)


def test_run_meta_records_the_knobs_that_invalidate_comparisons(profile):
    meta = build_run_meta(profile, command="test", hash_model=False)
    # spec.md Section 9 names judge drift and hardware as the two silent killers.
    assert meta["judge_model_configured"] == profile.judge.model
    assert meta["compiler_model_configured"] == profile.compiler.model
    assert meta["reader"]["n_ctx"] == profile.reader.n_ctx
    assert meta["reader"]["cache_type_k"] == profile.reader.cache_type_k
    assert meta["reader"]["temperature"] == 0.0
    assert meta["reader"]["seed"] == profile.reader.seed
    assert meta["platform"] and meta["python"]
    assert meta["profile"]["name"] == "test"


def test_run_meta_round_trips_to_disk(profile, tmp_path):
    path = write_run_meta(tmp_path / "nested" / "run_meta.json",
                          build_run_meta(profile, command="test", hash_model=False))
    loaded = json.loads(path.read_text())
    assert loaded["command"] == "test"
    assert loaded["started_at"] and loaded["finished_at"]


def test_file_sha256_of_a_missing_file_is_none(tmp_path):
    assert file_sha256(tmp_path / "absent.gguf") is None
    f = tmp_path / "present.bin"
    f.write_bytes(b"hello")
    assert file_sha256(f) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_conditions_cover_the_specs_eight_arms():
    for expected in [
        "fc_raw", "fc_raw_q4kv", "rag_top5",
        "wiki_light", "wiki_moderate", "wiki_aggressive",
        "wicer_iter1", "wicer_iter2", "random_pin",
    ]:
        assert expected in CONDITIONS


def test_jsonl_writer_flushes_so_a_killed_run_keeps_its_rows(tmp_path):
    path = tmp_path / "out" / "answers.jsonl"
    writer = JsonlWriter(path)
    writer.write({"question_id": "q1"})
    # Readable before close: an interrupted sweep must not lose everything.
    assert read_jsonl(path) == [{"question_id": "q1"}]
    writer.write({"question_id": "q2"})
    writer.close()
    assert completed_question_ids(path) == {"q1", "q2"}


def test_jsonl_writer_truncates_unless_resuming(tmp_path):
    path = tmp_path / "answers.jsonl"
    with JsonlWriter(path) as w:
        w.write({"question_id": "q1"})
    with JsonlWriter(path, resume=True) as w:
        w.write({"question_id": "q2"})
    assert completed_question_ids(path) == {"q1", "q2"}
    with JsonlWriter(path) as w:
        w.write({"question_id": "q3"})
    assert completed_question_ids(path) == {"q3"}


def test_reading_a_missing_jsonl_is_empty_not_an_error(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []
    assert completed_question_ids(tmp_path / "nope.jsonl") == set()


def test_write_skip_records_the_reason_and_detail(profile):
    path = write_skip(profile, "t", "fc_raw", "context_overflow", n_context_tokens=97_251, limit=96_000)
    data = json.loads(path.read_text())
    assert data == {"skipped": "context_overflow", "n_context_tokens": 97_251, "limit": 96_000}


def test_run_meta_records_the_backends_in_use(monkeypatch, profile):
    """Which service answered is as load-bearing as which model was asked for."""
    monkeypatch.setenv("WICER_JUDGE_BACKEND", "openrouter")
    monkeypatch.setenv("WICER_COMPILER_BACKEND", "openrouter")
    meta = build_run_meta(profile, command="test", hash_model=False)
    assert meta["judge_backend"] == "openrouter"
    assert meta["compiler_backend"] == "openrouter"


def test_run_meta_records_the_slug_openrouter_actually_receives(monkeypatch, profile):
    """A run_meta naming claude-sonnet-5 when anthropic/... was sent is a lie."""
    from dataclasses import replace

    monkeypatch.setenv("WICER_JUDGE_BACKEND", "openrouter")
    monkeypatch.setenv("WICER_COMPILER_BACKEND", "openrouter")
    p = replace(
        profile,
        judge=replace(profile.judge, model="claude-sonnet-5"),
        compiler=replace(profile.compiler, model="claude-sonnet-5"),
    )
    meta = build_run_meta(p, command="test", hash_model=False)
    assert meta["judge_model_resolved"] == "anthropic/claude-sonnet-5"
    assert meta["compiler_model_resolved"] == "anthropic/claude-sonnet-5"


def test_run_meta_leaves_the_model_untouched_off_openrouter(monkeypatch, profile):
    monkeypatch.setenv("WICER_JUDGE_BACKEND", "anthropic")
    monkeypatch.setenv("WICER_COMPILER_BACKEND", "anthropic")
    meta = build_run_meta(profile, command="test", hash_model=False)
    assert meta["judge_model_resolved"] == profile.judge.model
