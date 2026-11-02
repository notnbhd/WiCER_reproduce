"""TACT's results never land in the reproduction's results directory."""

from __future__ import annotations

from tact import RESULTS_DIR, get_profile


def test_paper_profile_writes_to_result_tact(monkeypatch):
    monkeypatch.delenv("WICER_RESULTS_DIR", raising=False)
    assert get_profile("paper").results_dir == RESULTS_DIR


def test_test_profile_writes_to_a_tact_subdirectory(monkeypatch):
    monkeypatch.delenv("WICER_RESULTS_DIR", raising=False)
    assert get_profile("test").results_dir == "results/test/tact"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("WICER_RESULTS_DIR", "elsewhere")
    assert get_profile("paper").results_dir == "elsewhere"
