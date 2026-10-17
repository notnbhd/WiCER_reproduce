"""Shared pytest fixtures.

Everything here runs offline: the reader is tests/fake_llama_server.py, the
judge and compiler use their ``fake`` backends, and results go to a tmp_path so
tests never touch a real run's output.
"""

from __future__ import annotations

import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.fake_llama_server import FakeLlamaServer  # noqa: E402
from wicer_core.config import get_profile  # noqa: E402

TOPIC = "mini_topic"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def fake_reader_port() -> int:
    port = _free_port()
    with FakeLlamaServer(port=port, n_ctx=8192, token_delay_s=0.0005):
        yield port


@pytest.fixture
def profile(tmp_path, fake_reader_port, monkeypatch):
    """The `test` profile, wired to the running fake server and a tmp results dir."""
    monkeypatch.setenv("WICER_JUDGE_BACKEND", "fake")
    monkeypatch.setenv("WICER_COMPILER_BACKEND", "fake")
    monkeypatch.setenv("WICER_EMBED_BACKEND", "fake")
    p = get_profile("test")
    p = replace(
        p,
        reader=replace(p.reader, port=fake_reader_port),
        judge=replace(p.judge, cache_dir=str(tmp_path / "judge-cache")),
        rag=replace(p.rag, index_dir=str(tmp_path / "rag_index")),
        results_dir=str(tmp_path / "results"),
    )
    return p


@pytest.fixture
def topic() -> str:
    return TOPIC
