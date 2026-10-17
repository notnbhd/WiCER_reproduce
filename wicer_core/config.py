"""Run profiles.

The paper runs on an Apple M4 Pro with Llama 3.1 8B Q5_K_M at a 96K context
window (paper Section 3.3).  Nothing about the code assumes that hardware: the
reader is reached over HTTP, so a profile is just a bundle of knobs.

Profiles
--------
paper  Faithful to Section 3.3.  Needs ~12 GB of VRAM (5.7 GB weights +
       ~6.4 GB of q8_0 KV at 96K).  Intended for a rented 24 GB GPU.
smoke  Small model, short context, few documents.  Fits a 6 GB laptop.
       Produces a *different* experiment -- see DEVIATIONS.md.
test   Points at tests/fake_llama_server.py.  No GPU, no API, no downloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any

from wicer_core import env

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every entry point imports this module, so this is the one place a `.env`
# needs loading.  The shell always wins -- see wicer_core/env.py.
env.load_dotenv(REPO_ROOT / env.DOTENV_NAME)

# Used when the openrouter backend is selected but the configured model is
# still an Anthropic-native id like "claude-sonnet-5", which OpenRouter does
# not recognise.  WICER_COMPILER_MODEL / WICER_JUDGE_MODEL override it.
OPENROUTER_DEFAULT_MODEL = "anthropic/claude-sonnet-5"


def resolve_openrouter_model(model: str) -> str:
    """Map an Anthropic-native model id onto an OpenRouter slug.

    OpenRouter addresses models as "provider/name".  Anything already carrying
    a slash is assumed to be deliberate and passed through untouched.
    """
    return model if "/" in model else OPENROUTER_DEFAULT_MODEL


@dataclass(frozen=True)
class ReaderConfig:
    """llama-server (llama.cpp) settings -- the model under test."""

    model_path: str = "models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
    host: str = "127.0.0.1"
    port: int = 8080
    n_ctx: int = 98_304  # 96K, paper Section 3.3
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    n_gpu_layers: int = 99
    flash_attn: bool = True
    # Greedy decoding, T=0 (paper Section 3.3).
    temperature: float = 0.0
    top_k: int = 1
    seed: int = 1234
    max_tokens: int = 512
    # Leave headroom for the chat template, the question suffix and generation.
    ctx_margin_tokens: int = 2_048
    request_timeout_s: float = 900.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class CorpusConfig:
    """RepLiQA subsampling.  Paper Section 3.1: 17 topics x 80 docs x 400 QA."""

    hf_repo: str = "ServiceNow/repliqa"
    split: str = "repliqa_0"
    n_docs_per_topic: int = 80
    n_qa_per_doc: int = 5
    # Which 80 of a topic's ~200 documents.  The paper does not say, and its
    # reported corpus sizes are only reachable by a length-biased selection:
    # "shortest" reproduces them to under 1% and reproduces its two
    # context-overflow topics exactly.  "lexicographic" is length-neutral and
    # was the original guess.  See DEVIATIONS.md 2.
    doc_selection: str = "shortest"
    data_dir: str = "data/repliqa"
    hf_cache_dir: str = "data/hf"


@dataclass(frozen=True)
class JudgeConfig:
    """LLM-as-judge (paper Section 3.5, rubric in Appendix B)."""

    # The paper says only "Claude Sonnet" and never pins a version; spec.md
    # Section 9 calls that out as a reproduction risk.  We pin an exact ID and
    # additionally record the `model` the API echoes back on every call.
    model: str = "claude-sonnet-5"
    max_tokens: int = 512
    effort: str = "low"
    max_workers: int = 8
    cache_dir: str = ".cache/judge"
    # USD per 1M tokens, used only for cost estimates and the --max-api-cost
    # guard.  Not authoritative billing.
    price_in_per_mtok: float = 3.00
    price_out_per_mtok: float = 15.00


@dataclass(frozen=True)
class CompilerConfig:
    """The knowledge compiler (paper Sections 6.1 and 7)."""

    model: str = "claude-sonnet-5"
    max_tokens: int = 32_000
    effort: str = "high"
    # Compression targets (paper Section 6.1 / Table 3).
    levels: dict[str, float] = field(
        default_factory=lambda: {"light": 0.75, "moderate": 0.50, "aggressive": 0.25}
    )
    seed: int = 0


@dataclass(frozen=True)
class WicerConfig:
    """Algorithm 1 parameters."""

    ratio: float = 0.50            # "moderate", paper Table 4
    max_iterations: int = 3        # T
    failure_threshold: int = 1     # score <= 1 is a catastrophic failure
    improvement_threshold: float = 0.10  # break if < 10% improvement and t > 0
    diagnosis_min_words: int = 50
    diagnosis_max_words: int = 100
    random_pin_seed: int = 20260508


@dataclass(frozen=True)
class RagConfig:
    """RAG baseline (paper Section 3.4)."""

    chunk_chars: int = 1_800
    overlap_chars: int = 360
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    index_dir: str = "data/rag_index"


@dataclass(frozen=True)
class Profile:
    name: str
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    wicer: WicerConfig = field(default_factory=WicerConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    results_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, Profile] = {
    "paper": Profile(name="paper"),
    "smoke": Profile(
        name="smoke",
        reader=ReaderConfig(
            model_path="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            n_ctx=16_384,
            n_gpu_layers=99,
        ),
        corpus=CorpusConfig(n_docs_per_topic=6, n_qa_per_doc=5),
        results_dir="results/smoke",
    ),
    "test": Profile(
        name="test",
        reader=ReaderConfig(
            model_path="<fake>",
            port=8299,
            n_ctx=8_192,
            n_gpu_layers=0,
            ctx_margin_tokens=256,
            max_tokens=128,
            request_timeout_s=60.0,
        ),
        corpus=CorpusConfig(n_docs_per_topic=12, n_qa_per_doc=5, data_dir="tests/fixtures"),
        judge=JudgeConfig(max_workers=2, cache_dir=".cache/judge-test"),
        compiler=CompilerConfig(max_tokens=4_000),
        # Diagnosis length is scaled to the fixture. What matters is the ratio
        # of pinned words to the compilation budget: in the paper's setting
        # (~44 failures x ~75 words against a ~28K-word budget) it is ~12%.
        # Keeping the paper's 50-100 words against this 12-document fixture
        # would pin ~30% of the budget and make knowledge displacement dominate
        # an effect it barely perturbs at full scale.
        wicer=WicerConfig(max_iterations=3, diagnosis_min_words=25, diagnosis_max_words=40),
        rag=RagConfig(chunk_chars=600, overlap_chars=120, index_dir="results/test/rag_index"),
        results_dir="results/test",
    ),
}

# 17 RepLiQA topic domains (paper Table 4).  Keys are our slugs; values are the
# `document_topic` strings as they appear in the HuggingFace dataset.
TOPICS: dict[str, str] = {
    "company_policies": "Company Policies",
    "cybersecurity_news": "Cybersecurity News",
    "incident_report": "Incident Report",
    "local_arts_and_culture": "Local Arts and Culture",
    "local_economy": "Local Economy and Market",
    "local_education": "Local Education Systems",
    "local_env_issues": "Local Environmental Issues",
    "local_health": "Local Health and Wellness",
    "local_news": "Local News",
    "local_politics": "Local Politics and Governance",
    "local_sports": "Local Sports and Activities",
    "local_tech": "Local Technology and Innovation",
    "neighborhood": "Neighborhood Stories",
    "news_stories": "News Stories",
    "regional_cuisine": "Regional Cuisine and Recipes",
    "regional_folklore": "Regional Folklore and Myths",
    "small_medium_ent": "Small and Medium Enterprises",
}


def get_profile(name: str | None = None) -> Profile:
    """Resolve a profile by name, then apply WICER_* environment overrides."""
    name = name or os.environ.get("WICER_PROFILE", "paper")
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    profile = PROFILES[name]

    reader_over: dict[str, Any] = {}
    if v := os.environ.get("WICER_READER_HOST"):
        reader_over["host"] = v
    if v := os.environ.get("WICER_READER_PORT"):
        reader_over["port"] = int(v)
    if v := os.environ.get("WICER_MODEL_PATH"):
        reader_over["model_path"] = v
    if v := os.environ.get("WICER_N_CTX"):
        reader_over["n_ctx"] = int(v)
    if v := os.environ.get("WICER_KV_TYPE"):
        reader_over["cache_type_k"] = v
        reader_over["cache_type_v"] = v
    if reader_over:
        profile = replace(profile, reader=replace(profile.reader, **reader_over))

    if v := os.environ.get("WICER_JUDGE_MODEL"):
        profile = replace(profile, judge=replace(profile.judge, model=v))
    if v := os.environ.get("WICER_COMPILER_MODEL"):
        profile = replace(profile, compiler=replace(profile.compiler, model=v))
    # Effort and the output cap compete for the same budget: reasoning tokens
    # bill as completion tokens and count against max_tokens, so a high effort
    # with a tight cap can consume the whole allowance and return no text.
    # Both are tunable without editing this file.
    if v := os.environ.get("WICER_JUDGE_EFFORT"):
        profile = replace(profile, judge=replace(profile.judge, effort=v))
    if v := os.environ.get("WICER_COMPILER_EFFORT"):
        profile = replace(profile, compiler=replace(profile.compiler, effort=v))
    if v := os.environ.get("WICER_COMPILER_MAX_TOKENS"):
        profile = replace(profile, compiler=replace(profile.compiler, max_tokens=int(v)))
    if v := os.environ.get("WICER_DOC_SELECTION"):
        profile = replace(profile, corpus=replace(profile.corpus, doc_selection=v))
    if v := os.environ.get("WICER_RESULTS_DIR"):
        profile = replace(profile, results_dir=v)
    return profile


def resolve(path: str | Path) -> Path:
    """Resolve a config path against the repo root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def add_profile_arg(parser) -> None:
    """Shared --profile flag for every CLI in the repo."""
    parser.add_argument(
        "--profile",
        default=os.environ.get("WICER_PROFILE", "paper"),
        choices=sorted(PROFILES),
        help="run profile (default: %(default)s)",
    )
