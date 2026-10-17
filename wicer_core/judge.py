"""LLM-as-judge scoring (paper Section 3.5, rubric Appendix B).

The judge score is the dependent variable in every table in the paper, so
three things get special care here:

* The system prompt is the Appendix B text verbatim.  Do not reword it.
* The model ID is pinned in config *and* the ``model`` the API echoes back is
  recorded on every row, because spec.md Section 9 names judge drift as a
  reproduction risk.
* Responses are cached content-addressed on disk.  The WiCER loop re-scores
  the same probe answers across iterations and across reruns; without the
  cache the same judgement gets paid for many times.

Score 1 is a "catastrophic failure" and is tracked separately from the mean
throughout (paper Table 3/4, spec.md Section 2).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from wicer_core.config import JudgeConfig, resolve

# Appendix B, verbatim.  The '--' double hyphens are the paper's.
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator comparing a generated answer against a gold-standard answer.

Score the generated answer from 1 to 5:
  5 = Perfect -- captures all key facts, accurate
  4 = Good -- mostly correct, minor omissions
  3 = Acceptable -- partially correct, misses details
  2 = Poor -- significant errors or major omissions
  1 = Wrong -- incorrect, irrelevant, misses the point

Respond ONLY with valid JSON:
{"score": <1-5>, "reasoning": "<explanation>"}"""

# The paper gives the system prompt but not the user-message layout.  Ours:
USER_TEMPLATE = """Question: {question}

Gold-standard answer: {gold}

Generated answer: {generated}"""


@dataclass
class Judgement:
    score: int
    reasoning: str
    model: str            # what the API actually served
    cached: bool = False
    parse_failure: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeRequest:
    question: str
    gold: str
    generated: str
    key: str = ""  # caller-side identifier, passed through untouched


class Judge:
    """Scores answers.  Thread-safe; judge calls are independent."""

    def __init__(
        self,
        cfg: JudgeConfig,
        backend: str | None = None,
        openrouter_client: Any = None,
    ):
        self.cfg = cfg
        self.backend = backend or os.environ.get("WICER_JUDGE_BACKEND", "anthropic")
        self._or_client = openrouter_client
        # OpenRouter reports what each call actually cost; the price table is
        # only a fallback for the paths that report nothing.
        self.measured_cost_usd = 0.0
        self.cache_dir = resolve(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_api_calls = 0
        self.n_cache_hits = 0
        self.n_parse_failures = 0

    # -- client ---------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    @property
    def openrouter(self):
        if self._or_client is None:
            from wicer_core.openrouter import OpenRouterClient

            self._or_client = OpenRouterClient()
        return self._or_client

    # -- caching --------------------------------------------------------------

    def _cache_path(self, user: str) -> Path:
        h = hashlib.sha256(
            "\x00".join([JUDGE_SYSTEM_PROMPT, user, self.cfg.model]).encode()
        ).hexdigest()
        return self.cache_dir / f"{h}.json"

    # -- scoring --------------------------------------------------------------

    def score(self, req: JudgeRequest) -> Judgement:
        user = USER_TEMPLATE.format(
            question=req.question, gold=req.gold, generated=req.generated
        )
        path = self._cache_path(user)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                with self._lock:
                    self.n_cache_hits += 1
                return Judgement(**{**data, "cached": True})
            except (json.JSONDecodeError, TypeError):
                path.unlink(missing_ok=True)

        if self.backend == "fake":
            judgement = self._score_fake(req)
        elif self.backend == "openrouter":
            judgement = self._score_openrouter(user)
        else:
            judgement = self._score_anthropic(user)
        with self._lock:
            self.n_api_calls += 1
            self.input_tokens += judgement.input_tokens
            self.output_tokens += judgement.output_tokens
            if judgement.parse_failure:
                self.n_parse_failures += 1
        path.write_text(json.dumps(judgement.to_dict()))
        return judgement

    def score_many(self, requests: Sequence[JudgeRequest]) -> list[Judgement]:
        if not requests:
            return []
        workers = max(1, min(self.cfg.max_workers, len(requests)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.score, requests))

    def _score_openrouter(self, user: str) -> Judgement:
        """Score via OpenRouter.

        No cache breakpoint: judge prompts differ on every question, so a
        breakpoint would pay the write premium and never be read.
        """
        from wicer_core.config import resolve_openrouter_model
        from wicer_core.openrouter import OpenRouterError

        try:
            reply = self.openrouter.chat(
                model=resolve_openrouter_model(self.cfg.model),
                system=JUDGE_SYSTEM_PROMPT,
                user=user,
                max_tokens=self.cfg.max_tokens,
                effort=self.cfg.effort,
                cache_system=False,
            )
        except OpenRouterError as e:
            # An unusable response is one bad score, not a dead sweep: the
            # judge runs 54K times and the compiler runs 150.  Record it the
            # same way an unparseable answer is recorded, never silently.
            return Judgement(
                score=1,
                reasoning=f"PARSE FAILURE: {e}"[:300],
                model=self.cfg.model,
                parse_failure=True,
            )
        with self._lock:
            self.measured_cost_usd += reply.cost_usd

        parsed = _parse_judgement(reply.text)
        if parsed is not None:
            score, reasoning = parsed
            return Judgement(
                score=score,
                reasoning=reasoning,
                model=reply.served_model,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
            )
        # Same contract as the Anthropic path: never guess a score, record it.
        return Judgement(
            score=1,
            reasoning=f"PARSE FAILURE: {reply.text[:200]}",
            model=reply.served_model,
            parse_failure=True,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
        )

    def _score_anthropic(self, user: str) -> Judgement:
        import anthropic

        last_text = ""
        in_tok = out_tok = 0
        served_model = self.cfg.model
        # One retry: a judge that emits prose instead of JSON usually complies
        # the second time.  A second failure is recorded, never silently dropped.
        for attempt in range(2):
            try:
                resp = self.client.messages.create(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    system=JUDGE_SYSTEM_PROMPT,
                    thinking={"type": "disabled"},
                    output_config={"effort": self.cfg.effort},
                    messages=[{"role": "user", "content": user}],
                )
            except anthropic.APIStatusError as e:
                if attempt == 0 and e.status_code >= 500:
                    continue
                raise
            served_model = resp.model
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            last_text = "".join(b.text for b in resp.content if b.type == "text")
            parsed = _parse_judgement(last_text)
            if parsed is not None:
                score, reasoning = parsed
                return Judgement(
                    score=score,
                    reasoning=reasoning,
                    model=served_model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )

        return Judgement(
            score=1,
            reasoning=f"JUDGE_PARSE_FAILURE: {last_text[:300]}",
            model=served_model,
            parse_failure=True,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    def _score_fake(self, req: JudgeRequest) -> Judgement:
        """Deterministic offline stand-in: token recall of the gold answer.

        Used by the test harness so the WiCER loop produces genuine failures,
        diagnoses and convergence behaviour with no API spend.  Never used in
        a real run.

        Numeric answers are judged strictly.  Plain token recall hands out
        partial credit when an answer names the right *kind* of thing with the
        wrong figure -- "10,300 volumes" against a gold of "6,030 volumes"
        shares the word "volumes" and scores 3.  The rubric calls that a 1
        ("incorrect"), and the distinction matters: without it, a random pinned
        passage that merely mentions libraries scores as well as the diagnosed
        fact that names the number, and the ablation this repo exists to run
        shows no separation.
        """
        gold_numbers = _numbers(req.gold)
        if gold_numbers and not (gold_numbers & _numbers(req.generated)):
            return Judgement(
                score=1,
                reasoning="fake judge: gold figures absent from the answer",
                model="fake",
            )

        gold = set(_content_words(req.gold))
        gen = set(_content_words(req.generated))
        if not gold:
            recall = 1.0
        else:
            recall = len(gold & gen) / len(gold)
        if recall >= 0.85:
            score = 5
        elif recall >= 0.6:
            score = 4
        elif recall >= 0.4:
            score = 3
        elif recall >= 0.2:
            score = 2
        else:
            score = 1
        return Judgement(
            score=score,
            reasoning=f"fake judge: gold-token recall {recall:.2f}",
            model="fake",
        )

    # -- accounting -----------------------------------------------------------

    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens / 1e6 * self.cfg.price_in_per_mtok
            + self.output_tokens / 1e6 * self.cfg.price_out_per_mtok
        )

    def usage_summary(self) -> dict[str, Any]:
        return {
            "judge_model": self.cfg.model,
            "backend": self.backend,
            "api_calls": self.n_api_calls,
            "cache_hits": self.n_cache_hits,
            "parse_failures": self.n_parse_failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
            "measured_cost_usd": round(self.measured_cost_usd, 6),
            "cost_source": "openrouter" if self.backend == "openrouter" else "estimate",
        }


_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "was", "were", "and", "or", "for",
    "on", "at", "by", "with", "as", "that", "this", "it", "its", "be", "are",
    "from", "has", "have", "had", "he", "she", "they", "his", "her", "their",
}


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    """Numeric tokens, comma-separators normalised away."""
    return {m.replace(",", "").rstrip(".") for m in _NUMBER_RE.findall(text)}


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_judgement(text: str) -> tuple[int, str] | None:
    """json.loads, then a braces-regex rescue.  Returns None if both fail."""
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "score" not in obj:
            continue
        try:
            score = int(obj["score"])
        except (TypeError, ValueError):
            continue
        if 1 <= score <= 5:
            return score, str(obj.get("reasoning", ""))
    return None


def _json_candidates(text: str) -> Iterable[str]:
    stripped = text.strip()
    yield stripped
    fenced = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()
    if fenced != stripped:
        yield fenced
    # Greedy outer-brace span first, then any non-greedy inner object.
    first, last = stripped.find("{"), stripped.rfind("}")
    if first != -1 and last > first:
        yield stripped[first : last + 1]
    yield from _JSON_RE.findall(stripped)
