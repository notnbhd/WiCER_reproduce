"""TACT diagnosis (draft section 3.5.2) and neighbourhood closure (3.5.1)."""

from __future__ import annotations

from tact.diagnose import (
    MISS_EXTRACT, MISS_READ, MISS_RELATION, MISS_RENDER,
    diagnose, extended_answer, neighbourhood, paragraphs,
)
from tact.ledger import Claim
from tact.probes import Probe

TEXT = "Mill opened in 1904.\n\nIt employed 120.\nA flood closed it.\n\nThe fair began."
P2 = TEXT.index("It employed")
FLOOD = TEXT.index("A flood")
FAIR = TEXT.index("The fair")


def _c(id, start, length=5, *, ref=False, v="x"):
    return Claim(id=id, doc="d", e="E", r="r", v=v, ref=ref, verbatim=["."],
                 spans=[[start, start + length]])


def _probe(relational=False):
    return Probe(id="d#p1", doc="d", question="?", answer="!", verbatim=["."],
                 spans=[[P2, P2 + 5]], kind="other", relational=relational)


def test_paragraphs_split_on_blank_lines():
    assert [TEXT[s:e].strip() for s, e in paragraphs(TEXT)] == [
        "Mill opened in 1904.", "It employed 120.\nA flood closed it.", "The fair began."]


def test_neighbourhood_closes_over_the_paragraph_and_nothing_else():
    claims = [_c("d#1", 0), _c("d#2", P2), _c("d#3", FLOOD), _c("d#4", FAIR)]
    assert neighbourhood(["d#2"], claims, TEXT) == ["d#2", "d#3"]
    assert neighbourhood([], claims, TEXT) == []


def test_extended_answer_adds_relations_into_a_relational_answer():
    claims = [_c("d#2", P2), _c("d#3", FLOOD), _c("d#9", FAIR, ref=True, v="d#2")]
    assert extended_answer(_probe(relational=True), ["d#2"], claims) == ["d#2", "d#9"]
    assert extended_answer(_probe(relational=False), ["d#2"], claims) == ["d#2"]


def test_the_four_way_diagnosis():
    rel = _c("d#9", FAIR, ref=True, v="d#2")
    by_id = {c.id: c for c in [_c("d#2", P2), _c("d#3", FLOOD), rel]}
    plain, relational = _probe(), _probe(relational=True)

    assert diagnose(plain, [], [], set(), by_id) == MISS_EXTRACT
    assert diagnose(plain, ["d#2"], ["d#2"], set(), by_id) == MISS_RENDER
    # partly realized: the remainder is a render loss, not a reader failure
    assert diagnose(plain, ["d#2", "d#3"], ["d#2", "d#3"], {"d#2"}, by_id) == MISS_RENDER
    assert diagnose(plain, ["d#2"], ["d#2"], {"d#2"}, by_id) == MISS_READ
    # relational, everything realized, no relation claim joins it
    assert diagnose(relational, ["d#2", "d#3"], ["d#2", "d#3"], {"d#2", "d#3"}, by_id) \
        == MISS_RELATION
    # the relation exists in the ledger but was not rendered
    assert diagnose(relational, ["d#2"], ["d#2", "d#9"], {"d#2"}, by_id) == MISS_RENDER
    # the relation is rendered and the reader still failed
    assert diagnose(relational, ["d#2"], ["d#2", "d#9"], {"d#2", "d#9"}, by_id) == MISS_READ


def test_a_block_without_paragraph_breaks_is_cut_at_sentences():
    long = " ".join(f"Sentence number {i} is here." for i in range(80))   # ~2,200 chars
    paras = paragraphs(long + "\n\nShort tail.")
    assert len(paras) > 3
    assert all(e - s <= 700 for s, e in paras[:-1])
    assert long[paras[0][0]:paras[0][1]].startswith("Sentence number 0")
    assert long[paras[0][0]:paras[0][1]].rstrip().endswith(".")
