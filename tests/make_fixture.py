#!/usr/bin/env python3
"""Generate the offline test fixture: a 6-document synthetic topic.

The documents are built so the pipeline's behaviour is observable end to end:
each carries ten distinct, checkable facts spread evenly through ~330 words,
and the five QA pairs per document target facts at positions 1, 3, 5, 7 and 9.
Moderate compression keeps roughly the first half of each document, so the
early facts survive and the late ones are dropped -- which is exactly the
catastrophic-failure pattern WiCER exists to repair.

Deterministic: rerunning produces byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fixtures" / "mini_topic"

# Twelve documents, not six. With too few documents a random 50-100 word
# passage from a random document lands on the needed fact often enough that the
# random-pinning control looks as good as diagnosis -- which would hide the
# paper's central claim behind a small-sample artefact.
TOWNS = [
    ("Ardenmoor", "Kestrel Valley"), ("Brightwater", "Lowen Basin"),
    ("Corrindale", "Harrow Plateau"), ("Dunhollow", "Vance Reach"),
    ("Everly Cross", "Marrow Fields"), ("Fennwick", "Stellan Coast"),
    ("Garrowhithe", "Pellin Downs"), ("Halloway", "Quintan Moor"),
    ("Ilmarsh", "Rundell Heath"), ("Jerrow Bank", "Torrin Shelf"),
    ("Kellworth", "Ubrecht Flats"), ("Larkendale", "Wyvern Gap"),
]

FIRST_NAMES = ["Ilse", "Tobias", "Wren", "Casimir", "Odile", "Petra",
               "Aurel", "Marit", "Ferran", "Sabine", "Leonid", "Nadja"]
SURNAMES = ["Hallam", "Krause", "Odabas", "Lindqvist", "Marchetti", "Nakagawa",
            "Vogel", "Brennan", "Sorokin", "Achterberg", "Duclos", "Iversen"]
RIVERS = ["Suden", "Calder", "Ophir", "Tessin", "Vardo", "Ryhope",
          "Almen", "Brisk", "Cadell", "Denne", "Elster", "Fennow"]
FESTIVALS = ["Lantern", "Harvest", "Kiln", "Saltwind", "Thresher", "Beacon",
             "Ember", "Weirstone", "Copperleaf", "Gallowsmoor", "Hollybank", "Ironvane"]
MILLS = ["Aldwin", "Bramm", "Coverly", "Denhart", "Esker", "Fallow",
         "Greelin", "Hobb", "Ivery", "Jarrow", "Kelm", "Lowther"]

# Filler carries no answerable content. It exists to push the later facts past
# the compression budget, which is what manufactures recoverable failures.
FILLER = (
    "Municipal records describe the period as one of steady administrative "
    "consolidation. Committee minutes from the era note routine discussion of "
    "zoning, drainage and seasonal staffing. Contemporary observers considered "
    "the arrangements unremarkable but durable. Subsequent reviews found the "
    "documentation complete and internally consistent. Clerks maintained the "
    "registers in a uniform hand throughout. The surviving correspondence is "
    "largely procedural in character. Auditors raised no material objections in "
    "any year of the sequence."
)


FILLER_SENTENCES = [s.rstrip(".") + "." for s in FILLER.split(". ")]


def facts_for(i: int, town: str, region: str, rng: random.Random) -> list[tuple[str, str, str]]:
    """Ten (statement, question, answer) triples for one document."""
    year = 1890 + i * 7
    mayor = f"{FIRST_NAMES[i]} {SURNAMES[i]}"
    river = f"{RIVERS[i]} River"
    festival = f"{FESTIVALS[i]} Festival"
    mill = f"{MILLS[i]} Mill"
    return [
        (f"{town} was chartered in {year} on the eastern bank of the {river}.",
         f"In what year was {town} chartered?", f"{town} was chartered in {year}."),
        (f"The town lies in the {region}, some {38 + i * 9} kilometres from the regional seat.",
         f"How far is {town} from the regional seat?", f"About {38 + i * 9} kilometres."),
        (f"Its first mayor was {mayor}, who served {6 + i} consecutive terms.",
         f"Who was the first mayor of {town}?", f"{mayor} was the first mayor."),
        (f"{mill} opened in {year + 14} and employed {120 + i * 35} workers at its peak.",
         f"How many workers did {mill} employ at its peak?", f"{120 + i * 35} workers."),
        (f"The {festival} has been held every autumn since {year + 22}.",
         f"Since what year has the {festival} been held?", f"Since {year + 22}."),
        (f"A flood in {year + 31} damaged {17 + i * 3} bridges across the {region}.",
         f"How many bridges were damaged in the {year + 31} flood?", f"{17 + i * 3} bridges."),
        (f"The municipal library holds {4200 + i * 610} volumes, including the {town} Ledger.",
         f"How many volumes does the {town} municipal library hold?", f"{4200 + i * 610} volumes."),
        (f"Population peaked at {9400 + i * 1250} residents in {year + 58}.",
         f"What was the peak population of {town}?", f"{9400 + i * 1250} residents in {year + 58}."),
        (f"The {river} bridge was rebuilt in {year + 71} at a cost of {2 + i} million marks.",
         f"What did rebuilding the {river} bridge cost?", f"{2 + i} million marks."),
        (f"Since {year + 84} the town council has met in the restored {mill} granary.",
         f"Where has the {town} council met since {year + 84}?",
         f"In the restored {mill} granary."),
    ]


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    docs, qa = [], []

    for i, (town, region) in enumerate(TOWNS):
        rng = random.Random(1000 + i)
        triples = facts_for(i, town, region, rng)
        paragraphs = [f"{town}: a municipal profile."]
        for j, (statement, _q, _a) in enumerate(triples):
            paragraphs.append(statement)
            # Filler between facts pushes the later ones past the compression
            # budget, which is what creates recoverable failures.
            paragraphs.append(FILLER_SENTENCES[j % len(FILLER_SENTENCES)])
        doc_id = f"mini{i:02d}"
        docs.append({"document_id": doc_id, "text": " ".join(paragraphs)})

        # Question numbering is rotated per document so that the lowest
        # question_id -- what probe selection picks -- is not systematically the
        # earliest, best-surviving fact. In RepLiQA the id order carries no
        # positional meaning either; without the rotation the probe set would
        # always pass and the WiCER loop would have nothing to diagnose.
        fact_indices = (0, 2, 4, 6, 8)
        for slot, k in enumerate(fact_indices):
            _s, question, answer = triples[k]
            q_num = (slot - i) % len(fact_indices) + 1
            qa.append({
                "question_id": f"{doc_id}-q{q_num}",
                "document_id": doc_id,
                "question": question,
                "answer": answer,
                "long_answer": triples[k][0],
            })
        qa.sort(key=lambda r: (r["document_id"], r["question_id"]))

    _write(OUT / "docs.jsonl", docs)
    _write(OUT / "qa.jsonl", qa)

    h = hashlib.sha256()
    for d in docs:
        h.update(d["document_id"].encode()); h.update(b"\0")
        h.update(d["text"].encode()); h.update(b"\0")
    for q in qa:
        h.update(q["question_id"].encode()); h.update(b"\0")
        h.update(q["question"].encode()); h.update(b"\0")
        h.update(q["answer"].encode()); h.update(b"\0")

    manifest = {
        "topic": "mini_topic",
        "hf_topic": "(synthetic fixture)",
        "hf_repo": None,
        "split": None,
        "n_documents": len(docs),
        "n_qa": len(qa),
        "n_words": sum(len(d["text"].split()) for d in docs),
        "document_ids": [d["document_id"] for d in docs],
        "dropped_unanswerable": 0,
        "documents_with_fewer_qa": [],
        "subsample_rule": "synthetic fixture generated by tests/make_fixture.py",
        "target_n_documents": len(docs),
        "target_n_qa_per_doc": 5,
        "content_sha256": h.hexdigest(),
        "built_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest['n_documents']} docs, {manifest['n_qa']} qa, "
          f"{manifest['n_words']} words, sha={manifest['content_sha256'][:12]}")


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build()
