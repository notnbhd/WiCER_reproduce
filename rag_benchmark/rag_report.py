#!/usr/bin/env python3
"""RAG aggregate report: quality metrics plus retrieval accuracy.

Restricts benchmark/report.py to the RAG condition; the retrieval-accuracy
section (paper Section 5.1) is rendered there for any condition that carries
retrieval fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.report import main as report_main
from rag_benchmark.run_rag_benchmark import CONDITION

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--condition" not in argv:
        argv += ["--condition", CONDITION]
    if "--out" not in argv:
        argv += ["--out", "rag_report.md"]
    raise SystemExit(report_main(argv))
