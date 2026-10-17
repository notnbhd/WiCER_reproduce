#!/usr/bin/env python3
"""Score RAG answers with the identical Claude judge and rubric.

Paper Section 3.5: "the identical judge and rubric are applied to both
conditions".  This is deliberately a thin wrapper over benchmark/evaluate.py
rather than a second implementation -- a second judge would silently break
every cross-condition comparison in the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.evaluate import main as evaluate_main
from rag_benchmark.run_rag_benchmark import CONDITION

if __name__ == "__main__":
    raise SystemExit(evaluate_main(default_condition=CONDITION))
