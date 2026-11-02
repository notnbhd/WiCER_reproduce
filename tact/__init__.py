"""TACT: Typed, Anchored Compilation with conTracts.

A new method built on this reproduction, not part of it. TACT reuses the
reproduction's corpus, reader, judge and benchmark runner unchanged, so its
wikis are scored exactly as WiCER's are; it never modifies ``wicer/``. The
method is specified in ``docs/tact/method_draft_v2.md`` (section numbers
in this package refer to that draft).

Results go to ``result-tact/`` so they never mix with the reproduction's
baseline in ``result-sweep/`` (git tag ``repro-sweep-2026-10-24``).
"""

from __future__ import annotations

import os
from dataclasses import replace

from wicer_core.config import Profile
from wicer_core.config import get_profile as _wicer_profile

RESULTS_DIR = "result-tact"


def get_profile(name: str | None = None) -> Profile:
    """The WiCER profile, with results redirected away from the baseline.

    ``WICER_RESULTS_DIR`` still wins, as it does everywhere else. The paper
    profile writes to ``result-tact/``; the test and smoke profiles write to a
    ``tact`` subdirectory of their own (gitignored) results directory.
    """
    profile = _wicer_profile(name)
    if os.environ.get("WICER_RESULTS_DIR"):
        return profile
    if profile.results_dir == "results":
        return replace(profile, results_dir=RESULTS_DIR)
    return replace(profile, results_dir=f"{profile.results_dir}/tact")
