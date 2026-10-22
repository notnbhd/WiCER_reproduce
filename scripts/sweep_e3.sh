#!/usr/bin/env bash
# E3 across topics, one run_e3_topic.sh at a time.
#
#   bash scripts/sweep_e3.sh                       # every topic not already done
#   TOPICS="local_health local_news" bash scripts/sweep_e3.sh
#   DRY_RUN=1 bash scripts/sweep_e3.sh             # print the plan, run nothing
#
# A topic is done when results/<topic>/.e3_done exists, written only after
# run_e3_topic.sh exits 0.  Re-running skips those, because wicer/compile.py
# recompiles (and respends) on every call and loop.py cannot resume mid-topic;
# a crash costs at most the topic in flight, about 80 min and $1.20.
#
# local_economy and local_arts_and_culture were piloted on an earlier rental
# and live in results-rental/, not results/ here, so they are in SKIP by
# default.  Set SKIP="" to rerun them.
#
# regional_cuisine and regional_folklore exceed the reader's window (paper
# Section 5, footnote 2): run_benchmark.py records them as skipped for FC raw
# and the rest of E3 still runs, so they get blind and WiCER scores but no
# recovery rate.  They go last.  Their sizes (97,427 and 97,322 tokens) were
# measured with the reader's own tokenizer.
#
# A failed topic is logged and the sweep moves on; the exit status is nonzero
# if any failed.  Pull results while this runs: scripts/pull_results.sh --watch.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Roughly slackest fit first, so an early failure is more likely a pipeline
# fault than a context-fit one.
DEFAULT_TOPICS=(
  company_policies local_health local_news local_education incident_report
  neighborhood small_medium_ent cybersecurity_news local_env_issues
  local_politics local_sports news_stories local_tech
  regional_cuisine regional_folklore
)
read -r -a TOPIC_LIST <<<"${TOPICS:-${DEFAULT_TOPICS[*]}}"
read -r -a SKIP_LIST <<<"${SKIP-local_economy local_arts_and_culture}"

in_list() { local x=$1; shift; for y in "$@"; do [[ "$x" == "$y" ]] && return 0; done; return 1; }

FAILED=()
for topic in "${TOPIC_LIST[@]}"; do
  if in_list "$topic" "${SKIP_LIST[@]}"; then echo "skip  $topic (SKIP)"; continue; fi
  if [[ -f "results/$topic/.e3_done" ]]; then echo "skip  $topic (done)"; continue; fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "would run $topic"; continue; fi

  printf '\n\033[1m######## %s  (%s)\033[0m\n' "$topic" "$(date +%H:%M:%S)"
  if bash scripts/run_e3_topic.sh "$topic"; then
    touch "results/$topic/.e3_done"
  else
    echo "FAILED: $topic" >&2
    FAILED+=("$topic")
  fi
done

if ((${#FAILED[@]})); then
  echo "failed topics: ${FAILED[*]}" >&2
  exit 1
fi
