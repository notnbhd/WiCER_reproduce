# Sourced by the runner scripts after .env is loaded.  Decides which API serves
# the judge and the compiler, picks the Haiku id that API expects, and stops
# before any GPU time is spent if the key for it is missing.
#
# Anthropic's own API is the default.  To go back to OpenRouter:
#   WICER_JUDGE_BACKEND=openrouter WICER_COMPILER_BACKEND=openrouter bash scripts/...
# The two spell model ids differently (claude-haiku-5-5 vs
# anthropic/claude-haiku-5.5); a slug sent to Anthropic is a 404 an hour in.
export WICER_JUDGE_BACKEND="${WICER_JUDGE_BACKEND:-anthropic}"
export WICER_COMPILER_BACKEND="${WICER_COMPILER_BACKEND:-$WICER_JUDGE_BACKEND}"

_wicer_default_model() {
  if [[ "$1" == openrouter ]]; then echo "anthropic/claude-haiku-5.5"; else echo "claude-haiku-5-5"; fi
}
export WICER_JUDGE_MODEL="${WICER_JUDGE_MODEL:-$(_wicer_default_model "$WICER_JUDGE_BACKEND")}"
export WICER_COMPILER_MODEL="${WICER_COMPILER_MODEL:-$(_wicer_default_model "$WICER_COMPILER_BACKEND")}"

_wicer_check_backend() {   # role backend model
  case "$2" in
    anthropic)
      : "${ANTHROPIC_API_KEY:?the $1 uses the Anthropic API: paste ANTHROPIC_API_KEY into .env first}"
      if [[ "$3" == */* ]]; then
        echo "FAILED: $1 model '$3' is an OpenRouter slug but the backend is anthropic" >&2
        exit 1
      fi ;;
    openrouter)
      : "${OPENROUTER_API_KEY:?the $1 uses OpenRouter: paste OPENROUTER_API_KEY into .env first}" ;;
    fake) ;;
    *) echo "FAILED: unknown $1 backend '$2' (anthropic, openrouter or fake)" >&2; exit 1 ;;
  esac
}
_wicer_check_backend judge "$WICER_JUDGE_BACKEND" "$WICER_JUDGE_MODEL"
_wicer_check_backend compiler "$WICER_COMPILER_BACKEND" "$WICER_COMPILER_MODEL"
echo "judge: $WICER_JUDGE_MODEL via $WICER_JUDGE_BACKEND; compiler: $WICER_COMPILER_MODEL via $WICER_COMPILER_BACKEND"
