#!/usr/bin/env bash
# run-condition.sh: run a rule's task bank against pi for one condition.
#
# Usage:
#   run-condition.sh A|B [output-subdir]
#
# Conditions (rule delivery = injection mode, pi -nc):
#   A: rule appended to the system prompt (base-system.md + rule.md)
#   B: no rule (failure baseline, base-system.md only)
#
# Environment (all optional):
#   SIDEVEC_PROJECT  project root      (default: directory above this script)
#   SIDEVEC_RULE     rule directory    (default: rules/reply-in-english)
#   PI_PROVIDER      pi provider name  (default: llamacpp)
#   PI_MODEL         pi model id       (default: qwen36-27b)
#   ARENA_RUN_DIR    scratch cwd       (default: $TMPDIR/side-vectors-run)
#
# Assumes the model server is already running. Each prompt runs as an
# independent ephemeral pi session. Output: one reply file per prompt +
# summary.tsv with detected language and PASS/FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${SIDEVEC_PROJECT:-$(dirname "$SCRIPT_DIR")}"
RULE_DIR="${SIDEVEC_RULE:-$PROJECT/rules/reply-in-english}"
PROVIDER="${PI_PROVIDER:-llamacpp}"
MODEL="${PI_MODEL:-qwen36-27b}"
ARENA_RUN="${ARENA_RUN_DIR:-${TMPDIR:-/tmp}/side-vectors-run}"

CONDITION="${1:?usage: run-condition.sh A|B [output-subdir]}"
OUT="$PROJECT/results/raw/${2:-harness}/condition-$CONDITION"
TASKS="$RULE_DIR/capture-tasks.txt"
EXPECTED_LANG="en"   # the rule under test: reply in English

[ -f "$TASKS" ] || { echo "no task bank: $TASKS" >&2; exit 2; }

rm -rf "$OUT" "$ARENA_RUN"
mkdir -p "$OUT" "$ARENA_RUN"

PI_FLAGS=(--provider "$PROVIDER" --model "$MODEL" --no-session --no-context-files
          --append-system-prompt "$RULE_DIR/base-system.md")
if [ "$CONDITION" = "A" ]; then
    PI_FLAGS+=(--append-system-prompt "$RULE_DIR/rule.md")
fi

echo "condition=$CONDITION provider=$PROVIDER model=$MODEL rule=$RULE_DIR"
printf 'n\tverdict\ten_share\tfi_share\tsentences\tresult\tprompt\n' > "$OUT/summary.tsv"

n=0
grep -v '^$' "$TASKS" | while IFS= read -r prompt; do
    n=$((n + 1))
    reply="$OUT/$(printf '%02d' "$n").txt"
    # </dev/null: pi reads stdin in -p mode and would consume the task list
    rc=0
    (cd "$ARENA_RUN" && pi "${PI_FLAGS[@]}" -p "$prompt" >"$reply" 2>&1 </dev/null) || rc=$?
    # detect_lang.py prints: <verdict> <en_share> <fi_share> <n_sentences>
    read -r verdict en_share fi_share nsent <<< "$("$SCRIPT_DIR/detect_lang.py" "$reply" || echo '?? 0 0 0')"
    # A non-zero pi exit or an unclassifiable reply is an ERROR, not a data
    # point: a broken server would otherwise language-classify its own error
    # text and record a plausible-looking PASS/FAIL.
    if [ "$rc" -ne 0 ] || [ "$verdict" = "??" ]; then
        result=ERROR
    elif [ "$verdict" = "$EXPECTED_LANG" ]; then result=PASS; else result=FAIL; fi
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$verdict" "$en_share" "$fi_share" "$nsent" "$result" "$prompt" >> "$OUT/summary.tsv"
    echo "  $n: $verdict (en=$en_share fi=$fi_share) $result"
done

total=$(($(wc -l < "$OUT/summary.tsv") - 1))
fails=$(grep -c $'\tFAIL\t' "$OUT/summary.tsv" || true)
echo "condition $CONDITION: $fails/$total replies NOT in $EXPECTED_LANG"
