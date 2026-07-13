#!/usr/bin/env bash
# verify-context-loading.sh: one-time empirical check of pi.dev context-file behavior.
#
# v2: pi session JSONL does NOT store the system prompt, so grep-the-session
# fails structurally (v1 finding, 2026-07-13). Instead use a self-report probe:
# ask the model to quote any reply-language rule it was given. Marker fragment
# "save tokens" appears iff the context file reached the model.
#
# Runs:
#   1. discovery mode, arena with CLAUDE.md          → rule expected
#   2. discovery mode, arena with AGENTS.md          → rule expected (filename check)
#   3. -nc mode, arena with both files               → NONE expected
#   4. discovery mode, cwd inside project tree       → project CLAUDE.md marker
#      (parent traversal)
#
# Usage: start your model server first, then run this script.
# Environment: SIDEVEC_PROJECT, SIDEVEC_RULE, PI_PROVIDER, PI_MODEL,
# ARENA_RUN_DIR (same defaults as run-condition.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${SIDEVEC_PROJECT:-$(dirname "$SCRIPT_DIR")}"
RULE_DIR="${SIDEVEC_RULE:-$PROJECT/rules/reply-in-english}"
ARENA_RUN="${ARENA_RUN_DIR:-${TMPDIR:-/tmp}/side-vectors-run}"
OUT="$PROJECT/results/raw/context-verify"
PROVIDER="${PI_PROVIDER:-llamacpp}"
MODEL="${PI_MODEL:-qwen36-27b}"

PROBE_LANG='If your instructions contain a rule about which language you must reply in, quote that rule verbatim. Otherwise reply with exactly: NONE'
PROBE_PROJECT='If your instructions contain a sentence about files being test subjects, quote it verbatim. Otherwise reply with exactly: NONE'

ARENA_FRAGMENT="save tokens"          # from arena rule "…reply in English, to save tokens."
PROJECT_FRAGMENT="test subjects"       # from this project's CLAUDE.md arena rule

rm -rf "$OUT" "$ARENA_RUN"
mkdir -p "$OUT"

run_pi() { # $1=cwd $2=label $3=probe $4=extra flags
    local cwd="$1" label="$2" probe="$3" flags="${4:-}"
    mkdir -p "$OUT/$label"
    # shellcheck disable=SC2086
    (cd "$cwd" && pi --provider "$PROVIDER" --model "$MODEL" --no-session \
        $flags -p "$probe" >"$OUT/$label/stdout.txt" 2>&1) || true
}

check() { # $1=label $2=fragment $3=expect(present|absent)
    local label="$1" fragment="$2" expect="$3" found
    if grep -qiF "$fragment" "$OUT/$label/stdout.txt" 2>/dev/null; then
        found=present
    else
        found=absent
    fi
    if [ "$found" = "$expect" ]; then
        echo "PASS  $label: '$fragment' $expect"
    else
        echo "FAIL  $label: '$fragment' expected $expect, got $found"
        echo "      stdout: $(head -c 300 "$OUT/$label/stdout.txt" 2>/dev/null | tr '\n' ' ')"
        FAILURES=$((FAILURES + 1))
    fi
}

FAILURES=0

echo "=== 1. discovery, arena CLAUDE.md ==="
mkdir -p "$ARENA_RUN"
cat "$RULE_DIR/base-system.md" "$RULE_DIR/rule.md" > "$ARENA_RUN/CLAUDE.md"
run_pi "$ARENA_RUN" discovery-claudemd "$PROBE_LANG" ""
check discovery-claudemd "$ARENA_FRAGMENT" present

echo "=== 2. discovery, arena AGENTS.md (filename check) ==="
rm -rf "$ARENA_RUN"
mkdir -p "$ARENA_RUN"
cat "$RULE_DIR/base-system.md" "$RULE_DIR/rule.md" > "$ARENA_RUN/AGENTS.md"
run_pi "$ARENA_RUN" discovery-agentsmd "$PROBE_LANG" ""
check discovery-agentsmd "$ARENA_FRAGMENT" present

echo "=== 3. -nc, arena with BOTH files ==="
cp "$ARENA_RUN/AGENTS.md" "$ARENA_RUN/CLAUDE.md"
run_pi "$ARENA_RUN" nc-both "$PROBE_LANG" "--no-context-files"
check nc-both "$ARENA_FRAGMENT" absent

echo "=== 4. discovery, cwd inside project (parent traversal) ==="
run_pi "$PROJECT/rules/reply-in-english" discovery-parent "$PROBE_PROJECT" ""
check discovery-parent "$PROJECT_FRAGMENT" present

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL CHECKS PASS"
else
    echo "$FAILURES CHECK(S) FAILED: inspect $OUT"
    exit 1
fi
