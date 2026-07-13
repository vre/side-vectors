#!/usr/bin/env bash
# run-phase1a-pretest.sh: phase 1a pre-test: condition B failure rate +
# condition A baseline, in one server lifetime.
#
# Run under a server wrapper, e.g.:
#   ./run_pi.sh <preset> --run /path/to/run-phase1a-pretest.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run-condition.sh" B phase1a-pretest
"$SCRIPT_DIR/run-condition.sh" A phase1a-pretest
