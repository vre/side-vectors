#!/usr/bin/env bash
# demo.sh: one key turn. Starts the server, asks the model one Finnish
# question twice (with and without the steering vector) and prints both
# answers side by side. Stops the server on exit.
#
# Follow SETUP.md first (build server, get model).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JLENS_GGUF_DIR="${JLENS_GGUF_DIR:-$ROOT/jlens-gguf}"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
PORT="${PORT:-8091}"

MODEL="$(find "$MODELS_DIR" \( -name '*.gguf' \) | head -1)"
[ -n "$MODEL" ] || { echo "no model in $MODELS_DIR, see SETUP.md" >&2; exit 1; }

echo "== starting jlens-server ($MODEL)"
"$JLENS_GGUF_DIR/native/jlens-server" -m "$MODEL" -ngl 99 -fa on -c 4096 \
    --port "$PORT" >/tmp/jlens-demo.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

until curl -s -o /dev/null "http://127.0.0.1:$PORT/props"; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died, see /tmp/jlens-demo.log" >&2; exit 1; }
    sleep 3
done
echo "== server ready"

export JLENS_URL="http://127.0.0.1:$PORT"
OUT="$ROOT/results/raw/demo"
mkdir -p "$OUT"
cp "$ROOT/rules/reply-in-english/vector.npz" "$OUT/deltas.npz"

echo "== asking in Finnish WITHOUT the vector"
"$ROOT/scripts/capture_deltas.py" --out "$OUT" steer \
    --vector d_bc --band 48-48 --alpha 0.0 --n-prompts 1 >/dev/null

echo "== asking the same WITH the vector (layer 48, alpha 1.0)"
"$ROOT/scripts/capture_deltas.py" --out "$OUT" steer \
    --vector d_bc --band 48-48 --alpha 1.0 --n-prompts 1 >/dev/null

python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
off = json.loads((out / "steer-d_bc-L48-48-a0.0.json").read_text())[0]
on = json.loads((out / "steer-d_bc-L48-48-a1.0.json").read_text())[0]
print()
print("QUESTION (Finnish):", off["prompt"])
print()
print("--- without vector ---")
print(off["reply"][:400])
print()
print("--- with vector (20 KB, layer 48) ---")
print(on["reply"][:400])
EOF

echo
echo "== done (server stops now)"
