#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "requests"]
# ///
"""Measure the steering intervention's speed cost on jlens-server.

Same prompt, same generation length, N reps with and without the vector.
Reports prefill and decode tok/s from the server's own timings.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("JLENS_PROJECT", SCRIPT_DIR.parent))
sys.path.insert(0, str(os.environ.get("JLENS_GGUF", PROJECT / "jlens-gguf")))

from jlens_gguf.client import NativeClient  # noqa: E402

URL = os.environ.get("JLENS_URL", "http://127.0.0.1:8091")
RULE_DIR = Path(os.environ.get("SIDEVEC_RULE",
                               PROJECT / "rules" / "reply-in-english"))
LAYER = int(os.environ.get("STEER_LAYER", "48"))
REPS = int(os.environ.get("PERF_REPS", "3"))
GEN = int(os.environ.get("PERF_GEN", "200"))

client = NativeClient(URL)
_vec_path = PROJECT / "results" / "raw" / "phase1a-deltas" / "deltas.npz"
if not _vec_path.exists():
    _vec_path = RULE_DIR / "vector.npz"   # shipped vector, no capture needed
vec = np.load(_vec_path)["d_bc"]

prompt = client.apply_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "Selitä lyhyesti miten aurinkopaneeli toimii ja mitä komponentteja tarvitaan mökkijärjestelmään."}],
    add_assistant=True,
) + "<think>\n\n</think>\n\n"
tokens = client.tokenize(prompt, add_special=False, parse_special=True)
print(f"prompt tokens: {len(tokens)}, gen: {GEN}, reps: {REPS}")

def run(ivs, label):
    rows = []
    for _ in range(REPS):
        res = client.forward(tokens, capture=False, interventions=ivs,
                             n_predict=GEN,
                             sampling={"greedy": True, "seed": 42})
        rows.append(res.timings)
    print(f"--- {label}")
    keys = sorted(rows[0]) if rows[0] else []
    if not keys:
        print("  (no timings in response)")
        return
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"  {k}: " + ", ".join(f"{v:.1f}" for v in vals))

run(None, "no intervention")
run([{"layer": LAYER, "pos_start": 0, "pos_end": -1, "mode": "add",
      "vector": 1.0 * vec[LAYER]}], f"vector at L{LAYER}, alpha 1.0")
