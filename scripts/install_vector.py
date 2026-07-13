#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "requests"]
# ///
"""Install (or clear) the steering vector as a live intervention on
jlens-server's /v1 endpoint. Every subsequent /v1 completion is steered.

Usage:
  install_vector.py            install d_bc at layer 48, alpha 1.0
  install_vector.py --clear    remove all live interventions
  install_vector.py --status   show the installed set
"""

import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("JLENS_PROJECT", SCRIPT_DIR.parent))
sys.path.insert(0, str(os.environ.get("JLENS_GGUF", PROJECT / "jlens-gguf")))

from jlens_gguf.client import NativeClient  # noqa: E402

URL = os.environ.get("JLENS_URL", "http://127.0.0.1:8091")
LAYER = int(os.environ.get("STEER_LAYER", "48"))
ALPHA = float(os.environ.get("STEER_ALPHA", "1.0"))
RULE_DIR = Path(os.environ.get("SIDEVEC_RULE",
                               PROJECT / "rules" / "reply-in-english"))
VECTOR = os.environ.get("STEER_VECTOR", str(RULE_DIR / "vector.npz"))

client = NativeClient(URL)

if "--clear" in sys.argv:
    print(client.live_interventions_clear())
elif "--status" in sys.argv:
    print(client.live_interventions_get())
else:
    vec = np.load(VECTOR)["d_bc"][LAYER]
    out = client.live_interventions_set(
        [{"layer": LAYER, "pos_start": 0, "pos_end": -1, "mode": "add",
          "vector": ALPHA * vec}],
        meta={"vector": Path(VECTOR).name, "layer": LAYER, "alpha": ALPHA},
    )
    print(out)
    print(f"installed: L{LAYER} alpha {ALPHA} from {Path(VECTOR).name}")
