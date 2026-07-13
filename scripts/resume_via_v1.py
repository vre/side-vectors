#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "requests", "langdetect"]
# ///
"""Run a long steered session through the /v1 endpoint, fast.

The /v1 path keeps a KV prefix cache, so each turn only prefills the new
tokens instead of the whole growing history (the interactive-sim path uses
/jlens/forward, which re-prefills everything every turn). Install the
steering vector server-side first (install_vector.py); it then applies to
every /v1 completion, so this measures how the vector holds over a real
multi-turn session.

Usage:
  install_vector.py                              # install the vector first
  resume_via_v1.py                    [TURNS]    # fresh session, 1..TURNS
  resume_via_v1.py TRAJECTORY.json    [TURNS]    # resume/extend a saved run

Gotcha (why the reply is fed back RAW below): the server caches each reply
verbatim, think tokens included. Feeding back a stripped copy diverges from
the cached prefix and breaks the KV cache (M-RoPE position assertion, HTTP
500). Strip only for language classification.

Writes the trajectory after every turn (crash-safe) and reports drift.
"""

import json
import os
import re
import sys
from pathlib import Path

import requests
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("SIDEVEC_PROJECT", SCRIPT_DIR.parent))


def lang_verdict(text):
    """Sentence-level shares; en/fi at >=0.8, or >=0.6 with the other <=0.1."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    counts, n = {}, 0
    for line in text.splitlines():
        s = re.sub(r"^([#>*+|-]|\d+\.)\s*", "", line.strip())
        for sent in re.split(r"(?<=[.!?:])\s+", s):
            if len(re.sub(r"[^A-Za-zÀ-ÿÄäÖö]", "", sent)) < 20:
                continue
            try:
                counts[detect(sent)] = counts.get(detect(sent), 0) + 1
                n += 1
            except LangDetectException:
                pass
    if n == 0:
        return "??", 0.0, 0.0
    en, fi = counts.get("en", 0) / n, counts.get("fi", 0) / n
    if en >= 0.8 or (en >= 0.6 and fi <= 0.1):
        return "en", en, fi
    if fi >= 0.8 or (fi >= 0.6 and en <= 0.1):
        return "fi", en, fi
    return "mixed", en, fi


RULE_DIR = Path(os.environ.get("SIDEVEC_RULE", PROJECT / "rules" / "reply-in-english"))
URL = os.environ.get("JLENS_URL", "http://127.0.0.1:8091")
OUT = Path(os.environ.get("OUT", PROJECT / "results" / "raw" / "session-v1"))

# args: [TRAJECTORY.json] [TURNS]. A first arg that is an existing file is a
# trajectory to resume; otherwise it (or the next arg) is the turn count.
RESUME, TURNS = None, 120
_args = sys.argv[1:]
if _args and Path(_args[0]).is_file():
    RESUME = _args.pop(0)
if _args:
    TURNS = int(_args[0])


def strip_think(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    return re.sub(r"<think>.*$", "", t, flags=re.DOTALL).strip()


def questions():
    qs = []
    for f in ("session-script.txt", "filler-questions.txt"):
        qs += [l.strip() for l in (RULE_DIR / f).read_text().splitlines() if l.strip()]
    return qs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = (RULE_DIR / "base-system.md").read_text()
    traj = json.loads(Path(RESUME).read_text()) if RESUME else []
    msgs = [{"role": "system", "content": base}]
    for t in traj:
        msgs.append({"role": "user", "content": t["q"]})
        msgs.append({"role": "assistant", "content": t["reply"]})
    qs = questions()
    start = "resuming from" if traj else "starting at"
    print(f"{start} turn {len(traj)}, target {TURNS}, vector server-side")

    for turn in range(len(traj) + 1, TURNS + 1):
        q = qs[turn - 1] + " /no_think"
        msgs.append({"role": "user", "content": q})
        r = requests.post(f"{URL}/v1/chat/completions", timeout=600, json={
            "messages": msgs, "max_tokens": 500, "temperature": 0.3,
            "top_p": 0.95, "top_k": 20, "seed": 42, "stream": False,
        })
        r.raise_for_status()
        data = r.json()
        # Feed the RAW reply back into history: the server cached it verbatim
        # (think tokens included). Feeding a stripped copy diverges from the
        # cached prefix and breaks the /v1 KV cache (M-RoPE position assert).
        raw = data["choices"][0]["message"]["content"]
        verdict, en, fi = lang_verdict(strip_think(raw))
        traj.append({"turn": turn, "verdict": verdict, "en": en, "fi": fi,
                     "q": qs[turn - 1], "reply": raw})
        msgs.append({"role": "assistant", "content": raw})
        out = OUT / f"session-{TURNS}t.json"
        out.write_text(json.dumps(traj, ensure_ascii=False, indent=1))  # crash-safe
        print(f"[t{turn:>3}] {verdict} (en={en:.2f} fi={fi:.2f})  {qs[turn-1][:38]}",
              flush=True)
    drift = [t["turn"] for t in traj if t["verdict"] != "en"]
    print(f"\nvector to turn {TURNS}: drift turns = {drift or 'NONE'}")


if __name__ == "__main__":
    main()
