#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["langdetect"]
# ///
"""Classify the reply language sentence by sentence.

Prints: <verdict> <en_share> <fi_share> <n_sentences>
where verdict is "en" / "fi" / "mixed" / "??".

Verdict rules: a language wins at >=0.8 share, or at >=0.6 when the other
language is <=0.1 (proper nouns and list fragments classify as random
languages and drag a clean reply below 0.8). Otherwise "mixed".
Thinking blocks and code fences are stripped. Markdown structure lines
(lists, tables, headers) are classified too but proper-noun-only fragments
are skipped via the min-letters threshold per sentence.
"""

import re
import sys

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # deterministic
MIN_LETTERS = 20
THRESHOLD = 0.8


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    return text.strip()


def sentences(text: str) -> list[str]:
    parts = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^([#>*+-]|\d+\.)\s*", "", s)  # strip markdown markers
        parts.extend(p.strip() for p in re.split(r"(?<=[.!?:])\s+", s) if p.strip())
    return parts


def letters(s: str) -> int:
    return len(re.sub(r"[^A-Za-zÀ-ÿÄäÖö]", "", s))


def main() -> int:
    if len(sys.argv) > 1:
        text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    else:
        text = sys.stdin.read()
    counts: dict[str, int] = {}
    n = 0
    for sent in sentences(clean(text)):
        if letters(sent) < MIN_LETTERS:
            continue
        try:
            lang = detect(sent)
        except LangDetectException:
            continue
        counts[lang] = counts.get(lang, 0) + 1
        n += 1
    if n == 0:
        print("?? 0.00 0.00 0")
        return 1
    en = counts.get("en", 0) / n
    fi = counts.get("fi", 0) / n
    # Majority language + near-zero opposition counts as clean: proper nouns
    # and list fragments classify as random languages and drag the majority
    # share below the threshold even when the other language is absent.
    if en >= THRESHOLD or (en >= 0.6 and fi <= 0.1):
        verdict = "en"
    elif fi >= THRESHOLD or (fi >= 0.6 and en <= 0.1):
        verdict = "fi"
    else:
        verdict = "mixed"
    print(f"{verdict} {en:.2f} {fi:.2f} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
