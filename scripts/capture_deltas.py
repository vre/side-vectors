#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "requests", "langdetect"]
# ///
"""Phase 1a delta capture + steering test against a running jlens-server.

States per Finnish prompt (text tasks only: no tools in this sandbox):
  B: base system prompt, no language rule        (expected reply: Finnish)
  A: system prompt WITH the reply-in-English rule (expected: English)
  C: B + model's own Finnish reply + correction message (teaching moment)

Capture: residual activations at the last prompt position (pre-generation)
for every layer. Deltas averaged over prompts:
  d_BC = mean(C) - mean(B)   (correction delta)
  d_BA = mean(A) - mean(B)   (prompt-rule delta)

Steering test: replay state-B prompts with `add` interventions of the delta
over a layer band; classify the generated language.

Subcommands:
  capture : run A/B/C forwards, save activations + replies to OUT/
  steer   : load saved deltas, run the steering sweep

Environment:
  JLENS_URL      jlens-server URL   (default http://127.0.0.1:8091)
  JLENS_GGUF     jlens-gguf checkout (default: <project>/jlens-gguf)
  JLENS_PROJECT  project root        (default: parent of this script's dir)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("JLENS_PROJECT", SCRIPT_DIR.parent))
JLENS_GGUF = Path(os.environ.get("JLENS_GGUF", PROJECT / "jlens-gguf"))
sys.path.insert(0, str(JLENS_GGUF))

from jlens_gguf.client import NativeClient  # noqa: E402

from langdetect import DetectorFactory, detect  # noqa: E402
from langdetect.lang_detect_exception import LangDetectException  # noqa: E402

DetectorFactory.seed = 0

# One directory per rule: base-system.md, rule.md, correction.txt,
# capture-tasks.txt, holdout.txt, session-script.txt, filler-questions.txt,
# vector.npz (once captured). Copy rules/reply-in-english to start your own.
RULE_DIR = Path(os.environ.get("SIDEVEC_RULE",
                               PROJECT / "rules" / "reply-in-english"))
OUT_DEFAULT = PROJECT / "results" / "raw" / "phase1a-deltas"
SAMPLING = {"greedy": False, "temp": 0.3, "top_p": 0.95, "top_k": 20, "seed": 42}
GEN_TOKENS = 350
# Prefill an empty think block so generation starts at the visible answer:
# replies stay complete within GEN_TOKENS, C states never contain reasoning,
# and the capture position ("about to answer") is consistent across states.
NOTHINK = "<think>\n\n</think>\n\n"


def strip_think(text: str) -> str:
    """Remove closed AND unclosed <think> blocks (a reply cut mid-thinking
    would otherwise be classified on its English reasoning text)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def lang_verdict(text: str) -> tuple[str, float, float]:
    """Sentence-level language shares; verdict en/fi/mixed at 0.8 threshold,
    "inc" when nothing visible remains (generation died inside thinking)."""
    text = strip_think(text)
    if not text:
        return "inc", 0.0, 0.0
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    counts: dict[str, int] = {}
    n = 0
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
    # Second clause: proper nouns and list fragments classify as random
    # languages and drag the majority share below 0.8 even when the other
    # language is entirely absent (observed: en=0.78, fi=0.00 on a fully
    # English reply). Majority language + near-zero opposition is clean.
    if en >= 0.8 or (en >= 0.6 and fi <= 0.1):
        verdict = "en"
    elif fi >= 0.8 or (fi >= 0.6 and en <= 0.1):
        verdict = "fi"
    else:
        verdict = "mixed"
    return verdict, en, fi


def load_stimuli() -> tuple[str, str, str, list[str]]:
    """Condition A system prompt = base + rule (single source, no duplicate
    files to keep in sync). Condition B/C = base alone."""
    base = (RULE_DIR / "base-system.md").read_text()
    rule = (RULE_DIR / "rule.md").read_text()
    sys_base = base
    sys_rule = base.rstrip() + "\n" + rule.strip() + "\n"
    correction = (RULE_DIR / "correction.txt").read_text().strip()
    prompts = []
    for line in (RULE_DIR / "capture-tasks.txt").read_text().splitlines():
        line = re.sub(r"^\d+\.\s+", "", line.strip())
        if line and "tiedostoon" not in line:  # skip file-write (tool) tasks
            prompts.append(line)
    return sys_base, sys_rule, correction, prompts


def load_delta(out: Path, name: str = "d_bc") -> np.ndarray:
    """Deltas from a capture run if present, else the rule's shipped vector."""
    for cand in (out / "deltas.npz", RULE_DIR / "vector.npz"):
        if cand.exists():
            return np.load(cand)[name]
    raise FileNotFoundError(f"no deltas.npz in {out} and no vector.npz in {RULE_DIR}")


def fwd(client: NativeClient, messages: list[dict], *, capture: bool, n_predict: int):
    prompt = client.apply_template(messages, add_assistant=True) + NOTHINK
    tokens = client.tokenize(prompt, add_special=False, parse_special=True)
    return tokens, client.forward(
        tokens,
        capture=capture,
        capture_layers=None,  # all layers
        n_predict=n_predict,
        sampling=SAMPLING,
    )


def last_prompt_vec(res) -> np.ndarray:
    """[n_layers, d] activation at the last prompt position."""
    layers = sorted(res.activations)
    return np.stack([res.activations[l][res.n_prompt - 1] for l in layers])


def cmd_capture(args) -> None:
    client = NativeClient(args.url)
    sys_base, sys_rule, correction, prompts = load_stimuli()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    acts: dict[str, list[np.ndarray]] = {"A": [], "B": [], "C": []}
    log = []
    for i, prompt in enumerate(prompts, 1):
        msgs_b = [{"role": "system", "content": sys_base}, {"role": "user", "content": prompt}]
        _, res_b = fwd(client, msgs_b, capture=True, n_predict=GEN_TOKENS)
        reply_b = res_b.generated_text
        acts["B"].append(last_prompt_vec(res_b))

        msgs_a = [{"role": "system", "content": sys_rule}, {"role": "user", "content": prompt}]
        _, res_a = fwd(client, msgs_a, capture=True, n_predict=GEN_TOKENS)
        acts["A"].append(last_prompt_vec(res_a))

        # Minimal pair vs B: after correction + acknowledgment the SAME
        # question is asked again, so both B and C capture at "about to
        # answer this question": the only difference is the correction.
        # (Capturing right after the correction instead encodes an
        # "about to acknowledge" state: steering with it produces
        # acknowledgment spam, observed 2026-07-13.)
        # Visible text only: real harnesses strip reasoning from history.
        # The acknowledgment is a FIXED inserted string, not generated by
        # the model. What matters for the delta is the correction being in
        # history when the question is re-asked; the ack is just a natural
        # assistant turn between them. Documented in RESULTS.md.
        ack = "Understood. I will respond in English from now on."
        msgs_c = msgs_b + [
            {"role": "assistant", "content": strip_think(reply_b)},
            {"role": "user", "content": correction},
            {"role": "assistant", "content": ack},
            {"role": "user", "content": prompt},
        ]
        _, res_c = fwd(client, msgs_c, capture=True, n_predict=GEN_TOKENS)
        acts["C"].append(last_prompt_vec(res_c))

        entry = {
            "prompt": prompt,
            "B": {"lang": lang_verdict(reply_b), "reply": reply_b},
            "A": {"lang": lang_verdict(res_a.generated_text), "reply": res_a.generated_text},
            "C": {"lang": lang_verdict(res_c.generated_text), "reply": res_c.generated_text},
        }
        log.append(entry)
        print(f"[{i}/{len(prompts)}] B={entry['B']['lang'][0]} A={entry['A']['lang'][0]} "
              f"C={entry['C']['lang'][0]}  {prompt[:50]}")

    np.savez_compressed(
        out / "activations.npz",
        A=np.stack(acts["A"]), B=np.stack(acts["B"]), C=np.stack(acts["C"]),
    )
    (out / "replies.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))

    A, B, C = (np.stack(acts[k]).mean(axis=0) for k in "ABC")  # [n_layers, d]
    d_bc, d_ba = C - B, A - B
    cos = (d_bc * d_ba).sum(-1) / (
        np.linalg.norm(d_bc, axis=-1) * np.linalg.norm(d_ba, axis=-1) + 1e-8
    )
    np.savez_compressed(out / "deltas.npz", d_bc=d_bc, d_ba=d_ba)
    print("\nlayer  |d_BC|   |d_BA|   cos(BC,BA)")
    for l in range(0, d_bc.shape[0], 4):
        print(f"{l:5d}  {np.linalg.norm(d_bc[l]):7.1f}  "
              f"{np.linalg.norm(d_ba[l]):7.1f}  {cos[l]:8.3f}")


def cmd_steer(args) -> None:
    client = NativeClient(args.url)
    sys_base, _, _, prompts = load_stimuli()
    if args.prompts_file:
        prompts = [
            l.strip() for l in Path(args.prompts_file).read_text().splitlines() if l.strip()
        ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vec = load_delta(out, args.vector)  # [n_layers, d]

    lo, hi = (int(x) for x in args.band.split("-"))
    results = []
    for i, prompt in enumerate(prompts[: args.n_prompts], 1):
        msgs = [{"role": "system", "content": sys_base}, {"role": "user", "content": prompt}]
        templ = client.apply_template(msgs, add_assistant=True) + NOTHINK
        tokens = client.tokenize(templ, add_special=False, parse_special=True)
        pos_start = len(tokens) if args.gen_only else 0
        ivs = [
            {"layer": l, "pos_start": pos_start, "pos_end": -1, "mode": "add",
             "vector": args.alpha * vec[l]}
            for l in range(lo, hi + 1)
        ]
        res = client.forward(tokens, capture=False, interventions=ivs,
                             n_predict=GEN_TOKENS, sampling=SAMPLING)
        verdict, en, fi = lang_verdict(res.generated_text)
        results.append({"prompt": prompt, "verdict": verdict, "en": en, "fi": fi,
                        "reply": res.generated_text})
        print(f"[{i}] {verdict} (en={en:.2f} fi={fi:.2f})  {prompt[:50]}")

    tag = f"steer-{args.vector}-L{lo}-{hi}-a{args.alpha}" + ("-genonly" if args.gen_only else "")
    (out / f"{tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    n_en = sum(1 for r in results if r["verdict"] == "en")
    print(f"\n{tag}: {n_en}/{len(results)} English")


def cmd_fillergen(args) -> None:
    """Generate a pool of Finnish Q&A filler turns with the model itself."""
    client = NativeClient(args.url)
    sys_base, _, _, _ = load_stimuli()
    questions = [
        l.strip()
        for l in (RULE_DIR / "filler-questions.txt").read_text().splitlines()
        if l.strip()
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = []
    for i, q in enumerate(questions, 1):
        msgs = [{"role": "system", "content": sys_base}, {"role": "user", "content": q}]
        _, res = fwd(client, msgs, capture=False, n_predict=380)
        a = strip_think(res.generated_text).strip()
        pool.append({"q": q, "a": a})
        print(f"[{i}/{len(questions)}] {len(a)} chars  {q[:50]}")
    (out / "filler-pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1))


def cmd_sweep(args) -> None:
    """Context-length sweep: rule-in-prompt vs steering vector vs nothing.

    Conversation = [system] + Finnish filler turns (to target size) +
    held-out question. Filler is model-generated Finnish Q&A: the same
    mirroring pressure as real long sessions.
    """
    client = NativeClient(args.url)
    sys_base, sys_rule, _, _ = load_stimuli()
    out = Path(args.out)
    pool = json.loads((out / "filler-pool.json").read_text())
    vec = load_delta(out)
    layer = args.layer
    holdout = [
        l.strip() for l in Path(args.prompts_file).read_text().splitlines() if l.strip()
    ]

    def build(system: str, target_tokens: int, question: str) -> list[int]:
        msgs = [{"role": "system", "content": system}]
        reached = False
        for turn in pool:
            probe = msgs + [{"role": "user", "content": question}]
            if len(client.tokenize(client.apply_template(probe, add_assistant=True),
                                   add_special=False, parse_special=True)) >= target_tokens:
                reached = True
                break
            msgs += [{"role": "user", "content": turn["q"]},
                     {"role": "assistant", "content": turn["a"]}]
        msgs.append({"role": "user", "content": question})
        templ = client.apply_template(msgs, add_assistant=True) + NOTHINK
        tokens = client.tokenize(templ, add_special=False, parse_special=True)
        if not reached:
            print(f"  WARNING: filler pool exhausted at {len(tokens)} tokens: "
                  f"target {target_tokens} NOT reached", flush=True)
        return tokens

    summary = []
    for target in args.lengths:
        for arm in ("rule", "vector", "none"):
            system = sys_rule if arm == "rule" else sys_base
            n_en = 0
            rows = []
            for q in holdout[: args.n_prompts]:
                tokens = build(system, target, q)
                ivs = None
                if arm == "vector":
                    ivs = [{"layer": layer, "pos_start": 0, "pos_end": -1,
                            "mode": "add", "vector": args.alpha * vec[layer]}]
                res = client.forward(tokens, capture=False, interventions=ivs,
                                     n_predict=GEN_TOKENS, sampling=SAMPLING)
                verdict, en, fi = lang_verdict(res.generated_text)
                n_en += verdict == "en"
                rows.append({"q": q, "n_ctx": len(tokens), "verdict": verdict,
                             "en": en, "fi": fi, "reply": res.generated_text})
                print(f"  {target:>6} {arm:<6} ctx={len(tokens):>6} {verdict}  {q[:40]}")
            summary.append({"target": target, "arm": arm,
                            "en_rate": n_en / len(rows), "rows": rows})
            print(f"{target} {arm}: {n_en}/{len(rows)} English")
    name = "context-sweep-" + "-".join(str(t) for t in args.lengths) + ".json"
    (out / name).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\ntarget  rule  vector  none")
    for t in args.lengths:
        vals = {s["arm"]: s["en_rate"] for s in summary if s["target"] == t}
        print(f"{t:>6}  {vals['rule']:.2f}  {vals['vector']:.2f}    {vals['none']:.2f}")


def cmd_interactive_sim(args) -> None:
    """Simulate an interactive session: Finnish user, English-rule assistant.

    Unlike the sweep (pre-contaminated history), the model starts COMPLIANT:
    the rule (or vector) is active from turn 1 and every reply the model
    produces: compliant or drifted: is appended to its own history. This
    measures the natural drift curve: which turn does the first slip happen,
    and does one slip cascade via self-precedent.
    """
    client = NativeClient(args.url)
    sys_base, sys_rule, _, _ = load_stimuli()
    # session-script.txt: coherent topic arcs with follow-ups referencing the
    # previous answer: identical user input across arms, unlike trivia bank.
    # filler-questions.txt extends the bank for long runs past the script.
    questions = [
        l.strip()
        for l in (RULE_DIR / "session-script.txt").read_text().splitlines()
        if l.strip()
    ] + [
        l.strip()
        for l in (RULE_DIR / "filler-questions.txt").read_text().splitlines()
        if l.strip()
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resume_traj = []
    if args.resume_from:
        resume_traj = json.loads(Path(args.resume_from).read_text())

    _, _, correction, _ = load_stimuli()
    seed_turns = []
    if args.seed_pairs:
        # Pre-contaminate: N Finnish Q&A pairs in the history before turn 1.
        # Guarantees the rule is already dead (death-point: one pair kills
        # it), so the correction half-life becomes measurable.
        pool = json.loads((Path(args.out) / "filler-pool.json").read_text())
        for turn in pool[: args.seed_pairs]:
            seed_turns += [{"role": "user", "content": turn["q"]},
                           {"role": "assistant", "content": turn["a"]}]
    for arm in args.arms:
        system = sys_rule if arm in ("rule", "rule-correct") else sys_base
        msgs = [{"role": "system", "content": system}] + list(seed_turns)
        traj = []
        n_corrections = 0
        for prev in resume_traj:  # rebuild history from a prior run
            msgs.append({"role": "user", "content": prev["q"]})
            msgs.append({"role": "assistant", "content": prev["reply"]})
            traj.append(prev)
        start = len(resume_traj)
        for turn, q in enumerate(questions[start: args.turns], start + 1):
            msgs.append({"role": "user", "content": q})
            templ = client.apply_template(msgs, add_assistant=True) + NOTHINK
            tokens = client.tokenize(templ, add_special=False, parse_special=True)
            ivs = None
            if arm == "vector":
                ivs = [{"layer": args.layer, "pos_start": 0, "pos_end": -1,
                        "mode": "add", "vector": args.alpha * vec[args.layer]}]
            res = client.forward(tokens, capture=False, interventions=ivs,
                                 n_predict=GEN_TOKENS, sampling=SAMPLING)
            reply = strip_think(res.generated_text).strip()
            verdict, en, fi = lang_verdict(reply)
            corrected = False
            msgs.append({"role": "assistant", "content": reply})
            if arm == "rule-correct" and verdict != "en":
                # inject the correction after a drifted reply;
                # measures correction half-life: turns until relapse
                msgs.append({"role": "user", "content": correction})
                msgs.append({"role": "assistant",
                             "content": "Understood. I will respond in English from now on."})
                n_corrections += 1
                corrected = True
            traj.append({"turn": turn, "n_ctx": len(tokens), "verdict": verdict,
                         "en": en, "fi": fi, "corrected_after": corrected,
                         "q": q, "reply": reply})
            print(f"[{arm} t{turn:02d} ctx={len(tokens):>6}] {verdict}"
                  f"{' +corr' if corrected else ''}  {q[:40]}", flush=True)
            # Spontaneous recovery never happens (months of empirical use
            # with this rule): once absorbed, further turns are waste.
            if args.absorb_stop and len(traj) >= args.absorb_stop and all(
                t["verdict"] != "en" for t in traj[-args.absorb_stop:]
            ):
                print(f"{arm}: absorbed ({args.absorb_stop} consecutive "
                      f"non-en): stopping early at turn {turn}", flush=True)
                break
        seed_tag = f"-seed{args.seed_pairs}" if args.seed_pairs else ""
        (out / f"interactive-{arm}-{args.turns}t{seed_tag}.json").write_text(
            json.dumps(traj, ensure_ascii=False, indent=1))
        drifted = [t["turn"] for t in traj if t["verdict"] != "en"]
        first = drifted[0] if drifted else None
        extra = f", corrections={n_corrections}" if arm == "rule-correct" else ""
        print(f"{arm}: first drift turn={first}, "
              f"drift {len(drifted)}/{len(traj)} turns{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("JLENS_URL", "http://127.0.0.1:8091"))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture")
    sub.add_parser("fillergen")
    sw = sub.add_parser("sweep")
    sw.add_argument("--lengths", type=int, nargs="+", default=[4000, 15000, 30000])
    sw.add_argument("--layer", type=int, default=48)
    sw.add_argument("--alpha", type=float, default=1.0)
    sw.add_argument("--n-prompts", type=int, default=4)
    sw.add_argument("--prompts-file",
                    default=str(RULE_DIR / "holdout.txt"))
    si = sub.add_parser("interactive-sim")
    si.add_argument("--arms", nargs="+", default=["rule", "vector", "none"],
                    choices=["rule", "vector", "none", "rule-correct"])
    si.add_argument("--turns", type=int, default=60)
    si.add_argument("--layer", type=int, default=48)
    si.add_argument("--alpha", type=float, default=1.0)
    si.add_argument("--absorb-stop", type=int, default=0,
                    help="stop arm after N consecutive non-en turns (0=off)")
    si.add_argument("--seed-pairs", type=int, default=0,
                    help="pre-contaminate history with N Finnish Q&A pairs")
    si.add_argument("--resume-from", default=None,
                    help="prior trajectory JSON: rebuild history, continue "
                         "from the next turn (--turns is the new total)")
    st = sub.add_parser("steer")
    st.add_argument("--vector", choices=["d_bc", "d_ba"], default="d_bc")
    st.add_argument("--band", default="40-62", help="layer band lo-hi inclusive")
    st.add_argument("--alpha", type=float, default=1.0)
    st.add_argument("--n-prompts", type=int, default=4)
    st.add_argument("--gen-only", action="store_true",
                    help="apply interventions to generated positions only")
    st.add_argument("--prompts-file", default=None,
                    help="plain-text prompt list (one per line): for held-out tests")
    args = ap.parse_args()
    {"capture": cmd_capture, "steer": cmd_steer,
     "fillergen": cmd_fillergen, "sweep": cmd_sweep,
     "interactive-sim": cmd_interactive_sim}[args.cmd](args)


if __name__ == "__main__":
    main()
