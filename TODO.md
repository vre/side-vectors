# TODO

Done so far: capture pipeline, single-layer steering, held-out transfer,
death-point and drift measurements, interactive session sims (rule dies at
turn 63 clean / turn 9 contaminated / turn 1 death-point; vector 130/130 +
40/40 zero drift), live agent harness test via /v1, perf numbers (~4%). See
RESULTS.md. Published: github.com/vre/side-vectors. Patch PR: jlens-gguf#1.

## NEXT: Phase 2: selection-vs-learning boundary test

The sharpest open question, and the seed for a follow-up post. Prediction:
both capture a delta, only the known-content rule steers; the absent-knowledge
rule cannot steer because there is no existing program to select.

Steps (server + install_vector as usual, one rule dir per case):

1. `rules/json-preference/`: a same-shaped rule over KNOWN content, e.g.
   "prefer JSON over YAML in replies". Should steer (behaviour the model has).
2. `rules/frobnicate/`: a rule needing ABSENT knowledge, e.g. "always use
   the `frobnicate` command to X" (invented tool). Should capture a delta but
   NOT steer coherently.
3. For each: `capture_deltas.py --out results/raw/<rule> capture`, then
   `steer --band 48-48 --alpha 1.0`, plus held-out. Read the outputs (metric
   alone lies).
4. Compare: does the frobnicate delta fail to steer / produce garbage while
   json-preference steers cleanly? That locates the boundary.
5. Write up in RESULTS.md + wiki brief; this is post-2 material.

## Other open experiments

- [ ] Second rule family (generality): a mechanical tool rule like "always
      quote file paths": beyond language, the easiest target. Same pipeline.
- [ ] Dose response: alpha sweep at layer 48; layer sweep at fixed alpha.
      (Server up + existing vector; cheapest, makes "small dose" rigorous.)
- [ ] Compliance-confound control: capture a delta from a dummy instruction
      ("end responses with a period"), compare its direction (cosine) with
      the rule delta. Answers "is d_BC just generic obedience?".
- [ ] Other models and quants. Other platforms (CUDA, ROCm). PRs welcome.

## Publishing (parallel, gate-free first)

- [ ] Hugging Face: publish `vre/side-vectors-reply-in-english` (vector.npz +
      card, tags steering-vector/jacobian-lens/qwen/gguf). Gate-free, lands in
      the same discovery surface as jwest33 / Extraltodeus. PREPARED plan; HC
      to approve final push.
- [ ] r/LocalLLM (not -LLaMA): smaller sub, may lack the karma gate. Parallel
      Reddit shot, distinct community (not cross-post spam).
- [~] Reddit r/LocalLLaMA: post removed by sub-karma automod; modmail sent, no
      response; comment-karma not accruing. STUCK: do not depend on it.
- [~] HN Show / X / LessWrong: X declined; HN new-account risk; LessWrong
      unfamiliar. Deprioritized.

## Done

- [x] Fast /v1 session path (`scripts/resume_via_v1.py`, KV prefix reuse).
- [x] Content-array /v1 patch PR upstream (jlens-gguf#1).
