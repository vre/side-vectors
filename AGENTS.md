# side-vectors: working notes for agents and contributors

A model is a pile of vectors. These live beside it.

Runtime behavior steering for frozen local models: capture a correction as an activation delta, inject it at one layer, and the rule sticks without retraining. See README.md for the what and RESULTS.md for the numbers.

## Layout

- `rules/<name>/`: one directory per rule experiment. Contains the base
  system prompt, the rule line, the correction message, task banks and the
  captured vector. Copy `rules/reply-in-english` to start your own.
- `scripts/`: capture_deltas.py (capture / steer / sweep / interactive-sim),
  install_vector.py (live /v1 interventions), detect_lang.py,
  run-condition.sh, verify-context-loading.sh, perf_test.py
- `patches/`: required patches to jlens-gguf (applied in SETUP.md)
- `jlens-gguf/`, `models/`, `lenses/`, `results/raw/`: local artifacts,
  not in git

## Rule files are test subjects

The files under `rules/<name>/` are experiment stimuli read by the model under test. Their exact content is the independent variable. Do not normalize, reformat or "improve" them. Byte changes alter the experiment.

## Experiment hygiene learned the hard way

- Capture the behavior, not the meta-state: capture at "about to answer the
  re-asked question", never right after the correction message.
- Verify on held-out prompts. The vector must encode the rule, not the
  capture prompts.
- One layer. Multi-layer injection of the same delta oversteers into
  repetition garbage.
- Never trust a shallow automated metric alone. Oversteered garbage can pass it. Read the actual output.
- Strip reasoning blocks from replies before building multi-turn states.
- pi reads stdin in `-p` mode: redirect `</dev/null` in loops.
- Long steered sessions: use scripts/resume_via_v1.py (/v1, KV prefix cache), not interactive-sim (/jlens/forward re-prefills every turn).
- Feeding a /v1 session: append the RAW model reply to history, think tokens included. A stripped copy diverges from the cached prefix and 500s the server (M-RoPE). Strip only to classify.

## Style

- Short sentences, plain language, no em dashes in any public text.
- Env-var parameterization in every script. No hardcoded personal paths.
- Git history is published too: fix mistakes with amend or squash before
  they land, not with a follow-up commit.
