# side-vectors

*A model is a pile of vectors. These live beside it.*

Make a system-prompt rule stick by injecting it as a steering vector instead of writing it in the prompt. The rule then lives neither in the prompt (decays) nor in the weights (expensive to change). It lives *beside* the model, a side-vector: a 20 KB file (a fixed direction added to the model's internal activations), applied at one layer on every forward pass, removable at any time. No retraining, no LoRA, no B200 clusters.

The example used throughout this repo is one such rule: *"the user writes Finnish, always reply in English"* (token economy, and some models are weak at Finnish). Every model breaks it in normal use. It turns out one turn pair of Finnish conversation in the history is enough to kill it. The vector does not care.

**Three ways to use this repo:**

- **Just read the numbers:** [Results](#results-qwen-36-27b-q8-llamacpp--jlens-gguf) below, full detail in [RESULTS.md](RESULTS.md).
- **Try it on your machine:** [SETUP.md](SETUP.md) then `./demo.sh`, one Finnish question steered and unsteered side by side.
- **Make a vector for your own rule:** [Isolate a vector for your own rule](#isolate-a-vector-for-your-own-rule).

## Results (Qwen 3.6 27B Q8, llama.cpp / jlens-gguf)

Each cell counts replies that stayed in English out of the test prompts for that row (a Finnish question should get an English answer). "History" is the prior conversation already in context. "Held out" means prompts that were not used to build the vector. A "turn pair" is one user question plus the model's answer.

| | rule in system prompt | steering vector, no rule | nothing |
|---|---|---|---|
| no history | 8/8 English | 4/4 English (held out) | 0/8 |
| 1 turn pair of Finnish history | **0/4** | 4/4 | 0/4 |
| ~600 to ~12K tokens of history | 0/4 | 4/4 | 0/4 |
| clean session, same script both arms | rule slips at turn 63, gone by 71 | **vector: 130/130 turns (~45K ctx), zero drift** | n/a |
| contaminated start (2 Finnish pairs) | rule dead from turn 9 | **vector: 40/40, zero drift** | n/a |

In-context corrections hold only while the model's own compliant history backs them. The vector cannot relapse. It never lives in the token stream.

It also works live under a real coding agent: [pi](https://github.com/earendil-works/pi) pointed at the steered server, Finnish in, English out, with no rule in its prompt at all. The KV cache still reused 97 percent of each turn, so the session ran at normal speed. Speed cost of the whole setup: about 4 percent of decode speed.

Full data, method and caveats: [RESULTS.md](RESULTS.md).

## Try it

The captured vector ships in `rules/reply-in-english/vector.npz` (all 64 layers, 2 MB; steering uses one layer, 20 KB).

Setup is a handful of documented commands, no blind scripts: [SETUP.md](SETUP.md). If you already have a Qwen3.6-27B GGUF on disk, you symlink it and skip the 30 GB download.

Then one key turn:

```bash
./demo.sh
```

`demo.sh` (short, read it first) starts the server, asks one Finnish question twice (with and without the vector) and prints both answers side by side. Then you have seen the whole thing.

The server is jlens-gguf's `jlens-server`: llama.cpp under the hood (same GGUF, same backends), plus the intervention API. A stock llama-server cannot apply the vector; the measured overhead of the introspection server is about 1 percent of decode speed.

No lens file is needed for steering or for capturing your own vectors. The 3.3 GB pre-fitted lens (`neuronpedia/jacobian-lens` on HF) is only for inspecting what the model is doing.

## How it works

1. Trigger the failure once (model answers Finnish despite the rule).
2. Correct it in the conversation, with a short acknowledgment turn.
3. Ask the same question again. Capture residual activations at the "about to answer" position.
4. The steering vector is the difference between this state and the uncorrected state, averaged over 8 prompts.
5. Inject it additively at one layer (48 of 64) on every forward pass.

## Steer a live server

Install the vector once; every /v1 completion is steered and the KV prefix cache keeps interactive sessions at normal speed:

```bash
scripts/install_vector.py            # install at layer 48, alpha 1.0
scripts/install_vector.py --clear    # back to stock
```

Point any OpenAI-compatible client (including agent CLIs like pi or opencode) at the server. Agent clients send OpenAI content arrays, which the /v1 endpoint only accepts with the small patch from [SETUP.md](SETUP.md) (in `patches/`). The `demo.sh` and capture paths do not need it.

## Run a long steered session (fast)

To see how a vector holds over many turns, drive a session through the /v1 endpoint. It keeps a KV prefix cache, so each turn only prefills the new tokens:

```bash
scripts/install_vector.py            # install the vector server-side first
scripts/resume_via_v1.py 130         # fresh session, 130 turns
scripts/resume_via_v1.py session.json 200   # or resume/extend a saved run
```

Do NOT use `capture_deltas.py interactive-sim` for long runs: it goes through `/jlens/forward`, which re-prefills the whole history every turn (fine for short capture and steer checks, painfully slow past a few dozen turns). The `interactive-sim` path exists for the rule-vs-vector drift comparison; `resume_via_v1.py` is the one to run a real long session on.

## Isolate a vector for your own rule

```bash
# start from a copy of the example rule
cp -r rules/reply-in-english rules/my-rule

# drop the example's vector, you will capture your own
rm rules/my-rule/vector.npz

# now edit the text files in rules/my-rule/ to describe YOUR rule:
#   base-system.md    the system prompt WITHOUT the rule
#   rule.md           the one rule line to make stick
#   correction.txt    what you would say to correct a violation
#   capture-tasks.txt tasks to capture the delta over
#   holdout.txt       different tasks, to verify the vector transfers

# point the scripts at your rule
export SIDEVEC_RULE=$PWD/rules/my-rule
OUT=results/raw/my-rule

# capture: run each task with and without the rule + correction,
# save the activation deltas
scripts/capture_deltas.py --out $OUT capture

# steer: replay the held-out tasks with the delta injected, check it holds
scripts/capture_deltas.py --out $OUT steer --band 48-48 --alpha 1.0

# simulate a session to see rule-decay vs vector-stability over many turns
scripts/capture_deltas.py --out $OUT interactive-sim --arms rule vector

# ship the captured vector into the rule directory
cp $OUT/deltas.npz rules/my-rule/vector.npz

# install it on a live server so every completion is steered
scripts/install_vector.py
```

The capture position decides what you isolate. Useful rules:

- **Capture the behavior, not the meta-state.** Capturing right after the correction captures the model reacting to being corrected, not doing the corrected behavior. Steering with that reproduces the reaction. Re-ask the task after the correction and capture there, so both the corrected and uncorrected states sit at the same "about to act" position and the only difference between them is the correction.
- **Verify on held-out inputs.** The vector must encode the rule, not the inputs it was captured from. Test it on cases it never saw.
- **Keep the dose small.** A single layer is often enough. The same delta applied across many layers compounds past the natural activation scale and degrades output into garbage.
- **Judge the output, not a proxy.** An oversteered reply can still pass a shallow automated check. Read the actual output.

## Repo layout

- `rules/<name>/`: one directory per rule (system prompts, correction, task
  banks, captured vector)
- `scripts/`: capture, steer, sim, live install, language classification,
  harness runs
- `patches/`: required jlens-gguf patches (see SETUP.md)
- `RESULTS.md`: full numbers

## Requirements

Developed and tested ONLY on a MacBook Pro M5 Max 128 GB (Metal). That is the target environment. jlens-gguf builds on llama.cpp, so CUDA and ROCm should work in principle, but nobody has tested this there. Expect to modify build flags and `-ngl` handling. The 27B Q8 model wants about 30 GB of memory plus context. Python via `uv`. The scripts assume a Qwen-style chat template (an empty think block is prefilled so replies start at the visible answer); other model families need that constant adjusted. PRs for other platforms and models welcome.

## Credits

- Anthropic: the [Jacobian Lens research](https://www.anthropic.com/research/global-workspace) and [reference code](https://github.com/anthropics/jacobian-lens), and the [pre-fitted lens weights](https://huggingface.co/neuronpedia/jacobian-lens) (via Neuronpedia)
- [igorbarshteyn/jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf): the GGUF/llama.cpp port with the intervention API
- [llama.cpp](https://github.com/ggml-org/llama.cpp) under it all
- Model: [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)

MIT license. See [LICENSE](LICENSE).
