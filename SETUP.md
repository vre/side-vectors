# Setup

Every command is listed here so you can read before you run. Nothing needs root. Everything lands inside this directory (plus the model download if you take that path).

Steps 1 to 3 are the same whether you want to **try** the shipped vector or **capture** your own. They give you the server and the model. After that:

- **Just try it:** start the server (step 4) and run the demo (step 5). Done.
- **Capture your own rule:** same server, then follow "Isolate a vector for your own rule" in [README.md](README.md). No extra setup.

The lens (step 6) is optional and needed for neither. It is only for inspecting what the model is doing.

Requirements: git, cmake, a C++ toolchain, [uv](https://docs.astral.sh/uv/). Optional: the [pi](https://github.com/earendil-works/pi) coding agent CLI for the live harness scripts (run-condition.sh, verify-context-loading.sh). Tested on macOS (M5 Max, Metal). CUDA and ROCm are untested and may need build flag changes.

## 1. Get and build jlens-gguf

The introspection server. Pinned to the commit this repo was tested with.

```bash
git clone https://github.com/igorbarshteyn/jlens-gguf
cd jlens-gguf
git checkout 16e5526ced3b332467210eb4b10018c480fd6360
git submodule update --init     # brings llama.cpp
git apply ../patches/0001-v1-openai-content-array.patch
native/build.sh                 # builds llama.cpp + the jlens-server binary
```

The patch (35 lines, read it) lets the /v1 endpoint accept OpenAI content arrays, which agent clients like pi send. Without it only plain-string content works. Could be proposed upstream.

## 2. Python environment

```bash
uv venv
uv pip install -e .
cd ..
```

## 3. The model

Already have a Qwen3.6-27B GGUF on disk? Point your model here:

```bash
mkdir -p models
ln -s /path/to/your/Qwen3.6-27B-whatever.gguf models/
```

No model yet? This downloads about 30 GB:

```bash
mkdir -p models
uvx --from huggingface_hub hf download unsloth/Qwen3.6-27B-GGUF \
    Qwen3.6-27B-UD-Q8_K_XL.gguf --local-dir models
```

## 4. Start the server

```bash
jlens-gguf/native/jlens-server -m models/<your-model>.gguf \
    -ngl 99 -fa on -c 8192 --port 8091
```

`-c 8192` is enough for the demo and single steer calls. Long sessions (`resume_via_v1.py`) grow the context every turn, so use `-c 32768` or more for those.

The server binds to 127.0.0.1 by default. Keep it that way: the API has no authentication, and anyone who can reach the port can install interventions that silently change the model's behavior.

## 5. Try it

`demo.sh` (short, read it first) asks one Finnish question twice, with and without the steering vector, and prints both answers side by side. It starts and stops its own server, so stop the one from step 4 first, or just run the demo on its own.

```bash
./demo.sh
```

Or steer manually against the running server:

```bash
scripts/capture_deltas.py steer --vector d_bc --band 48-48 --alpha 1.0
```

The shipped vector is found automatically (`rules/reply-in-english/vector.npz`).

To capture a vector for your OWN rule instead, keep the server running and follow "Isolate a vector for your own rule" in [README.md](README.md).

## 6. Optional: the lens (inspection only)

Neither trying nor capturing needs this file. Capture reads activations straight from the running model server. The lens is only for the readout/inspection views (projecting activations to tokens). About 3.3 GB.

```bash
mkdir -p lenses
uvx --from huggingface_hub hf download neuronpedia/jacobian-lens \
    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt" \
    --local-dir lenses
jlens-gguf/.venv/bin/python -m jlens_gguf convert-pt \
    "lenses/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt" \
    lenses/qwen3.6-27b-lens.gguf
```
