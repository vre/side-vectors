# Patches to jlens-gguf

Applied on top of the pinned commit in SETUP.md.

- `0001-v1-openai-content-array.patch`: the /v1 endpoint only accepted
  plain-string message content. Agent clients (pi and others) send OpenAI
  content-part arrays. The patch flattens text parts. Proposed upstream:
  https://github.com/igorbarshteyn/jlens-gguf/pull/1

The patch modifies jlens-gguf source, which is Apache-2.0 licensed by its author. The patch content follows that license; the rest of this repository is MIT.
