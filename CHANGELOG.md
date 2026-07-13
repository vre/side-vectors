# Changelog

## 2026-07-13: initial public release

- Capture pipeline: correction-based activation deltas on Qwen 3.6 27B Q8
  via jlens-gguf, minimal-pair capture at the "about to answer" position.
- Single-layer steering (layer 48, alpha 1.0) makes the reply-in-English
  rule stick: 100/100 session turns, zero drift, content intact.
- Comparison data: prompt rule dies from one bad example in the history and
  never recovers; in-context corrections hold only while compliant history
  backs them. Full numbers in RESULTS.md.
- Live agent harness test: pi against the steered /v1 endpoint, 97 percent
  KV prefix reuse between turns. Needs the content-array patch in
  `patches/` (could be proposed upstream).
- Ships the captured vector (`rules/reply-in-english/vector.npz`, 2 MB, all
  64 layers) and DIY scripts for capturing vectors for your own rules.
- Tested only on a Mac (M5 Max, Metal).
