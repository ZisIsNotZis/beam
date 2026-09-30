# beam — beam search for llama.cpp (experiment log)

**One-line what:** A hand-written beam-search decoder layered on llama.cpp's
`common`/`sampling` example infrastructure, exploring how to keep N candidate
continuations alive in the KV cache via sequence copy/clone.

**Why it exists:** llama.cpp's bundled examples do greedy/sampling decode, not
general beam search. This was a scratch investigation into implementing beam
search on top of `llama_kv_self_seq_cp` / `llama_kv_self_seq_rm` and
`common_sampler_clone`, iterating to find a correct-and-fast sequence-reuse
strategy.

**How to run:** There is no standalone build. Each file is meant to be dropped
into a llama.cpp source tree (as an `examples/`-style target) and compiled
against that tree's `common`, `llama` and `ggml` libraries. The author's own
recorded build line is preserved as a trailing comment in `beam7.cpp`,
`beam8.cpp` and `beamUnknown.cpp`; it references a local llama.cpp checkout and
local BLAS build directories that are not part of this directory. No
self-contained repro exists.

**Status:** **deprecated (2025 era) — kept as an archive.**
The files date from April 2025 (`beam.cpp` … `beam7.cpp`) with one later
revision in November 2025 (`beamUnknown.cpp`). They are pinned to a specific
snapshot of llama.cpp's *internal* example API (`common_sampler`,
`llama_kv_self_*`, `common_batch_add`), which is explicitly unstable and has
changed since. To become current again the code needs porting to the present
KV-cache/sampler API; the beam-search technique itself remains valid.

**License note:** No LICENSE file is present and no upstream project is cited,
so this is the owner's own code and the owner may license it as they choose.
However the surrounding scaffolding is llama.cpp's example framework
(MIT-licensed) — publishing this as part of or alongside llama.cpp is the
natural and license-compatible route; a standalone repo would need to state its
dependence on llama.cpp's MIT license.
