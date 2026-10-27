<p align="center">
  <img src="assets/logo.svg" width="96" height="96" alt="beam logo">
</p>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

# beam

### Beam-search decoding experiments for llama.cpp

**Eight hand-written decoders that keep several continuations alive at once by branching the model's KV cache — and a record of the trade-offs between sequence-reuse strategies.**

**Status:** archived, not maintained · no standalone build · pinned to llama.cpp's internal example API from early 2025 · [AGPL-3.0-only](LICENSE)

---

## Why this exists

llama.cpp's bundled example decoders keep one token stream alive. Beam search
needs several hypotheses at the same time, and the expensive part is not the
sampling — it is the KV cache. Every live hypothesis needs its own sequence,
and branching one hypothesis into two means copying or remapping that sequence
without corrupting its siblings.

These files are the working record of that problem. The sampling calls are
ordinary; the interesting decisions are when to copy a KV sequence, when to
reuse it in place, and when sampler state has to be cloned along with it. The
code targets an internal llama.cpp API that has since moved, so treat this as
an annotated design archive rather than a current library.

## What it can do now

- Draw several candidate tokens per live beam, deduplicate them, and prune the
  global candidate pool by cumulative log-probability.
- Keep multiple hypotheses resident in one `llama_context`, one KV sequence per
  live beam.
- Branch a beam either by reusing the parent sequence in place or by allocating
  a fresh sequence and copying the parent into it.
- Share one sampler across beams, or clone sampler state per beam when the
  sampler is stateful.
- Emit a finished beam on end-of-generation while the remaining beams continue.
- Run more than one independent search inside a single batch (`beam3.cpp`), and
  report KV usage plus decode/planning time (`beam7.cpp`, `beam8.cpp`).

What is **not** here: a build system, a public API, tests, a benchmark, a CLI
layer beyond llama.cpp's common arguments, or compatibility with the current
llama.cpp tree. Some snapshots hard-code a Qwen chat prompt; later ones use
`-p`. That is the archive, not a promise.

## Quick start

There is nothing to compile or run in this repository on its own. The fastest
way in is to read the arc:

1. [`beam.cpp`](beam.cpp) — 92 lines, the whole idea in one `main`.
2. [`beam4.cpp`](beam4.cpp) — the first clean sequence-reuse design.
3. [`beam8.cpp`](beam8.cpp) — the final branch-allocation variant.

To try the technique today, port one file into a llama.cpp tree as an
`examples/` target and replace its `llama_kv_self_*` and `common_sampler_*`
calls with the current equivalents. The original local build lines survive as
trailing comments in `beam4.cpp` through `beam8.cpp`; they reference the
author's private llama.cpp and BLAS checkouts, so read them as provenance
rather than instructions.

## The eight snapshots

| File | What changed | What it teaches |
| --- | --- | --- |
| [`beam.cpp`](beam.cpp) | First cut. One sampler per beam, copy KV when branches collide, fixed Qwen chat prompt. | The core loop works, but per-beam samplers and ad-hoc pruning get messy quickly. |
| [`beam2.cpp`](beam2.cpp) | Wrapped in a `BeamSearch` object with token history and configurable branch factor and beam width. | Separating search state from `main` makes the variants comparable. |
| [`beam3.cpp`](beam3.cpp) | Candidate indices instead of beam objects; several independent searches share one batch and contiguous sequence slots. | Batching works, but index bookkeeping and slot ownership become the hard part. |
| [`beam4.cpp`](beam4.cpp) | One shared sampler, additive log-probabilities, in-place donor-sequence reuse, finished beams emitted on EOG. | The sequence-remap approach is the cleanest of the early designs. |
| [`beam5.cpp`](beam5.cpp) | `beam4` wired to `-p` and streaming the finished beams. | A small driver change turns the experiment into something runnable against that llama.cpp revision. |
| [`beam6.cpp`](beam6.cpp) | Per-beam sampler clones return, but KV is copied only when a branch actually forks. | Sampler state and KV state fork at different moments and must be tracked separately. |
| [`beam7.cpp`](beam7.cpp) | Compile-time switch for sampler cloning, cleaner branch remapping, KV and timing instrumentation. | One implementation can compare both sampler strategies without forking the search code. |
| [`beam8.cpp`](beam8.cpp) | Allocate a fresh sequence id for each fork instead of reusing the donor slot. | The simplest branching logic, at the cost of more KV copying. |

`beam.cpp` is iteration 1; there is no `beam1.cpp`. The numbering is the order
the ideas were tried, and the files stay untouched as history.

## How it works

The later files share one model:

```
prompt ── seq 0
            │
            ├── token A ── seq 0 ── ...
            ├── token B ── seq 1 ── ...   KV copied from seq 0
            └── token C ── seq 2 ── ...   KV copied from seq 0
```

- `BeamTok { sup, tok }` is an append-only token tree. New candidates point at
  their parent's index; walking `sup` backwards reconstructs a hypothesis.
- `Beam { seq, logit }` is the live tip: which KV sequence it owns and its
  cumulative score.
- `step()` samples candidates for every live beam, deduplicates them, sorts by
  score, removes finished beams, remaps or copies KV sequences for the
  survivors, and queues the next tokens in one batch.

`beam4.cpp`–`beam7.cpp` reuse the winning beam's sequence in place and copy only
the remaining forks. `beam8.cpp` copies every additional fork into a fresh
sequence, trading extra KV work for simpler ownership rules. Sampler state is a
separate axis: shared sampler, per-beam clone, or a compile-time choice in
`beam7.cpp`.

## How to develop

There is no repository-level build. Each `.cpp` file is a translation unit with
its own `main`, written to sit beside llama.cpp's `common` and `sampling`
helpers. The intended development loop is:

1. Pick a llama.cpp revision whose internal example API matches the file.
2. Add the file as an `examples/` target in that tree.
3. Iterate in a scratch copy; the numbered snapshots stay as they are.
4. If porting to today's API, change the KV and sampler calls, not the search
   shape — the sequence-ownership model is the part worth preserving.

No CMakeLists, package manifest or test harness is provided. That is a
deliberate boundary of the archive, not a missing release artifact.

## How to contribute

This is an archive, not a maintained project: a new design goes in a new
numbered file, and the historical snapshots stay as they are. The rules for
ports, corrections and review are in [CONTRIBUTING.md](CONTRIBUTING.md);
anything larger starts as an issue on
[GitHub](https://github.com/ZisIsNotZis/beam/issues).

## Roadmap

### Implemented

- [x] Beam search over live KV-cache sequences with global pruning by
      cumulative log-probability.
- [x] Two sequence-reuse strategies: in-place donor reuse and fresh-sequence
      allocation per fork.
- [x] Shared sampler versus per-beam sampler clones.
- [x] Multiple independent searches batched together (`beam3.cpp`).
- [x] Finished-beam emission and KV/timing instrumentation.
- [x] The experiment log, provenance notes and an honest archive status.

### Planned

- [ ] Port one `BeamSearch` implementation to the current llama.cpp KV and
      sampler API.
- [ ] Add a minimal `examples/` CMake target around that port.
- [ ] Add a small reproducible harness comparing greedy decoding and beam
      search on one GGUF model.
- [ ] Audit candidate deduplication and error paths; replace silent failures
      with assertions.
- [ ] Keep the old snapshots untouched as the design history.

## Future vision

If this archive is revived, the clean target is a small `BeamSearch` class on
current llama.cpp: one sequence per live beam, copies only at real forks,
sampler state cloned only when the sampler is stateful, and a harness that
reports quality per second against greedy decoding. The old files remain the
record of how that design was found.

## License

The code in this repository is licensed under
[AGPL-3.0-only](LICENSE). It depends on llama.cpp's example framework, which is
MIT-licensed and is not included here.
