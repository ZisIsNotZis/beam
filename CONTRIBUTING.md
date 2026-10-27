# Contributing

`beam` is a frozen design archive: eight numbered snapshots of the same
beam-search decoder, kept as they were written. It is not a maintained library
and has no repository-level build. Contributions are welcome in two shapes
only — **new snapshots** and **corrections** — and both are judged by whether a
reader can reproduce the result.

## Scope

- A new idea goes in a new file: `beam9.cpp`, `beam10.cpp`, … Each file is a
  standalone translation unit with its own `main`, written to sit inside a
  llama.cpp tree beside `common` and `sampling`.
- `beam.cpp` through `beam8.cpp` are history. Do not reformat, rename or
  "clean up" them; a diff against those files is only accepted for a factual
  error in a comment.
- No build system, package manifest, dependency change or public API is in
  scope. The absence of those is the point of the archive.

## How to check your change

There is nothing in this repository to run. The check is external:

1. Pick a llama.cpp revision whose internal example API matches your file, and
   record the commit hash in a comment at the top of the file.
2. Build it inside that tree as an `examples/` target, or with the `g++` line
   the earlier snapshots keep as a trailing comment.
3. Run it once with a real model and paste the command and its output in the
   pull request.

A pull request that cannot name its llama.cpp commit and show one reproducible
generation will be asked for them before review.

## Pull requests

- One idea per file, one file per pull request.
- Say what changed in the search strategy and what it cost: KV copies, memory,
  or decode time. `beam7.cpp` and `beam8.cpp` print KV and timing numbers —
  reuse that instrumentation rather than adding a new one.
- Keep the historical files untouched, so `git log` stays the record of how
  the design evolved.
- Conventional Commits for the subject line (`docs:`, `chore:`, `feat:`), and
  no mixed refactor-plus-feature changes.

## Issues

Open an issue before any change larger than a new snapshot — a port to the
current llama.cpp API, a harness, or a rewrite of the sequence-ownership model.
Say which llama.cpp revision you target and what you observed; "it does not
compile" without a revision cannot be acted on. Corrections to a claim in the
README are always welcome and need no prior issue.
