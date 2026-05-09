# MTPLX perf stack share - 2026-05-08

A curated branch off `daniel-farina/MTPLX` that bundles all the perf and cache-correctness work from the 2026-05-08 session into one place that a teammate can pull, run, and read.

The branch name is `share/perf-stack-2026-05-08`. It is based on `perf/memory-caps-and-adaptive-depth` (no rebase, no rewrite) so the actual commit history, including merges and bench-only commits, is preserved exactly as the work happened.

## What's on this branch

Ten code commits and five bench/doc commits. The code commits cluster into six fix areas:

### 1. Memory caps (upstream PR #32, OPEN)

Pin MLX Metal memory caps explicitly and lower the `clear_cache` threshold for long-context runs. Without this, sustained 32K-context loads on a 128GB box could drift into swap territory before `clear_cache` was tripped, causing latency cliffs that looked like cache misses but were actually allocator pressure. Commit `2debb9a`.

### 2. Prefill chunk size (upstream PR #33, OPEN)

Bump the sustained-profile prefill chunk size from 2048 to 4096. The smaller chunk was leaving Metal kernel bandwidth on the table for prompts above 8K tokens. The A/B numbers (see `findings-chunk-4096-2026-05-08.md`) showed a clean win across the long-context regime with no regression at short context. Commit `383da4f`.

### 3. Postcommit-wait (no upstream PR yet)

Wait briefly for the prior postcommit to land before the next request's cache lookup. The race here was: request N+1 would arrive, do its lookup, and miss because request N's postcommit hadn't yet written the prefix back into the bank. A short bounded wait fixes this without serializing the request path. Commit `12d7ee8`.

### 4. Content-based cache matching (no upstream PR yet)

Match cache entries by content rather than by exact session_id, and lift the implicit "two agents max" assumption so the bank works for arbitrary numbers of agents sharing a prefix. Combined with the per-session divergence threading (`d1cf628`), this is what makes multi-agent loops actually share a KV prefix instead of each blowing their own cold cache. Commits `15ce31b` and `d1cf628`.

### 5. Preamble fix (upstream PR #35, OPEN)

Keep the tool-call preamble text in the stored `assistant_content`. Previously the preamble was being stripped before storage, so the next turn's lookup would hash a slightly different content prefix and miss. Commit `19b2849`.

### 6. Divergence diagnostic (no upstream PR; tooling only)

Surface `cache_miss_reason` on postcommit, distinguish `new_session` from `prefix_divergence`, and have the divergence dump include cross-session overlapping entries so the operator can actually see *where* two flows forked. Commits `43bd51f`, `37c8cd7`, `179bc8c`, `e37181e`. Set `MTPLX_DUMP_DIVERGENCE=1` in the environment to capture per-session divergence diagnostics on every miss.

## Findings docs (already in `tools/bench/`)

- `findings-40pass-2026-05-08.md` - 40-pass game benchmark, raw TSV in `results-40pass-2026-05-08.tsv`
- `findings-adaptive-vs-baseline-2026-05-08.md` - adaptive depth A/B and sustained-profile knob triage
- `findings-chunk-4096-2026-05-08.md` - prefill chunk 2048 vs 4096
- `baseline-context-ladder-2026-05-08.txt` and `with-caps-context-ladder-2026-05-08.txt` - context-ladder traces before and after the memory caps
- `repro_tool_call_divergence.py` - standalone reproducer for the tool_call divergence bug
- `upstream-pr-plan-2026-05-08.md` and `upstream-pr-execution-2026-05-08.md` - the plan and the execution log
- `research-roadmap-2026-05-08.md` - what's still open

## Upstream PR status

| PR  | Topic                                    | Status                                             |
|-----|------------------------------------------|----------------------------------------------------|
| #32 | MLX Metal memory caps + clear_cache      | OPEN                                               |
| #33 | Prefill chunk 2048 -> 4096               | OPEN                                               |
| #34 | (earlier attempt)                        | CLOSED with feedback - needs to be rebuilt         |
| #35 | Preamble in stored assistant_content     | OPEN                                               |

Issue #36 has the executive summary that ties all four PRs together and gives upstream maintainers the context for the bench numbers.

The PR #34 rebuild is the main piece of in-flight work. The upstream feedback has been read and a fresh patch is on the to-do list; it is *not* on this branch yet.

## How to run

The launcher is local to Daniel's machine:

```
bash /Users/dan/code-2/mpt/run-mtplx-fork.sh
```

If you are running this on a different machine, adapt the launcher path. It expects the repo at `/Users/dan/code-2/forks/MTPLX` and the dashboard at port 9099; either match those or edit the launcher.

To capture per-session divergence diagnostics on cache miss:

```
MTPLX_DUMP_DIVERGENCE=1 bash /path/to/your/run-mtplx-fork.sh
```

Diagnostics land in the dashboard postcommit log with `cache_miss_reason` populated and, on `prefix_divergence`, a per-session dump of the overlapping prefix.

## Quick acceptance test

Five-pass cache test. Hit the same conversation prompt five times back-to-back through the dashboard or a scripted client. Expected behavior:

- Turn 1: `cache_hit: false`, `cache_miss_reason: new_session`
- Turns 2-5: `cache_hit: true`, no miss reason

If turn 2 still shows `cache_hit: false` with `cache_miss_reason: prefix_divergence`, set `MTPLX_DUMP_DIVERGENCE=1` and re-run; the per-session dump in the postcommit log will show which token boundary diverged. That used to be a real bug and is what the preamble fix (`19b2849`) and the content-match work (`15ce31b`) address.

## Clone

```
git clone https://github.com/daniel-farina/MTPLX.git
cd MTPLX
git checkout share/perf-stack-2026-05-08
```
