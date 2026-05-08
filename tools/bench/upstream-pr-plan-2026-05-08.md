# Upstream PR plan for `perf/memory-caps-and-adaptive-depth`

Date: 2026-05-08
Local repo: `/Users/dan/code-2/forks/MTPLX`
Fork branch: `daniel-farina:perf/memory-caps-and-adaptive-depth`
Upstream: `youssofal/MTPLX` (default branch `main`)

## Upstream state

- Upstream description: "Native MTP Speculative Decoding On Apple Silicon | 2x - 2.5x decode TPS increase at temp 0.6 | MLX-native, OpenAI API/Anthropic-compatible serving, no external drafter."
- Default branch: `main`.
- PR template (`.github/pull_request_template.md`) has three required sections:
  - `## Summary`
  - `## Verification`
  - `## Benchmark Evidence` ("If this changes runtime performance, include hardware, model, quantization, sampler, token count, profile, fan mode, date, and commit.")
- `CONTRIBUTING.md` requires before opening a PR:
  - `python -m pip install -e ".[dev,server]"`
  - `python -m pytest tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py`
  - `python -m build`
  - `scripts/fresh_venv_smoke.sh`
  - "Benchmarks must include hardware, model, quantization, sampler, token count, profile, fan mode, date, and commit. Do not use fan-controlled runs for product headline claims."

### Recent activity tempo (default branch, last ~25 commits)

High velocity. Maintainer (`youssofal`) merges multi-times-per-day. Most recent default-branch tip `d4315b7 2026-05-08T16:44:02Z Fix SessionBank postcommit prefix reachability` (today). Recent merges include "Batch-ready SessionBank model scheduler" (PR #29), v0.2.1 hotfix, and an active stream of perf/correctness fixes - the maintainer is clearly receptive to small, evidence-backed PRs in this exact area.

### Open / recent PRs (sample)

| # | State | Author | Title |
|---|---|---|---|
| 31 | OPEN | daniel-farina | perf: SessionBank cache reuse for tool-using subagents (entry-cap env, 16K cliff, postcommit serialization, storage prefix-match) |
| 30 | MERGED | youssofal | Fix SessionBank postcommit prefix reachability |
| 29 | MERGED | youssofal | Batch-ready SessionBank model scheduler |
| 28 | OPEN | daniel-farina | Fix: TurboQuant must not crash when vllm-metal external ops are unavailable |
| 25 | CLOSED | AlexsJones | Fix tool-call streaming dying when model hallucinates |
| 24 | CLOSED | daniel-farina | Fix subagent SessionBank caching: idle postcommit + tools plumbing + cross-thread MLX |
| 21 | MERGED | daniel-farina | Handle mixed text + `<tool_call>` responses in streaming translator |
| 19 | MERGED | AlexsJones | Fix install_macos.sh crash on global launcher permission denied |
| 18 | MERGED | daniel-farina | Add `MTPLX_SESSION_BANK_*_BYTES` env-var overrides for bank caps |
| 17 | MERGED | daniel-farina | Fix: async SessionBank commit for tool-call responses |

## User's existing PRs upstream

| # | State | Branch | Status / disposition |
|---|---|---|---|
| 17 | MERGED | `fix/tool-call-postcommit-async` | landed |
| 18 | MERGED | `feat/configurable-bank-caps` | landed |
| 21 | MERGED | `fix/mixed-content-tool-call-streaming` | landed |
| 24 | CLOSED | `fix/idle-postcommit-foreground-busy` | superseded by #31 |
| 28 | OPEN | `fix/turboquant-graceful-fallback-no-vllm-metal` | TurboQuant crash fix |
| 31 | OPEN | `perf/postcommit-off-generation-executor` | rollup of 4 SessionBank tiers; INCLUDES `37c8cd7` and `43bd51f` |

Critical: PR #31 already contains commits `37c8cd7` (storage prefix-match) and `43bd51f` (cache_miss_reason observability). The `perf/memory-caps-and-adaptive-depth` branch is layered on top of PR #31's tip, so anything we PR from this branch must NOT include those two SHAs.

## Commit inventory

The branch contains 11 user commits beyond `origin/main`. The 8 above the `perf/postcommit-off-generation-executor` tip are net-new and candidates for new PRs; the 3 below are part of PR #31 and must be excluded.

| SHA | Message | New on this branch? | Recommended disposition |
|---|---|---|---|
| `3a8fef5` | tools/bench: standalone tool_call divergence reproducer | yes | fork-only (bench artifact) |
| `d1cf628` | perf: thread session_id through restore() so divergence dumps are per-session | yes | PR (pairs with `12d7ee8`) |
| `179bc8c` | perf: cosmetic - distinguish new_session from prefix_divergence in miss reasons | yes | PR (pairs with `12d7ee8`) |
| `12d7ee8` | perf: wait briefly for prior postcommit before next request lookup | yes | PR (highest impact) |
| `383da4f` | perf: bump sustained-profile prefill chunk size 2048 -> 4096 | yes | PR (small, clean win) |
| `d838d64` | tools/bench: adaptive depth A/B + sustained-profile knob triage | yes | fork-only (negative result + bench) |
| `add6505` | tools/bench: 40-pass game benchmark + findings | yes | fork-only (bench harness) |
| `2debb9a` | perf: pin MLX Metal memory caps + lower clear_cache threshold for long ctx | yes | PR |
| `37c8cd7` | perf: prefix-match storage encoding to lookup encoding | already in PR #31 | drop / do not double-ship |
| `43bd51f` | perf: surface postcommit cache_miss_reason for observability | already in PR #31 | drop / do not double-ship |
| `9f7446a` / `960efc1` / `110bb9d` | merge commits / Tier 1.2+2.1 rollup | already in PR #31 | drop |

## Proposed PRs

Three code PRs and one optional follow-up. Each is independently mergeable and based directly on `origin/main` (no dependency on PR #31 staying open) except where noted.

### PR A: pin MLX Metal memory caps + tune clear_cache for long context

- Title: `perf: pin MLX Metal memory caps and lower clear_cache threshold for long context`
- Branch: `perf/metal-memory-caps`
- Base on: `origin/main`
- Commits: cherry-pick `2debb9a` only.
- Summary: at >30K context on Apple Silicon the wired memory pool can grow unbounded across back-to-back requests, occasionally crashing or collapsing decode ~10x. This PR adds hard caps via `mx.set_memory_limit` / `mx.set_wired_limit` (with fallback to deprecated `mx.metal.*`) called once during `ServerState` init - defaults `memory_limit = 75% RAM`, `wired_limit = 60% RAM`, both env-overridable (`MTPLX_MEMORY_LIMIT_BYTES`, `MTPLX_WIRED_LIMIT_BYTES`, K/M/G/T suffix accepted). Lowers `MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD` from 98304 to 16384 to actually fire in the typical agent regime, bumps `MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT` from 16 to 256 to amortize barrier cost, switches default mode from `0` to `auto`.
- Test/benchmark evidence: cite `tools/bench/baseline-context-ladder-2026-05-08.txt` and `tools/bench/with-caps-context-ladder-2026-05-08.txt` (consistent -2.7 GB peak across 4K -> 48K, regression-free TTFT/decode). 42/42 in-scope unit tests pass.
- Files touched: `mtplx/generation.py` (+13/-3), `mtplx/server/openai.py` (+73/0).
- Risk: changes default behavior (clear_cache mode flips to `auto`, threshold drops). Maintainer has been merging similar SessionBank/cache reachability fixes; the env-var ergonomics match the existing `MTPLX_SESSION_BANK_*_BYTES` pattern from PR #18 (already merged from this author). Likely welcome. Concern worth flagging: any operator currently relying on `MTPLX_CLEAR_CACHE_EVERY_*=0` will see new behavior; the PR should explicitly call out the default change.

### PR B: prefill chunk 2048 -> 4096 for sustained profile

- Title: `perf: bump sustained-profile prefill chunk size 2048 -> 4096`
- Branch: `perf/prefill-chunk-4096`
- Base on: `origin/main`
- Commits: cherry-pick `383da4f` only.
- Summary: the sustained profile's prefill chunk size of 2048 leaves a TTFT cliff at >29K context. Bumping to 4096 amortizes Python loop overhead during chunked contiguous prefill. On the 40-pass context-growth benchmark this is a clean win across the board: decode +29% (39.3 -> 50.6 t/s), TTFT mean -35% (2.29s -> 1.48s), TTFT @29K -75% (6.62s -> 1.66s, cliff gone), reliability 40/40 vs 39/40, peak memory flat (37.9 GB vs 38.8 GB).
- Test/benchmark evidence: `tools/bench/findings-chunk-4096-2026-05-08.md`. Hardware: M5 Max 128GB. Model: Qwen3.6-27B (TurboQuant). Profile: sustained. Date: 2026-05-08. Fan: not specified - this is the one piece of metadata the PR body must add to satisfy CONTRIBUTING.md. Sampler/temp values are in the bench script.
- Files touched: `mtplx/profiles.py` (+4/-2) plus the findings doc.
- Risk: trivial diff, isolated profile constant. The dense verifier still fits comfortably at 4096 on M5 Max 128GB - lower-RAM hosts (32GB / 64GB) need confirmation. Recommend asking the maintainer whether to gate by RAM or just ship as-is for the sustained profile (which is already the high-RAM profile).

### PR C: serialize subagent T1->T2 by waiting briefly for prior postcommit

- Title: `perf: wait briefly for prior postcommit before next request lookup`
- Branch: `perf/postcommit-await-before-next-lookup`
- Base on: `origin/main` (or `perf/postcommit-off-generation-executor` if PR #31 lands first - see Open Questions)
- Commits: cherry-pick `12d7ee8`, `179bc8c`, `d1cf628`. Optionally include `3a8fef5` as tooling.
- Summary: in opencode-style subagent fan-out, T1's slow ~10s postcommit doesn't land before T2 arrives, so T2 cold-prefills, T2's postcommit doesn't land before T3, etc. Steady-state cache reuse only kicks in around T5. This PR has `EngineSession` track the in-flight postcommit future; the next request in the same session waits up to 10s for it to land BEFORE acquiring the session lock. Critical ordering note (and the reason this commit had a deadlock in an earlier draft that was redesigned in `12d7ee8` itself): the postcommit's `_run_store_on_generation_executor` blocks on `generation_executor`, so waiting AFTER acquiring the session lock would deadlock. The wait is therefore explicitly placed before lock acquisition. Companion commits `179bc8c` and `d1cf628` thread `session_id` through `SessionBank.restore()` so the miss-reason label is honest (`new_session` vs `prefix_divergence_at_token`), and gate the divergence diagnostic dump (`MTPLX_DUMP_DIVERGENCE=1` -> `/tmp/mtplx-divergence-<sid>-<ts>.json`) per-session and one-shot.
- Test/benchmark evidence: cite `tools/bench/findings-40pass-2026-05-08.md` (39/40 cache hits across 75 -> 38K context, sub-2s TTFT through 38K) and the per-session dump format. Add a unit test for `EngineSession`'s in-flight-future tracking and the timeout path.
- Files touched: `mtplx/engine_session.py` (+51), `mtplx/server/openai.py` (+~14), `mtplx/session_bank.py` (+~90 across the three commits), `mtplx/generation.py` (+~10).
- Risk: this is the most impactful single change. Two concerns to address up front in the PR body:
  - Deadlock: the wait must be before `state.lock.acquire()`; the PR body should call this out explicitly because a maintainer reviewing in isolation would otherwise not see why.
  - Foreground latency: the new wait can add up to 10s to inter-turn TTFT in pathological cases; should default the timeout via env var (`MTPLX_POSTCOMMIT_AWAIT_S`?) and include a benchmark showing the win is net-positive vs the worst-case wait.

### PR D (optional follow-up): standalone tool_call divergence reproducer

- Title: `tools/bench: standalone tool_call divergence reproducer`
- Branch: `tools/bench-divergence-repro`
- Base on: `origin/main`
- Commits: cherry-pick `3a8fef5` only.
- Summary: a server-less script that diffs storage-encoded vs lookup-encoded tokens for a synthetic 2-turn opencode-shaped conversation, used to confirm that storage IS a strict prefix of lookup once `37c8cd7` (in PR #31) is applied. Useful as a regression canary for future chat-template changes.
- Risk: low - it's a standalone tool with no runtime impact. May or may not be welcome (the maintainer has merged some bench harnesses, e.g. `scripts/fresh_venv_smoke.sh` is referenced by CONTRIBUTING.md, but the project doesn't have a `tools/bench/` convention). Easiest to fold this in alongside PR C rather than open a standalone PR.

## Fork-only commits (not proposed for upstream)

- `add6505` (40-pass game bench + findings docs): real-world workload harness driven by user-specific opencode prompts, plus four dated findings docs that read more like a working diary than a reference. Keep on the fork; reference specific docs in PR bodies as evidence.
- `d838d64` (adaptive depth A/B + knob triage): documents a NEGATIVE result - adaptive depth gives +6% decode but breaks 5/40 streams. Useful internally as "do not enable" but not something to upstream as code (no code change is even being proposed by it).
- `3a8fef5` (divergence reproducer): could go either way - see PR D above. Recommend folding into PR C if at all.

## Open questions

1. **PR base branch for PR C.** Should this be opened against `main` directly (carrying the assumption that PR #31 lands first), or should it stack on `perf/postcommit-off-generation-executor` and be marked as dependent on #31? Stacking is cleaner because `12d7ee8` interacts with the same `_store_retokenized_history_snapshot` code paths PR #31 modifies; opening on `main` will require a manual merge resolution. Recommend stacking and noting the dependency in the PR body, the same pattern PR #31 itself uses with #24 / #28.

2. **PR B fan-mode metadata.** `findings-chunk-4096-2026-05-08.md` does not record fan mode. CONTRIBUTING.md requires it for benchmark evidence and explicitly says "Do not use fan-controlled runs for product headline claims." If the run was fan-controlled, the PR body must downgrade the headline language; if it was uncontrolled, the body must say so explicitly.

3. **PR A defaults.** Switching `MTPLX_CLEAR_CACHE_EVERY_*` default mode from `0`/disabled to `auto` is a behavior change. Worth asking the maintainer whether they prefer an opt-in env (`MTPLX_CLEAR_CACHE_EVERY=auto` only on demand) versus the opt-out the commit currently ships. Either is defensible; the PR body should at minimum call this out as an operator-facing change in the v0.2.x line.

4. **Whether to include `179bc8c` + `d1cf628` with PR C, or split them.** They are pure observability/labeling improvements that are technically independent of `12d7ee8` (the wait fix). Splitting would let `12d7ee8` ship on its own merit. Bundling keeps the diagnostic story honest, since `12d7ee8`'s PR body cites `cache_miss_reason=new_session vs prefix_divergence_at_token` evidence that only renders correctly with `179bc8c` and `d1cf628` applied. Recommend bundling, but split is fine if the maintainer prefers smaller PRs.

5. **PR D disposition.** Maintainer has not historically taken pure tooling PRs (the merged PRs are runtime fixes). Default: skip PR D, keep `3a8fef5` on the fork.
