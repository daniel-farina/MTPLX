# Upstream PR execution report

Date: 2026-05-08
Local repo: `/Users/dan/code-2/forks/MTPLX`
Fork: `daniel-farina/MTPLX`
Upstream: `youssofal/MTPLX`
Plan: [`upstream-pr-plan-2026-05-08.md`](./upstream-pr-plan-2026-05-08.md)

## PRs created

### PR #32 - Metal memory caps + clear_cache tuning

- URL: https://github.com/youssofal/MTPLX/pull/32
- Title: `perf: pin MLX Metal memory caps and lower clear_cache threshold for long context`
- Branch (fork): `daniel-farina:pr/metal-memory-caps`
- Base: `youssofal:main`
- Commits: `2debb9a` (cherry-picked, lands as `3b0fb1f` on the new branch)
- Files: `mtplx/generation.py` (+13/-3), `mtplx/server/openai.py` (+73/0)
- Status: open, mergeable
- Notes:
  - Body explicitly flags the operator-facing default change (`MTPLX_CLEAR_CACHE_EVERY_*` mode flips from `0`/disabled to `auto`, threshold drops 98304 -> 16384, long-context interval bumps 16 -> 256). Maintainer is told they can gate behind opt-in if they prefer.
  - Benchmark evidence cites both ladder files via permalinks on the working branch.

### PR #33 - Prefill chunk 2048 -> 4096

- URL: https://github.com/youssofal/MTPLX/pull/33
- Title: `perf: bump sustained-profile prefill chunk size 2048 -> 4096`
- Branch (fork): `daniel-farina:pr/prefill-chunk-4096`
- Base: `youssofal:main`
- Commits: `383da4f` (cherry-picked, lands as `9de29ed`)
- Files: `mtplx/profiles.py` (+4/-2), `tools/bench/findings-chunk-4096-2026-05-08.md` (+85)
- Status: open, mergeable
- Notes:
  - Body documents the fan-mode explicitly: sustained profile is not fan-controlled (`fan_control_allowed=False`); CONTRIBUTING.md's caveat about fan-controlled headlines is therefore satisfied.
  - Body flags the lower-RAM (32GB / 64GB) host as not validated and notes the change is scoped to the high-RAM sustained profile.
  - The cherry-picked commit bundles the findings doc into `tools/bench/`. Left as-is since it's small and supports the evidence inline; maintainer can drop if undesired.

### PR #34 - Postcommit await + per-session miss reasons

- URL: https://github.com/youssofal/MTPLX/pull/34
- Title: `perf: wait briefly for prior postcommit before next request lookup`
- Branch (fork): `daniel-farina:pr/postcommit-await-before-next-lookup`
- Base: `youssofal:main` (with explicit dependency on #31 in body)
- Commits cherry-picked: `12d7ee8`, `179bc8c`, `d1cf628` (top-of-stack on PR #31's branch)
- Files (relative to #31 tip): `mtplx/engine_session.py` (+51), `mtplx/server/openai.py` (+~14 net), `mtplx/session_bank.py` (+~90), `mtplx/generation.py` (+~10)
- Status: open, depends on #31 landing first
- Notes:
  - Cherry-picking onto `main` directly produces a structural conflict in `mtplx/server/openai.py`: HEAD now uses the `_submit_idle_postcommit_model_work` abstraction (post-#29 era), while `12d7ee8` was written against the pre-#29 `executor.submit(async_postcommit)` shape. Resolving that conflict requires deciding how the future-tracking hook integrates with the new abstraction - more than mechanical, so the PR was stacked on PR #31's head instead, per the plan's recommendation.
  - GitHub diff against `main` includes the 13 commits from #31 plus the 3 from this PR (16 total). Body explicitly tells the reviewer this and offers to rebase post-#31.
  - Deadlock-avoidance ordering note is in the body (wait must precede `state.lock.acquire()`).
  - Real-world impact cited: 39/40 cache hits on the 40-pass benchmark, T2 TTFT 1.06s, mean TTFT 1.54s sub-2s through 38K ctx.

## PR D - Skipped per plan

The plan recommended skipping the standalone divergence reproducer (`3a8fef5`) since the maintainer has not historically taken pure tooling PRs. The commit remains on the fork's working branch.

## Branch state

### Fork (`daniel-farina/MTPLX`)

| Branch | Purpose | State |
|---|---|---|
| `perf/memory-caps-and-adaptive-depth` | User's working branch | UNTOUCHED |
| `perf/postcommit-off-generation-executor` | PR #31 head | UNTOUCHED |
| `pr/metal-memory-caps` | PR #32 head | NEW |
| `pr/prefill-chunk-4096` | PR #33 head | NEW |
| `pr/postcommit-await-before-next-lookup` | PR #34 head | NEW |
| `fix/turboquant-graceful-fallback-no-vllm-metal` | PR #28 head | UNTOUCHED |

### Upstream (`youssofal/MTPLX`)

| PR | State | Title | Notes |
|---|---|---|---|
| #28 | OPEN | TurboQuant graceful fallback | UNTOUCHED |
| #31 | OPEN | SessionBank cache reuse rollup | UNTOUCHED (still contains `37c8cd7`, `43bd51f`) |
| #32 | OPEN | Metal memory caps | NEW |
| #33 | OPEN | Prefill chunk 4096 | NEW |
| #34 | OPEN | Postcommit await | NEW (depends on #31) |

No closed/merged PRs were touched. Nothing was pushed to upstream.

## Verification of constraint compliance

- PR #31 is untouched (still open with `37c8cd7`, `43bd51f` at tip).
- Commits `37c8cd7` and `43bd51f` do NOT appear in PR #32 or PR #33 (verified: each contains exactly 1 commit).
- They DO appear in PR #34's diff, but only because PR #34 stacks on PR #31's branch as the plan recommended; the body discloses this explicitly. After #31 merges, #34 should be rebased and its diff will collapse to the 3 cherry-picked commits.
- Working branch `perf/memory-caps-and-adaptive-depth` was checked back out at the end and was not modified.
- All pushes went to `fork` only. Upstream was used only via `gh pr create`.

## Follow-up actions for the user

1. **Wait for #31 to merge before #34 review starts in earnest.** Once #31 is merged, rebase `pr/postcommit-await-before-next-lookup` onto `origin/main` and resolve the `_submit_idle_postcommit_model_work` conflict. The conflict resolution requires deciding how to thread the future-tracking hook into the new abstraction.
2. **Address open question 3 from the plan** (PR #32 default flip) if the maintainer pushes back. The body offers to switch to opt-in (`MTPLX_CLEAR_CACHE_EVERY=auto` only on demand) - a one-commit follow-up.
3. **Add a unit test for `EngineSession`'s in-flight-future tracking and timeout path** before #34 review goes deep. The plan called this out as in-scope; it's not in the cherry-picked commits.
4. **Optional**: open PR D (`3a8fef5`) standalone if maintainer expresses interest. Default is to keep on fork.
5. **Optional**: provide a benchmark comparing decode/TTFT with `MTPLX_POSTCOMMIT_AWAIT_S` set to 0 vs 10s, to give the maintainer net-impact data on the worst-case 10s wait. Could be added as a comment to #34.

## Summary of what's neat now

- 3 new clean PRs upstream, each with the required template sections (Summary / Verification / Benchmark Evidence) and CONTRIBUTING.md-mandated metadata (hardware, model, quantization, sampler, token count, profile, fan mode, date, commit).
- Each new PR has its own dedicated branch on the fork (`pr/<short-name>`).
- The user's working branch is untouched.
- The user's existing open PRs (#28, #31) are untouched.
- All concerns flagged in the plan's open-questions section are surfaced explicitly in PR bodies (default flip in #32, fan-mode in #33, deadlock-avoidance + foreground-latency + dependency on #31 in #34).
