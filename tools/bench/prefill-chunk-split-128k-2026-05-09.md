# Prefill chunk-size split (dense=4096 / repage=2048) - 128k validation plan

**Status: PENDING execution against a real MTPLX serving stack.**

This file is a reproducible plan for the 128k bench requested in the PR #33
review. The plan is intentionally laid out so anyone with the hardware can run
it and append numbers to the "Results" section below without rewriting the
methodology.

The numbers themselves are NOT fabricated. When the run executes against a
live model, paste the JSON outputs and the headline TTFT / prefill-rate delta
into the `Results` table.

## Hypothesis

PR #33 originally bumped a single chunk-size knob from 2048 -> 4096. Maintainer
review on 2026-05-09 flagged that 4096 in the **repage path** (contexts > 64k)
regresses 128k-context TTFT, while it remains a clear win on the **dense path**
(contexts <= 64k).

The reshape splits the knob:

- `MTPLX_PREFILL_CHUNK_SIZE_DENSE`  default `4096` (contexts <= 64k)
- `MTPLX_PREFILL_CHUNK_SIZE_REPAGE` default `2048` (contexts >  64k)
- `MTPLX_PREFILL_CHUNK_SIZE` retained as a legacy single-knob fallback.
- `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT` locked at `65536` so any prompt
  above 64k tokens always takes the repage path.

Expected outcome at 128k:
- "Old" behavior (`_REPAGE=4096`) regresses TTFT and / or prefill rate vs 2048.
- "New" behavior (`_REPAGE=2048`) holds the prior 128k baseline.
- The 32k / 64k dense-path numbers stay consistent with the +29% decode /
  -35% TTFT win recorded in `tools/bench/findings-chunk-4096-2026-05-08.md`.

## Hardware target

- Apple M5 Max, 128 GB unified memory.
- macOS 14+, MLX 0.31.x with the MTPLX fork at commit `2377a99f`.
- Default `sustained` profile (no Metal cache trim, no fan control overrides).

## Reproduction

The bench driver lives in `mtplx.prefill_bench`; the runnable command is the
ladder it emits internally for the 128k row:

```bash
# A. Dense=4096 / Repage=2048 (THIS PR, post-reshape)
MTPLX_PREFILL_CHUNK_SIZE=auto \
MTPLX_PREFILL_CHUNK_SIZE_DENSE=4096 \
MTPLX_PREFILL_CHUNK_SIZE_REPAGE=2048 \
MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=65536 \
uv run python -m mtplx.cli bench prefill-ladder \
    --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
    --profile sustained --max \
    --prompt-style coding-agent \
    --prompt-format chat \
    --disable-thinking \
    --max-tokens 256 \
    --contexts 131072 \
    --output benchmarks/results/prefill-chunk-split-new-128k-2026-05-09.json

# B. Dense=4096 / Repage=4096 (PR #33 head, pre-reshape - the regressing config)
MTPLX_PREFILL_CHUNK_SIZE=auto \
MTPLX_PREFILL_CHUNK_SIZE_DENSE=4096 \
MTPLX_PREFILL_CHUNK_SIZE_REPAGE=4096 \
MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=65536 \
uv run python -m mtplx.cli bench prefill-ladder \
    --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
    --profile sustained --max \
    --prompt-style coding-agent \
    --prompt-format chat \
    --disable-thinking \
    --max-tokens 256 \
    --contexts 131072 \
    --output benchmarks/results/prefill-chunk-split-old-128k-2026-05-09.json
```

Compare `ttft_seconds` and `prefill_tokens_per_second` from the two JSONs at
`contexts=131072`. The relevant gate is `tests/test_prefill_tps_regression.py`
which expects M5 Max prefill rate >= 240 t/s at 131072 (and >=325 at 65536,
>=500 at 32768).

A short sanity check on the dense path is also useful:

```bash
# C. Dense path sanity at 32k / 64k - should match the 4096 win.
MTPLX_PREFILL_CHUNK_SIZE=auto \
MTPLX_PREFILL_CHUNK_SIZE_DENSE=4096 \
MTPLX_PREFILL_CHUNK_SIZE_REPAGE=2048 \
MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=65536 \
uv run python -m mtplx.cli bench prefill-ladder \
    --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
    --profile sustained --max \
    --prompt-style coding-agent \
    --prompt-format chat \
    --disable-thinking \
    --max-tokens 256 \
    --contexts 32768,65536 \
    --output benchmarks/results/prefill-chunk-split-dense-2026-05-09.json
```

## Results

> Run was NOT executed in this session: no MTPLX serving stack and no model
> weights were available on the bench machine when this PR comment was
> assembled. Numbers below are intentionally left as `<pending>`. Replace them
> in-place after running the three commands above; do not amend the
> hypothesis section.

| Context | Variant                          | TTFT (s)   | Prefill t/s | Decode t/s | Peak mem (MB) | Notes |
|---------|----------------------------------|------------|-------------|------------|---------------|-------|
| 32,768  | C: dense 4096 / repage 2048      | `<pending>`| `<pending>` | `<pending>`| `<pending>`   | dense-path sanity |
| 65,536  | C: dense 4096 / repage 2048      | `<pending>`| `<pending>` | `<pending>`| `<pending>`   | dense-path boundary |
| 131,072 | A: dense 4096 / repage 2048 (new)| `<pending>`| `<pending>` | `<pending>`| `<pending>`   | repage path |
| 131,072 | B: dense 4096 / repage 4096 (old)| `<pending>`| `<pending>` | `<pending>`| `<pending>`   | repage path, regressing |

### Headline delta at 131,072

`<pending>` - replace with: e.g. `old (repage 4096) ttft N.NNs vs new (repage
2048) ttft M.MMs at 128k - prefill rate X t/s vs Y t/s.`

## Why "pending" is the honest answer here

The reshape itself does not depend on the bench - it is a code revert of a
single line in `SUSTAINED_PREFILL_ENV` plus a profile lock at the dense cutoff
plus 4 unit tests that pin path selection and env honoring. The unit tests
are sufficient to gate the code change. The 128k bench is required to confirm
the *empirical* maintainer claim, which can only be measured against the real
model on M5 Max hardware.

If you are reviewing this PR and have access to the bench rig, please run the
three commands above and replace the `<pending>` cells. If the new behavior
fails to recover the 128k baseline, the dense cutoff in
`MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT` should be revisited - 64k is the
maintainer-recommended setting per the 2026-05-09 review.
