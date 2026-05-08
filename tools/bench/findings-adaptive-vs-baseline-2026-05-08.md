# Adaptive depth policy A/B (2026-05-08)

Compared `--adaptive-policy expected_value` vs default `none` on the 40-pass
context-growth benchmark (75 -> ~38K tokens).

## Headline

Adaptive shows a **+6% mean decode** improvement but introduces a **12.5%
error rate** at high context (>30K). Net regression. Keeping the flag
available for future tuning but **not enabling in launcher**.

## Numbers

|                  | Baseline (none) | Adaptive (expected_value) |
|------------------|-----------------|---------------------------|
| Successful       | 39/40           | 35/40                     |
| Cache hits       | 39/40           | 34/35                     |
| Decode mean t/s  | 39.3            | 41.7                      |
| Decode max t/s   | ~55             | 54.5                      |
| Errors (no usage)| 0               | 5 (passes 32,34,36,38,40) |

## Failure pattern

Errors occur on every other pass starting at pass 32 (RoadCones), through
pass 40 (Trees). All failures are stream cut-offs where the final SSE chunk
with usage stats is never emitted; client elapsed ~6.5-12.7s.

The interleaved odd/even pattern is suspicious - successful passes between
errors show 6.5s TTFT and ~33 t/s decode, while errored passes elapse the
same ~6.5s but emit no usage. Likely the adaptive controller hits a state
that occasionally aborts the stream finalization when context is large.

Server stderr is silent (no traceback/exception). Reproducing this with
extra instrumentation would require an mtplx code patch; deferring.

## Action

- Do NOT add `--adaptive-policy expected_value` to run-mtplx-fork.sh
- Keep the flag opt-in for future investigation
- Move to next perf candidate per research roadmap

## Other knobs probed (n=5 sanity, ctx 0-4K, decode mean)

Small-sample sanity to triage which sustained-profile knobs are worth a full
40-pass A/B. None showed a clear positive signal beyond noise; all left the
launcher unchanged.

| Variant                                         | mean decode t/s |
|-------------------------------------------------|-----------------|
| Baseline (top-k 20, temp 0.7, stream-interval 1)| 51.6            |
| Sharp drafts (top-k 10, temp 0.4)               | 43.4 (worse)    |
| stream-interval 4                                | 53.4 (+3%)      |
| --online-correction-cache + --prompt-correction-cache | 45.4        |

Conclusions:
- Tighter draft sampling regresses; the stock broader sampling already lands
  near the model's accept-rate sweet spot for this workload.
- stream-interval=4 is within noise; not worth changing default.
- correction caches did not improve and are likely tuned for different
  workloads; revisit only with a 40-pass A/B if a workload triggers
  correction-cache hits visibly in stats.

Larger gains past current baseline (mean 39 t/s at 17K ctx, 33 t/s at 30K+)
likely require code work: chunked prefill (Action #3 in research roadmap),
recent-window fp16 (Action #4), or instrumenting the adaptive policy stream
finalization to find the silent abort.
