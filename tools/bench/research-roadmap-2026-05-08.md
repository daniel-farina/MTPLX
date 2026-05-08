# MTPLX Long-Context Perf Research (40K, Qwen3.6-27B, M5 Max 128GB)

## Executive Summary
- The 30K+ collapse to ~2.5 t/s + crash is consistent with the documented MLX wired-memory + KV cache pathology: wired pool keeps growing, `mx.metal.clear_cache()` is never called or threshold is too high, swap kicks in, throughput drops 10x and the kernel kills the process. Fixing the memory cap and adding a per-N-token cache clear is the highest ROI single change.
- Speculative depth=3 is too aggressive at 40K. Published Qwen3.6 numbers show n=4 peaks only at short context; at long context the optimum is n=1-2 because draft accept rate drops and the verify forward pass dominates. Adaptive depth schedule (3 -> 2 -> 1 by ctx threshold) is a one-day change with documented gains.
- KV cache quantization (already present via TurboQuant 3-bit/group=64/affine) is correct in principle but you are paying the per-token dequant cost. Published numbers: q4_0 KV gives a 36.8% decode penalty at 110K context (38 -> 24 t/s). This is consistent with your 40-50 -> 2.5 t/s curve being dominated by something *else* (memory swap), not the quant cost. Don't blame the quant; fix memory first.
- Sustained mode in MTPLX explicitly skips oversized SessionBank snapshots and avoids full-K/V materialization but it still relies on Metal not being swap-bound. Tighten `set_wired_limit` to 60% (77 GB on 128 GB), `set_memory_limit` to 96 GB hard cap, and add chunked prefill below 8K chunks.
- Qwen3.6-27B dense is documented to degrade harder at 100K+ than its MoE 35B-A3B sibling. On 128 GB you are not memory-bound at 40K (KV at 40K, dense 27B, q3 = ~6-8 GB), so the crash is allocator/wired-limit fragmentation, not absolute memory. This means it is fixable in config.

## Top 5 Actions (impact / effort)

### 1. Tighten MLX memory caps + add periodic `mx.metal.clear_cache()`  [HIGH / 1 hour]
- **Change**: At server startup call `mx.metal.set_memory_limit(96 * 1024**3)` and `mx.set_wired_limit(int(0.60 * total_ram))`. In the decode loop, call `mx.metal.clear_cache()` every ~512 generated tokens, and unconditionally at end-of-request.
- **Why**: GitHub mlx-lm issue #883 documents exactly your failure mode (kernel panic / OOM-ish at long ctx because wired memory bypasses macOS pressure monitor). The "MLX Memory Safety Checklist" recommends both calls; per-call clears cost ~5-10ms but threshold-based (every 512 tok) is essentially free. Issue #945 in mlx-vlm specifically warns that per-chunk `mx.eval()` + `mx.clear_cache()` causes sync barriers, so do it on a counter, not per-step.
- **Expected**: Eliminates the 30K crash. Decode at 30-40K should hold within 10-15% of the 16-20K rate (so ~35-45 t/s instead of 2.5 t/s). This is the dominant fix.
- **Effort**: One-line set at startup, ~10 line counter in the decode loop. (Sources: ml-explore/mlx-lm#883, dev.to MLX Memory Safety Checklist, Blaizzy/mlx-vlm#945)

### 2. Adaptive speculative depth (depth schedule by context length)  [HIGH / 1 day]
- **Change**: Replace fixed `spec_depth=3` with `depth = 3 if ctx<16K else 2 if ctx<32K else 1`. Plumb `n_speculative` as a per-decode-step parameter, not a server-wide constant.
- **Why**: Qwen3.6 official docs note "n=2 underutilises the head; n=6+ drops accept rate faster than throughput grows. n=4 lands at ~85% accept × ~101 tok/s" - but those numbers are at short context. The MagicDec paper and SpecKV both show acceptance drops at long ctx and the optimum γ collapses to 1-2. Your decode at 16-20K (40-50 t/s) is already showing the depth=3 being marginal; at 40K it is actively hurting because every rejected speculative slot costs a full verify pass on a 40K KV.
- **Expected**: 1.3-1.8x decode at 30-40K (so ~45-65 t/s if memory issue #1 is also fixed). Independent of #1.
- **Effort**: ~50 LOC in the spec-decode path, plus a config flag for the schedule. (Sources: Qwen3.6-27B HF discussion #17, MagicDec arXiv 2408.11049, SpecKV arXiv 2605.02888, Decoding Speculative Decoding NAACL 2025)

### 3. Chunked prefill at 4K-8K chunks  [MEDIUM-HIGH / 2-3 days]
- **Change**: Split the 40K prompt into 4K or 8K chunks, run prefill chunk-by-chunk into the paged KV. The vllm-mlx paper explicitly notes: "Neither MLX nor MLC yet implement chunked prefill as in vLLM, leaving very long inputs (e.g., 100k+) with poor TTFT." MTPLX sustained mode claims chunked prefill but verify the chunk size; vllm-mlx defaults to 2K-4K.
- **Why**: Your TTFT of 60-180s at 40K means prefill is monolithic or chunks are huge. vllm-mlx hits 525 t/s text on M4 Max with 2-4K chunks. M5 Max should beat that.
- **Expected**: TTFT 60-180s -> 30-60s at 40K. Smaller chunks also reduce peak memory during prefill (helps #1).
- **Effort**: If MTPLX already has the path: ~half day to tune chunk size + verify. If not: ~3 days to wire it through paged-attn correctly. (Sources: arXiv 2601.19139 vllm-mlx paper, github.com/waybarrios/vllm-mlx)

### 4. Lower KV-quant aggressiveness on the most-recent window  [MEDIUM / 1 day]
- **Change**: Keep TurboQuant 3-bit/g64/affine for tokens older than 4K from the head, but keep the most recent 1K-2K tokens at fp16 (or 4-bit). This is the SnapKV/StreamingLLM "sink + recent window" pattern.
- **Why**: TurboQuant community note: "2-bit values cause significant cosine similarity degradation (~0.94), 4-bit maintains 0.997. TurboQuant really shines at 4K+ tokens, with most implementations keeping the most recent 128-256 tokens in full FP16." Your 3-bit MTP draft head accept rate likely *is* dropping at long context partly because the recent KV window is over-quantized; the draft head reads from quantized K/V and its distribution diverges from the verify model's.
- **Expected**: +5-15% draft accept rate at 30K+ -> ~10-20% net decode improvement. Compounds with #2.
- **Effort**: ~1 day if TurboQuant exposes a per-position quant policy hook; otherwise a small kernel modification. (Sources: ggml-org/llama.cpp discussion #20969, dasroot.net KV quant agentic coding 2026)

### 5. M5 Max neural-accelerator path + thermal-aware sustained throttle  [LOW-MEDIUM / variable]
- **Change**: (a) Verify MLX is using the M5 GPU neural accelerators (Apple ML research blog confirms MLX has M5-specific kernels). (b) Read `powermetrics` thermal pressure and back off speculative depth one notch when GPU enters "Heavy" or "Trapping". M5 Max in MacBook Pro chassis sustains full perf per Apple's docs, but Mac Studio thermal envelope is different.
- **Why**: 5-10% throughput loss reported in 30+ minute sessions on M5 Max even in actively cooled chassis. Combining thermal back-off with depth schedule (#2) gives a smoother sustained curve. Apple's WWDC25 MLX talk explicitly recommends checking neural accelerator dispatch for M5.
- **Expected**: 5-10% sustained-rate improvement on long sessions; mostly a stability win, not a peak win.
- **Effort**: Thermal monitor: half day. Neural accelerator verification: depends on MLX version. (Sources: machinelearning.apple.com "Exploring LLMs with MLX and Neural Accelerators in M5 GPU", WWDC25 session 298, creativestrategies M5 Max chiplets)

## Things to skip
- Continuous batching: MTPLX explicitly disclaims it; single-stream is fine for your workload.
- Switching to Qwen3.6-35B-A3B MoE: documented to degrade more gracefully at 190K, but at 40K the 27B dense is faster on a 128GB Mac and you ruled out model swaps.
- EAGLE-3 / Medusa drafter: re-training a drafter is weeks of work; native MTP heads + adaptive depth (#2) gets 80% of the benefit.

## Sources
- https://github.com/ml-explore/mlx-lm/issues/883 (kernel panic / unbounded memory)
- https://github.com/Blaizzy/mlx-vlm/issues/945 (clear_cache sync barrier)
- https://dev.to/sleepyquant/mlx-memory-safety-checklist-6-layer-defense-for-m1m2-apple-silicon-2cbj
- https://huggingface.co/Qwen/Qwen3.6-27B/discussions/17 (MTP n=3 accept rates 97/95/91, no gains beyond)
- https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914
- https://arxiv.org/html/2408.11049 (MagicDec long-context spec decoding)
- https://arxiv.org/html/2605.02888 (SpecKV adaptive gamma)
- https://github.com/ggml-org/llama.cpp/discussions/20969 (TurboQuant decode degradation)
- https://dasroot.net/posts/2026/05/kv-cache-quantization-agentic-coding-long-horizon/
- https://arxiv.org/abs/2601.19139 (vllm-mlx paper, chunked prefill gap)
- https://github.com/waybarrios/vllm-mlx
- https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- https://developer.apple.com/videos/play/wwdc2025/298/
- https://github.com/youssofal/MTPLX/releases (sustained mode semantics)
- https://creativestrategies.com/research/m5-max-chiplets-thermals-and-performance-per-watt/
- https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
