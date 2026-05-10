"""Runtime sampler helpers for sparse top-k speculative sampling."""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from .sampling import SamplerConfig, SparseDistribution


def _trunk_logits_nan_guard_enabled() -> bool:
    """Return whether the NaN/Inf guard on trunk logits is active.

    Defaults to enabled. Set ``MTPLX_TRUNK_LOGITS_NAN_GUARD=0`` to disable
    (only for debugging - disabling re-exposes the long-context
    ``SparseDistribution probabilities must have positive mass`` 500.).
    """

    raw = os.environ.get("MTPLX_TRUNK_LOGITS_NAN_GUARD")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _sanitize_logits_row(row: np.ndarray) -> tuple[np.ndarray, bool]:
    """Replace non-finite entries with ``-inf`` so argpartition/logsumexp behave.

    Returns the (possibly copied) row and a flag indicating whether any
    sanitization happened. The caller uses the flag to log diagnostic state
    so we can tie production crashes back to upstream attention NaN sources.
    """

    finite_mask = np.isfinite(row)
    if bool(finite_mask.all()):
        return row, False
    sanitized = np.where(finite_mask, row, -np.inf)
    return sanitized, True


def _record_trunk_logits_nan(*, batched: bool, rows: int) -> None:
    """Log a single line per occurrence so operators can see frequency.

    Kept intentionally cheap (one line, no traceback) - the goal is to
    surface that the guard fired without spamming the log when an entire
    batch hits the same upstream NaN.
    """

    if os.environ.get("MTPLX_TRUNK_LOGITS_NAN_GUARD_QUIET"):
        return
    import sys

    print(
        "mtplx_trunk_logits_nan_guard "
        f"batched={int(bool(batched))} rows_with_nan={int(rows)}",
        file=sys.stderr,
    )


class BatchedSparseDistributions:
    def __init__(
        self,
        token_ids: np.ndarray,
        probs: np.ndarray,
        *,
        vocab_size: int,
    ) -> None:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        if token_ids.ndim != 2 or probs.ndim != 2:
            raise ValueError("BatchedSparseDistributions expects 2D arrays")
        if token_ids.shape != probs.shape:
            raise ValueError("token_ids/probs shape mismatch")
        row_sums = probs.sum(axis=1)
        if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
            raise ValueError("each sparse distribution row needs positive mass")
        self.token_ids = token_ids
        self.probs = probs / row_sums[:, None]
        self.vocab_size = int(vocab_size)

    def probability(self, row: int, token_id: int) -> float:
        hits = np.nonzero(self.token_ids[int(row)] == int(token_id))[0]
        if hits.size == 0:
            return 0.0
        return float(self.probs[int(row), int(hits[0])])

    def to_distribution(self, row: int) -> SparseDistribution:
        row = int(row)
        keep = self.probs[row] > 0
        return SparseDistribution(
            self.token_ids[row, keep],
            self.probs[row, keep],
            self.vocab_size,
        )

    def sample(self, row: int, rng: np.random.Generator) -> int:
        row = int(row)
        keep = self.probs[row] > 0
        return int(rng.choice(self.token_ids[row, keep], p=self.probs[row, keep]))


def sparse_distribution_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> SparseDistribution | None:
    """Return an exact sparse distribution for top-p then top-k sampling.

    The Qwen coding sampler uses `top_k=20`, so the final support can never be
    larger than 20 tokens. We still compute the full-vocab logsumexp on MLX so
    top-p decisions use true full-distribution probability mass, then move only
    the small support to NumPy for deterministic speculative correction.
    """

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    flat = logits.reshape(-1).astype(mx.float32) / float(config.temperature)
    vocab_size = int(flat.shape[-1])
    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None

    # Sanitize non-finite logits before argpartition / logsumexp. At long
    # context (>~25K cumulative) the dense SDPA fallback in
    # ``cache_state.py:_large_q_split_sdpa_fallback`` and chunked-attention
    # paths can produce NaN/Inf trunk logits (e.g. fp16 overflow in
    # `running_acc` or rare divide-by-eps when `running_denom` underflows).
    # Without this guard the NaN propagates: ``mx.logsumexp`` returns NaN,
    # ``mx.exp(top_vals - log_total)`` is NaN, and the downstream
    # ``probs.sum() <= 0`` rescue at the bottom of this function does NOT
    # catch NaN (NaN <= 0 is False). The result is a 500 raised at
    # ``sampling.py:40`` (``SparseDistribution probabilities must have
    # positive mass``). Replacing non-finite entries with a large negative
    # constant degrades to argmax-over-finite-support without crashing.
    if _trunk_logits_nan_guard_enabled():
        flat = mx.where(mx.isfinite(flat), flat, mx.array(-1.0e30, dtype=flat.dtype))

    top_idx = mx.argpartition(-flat, kth=k - 1, axis=-1)[:k]
    top_vals = flat[top_idx]
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = top_idx[order]
    top_vals = top_vals[order]

    log_total = mx.logsumexp(flat, axis=-1)
    top_probs_full = mx.exp(top_vals - log_total)
    mx.eval(top_idx, top_probs_full)

    token_ids = np.asarray(top_idx, dtype=np.int64).reshape(-1)
    probs_full = np.asarray(top_probs_full, dtype=np.float64).reshape(-1)

    # Defense in depth: if logsumexp itself was non-finite (e.g. every
    # logit was -inf because every entry was non-finite, or fp32 overflow
    # produced +inf), report it once and degrade to a uniform distribution
    # over the top-k support. This must happen BEFORE the top-p mask so
    # the rescue branch survives the np.cumsum(NaN < top_p) collapse.
    if not bool(np.all(np.isfinite(probs_full))):
        _record_trunk_logits_nan(batched=False, rows=1)
        token_ids = token_ids[:1]
        probs = np.array([1.0], dtype=np.float64)
        return SparseDistribution(token_ids=token_ids, probs=probs, vocab_size=vocab_size)

    if 0 < config.top_p < 1.0:
        cumulative_before = np.concatenate(([0.0], np.cumsum(probs_full[:-1])))
        keep = cumulative_before < float(config.top_p)
        if keep.size:
            keep[0] = True
    else:
        keep = np.ones_like(probs_full, dtype=bool)

    token_ids = token_ids[keep]
    probs = probs_full[keep]
    # Use !(>0) instead of `<= 0` so NaN totals also trip the rescue
    # branch (NaN <= 0 is False; not (NaN > 0) is True).
    total = float(probs.sum())
    if not (total > 0):
        token_ids = token_ids[:1]
        probs = np.array([1.0], dtype=np.float64)

    return SparseDistribution(token_ids=token_ids, probs=probs, vocab_size=vocab_size)


def sparse_distributions_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> list[SparseDistribution] | None:
    """Return exact sparse distributions for a batch of logit rows.

    This is the batched equivalent of ``sparse_distribution_from_mlx_logits``.
    It keeps the same top-k/top-p semantics but shares the MLX materialization
    boundary across rows.
    """

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32) / float(config.temperature)
    vocab_size = int(rows.shape[-1])
    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None

    # See sparse_distribution_from_mlx_logits for rationale on this guard.
    if _trunk_logits_nan_guard_enabled():
        rows = mx.where(mx.isfinite(rows), rows, mx.array(-1.0e30, dtype=rows.dtype))

    top_idx = mx.argpartition(-rows, kth=k - 1, axis=-1)[:, :k]
    top_vals = mx.take_along_axis(rows, top_idx, axis=-1)
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = mx.take_along_axis(top_idx, order, axis=-1)
    top_vals = mx.take_along_axis(top_vals, order, axis=-1)

    log_total = mx.logsumexp(rows, axis=-1)
    top_probs_full = mx.exp(top_vals - log_total[:, None])
    mx.eval(top_idx, top_probs_full)

    token_rows = np.asarray(top_idx, dtype=np.int64)
    prob_rows = np.asarray(top_probs_full, dtype=np.float64)
    distributions: list[SparseDistribution] = []
    nan_rows = 0

    for token_ids, probs_full in zip(token_rows, prob_rows, strict=True):
        if not bool(np.all(np.isfinite(probs_full))):
            # Degenerate row - degrade to greedy on the top-k argmax.
            nan_rows += 1
            distributions.append(
                SparseDistribution(
                    token_ids=token_ids[:1],
                    probs=np.array([1.0], dtype=np.float64),
                    vocab_size=vocab_size,
                )
            )
            continue

        if 0 < config.top_p < 1.0:
            cumulative_before = np.concatenate(([0.0], np.cumsum(probs_full[:-1])))
            keep = cumulative_before < float(config.top_p)
            if keep.size:
                keep[0] = True
        else:
            keep = np.ones_like(probs_full, dtype=bool)

        kept_ids = token_ids[keep]
        probs = probs_full[keep]
        # Use !(>0) instead of `<= 0` so NaN totals also rescue.
        total = float(probs.sum())
        if not (total > 0):
            kept_ids = kept_ids[:1]
            probs = np.array([1.0], dtype=np.float64)

        distributions.append(
            SparseDistribution(
                token_ids=kept_ids,
                probs=probs,
                vocab_size=vocab_size,
            )
        )

    if nan_rows:
        _record_trunk_logits_nan(batched=True, rows=nan_rows)

    return distributions


def batched_sparse_distributions_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> BatchedSparseDistributions | None:
    """Return batched sparse distributions without per-row Python objects."""

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32) / float(config.temperature)
    vocab_size = int(rows.shape[-1])
    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None

    # See sparse_distribution_from_mlx_logits for rationale on this guard.
    if _trunk_logits_nan_guard_enabled():
        rows = mx.where(mx.isfinite(rows), rows, mx.array(-1.0e30, dtype=rows.dtype))

    top_idx = mx.argpartition(-rows, kth=k - 1, axis=-1)[:, :k]
    top_vals = mx.take_along_axis(rows, top_idx, axis=-1)
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = mx.take_along_axis(top_idx, order, axis=-1)
    top_vals = mx.take_along_axis(top_vals, order, axis=-1)

    log_total = mx.logsumexp(rows, axis=-1)
    top_probs_full = mx.exp(top_vals - log_total[:, None])
    mx.eval(top_idx, top_probs_full)

    token_rows = np.asarray(top_idx, dtype=np.int64)
    prob_rows = np.asarray(top_probs_full, dtype=np.float64)

    # Defense in depth: collapse non-finite rows to a one-hot on their
    # top-1 token before downstream arithmetic. Mirrors the per-row
    # sanitization in ``sparse_distributions_from_mlx_logits``.
    finite_per_row = np.all(np.isfinite(prob_rows), axis=1)
    if not bool(finite_per_row.all()):
        nan_rows = int((~finite_per_row).sum())
        _record_trunk_logits_nan(batched=True, rows=nan_rows)
        bad_rows = ~finite_per_row
        prob_rows[bad_rows, :] = 0.0
        prob_rows[bad_rows, 0] = 1.0

    if 0 < config.top_p < 1.0:
        cumulative_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        keep = cumulative_before < float(config.top_p)
        if keep.size:
            keep[:, 0] = True
        prob_rows = np.where(keep, prob_rows, 0.0)

    row_sums = prob_rows.sum(axis=1)
    # Use ``not (>0)`` so NaN row-sums also trigger the rescue. NaN <= 0
    # is False under IEEE-754, which is why the original guard missed
    # the long-context degenerate-distribution case at sampling.py:40.
    bad = ~(row_sums > 0)
    if np.any(bad):
        prob_rows[bad, :] = 0.0
        prob_rows[bad, 0] = 1.0

    return BatchedSparseDistributions(token_rows, prob_rows, vocab_size=vocab_size)
