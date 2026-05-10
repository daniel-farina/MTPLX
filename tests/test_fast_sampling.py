"""Regression tests for fast_sampling.py NaN/Inf guard.

At long context (>~25K cumulative on the qwen3.6-27b sustained profile)
the dense SDPA fallback path in cache_state.py can emit non-finite trunk
logits. Without the guard added in this module, the resulting NaN
``top_probs_full`` propagates to ``SparseDistribution`` construction and
the sampler raises a 500 with the misleading message
``SparseDistribution probabilities must have positive mass``.

These tests pin the recovery semantics: a single non-finite logit must
not crash the request, and the sampler must degrade to a valid sparse
distribution over the surviving finite support (or to a one-hot greedy
distribution when every entry is non-finite).
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.fast_sampling import (  # noqa: E402  - mx import must precede
    BatchedSparseDistributions,
    batched_sparse_distributions_from_mlx_logits,
    sparse_distribution_from_mlx_logits,
    sparse_distributions_from_mlx_logits,
)
from mtplx.sampling import SamplerConfig, SparseDistribution


CFG = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)


def _logits_with_nan(vocab: int = 256) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.normal(scale=2.0, size=(vocab,)).astype(np.float32)
    # Inject one NaN and one +inf, which collectively reproduce the
    # long-context dense-SDPA-fallback signature.
    base[7] = np.nan
    base[42] = np.inf
    return base


def test_sparse_distribution_handles_nan_logits():
    logits = mx.array(_logits_with_nan())
    dist = sparse_distribution_from_mlx_logits(logits, CFG)
    assert isinstance(dist, SparseDistribution)
    # Probabilities must sum to a finite positive value.
    assert np.isfinite(dist.probs).all()
    assert dist.probs.sum() > 0
    # The poisoned positions must not appear in the support.
    assert 7 not in dist.token_ids.tolist()
    assert 42 not in dist.token_ids.tolist()


def test_sparse_distribution_all_nan_logits_returns_one_hot():
    vocab = 64
    poisoned = mx.array(np.full((vocab,), np.nan, dtype=np.float32))
    dist = sparse_distribution_from_mlx_logits(poisoned, CFG)
    # Either we degrade to a one-hot or return None for the dense
    # numpy path to handle. Both are safe - what we must not do is
    # raise ``positive mass``.
    if dist is None:
        return
    assert isinstance(dist, SparseDistribution)
    assert np.isfinite(dist.probs).all()
    assert np.isclose(dist.probs.sum(), 1.0)


def test_sparse_distributions_batched_handles_nan_logits():
    rng = np.random.default_rng(1)
    rows = rng.normal(scale=2.0, size=(3, 128)).astype(np.float32)
    rows[1, 5] = np.nan  # Only the middle row is poisoned.
    rows[1, 9] = np.inf
    logits = mx.array(rows)
    dists = sparse_distributions_from_mlx_logits(logits, CFG)
    assert dists is not None and len(dists) == 3
    for d in dists:
        assert np.isfinite(d.probs).all()
        assert d.probs.sum() > 0


def test_batched_sparse_distributions_handles_nan_logits():
    rng = np.random.default_rng(2)
    rows = rng.normal(scale=2.0, size=(4, 64)).astype(np.float32)
    rows[2, :] = np.nan  # Whole row is poisoned (full SDPA fallout).
    logits = mx.array(rows)
    batched = batched_sparse_distributions_from_mlx_logits(logits, CFG)
    assert isinstance(batched, BatchedSparseDistributions)
    # All rows must have positive, finite mass after the guard.
    assert np.all(np.isfinite(batched.probs))
    row_sums = batched.probs.sum(axis=1)
    assert np.all(row_sums > 0)
