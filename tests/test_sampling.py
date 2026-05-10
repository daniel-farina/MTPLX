from __future__ import annotations

import numpy as np

from mtplx.sampling import (
    SamplerConfig,
    SparseDistribution,
    acceptance_probability,
    distribution_from_logits,
    residual_distribution,
    sample_from_distribution,
    speculative_output_marginal,
    verify_one_token,
)


def test_distribution_from_logits_normalizes_after_filtering():
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    probs = distribution_from_logits(logits, SamplerConfig(temperature=0.6, top_p=0.9, top_k=2))
    assert np.isclose(probs.sum(), 1.0)
    assert np.count_nonzero(probs) <= 2


def test_acceptance_probability_caps_at_one():
    target = np.array([0.8, 0.2])
    draft = np.array([0.4, 0.6])
    assert acceptance_probability(target, draft, 0) == 1.0
    assert np.isclose(acceptance_probability(target, draft, 1), 1.0 / 3.0)


def test_residual_distribution_uses_positive_target_minus_draft_mass():
    target = np.array([0.6, 0.3, 0.1])
    draft = np.array([0.2, 0.5, 0.3])
    residual = residual_distribution(target, draft)
    assert np.isclose(residual.sum(), 1.0)
    assert residual[0] == 1.0
    assert residual[1] == 0.0


def test_verify_one_token_rejects_into_residual_when_random_is_high():
    target = np.array([0.2, 0.8])
    draft = np.array([0.8, 0.2])
    rng = np.random.default_rng(1)
    decision = verify_one_token(target, draft, 0, rng)
    assert decision.accept_probability == 0.25
    if not decision.accepted:
        assert decision.token_id == 1


def test_speculative_output_marginal_recovers_target_distribution():
    target = np.array([0.55, 0.25, 0.15, 0.05])
    draft = np.array([0.10, 0.55, 0.20, 0.15])
    marginal = speculative_output_marginal(target, draft)
    assert np.allclose(marginal, target)


def test_sparse_distribution_acceptance_and_residual():
    target = SparseDistribution(
        token_ids=np.array([2, 5, 9]),
        probs=np.array([0.5, 0.3, 0.2]),
        vocab_size=12,
    )
    draft = SparseDistribution.one_hot(5, vocab_size=12)

    assert np.isclose(acceptance_probability(target, draft, 5), 0.3)
    residual = residual_distribution(target, draft)
    assert isinstance(residual, SparseDistribution)
    assert residual.token_ids.tolist() == [2, 9]
    assert np.allclose(residual.probs, [5 / 7, 2 / 7])


def test_sparse_distribution_sampling_returns_original_token_ids():
    dist = SparseDistribution(
        token_ids=np.array([7, 11]),
        probs=np.array([0.0, 1.0]),
        vocab_size=12,
    )
    assert sample_from_distribution(dist, np.random.default_rng(0)) == 11


def test_softmax_recovers_from_partial_nan_logits():
    """At long context the dense SDPA fallback can emit NaN trunk logits.

    The previous softmax raised ``Cannot normalize logits into a
    probability distribution`` because ``np.max`` propagated the NaN. We
    now sanitize non-finite entries to ``-inf`` and degrade gracefully to
    a softmax over the finite support so the request returns a valid
    token instead of a 500.
    """
    from mtplx.sampling import softmax

    logits = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 2.0])
    probs = softmax(logits, temperature=1.0)
    assert np.isclose(probs.sum(), 1.0)
    # The +inf and NaN positions must contribute zero mass after sanitization.
    assert probs[1] == 0.0
    assert probs[3] == 0.0
    assert probs[4] == 0.0
    # The finite-logit positions must keep their relative ordering.
    assert probs[2] > probs[5] > probs[0]


def test_softmax_full_nan_logits_raises_clear_error():
    """If every logit is non-finite there is no recoverable distribution."""
    import warnings

    from mtplx.sampling import softmax

    logits = np.array([np.nan, np.nan, np.nan])
    # The implementation intentionally subtracts the (-inf) max, which
    # produces an IEEE-754 ``invalid`` warning. Suppress it locally so the
    # test asserts only on the resulting exception.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            softmax(logits, temperature=1.0)
    except ValueError as exc:
        assert "probability distribution" in str(exc)
    else:
        raise AssertionError("softmax must raise on all-non-finite logits")
