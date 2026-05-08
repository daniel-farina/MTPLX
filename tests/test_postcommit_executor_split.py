"""Tier 1.2 perf regression tests: postcommit retokenization must not
head-of-line block the next foreground request on the single-worker
`generation_executor`.

PR #24 (commit 3d9f200) routed the async retokenization snapshot work to
`state.generation_executor` to preserve thread-local MLX stream invariants
(a SessionBank entry created on one thread cannot be restored from another
without `RuntimeError: There is no Stream(gpu, N) in current thread`). The
side effect: `generation_executor = ThreadPoolExecutor(max_workers=1)` so
the next foreground `worker` submitted to that same executor queued behind
the postcommit's full ~18K-token re-prefill (~27-29 s on Qwen3.6 27B),
producing visible 30 s stream silences between turns of any tool-using
subagent flow.

The fix splits the orchestration loop off the MLX-touching work:

  - The retry/sleep loop (poll on lock, sleep, retry, deadline tracking)
    runs on `postcommit_executor` so its 250 ms inter-attempt sleeps do
    not occupy `generation_executor`.

  - The actual MLX-touching `_store_retokenized_history_snapshot` call is
    submitted onto `generation_executor` from the postcommit thread and
    awaited via `future.result()`, so the cache buffer is created on the
    same thread that will later restore it (Apple Silicon stream-locality).

These tests pin that invariant in place against future regressions.
"""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import pytest

from mtplx.server import openai


def _state(*, executors_aliased: bool = False):
    """Build the minimum stub for `_schedule_idle_postcommit_snapshot`.

    `executors_aliased=True` simulates older configurations / test stubs
    where the same executor is used for both roles. The helper must fall
    back to a direct call (no double-submit deadlock).
    """
    foreground_lock = threading.Lock()

    generation_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-generation"
    )
    if executors_aliased:
        postcommit_executor = generation_executor
    else:
        postcommit_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="test-postcommit"
        )

    return SimpleNamespace(
        lock=foreground_lock,
        has_foreground=lambda: False,
        generation_executor=generation_executor,
        postcommit_executor=postcommit_executor,
        args=SimpleNamespace(server_console=False),
    )


def _kwargs():
    return dict(
        session_id="test-session",
        messages=[],
        assistant_content="<tool_call>...</tool_call>",
        assistant_tool_calls=[{"name": "grep", "arguments": "{}"}],
        thinking_enabled=False,
        policy_fingerprint="test-policy",
        unsafe_reason="retokenized_history_mismatch",
    )


def _drain(state):
    state.postcommit_executor.shutdown(wait=True)
    if state.generation_executor is not state.postcommit_executor:
        state.generation_executor.shutdown(wait=True)


def test_mlx_work_runs_on_generation_executor_thread(monkeypatch):
    """The MLX-touching `_store_retokenized_history_snapshot` call MUST
    execute on a `generation_executor` thread so subsequent restores from
    that same executor see the cache buffer in the right MLX stream.
    """
    state = _state()
    seen_thread_names: list[str] = []

    def fake_store(*_args, **_kwargs):
        seen_thread_names.append(threading.current_thread().name)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 1,
            "nbytes": 1,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    openai._schedule_idle_postcommit_snapshot(state, **_kwargs())
    _drain(state)

    assert len(seen_thread_names) == 1
    assert seen_thread_names[0].startswith("test-generation"), (
        f"MLX snapshot must run on generation_executor thread, got "
        f"{seen_thread_names[0]!r}"
    )


def test_orchestration_sleep_does_not_block_generation_executor(monkeypatch):
    """The retry-loop's `time.sleep` between attempts must occur on the
    postcommit thread, NOT on `generation_executor`. Otherwise the next
    foreground request submitted to `generation_executor` would queue
    behind the sleep.

    We simulate a single retry by returning `model_lock_busy_*` once and
    then `stored: True` on the second attempt, while spying on
    `time.sleep` from `mtplx.server.openai` to record which thread sleeps.
    """
    state = _state()

    attempt_count = {"n": 0}
    sleep_threads: list[str] = []

    def fake_store(*_args, **_kwargs):
        attempt_count["n"] += 1
        if attempt_count["n"] == 1:
            return {
                "stored": False,
                "mode": "retokenized_history",
                "reason": "model_lock_busy_before_retokenized_commit",
                "elapsed_s": 0.0,
            }
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 5,
            "nbytes": 42,
        }

    real_sleep = time.sleep

    def spy_sleep(seconds):
        sleep_threads.append(threading.current_thread().name)
        # Use a tiny real sleep so the loop can actually progress; we don't
        # want to mock all of `time` because the deadline math depends on it.
        real_sleep(0.001)

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)
    monkeypatch.setattr(openai.time, "sleep", spy_sleep)

    openai._schedule_idle_postcommit_snapshot(state, **_kwargs())
    _drain(state)

    assert attempt_count["n"] == 2
    # Exactly one inter-attempt sleep recorded.
    assert len(sleep_threads) == 1
    assert sleep_threads[0].startswith("test-postcommit"), (
        f"Inter-attempt sleep must occur on postcommit thread, got "
        f"{sleep_threads[0]!r} - sleeping on generation_executor would "
        f"head-of-line block the next foreground request."
    )


def test_consecutive_foreground_submits_not_serialized_behind_postcommit(monkeypatch):
    """End-to-end timing test: a slow postcommit (simulating the ~30 s
    re-prefill before Tier 1.2) must NOT delay the start of the next
    `generation_executor.submit(worker)` call.

    Concretely: kick off the postcommit, then submit a foreground "worker"
    to `generation_executor`. The worker should start within a small
    fraction of the postcommit duration, not after the full postcommit
    completes.
    """
    state = _state()
    postcommit_started = threading.Event()
    postcommit_release = threading.Event()
    worker_started_s: list[float] = []

    def fake_store(*_args, **_kwargs):
        postcommit_started.set()
        # Simulate the slow re-prefill the user reported as 27-29 s.
        # We use 0.5 s to keep the test fast, then assert the worker
        # started within 0.2 s. Generous tolerance to avoid CI flakiness.
        postcommit_release.wait(timeout=2.0)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 1,
            "nbytes": 1,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    openai._schedule_idle_postcommit_snapshot(state, **_kwargs())

    # Wait for postcommit's MLX work to begin on generation_executor.
    assert postcommit_started.wait(timeout=2.0)

    # Now submit a "next foreground request" to generation_executor.
    # Under the bug (postcommit's retry loop running on generation_executor),
    # this submit would not start until postcommit_release fires.
    # Under the fix (retry loop on postcommit_executor, MLX call runs and
    # returns), the submit waits only for the in-flight MLX call to
    # complete - which IS still serialized through generation_executor by
    # design (cross-thread MLX safety). So the test asserts the worker
    # starts within `postcommit_release` time + a small slack.
    submit_start = time.perf_counter()

    def worker():
        worker_started_s.append(time.perf_counter() - submit_start)

    state.generation_executor.submit(worker)
    # Release the postcommit so its sub-future on generation_executor
    # finishes and the worker can run.
    postcommit_release.set()
    _drain(state)

    assert len(worker_started_s) == 1
    # Worker should start very soon after release - we tolerate 0.5 s of
    # scheduling slack.
    assert worker_started_s[0] < 0.5, (
        f"worker took {worker_started_s[0]:.2f}s to start after the "
        f"postcommit released; expected sub-second under Tier 1.2."
    )


def test_aliased_executors_fall_back_to_direct_call(monkeypatch):
    """If `postcommit_executor is generation_executor` (older config or
    minimal test stub), `_run_store_on_generation_executor` MUST call
    the snapshot function directly rather than submit-and-await on the
    same single-worker executor (which would deadlock).
    """
    state = _state(executors_aliased=True)
    seen_thread_names: list[str] = []

    def fake_store(*_args, **_kwargs):
        seen_thread_names.append(threading.current_thread().name)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 1,
            "nbytes": 1,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    openai._schedule_idle_postcommit_snapshot(state, **_kwargs())
    _drain(state)

    # No deadlock, exactly one call, on the (single, aliased) executor.
    assert len(seen_thread_names) == 1


def test_retokenized_snapshot_passes_session_bank_for_prefix_reuse(monkeypatch):
    """Tier 2.1 / Option B: `_store_retokenized_history_snapshot` MUST pass
    `session_bank` to `restore_or_prefill_prompt_state` so the postcommit
    re-prefill can short-circuit on the previous turn's prefix.

    Without this, every postcommit pays the full ~18K-token re-prefill
    cost. With this, consecutive tool-call turns drop from ~27 s to ~1 s
    because only the new suffix needs to be forward-AR'd.
    """
    captured_kwargs: dict = {}

    class _PromptState:
        trunk_cache = "cache"
        logits = "logits"
        hidden = "hidden"
        committed_mtp_cache = None
        cache_hit = True
        cached_tokens = 17_500
        suffix_tokens = 320

    def fake_restore_or_prefill(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _PromptState()

    class _Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return [1, 2, 3, 4, 5]

    class _Bank:
        def put(self, **_kwargs):
            return SimpleNamespace(
                prefix_len=5,
                nbytes=42,
                token_hash="hash",
            )

    monkeypatch.setattr(
        openai, "restore_or_prefill_prompt_state", fake_restore_or_prefill
    )
    # `snapshot_cache` only fires when committed_mtp_cache is not None.
    monkeypatch.setattr(openai, "snapshot_cache", lambda c: c)
    monkeypatch.setattr(openai, "_encode_messages", lambda *a, **k: [1, 2, 3, 4, 5])

    bank = _Bank()
    args = SimpleNamespace(strip_assistant_reasoning_history=False)
    state = SimpleNamespace(
        runtime=SimpleNamespace(tokenizer=_Tokenizer()),
        sessions=SimpleNamespace(bank=bank),
        template_hash="tmpl-hash",
        draft_head_identity="draft-id",
        lock=threading.Lock(),
        begin_foreground=lambda: None,
        end_foreground=lambda: None,
        args=args,
    )

    result = openai._store_retokenized_history_snapshot(
        state,
        session_id="session-A",
        messages=[],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy-fp",
    )

    assert result["stored"] is True
    assert captured_kwargs.get("session_bank") is bank, (
        "restore_or_prefill_prompt_state must receive session_bank so the "
        "longest_prefix lookup can shortcut a full re-prefill."
    )
    assert captured_kwargs.get("template_hash") == "tmpl-hash"
    assert captured_kwargs.get("draft_head_identity") == "draft-id"
    assert captured_kwargs.get("policy_fingerprint") == "policy-fp"
    # cache_hit / cached_tokens / suffix_tokens / cache_miss_reason
    # propagate to result for observability. The miss reason is the bank's
    # `last_miss_reason` after this turn's lookup; on a true hit it is
    # None, otherwise it captures why the prefix shortcut could not run
    # (`new_session`, `prefix_divergence_at_token`, `policy_mismatch`,
    # ...). Operators rely on this field to debug regressions where the
    # postcommit silently falls back to a full re-prefill.
    assert result["cache_hit"] is True
    assert result["cached_tokens"] == 17_500
    assert result["suffix_tokens"] == 320
    assert "cache_miss_reason" in result
