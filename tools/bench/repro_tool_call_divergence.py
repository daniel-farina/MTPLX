"""Reproduce the tool_call_history_rewrite -> prefix_divergence_at_token bug.

Builds the storage encoding (turn N: messages + assistant(tool_calls) + sentinel
user, truncated before the last im_start) and the lookup encoding (turn N+1:
messages + assistant(tool_calls) + tool_response + new_user) and diffs them.

Storage SHOULD be a strict prefix of lookup. When it isn't, find the first
divergent token and dump enough surrounding bytes to identify what the chat
template re-rendered between turns.

Run: python tools/bench/repro_tool_call_divergence.py
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer  # type: ignore

MODEL_DIR = "/Users/dan/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
IM_START = "<|im_start|>"


def encode(tokenizer, messages, *, tools=None, add_generation_prompt=True):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        tools=tools,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def encode_for_storage(tokenizer, messages, *, tools=None):
    """Mirror mtplx.server.openai._encode_history_for_storage."""
    full = list(messages) + [{"role": "user", "content": "_"}]
    encoded = encode(tokenizer, full, tools=tools, add_generation_prompt=False)
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
    if not isinstance(im_start_id, int) or im_start_id < 0:
        return encoded
    last = -1
    for i in range(len(encoded) - 1, -1, -1):
        if encoded[i] == im_start_id:
            last = i
            break
    if last < 0:
        return encoded
    return encoded[:last]


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def main():
    if not Path(MODEL_DIR).exists():
        print(f"model dir not found: {MODEL_DIR}", file=sys.stderr)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # opencode-style tool definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "task",
                "description": "Dispatch a subagent to investigate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                    },
                    "required": ["description"],
                },
            },
        }
    ]

    system = {"role": "system", "content": "You are a coding assistant."}
    user1 = {"role": "user", "content": "list the game objects"}

    # Assistant emits a tool_call (this is what opencode produces).
    assistant_tc = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "task",
                    # Qwen3 chat template iterates arguments|items so it must
                    # be a mapping, not a JSON string.
                    "arguments": {"description": "List game objects"},
                },
            }
        ],
    }

    tool_resp = {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "Found 56 object classes in src/objects/",
    }
    user2 = {"role": "user", "content": "Now plan a fix."}

    # STORAGE: what postcommit stores at end of turn N.
    storage_msgs = [system, user1, assistant_tc]
    storage_ids = encode_for_storage(tok, storage_msgs, tools=tools)

    # LOOKUP: what the next request looks like when opencode echoes back the
    # assistant + tool_response + new user message.
    lookup_msgs = [system, user1, assistant_tc, tool_resp, user2]
    lookup_ids = encode(
        tok, lookup_msgs, tools=tools, add_generation_prompt=True
    )

    print(f"storage tokens: {len(storage_ids)}")
    print(f"lookup tokens : {len(lookup_ids)}")

    div = first_divergence(storage_ids, lookup_ids)
    if div < 0:
        print("\nstorage IS a strict prefix of lookup -> cache would hit.")
        return

    print(f"\nfirst divergence at token index {div}")
    span = 30
    lo = max(0, div - span)
    hi_s = min(len(storage_ids), div + span)
    hi_l = min(len(lookup_ids), div + span)

    print(f"\n--- storage[{lo}:{hi_s}] ---")
    print(f"ids: {storage_ids[lo:hi_s]}")
    print(f"text: {tok.decode(storage_ids[lo:hi_s])!r}")

    print(f"\n--- lookup[{lo}:{hi_l}] ---")
    print(f"ids: {lookup_ids[lo:hi_l]}")
    print(f"text: {tok.decode(lookup_ids[lo:hi_l])!r}")

    print(f"\n--- divergent token ---")
    print(f"storage[{div}] = {storage_ids[div]} -> {tok.decode([storage_ids[div]])!r}")
    print(f"lookup[{div}]  = {lookup_ids[div]}  -> {tok.decode([lookup_ids[div]])!r}")

    # Dump the full prefix around divergence as plain text for visual diff
    print(f"\n--- storage tail (last 200 chars before+including divergence) ---")
    print(repr(tok.decode(storage_ids[max(0, div - 100) : min(len(storage_ids), div + 30)])))
    print(f"\n--- lookup tail (same window) ---")
    print(repr(tok.decode(lookup_ids[max(0, div - 100) : min(len(lookup_ids), div + 30)])))


if __name__ == "__main__":
    main()
