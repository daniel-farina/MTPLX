"""40-pass benchmark, but context truly grows from ~512 to ~40K.
Each pass injects a synthetic tool_response of ~1K tokens between the user
ask and the next ask. Mimics opencode's pattern of huge tool results.
"""
import json, time, urllib.request, sys, random
from datetime import datetime

URL = "http://127.0.0.1:8088/v1/chat/completions"
SID = f"game-bench-long-{int(time.time())}"
TSV = "/tmp/mtplx-game-bench-long-results.tsv"
MD = "/tmp/mtplx-game-bench-long-findings.md"

random.seed(42)
WORDS = ["function", "class", "import", "export", "const", "if", "else", "return",
         "static", "public", "private", "void", "string", "number", "array",
         "object", "true", "false", "while", "for", "try", "catch", "throw",
         "new", "this", "self", "await", "async", "let", "var", "interface",
         "extends", "implements", "type", "enum", "namespace", "module"]

ASKS = [
    ("Airplane", "add a propeller-blur disc that fades in with airspeed"),
    ("AIFollower", "reduce cylinder/sphere segment counts by 30%"),
    ("Airport", "add a runway-light intensity toggle in settings"),
    ("Balloons", "make balloons sway with a sin(time) offset"),
    ("Birds", "lower bird poly count and instance them"),
    ("Bridges", "add a parametric span count for variation"),
    ("Building", "add window emissive at night based on time-of-day"),
    ("Car", "add a brake-light material toggle"),
    ("Castle", "add subtle banner sway on towers"),
    ("Checkpoints", "add a glowing ring for the next active checkpoint"),
    ("City", "instance distant building meshes for perf"),
    ("Clouds", "drift clouds horizontally at constant speed"),
    ("Coins", "make coins rotate at fixed rate"),
    ("ControlTower", "add a rotating radar dish on top"),
    ("Drone", "add propeller wash particles"),
    ("ExplosionEffect", "reduce particle count to 50% of current"),
    ("Fence", "instance fence sections along a path"),
    ("Fireworks", "lower particle counts by 30%"),
    ("Helicopter", "add tail-rotor spin"),
    ("Highway", "add lane markings as instanced quads"),
    ("Hills", "lower hill mesh segment density"),
    ("HotAirBalloon", "make balloon bob slightly while ascending"),
    ("House", "vary house roof colors deterministically"),
    ("Lighthouse", "add a rotating beam"),
    ("Mountains", "instance distant mountain LODs"),
    ("Pine", "lower pine-tree segment count"),
    ("PowerLines", "add subtle wire sag using catenary"),
    ("Rain", "add a settings toggle to disable rain"),
    ("Refinery", "add steam particles from stacks"),
    ("River", "make river surface scroll with a UV offset"),
    ("Road", "add a wear texture variation"),
    ("RoadCones", "instance cones for a row layout"),
    ("Rocks", "lower rock mesh complexity"),
    ("Runway", "add edge lights with intensity setting"),
    ("Sky", "add a configurable haze density"),
    ("Stadium", "add a crowd-noise audio toggle"),
    ("Streetlights", "add flicker animation when on"),
    ("Tank", "add turret rotation interpolation"),
    ("Trains", "add wheel-spin animation"),
    ("Trees", "instance trees in clusters with random rotation"),
]

# Pre-generate per-pass synthetic tool results (~1000 tokens each).
SYNTH_TOOL_RESULTS = [" ".join(random.choices(WORDS, k=900)) for _ in range(40)]

def req(messages, max_tokens=200):
    body = json.dumps({
        "model": "mtplx-qwen36-27b-optimized-speed",
        "enable_thinking": False,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    headers = {"Content-Type": "application/json", "x-mtplx-session-id": SID}
    r = urllib.request.Request(URL, data=body, headers=headers)
    chunks = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=400) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore")
                if line.startswith("data: "):
                    d = line[6:].strip()
                    if d == "[DONE]" or not d: continue
                    try: chunks.append(json.loads(d))
                    except: pass
    except Exception as e:
        return {"error": str(e)[:200], "elapsed": time.time() - t0}
    elapsed = time.time() - t0
    final = next((c for c in reversed(chunks) if "usage" in c), None)
    if not final:
        return {"error": "no usage", "elapsed": elapsed}
    ms = final.get("mtplx_stats", {})
    u = final.get("usage", {})
    contents = []
    for c in chunks:
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"): contents.append(d["content"])
    return {
        "ctx": u.get("prompt_tokens", 0),
        "out": u.get("completion_tokens", 0),
        "ttft_s": ms.get("ttft_s", 0) or 0,
        "decode_tok_s": ms.get("decode_tok_s", 0) or 0,
        "elapsed_total": elapsed,
        "peak_mem_mb": (ms.get("peak_memory_bytes", 0) or 0) / 1024 / 1024,
        "session_cache_hit": ms.get("session_cache_hit", False),
        "cache_miss_reason": ms.get("cache_miss_reason"),
        "content": "".join(contents),
    }

sys_prompt = ("You are a Three.js senior engineer reviewing the 3dworld flight game. "
              "When asked to improve an object, suggest a specific code change in 2-3 sentences.")

messages = [{"role": "system", "content": sys_prompt}]

with open(TSV, "w") as f:
    f.write("pass\tobject\tctx\tttft_s\tdecode_tok_s\tout\tpeak_mem_mb\thit\tmiss\tresponse_chars\telapsed_s\n")

results = []
print(f"{'#':>3} {'object':<14} {'ctx':>7} {'ttft':>7} {'decode':>10} {'out':>4} {'mem_MB':>7} {'hit':>5}")
sys.stdout.flush()

for i, (obj, feature) in enumerate(ASKS, start=1):
    user_prompt = f"Improve **{obj}**: {feature}. Reply briefly."
    messages.append({"role": "user", "content": user_prompt})
    r = req(messages, max_tokens=150)

    if "error" in r:
        print(f"{i:>3} {obj:<14}  ERROR: {r['error']}  elapsed={r.get('elapsed',0):.1f}s")
        with open(TSV, "a") as f:
            f.write(f"{i}\t{obj}\tERROR\t\t\t\t\t\t{r['error']}\t\t{r.get('elapsed',0):.1f}\n")
        results.append({"pass": i, "object": obj, "error": r["error"]})
        messages.append({"role": "assistant", "content": "(error)"})
        sys.stdout.flush()
        continue

    print(f"{i:>3} {obj:<14} {r['ctx']:>7} {r['ttft_s']:>5.1f}s {r['decode_tok_s']:>7.1f}t/s {r['out']:>4} {r['peak_mem_mb']:>6.0f} {str(r['session_cache_hit']):>5}")
    sys.stdout.flush()

    with open(TSV, "a") as f:
        f.write(f"{i}\t{obj}\t{r['ctx']}\t{r['ttft_s']:.2f}\t{r['decode_tok_s']:.1f}\t{r['out']}\t{r['peak_mem_mb']:.0f}\t{r['session_cache_hit']}\t{r['cache_miss_reason'] or ''}\t{len(r['content'])}\t{r['elapsed_total']:.1f}\n")
    results.append({
        "pass": i, "object": obj,
        "ctx": r["ctx"], "ttft_s": r["ttft_s"], "decode_tok_s": r["decode_tok_s"],
        "out": r["out"], "peak_mem_mb": r["peak_mem_mb"],
        "hit": r["session_cache_hit"], "miss": r["cache_miss_reason"],
        "elapsed_s": r["elapsed_total"],
    })

    # Append assistant + a synthetic tool-result-shaped user message to grow context
    messages.append({"role": "assistant", "content": r["content"]})
    messages.append({"role": "user", "content": f"Reference dump for {obj}:\n{SYNTH_TOOL_RESULTS[i-1]}\nNoted, continue with the next object."})
    # Simulate model accepting the dump; append assistant ack
    messages.append({"role": "assistant", "content": "Noted."})

# Findings
ok = [r for r in results if "error" not in r]

def curve(rows, key):
    vals = [r[key] for r in rows]
    return (min(vals), max(vals), sum(vals)/len(vals)) if vals else (0,0,0)

ttft_min, ttft_max, ttft_avg = curve(ok, "ttft_s")
dec_min, dec_max, dec_avg = curve(ok, "decode_tok_s")
mem_min, mem_max, mem_avg = curve(ok, "peak_mem_mb")
hit_count = sum(1 for r in ok if r["hit"])

# Find cliffs
cliffs = []
for i in range(1, len(ok)):
    p, c = ok[i-1], ok[i]
    if p["decode_tok_s"] > 0 and c["decode_tok_s"] / p["decode_tok_s"] < 0.5:
        cliffs.append((c["pass"], c["object"], p["decode_tok_s"], c["decode_tok_s"], c["ctx"]))

with open(MD, "w") as f:
    f.write(f"""# 40-Pass Game Bench (TRUE 40K context growth)

Run: {datetime.now().isoformat()}
Session ID: {SID}
Branch: perf/memory-caps-and-adaptive-depth

## Summary

- Passes attempted: {len(ASKS)}
- Successful: {len(ok)}
- Errors: {len(results) - len(ok)}
- Cache hits: {hit_count}/{len(ok)}
- Cache misses: {len(ok) - hit_count}/{len(ok)}

## Context growth

- Pass 1 ctx: {ok[0]['ctx'] if ok else 'n/a'}
- Pass 40 ctx: {ok[-1]['ctx'] if ok else 'n/a'}
- Mean ctx: {sum(r['ctx'] for r in ok) // max(1,len(ok))}

## TTFT

- min: {ttft_min:.2f}s | max: {ttft_max:.2f}s | mean: {ttft_avg:.2f}s

## Decode tok/s

- min: {dec_min:.1f} | max: {dec_max:.1f} | mean: {dec_avg:.1f}

## Peak memory MB

- min: {mem_min:.0f} | max: {mem_max:.0f} | mean: {mem_avg:.0f}

## Decode cliffs (50%+ consecutive drops)

""")
    if cliffs:
        for p, o, prev, cur, ctx in cliffs:
            f.write(f"- Pass {p} ({o}): decode {prev:.1f} -> {cur:.1f} t/s at ctx={ctx}\n")
    else:
        f.write("- None detected\n")
    f.write("\n## Per-pass detail\n\n| # | object | ctx | ttft_s | decode | out | mem_MB | hit |\n|---|---|---|---|---|---|---|---|\n")
    for r in ok:
        f.write(f"| {r['pass']} | {r['object']} | {r['ctx']} | {r['ttft_s']:.2f} | {r['decode_tok_s']:.1f} | {r['out']} | {r['peak_mem_mb']:.0f} | {r['hit']} |\n")
    if any("error" in r for r in results):
        f.write("\n## Errors\n\n")
        for r in results:
            if "error" in r:
                f.write(f"- Pass {r['pass']} ({r['object']}): {r['error']}\n")

print()
print(f"=== findings: {MD} ===")
print(f"=== TSV: {TSV} ===")
