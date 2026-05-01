"""Verify Ollama chat-template alignment between /api/generate and /v1/chat/completions.

Speculative LLM prefill plan: spec calls fire on streaming ASR partials, real
calls go through pipecat's OllamaLLMService (which uses /v1/chat/completions).
For the speculation to actually save prefill time, both endpoints must render
the same byte string for equivalent inputs — otherwise the second call's KV
cache prefix won't match the first's.

This script gates the speculation architecture:

  Test 1  Render equivalence: same logical (system + user) → same prompt_eval_count?
  Test 2  Cross-endpoint cache hit (the critical test):
            spec via /api/generate (short prefix) → real via /v1/chat (extension)
            does real's prefill drop vs cold?
  Test 3  Same-endpoint /v1/chat baseline: prefix extension within OpenAI-compat
            does it work at all?
"""
import argparse
import json
import statistics
import time

import requests

OLLAMA = "http://localhost:11434"
GEN_URL = f"{OLLAMA}/api/generate"
CHAT_URL = f"{OLLAMA}/v1/chat/completions"
MODEL = "qwen2.5:1.5b"
KEEP_ALIVE = "30m"

SYSTEM = (
    "You are a helpful voice assistant running locally on a Raspberry Pi. "
    "Keep answers brief and natural, under 2 sentences. "
    "You may share specific HR/BR values if the user explicitly asks."
)


def call_generate(prompt, system=None, num_predict=1, keep_alive=KEEP_ALIVE):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }
    if system is not None:
        body["system"] = system
    t0 = time.time()
    r = requests.post(GEN_URL, json=body, timeout=180)
    r.raise_for_status()
    j = r.json()
    return {
        "endpoint": "/api/generate",
        "wallclock_ms": (time.time() - t0) * 1000,
        "prompt_eval_count": j.get("prompt_eval_count", 0),
        "prompt_eval_ms": j.get("prompt_eval_duration", 0) / 1e6,
        "eval_count": j.get("eval_count", 0),
        "eval_ms": j.get("eval_duration", 0) / 1e6,
        "response": j.get("response", "")[:60],
    }


def call_chat(messages, max_tokens=1, keep_alive=KEEP_ALIVE):
    """POST /v1/chat/completions and parse Ollama's OpenAI-compat response.

    Ollama tucks native metrics under non-standard keys (prompt_eval_count etc.)
    so we capture both standard usage.* and any Ollama-only fields it exposes.
    """
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # keep_alive is Ollama extension; OpenAI-compat may or may not honor it.
        # Pass it via a top-level field so Ollama sees it; OpenAI clients ignore
        # unknown top-level fields.
        "keep_alive": keep_alive,
    }
    t0 = time.time()
    r = requests.post(CHAT_URL, json=body, timeout=180)
    r.raise_for_status()
    j = r.json()
    usage = j.get("usage", {})
    return {
        "endpoint": "/v1/chat",
        "wallclock_ms": (time.time() - t0) * 1000,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        # OpenAI compat usually does NOT expose prompt_eval_duration; fall back
        # to wallclock approximation. We separately capture real timing in Test
        # 2 by minimizing the eval portion (max_tokens=1).
        "prompt_eval_ms": j.get("prompt_eval_duration", 0) / 1e6,
        "eval_count": usage.get("completion_tokens", 0),
        "eval_ms": j.get("eval_duration", 0) / 1e6,
        "response": (
            j.get("choices", [{}])[0].get("message", {}).get("content", "")[:60]
        ),
        "raw_json_keys": sorted(j.keys()),
    }


def unload():
    requests.post(GEN_URL, json={
        "model": MODEL, "prompt": "", "stream": False, "keep_alive": 0,
        "options": {"num_predict": 1},
    }, timeout=60)
    time.sleep(2)


def warmup():
    call_generate("hi", num_predict=1)


def wash():
    """Clear KV cache by feeding a totally unrelated prompt."""
    call_generate(f"@@@ wash {time.time_ns()} ignore", num_predict=1)


def fmt(r):
    return (f"prefill={r['prompt_eval_ms']:7.1f}ms "
            f"({r['prompt_eval_count']} tok), "
            f"wall={r['wallclock_ms']:.0f}ms, "
            f"resp='{r['response']}'")


def median(xs):
    return statistics.median(xs)


def test_1_render_equivalence():
    print("\n" + "=" * 70)
    print("Test 1: Render equivalence — same input via both endpoints")
    print("=" * 70)
    print("Compare prompt_eval_count for identical (system + user) input.")
    print("If counts are equal, both endpoints render the same template.\n")

    user = "What is the capital of France?"
    warmup()

    # Run 3 times each, alternating, to mitigate cache effects
    gen_counts = []
    chat_counts = []
    for i in range(3):
        wash()
        g = call_generate(user, system=SYSTEM, num_predict=1)
        wash()
        c = call_chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user}],
            max_tokens=1,
        )
        gen_counts.append(g["prompt_eval_count"])
        chat_counts.append(c["prompt_eval_count"])
        print(f"  [{i+1}] /api/generate: {fmt(g)}")
        print(f"      /v1/chat:      {fmt(c)}")

    if i == 0:  # only print once
        print(f"\n  /v1/chat response keys: {c.get('raw_json_keys', [])}")

    g_med = median(gen_counts)
    c_med = median(chat_counts)
    print(f"\n  Median prompt_eval_count: /api/generate={g_med}, /v1/chat={c_med}")
    if g_med == c_med:
        print(f"  ✅ PASS: counts identical → same template render")
        return "pass", g_med, c_med
    else:
        diff = abs(g_med - c_med)
        print(f"  ⚠️  Differ by {diff} tokens — render is NOT byte-identical")
        return "fail", g_med, c_med


def test_2_cross_endpoint_cache():
    print("\n" + "=" * 70)
    print("Test 2: Cross-endpoint cache hit (CRITICAL)")
    print("=" * 70)
    print("spec via /api/generate, real via /v1/chat. If real's wallclock drops")
    print("dramatically (vs cold), KV cache transferred across endpoints.\n")

    warmup()

    # First, get a baseline: cold /v1/chat call (no spec)
    print("Step A: cold /v1/chat baseline (no spec)")
    wash()
    cold = call_chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "What is the capital of France?"}],
        max_tokens=1,
    )
    print(f"  cold: {fmt(cold)}")
    cold_wall = cold["wallclock_ms"]

    # Now: speculation via /api/generate, then real via /v1/chat
    print("\nStep B: spec via /api/generate, real via /v1/chat (3 runs)")
    real_walls = []
    for i in range(3):
        wash()
        # Spec: prompt is just the prefix portion of what real will send
        spec = call_generate(
            "What is the capital of",  # prefix
            system=SYSTEM,
            num_predict=1,
        )
        # Real: full chat call
        real = call_chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "What is the capital of France?"}],
            max_tokens=1,
        )
        real_walls.append(real["wallclock_ms"])
        print(f"  [{i+1}] spec: {fmt(spec)}")
        print(f"       real: {fmt(real)}")

    real_med = median(real_walls)
    speedup = cold_wall / real_med if real_med > 0 else 0
    saved_pct = (1 - real_med / cold_wall) * 100 if cold_wall > 0 else 0
    print(f"\n  cold /v1/chat: {cold_wall:.0f}ms")
    print(f"  real /v1/chat after spec (median): {real_med:.0f}ms")
    print(f"  speedup: {speedup:.2f}× / saved {saved_pct:+.1f}%")

    if saved_pct >= 50:
        verdict = "pass"
        print(f"  ✅ PASS: cross-endpoint speculation works (≥50% saved)")
    else:
        verdict = "fail"
        print(f"  ❌ FAIL: cross-endpoint cache transfer did NOT happen")
    return verdict, cold_wall, real_med


def test_3_chat_only_baseline():
    print("\n" + "=" * 70)
    print("Test 3: /v1/chat only — does prefix extension hit cache within")
    print("        OpenAI-compat at all?")
    print("=" * 70)

    warmup()

    # Cold /v1/chat first (already have a baseline)
    print("Step A: cold /v1/chat (with unique prompt to avoid leftover cache)")
    wash()
    cold = call_chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"explain photosynthesis briefly {time.time_ns()}"}],
        max_tokens=1,
    )
    print(f"  cold: {fmt(cold)}")
    cold_wall = cold["wallclock_ms"]

    # Now: spec + real both via /v1/chat
    print("\nStep B: spec via /v1/chat, real via /v1/chat (3 runs)")
    real_walls = []
    for i in range(3):
        wash()
        spec = call_chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "What is the capital of"}],
            max_tokens=1,
        )
        real = call_chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "What is the capital of France?"}],
            max_tokens=1,
        )
        real_walls.append(real["wallclock_ms"])
        print(f"  [{i+1}] spec: {fmt(spec)}")
        print(f"       real: {fmt(real)}")

    real_med = median(real_walls)
    saved_pct = (1 - real_med / cold_wall) * 100 if cold_wall > 0 else 0
    print(f"\n  cold /v1/chat: {cold_wall:.0f}ms")
    print(f"  real /v1/chat after spec (median): {real_med:.0f}ms")
    print(f"  saved {saved_pct:+.1f}%")

    verdict = "pass" if saved_pct >= 50 else "fail"
    if verdict == "pass":
        print(f"  ✅ PASS: speculation works within /v1/chat alone")
    else:
        print(f"  ❌ FAIL: even within /v1/chat, prefix extension does not hit cache")
    return verdict, cold_wall, real_med


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip", default="",
                   help="comma-separated tests to skip, e.g. '1,3'")
    args = p.parse_args()
    skip = set(int(x) for x in args.skip.split(",") if x.strip())

    print(f"Ollama chat-template alignment test — model={MODEL}")
    print(f"Skipped: {sorted(skip) if skip else 'none'}")

    results = {}
    if 1 not in skip:
        results[1] = test_1_render_equivalence()
    if 2 not in skip:
        results[2] = test_2_cross_endpoint_cache()
    if 3 not in skip:
        results[3] = test_3_chat_only_baseline()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for n, (verdict, *vals) in results.items():
        marker = "✅" if verdict == "pass" else "❌"
        print(f"  Test {n}: {marker} {verdict.upper()} — {vals}")

    print("\n--- Decision matrix ---")
    t1 = results.get(1, ("skip",))[0]
    t2 = results.get(2, ("skip",))[0]
    t3 = results.get(3, ("skip",))[0]
    if t2 == "pass":
        print("  → spec via /api/generate works (preferred path)")
        print("    Use /api/generate for spec, keep simple num_predict/keep_alive control.")
    elif t3 == "pass":
        print("  → spec must use /v1/chat/completions to match pipecat's OllamaLLMService")
        print("    Architectural choice: send chat-format spec calls.")
    else:
        print("  → ❌ speculation does NOT work across endpoints. Need to investigate")
        print("    Ollama version, model template, or call patterns.")


if __name__ == "__main__":
    main()
