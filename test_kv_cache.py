"""KV-cache reuse benchmark for Ollama on Pi 5.

Verifies whether qwen2.5:1.5b's KV cache is reused across consecutive
/api/generate requests when the second prompt is a byte-level prefix
extension of the first. Decision input for speculative LLM prefill.

Usage:
    ~/ollama/bin/python test_kv_cache.py            # ~3-5 min
    ~/ollama/bin/python test_kv_cache.py --long-decay   # +5 min for 300s wait
    ~/ollama/bin/python test_kv_cache.py --skip 1,6     # skip slow scenarios
"""
import argparse
import statistics
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"
KEEP_ALIVE = "30m"

BASE_PERSONA = (
    "You are a helpful voice assistant running locally on a Raspberry Pi. "
    "You have access to the user's real-time physiological signals from a "
    "60GHz mmWave radar (heart rate, breathing rate, presence)."
)
NORMAL_STYLE = (
    "The user appears calm and relaxed. "
    "Keep answers brief and natural, under 2 sentences."
)
READING_RULE = (
    "You may share specific HR/BR values if the user explicitly asks. "
    "Do not bring them up unprompted unless directly relevant."
)


def static_system():
    """Static-only system prompt (no HR/BR injection)."""
    return f"{BASE_PERSONA}\n\n{NORMAL_STYLE}\n\n{READING_RULE}"


def dynamic_system(hr, br):
    """Mirrors prompts.py post-refactor: dynamic readings at the very end."""
    readings = (
        f"Current readings: heart rate {hr:.0f} bpm, "
        f"breathing rate {br:.0f}/min, "
        f"presence=yes, inferred state: normal."
    )
    return f"{BASE_PERSONA}\n\n{NORMAL_STYLE}\n\n{READING_RULE}\n\n{readings}"


def call(prompt, system=None, num_predict=1, keep_alive=KEEP_ALIVE):
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
    r = requests.post(OLLAMA_URL, json=body, timeout=180)
    r.raise_for_status()
    j = r.json()
    return {
        "wallclock_ms": (time.time() - t0) * 1000,
        "prompt_eval_count": j.get("prompt_eval_count", 0),
        "prompt_eval_ms": j.get("prompt_eval_duration", 0) / 1e6,
        "eval_count": j.get("eval_count", 0),
        "eval_ms": j.get("eval_duration", 0) / 1e6,
        "total_ms": j.get("total_duration", 0) / 1e6,
        "load_ms": j.get("load_duration", 0) / 1e6,
        "response": j.get("response", "")[:60],
    }


def unload():
    body = {
        "model": MODEL,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
        "options": {"num_predict": 1},
    }
    try:
        requests.post(OLLAMA_URL, json=body, timeout=60)
    except requests.RequestException:
        pass
    time.sleep(2)


def warmup():
    call("hi", num_predict=1)


def wash():
    """Replace KV cache with garbage so next test prompt sees a cold cache.
    Wash prompt starts with '@' which test prompts never use."""
    call(f"@@@ wash {time.time_ns()} ignore", num_predict=1)


def fmt(r):
    return (f"prefill={r['prompt_eval_ms']:7.1f}ms "
            f"({r['prompt_eval_count']} tok), "
            f"load={r['load_ms']:.0f}ms, "
            f"gen={r['eval_ms']:.0f}ms")


def median(xs):
    return statistics.median(xs)


def scenario_1_cold():
    print("\n=== Scenario 1: Cold baseline (model unloaded each trial) ===")
    vals = []
    for i in range(3):
        unload()
        r = call(f"{static_system()}\n\nUser: What is the capital of France?",
                 num_predict=1)
        print(f"  [{i+1}] {fmt(r)}")
        vals.append(r["prompt_eval_ms"])
    m = median(vals)
    print(f"  median prefill = {m:.1f}ms (load_duration is separate)")
    return m


def scenario_2_warm():
    print("\n=== Scenario 2: Warm baseline (pure prefill, washed cache) ===")
    warmup()
    vals = []
    for i in range(3):
        wash()
        # Unique prompt each trial; starts with '#' so no overlap with wash ('@')
        prompt = (f"#{i} {static_system()}\n\n"
                  f"User: trial-{i}-{time.time_ns()} please answer briefly")
        r = call(prompt, num_predict=1)
        print(f"  [{i+1}] {fmt(r)}")
        vals.append(r["prompt_eval_ms"])
    m = median(vals)
    print(f"  median prefill = {m:.1f}ms  <- BASELINE for comparison")
    return m


def scenario_3_perfect_hit():
    print("\n=== Scenario 3: Perfect prefix extension (prompt-only) ===")
    warmup()
    vals = []
    for i in range(3):
        wash()
        spec_prompt = "What is the capital of"
        real_prompt = "What is the capital of France? Answer in one sentence."
        s = call(spec_prompt, num_predict=1)
        r = call(real_prompt, num_predict=10)
        print(f"  [{i+1}] spec: {fmt(s)}")
        print(f"      real: {fmt(r)}  -> '{r['response']}'")
        vals.append(r["prompt_eval_ms"])
    m = median(vals)
    print(f"  median real prefill = {m:.1f}ms")
    return m


def scenario_4_system_hit():
    print("\n=== Scenario 4: Static system + extending user prompt ===")
    warmup()
    sys = static_system()
    vals = []
    for i in range(3):
        wash()
        s = call("What is the capital of", system=sys, num_predict=1)
        r = call("What is the capital of France? Answer in one sentence.",
                 system=sys, num_predict=10)
        print(f"  [{i+1}] spec: {fmt(s)}")
        print(f"      real: {fmt(r)}  -> '{r['response']}'")
        vals.append(r["prompt_eval_ms"])
    m = median(vals)
    print(f"  median real prefill = {m:.1f}ms")
    return m


def scenario_5_miss():
    print("\n=== Scenario 5: Speculation miss (unrelated real prompt) ===")
    warmup()
    vals = []
    for i in range(3):
        wash()
        s = call("What is the weather in Paris like today?", num_predict=1)
        r = call("Tell me about the history of Mongolia in one sentence.",
                 num_predict=10)
        print(f"  [{i+1}] spec: {fmt(s)}")
        print(f"      real: {fmt(r)}  -> '{r['response']}'")
        vals.append(r["prompt_eval_ms"])
    m = median(vals)
    print(f"  median real prefill = {m:.1f}ms (should ≈ scenario 2)")
    return m


def scenario_6_decay(include_long=False):
    print("\n=== Scenario 6: Cache decay over idle time ===")
    waits = [5, 30, 90]
    if include_long:
        waits.append(300)
    out = {}
    for w in waits:
        warmup()
        wash()
        s = call("What is the capital of", num_predict=1)
        print(f"  spec: {fmt(s)}, idling {w}s...")
        time.sleep(w)
        r = call("What is the capital of France? Answer in one sentence.",
                 num_predict=10)
        print(f"  real after {w}s: {fmt(r)}  -> '{r['response']}'")
        out[w] = r["prompt_eval_ms"]
    return out


def scenario_7_dynamic_system():
    print("\n=== Scenario 7: Dynamic system prompt (HR injection) ===")
    warmup()
    vals = []
    for i in range(3):
        wash()
        sys_a = dynamic_system(72, 16)
        sys_b = dynamic_system(73, 17)  # one bpm change
        user = "What is the capital of France? Answer in one sentence."
        s = call("What is the capital of", system=sys_a, num_predict=1)
        r = call(user, system=sys_b, num_predict=10)
        print(f"  [{i+1}] spec: {fmt(s)}")
        print(f"      real: {fmt(r)}  -> '{r['response']}'")
        vals.append(r["prompt_eval_ms"])
    m = median(vals)
    print(f"  median real prefill = {m:.1f}ms (≈ scenario 2 means cache invalidated)")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--long-decay", action="store_true",
                   help="include 300s wait in scenario 6 (+5 min)")
    p.add_argument("--skip", default="",
                   help="comma-separated scenario numbers to skip, e.g. '1,6'")
    args = p.parse_args()
    skip = set(int(x) for x in args.skip.split(",") if x.strip())

    print(f"Ollama KV-cache test — model={MODEL}, keep_alive={KEEP_ALIVE}")
    print(f"Skipped scenarios: {sorted(skip) if skip else 'none'}")
    print("=" * 60)

    R = {}
    if 1 not in skip: R[1] = scenario_1_cold()
    if 2 not in skip: R[2] = scenario_2_warm()
    if 3 not in skip: R[3] = scenario_3_perfect_hit()
    if 4 not in skip: R[4] = scenario_4_system_hit()
    if 5 not in skip: R[5] = scenario_5_miss()
    if 6 not in skip: R[6] = scenario_6_decay(args.long_decay)
    if 7 not in skip: R[7] = scenario_7_dynamic_system()

    print("\n" + "=" * 60)
    print("Summary (median real prefill_duration in ms)")
    print("=" * 60)
    print("\n| Scenario | Median prefill (ms) | Notes |")
    print("|----------|--------------------:|-------|")
    if 1 in R: print(f"| 1 cold baseline | {R[1]:8.1f} | model load excluded from prefill |")
    if 2 in R: print(f"| 2 warm baseline | {R[2]:8.1f} | **comparison reference** |")
    if 3 in R: print(f"| 3 perfect hit | {R[3]:8.1f} | prompt-only extension |")
    if 4 in R: print(f"| 4 system+user hit | {R[4]:8.1f} | static system + extending user |")
    if 5 in R: print(f"| 5 miss (no prefix) | {R[5]:8.1f} | should ≈ scenario 2 |")
    if 6 in R:
        for w, v in R[6].items():
            print(f"| 6 decay {w}s | {v:8.1f} | cache after {w}s idle |")
    if 7 in R: print(f"| 7 dynamic sys prompt | {R[7]:8.1f} | HR change → cache invalidates |")

    print("\n--- Decision ---")
    if 2 in R and 3 in R:
        savings = (1 - R[3] / R[2]) * 100
        print(f"Scenario 3 vs 2: {R[3]:.1f} vs {R[2]:.1f} ms — saves {savings:+.1f}%")
        verdict = "GO" if savings >= 50 else "NO-GO"
        print(f"  Threshold (≥50% savings): {verdict}")
    if 5 in R and 2 in R:
        miss_ratio = R[5] / R[2]
        print(f"Scenario 5 (miss) / scenario 2: {miss_ratio:.2f}x")
        ok = "OK" if miss_ratio <= 1.1 else "PENALTY"
        print(f"  Threshold (≤1.1x): {ok}")
    if 6 in R and 2 in R:
        for w, v in R[6].items():
            ratio = v / R[2]
            tag = "still cached" if ratio < 0.5 else "cache lost"
            print(f"  Decay {w}s: real/baseline = {ratio:.2f} → {tag}")


if __name__ == "__main__":
    main()
