"""Spec behavior unit test.

Drives SpeculativePrefillProcessor directly (bypassing the pipecat pipeline)
to verify it fires/cancels as designed. Also measures whether the "real" call
afterwards hits the cache loaded by the spec sequence.

Sequence:
  T0  fire spec("what is the capital of")           -- starts loading cache
  T0+200ms fire spec("what is the capital of France?")  -- cancel old, fire new
  wait for completion
  warm /v1/chat with "what is the capital of France? answer briefly" → measure prefill speedup vs cold
"""
import asyncio
import sys
import time
from pathlib import Path

import requests

# Tests live in tests/, project modules at repo root — make them importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speculation import SpeculativePrefillProcessor

# fake state callback that returns a stable dict (no radar dependency for this test)
def fake_get_state():
    return {"hr": 72, "br": 16, "presence": True, "category": "normal", "source": "fake"}


def fake_build_system_prompt(state):
    return (
        "You are a helpful voice assistant running locally on a Raspberry Pi. "
        "You have access to the user's real-time physiological signals from a "
        "60GHz mmWave radar (heart rate, breathing rate, presence).\n\n"
        "The user appears calm and relaxed. Keep answers brief and natural, "
        "under 2 sentences.\n\n"
        "You may share specific HR/BR values if the user explicitly asks "
        "(e.g., 'what is my heart rate?'). Do not bring them up unprompted "
        "unless directly relevant to the user's question.\n\n"
        f"Current readings: heart rate {state['hr']:.0f} bpm, "
        f"breathing rate {state['br']:.0f}/min, "
        f"presence={'yes' if state['presence'] else 'no'}, "
        f"inferred state: {state['category']}."
    )


def call_chat_real(prompt: str, system: str) -> dict:
    """Real call equivalent to what pipecat's OLLamaLLMService does."""
    body = {
        "model": "qwen2.5:1.5b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 1,
        "temperature": 0.0,
        "keep_alive": "30m",
    }
    t0 = time.time()
    r = requests.post("http://localhost:11434/v1/chat/completions",
                      json=body, timeout=180)
    r.raise_for_status()
    j = r.json()
    return {
        "wallclock_ms": (time.time() - t0) * 1000,
        "prompt_tokens": j.get("usage", {}).get("prompt_tokens", 0),
        "response": j.get("choices", [{}])[0].get("message", {})
                     .get("content", "")[:60],
    }


def wash():
    requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:1.5b",
        "prompt": f"@@@ wash {time.time_ns()} ignore",
        "stream": False,
        "options": {"num_predict": 1},
        "keep_alive": "30m",
    }, timeout=60)


async def main():
    print("=== Spec unit test ===\n")

    # Warmup ollama
    requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:1.5b", "prompt": "hi", "stream": False,
        "options": {"num_predict": 1}, "keep_alive": "30m",
    }, timeout=60)

    # Cold baseline first (no spec)
    wash()
    cold = call_chat_real(
        "What is the capital of France? Answer in one sentence.",
        fake_build_system_prompt(fake_get_state()),
    )
    print(f"[cold]   wall={cold['wallclock_ms']:.0f}ms, "
          f"tokens={cold['prompt_tokens']}, '{cold['response']}'")

    # Now fire spec sequence + warm real call
    wash()  # clear baseline cache
    proc = SpeculativePrefillProcessor(
        get_state=fake_get_state,
        build_system_prompt=fake_build_system_prompt,
        model="qwen2.5:1.5b",
    )

    print("\n[spec] firing sequence (each cancels prior)...")
    proc._fire_async("what is")
    await asyncio.sleep(0.2)
    proc._fire_async("what is the cap")
    await asyncio.sleep(0.2)
    proc._fire_async("what is the capital of France?")

    # Wait for last spec to finish
    if proc._in_flight:
        try:
            await proc._in_flight
        except asyncio.CancelledError:
            pass

    # Allow tasks that were 'cancelled' to finish their HTTP work in the
    # background thread (Ollama queues requests and runs them sequentially).
    print("[spec] waiting 2s for any queued backend work to drain...")
    await asyncio.sleep(2.0)

    print(f"[stats] {proc.stats}")

    # Now real call — should hit cache from the spec
    real = call_chat_real(
        "What is the capital of France? Answer in one sentence.",
        fake_build_system_prompt(fake_get_state()),
    )
    print(f"\n[warm]   wall={real['wallclock_ms']:.0f}ms, "
          f"tokens={real['prompt_tokens']}, '{real['response']}'")

    saved_pct = (1 - real["wallclock_ms"] / cold["wallclock_ms"]) * 100
    print(f"\n→ cold {cold['wallclock_ms']:.0f}ms → warm {real['wallclock_ms']:.0f}ms"
          f" = saved {saved_pct:+.1f}%")
    if saved_pct >= 50:
        print(f"✅ PASS: spec → real hit cache (≥50% saved)")
    else:
        print(f"⚠️  warning: speedup below 50%. Check stats and Ollama logs.")

    # --- Debounce gate test ---
    print("\n=== Debounce gate test ===")
    proc2 = SpeculativePrefillProcessor(
        get_state=fake_get_state,
        build_system_prompt=fake_build_system_prompt,
        model="qwen2.5:1.5b",
    )
    # Rapid-fire small extensions: only the first should fire.
    proc2._fire_async("what is")          # first → fires (elapsed huge)
    proc2._fire_async("what is t")        # growth 2, elapsed ~0ms → skip
    proc2._fire_async("what is th")       # growth 3 vs last_fired, elapsed ~0ms → skip
    proc2._fire_async("what is tha")      # growth 4 vs last_fired, elapsed ~0ms → skip
    print(f"  rapid-fire stats: {proc2.stats}")
    if proc2.stats["fired"] == 1 and proc2.stats["skipped"] == 3:
        print("  ✅ debounce gate working: 1 fired, 3 skipped")
    else:
        print(f"  ⚠️  unexpected stats: expected fired=1 skipped=3")
    # Cancel the in-flight task so it doesn't linger
    if proc2._in_flight and not proc2._in_flight.done():
        proc2._in_flight.cancel()
        try:
            await proc2._in_flight
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
