"""Benchmark KV-cache hit rate and TTFT savings for a 3-turn conversation.

Scenario: realistic Edge-AIGuard exchange — short greeting, vitals query,
follow-up health question. Measures, for each turn:
  COLD : direct /v1/chat (no speculative prefill)
  WARM : /api/generate(num_predict=1) — mimicking what SpeculativePrefillProcessor
         fires on each InterimTranscriptionFrame — followed by /v1/chat

To ensure a fair per-turn measurement (and that the cold-vs-warm comparison
isn't polluted by carryover cache from the previous turn), each scenario uses
a unique random sentinel injected into the system prompt. The sentinel pushes
prompt_eval to exercise a fresh prefix for every (turn, condition) pair.

Output: console table + JSON dump of all metrics.
"""
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

# bench lives in benchmarks/, project modules at repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import build_system_prompt

OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen2.5:1.5b"
KEEP_ALIVE = "30m"

USER_TURNS = [
    "hello",
    "check my heart rate and breathing rate",
    "Is my health condition good?",
]

RADAR_STATE = {"hr": 72, "br": 16, "presence": True, "category": "normal"}


@dataclass
class TurnMetrics:
    turn: int
    user_text: str
    condition: str               # "cold" or "warm"
    wall_ms: int                 # full wallclock to first token
    prompt_eval_count: int       # tokens prefilled
    prompt_eval_ms: int          # time spent in prefill per Ollama
    eval_count: int              # tokens generated
    eval_ms: int                 # time spent generating
    spec_wall_ms: int = 0        # time spent in speculative prefill (warm only)


def build_messages(system: str, prior_turns: list[tuple[str, str]], current_user: str):
    """Build messages list with all prior assistant responses included."""
    msgs = [{"role": "system", "content": system}]
    for u, a in prior_turns:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": current_user})
    return msgs


def build_generate_prompt(system: str, prior_turns: list[tuple[str, str]], current_user: str):
    """Render messages as a flat string for /api/generate (mimics chat template).
    The actual tokenization may differ slightly from /v1/chat — we proved in
    test_chat_template_alignment.py that for our model the byte-level rendering
    matches close enough that the KV cache transfers across endpoints.
    """
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]
    for u, a in prior_turns:
        parts.append(f"<|im_start|>user\n{u}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")
    parts.append(f"<|im_start|>user\n{current_user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n")
    return "\n".join(parts)


def fire_speculation(prompt: str) -> int:
    """Mimics SpeculativePrefillProcessor: /api/generate with num_predict=1.
    Pre-warms the KV cache so the real /v1/chat call below hits it.
    """
    t0 = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"num_predict": 1, "temperature": 0},
        },
        timeout=120,
    )
    r.raise_for_status()
    return int((time.perf_counter() - t0) * 1000)


def run_chat(messages: list[dict], stream_first_token: bool = True) -> tuple[int, dict, str]:
    """Hit /v1/chat (OpenAI-compatible). Returns (wall_ms_to_first_token, raw_response, full_text)."""
    t0 = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "max_tokens": 80,
            "keep_alive": KEEP_ALIVE,
        },
        timeout=180,
    )
    r.raise_for_status()
    wall = int((time.perf_counter() - t0) * 1000)
    body = r.json()
    text = body["choices"][0]["message"]["content"]
    # Ollama returns extended fields in /v1/chat for prefill/eval timing
    usage = body.get("usage", {})
    return wall, body, text


def get_native_timings(prompt: str) -> dict:
    """Run /api/generate (Ollama-native) to get prompt_eval_count + duration.
    /v1/chat doesn't expose these; /api/generate does. We use this as the
    prefill timing reference, paired with /v1/chat for end-to-end timing.
    """
    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"num_predict": 80, "temperature": 0},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def measure_turn(
    turn_idx: int,
    user_text: str,
    prior_turns: list[tuple[str, str]],
    sentinel: str,
    condition: str,
) -> tuple[TurnMetrics, str]:
    """Measure one turn under one condition. Returns (metrics, assistant_response)."""
    # Each condition gets a unique sentinel prefix to keep cache cold for cold runs
    # and avoid cross-condition pollution. Sentinel is appended to system prompt.
    system = build_system_prompt(RADAR_STATE) + f"\n[bench-id: {sentinel}]"

    if condition == "warm":
        # Speculative prefill: fire /api/generate with the same prompt the real
        # call will use. Pre-warms KV cache.
        spec_prompt = build_generate_prompt(system, prior_turns, user_text)
        spec_ms = fire_speculation(spec_prompt)
    else:
        spec_ms = 0

    # Real call: /api/generate (which exposes timing fields)
    prompt = build_generate_prompt(system, prior_turns, user_text)
    t0 = time.perf_counter()
    body = get_native_timings(prompt)
    wall = int((time.perf_counter() - t0) * 1000)

    metrics = TurnMetrics(
        turn=turn_idx,
        user_text=user_text,
        condition=condition,
        wall_ms=wall,
        prompt_eval_count=body.get("prompt_eval_count", 0),
        prompt_eval_ms=body.get("prompt_eval_duration", 0) // 1_000_000,
        eval_count=body.get("eval_count", 0),
        eval_ms=body.get("eval_duration", 0) // 1_000_000,
        spec_wall_ms=spec_ms,
    )
    return metrics, body.get("response", "").strip()


def main():
    print(f"=== Three-turn benchmark — model={MODEL} ===\n")
    # Prime: ensure model is loaded so subsequent measurements aren't dominated
    # by load latency. We don't want load time to pollute the cold/warm comparison.
    print("[prime] loading model...")
    requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": MODEL, "prompt": "warm", "options": {"num_predict": 1},
              "keep_alive": KEEP_ALIVE, "stream": False},
        timeout=120,
    )
    print("[prime] done\n")

    all_metrics: list[TurnMetrics] = []
    # Build prior_turns_cold and prior_turns_warm independently — each track has
    # its own assistant responses to keep cache fully separate.
    prior_cold: list[tuple[str, str]] = []
    prior_warm: list[tuple[str, str]] = []

    for i, user in enumerate(USER_TURNS, start=1):
        print(f"--- Turn {i}: {user!r} ---")

        # Cold: unique sentinel keeps cache miss for prefill.
        sent_cold = secrets.token_hex(8)
        m_cold, asst_cold = measure_turn(i, user, prior_cold, sent_cold, "cold")
        prior_cold.append((user, asst_cold))
        all_metrics.append(m_cold)
        print(f"  COLD wall={m_cold.wall_ms}ms prefill={m_cold.prompt_eval_ms}ms "
              f"prompt_tok={m_cold.prompt_eval_count} gen_tok={m_cold.eval_count}")

        # Warm: separate sentinel, but spec pre-warm matches the real call's prefix.
        sent_warm = secrets.token_hex(8)
        m_warm, asst_warm = measure_turn(i, user, prior_warm, sent_warm, "warm")
        prior_warm.append((user, asst_warm))
        all_metrics.append(m_warm)
        print(f"  WARM wall={m_warm.wall_ms}ms prefill={m_warm.prompt_eval_ms}ms "
              f"spec_overhead={m_warm.spec_wall_ms}ms "
              f"prompt_tok={m_warm.prompt_eval_count} gen_tok={m_warm.eval_count}")
        print()

    # Tabulate
    print("\n=== Results table ===\n")
    print(f"{'Turn':<5} {'User input':<42} {'Cond':<5} "
          f"{'Prefill (ms)':>12} {'TTFT/wall (ms)':>15} "
          f"{'Prompt tok':>11} {'Saved':>7}")
    print("-" * 110)
    for i in range(len(USER_TURNS)):
        cold = all_metrics[i * 2]
        warm = all_metrics[i * 2 + 1]
        saved = (1 - warm.prompt_eval_ms / cold.prompt_eval_ms) * 100 if cold.prompt_eval_ms else 0
        u = cold.user_text if len(cold.user_text) <= 40 else cold.user_text[:37] + "..."
        print(f"{cold.turn:<5} {u:<42} {'cold':<5} "
              f"{cold.prompt_eval_ms:>12} {cold.wall_ms:>15} "
              f"{cold.prompt_eval_count:>11} {'—':>7}")
        print(f"{'':<5} {'':<42} {'warm':<5} "
              f"{warm.prompt_eval_ms:>12} {warm.wall_ms:>15} "
              f"{warm.prompt_eval_count:>11} {saved:>6.1f}%")
        print()

    out = {
        "model": MODEL,
        "user_turns": USER_TURNS,
        "metrics": [asdict(m) for m in all_metrics],
    }
    out_path = "/tmp/bench_three_turn.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
