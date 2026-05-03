"""bench_streaming_tts.py — A/B/C TTFA on the LLM→TTS path.

Isolates the contribution of full-duplex streaming + phrase-level
TTS aggregation. Three text-flushing strategies on the same fixed
query and system prompt:

  A. Whole-response   stream=False, run piper on full reply
                      (naive sync; everything blocks on full LLM)
  B. Sentence-flush   stream=True, flush on . ! ?
                      (pipeline.py Day 1 streaming behavior)
  C. Phrase-flush     stream=True, flush on . ! ? OR , : ; (≥6 chars)
                      (pipecat PhraseAwareTextAggregator, our shipped path)

Why this is the right A/B:
  The shipped pipeline's TTFA = (time until enough text is in the
  buffer to start TTS) + (piper processing first audio byte). The
  first term is the only thing the flushing policy controls; the
  second is fixed by the model. So we measure the first directly
  (`flush_ms`) and report the sum (`ttfa_ms`) for the system-level
  number a user would feel.

Cache control: each trial appends a unique 16-hex sentinel to the
system prompt to defeat Ollama's KV cache, so all three modes start
from the same cold prefix and only differ in flush policy. Modes
are interleaved (A,B,C, A,B,C, A,B,C) so any drift over time hits
every mode equally.

Usage:
  cd ~/Edge-AIGuard
  source ~/ollama/bin/activate
  python benchmarks/bench_streaming_tts.py [--n 3]

Output: console table + benchmarks/bench_streaming_tts.results.json
        (alongside this script, so the latest committed-comparable data
        always lives next to its harness)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

# bench lives in benchmarks/, project modules at repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import build_system_prompt  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")
PIPER_BIN = Path(os.environ.get("PIPER_BIN", "~/piper/piper/piper")).expanduser()
PIPER_MODEL = Path(
    os.environ.get("PIPER_MODEL", "~/piper/piper/en_US-amy-medium.onnx")
).expanduser()

# Use the production system prompt verbatim — same template the live
# pipeline injects every turn, so TTFA numbers reflect what a real user
# hears. Fix a "normal" radar reading so all trials see identical context;
# only the bench-id sentinel varies (cache control).
RADAR_STATE = {
    "hr": 72,
    "br": 16,
    "presence": True,
    "category": "normal",
}
BASE_SYSTEM = build_system_prompt(RADAR_STATE)

# Three queries spanning the short / medium / multi-clause spectrum, all
# grounded in the project's physiological-awareness context (HR, BR,
# body condition, mental state). All three are phrased to fall into the
# system prompt's "any other input → 1-2 sentences, read values from the
# readings line" fallback (avoiding DETECT/GUIDE/CONFIRM which produce
# templated replies that wouldn't exercise phrase aggregation). Each
# explicitly names HR / BR / presence so the LLM has to enumerate them,
# which naturally produces commas — the signal phrase-flush mode can
# act on but sentence-flush mode cannot.
#
#   Q1  short factual — "Your heart rate is 72 bpm." (1 clause)
#   Q2  two-aspect compare — naturally yields a 2-clause reply
#   Q3  three-aspect analytic — yields 3+ clauses
#
# Q3 is where phrase-flush should win the most over sentence-flush; Q1
# is a control that should show A≈B≈C convergence.
QUERIES = [
    "What is my heart rate?",
    "What are my heart rate and my breathing rate?",
    "What are my heart rate, my breathing rate, and my presence status?",
]

PHRASE_BREAK = set(",;:")
SENTENCE_END = set(".!?")
MIN_PHRASE_LEN = 6  # matches tts_aggregator.py


def system_with_sentinel() -> str:
    """Defeat Ollama's KV cache between trials."""
    return f"{BASE_SYSTEM}\n[bench-id: {secrets.token_hex(8)}]"


def post_generate(prompt: str, system: str, stream: bool):
    # temperature=0.2: low enough for shape stability across trials of the
    # same query (so medians are meaningful), high enough that the per-trial
    # sentinel still nudges the prefix encoding so we don't collapse into
    # an identical KV-cache hit. The exact response wording still varies a
    # little but the punctuation pattern stays consistent, which is what
    # the flush-policy comparison depends on.
    body = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": stream,
        "options": {"num_predict": 80, "temperature": 0.2},
        "keep_alive": "30m",
    }
    return requests.post(OLLAMA_URL, json=body, stream=stream, timeout=120)


def stream_chunks(query: str, system: str):
    """Yield (chunk_text, t_offset_seconds) tuples from streaming LLM call.

    t_offset is wall-clock seconds since the POST started (so the
    first yield's t is the LLM's TTFT)."""
    t0 = time.perf_counter()
    r = post_generate(query, system, stream=True)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line).get("response", "")
        if chunk:
            yield chunk, time.perf_counter() - t0


def mode_a_whole(query: str, system: str) -> tuple[float, str]:
    """No streaming. Wait for full response; flush_t is when full text exists."""
    t0 = time.perf_counter()
    r = post_generate(query, system, stream=False)
    r.raise_for_status()
    full = r.json().get("response", "")
    return time.perf_counter() - t0, full


def mode_b_sentence(query: str, system: str) -> tuple[float, str]:
    """Stream; flush at first . ! ?"""
    buf = ""
    last_t = 0.0
    for chunk, t in stream_chunks(query, system):
        buf += chunk
        last_t = t
        for i, ch in enumerate(buf):
            if ch in SENTENCE_END:
                return t, buf[: i + 1]
    return last_t, buf  # no terminal punctuation seen — return whole reply


def mode_c_phrase(query: str, system: str) -> tuple[float, str]:
    """Stream; flush at first . ! ? OR , : ; (with ≥6 chars accumulated)."""
    buf = ""
    last_t = 0.0
    for chunk, t in stream_chunks(query, system):
        buf += chunk
        last_t = t
        for i, ch in enumerate(buf):
            if ch in SENTENCE_END:
                return t, buf[: i + 1]
            if ch in PHRASE_BREAK:
                stripped = buf[: i + 1].strip()
                if len(stripped) >= MIN_PHRASE_LEN:
                    return t, buf[: i + 1]
    return last_t, buf


def piper_first_byte(text: str) -> float:
    """Time from piper subprocess launch to first stdout audio byte (seconds).

    Mirrors pipeline.py's per-flush invocation: fresh subprocess per
    flush, including model load. This is the conservative figure;
    pipecat's PiperTTSService keeps a daemon and would amortize the
    load across a session — that improvement is orthogonal to flush
    policy and would help all three modes equally."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [str(PIPER_BIN), "--model", str(PIPER_MODEL), "--output_raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(text.encode())
    proc.stdin.close()
    first_byte = proc.stdout.read(1)
    t_first = time.perf_counter() - t0
    if not first_byte:
        # piper produced no audio (probably empty text); fail loudly
        raise RuntimeError(f"piper emitted no audio for: {text!r}")
    # drain remaining audio so the subprocess can exit cleanly
    while proc.stdout.read(8192):
        pass
    proc.wait()
    return t_first


def warmup():
    """Warm Ollama (model in RAM) and piper (kernel cache + ONNX warmup)."""
    print("[warmup] Ollama...", flush=True)
    requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": "hi",
            "system": BASE_SYSTEM,
            "stream": False,
            "options": {"num_predict": 1},
            "keep_alive": "30m",
        },
        timeout=120,
    ).raise_for_status()
    print("[warmup] Piper...", flush=True)
    piper_first_byte("warming up the speech synthesizer.")
    print("[warmup] done.\n", flush=True)


MODES = [
    ("A.whole", mode_a_whole),
    ("B.sentence", mode_b_sentence),
    ("C.phrase", mode_c_phrase),
]


def median_int(xs):
    return int(statistics.median(xs) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="trials per (mode, query)")
    args = ap.parse_args()

    print(f"Edge-AIGuard streaming-TTS A/B benchmark")
    print(f"  model={MODEL}, piper={PIPER_MODEL.name}, n={args.n}")
    print()
    warmup()

    rows = []  # accumulated per-trial rows
    summary = {}  # (mode, query_idx) -> dict of medians

    # Interleave modes per trial so any cache / thermal drift hits all modes.
    for trial in range(args.n):
        for q_idx, query in enumerate(QUERIES):
            for mode_name, mode_fn in MODES:
                system = system_with_sentinel()
                t_flush, text = mode_fn(query, system)
                t_piper = piper_first_byte(text) if text.strip() else 0.0
                t_ttfa = t_flush + t_piper
                rows.append({
                    "trial": trial,
                    "query_idx": q_idx,
                    "query": query,
                    "mode": mode_name,
                    "flush_s": round(t_flush, 3),
                    "piper_first_byte_s": round(t_piper, 3),
                    "ttfa_s": round(t_ttfa, 3),
                    "first_text_chars": len(text),
                    "first_text_preview": text[:60],
                })
                print(
                    f"  [trial {trial+1}/{args.n}] q{q_idx} {mode_name:11s} "
                    f"flush={t_flush*1000:>6.0f}ms  "
                    f"piper={t_piper*1000:>5.0f}ms  "
                    f"ttfa={t_ttfa*1000:>6.0f}ms  "
                    f"text[:40]={text[:40]!r}",
                    flush=True,
                )

    # Aggregate
    print("\n" + "=" * 78)
    print("Median TTFA (ms) — lower is better")
    print("=" * 78)
    header = f"{'query':40s}  {'A.whole':>10s}  {'B.sent':>10s}  {'C.phrase':>10s}"
    print(header)
    print("-" * len(header))

    for q_idx, query in enumerate(QUERIES):
        cells = [query[:38] + ".." if len(query) > 40 else query]
        for mode_name, _ in MODES:
            ttfas = [r["ttfa_s"] for r in rows
                     if r["query_idx"] == q_idx and r["mode"] == mode_name]
            cells.append(f"{median_int(ttfas):>9d}")
        print(f"{cells[0]:40s}  {cells[1]:>10s}  {cells[2]:>10s}  {cells[3]:>10s}")
        # Per-query speedup of C over A
        a_med = statistics.median([r["ttfa_s"] for r in rows
                                   if r["query_idx"] == q_idx and r["mode"] == "A.whole"])
        c_med = statistics.median([r["ttfa_s"] for r in rows
                                   if r["query_idx"] == q_idx and r["mode"] == "C.phrase"])
        if a_med > 0:
            print(f"{'  → C/A saved':40s}  {(1 - c_med / a_med) * 100:>9.1f}%")

    print("\nMedian flush_ms (LLM-only contribution to TTFA):")
    for q_idx, query in enumerate(QUERIES):
        cells = [query[:38] + ".." if len(query) > 40 else query]
        for mode_name, _ in MODES:
            xs = [r["flush_s"] for r in rows
                  if r["query_idx"] == q_idx and r["mode"] == mode_name]
            cells.append(f"{median_int(xs):>9d}")
        print(f"{cells[0]:40s}  {cells[1]:>10s}  {cells[2]:>10s}  {cells[3]:>10s}")

    # Write next to the harness so committed results stay in sync with the
    # version of the script that produced them.
    out = Path(__file__).resolve().parent / "bench_streaming_tts.results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nRaw results: {out}")


if __name__ == "__main__":
    main()
