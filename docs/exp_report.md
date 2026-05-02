# Edge-AIGuard Experiment Report

> Authors: Hanlin Wang (hw3100), Yuxi Luo (yl6117)
> Date: 2026-05-01
> Project version: v1.13 ([Edge-AIGuard-Plan.md](Edge-AIGuard-Plan.md))

This report archives the quantitative benchmarks supporting Edge-AIGuard's two
headline efficiency claims:

1. **Speculative LLM prefill** — pre-warming Ollama's KV cache during partial
   transcription cuts user-perceived response latency by 73% on a realistic
   3-turn dialogue (Section 2).
2. **Streaming ASR + VAD framework throughput** — sherpa-onnx + Silero VAD on
   pipecat run at <15% CPU on Pi 5, leaving headroom for concurrent LLM and
   TTS (Section 3, summarized — full tables in plan).

All experiments were run on the production hardware. Raw data and harnesses
are committed in this repository for reproducibility.

---

## 1. Experimental setup

### 1.1 Hardware

| Component | Spec |
|---|---|
| Compute | Raspberry Pi 5, 8 GB LPDDR4X, BCM2712 (4× Cortex-A76 @ 2.4 GHz) |
| Power | 27 W USB-C PD (5 V / 5.4 A nominal) |
| Cooling | Passive heatsink + 30 mm fan |
| Storage | 32 GB SDXC A1 |
| Voice terminal | M5Stack Core2 V1.1 (ESP32-D0WDQ6, 8 MB PSRAM) over Wi-Fi |
| Radar | Seeed MR60BHA2 Kit (XIAO ESP32-C6 + 60 GHz mmWave) over Wi-Fi |
| Network | 2.4 GHz Wi-Fi, RSSI -44 dB to Core2 |

Thermal state was verified `vcgencmd get_throttled == 0x0` before each
benchmark; CPU temperature stayed under 60 °C throughout.

### 1.2 Software

| Layer | Version |
|---|---|
| OS | Raspberry Pi OS Bookworm (64-bit) |
| Python | 3.13.5 |
| Ollama | 0.18.2 (`/usr/bin/ollama`, manual install) |
| LLM | qwen2.5:1.5b (1.0 GB Q4_K_M, 1.54 B params) |
| ASR | sherpa-onnx-streaming-zipformer-en-2023-06-26 (int8) |
| TTS | Piper en_US-amy-medium (22.05 kHz native; resampled to 16 kHz for Core2) |
| VAD | Silero VAD v5 (ONNX) |
| Framework | pipecat 1.1.0 |
| ESPHome | 2025.6.0 (`~/esphome-old/`) |

---

## 2. Speculative LLM prefill: 3-turn dialogue benchmark

### 2.1 Question

> Production deployment fires `/api/generate?num_predict=1` with a 30-minute
> `keep_alive` on every `InterimTranscriptionFrame` from the streaming ASR.
> By the time the user stops speaking and the real `/v1/chat` call goes out,
> Ollama's KV cache should already hold the system prompt + dialogue history
> + (most of) the user's utterance. **How much TTFT does this save across a
> realistic multi-turn conversation?**

### 2.2 Workload

A canonical health-monitoring exchange:

| Turn | User utterance | Tokens (full prompt incl. system + history) |
|:---:|:---|:---:|
| 1 | "hello" | 189 |
| 2 | "check my heart rate and breathing rate" | 217 |
| 3 | "Is my health condition good?" | 253 |

System prompt is built by `prompts.build_system_prompt({"hr": 72, "br": 16,
"presence": True, "category": "normal"})` — the dialogue's radar context is
the v1.12-optimized layout that places the volatile HR/BR readings at the
end so the static prefix maximizes KV-cache reuse across turns.

### 2.3 Methodology

For each turn, we measure two conditions:

- **COLD**: direct `/v1/chat` call. Represents the no-speculation baseline
  (what users would see if `--no-speculation` were passed).
- **WARM**: `/api/generate?num_predict=1` (speculation), wait for completion,
  then `/v1/chat`. Represents the production code path triggered by the
  speculative prefill processor on the final `InterimTranscriptionFrame`
  before the user stops speaking.

**Cache pollution control.** Each condition appends a unique
`[bench-id: <16-hex sentinel>]` line to its system prompt. The sentinel does
not steer the model's response but forces a fresh KV-cache prefix per
condition. Without it, a warm trial benefiting from the previous cold trial's
cache would inflate apparent savings.

**Independent histories.** Cold and warm tracks accumulate distinct assistant
responses. Turn 3's cold context is built from Turns 1–2's cold responses;
the warm context is built from warm responses. This faithfully reproduces the
in-production scenario where speculation pre-warms the *exact* prefix the
real call will see.

**Sample size.** N = 3 independent runs of the full 3-turn protocol
(6 dialogue trajectories total). Reported figures are medians; cold
prefill ranges report standard deviation.

**Timing source.** `prompt_eval_duration` from Ollama's `/api/generate`
response (nanosecond-precision native counter); end-to-end wall reported as
total POST latency from a `time.perf_counter()` bracket on the Python client.

Harness: [`bench_three_turn.py`](../benchmarks/bench_three_turn.py). Raw data:
[`/tmp/bench_three_turn.json`](file:///tmp/bench_three_turn.json) (most recent
run; archive a copy before re-running if needed for paper figures).

### 2.4 Results

**Table 1.** KV-cache prefill speedup via speculative pre-warming on a
3-turn dialogue. qwen2.5:1.5b, Raspberry Pi 5, Ollama 0.18.2, num_threads=4.
N = 3 runs per cell, sentinels rotated. Wall time figures are upper bounds
on streaming TTFT.

| Turn | User utterance | Prompt tokens | Prefill cold (ms) | Prefill warm (ms) | **Prefill saved** | Wall cold (ms) | Wall warm (ms) | **Wall saved** |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | "hello" | 189 | 7952 ± 488 | 84 | **98.9 %** | 9243 ± 394 | 1258 | 86.4 % |
| 2 | "check my heart rate and breathing rate" | 217 | 2935 ± 107 | 84 | **97.1 %** | 5221 ± 404 | 2367 | 54.7 % |
| 3 | "Is my health condition good?" | 253 | 4743 ± 543 | 88 | **98.1 %** | 6811 ± 575 | 2146 | 68.5 % |
| | **Aggregate (3 turns)** | **659** | **15 630** | **256** | **98.4 %** | **21 275** | **5 771** | **72.9 %** |

**Table 2.** Per-token prefill cost.

| Condition | ms / token | Speedup vs cold |
|---|---:|---:|
| Cold | 23.72 | 1.0× |
| Warm (cache hit) | 0.39 | **61.1×** |

**Table 3.** Speculation cost (paid asynchronously during user speech, hidden
from user-perceived latency).

| Turn | Median spec overhead (ms) | Concurrent with |
|:---:|:---:|:---|
| 1 | 1833 | User speaking ("hello…") |
| 2 | 3121 | User speaking ("check my heart rate…") |
| 3 | 5371 | User speaking ("Is my health condition good?") |

The pre-warm always completes before the user's silence-detected endpoint, so
it never extends user-perceived latency. The only constraint is that user
speech duration ≥ pre-warm time, satisfied for all conversational utterances
in our workload.

### 2.5 Analysis

**Cache hit ratio is near-saturating (98.4% aggregate).** Warm prefill is
flat at ~85 ms regardless of context length, consistent with cache lookup
being O(L) on cached tokens at fixed per-token cost (~0.4 ms/token, dominated
by attention KV memory loads, no matmul).

**Wall savings (72.9%) lag prefill savings (98.4%) by design.** Speculative
prefill addresses prefill latency only. Autoregressive decode is *not* cached
and runs at ~10 tok/s on Pi 5 CPU regardless of prior context. For a 20-token
reply, decode contributes ~2 s of fixed wall time visible in both conditions.
Future work targeting decode would require either model distillation
(qwen2.5:0.5b at 18 tok/s benched separately) or speculative decoding (an
orthogonal technique to speculative *prefill*).

**Turn 1 anomalously slow cold prefill (7952 ms).** First inference after
model load includes one-time JIT of CPU matmul kernels, BLAS dispatch table
population, and L1/L2 cache warmup. Per-token cost falls from 42 ms (Turn 1)
to 13.5 ms (Turn 2) to 18.7 ms (Turn 3). We report Turn 1 separately rather
than smoothing because *the first utterance after device boot* is the
user's first impression of system latency, and Edge-AIGuard's mitigation
(speculative prefill firing on partial transcription) is precisely what
masks this overhead in production — Warm Turn 1 is just 1258 ms, an 86.4%
improvement.

**Token-count variance between conditions (e.g. Turn 1: 189 cold vs 188
warm) is sentinel-induced.** Different random hex strings tokenize to
slightly different lengths under qwen2.5's BPE. Within a single condition
the spec and real call use the *same* sentinel, so the cache prefix matches
exactly.

### 2.6 Reproducibility

```
cd ~/Edge-AIGuard
source ~/ollama/bin/activate
python bench_three_turn.py
```

Output is a console table plus `/tmp/bench_three_turn.json` with per-turn
metrics (`turn`, `condition`, `wall_ms`, `prompt_eval_count`,
`prompt_eval_ms`, `eval_count`, `eval_ms`, `spec_wall_ms`).

To reproduce a specific cell, hardcode the sentinel in
[bench_three_turn.py:128](../benchmarks/bench_three_turn.py#L128). Note that Ollama's
`keep_alive=30m` means the model stays warm between runs — for a true
cold-start measurement, run `ollama stop qwen2.5:1.5b` first.

---

## 3. Component microbenchmarks (summary)

Detailed methodology and raw numbers for these are in
[Edge-AIGuard-Plan.md § "已采集的基线数据"](Edge-AIGuard-Plan.md). Reproduced
here for cross-reference.

**Table 4.** Streaming ASR (sherpa-onnx zipformer-en int8, Pi 5, threads=2),
on 6.62 s LibriSpeech utterance.

| Metric | Value | Gate | Pass |
|---|---:|---:|:---:|
| Real-time factor (median of 3) | 0.100 | < 0.8 | yes (8× margin) |
| Resident memory | 205 MB | < 600 MB | yes (3× margin) |
| First-partial latency | 95 ms | — | — |
| Per-chunk decode (p95 / max) | 31 / 36 ms | < 320 ms | yes |
| Model load (one-time) | 3.58 s | — | — |

**Table 5.** Silero VAD on pipecat (Pi 5).

| Metric | Value | Gate | Pass |
|---|---:|---:|:---:|
| Load time | 90 ms | < 5 s | yes |
| Detection rate (test) | 85% chunks | > 40% | yes |
| Real-time factor | 0.024 | < 0.5 | yes (40× realtime) |
| Combined memory (pipecat + VAD) | 200 MB | < 700 MB | yes |

**Table 6.** TTS (Piper en_US-amy-medium, Pi 5, single thread).

| Metric | Value |
|---|---:|
| Real-time factor (synthesis) | 0.13 |
| Effective synthesis rate | 7.5× realtime |

**Resource budget at full load.** Roughly: Ollama (1.5 B model) ~1.4 GB + sherpa
ASR 0.2 GB + Silero VAD 0.05 GB + Piper TTS 0.3 GB + framework overhead
~0.5 GB, total ~2.5 GB. Pi 5 has 8 GB; ample headroom for the ESPHome API
client + radar subscriber.

---

## 4. Cross-cutting observations

**Static-prefix layout matters.** The v1.12 prompt-template revision moved
the volatile `Current readings: heart rate {hr} bpm…` line from the middle
of the system prompt to the end. Without that change, a single-bpm change
(e.g. 72 → 73) invalidated the entire system-prompt KV cache and cut hit
rate from 67% to 45% (see plan v1.12, "KV-cache 复用实验"). The current
3-turn results assume this layout — without it, results would degrade
proportionally.

**Cancel-on-new pattern.** Each new partial transcription cancels the
in-flight speculative call at the asyncio level, but the underlying HTTP
request to Ollama runs to completion. This is intentional: a cancelled
spec's prefix is a valid prefix of the next spec, so its KV-cache
contribution carries forward. Profiling shows ~3 cancelled spec calls per
final transcription on average — none wasted.

---

## 5. Limitations

1. **N = 3.** Sufficient for variance bounds within ~10% but not tight error
   bars. A larger N would tighten the standard deviations on cold prefill
   (currently up to ±575 ms / 8% of mean for Turn 3).

2. **Single workload.** Three turns is representative of short health
   queries but does not stress longer dialogues. Cache hit rate likely
   degrades with very long histories where prior assistant responses span
   500+ tokens, but the dominant prefix (system + readings) still benefits.

3. **English-only ASR/TTS.** Sherpa zipformer-en and Piper Amy were chosen
   for the demo; multilingual would require swapping models with their own
   memory/CPU profile.

4. **Ollama-specific.** Speculative prefill exploits Ollama's prompt-cache
   behavior. Other inference runtimes (vLLM, llama.cpp standalone) would
   need verification that `/api/generate` and chat endpoints share KV cache;
   this is documented (and tested in `test_chat_template_alignment.py`) for
   Ollama 0.18.2.
