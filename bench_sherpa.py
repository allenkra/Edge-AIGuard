"""Sherpa-onnx streaming ASR benchmark on Pi 5.

Measures RTF, memory footprint, and per-chunk streaming latency for the
streaming-zipformer-en-2023-06-26 int8 model. Decision input for whether
to adopt sherpa-onnx as the streaming ASR for speculative LLM prefill.

Gate: RTF < 0.8 (with margin for concurrent LLM/TTS load) and memory < 600 MB.

Usage:
    ~/ollama/bin/python bench_sherpa.py
    ~/ollama/bin/python bench_sherpa.py --chunk-ms 100  # try smaller chunks
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import psutil
import sherpa_onnx
from scipy.io import wavfile

MODEL_DIR = Path(os.path.expanduser("~/Edge-AIGuard/models/streaming-zipformer-en"))
ENCODER = MODEL_DIR / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
DECODER = MODEL_DIR / "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
JOINER = MODEL_DIR / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
TOKENS = MODEL_DIR / "tokens.txt"
TEST_WAV = MODEL_DIR / "test_0.wav"


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-ms", type=int, default=100,
                   help="audio chunk size to feed per iteration (ms)")
    p.add_argument("--num-threads", type=int, default=2,
                   help="ONNX runtime intra-op threads")
    p.add_argument("--repeat", type=int, default=3,
                   help="how many times to replay the wav for stable timing")
    args = p.parse_args()

    print(f"=== sherpa-onnx streaming ASR benchmark on Pi 5 ===")
    print(f"  chunk_ms={args.chunk_ms}, num_threads={args.num_threads}, "
          f"repeat={args.repeat}")
    print(f"  model: streaming-zipformer-en-2023-06-26 (int8)")

    print(f"\n[mem] before load: {rss_mb():.1f} MB")

    t0 = time.perf_counter()
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(ENCODER),
        decoder=str(DECODER),
        joiner=str(JOINER),
        tokens=str(TOKENS),
        num_threads=args.num_threads,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
        decoding_method="greedy_search",
        provider="cpu",
    )
    load_s = time.perf_counter() - t0
    print(f"[load] {load_s:.2f}s, mem now: {rss_mb():.1f} MB "
          f"(Δ={rss_mb()-50:.0f} MB rough)")

    sr, samples = wavfile.read(TEST_WAV)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32) / 32768.0
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    audio_dur_s = len(samples) / sr
    print(f"\n[wav] {TEST_WAV.name}: sr={sr}, "
          f"duration={audio_dur_s:.2f}s, {len(samples)} samples")

    chunk_samples = int(args.chunk_ms / 1000 * sr)
    print(f"[chunks] {chunk_samples} samples per chunk "
          f"({args.chunk_ms}ms each, "
          f"{(len(samples)+chunk_samples-1)//chunk_samples} chunks total)")

    print(f"\n--- Running {args.repeat} repeats ---")
    rtfs = []
    first_partial_times = []
    chunk_decode_times_all = []

    for run_i in range(args.repeat):
        stream = recognizer.create_stream()
        prev_text = ""
        first_partial_t = None
        chunk_decode_times = []
        last_partial_idx = -1

        run_start = time.perf_counter()
        for i in range(0, len(samples), chunk_samples):
            chunk = samples[i:i + chunk_samples]
            chunk_t0 = time.perf_counter()
            stream.accept_waveform(sr, chunk)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            chunk_dt = time.perf_counter() - chunk_t0
            chunk_decode_times.append(chunk_dt)

            text = recognizer.get_result(stream).strip()
            if text and first_partial_t is None:
                first_partial_t = time.perf_counter() - run_start
            if text and text != prev_text:
                if i // chunk_samples != last_partial_idx:
                    last_partial_idx = i // chunk_samples
                    prev_text = text

        # Flush
        tail = np.zeros(int(0.5 * sr), dtype=np.float32)
        stream.accept_waveform(sr, tail)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        run_dur = time.perf_counter() - run_start
        rtf = run_dur / audio_dur_s
        final_text = recognizer.get_result(stream).strip()
        rtfs.append(rtf)
        first_partial_times.append(first_partial_t or float("nan"))
        chunk_decode_times_all.extend(chunk_decode_times)

        print(f"  [run {run_i+1}] proc_time={run_dur:.3f}s, "
              f"audio={audio_dur_s:.2f}s, RTF={rtf:.3f}, "
              f"first_partial={first_partial_t:.3f}s")
        print(f"           final: '{final_text[:90]}'")

    print(f"\n[mem] after {args.repeat} runs: {rss_mb():.1f} MB")

    cdt = np.array(chunk_decode_times_all) * 1000  # to ms
    print(f"\n--- Summary ---")
    print(f"RTF: median={np.median(rtfs):.3f}, "
          f"mean={np.mean(rtfs):.3f}, "
          f"min={min(rtfs):.3f}, max={max(rtfs):.3f}")
    print(f"First-partial latency: median={np.nanmedian(first_partial_times):.3f}s")
    print(f"Per-chunk decode time (ms): "
          f"median={np.median(cdt):.1f}, "
          f"p95={np.percentile(cdt, 95):.1f}, "
          f"max={cdt.max():.1f}")
    print(f"Memory: {rss_mb():.0f} MB resident")

    print(f"\n--- Gate ---")
    median_rtf = float(np.median(rtfs))
    median_mem = rss_mb()
    rtf_ok = median_rtf < 0.8
    mem_ok = median_mem < 600
    print(f"RTF < 0.8: {median_rtf:.3f} → {'PASS' if rtf_ok else 'FAIL'}")
    print(f"Mem < 600 MB: {median_mem:.0f} MB → {'PASS' if mem_ok else 'FAIL'}")
    if rtf_ok and mem_ok:
        print("✓ GO: sherpa-onnx is feasible on Pi 5")
    else:
        print("✗ NO-GO: see failed gate above")


if __name__ == "__main__":
    main()
