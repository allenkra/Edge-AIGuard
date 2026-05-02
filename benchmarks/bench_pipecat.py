"""Pipecat minimum-viable verification on Pi 5.

Confirms:
  1. pipecat framework constructs run on Python 3.13
  2. Silero VAD model loads and emits per-chunk speech probabilities
  3. Memory footprint of pipecat + Silero is acceptable
  4. VAD on test wav (Joyce LibriSpeech sample) returns speech=True

Gate:
  - Framework imports + VAD analyzer instantiates without error
  - VAD detects speech in the test wav (probability > 0.5 on most chunks)
  - Total RSS < 700 MB after VAD model loaded
"""
import os
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.io import wavfile

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

TEST_WAV = (
    Path(__file__).resolve().parent.parent
    / "models" / "streaming-zipformer-en" / "test_0.wav"
)


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


class CountingProcessor(FrameProcessor):
    """No-op processor that just counts frames passing through."""
    def __init__(self):
        super().__init__()
        self.count = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.count += 1
        await self.push_frame(frame, direction)


async def test_silero_vad():
    """Load Silero VAD, run on test_wav, report speech detection rate."""
    print(f"\n=== Silero VAD on test wav ===")
    print(f"[mem] before VAD load: {rss_mb():.1f} MB")
    t0 = time.perf_counter()
    vad = SileroVADAnalyzer(
        sample_rate=16000,
        params=VADParams(
            confidence=0.5,
            start_secs=0.2,
            stop_secs=0.5,
        ),
    )
    # In framework usage the transport calls set_sample_rate; standalone we
    # must call it ourselves to compute internal byte buffer sizing.
    vad.set_sample_rate(16000)
    load_s = time.perf_counter() - t0
    print(f"[load] Silero VAD: {load_s:.2f}s, mem now: {rss_mb():.1f} MB")

    sr, samples = wavfile.read(TEST_WAV)
    if samples.dtype != np.int16:
        samples = (samples * 32768).astype(np.int16)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.int16)

    audio_dur = len(samples) / sr
    print(f"[wav] {TEST_WAV.name}: {audio_dur:.2f}s, {len(samples)} samples @ {sr}Hz")

    chunk_samples = vad.num_frames_required()
    print(f"[chunks] VAD wants {chunk_samples} samples per call "
          f"({1000*chunk_samples/sr:.0f}ms windows)")

    speech_chunks = 0
    states = []
    t_start = time.perf_counter()
    for i in range(0, len(samples) - chunk_samples + 1, chunk_samples):
        chunk = samples[i:i + chunk_samples].tobytes()
        state = await vad.analyze_audio(chunk)
        states.append(state)
        if state in (VADState.SPEAKING, VADState.STARTING):
            speech_chunks += 1
    proc_s = time.perf_counter() - t_start
    rtf = proc_s / audio_dur

    print(f"[vad] processed {len(states)} chunks in {proc_s*1000:.0f}ms "
          f"(RTF={rtf:.3f})")
    speech_frac = speech_chunks / max(1, len(states))
    speaking_count = sum(1 for s in states if s == VADState.SPEAKING)
    starting_count = sum(1 for s in states if s == VADState.STARTING)
    quiet_count = sum(1 for s in states if s == VADState.QUIET)
    stopping_count = sum(1 for s in states if s == VADState.STOPPING)
    print(f"[vad] state breakdown: "
          f"SPEAKING={speaking_count}, STARTING={starting_count}, "
          f"STOPPING={stopping_count}, QUIET={quiet_count}")
    print(f"[vad] {100*speech_frac:.0f}% chunks classified as speech")

    return {
        "load_s": load_s,
        "vad_rtf": rtf,
        "speech_frac": speech_frac,
        "mem_mb": rss_mb(),
        "ok": speech_frac > 0.4,
    }


async def test_pipeline_framework():
    """Run a tiny pipeline to verify async framework executes on Pi."""
    print(f"\n=== Minimal pipecat Pipeline run ===")
    counter = CountingProcessor()
    pipeline = Pipeline([counter])
    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False)

    async def feed():
        await task.queue_frames([
            TextFrame("hello"),
            TextFrame("world"),
            EndFrame(),
        ])

    import asyncio
    t0 = time.perf_counter()
    await asyncio.gather(runner.run(task), feed())
    dt = time.perf_counter() - t0
    print(f"[pipeline] ran in {dt*1000:.0f}ms, "
          f"counter saw {counter.count} frames")
    return counter.count


async def amain():
    print(f"=== pipecat minimum-viable verification on Pi 5 ===")
    print(f"[mem] start: {rss_mb():.1f} MB")

    vad_result = await test_silero_vad()
    pipeline_count = await test_pipeline_framework()
    return vad_result, pipeline_count


def main():
    import asyncio
    vad_result, pipeline_count = asyncio.run(amain())

    print(f"\n--- Gate ---")
    framework_ok = pipeline_count >= 2
    vad_ok = vad_result["ok"]
    mem_ok = vad_result["mem_mb"] < 700

    print(f"Pipeline framework runs: {framework_ok} ({pipeline_count} frames)")
    print(f"Silero VAD detects speech (>40% chunks): "
          f"{vad_ok} ({100*vad_result['speech_frac']:.0f}%)")
    print(f"Mem < 700 MB: {mem_ok} ({vad_result['mem_mb']:.0f} MB)")
    print(f"VAD RTF: {vad_result['vad_rtf']:.3f}")

    if framework_ok and vad_ok and mem_ok:
        print("\n✓ GO: pipecat + Silero VAD work on Pi 5")
    else:
        print("\n✗ NO-GO: see failed gate above")


if __name__ == "__main__":
    main()
