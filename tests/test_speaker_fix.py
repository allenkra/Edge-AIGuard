"""Direct test of Core2 speaker fix — bypasses sherpa STT entirely.

Replays the exact event sequence Core2OutputTransport produces, with STT_VAD_END
inserted at the right point. Streams a pre-baked TTS WAV (or any 22050 Hz mono
int16 PCM) into Core2 via voice_assistant_audio.

If speaker plays the audio without "Cannot receive audio, buffer is full"
errors, the I2S0 release fix is confirmed working.

Usage: python test_speaker_fix.py [path/to/22050_mono_s16.raw]
       defaults to /tmp/edge_aiguard_last.raw produced by an earlier text run.
"""
import asyncio
import os
import sys

from aioesphomeapi import APIClient
from aioesphomeapi.model import VoiceAssistantEventType as VA


async def main(audio_path: str) -> int:
    host = os.environ.get("CORE2_HOST", "edge-aiguard-core2.local")
    psk = os.environ.get("CORE2_NOISE_PSK", "")
    if not psk:
        print("CORE2_NOISE_PSK env var required", file=sys.stderr)
        return 2

    if not os.path.exists(audio_path):
        print(f"audio file missing: {audio_path}", file=sys.stderr)
        return 2
    audio = open(audio_path, "rb").read()
    print(f"[test] loaded {len(audio)} bytes of audio from {audio_path}")
    # Auto-resample 22050 → 16000 if filename contains 'last' (heuristic for
    # pipeline-generated raw which is always 22050). 16k file passthrough.
    if "16k" not in audio_path:
        import numpy as np
        from scipy import signal
        samples = np.frombuffer(audio, dtype=np.int16)
        new_n = int(len(samples) * 16000 / 22050)
        resampled = signal.resample(samples.astype(np.float32), new_n)
        np.clip(resampled, -32768, 32767, out=resampled)
        audio = resampled.astype(np.int16).tobytes()
        print(f"[test] resampled to 16 kHz: {len(audio)} bytes")

    client = APIClient(host, 6053, password="", noise_psk=psk)
    await client.connect(login=True)

    va_started = asyncio.Event()

    async def handle_start(conv, flags, settings, wake):
        print(f"[test] VA start (conv={conv}, flags={flags})")
        va_started.set()
        return 0

    async def handle_stop(server_side):
        print(f"[test] VA stop server_side={server_side}")

    async def handle_audio(data):
        # ignore mic data — we're testing speaker only
        pass

    client.subscribe_voice_assistant(
        handle_start=handle_start,
        handle_stop=handle_stop,
        handle_audio=handle_audio,
    )

    _, services = await client.list_entities_services()
    start_listen = next((s for s in services if s.name == "start_listening"), None)
    if start_listen is None:
        print("start_listening service missing — flash latest core2.yaml", file=sys.stderr)
        return 1

    print("[test] triggering start_listening")
    await client.execute_service(start_listen, {})

    try:
        await asyncio.wait_for(va_started.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        print("[test] VA never started — voice_assistant.start probably failed", file=sys.stderr)
        return 1

    # Mimic the exact sequence Pi sends after touch + speech + endpoint:
    print("[test] sending RUN_START + STT_START")
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_RUN_START, {})
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_STT_START, {})

    # Let mic stream a tiny bit so we're squarely in STREAMING_MICROPHONE
    await asyncio.sleep(1.5)

    # **THE FIX**: STT_VAD_END forces voice_assistant to STOP_MICROPHONE,
    # releasing I2S0 before speaker_->start() runs.
    print("[test] sending STT_VAD_END (the critical fix)")
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_STT_VAD_END, {})
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_STT_END, {})
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_INTENT_START, {})

    # Give Core2 time to actually stop mic and free I2S0
    await asyncio.sleep(0.8)

    print("[test] sending INTENT_END + TTS_START + TTS_END(url) + TTS_STREAM_START")
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_INTENT_END, {})
    # text="..." required: voice_assistant.cpp:652 early-returns on empty text
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_TTS_START, {"text": "speaker test"})
    # url="..." required: voice_assistant.cpp:674 early-returns on empty url.
    # Without this the state stays AWAITING_RESPONSE forever and write_speaker_
    # never runs → 16KB pre-buffer fills in <1s → all subsequent chunks dropped.
    # url is only used for media_player path; for local speaker any non-empty
    # string works to advance the state machine to STREAMING_RESPONSE.
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_TTS_END, {"url": "tts://local"})
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_TTS_STREAM_START, {})

    # ESPHome speaker uses default AudioStreamInfo (16 kHz mono int16) because
    # voice_assistant never calls set_audio_stream_info(). Drain = 32 KB/s.
    # 1KB @ 35ms = 29.3 KB/s = 91% of realtime, no overflow.
    # Audio file MUST be 16 kHz mono int16 — 22050 Hz inputs slow-mo + overflow.
    CHUNK = 1024
    SLEEP_S = 0.035
    print(f"[test] streaming {len(audio)} bytes in {(len(audio) + CHUNK - 1) // CHUNK} chunks")
    for i in range(0, len(audio), CHUNK):
        client.send_voice_assistant_audio(audio[i:i + CHUNK])
        await asyncio.sleep(SLEEP_S)

    print("[test] sending TTS_STREAM_END + RUN_END (TTS_END already sent earlier)")
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_TTS_STREAM_END, {})
    client.send_voice_assistant_event(VA.VOICE_ASSISTANT_RUN_END, {})

    # Hold connection a moment so Core2 finishes draining its buffer
    await asyncio.sleep(2.0)

    await client.disconnect()
    print("[test] done — check Core2 log for speaker behavior")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/edge_aiguard_last.raw"
    sys.exit(asyncio.run(main(path)))
