"""
Edge-AIGuard - Day 1 Pipeline
ASR (Whisper) -> LLM (Ollama) -> TTS (Piper)

Modes:
  default  按 ENTER 录音 5 秒 (需要麦克风)
  --text   纯文字输入 (无麦克风时用)
  --no-audio 不播放 TTS (生成 WAV 到 /tmp 但不 aplay)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests

from prompts import build_system_prompt
from radar import RadarReader
from tools import chat_with_tools, should_use_tools

# === 配置 ===
WHISPER_MODEL = "base.en"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")
PIPER_BIN = os.path.expanduser("~/piper/piper/piper")
PIPER_MODEL = os.path.expanduser("~/piper/piper/en_US-amy-medium.onnx")
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
# Use mDNS hostname instead of IP — survives DHCP changes
RADAR_IP = os.environ.get("RADAR_IP", "seeedstudio-mr60bha2-kit-12fd18.lan")
ENABLE_TOOLS = os.environ.get("ENABLE_TOOLS", "0") == "1"


def record(duration=RECORD_SECONDS):
    import sounddevice as sd
    import numpy as np
    print(f"[mic] recording {duration}s...")
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def transcribe(audio, asr_model):
    t0 = time.time()
    segments, _ = asr_model.transcribe(audio, beam_size=1, language="en")
    text = " ".join([s.text for s in segments]).strip()
    dt = time.time() - t0
    print(f"[asr] ({dt:.2f}s) {text}")
    return text, dt


def ask_llm_streaming(prompt, system_prompt, on_sentence):
    t0 = time.time()
    first_token_time = None

    r = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {"num_predict": 80, "temperature": 0.7},
    }, stream=True, timeout=60)

    buffer = ""
    full = ""
    sentence_re = re.compile(r"([.!?])\s")

    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line).get("response", "")
        if chunk and first_token_time is None:
            first_token_time = time.time() - t0
            print(f"[llm] first token {first_token_time:.2f}s")
        buffer += chunk
        full += chunk

        while True:
            m = sentence_re.search(buffer)
            if not m:
                break
            sentence = buffer[:m.end()].strip()
            buffer = buffer[m.end():]
            if sentence:
                on_sentence(sentence)

    if buffer.strip():
        on_sentence(buffer.strip())

    total = time.time() - t0
    print(f"[llm] total {total:.2f}s")
    return full, first_token_time, total


def speak(text, no_audio=False):
    t0 = time.time()
    cmd = [PIPER_BIN, "--model", PIPER_MODEL, "--output_raw"]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    raw, _ = proc.communicate(input=text.encode())

    if no_audio:
        # 静默模式: 把音频追加到 /tmp/edge_aiguard_last.raw 以便事后听
        with open("/tmp/edge_aiguard_last.raw", "ab") as f:
            f.write(raw)
        print(f"[tts] {time.time()-t0:.2f}s '{text[:40]}' -> /tmp/edge_aiguard_last.raw")
    else:
        aplay = subprocess.Popen(
            ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        aplay.communicate(input=raw)
        print(f"[tts] {time.time()-t0:.2f}s '{text[:40]}'")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="store_true",
                   help="text input mode (no microphone)")
    p.add_argument("--no-audio", action="store_true",
                   help="generate TTS but don't play")
    p.add_argument("--no-radar", action="store_true",
                   help="skip radar connection (use fake mode only)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("Edge-AIGuard Day 1 Pipeline")
    print(f"  LLM={LLM_MODEL} | Whisper={WHISPER_MODEL}")
    print(f"  Radar={'OFF' if args.no_radar else RADAR_IP}")
    print(f"  Mode={'TEXT' if args.text else 'VOICE'} | "
          f"Audio={'OFF' if args.no_audio else 'ON'} | "
          f"Tools={'ON' if ENABLE_TOOLS else 'OFF'}")
    print("Commands: ENTER=talk/submit, m=cycle fake mode, "
          "s=show state, q=quit")
    print("=" * 50)

    asr_model = None
    if not args.text:
        from faster_whisper import WhisperModel
        print("[init] loading Whisper...")
        asr_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    if args.no_radar:
        radar = None
        print("[init] radar disabled, fake mode required")
    else:
        print("[init] connecting to radar...")
        radar = RadarReader(host=RADAR_IP)

    fake_modes = ["normal", "stressed", "relaxed", "exercising", None]
    fake_idx = 0
    # default to "normal" fake mode if no radar so we can demo
    if args.no_radar or not radar:
        from radar import RadarReader as _R
        radar = _R.__new__(_R)
        radar.fake_mode = "normal"
        radar.hr_window = []
        radar.br_window = []
        radar.presence = True
        radar.running = False

    def get_state():
        if radar is None:
            return {"hr": None, "br": None, "presence": False, "category": "unknown"}
        return radar.get_state()

    def on_speak(s):
        speak(s, no_audio=args.no_audio)

    try:
        while True:
            try:
                cmd = input("\n> ").strip()
            except EOFError:
                break

            if cmd.lower() == "q":
                break

            if cmd.lower() == "s":
                print(f"[state] {get_state()}")
                continue

            if cmd.lower() == "m":
                fake_idx = (fake_idx + 1) % len(fake_modes)
                if radar is not None:
                    radar.fake_mode = fake_modes[fake_idx]
                print(f"[fake] {fake_modes[fake_idx] or 'REAL (no fake)'}")
                continue

            # Get user text
            if args.text:
                if not cmd:
                    print("(empty, type something or 'q')")
                    continue
                text = cmd
            else:
                audio = record()
                text, _ = transcribe(audio, asr_model)
                if not text:
                    print("(empty transcription, skip)")
                    continue

            state = get_state()
            print(f"[state] {state}")
            system = build_system_prompt(state)

            if ENABLE_TOOLS and should_use_tools(text):
                print("[route] tool calling path")
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ]
                answer = chat_with_tools(messages, LLM_MODEL,
                                         OLLAMA_CHAT_URL, radar)
                on_speak(answer)
            else:
                ask_llm_streaming(text, system, on_sentence=on_speak)

    finally:
        if radar is not None and getattr(radar, "running", False):
            radar.stop()
        print("bye")


if __name__ == "__main__":
    main()
