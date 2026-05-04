# Edge-AIGuard

An always-on, fully on-device health companion for the home. Edge-AIGuard
runs streaming ASR, a 1.5 B-parameter LLM, and neural TTS on a single
Raspberry Pi 5 while a 60 GHz mmWave radar fuses heart-rate and
breathing-rate readings into the LLM's system prompt. No cloud, no
internet required at runtime.


## Hardware Requirements

| Device | Required | Notes |
|--------|----------|-------|
| Raspberry Pi 5 (8 GB) | yes | 27 W USB-C PD supply; `vcgencmd get_throttled` should read `0x0`. Active cooler recommended. |
| MR60BHA2 mmWave radar kit | yes for radar | Seeed MR60BHA2 + XIAO ESP32-C6, ESPHome firmware, on the Pi's WiFi. |
| M5Stack Core2 V1.1 | yes for voice terminal | V1.1 only (V1.0's AXP192 is incompatible with the AXP2101 component). |
| USB-C cable Pi ↔ Core2 | first flash only | Subsequent flashes go OTA over WiFi. |
| USB mic / DAC / HDMI speaker | optional | For the Pi-local demo (`pipeline_pipecat.py` with no `--core2`). |

The full pipeline runs in ≈2.5 GB resident memory at <80 % single-core
peak on the Pi 5, with ≈10 tok/s decode on Qwen2.5-1.5B Q4_K_M.

---

## Environment Setup

All paths assume `~` is your Pi user's home directory.

### 1. Pi packages and Ollama model

```bash
sudo apt install -y nmap python3.13 python3.13-venv

python3 -m venv ~/ollama
source ~/ollama/bin/activate
pip install --upgrade pip
pip install pipecat-ai sherpa-onnx silero-vad aioesphomeapi loguru \
            scipy numpy pyserial flask matplotlib requests psutil

ollama pull qwen2.5:1.5b
```

### 2. Streaming ASR model

Download the sherpa-onnx Zipformer English model into
`models/streaming-zipformer-en/`:

```
encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx
tokens.txt
```

See <https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html>.

Optional — enable hotword biasing for higher accuracy on domain phrases:

```bash
cd models/streaming-zipformer-en
wget https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/bpe.model
pip install sentencepiece
python -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor(); sp.load('bpe.model')
with open('bpe.vocab', 'w') as f:
    for i in range(sp.get_piece_size()):
        f.write(f'{sp.id_to_piece(i)} {sp.get_score(i)}\n')
"
```

`hotwords.txt` ships in the repo with project defaults. The pipeline
auto-detects both files; if either is missing it falls back to plain
greedy decoding.

### 3. Piper TTS

```bash
mkdir -p ~/piper && cd ~/piper
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz && rm piper_linux_aarch64.tar.gz
cd piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

### 4. ESPHome (only if flashing Core2 / radar yourself)

Pin 2025.6.x — the AXP2101 community component breaks on 2026.x.

```bash
python3 -m venv ~/esphome-old
~/esphome-old/bin/pip install "esphome==2025.6.0"
```

---

## Configuration

### `esphome/secrets.yaml` (required, gitignored)

A template ships at [`esphome/secrets.yaml.example`](esphome/secrets.yaml.example).
Copy it and fill in real values before flashing or running the pipeline:

```bash
cp esphome/secrets.yaml.example esphome/secrets.yaml

# Generate fresh keys for your install
echo "api_password: \"$(openssl rand -base64 32)\""
echo "ota_password: \"$(openssl rand -hex 12)\""

# Edit esphome/secrets.yaml and paste in your WiFi creds + the keys above.
```

The four required keys are `wifi_ssid`, `wifi_password`, `api_password`,
`ota_password`. Both Core2 firmware and the Pi-side pipeline read this
file (the launcher `scripts/start.sh` auto-loads `api_password` into
`CORE2_NOISE_PSK` for you).

### Pipeline environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_MODEL` | `qwen2.5:1.5b` | Ollama model tag |
| `RADAR_IP` | `seeedstudio-mr60bha2-kit-12fd18.lan` | radar mDNS / IP |
| `PIPER_VOICE` | `en_US-amy-medium` | Piper voice id |
| `PIPER_DIR` | `~/piper/piper` | Piper voice files |
| `CORE2_HOST` | `edge-aiguard-core2.local` | Core2 mDNS / IP |
| `CORE2_NOISE_PSK` | (none) | Must equal `api_password` from `secrets.yaml` |

`CORE2_NOISE_PSK` is mandatory for `--core2` and `--core2-audio` modes.

---

## Flashing Core2 (one-time)

1. Connect Core2 USB-C → Pi USB-A; verify `/dev/ttyACM0` appears.
2. Add yourself to `dialout` if needed: `sudo usermod -aG dialout $USER`.
3. Build and flash:
   ```bash
   cd esphome
   ~/esphome-old/bin/esphome run core2.yaml --no-logs --device /dev/ttyACM0
   ```
   First build pulls the ESP32 toolchain (~10 min). Subsequent flashes
   use OTA — drop `--device /dev/ttyACM0`.

After flash, the device exposes:
- ESPHome native API at `edge-aiguard-core2.local:6053`
- Services: `update_status(status, hr, br)`, `start_listening`
- Display, touchscreen, mic, and speaker

---

## Running

The convenience launcher starts Ollama, warms the prompt cache, and
runs the pipeline:

```bash
scripts/start.sh                       # full Core2 mode (default)
scripts/start.sh --text --no-audio     # text-only mode (no mic, no speaker)
```

Or run the pipeline directly:

### Full Core2 voice terminal (mic + speaker on Core2)

```bash
export CORE2_NOISE_PSK="<api_password from secrets.yaml>"
python pipeline_pipecat.py --core2 --core2-audio
```


### STT smoke test on a WAV

```bash
python pipeline_pipecat.py --wav /path/to/16khz.wav
```

Feeds the file through sherpa-onnx STT and prints partials + finals.
No LLM, no TTS.

---

## Benchmarks

All harnesses run on the Pi with the production stack (`~/ollama` venv,
Ollama daemon up, ASR / Piper models in place). They write JSON
artifacts next to the script for reproducibility.

```bash
source ~/ollama/bin/activate
```

### `benchmarks/bench_three_turn.py` — Speculative prefill speedup (RQ2)

Measures KV-cache hit rate and TTFT savings on a realistic three-turn
health dialogue (`hello` → vitals query → health follow-up). Each turn
runs cold (direct chat) and warm (single-token `/api/generate` first,
mimicking `SpeculativePrefillProcessor`), with a unique sentinel per
trial to defeat carryover caching.

```bash
python benchmarks/bench_three_turn.py
```

Reports per-turn `prompt_eval_ms`, `eval_ms`, and end-to-end wall time
in cold vs. warm conditions, plus aggregate savings.

### `benchmarks/bench_streaming_tts.py` — Phrase-level TTS aggregation (RQ1)

A/B/C of three flush policies on the LLM → TTS path:

- **A. Whole-response** — `stream=False`, run Piper on the full reply.
- **B. Sentence-flush** — `stream=True`, flush at `.\,!\,?`.
- **C. Phrase-flush** — `stream=True`, also flush at `,\,:\,;` once
  ≥6 characters have buffered (the shipped aggregator).

```bash
python benchmarks/bench_streaming_tts.py [--n 3]
```

Reports median TTFA per query × policy and writes
`benchmarks/bench_streaming_tts.results.json`.

### `benchmarks/bench_sherpa.py` — Streaming ASR throughput

Measures RTF, memory, and per-chunk latency for streaming-zipformer-en
on the Pi 5.

```bash
python benchmarks/bench_sherpa.py [--chunk-ms 100]
```

Gate: RTF < 0.8 and resident memory < 600 MB.

### `benchmarks/bench_pipecat.py` — Pipecat + VAD smoke test

Confirms pipecat constructs and Silero-VAD load on Python 3.13 and
that VAD detects speech on a reference WAV.

```bash
python benchmarks/bench_pipecat.py
```

---

## Repository Layout

```
pipeline_pipecat.py          Main streaming full-duplex pipeline (sherpa + Ollama + Piper)
prompts.py                   Radar-state-conditioned system prompt builder
radar.py                     MR60BHA2 ESPHome client; rolling HR/BR window
speculation.py               SpeculativePrefillProcessor (KV-cache pre-warm via ASR partials)
tts_aggregator.py            Phrase-aware aggregator (commas, colons, semicolons)
core2_transport.py           Core2 voice_assistant <-> pipecat BaseTransport
esp_client.py                Core2 status push (Listening / Thinking / ...) + radar pulse
benchmarks/                  See "Benchmarks" section
tests/                       Unit tests (KV-cache, chat-template, speculation, speaker fix)
scripts/start.sh             Launcher: ollama + warmup + pipeline
esphome/core2.yaml           Core2 firmware (AXP2101 + display + mic + speaker + voice_assistant)
esphome/secrets.yaml.example Template for the required secrets file
esphome/secrets.yaml         gitignored; WiFi + API + OTA passwords
models/                      gitignored; sherpa-onnx ASR weights
```
