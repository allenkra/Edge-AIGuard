# Edge-AIGuard

Privacy-preserving, low-latency, full-duplex voice assistant that adapts
its responses to the user's real-time physiological state (heart rate /
breathing rate from a 60 GHz mmWave radar). 100% on-device — no cloud,
no internet required at runtime.

Hardware: **Raspberry Pi 5** (LLM/STT/TTS) + **M5Stack Core2 V1.1**
(mic + display) + **Seeed MR60BHA2** mmWave radar (HR/BR over WiFi).

For implementation history, design rationale, and decisions, see
[`Edge-AIGuard-Plan.md`](Edge-AIGuard-Plan.md).

---

## Architecture

```
┌── Core2 ──────────────┐                  ┌── Pi 5 ────────────────────────┐
│  mic (PDM) ──────────── voice_assistant ──→ Core2Transport.input        │
│                          (ESPHome native    └─ sherpa-onnx STT          │
│                           API + UDP audio)     ├─ RadarSystemPromptUpdater
│                                                │   ↑                     │
│  display (status,                              │   │ radar.get_state    │
│  HR, BR text)  ←────── update_status ────── esp_client / Core2Client    │
│                          service             ├─ OLLamaLLMService        │
│                                              ├─ PiperTTSService         │
│  speaker (NS4168)         (Future Work,      └─ Core2Transport.output ──┘
│  ▲ silent today           see Plan §2.4)                ↑
│  └ I2S0 conflict                                Pi-local also works
└───────────────────────┘                        (LocalAudioTransport)

  Seeed MR60BHA2 ──── ESPHome native API ──── radar.py (HR/BR/presence)
       (WiFi)
```

---

## Hardware Setup

| Device | Required | Notes |
|--------|----------|-------|
| Raspberry Pi 5 (8 GB) | yes | 27W USB-C PD power supply needed (`vcgencmd get_throttled` must read `0x0`) |
| MR60BHA2 mmWave radar | for radar features | Reflashes itself to ESPHome firmware; on the same WiFi as Pi |
| M5Stack Core2 V1.1 | for Core2 features | V1.1 has AXP2101 PMIC (V1.0 has AXP192 — schema differs) |
| USB-C cable Pi↔Core2 | for first flash only | After first USB flash, OTA updates over WiFi |
| USB mic / USB DAC / HDMI speaker | recommended | For LocalAudioTransport demo (Core2 speaker is parked, see Limitations) |

---

## Pi Setup

All paths assume `~` is `/home/hw3100`. Adjust if different.

### 1. Pi packages + Ollama model

```bash
# Pi base packages (install once)
sudo apt install -y nmap python3.13 python3.13-venv

# Python venv for the pipeline
python3 -m venv ~/ollama
source ~/ollama/bin/activate

pip install --upgrade pip
pip install pipecat-ai sherpa-onnx silero-vad aioesphomeapi loguru \
            scipy numpy pyserial flask matplotlib

# Ollama model (Pi-friendly 1.5B)
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

(See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)

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

```bash
# Side-installed in its own venv. Pin 2025.6.x because the AXP2101
# community component breaks on 2026.x (Wire.h / XPowersLib gone).
python3 -m venv ~/esphome-old
~/esphome-old/bin/pip install "esphome==2025.6.0"
```

---

## Configuration

### `esphome/secrets.yaml` (gitignored)

Generate fresh keys for your install:

```bash
echo "API key:";    openssl rand -base64 32
echo "OTA pass:";   openssl rand -hex 12
```

Then write to `esphome/secrets.yaml`:

```yaml
wifi_ssid:     "YourSSID"
wifi_password: "YourWiFiPassword"
api_password:  "<base64 from openssl rand>"
ota_password:  "<hex from openssl rand>"
```

### Pipeline env vars (export in shell or `.env`)

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_MODEL` | `qwen2.5:1.5b` | Ollama model tag |
| `RADAR_IP` | `seeedstudio-mr60bha2-kit-12fd18.lan` | mDNS hostname or IP of the MR60BHA2 |
| `PIPER_VOICE` | `en_US-amy-medium` | Piper voice id |
| `PIPER_DIR` | `~/piper/piper` | Where the Piper voice files live |
| `CORE2_HOST` | `edge-aiguard-core2.local` | mDNS hostname or IP of Core2 |
| `CORE2_NOISE_PSK` | (none) | API encryption key from `secrets.yaml`, base64 |

`CORE2_NOISE_PSK` MUST match the `api_password` you put in
`esphome/secrets.yaml` and flashed onto Core2. Without it, `--core2`
and `--core2-audio` modes refuse to start.

---

## Flashing Core2 (one-time)

1. Connect Core2 USB-C → Pi USB-A. Verify `/dev/ttyACM0` appears.
2. Verify your user is in the `dialout` group:
   `groups | grep dialout || sudo usermod -aG dialout $USER`
   (log out / back in if you had to add it)
3. Compile + flash via the side-installed ESPHome:
   ```bash
   cd esphome
   ~/esphome-old/bin/esphome run core2.yaml --no-logs --device /dev/ttyACM0
   ```
   First build pulls the ESP32 toolchain + axp2101 component (~10 min).
   Subsequent OTA flashes take seconds — drop `--device /dev/ttyACM0`
   to use OTA over WiFi.

After flash the device exposes:

- ESPHome native API at `edge-aiguard-core2.local:6053`
- two services: `update_status(status, hr, br)` and `start_listening`
- a touchscreen + binary sensor (FT6336 — see Limitations)

---

## Running

### Default — Pi-local mic + speaker, real radar

```bash
source ~/ollama/bin/activate
python pipeline_pipecat.py
```

### Headless dev (no mic, no speaker, fake radar)

```bash
python pipeline_pipecat.py --text --no-audio --no-radar
```

Type a query at the `>` prompt; TTS audio is appended to
`/tmp/edge_aiguard_last.raw`. Quit with `q`.

### Core2 status display (Pi mic + speaker locally)

```bash
export CORE2_NOISE_PSK="<api_password from secrets.yaml>"
python pipeline_pipecat.py --core2
```

Status text + HR/BR push to Core2 screen on every pipeline state change
plus once every 2 s.

### Core2 mic input + Core2 speaker output (full remote)

```bash
export CORE2_NOISE_PSK="<api_password from secrets.yaml>"
python pipeline_pipecat.py --core2 --core2-audio
```

Then trigger a session from a separate shell — the FT6336 touch IC
on this Core2 V1.1 isn't responsive, so we use a Pi-side service call:

```bash
~/ollama/bin/python -c "
import asyncio
from aioesphomeapi import APIClient
async def main():
    c = APIClient('edge-aiguard-core2.local', 6053, password='',
                  noise_psk='<api_password>')
    await c.connect(login=True)
    _, services = await c.list_entities_services()
    svc = next(s for s in services if s.name == 'start_listening')
    await c.execute_service(svc, {})
    await c.disconnect()
asyncio.run(main())
"
```

Speak after the script prints; voice_assistant's endpoint detector
ends the session on silence. **Note:** Core2 speaker output is currently
silent — see Limitations.

### STT smoke test on a WAV

```bash
python pipeline_pipecat.py --wav /path/to/16khz.wav
```

Feeds the file directly through sherpa-onnx STT and prints partials +
finals. No LLM, no TTS.

---

## Known Limitations

1. **Core2 speaker silent (parked).** The PDM mic and NS4168 speaker
   on Core2 V1.1 share `I2S0` (LRCK GPIO0). ESPHome `voice_assistant`
   doesn't stop the mic before starting the speaker, so the speaker
   task fails to claim the I2S peripheral and exits immediately.
   Pi sends TTS audio correctly; Core2 just can't play it. See
   [`Edge-AIGuard-Plan.md` §"Phase 2.4 最终状态"](Edge-AIGuard-Plan.md)
   for repro logs and Future Work options.

2. **Core2 touch IC unresponsive.** FT6336 doesn't appear on the I2C
   scan after the AXP2101 was wedged by an earlier (wrong) PMIC component.
   Worked around with a Pi-side `start_listening` ESPHome service.

3. **No echo cancellation.** Demos with the speaker would feed back
   into the mic; use headphones or a directional speaker.

4. **TTFT.** End-to-end latency is dominated by qwen2.5:1.5b prefill
   on Pi 5 (~3-5 s for short prompts after the first warm call).
   Acceptable for the demo, not for production.

---

## Repo Layout

```
pipeline_pipecat.py    main pipeline (pipecat + sherpa + Ollama + Piper)
pipeline.py            simpler Day-1 reference path (no streaming, no VAD)
prompts.py             radar-state-conditioned system prompt builder
radar.py               MR60BHA2 ESPHome client; sliding HR/BR window + fake mode
core2_transport.py     Core2 voice_assistant <-> pipecat BaseTransport
esp_client.py          Core2 status push (Listening/Thinking/...) + radar pulse
tools.py               deferred function-calling stub
bench_pipecat.py       latency benchmark
bench_sherpa.py        STT throughput benchmark
test_kv_cache.py       Ollama KV-cache reuse validation
esphome/core2.yaml     Core2 firmware (AXP2101 + display + mic + speaker + voice_assistant)
esphome/secrets.yaml   gitignored; WiFi + API + OTA passwords
models/                gitignored; sherpa-onnx ASR weights
Edge-AIGuard-Plan.md   3-day project plan with full implementation log
```

---

## Project Plan

The full design + day-by-day execution log is in
[`Edge-AIGuard-Plan.md`](Edge-AIGuard-Plan.md). Read it for:

- why qwen2.5:1.5b not 7B+ (TTFT trade-off)
- why pipecat over a synchronous loop (Day 1.5 evolution)
- the AXP192 → AXP2101 hardware mismatch saga (Phase 2 Path B)
- the 10 fixes that got Pi → Core2 audio working end-to-end
- optional Fine-tuning path on a 5070 Laptop GPU
