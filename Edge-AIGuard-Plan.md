# Edge-AIGuard: 三天开发完整实施计划

> **项目**: 哥伦比亚大学 Embedded AI 课程 Project
> **作者**: Hanlin Wang (hw3100), Yuxi Luo (yl6117)
> **目标**: 在 Pi 5 + M5Stack Core2 + MR60BHA2 Kit 上实现生理感知的本地语音助手
> **核心创新**: 根据用户实时心率/呼吸率自动调整 LLM 回应风格

---

## 目录

- [项目核心](#项目核心)
- [硬件清单](#硬件清单)
- [系统架构](#系统架构)
- [当前状态](#当前状态)
- [Day 1: Pi 端语音 Pipeline](#day-1-pi-端语音-pipeline)
- [Day 2: M5Stack Core2 集成](#day-2-m5stack-core2-集成)
- [Day 3: 端到端联调 + 评测 + 论文](#day-3-端到端联调--评测--论文)
- [完整文件清单](#完整文件清单)
- [风险与降级方案](#风险与降级方案)
- [可选: Fine-tuning 路径](#可选-fine-tuning-路径)
- [快速参考](#快速参考)

---

## 项目核心

### 论文 Contribution

**Multimodal Edge AI Assistant**：将 60GHz mmWave 雷达的生理信号融入本地 LLM 的对话上下文，实现**情绪感知**的人机交互。所有计算 100% 本地完成（隐私保护、无网络依赖）。

### 关键设计决策

1. **使用 1.5B-3B 小模型**而非 7B-8B：Pi 5 上 1.5B 模型可达 15+ tok/s，3B 约 6 tok/s，7B+ 实测仅 1-2 tok/s 无法满足实时性
2. **TTFA (Time-to-First-Audio) 而非 end-to-end 作为关键指标**：流式 TTS 让用户在 LLM 全部生成完成前就能听到响应
3. **Push-to-Talk + 唤醒词**双轨设计：先保证按钮触发可靠，唤醒词作为 stretch goal
4. **MR60BHA2 Kit 通过 WiFi 接入**：利用其内置 ESP32C6 + ESPHome 固件,免去 UART 协议解析

---

## 硬件清单

| 设备 | 型号 | 用途 | 状态 |
|------|------|------|------|
| 主机 | **Raspberry Pi 5 (8GB)** | LLM/ASR/TTS 推理大脑 | ✅ 已就绪 |
| 语音终端 | **M5Stack Core2 (ESP32)** | 唤醒词 + 麦克风 + 显示 | 待配置 |
| 雷达 | **MR60BHA2 Sensor Kit** (含 XIAO ESP32C6) | 心率/呼吸/存在感知,WiFi 接入 | 待配置 |
| 麦克风 | Core2 内置 SPM1423 (PDM) | 语音输入 | 随 Core2 |
| 喇叭 | Pi 5 接 USB DAC 或 3.5mm | 语音输出 | 待确认 |
| 电源 | **27W USB-C PD** | Pi 5 供电 | ⚠️ 必须确认 |
| 散热 | 风扇 + 散热片 | 防止降频 | 推荐 |
| 存储 | 32GB SD 卡 | 系统盘 | ⚠️ 偏小,注意空间 |

### 硬件采购建议（如缺）

- 27W USB-C PD 电源（Pi 官方或第三方支持 5V/5A）
- 64GB+ A2 SD 卡（如果空间持续紧张）
- USB DAC 或 USB 喇叭（音质比 Pi 3.5mm 接口好很多）

---

## 系统架构

```
                       WiFi 局域网 (192.168.x.x)
   ┌────────────────┬──────────────────┬──────────────────┐
   │                │                  │                  │
┌──┴──────┐   ┌────┴─────┐      ┌─────┴──────┐      ┌────┴───┐
│  Pi 5   │   │ M5Stack  │      │ MR60BHA2   │      │   PC   │
│ (Brain) │   │  Core2   │      │    Kit     │      │ (开发) │
└──┬──────┘   └────┬─────┘      └─────┬──────┘      └────────┘
   │               │                   │
   │           麦克风                  │
   │           显示屏                  │
   │           按键 / 唤醒词            │
   │                                  │
   ├─→ Whisper.cpp (ASR)              │
   ├─→ Ollama qwen2.5:1.5b (LLM)      │
   ├─→ Piper (TTS)                    │
   ├─→ Radar Subscriber ←─────────────┘
   └─→ Prompt Composer (融合状态)
```

### 数据流

```
[用户] ──说话──> [Core2 唤醒/按键] ──HTTP/Wyoming──> [Pi: ASR]
                                                         │
                                                         ▼
[Radar] ──ESPHome API──> [Pi: State Classifier] ──> [Pi: Prompt Composer]
                                                         │
                                                         ▼
                                                  [Pi: Ollama LLM]
                                                         │
                                                         ▼
                                              [Pi: Streaming TTS]
                                                         │
                                                         ▼
                                                    [喇叭播放]
```

---

## 当前状态

### ✅ 已完成

**初始环境 (Day 0)**
- [x] Pi 5 系统安装 (Raspberry Pi OS Bookworm)
- [x] SSH 远程访问 (主机名 `hw3100`)
- [x] Ollama v0.18.2 安装到 `/usr/bin/ollama`
- [x] Swap 扩到 6 GB (zram 2GB + /var/swap 4GB)
- [x] `/etc/fstab` 加入 `/var/swap none swap sw 0 0`, `/etc/rpi/swap.conf.d/fixed.conf` 写入 `[File] FixedSizeMiB=4096`
- [x] Python 虚拟环境 `~/ollama` 已创建

**模型 (Day 0 + Day 1)**
- [x] 已装模型 (位于 `~/.ollama/models/`):
  - llama3.2:1b (1.3 GB)
  - llama3.2:3b (2.0 GB)
  - gemma2:2b (1.6 GB)
  - phi3.5:3.8b (2.2 GB)
  - **qwen2.5:1.5b (986 MB)** ← Day 1.1 新增, 主要推理模型
- [x] llava-phi3:3.8b 删除 (释放 2.9 GB,本项目无需视觉)
- [x] 1B/3B/2B 模型基线性能数据已采集 (见下表)

**Day 1 实施 (2026-04-29)**
- [x] Day 1.1.1 电源/温度健康 (`throttled=0x0`, 52.7°C)
- [x] Day 1.1.2 Python 依赖装好 (faster-whisper 1.2.1, ctranslate2 4.7.1, aioesphomeapi 44.22, sounddevice, scipy, flask, matplotlib)
- [x] Day 1.1.3 Piper TTS + Amy 语音模型 (real-time factor 0.13, **7.5× realtime**)
- [x] Day 1.1.5 项目目录建立 (后改为 `~/Edge-AIGuard/`,与 git 仓库一致)
- [x] Day 1.3.1 `radar.py` 编写 (含 fake_mode, 待真机验证)
- [x] Day 1.3.2 `prompts.py` 编写 (HR/BR 实时注入)
- [x] Day 1.3.3 `pipeline.py` 编写 (含 `--text/--no-audio/--no-radar` headless fallback)
- [x] Day 1.3.5 `tools.py` 编写 (tool calling 接口预留, `ENABLE_TOOLS=1` 启用)
- [x] **冒烟测试**: text mode + fake radar 验证 state-conditioned response 工作 (normal "The capital of France is Paris." → stressed "Paris.")
- [x] git 仓库初始化 + push 到 `git@github.com:allenkra/Edge-AIGuard.git`

**Day 1.2 雷达真机验证 (2026-04-29)**
- [x] MR60BHA2 Kit WiFi 配通 (FortenLLLord), nmap 扫到 IP `192.168.1.140`
- [x] mDNS 名: `seeedstudio-mr60bha2-kit-12fd18.lan` (路由用 `.lan` 后缀)
- [x] ESPHome API 实体发现, 7 个实体, 关键三个: `Real-time heart rate` (bpm), `Real-time respiratory rate`, `Person Information` (binary)
- [x] 修复 `radar.py`: aioesphomeapi 44.x `subscribe_states` 不再是 coroutine; entity matcher 改 `respir`/`person` 词根
- [x] 修复 presence 推导: 从 HR 数据 recency 反推 (binary sensor 只在状态切换时发,初始状态丢失)
- [x] 修复 `_classify`: 把 `exercising` (HR≥110) 排到 `stressed` 前面
- [x] 真机数据验证: HR 86-117 bpm, BR 6-20/min, distance 40-120cm 实时流入
- [x] **完整集成测试** (text + 真雷达 + 流式 LLM):
  - "How am I doing?" (presence=False 时) → "I can't tell from your current state..." 不编数字 ✅
  - "What is my heart rate?" (HR=86) → "Your heart rate is 86 bpm." 精确报数 ✅

### 📊 已采集的基线数据

**Day 0 距离题基线** (参考真实距离: ~11650 km):

| 模型 | 参数量 | Eval Rate | 距离题答案 | 准确性 |
|------|--------|-----------|-----------|--------|
| llama3.2:1b | 1B | 8.3 tok/s | 11960 km | 接近 + 编驾驶距离 |
| llama3.2:3b | 3B | 5.8 tok/s | 7841 km | 偏低 -3800 km |
| gemma2:2b | 2.6B | 6.8 tok/s | 8400 km | 偏低 -3250 km |
| phi3.5:3.8b | 3.8B | 4.77 tok/s | 12309 km | 最准 +660 km |

**Day 1 冒烟测试** (qwen2.5:1.5b, system prompt ~150 token):

| 状态 | Query | Response | TTFT | LLM total |
|------|-------|----------|------|-----------|
| normal (HR=72) | "What is the capital of France?" | "The capital of France is Paris." | 8.41s | 9.76s |
| stressed (HR=95) | (same) | "Paris." | 7.44s | 8.12s |

→ State-conditioned response adaptation **已验证**, 但 TTFT 远超 plan 目标 (sub-3s),需优化 system prompt 长度。

### ⚠️ 已知问题

1. **跑 phi3.5:3.8b 时触发红灯欠压** → 必须使用 27W PD 电源 (Day 1 实测 0x0,健康)
2. **SD 卡空间** → Day 1 删 llava-phi3 后剩 6.0 GB,暂时宽松,装完所有依赖+模型后约剩 5 GB
3. **`ollama` 目录命名冲突** → 虚拟环境也叫 `~/ollama`,不影响功能但注意区分
4. **TTFT 8s 偏慢 (Day 1 实测)** → plan 目标 sub-3s 未达成,瓶颈是 system prompt prefill (~150 token)。优化方向: 精简 prompt / 试 0.5b 模型 / 预热模型
5. **没有 USB 麦克风 + 没有音频输出** → Day 1 用 `--text` + `--no-audio` headless 模式开发,Day 2 由 Core2 提供麦,音频输出待补 (HDMI 音箱 / USB DAC)
6. **MR60BHA2 `Person Information` binary sensor 只在状态切换时发** → 订阅时刻已是 True 就再也收不到事件; 已通过从 HR 新鲜度反推 presence 解决
7. **aioesphomeapi 44.x 与 plan 代码不兼容** → `subscribe_states` 不再是 coroutine, `await` 会报 TypeError; 已修
8. **DHCP 分配的 IP 可能变** → 已切换到 mDNS hostname `seeedstudio-mr60bha2-kit-12fd18.lan` (固件烧死,永远有效); 路由器加 DHCP 保留作为兜底 (待用户操作)

### ⏳ 待完成

**Day 1 收尾**
- [ ] TTFT 优化到 sub-3s (当前 5-8s, 取决于是否冷启动)

**Day 2**
- [ ] 配置 M5Stack Core2 (ESPHome)
- [ ] 麦克风录音 + HTTP 上传到 Pi
- [ ] Pi 端状态推送到 Core2 显示

**Day 3**
- [ ] 端到端 4 个场景测试
- [ ] `eval.py` + 评测数据
- [ ] 三张图 + Demo 视频
- [ ] 论文实验数据填入

---

## Day 1: Pi 端语音 Pipeline

**目标**: Pi 单机能完整跑通"语音输入 → 智能回应 → 语音输出",并根据雷达状态调整风格。Day 1 结束时即使没有 Core2,也有完整可演示的命令行版语音助手。

### Day 1.1 - 环境准备 (上午)

#### 1.1.1 解决电源 (必做)

```bash
# 检查当前电源/温度状态
vcgencmd get_throttled        # 期望 0x0
vcgencmd measure_volts         # 期望 ~0.85V
vcgencmd measure_temp          # 期望 < 70°C
dmesg | grep -i 'under-volt' | tail -5  # 期望空输出
```

如 `get_throttled` 不为 `0x0`,**必须**换 27W PD 电源后再继续。

#### 1.1.2 安装 Python 依赖

```bash
# 激活虚拟环境
source ~/ollama/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装核心依赖
pip install \
  faster-whisper \
  sounddevice \
  numpy \
  requests \
  pyserial \
  scipy \
  flask \
  aioesphomeapi \
  matplotlib

# 拉一个更适合的模型 (中英都好,1.5B 速度最快)
ollama serve > ~/ollama.log 2>&1 &
sleep 2
ollama pull qwen2.5:1.5b
```

#### 1.1.3 下载 Piper TTS

```bash
mkdir -p ~/piper && cd ~/piper

# Piper 二进制 (ARM64)
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz
rm piper_linux_aarch64.tar.gz

# 语音模型 (Amy - 美式女声)
cd piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# 测试 TTS
echo "Hello world, this is Piper text to speech." | \
  ./piper --model en_US-amy-medium.onnx --output_file /tmp/test.wav
aplay /tmp/test.wav
```

#### 1.1.4 测试音频设备

```bash
# 列出所有录音设备
arecord -l

# 列出所有播放设备
aplay -l

# 测试录音 (3 秒)
arecord -d 3 -f S16_LE -r 16000 -c 1 /tmp/test_rec.wav
aplay /tmp/test_rec.wav
```

如果没有 USB 麦,Day 1 可以临时用文字输入测试 LLM+TTS 链路,Day 2 用 Core2 麦克风。

#### 1.1.5 创建项目目录

```bash
mkdir -p ~/Edge-AIGuard
cd ~/Edge-AIGuard
```

### Day 1.2 - 雷达 Kit 配置 (中午)

#### 1.2.1 给 MR60BHA2 Kit 通电

1. 用 USB-C 数据线给 Kit 通电
2. 第一次开机会启动 WiFi 配网热点 (类似 `mr60bha2-xxxxxx`)
3. 用手机连这个热点,会自动跳转配置页 (或访问 `192.168.4.1`)
4. 输入家里 WiFi 名称和密码,保存
5. Kit 重启后自动连家庭 WiFi

#### 1.2.2 找到 Kit 的 IP

```bash
# 方法 1: 扫描局域网
sudo apt install nmap -y
sudo nmap -sn 192.168.1.0/24    # 网段按实际改

# 方法 2: 查 mDNS (ESPHome 设备一般有 .local 域名)
ping mr60bha2.local             # 或 kit 的具体名字
```

记录 Kit IP, 假设为 `192.168.1.100`。

#### 1.2.3 验证 ESPHome API 可达

```python
# /tmp/test_radar.py
import asyncio
from aioesphomeapi import APIClient

async def main():
    client = APIClient("192.168.1.100", 6053, "")  # 改成你的 IP
    await client.connect(login=True)
    entities, _ = await client.list_entities_services()
    print(f"Found {len(entities)} entities:")
    for e in entities:
        name = getattr(e, 'name', '?')
        obj_id = getattr(e, 'object_id', '?')
        print(f"  - {name} (id={obj_id}, key={e.key})")
    await client.disconnect()

asyncio.run(main())
```

```bash
python /tmp/test_radar.py
```

应输出实体列表,包含 `Heart Rate`, `Breath Rate`, `Presence` 等。**记录这些实体名,Day 1 下午写代码要用**。

### Day 1.3 - 核心代码 (下午)

#### 1.3.1 雷达驱动 `radar.py`

创建文件 `~/Edge-AIGuard/radar.py`:

```python
"""
MR60BHA2 Sensor Kit - 通过 ESPHome API 读取
"""
import asyncio
import threading
import time
from collections import deque
from aioesphomeapi import APIClient, EntityState


class RadarReader:
    def __init__(self, host, password="", port=6053):
        self.host = host
        self.password = password
        self.port = port

        self.hr_window = deque(maxlen=30)
        self.br_window = deque(maxlen=30)
        self.presence = False
        self.fake_mode = None  # 调试用,设 "normal"/"stressed"/"relaxed"/"exercising"

        self._entity_map = {}
        self._latest = {}

        self.running = True
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_async, daemon=True)
        self.thread.start()

    def _run_async(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self):
        while self.running:
            try:
                await self._connect()
            except Exception as e:
                print(f"⚠️  Radar connection error: {e}, retrying in 5s")
                await asyncio.sleep(5)

    async def _connect(self):
        client = APIClient(self.host, self.port, self.password)
        await client.connect(login=True)
        print(f"✅ Connected to radar at {self.host}")

        entities, _ = await client.list_entities_services()
        for e in entities:
            name = (getattr(e, 'name', '') or '').lower()
            obj_id = (getattr(e, 'object_id', '') or '').lower()
            self._entity_map[e.key] = (name, obj_id)

        def on_state(state: EntityState):
            if state.key not in self._entity_map:
                return
            name, obj_id = self._entity_map[state.key]
            value = getattr(state, 'state', None)
            if value is None:
                return

            search = f"{name} {obj_id}"
            now = time.time()

            if any(k in search for k in ['heart', 'hr', 'pulse', 'bpm']):
                if isinstance(value, (int, float)) and 30 < value < 200:
                    self.hr_window.append((now, float(value)))
            elif any(k in search for k in ['breath', 'br', 'respiration']):
                if isinstance(value, (int, float)) and 5 < value < 40:
                    self.br_window.append((now, float(value)))
            elif any(k in search for k in ['presence', 'occupancy', 'detected']):
                self.presence = bool(value)

        await client.subscribe_states(on_state)

        while self.running:
            await asyncio.sleep(1)

    def get_state(self):
        # Fake mode (用于测试)
        if self.fake_mode:
            presets = {
                "normal":     (72, 16),
                "stressed":   (95, 22),
                "relaxed":    (60, 12),
                "exercising": (130, 25),
            }
            hr, br = presets.get(self.fake_mode, (72, 16))
            return {
                "hr": hr, "br": br, "presence": True,
                "category": self._classify(hr, br),
                "source": "fake"
            }

        if not self.hr_window:
            return {
                "hr": None, "br": None,
                "presence": self.presence,
                "category": "unknown",
                "source": "real"
            }

        hr = sum(v for _, v in self.hr_window) / len(self.hr_window)
        br = (sum(v for _, v in self.br_window) / len(self.br_window)
              if self.br_window else 0)
        return {
            "hr": round(hr, 1),
            "br": round(br, 1),
            "presence": self.presence,
            "category": self._classify(hr, br),
            "source": "real"
        }

    def _classify(self, hr, br):
        if not self.presence and not self.fake_mode:
            return "no_user"
        if hr > 90 and br > 20:
            return "stressed"
        elif hr > 110:
            return "exercising"
        elif hr < 65 and br < 14:
            return "relaxed"
        else:
            return "normal"

    def stop(self):
        self.running = False


if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.100"
    radar = RadarReader(host=host)
    try:
        for _ in range(60):
            print(radar.get_state())
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        radar.stop()
```

#### 1.3.2 Prompt 模板 `prompts.py`

创建文件 `~/Edge-AIGuard/prompts.py`:

```python
"""
基于雷达状态的 system prompt 模板
HR/BR 数值在每轮对话开始前实时注入 (radar.get_state() 是 O(1))
"""

BASE_PERSONA = (
    "You are a helpful voice assistant running locally on a Raspberry Pi. "
    "You have access to the user's real-time physiological signals from a "
    "60GHz mmWave radar (heart rate, breathing rate, presence)."
)

# 状态相关的风格指令 (不再包含数字, 数字在 build_system_prompt 里拼接)
STATE_INSTRUCTIONS = {
    "normal": (
        "The user appears calm and relaxed. "
        "Keep answers brief and natural, under 2 sentences."
    ),
    "stressed": (
        "The user's elevated heart rate and breathing suggest stress or anxiety. "
        "Speak gently and slowly. Keep answers very brief, under 2 sentences. "
        "If appropriate, you may gently suggest taking a deep breath. "
        "Do NOT proactively mention specific numbers or cause alarm — "
        "but if the user explicitly asks about their vitals, you may share them calmly."
    ),
    "relaxed": (
        "The user appears very relaxed. Match their calm, unhurried energy. "
        "Up to 3 sentences is fine."
    ),
    "exercising": (
        "The user appears physically active. "
        "Give one-sentence answers only. Be direct and clear."
    ),
    "no_user": (
        "No user is currently detected. Respond briefly."
    ),
    "unknown": (
        "Keep answers brief and natural, under 2 sentences."
    ),
}

READING_RULE = (
    "You may share specific HR/BR values if the user explicitly asks "
    "(e.g., 'what is my heart rate?'). Do not bring them up unprompted "
    "unless directly relevant to the user's question."
)


def build_system_prompt(radar_state):
    category = radar_state.get("category", "unknown")
    style = STATE_INSTRUCTIONS.get(category, STATE_INSTRUCTIONS["unknown"])

    hr = radar_state.get("hr")
    br = radar_state.get("br")
    presence = radar_state.get("presence", False)

    if hr is not None and br is not None:
        readings = (
            f"Current readings: heart rate {hr:.0f} bpm, "
            f"breathing rate {br:.0f}/min, "
            f"presence={'yes' if presence else 'no'}, "
            f"inferred state: {category}."
        )
    else:
        readings = f"Physiological readings unavailable. State: {category}."

    return f"{BASE_PERSONA}\n\n{style}\n\n{readings}\n\n{READING_RULE}"
```

#### 1.3.3 主控 `pipeline.py`

创建文件 `~/Edge-AIGuard/pipeline.py`:

```python
"""
Edge-AIGuard - Day 1 Pipeline
ASR (Whisper) -> LLM (Ollama) -> TTS (Piper)
按 ENTER 录音 5 秒,跑完整链路
"""
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import requests
import subprocess
import time
import json
import re
import os
import sys

from radar import RadarReader
from prompts import build_system_prompt

# === 配置 ===
WHISPER_MODEL = "base.en"           # tiny.en 更快但精度略低
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:1.5b"
PIPER_BIN = os.path.expanduser("~/piper/piper/piper")
PIPER_MODEL = os.path.expanduser("~/piper/piper/en_US-amy-medium.onnx")
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
RADAR_IP = os.environ.get("RADAR_IP", "192.168.1.100")  # 改成你的

# === 加载组件 ===
print("Loading Whisper...")
asr_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("Connecting to radar...")
radar = RadarReader(host=RADAR_IP)


def record(duration=RECORD_SECONDS):
    print(f"🎤 Recording {duration}s...")
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def transcribe(audio):
    t0 = time.time()
    segments, _ = asr_model.transcribe(audio, beam_size=1, language="en")
    text = " ".join([s.text for s in segments]).strip()
    dt = time.time() - t0
    print(f"📝 ASR ({dt:.2f}s): {text}")
    return text, dt


def ask_llm_streaming(prompt, system_prompt, on_sentence):
    t0 = time.time()
    first_token_time = None

    r = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {"num_predict": 80, "temperature": 0.7}
    }, stream=True, timeout=60)

    buffer = ""
    full = ""
    sentence_re = re.compile(r'([.!?])\s')

    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line).get("response", "")
        if chunk and first_token_time is None:
            first_token_time = time.time() - t0
            print(f"🤖 First token at {first_token_time:.2f}s")
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

    total_time = time.time() - t0
    print(f"🤖 LLM total: {total_time:.2f}s")
    return full, first_token_time, total_time


def speak(text):
    t0 = time.time()
    cmd = [PIPER_BIN, "--model", PIPER_MODEL, "--output_raw"]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    raw, _ = proc.communicate(input=text.encode())

    aplay = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    aplay.communicate(input=raw)
    print(f"🔊 TTS '{text[:30]}...' ({time.time()-t0:.2f}s)")


def main():
    print("=" * 50)
    print("Edge-AIGuard Day 1 - Console Mode")
    print("Commands: ENTER=talk, m=cycle fake mode, s=show state, q=quit")
    print("=" * 50)

    fake_modes = ["normal", "stressed", "relaxed", "exercising", None]
    fake_idx = 0

    try:
        while True:
            cmd = input("\n> ").strip().lower()

            if cmd == 'q':
                break

            if cmd == 's':
                print(f"📊 State: {radar.get_state()}")
                continue

            if cmd == 'm':
                fake_idx = (fake_idx + 1) % len(fake_modes)
                radar.fake_mode = fake_modes[fake_idx]
                mode_name = fake_modes[fake_idx] or "REAL (no fake)"
                print(f"🎛️  Fake mode: {mode_name}")
                continue

            # 默认: 录音对话
            audio = record()
            text, _ = transcribe(audio)
            if not text:
                print("(empty, skip)")
                continue

            state = radar.get_state()
            print(f"📊 State: {state}")
            system = build_system_prompt(state)
            ask_llm_streaming(text, system, on_sentence=speak)

    finally:
        radar.stop()
        print("Bye.")


if __name__ == "__main__":
    main()
```

#### 1.3.4 第一次运行

```bash
cd ~/Edge-AIGuard
source ~/ollama/bin/activate
ollama serve > ~/ollama.log 2>&1 &
sleep 2

# 把 RADAR_IP 改成你的
RADAR_IP=192.168.1.100 python pipeline.py
```

**测试流程**:
1. 按 ENTER, 对着麦说 "What is the capital of France?"
2. 等几秒,听 Pi 回应
3. 按 'm' 切换到 stressed 模式
4. 再按 ENTER 问同样问题,听到温柔/简短的版本
5. 按 'q' 退出

#### 1.3.5 Tool Calling 接口 (预留, Day 1 不启用)

**设计原则**: HR/BR 走 system prompt 注入(廉价、永远可用、不破坏流式 TTFA)。Tool calling 留给真正"稀疏 + 昂贵"的能力,作为 agentic demo。

主路径 `pipeline.py` 继续用 `/api/generate` 流式推理,不开 tools。Tool 路径作为独立模块,Day 3 评测主线后视时间决定是否激活做额外 demo。

创建文件 `~/Edge-AIGuard/tools.py`:

```python
"""
Tool calling 接口 - 为 agentic 能力预留
注意: tool calling 必须用 /api/chat (非流式), 会牺牲 TTFA
仅在用户明确触发 agentic 命令时调用,主对话路径不启用
"""
import json
import time
import requests

# Ollama / OpenAI 兼容的 tool schema
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time on the device.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vital_signs",
            "description": (
                "Get the user's latest heart rate (bpm) and breathing rate "
                "(breaths/min) from the mmWave radar. Use only when the user "
                "explicitly asks about their vitals or current physical state."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # 后续可加: get_weather, set_timer, control_light...
]


def dispatch_tool(name, args, radar):
    """根据 tool name 路由到实际函数, 返回 JSON 字符串供 LLM 续写"""
    if name == "get_current_time":
        return json.dumps({"local_time": time.strftime("%H:%M:%S, %Y-%m-%d")})
    elif name == "get_vital_signs":
        s = radar.get_state()
        return json.dumps({
            "heart_rate_bpm": s.get("hr"),
            "breathing_rate_per_min": s.get("br"),
            "presence_detected": s.get("presence"),
            "inferred_state": s.get("category"),
        })
    return json.dumps({"error": f"unknown tool: {name}"})


def chat_with_tools(messages, model, ollama_chat_url, radar, max_iters=3):
    """
    /api/chat + tools loop. 非流式 (tool calling + streaming 在小模型上不稳)
    返回最终的 assistant content 字符串
    """
    for _ in range(max_iters):
        r = requests.post(ollama_chat_url, json={
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        }, timeout=60)
        msg = r.json().get("message", {})
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content", "")

        for call in tool_calls:
            fn = call["function"]["name"]
            args = call["function"].get("arguments", {}) or {}
            result = dispatch_tool(fn, args, radar)
            messages.append({"role": "tool", "name": fn, "content": result})

    return messages[-1].get("content", "(tool loop max iters exceeded)")
```

**集成方式 (Day 3 可选启用)**: 在 `pipeline.py` 加一个轻量意图判断,前缀触发即可,不增加默认路径延迟:

```python
# 在 main() 主循环里,拿到 ASR text 后:
TOOL_KEYWORDS = ["what time", "my heart rate", "my breathing", "my vitals"]
if any(kw in text.lower() for kw in TOOL_KEYWORDS):
    from tools import chat_with_tools
    OLLAMA_CHAT = OLLAMA_URL.replace("/generate", "/chat")
    messages = [
        {"role": "system", "content": build_system_prompt(state)},
        {"role": "user", "content": text},
    ]
    answer = chat_with_tools(messages, LLM_MODEL, OLLAMA_CHAT, radar)
    speak(answer)
else:
    # 默认路径: 流式生成, 保 TTFA
    ask_llm_streaming(text, build_system_prompt(state), on_sentence=speak)
```

**论文里的措辞**: "We expose two parallel inference paths: a streaming path optimized for TTFA, and a tool-calling path for agentic queries. The system routes based on lightweight keyword intent detection, preserving low-latency response for the common case."

**Day 3 决策点**: 评测主表跑完后,如果 qwen2.5:1.5b 的 tool call 准确率 > 80% (人工跑 10 条 tool 触发样本),就在 demo 视频里加一段;低于 80% 就只在论文 Future Work 提及。

### Day 1.4 - 优化 + Deliverable (晚上)

#### 1.4.1 性能调优

如果 ASR 太慢,试试 `tiny.en` 模型:
```python
WHISPER_MODEL = "tiny.en"   # 快 2x, 精度略低
```

如果 LLM 太慢,确认监控:
```bash
# 另开终端监控
watch -n 1 'echo "TEMP:"; vcgencmd measure_temp; echo "THROTTLE:"; vcgencmd get_throttled; echo "MEM:"; free -h | head -2'
```

如果温度持续 > 80°C,跑模型时频率会被限制,需要散热。

#### 1.4.2 Day 1 Deliverable Checklist

- [ ] 命令行版语音助手能完整跑通
- [ ] 雷达数据能稳定读到 (或 fake mode 能切换)
- [ ] 不同 fake mode 下回答风格明显不同
- [ ] 记录基线延迟数据 (ASR、LLM first token、TTS、TTFA)
- [ ] 打 ASR/LLM/TTS 各阶段的耗时日志

---

## Day 2: M5Stack Core2 集成

**目标**: 用 M5Stack Core2 做唤醒/录音/显示,实现真正的"无屏远场"语音助手。

### Day 2.1 - 工具链准备 (上午)

#### 2.1.1 PC 上装驱动

将 M5Stack Core2 用 USB-C 接 **Windows PC** (开发都在 PC 上做):

1. 打开设备管理器,看 "端口 (COM 和 LPT)"
2. 如果没识别,从 [docs.m5stack.com/en/download](https://docs.m5stack.com/en/download) 下载 **CH9102 驱动**安装
3. 记录 COM 端口号 (如 `COM3`)

#### 2.1.2 Arduino IDE 验证硬件 (30 分钟)

**目的**: 验证 PC 能成功烧录 Core2,排除驱动问题。

1. 下载安装 [Arduino IDE 2.x](https://www.arduino.cc/en/software)
2. **File → Preferences**, "Additional Board Manager URLs" 填:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
3. **Boards Manager** 搜 `esp32`, 安装 (5-10 分钟,要下载 200MB+)
4. **Library Manager** 搜 `M5Core2`, 安装
5. **Tools → Board → ESP32 Arduino → M5Stack-Core2**
6. **Tools → Port → COMx** (你看到的)
7. **File → Examples → M5Core2 → Basics → HelloWorld**
8. 点 Upload 烧录

成功标志: Core2 屏幕显示 "Hello World"。

烧录失败时按住 Core2 红色按钮再点 Upload。

#### 2.1.3 装 ESPHome

```powershell
# Windows PowerShell
python -m venv esphome-env
.\esphome-env\Scripts\Activate.ps1
pip install esphome
esphome version
```

或 Docker:
```powershell
docker run --rm -v "${PWD}:/config" -p 6052:6052 -it ghcr.io/esphome/esphome dashboard /config
```

### Day 2.2 - 第一个 ESPHome 配置 (中午)

#### 2.2.1 项目目录

```powershell
mkdir edge-aiguard-esp
cd edge-aiguard-esp
```

#### 2.2.2 `secrets.yaml`

```yaml
wifi_ssid: "你的WiFi名"
wifi_password: "WiFi密码"
api_password: "edge-aiguard-2026"
ota_password: "edge-aiguard-2026"
pi_ip: "192.168.1.50"     # 改成 Pi 5 的实际 IP
```

#### 2.2.3 基础配置 `core2.yaml` (先验证显示和按键)

```yaml
esphome:
  name: edge-aiguard
  friendly_name: Edge-AIGuard

esp32:
  board: m5stack-core2
  framework:
    type: arduino

logger:
  level: INFO

api:
  encryption:
    key: !secret api_password

ota:
  - platform: esphome
    password: !secret ota_password

wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password
  ap:
    ssid: "EdgeAIGuard-Setup"
    password: "12345678"

# Core2 屏幕 SPI
spi:
  clk_pin: GPIO18
  mosi_pin: GPIO23
  miso_pin: GPIO38

display:
  - platform: ili9xxx
    model: M5STACK
    cs_pin: GPIO5
    dc_pin: GPIO15
    update_interval: 1s
    lambda: |-
      it.fill(Color::BLACK);
      it.printf(160, 30, id(font_title), Color(0xFF, 0xFF, 0xFF),
                TextAlign::CENTER, "Edge-AIGuard");
      it.printf(160, 80, id(font_main), Color(0x00, 0xFF, 0x00),
                TextAlign::CENTER, "Status: %s", id(status).c_str());
      it.printf(20, 140, id(font_main), Color(0xFF, 0xAA, 0x00),
                TextAlign::TOP_LEFT, "HR: %.0f bpm", id(hr_value));
      it.printf(20, 175, id(font_main), Color(0x00, 0xAA, 0xFF),
                TextAlign::TOP_LEFT, "BR: %.0f /min", id(br_value));
      it.printf(160, 220, id(font_small), Color(0x88, 0x88, 0x88),
                TextAlign::CENTER, "Press button to talk");

font:
  - file: "gfonts://Roboto"
    id: font_title
    size: 28
  - file: "gfonts://Roboto"
    id: font_main
    size: 20
  - file: "gfonts://Roboto"
    id: font_small
    size: 14

globals:
  - id: status
    type: std::string
    restore_value: no
    initial_value: '"Ready"'
  - id: hr_value
    type: float
    restore_value: no
    initial_value: '0.0'
  - id: br_value
    type: float
    restore_value: no
    initial_value: '0.0'

# 触摸按键 A (屏幕下方左侧触摸点)
binary_sensor:
  - platform: gpio
    pin:
      number: GPIO39
      inverted: true
    name: "Button A"
    on_press:
      then:
        - lambda: |-
            id(status) = "Listening...";
        - logger.log: "Button A pressed"
```

#### 2.2.4 烧录

```powershell
esphome run core2.yaml
```

第一次编译 5-15 分钟。完成后看到屏幕显示 "Edge-AIGuard / Status: Ready / HR: 0 / BR: 0"。

按屏幕下方左侧触摸点,日志看到 "Button A pressed"。

### Day 2.3 - 麦克风 + 音频上传 (下午)

#### 2.3.1 在 `core2.yaml` 加麦克风部分

```yaml
# I2S 麦克风 (M5Stack Core2 内置 SPM1423 PDM)
i2s_audio:
  - id: i2s_input
    i2s_lrclk_pin: GPIO0

microphone:
  - platform: i2s_audio
    id: m5_mic
    i2s_din_pin: GPIO34
    adc_type: external
    pdm: true
    sample_rate: 16000
    bits_per_sample: 16bit

http_request:
  verify_ssl: false

# 修改按键 A 行为: 录音 + 上传
binary_sensor:
  - platform: gpio
    pin:
      number: GPIO39
      inverted: true
    name: "Button A"
    on_press:
      then:
        - lambda: |-
            id(status) = "Recording...";
        - microphone.capture:
            id: m5_mic
            duration: 3s
            on_data:
              then:
                - http_request.post:
                    url: !lambda |-
                      return ((std::string)"http://" + "192.168.1.50" + ":8000/audio").c_str();
                    body: !lambda |-
                      return std::string((char*)data, len);
                    headers:
                      Content-Type: "audio/raw"
        - lambda: |-
            id(status) = "Processing...";
```

> 注: ESPHome 不同版本的 microphone API 可能略有差异,如报错按错误信息查文档。

#### 2.3.2 Pi 端接收音频

修改 `pipeline.py`,加 Flask HTTP 服务器:

```python
# 在 pipeline.py 顶部加
from flask import Flask, request, jsonify
import threading
import io

app = Flask(__name__)
pending_audio = []

@app.route('/audio', methods=['POST'])
def receive_audio():
    raw = request.data
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    pending_audio.append(audio)
    return jsonify({"status": "received", "samples": len(audio)})

def run_server():
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

# 在 main() 之前启动
threading.Thread(target=run_server, daemon=True).start()
```

修改 `main()` 函数,改为轮询模式:

```python
def main():
    print("=" * 50)
    print("Edge-AIGuard - Voice Assistant Mode")
    print("Press button on Core2 to talk")
    print("Console: m=fake mode, s=show state, q=quit")
    print("=" * 50)

    fake_modes = ["normal", "stressed", "relaxed", "exercising", None]
    fake_idx = 0

    # 启动一个键盘监听线程
    import threading
    def keyboard_listener():
        nonlocal fake_idx
        while True:
            cmd = input().strip().lower()
            if cmd == 'q':
                os._exit(0)
            elif cmd == 's':
                print(f"📊 State: {radar.get_state()}")
            elif cmd == 'm':
                fake_idx = (fake_idx + 1) % len(fake_modes)
                radar.fake_mode = fake_modes[fake_idx]
                print(f"🎛️  Fake: {fake_modes[fake_idx] or 'REAL'}")

    threading.Thread(target=keyboard_listener, daemon=True).start()

    try:
        while True:
            if pending_audio:
                audio = pending_audio.pop(0)
                text, _ = transcribe(audio)
                if not text:
                    continue

                state = radar.get_state()
                print(f"📊 State: {state}")
                system = build_system_prompt(state)
                ask_llm_streaming(text, system, on_sentence=speak)
            time.sleep(0.05)
    finally:
        radar.stop()
```

### Day 2.4 - 状态推送 + 显示 (晚上)

#### 2.4.1 Pi → Core2 状态推送

ESPHome 可以接收 HTTP 数据更新内部传感器。在 `core2.yaml` 加:

```yaml
# 暴露 web 端点接收 Pi 推数据
api:
  encryption:
    key: !secret api_password
  services:
    - service: update_status
      variables:
        new_status: string
        new_hr: float
        new_br: float
      then:
        - lambda: |-
            id(status) = new_status;
            id(hr_value) = new_hr;
            id(br_value) = new_br;
```

#### 2.4.2 Pi 端调用 ESPHome API

新建 `~/Edge-AIGuard/esp_client.py`:

```python
"""
推送状态到 M5Stack Core2 (通过 ESPHome native API)
"""
import asyncio
import threading
from aioesphomeapi import APIClient

class CoreClient:
    def __init__(self, host, password=""):
        self.host = host
        self.password = password
        self.loop = asyncio.new_event_loop()
        self.client = None
        self._lock = threading.Lock()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=2)

    def start(self):
        threading.Thread(target=self._loop_runner, daemon=True).start()

    def _loop_runner(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())
        self.loop.run_forever()

    async def _connect(self):
        self.client = APIClient(self.host, 6053, self.password)
        await self.client.connect(login=True)
        print(f"✅ Connected to Core2 at {self.host}")

    async def _push(self, status, hr, br):
        if not self.client:
            return
        try:
            await self.client.execute_service({
                "name": "update_status"
            }, {"new_status": status, "new_hr": hr, "new_br": br})
        except Exception as e:
            print(f"⚠️  ESP push error: {e}")

    def push(self, status, hr=0.0, br=0.0):
        try:
            self._run(self._push(status, hr, br))
        except:
            pass
```

在 `pipeline.py` 集成:

```python
from esp_client import CoreClient

esp = CoreClient(host="192.168.1.51")  # Core2 IP
esp.start()

# 在主循环里:
state = radar.get_state()
esp.push("Listening", state.get("hr") or 0, state.get("br") or 0)
text, _ = transcribe(audio)

esp.push("Thinking", state.get("hr") or 0, state.get("br") or 0)
ask_llm_streaming(...)

esp.push("Ready", state.get("hr") or 0, state.get("br") or 0)
```

#### 2.4.3 Day 2 Deliverable Checklist

- [ ] Core2 屏幕能显示状态、HR、BR
- [ ] 按 Core2 按钮 A 录音并上传 Pi
- [ ] Pi 处理完语音回应,通过 Pi 喇叭播放
- [ ] Core2 屏幕实时显示 "Listening / Thinking / Ready" 状态切换
- [ ] 雷达数据从 Pi 推到 Core2 显示

### (可选) 唤醒词 - Stretch Goal

如果 Day 2 时间充裕,加唤醒词:

```yaml
voice_assistant:
  microphone: m5_mic
  use_wake_word: true
  noise_suppression_level: 2

micro_wake_word:
  models:
    - model: hey_jarvis
  on_wake_word_detected:
    then:
      - lambda: |-
          id(status) = "Wake!";
      - microphone.capture:
          id: m5_mic
          duration: 4s
          on_data:
            # ... HTTP POST 同上
```

如果跑不通,跳过即可,按钮触发已经够 demo。

---

## Day 3: 端到端联调 + 评测 + 论文

**目标**: 跑通完整链路,采集评测数据,录视频,写论文。

### Day 3.1 - 联调 + Bug 修复 (上午)

#### 3.1.1 完整测试场景

跑通以下 4 个场景:

**Scene 1: 默认状态**
- 按 Core2 按键 A
- 说 "What is the capital of France?"
- 听 Pi 简短回应
- Core2 显示 HR=72, BR=16, Status: Ready

**Scene 2: 紧张状态切换**
- Pi 终端按 'm' 切到 stressed
- Core2 显示 HR=95 (红色)
- 同样问题,听到温柔/简短的回应

**Scene 3: 放松状态**
- 切到 relaxed (HR=60)
- 同样问题,听到更对话化的回应

**Scene 4: 真实雷达 (如果 Kit 数据稳定)**
- 关掉 fake mode (按 'm' 循环到 None)
- 站起做几下深蹲,坐到 Kit 前面
- 等 30 秒让心率下降但还略高
- 问问题,看系统是否识别

#### 3.1.2 常见问题排查

| 现象 | 排查 |
|------|------|
| WiFi 不稳/丢包 | 用以太网,或换路由器 |
| 音频杂音 | Piper 输出 22050Hz, aplay 也要 22050Hz |
| 雷达数据全 0 | 检查 Kit 距离/朝向,坐近一点 |
| ESP32 死机 | 录音时长改短 (3s→2s) |
| LLM 太慢 | 换更小模型 (qwen2.5:0.5b) |
| ASR 不准 | 换 base.en (但更慢),或说话清晰 |
| Core2 屏幕黑屏 | 重新烧录或按 reset |

### Day 3.2 - 自动化评测 (下午)

#### 3.2.1 评测脚本 `eval.py`

创建 `~/Edge-AIGuard/eval.py`:

```python
"""
自动化评测脚本
跑预定 query × 状态,记录所有延迟和回答
"""
import json
import time
from radar import RadarReader
from prompts import build_system_prompt
from pipeline import ask_llm_streaming
import os

TEST_QUERIES = [
    ("What is the capital of France?", "factual"),
    ("Tell me a fun fact about space.", "general"),
    ("How are you feeling today?", "social"),
    ("What time is it in Tokyo?", "factual"),
    ("I had a long day at work.", "emotional"),
    ("Recommend a book to read.", "open"),
    ("What's 15 plus 27?", "math"),
    ("Tell me how to make tea.", "instruction"),
    ("Who wrote Hamlet?", "factual"),
    ("Are you a real person?", "philosophical"),
]

STATES = ["normal", "stressed", "relaxed", "exercising"]


def evaluate(radar):
    results = []
    total = len(TEST_QUERIES) * len(STATES)
    count = 0

    for query, qtype in TEST_QUERIES:
        for state_name in STATES:
            count += 1
            radar.fake_mode = state_name
            time.sleep(2)

            state = radar.get_state()
            system = build_system_prompt(state)

            t_start = time.time()
            collected = []
            ttfa = [None]

            def on_sent(s):
                if ttfa[0] is None:
                    ttfa[0] = time.time() - t_start
                collected.append(s)

            full, ftt, llm_total = ask_llm_streaming(query, system, on_sent)
            total_time = time.time() - t_start

            result = {
                "query": query,
                "type": qtype,
                "state": state_name,
                "system_prompt": system,
                "answer": full,
                "first_token_s": ftt,
                "llm_total_s": llm_total,
                "ttfa_s": ttfa[0],
                "total_s": total_time,
                "answer_length_chars": len(full),
                "num_sentences": len(collected),
            }
            results.append(result)
            print(f"[{count}/{total}] {state_name:11s} | {query[:35]:35s} | "
                  f"TTFA={ftt:.2f}s | {len(full)}c")

    return results


def stats(results):
    from statistics import mean, stdev

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    ttfa_all = [r["first_token_s"] for r in results if r["first_token_s"]]
    print(f"\nTTFA (Time-to-First-Audio):")
    print(f"  Mean: {mean(ttfa_all)*1000:.0f} ms")
    print(f"  Std:  {stdev(ttfa_all)*1000:.0f} ms")

    print(f"\nAnswer length by state:")
    for s in STATES:
        lens = [r["answer_length_chars"] for r in results if r["state"] == s]
        sents = [r["num_sentences"] for r in results if r["state"] == s]
        print(f"  {s:12s}: {mean(lens):5.0f} chars, {mean(sents):.1f} sentences (avg)")


if __name__ == "__main__":
    radar = RadarReader(host=os.environ.get("RADAR_IP", "192.168.1.100"))
    try:
        results = evaluate(radar)
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        stats(results)
        print(f"\nSaved to results.json ({len(results)} samples)")
    finally:
        radar.stop()
```

跑评测:

```bash
cd ~/Edge-AIGuard
source ~/ollama/bin/activate
python eval.py
# 大约 15-30 分钟跑完 40 个样本
```

#### 3.2.2 数据可视化 `plot.py`

创建 `~/Edge-AIGuard/plot.py`:

```python
"""
评测结果可视化
"""
import json
import matplotlib.pyplot as plt
from statistics import mean

with open("results.json") as f:
    results = json.load(f)

# === Plot 1: TTFA 分布直方图 ===
ttfa = [r["first_token_s"] for r in results if r["first_token_s"]]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(ttfa, bins=15, color='steelblue', edgecolor='black')
ax.axvline(mean(ttfa), color='red', linestyle='--', label=f'Mean = {mean(ttfa):.2f}s')
ax.set_xlabel("Time-to-First-Audio (s)")
ax.set_ylabel("Count")
ax.set_title(f"TTFA Distribution (n={len(ttfa)})")
ax.legend()
plt.tight_layout()
plt.savefig("plot_ttfa.png", dpi=150)
print("Saved plot_ttfa.png")

# === Plot 2: 回答长度按状态 ===
states = ["normal", "stressed", "relaxed", "exercising"]
lengths = {s: [r["answer_length_chars"] for r in results if r["state"] == s]
           for s in states}
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#4682B4', '#DC143C', '#228B22', '#FF8C00']
ax.bar(states, [mean(lengths[s]) for s in states],
       color=colors, edgecolor='black',
       yerr=[max(lengths[s])-mean(lengths[s]) for s in states],
       capsize=5)
ax.set_ylabel("Avg answer length (chars)")
ax.set_title("LLM Response Length by Physiological State")
plt.tight_layout()
plt.savefig("plot_length.png", dpi=150)
print("Saved plot_length.png")

# === Plot 3: 各阶段延迟分解 ===
fig, ax = plt.subplots(figsize=(8, 4))
phases = {
    "First Token": [r["first_token_s"] for r in results],
    "LLM Total": [r["llm_total_s"] for r in results],
    "End-to-End": [r["total_s"] for r in results],
}
positions = list(range(len(phases)))
ax.boxplot(phases.values(), positions=positions, labels=phases.keys())
ax.set_ylabel("Latency (s)")
ax.set_title("Latency Breakdown")
plt.tight_layout()
plt.savefig("plot_latency.png", dpi=150)
print("Saved plot_latency.png")
```

```bash
python plot.py
```

#### 3.2.3 经验性观察记录

跑完评测,人工分析 results.json,记录论文里要写的发现:

- 不同状态下回答长度的差异 (stressed 应最短)
- 不同状态下用词的差异 (stressed 是否用了 "calm/breath/gentle" 类词)
- 失败案例 (哪些 query 类型 LLM 表现差)

### Day 3.3 - 视频 + 论文 (晚上)

#### 3.3.1 Demo 视频脚本 (4 分钟)

```
[00:00 - 00:20] 项目介绍
  画面: 硬件全家福 (Pi 5 + Core2 + MR60BHA2 Kit)
  字幕: "Edge-AIGuard: Physiologically-Aware Voice Assistant"
  解说: 100% local, multimodal, privacy-preserving

[00:20 - 01:30] Scene 1: 正常状态对话
  画面: 用户坐姿放松
  动作: 按 Core2 按键 A
  屏幕显示: HR=72, Status: Listening → Thinking → Ready
  说话: "Tell me about Paris."
  听到: 正常长度回答 (~2 句话)

[01:30 - 02:30] Scene 2: 切换到压力状态
  画面: 终端切换 fake mode 到 stressed
  屏幕显示: HR=95 (红色加粗)
  说话: 同样问题
  听到: 温柔/简短/可能含 "take a deep breath"
  字幕对比: 两次回答并列展示

[02:30 - 03:10] Scene 3: 数据展示
  画面: 终端显示评测数据
  TTFA 直方图、回答长度对比图
  解说: "Sub-3s TTFA, context-aware response"

[03:10 - 03:40] Scene 4 (可选): 真实雷达
  画面: 用户走开,做深蹲,回来
  屏幕: HR 实时上升到 110+
  动作: 按按键问问题
  听到: 极简短回答

[03:40 - 04:00] 总结
  字幕: 100% Local · Multimodal · Privacy-Preserving · <$200
  Logo + GitHub link
```

录制工具:
- OBS Studio 录屏 (PC 端 SSH 终端)
- 手机直接拍硬件特写
- DaVinci Resolve 或 Adobe Premiere 剪辑

#### 3.3.2 论文修订要点

把 abstract / Expected Outcomes 部分按实测数据改:

**修订前**:
> sub-2s end-to-end response time
> 7B-8B model with 4-bit quantization on Pi 5

**修订后**:
> sub-3s Time-to-First-Audio (TTFA), with full response streaming continuously thereafter
> 1.5B-3B parameter SLMs (Qwen2.5, Llama3.2) on Pi 5

#### 3.3.3 论文实验部分模板

```latex
\section{Experimental Evaluation}

\subsection{Latency Breakdown}

We evaluated end-to-end latency across N=40 query-state pairs (10 queries × 4
states). Table I shows the breakdown:

\begin{table}[h]
\centering
\caption{End-to-End Latency Breakdown (n=40)}
\begin{tabular}{lcc}
\hline
Component & Mean (s) & Std (s) \\
\hline
ASR (Whisper base.en, 3s) & X.XX & X.XX \\
LLM First Token (Qwen2.5 1.5B) & X.XX & X.XX \\
TTS First Sentence (Piper) & X.XX & X.XX \\
\hline
Time-to-First-Audio & X.XX & X.XX \\
\hline
\end{tabular}
\end{table}

\subsection{Context-Aware Response Adaptation}

We measured the impact of physiological state injection on response
characteristics. Table II shows that the system produces meaningfully
shorter responses when the user is detected as stressed.

\begin{table}[h]
\centering
\caption{Response Length by Inferred State}
\begin{tabular}{lcc}
\hline
State & Avg Length (chars) & Avg Sentences \\
\hline
Normal & XXX & X.X \\
Stressed & XXX & X.X \\
Relaxed & XXX & X.X \\
Exercising & XXX & X.X \\
\hline
\end{tabular}
\end{table}

\subsection{Limitations}

\begin{itemize}
  \item \textbf{Echo cancellation} is not implemented; demos use a
        directional speaker or earphones.
  \item \textbf{Wake word} detection on Core2 is provided as a
        stretch capability; primary interaction uses push-to-talk.
  \item \textbf{Radar baseline calibration} requires ~30s of stable
        sitting before classifications stabilize.
  \item \textbf{Hallucinations} on factual queries (e.g., distance
        calculations) suggest function calling with external APIs as
        future work.
\end{itemize}
```

#### 3.3.4 Day 3 Deliverable Checklist

- [ ] `results.json` (40+ 样本评测数据)
- [ ] 三张论文图: `plot_ttfa.png`, `plot_length.png`, `plot_latency.png`
- [ ] Demo 视频 (3-5 分钟)
- [ ] 论文 LaTeX 修订完成 (实测数据填入,Limitations 添加)
- [ ] GitHub 仓库整理 (代码 + README + 视频链接)

---

## 完整文件清单

### Pi 5 上 (`~/Edge-AIGuard/`)

```
~/Edge-AIGuard/
├── pipeline.py          # 主控:ASR/LLM/TTS/HTTP server
├── radar.py             # MR60BHA2 Kit 客户端
├── prompts.py           # 状态-prompt 模板
├── esp_client.py        # 推送状态到 Core2
├── eval.py              # 自动评测
├── plot.py              # 可视化
├── results.json         # 评测数据
├── plot_ttfa.png        # TTFA 直方图
├── plot_length.png      # 长度对比
└── plot_latency.png     # 延迟分解
```

### PC 上 (`edge-aiguard-esp/`)

```
edge-aiguard-esp/
├── core2.yaml           # ESPHome 配置
└── secrets.yaml         # WiFi + API 密钥 (不进 git)
```

### 论文 (`paper/`)

```
paper/
├── main.tex             # 修订后的 LaTeX
├── references.bib       # 引用
├── figs/
│   ├── system_arch.pdf
│   ├── plot_ttfa.png
│   ├── plot_length.png
│   └── plot_latency.png
└── demo.mp4
```

---

## 风险与降级方案

| 风险 | 概率 | 影响 | 降级方案 |
|------|------|------|---------|
| Pi 电源欠压 | 高 | 致命 | **必须解决**,换 27W PD 电源 |
| 雷达 Kit WiFi 配不上 | 中 | 高 | 用 Fake Radar 演示,论文写 "controlled experiment" |
| ESPHome microphone 组件不稳定 | 中 | 中 | 改用 Pi USB 麦,Core2 只做显示 |
| ESP32 ↔ Pi WiFi 延迟大 | 中 | 中 | 用以太网共局域网 |
| 唤醒词跑不通 | 高 | 低 | 用按键触发即可 |
| LLM 太慢 | 低 | 中 | 换 qwen2.5:0.5b 或 llama3.2:1b |
| SD 卡空间满 | 中 | 高 | 删 phi3.5/llava 等大模型 |
| 时间不够 | - | - | Day 1 单机版已是完整 demo,可独立交付 |

### 应急 Plan B (如果到 Day 2 末仍未跑通 Core2)

把 Core2 部分**完全砍掉**,专注 Pi 端:
- Pi 上接 USB 麦
- Pi 上接 USB 喇叭
- 用键盘代替 Core2 按键触发录音
- 论文里说 "Hardware integration with Core2 deferred to future work"

这样 Day 2 时间全部投入到 Day 3 的评测和论文上,deliverable 仍然完整。

### 应急 Plan C (如果雷达 Kit 完全失效)

- 全程用 Fake Radar
- 论文里在 Methodology 写: "We evaluate the contextual injection mechanism using simulated radar inputs corresponding to four physiological states. Real-time radar integration is demonstrated separately in Section X."
- 这是合理的实验设计,审稿不会扣分

---

## 可选: Fine-tuning 路径

> **状态**: 待决策 (Day 3 评测后视情况启用)
> **硬件确认**: 笔记本 RTX 5070 Laptop, 8GB VRAM, CUDA 13.1, 驱动 592.01
> **目标模型**: qwen2.5:1.5b (训练目标 = 部署目标)
> **总时长**: 数据 2h + 训练 1-2h + 部署评测 1.5h = **半天**

### 决策门槛

Day 3.2 评测主表跑完后,看 prompt engineering 路径下不同 state 的 length 对比:

| 情况 | 决策 |
|------|------|
| stressed vs normal length 差 ≥ 30%, 用词区分明显 | prompt 工程足够, fine-tuning 写进 Future Work |
| 长度差 < 30% 或 style 不稳定 | 启用 fine-tuning, 用对比图证明改进 |
| Day 1/2 已经延期 | 砍掉,不做 |

### 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 训练框架 | unsloth | 4× 加速 + 4-bit QLoRA, 8GB VRAM 充裕 |
| 基模 | qwen2.5:1.5b | 与 Pi 部署一致,避免 train/serve mismatch |
| 训练方法 | LoRA r=8, alpha=16, 3 epoch | 小模型避免过拟合 |
| 数据生成 | Claude Sonnet 4.6 API | knowledge distillation, 1000 条约 $2-5 |
| 部署 | merge → GGUF q4_k_m → Ollama Modelfile | 与现有推理路径无缝衔接 |

### 实施步骤 (笔记本上)

#### F.1 环境准备 (10 分钟)

```powershell
# Windows PowerShell 或 WSL
conda create -n unsloth python=3.11 -y
conda activate unsloth

# CUDA 13.1 驱动向后兼容, 装 CUDA 12.4 wheels 即可
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install unsloth anthropic transformers datasets trl peft accelerate

# 验证
python -c "from unsloth import FastLanguageModel; import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出: `CUDA: True NVIDIA GeForce RTX 5070 Laptop GPU`

#### F.2 数据生成 `data_gen.py` (30-60 分钟)

并发调 Claude API 生成 (state, query, response) 三元组:
- 4 种 state × 25 种 query type × 10 条/格 = **1000 条**
- 用 `asyncio.Semaphore(15)` 控制并发 15 路
- 输出 `train.jsonl`, 每行格式:
  ```json
  {"messages": [
    {"role": "system", "content": "<拼接好的 system prompt with 具体 HR/BR 数字>"},
    {"role": "user", "content": "<query>"},
    {"role": "assistant", "content": "<风格匹配的 response>"}
  ]}
  ```

**关键 prompt 设计原则**:
- 在 generator prompt 里明确禁止 "Certainly!", "I'd be happy to help", "Of course!" 等 Claude 套话
- 给 generator 提供 reference example, 让它模仿本地小助手的简洁风格
- 每 100 条人工抽查 5 条, 发现风格漂移就调 prompt 重跑

#### F.3 训练 `train.py` (1-2 小时)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

ds = load_dataset("json", data_files="train.jsonl", split="train")

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        max_seq_length=2048,
        output_dir="out",
        logging_steps=10,
    ),
)
trainer.train()

# 合并 LoRA → 16-bit 权重 (后续转 GGUF 用)
model.save_pretrained_merged("qwen-edge-aiguard", tokenizer, save_method="merged_16bit")
```

预估: 1000 条 × 3 epoch, batch 2, grad_accum 4 → ~375 steps, 5070 上约 1-1.5 小时。

#### F.4 转 GGUF + 部署到 Pi (30 分钟)

```bash
# 笔记本上 (WSL 或 Linux 环境)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt
python convert_hf_to_gguf.py ../qwen-edge-aiguard \
  --outfile qwen-edge-aiguard-q4.gguf --outtype q4_k_m

# scp 到 Pi
scp qwen-edge-aiguard-q4.gguf hw3100:~/

# Pi 上注册到 Ollama
cat > ~/Modelfile <<EOF
FROM ./qwen-edge-aiguard-q4.gguf
PARAMETER temperature 0.7
PARAMETER num_predict 80
EOF
ollama create qwen-edge-aiguard -f ~/Modelfile
ollama list  # 确认出现 qwen-edge-aiguard
```

`pipeline.py` 改一行: `LLM_MODEL = "qwen-edge-aiguard"` 即可切换。

#### F.5 对比评测 (1 小时)

跑两遍 `eval.py`,生成 `results_baseline.json` 和 `results_finetuned.json`:

```bash
# baseline (prompt engineering only)
LLM_MODEL=qwen2.5:1.5b python eval.py --out results_baseline.json

# fine-tuned
LLM_MODEL=qwen-edge-aiguard python eval.py --out results_finetuned.json
```

加一张论文图 `plot_finetune_compare.png`:
- X 轴: 4 个 state
- Y 轴: 平均回答长度 (chars)
- 双柱状: baseline (蓝) vs fine-tuned (橙)
- 期望: fine-tuned 在 stressed 下更短、relaxed 下稍长,**state 之间的 separation 更大**

### 论文叙事

Fine-tuning 不是必须,但能加分:

**仅做 prompt engineering**:
> "We demonstrate context-aware response generation through dynamic system prompt injection. The 1.5B model adheres to physiologically-conditioned style guidance via natural language instructions alone."

**额外做 fine-tuning** (作为增量贡献):
> "We further train a LoRA adapter on N=1000 synthetic context-conditioned dialogues generated via knowledge distillation from a frontier model. The fine-tuned variant exhibits stronger style adherence (Δlength=X% greater between stressed/normal states) and reduces reliance on lengthy prompt engineering."

### 风险

| 风险 | 缓解 |
|------|------|
| CUDA 13.1 + PyTorch 兼容性 | 装 cu124 wheels (驱动向后兼容); 跑环境验证脚本 |
| 数据集风格泄漏 (Claude 套话) | generator prompt 明确禁止 + 抽查 50 条 + 提供 reference style |
| Fine-tuned 不如 baseline | 写成 negative result, 论文里诚实报告 (审稿欢迎诚实) |
| 训练时间挤占 Day 3 | 早上启动数据生成, 训练后台跑, 不阻塞主线评测 |
| 8GB VRAM OOM (扩到 3B 时) | 锁死 1.5B; 如要试 3B 用 max_seq_length=1024 + batch=1 |
| GGUF 转换失败 | llama.cpp 偶发问题, fallback: Ollama 直接吃 PEFT adapter (新版支持) |

### 并行 Day 3 时间线

如果决定做,Day 3 时间分配:

| 时段 | 主线 (Pi) | 副线 (笔记本) |
|------|----------|--------------|
| 上午 | 4 个测试场景跑通 + 录 Demo 初稿 | 数据生成 (1000 条) |
| 下午 | `eval.py` baseline 评测,出基线图 | LoRA 训练 (后台跑) |
| 晚上 | 论文实验数据填入 + GitHub 整理 | GGUF 转换 + Pi 部署 + fine-tuned 评测 + 对比图 |

如果 Day 3 上午 baseline 已经达标 (length 差 ≥30%), 副线全程可砍, 把笔记本时间用来剪视频。

---

## 快速参考

### 重启后必跑命令

```bash
# 启动 Ollama 服务
ollama serve > ~/ollama.log 2>&1 &
sleep 2

# 激活 Python 虚拟环境
source ~/ollama/bin/activate

# 启动语音助手
cd ~/Edge-AIGuard
RADAR_IP=192.168.1.100 python pipeline.py
```

### 健康检查命令

```bash
# 系统资源
df -h /                              # 磁盘
free -h                              # 内存
vcgencmd measure_temp                # 温度
vcgencmd get_throttled               # 0x0 = 健康
vcgencmd measure_clock arm           # CPU 频率

# Ollama
ollama list                          # 已装模型
ps aux | grep ollama                 # 服务状态
curl http://localhost:11434/api/tags # API 可达?

# 网络
hostname -I                          # 本机 IP
nmap -sn 192.168.1.0/24              # 扫描局域网
```

### 论文关键引用 (BibTeX 已在 references.bib)

- llama.cpp: gerganov2023llamacpp
- Whisper: radford2022robustspeechrecognitionlargescale
- Piper: kim2023piper
- GPTQ Quantization: frantar2023gptqaccurateposttrainingquantization
- ESP-SR: espressif2024espsr
- Llama 2: Touvron2023Llama2O

需要补充的引用:
- Qwen2.5 技术报告
- ESPHome / Home Assistant 项目
- faster-whisper / CTranslate2
- microWakeWord (Home Assistant 项目)

---

## 进度追踪

### Day 1 ✅ (主体完成,仅余 TTFT 优化)
- [x] 环境就绪 (依赖装好, Ollama 服务跑着)
- [x] Piper TTS 验证可用 (real-time factor 0.13)
- [x] 雷达 Kit IP 找到 (`192.168.1.140`), API 测试通过
- [x] `radar.py` 跑通 (真机 HR/BR/presence 实时数据, presence 从 HR 新鲜度反推)
- [x] `pipeline.py` 命令行版完整跑通 (text mode + 真雷达, voice mode 待麦克风)
- [x] Fake mode 切换 → 回答风格变化明显 ("The capital of France is Paris." → "Paris.")
- [x] **真雷达集成验证**: 不主动报数字, 被问就准确报 ("Your heart rate is 86 bpm.")

### Day 2 ☐
- [ ] M5Stack Core2 驱动装好,Arduino HelloWorld 验证
- [ ] ESPHome 装好
- [ ] `core2.yaml` 基础配置烧录成功 (显示 + 按键工作)
- [ ] 麦克风录音 + HTTP 上传到 Pi 工作
- [ ] Pi 接收音频 → ASR → LLM → TTS 全流程通
- [ ] Core2 屏幕实时显示状态 (Listening/Thinking/Ready)

### Day 3 ☐
- [ ] 4 个测试场景跑通
- [ ] `eval.py` 跑完,生成 results.json
- [ ] 三张图生成
- [ ] Demo 视频录制
- [ ] 论文实验数据填入
- [ ] GitHub 仓库整理

---

**最后更新**: 2026-04-29 (Day 1 完成,真雷达验证通过)
**版本**: v1.2
