"""Core2Transport: streams audio between Core2 (ESPHome voice_assistant) and
the pipecat pipeline. Replaces LocalAudioTransport when --core2-audio is set.

Wire model:
  Core2 mic (PDM, 16k int16)   ──API audio──▶  handle_audio  ──▶  AudioRawFrame
  TTSAudioRawFrame (22050 int16)  ──▶  send_voice_assistant_audio  ──▶  Core2 speaker

Pipeline integration mirrors LocalAudioTransport's interface:
  pipeline = [transport.input(), stt, ..., tts, transport.output(), ...]

Event sequencing into Core2 (drives the on_listening / on_tts_start / on_end
lambdas in core2.yaml):
  RUN_START + STT_START   on user tap (handle_start)
  STT_END                 on user stops talking (handle_stop, server_side=False)
  TTS_START + TTS_STREAM_START  on first TTSAudioRawFrame
  TTS_STREAM_END + TTS_END + RUN_END  on TTSStoppedFrame
"""
import asyncio
from typing import Optional

from loguru import logger

from aioesphomeapi import APIClient, ReconnectLogic
from aioesphomeapi.model import VoiceAssistantEventType

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

VA = VoiceAssistantEventType


class Core2InputTransport(FrameProcessor):
    """Audio source: pushes AudioRawFrame downstream as Core2 streams mic data."""

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self._sample_rate = sample_rate
        self._started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._started = True
        elif isinstance(frame, EndFrame):
            self._started = False
        await self.push_frame(frame, direction)

    async def push_audio(self, audio_bytes: bytes):
        if not self._started:
            return
        # InputAudioRawFrame (not AudioRawFrame mixin) carries the Frame
        # metadata (id, pts, name) that pipecat observers and downstream
        # processors expect. Using the bare mixin caused observer crashes
        # and STT to silently drop chunks.
        frame = InputAudioRawFrame(
            audio=audio_bytes,
            sample_rate=self._sample_rate,
            num_channels=1,
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)


class Core2OutputTransport(FrameProcessor):
    """Audio sink: forwards TTSAudioRawFrame to Core2 + drives VA TTS events."""

    def __init__(self, transport: "Core2Transport"):
        super().__init__()
        self._transport = transport
        self._tts_streaming = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            logger.info("[core2.audio.out] TTSStartedFrame -> TTS_START + TTS_STREAM_START")
            # voice_assistant protocol expects INTENT_END before TTS_START so
            # the device's pipeline state machine moves out of "intent" phase.
            self._transport._send_event(VA.VOICE_ASSISTANT_INTENT_END)
            # ESPHome voice_assistant.cpp checks for arg.name == "text" and
            # early-returns + skips speaker->start() if it's empty. Field is
            # literally "text", NOT "tts_text". With empty/missing text the
            # device never enters playback state — silent speaker.
            self._transport._send_event(VA.VOICE_ASSISTANT_TTS_START, {"text": "..."})
            self._transport._send_event(VA.VOICE_ASSISTANT_TTS_STREAM_START)
            self._tts_streaming = True
        elif isinstance(frame, TTSAudioRawFrame):
            logger.debug(f"[core2.audio.out] TTSAudioRawFrame: {len(frame.audio)} bytes")
            # Await the chunked send instead of fire-and-forget so
            # TTSStoppedFrame can't race ahead and emit RUN_END while
            # chunks are still in flight (which makes Core2 drop them).
            await self._transport._send_audio_chunked_inline(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            logger.info("[core2.audio.out] TTSStoppedFrame -> TTS_STREAM_END + TTS_END + RUN_END")
            if self._tts_streaming:
                self._transport._send_event(VA.VOICE_ASSISTANT_TTS_STREAM_END)
                self._transport._send_event(VA.VOICE_ASSISTANT_TTS_END)
                self._transport._send_event(VA.VOICE_ASSISTANT_RUN_END)
                self._tts_streaming = False

        await self.push_frame(frame, direction)


class Core2Transport:
    """Manages Core2 connection + voice_assistant subscription.

    Provides .input() and .output() to slot into a pipecat Pipeline like
    LocalAudioTransport does. The aioesphomeapi connection is re-used by
    Core2Client (status display) — but to keep transport ownership clean,
    this class opens its own connection. Two parallel connections to the
    same device are fine; ESPHome handles them as separate clients.
    """

    def __init__(self, host: str, noise_psk: str,
                 sample_rate_in: int = 16000, sample_rate_out: int = 22050):
        self._host = host
        self._noise_psk = noise_psk
        self._sample_rate_in = sample_rate_in
        self._sample_rate_out = sample_rate_out
        self._client: Optional[APIClient] = None
        self._reconnect: Optional[ReconnectLogic] = None
        self._input = Core2InputTransport(sample_rate=sample_rate_in)
        self._output = Core2OutputTransport(self)
        self._closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def input(self) -> Core2InputTransport:
        return self._input

    def output(self) -> Core2OutputTransport:
        return self._output

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._client = APIClient(
            self._host, 6053, password="", noise_psk=self._noise_psk
        )

        async def on_connect():
            logger.info(f"[core2.audio] connected to {self._host}")
            self._client.subscribe_voice_assistant(
                handle_start=self._on_va_start,
                handle_stop=self._on_va_stop,
                handle_audio=self._on_va_audio,
            )

        async def on_disconnect(expected: bool):
            level = logger.debug if expected else logger.warning
            level(f"[core2.audio] disconnected (expected={expected})")

        self._reconnect = ReconnectLogic(
            client=self._client,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
            zeroconf_instance=None,
            name=self._host,
        )
        await self._reconnect.start()

    def _send_event(self, event_type: VoiceAssistantEventType, data=None):
        if self._client is None or self._closed:
            return
        try:
            self._client.send_voice_assistant_event(event_type, data)
        except Exception as e:
            logger.warning(f"[core2.audio] event send failed ({event_type}): {e!r}")

    async def _send_audio_chunked_inline(self, audio_bytes: bytes):
        # ESPHome voice_assistant.cpp: SPEAKER_BUFFER_SIZE = 16 * 1024 (16KB).
        # on_audio drops bytes with "Cannot receive audio, buffer is full"
        # when the chunk can't fit. A whole TTS sentence from Piper
        # (35KB-150KB) overflows in one go, so we chunk + pace.
        # 4KB per chunk = ~93ms at 22050Hz int16 mono. Sleep 70ms between
        # chunks (slightly slower than realtime drain to let speaker keep up
        # without underflowing — DMA underflow makes the speaker exit too).
        if self._client is None or self._closed:
            return
        CHUNK = 4096
        SLEEP_S = 0.07
        try:
            for i in range(0, len(audio_bytes), CHUNK):
                if self._closed or self._client is None:
                    break
                self._client.send_voice_assistant_audio(audio_bytes[i:i + CHUNK])
                await asyncio.sleep(SLEEP_S)
        except Exception as e:
            logger.warning(f"[core2.audio] audio send failed: {e!r}")

    async def _on_va_start(self, conversation_id: str, flags: int,
                            audio_settings, wake_word_phrase):
        logger.info(
            f"[core2.audio] VA start (conv={conversation_id}, "
            f"flags={flags}, wake={wake_word_phrase})"
        )
        # Acknowledge by sending Run Start + STT Start so Core2's on_listening
        # lambda fires and the user sees the state change immediately.
        self._send_event(VA.VOICE_ASSISTANT_RUN_START)
        self._send_event(VA.VOICE_ASSISTANT_STT_START)
        # aioesphomeapi treats None as an error and sends back error=True.
        # For API audio (handle_audio callback) the UDP port is irrelevant;
        # return 0 to signal "session accepted, no UDP listener needed".
        return 0

    async def _on_va_stop(self, server_side: bool):
        logger.info(f"[core2.audio] VA stop (server_side={server_side})")
        # User finished speaking. Tell Core2 STT is done; INTENT/TTS events
        # follow naturally as the pipeline progresses.
        self._send_event(VA.VOICE_ASSISTANT_STT_END)
        self._send_event(VA.VOICE_ASSISTANT_INTENT_START)

    async def _on_va_audio(self, audio_bytes: bytes):
        # Mic chunk from Core2; push into pipecat input.
        # Log every 50th chunk so we don't flood (chunks ~10ms = 100/sec).
        self._chunks_in = getattr(self, "_chunks_in", 0) + 1
        if self._chunks_in % 50 == 1:
            logger.info(f"[core2.audio.in] chunk #{self._chunks_in} ({len(audio_bytes)} bytes)")
        # Dump first 200 chunks (~6.4s) to /tmp/core2_mic.raw for debugging
        if self._chunks_in <= 200:
            with open("/tmp/core2_mic.raw", "ab") as f:
                f.write(audio_bytes)
        await self._input.push_audio(audio_bytes)

    async def close(self):
        self._closed = True
        if self._reconnect is not None:
            await self._reconnect.stop()
