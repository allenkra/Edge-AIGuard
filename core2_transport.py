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
    AudioRawFrame,
    EndFrame,
    Frame,
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
        frame = AudioRawFrame(
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
            self._transport._send_event(VA.VOICE_ASSISTANT_TTS_START)
            self._transport._send_event(VA.VOICE_ASSISTANT_TTS_STREAM_START)
            self._tts_streaming = True
        elif isinstance(frame, TTSAudioRawFrame):
            self._transport._send_audio(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
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

    def _send_audio(self, audio_bytes: bytes):
        if self._client is None or self._closed:
            return
        try:
            self._client.send_voice_assistant_audio(audio_bytes)
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
        # Returning None → use API audio path (handle_audio callback).
        return None

    async def _on_va_stop(self, server_side: bool):
        logger.info(f"[core2.audio] VA stop (server_side={server_side})")
        # User finished speaking. Tell Core2 STT is done; INTENT/TTS events
        # follow naturally as the pipeline progresses.
        self._send_event(VA.VOICE_ASSISTANT_STT_END)
        self._send_event(VA.VOICE_ASSISTANT_INTENT_START)

    async def _on_va_audio(self, audio_bytes: bytes):
        # Mic chunk from Core2; push into pipecat input.
        await self._input.push_audio(audio_bytes)

    async def close(self):
        self._closed = True
        if self._reconnect is not None:
            await self._reconnect.stop()
