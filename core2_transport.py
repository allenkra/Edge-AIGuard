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
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

VA = VoiceAssistantEventType


class Core2InputTransport(FrameProcessor):
    """Audio source: pushes AudioRawFrame downstream as Core2 streams mic data."""

    def __init__(self, sample_rate: int = 16000, transport: "Core2Transport | None" = None):
        super().__init__()
        self._sample_rate = sample_rate
        self._started = False
        self._transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._started = True
        elif isinstance(frame, EndFrame):
            self._started = False
        elif isinstance(frame, UserStoppedSpeakingFrame) and self._transport is not None:
            # Pipecat detected end-of-speech via sherpa STT endpoint. Core2 has no
            # local VAD so it would otherwise stream mic audio forever, holding
            # I2S0 and blocking speaker_->start() when TTS_START arrives. Tell
            # Core2 to stop the mic NOW: STT_VAD_END is the only event that
            # transitions voice_assistant to STOP_MICROPHONE → mic_source_->stop()
            # → I2S0 released. Without this, all subsequent TTS audio gets
            # dropped with "Cannot receive audio, buffer is full" errors.
            logger.info("[core2.audio.in] UserStoppedSpeakingFrame -> STT_VAD_END")
            self._transport._send_event(VA.VOICE_ASSISTANT_STT_VAD_END)
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
            logger.info("[core2.audio.out] TTSStartedFrame -> INTENT_END + TTS_START + TTS_END + TTS_STREAM_START")
            # voice_assistant protocol expects INTENT_END before TTS_START so
            # the device's pipeline state machine moves out of "intent" phase.
            self._transport._send_event(VA.VOICE_ASSISTANT_INTENT_END)
            # text="..." required: voice_assistant.cpp:652 early-returns on
            # empty text so speaker_->start() never runs.
            self._transport._send_event(VA.VOICE_ASSISTANT_TTS_START, {"text": "..."})
            # CRITICAL: send TTS_END *now*, before any audio chunks. The state
            # machine only moves to STREAMING_RESPONSE on TTS_END (line 692),
            # and STREAMING_RESPONSE is the only state that runs write_speaker_
            # to drain on_audio's 16KB pre-buffer. If TTS_END comes after the
            # audio (the obvious order), the pre-buffer fills in <1s and every
            # remaining chunk is dropped with "Cannot receive audio, buffer is
            # full". url must be non-empty (line 674) — only used by media_player
            # path which we don't use, so any string works.
            self._transport._send_event(VA.VOICE_ASSISTANT_TTS_END, {"url": "tts://local"})
            self._transport._send_event(VA.VOICE_ASSISTANT_TTS_STREAM_START)
            self._tts_streaming = True
        elif isinstance(frame, TTSAudioRawFrame):
            logger.debug(f"[core2.audio.out] TTSAudioRawFrame: {len(frame.audio)} bytes")
            # Await the chunked send instead of fire-and-forget so
            # TTSStoppedFrame can't race ahead and emit RUN_END while
            # chunks are still in flight (which makes Core2 drop them).
            await self._transport._send_audio_chunked_inline(frame.audio, frame.sample_rate)
        elif isinstance(frame, TTSStoppedFrame):
            logger.info("[core2.audio.out] TTSStoppedFrame -> TTS_STREAM_END + RUN_END")
            if self._tts_streaming:
                # TTS_END was already sent at TTSStartedFrame to advance the
                # state machine early. Just close the stream now.
                self._transport._send_event(VA.VOICE_ASSISTANT_TTS_STREAM_END)
                self._transport._send_event(VA.VOICE_ASSISTANT_RUN_END)
                self._tts_streaming = False
                # Auto re-arm Core2 for next turn — continuous conversation
                # without manual trigger.
                asyncio.create_task(self._transport.rearm_listen())

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
        self._input = Core2InputTransport(sample_rate=sample_rate_in, transport=self)
        self._output = Core2OutputTransport(self)
        self._closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._start_listen_svc = None

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
            # Cache start_listening service handle so we can re-arm Core2
            # after each TTS finishes — gives continuous conversation
            # without depending on the dead touch IC.
            _, services = await self._client.list_entities_services()
            self._start_listen_svc = next(
                (s for s in services if s.name == "start_listening"), None
            )
            if not self._start_listen_svc:
                logger.warning("[core2.audio] start_listening service missing — auto-rearm disabled")

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

    async def _send_audio_chunked_inline(self, audio_bytes: bytes, src_sample_rate: int = 22050):
        # ESPHome's voice_assistant + i2s_audio speaker uses the default
        # AudioStreamInfo (16 kHz mono int16) because voice_assistant.cpp never
        # calls speaker_->set_audio_stream_info(). The yaml `sample_rate: 22050`
        # only validates SLAVE-mode I2S and is otherwise ignored — the I2S DMA
        # is configured at audio_stream_info.get_sample_rate() = 16000.
        # So we MUST resample Piper's 22050 Hz output to 16 kHz before sending,
        # or audio plays slow-mo + the 32 KB/s drain rate can't keep up with our
        # send rate, causing 16KB pre-buffer overflow.
        if self._client is None or self._closed:
            return
        TARGET_RATE = 16000
        if src_sample_rate != TARGET_RATE:
            audio_bytes = self._resample_int16(audio_bytes, src_sample_rate, TARGET_RATE)
        # 1KB @ 30ms = 34.1 KB/s = 107% of 16-kHz realtime drain (32 KB/s).
        # Slight overrun keeps Core2's pre-buffer pleasantly full so inter-sentence
        # gaps from Piper (~1s while it generates the next utterance) don't
        # underrun the speaker mid-playback. The 16KB pre-buffer + 8KB ring buffer
        # absorb ~750ms of headroom at this rate; cumulative overrun on a 5s
        # sentence is ~10KB, well under the 16KB limit.
        CHUNK = 1024
        SLEEP_S = 0.030
        try:
            for i in range(0, len(audio_bytes), CHUNK):
                if self._closed or self._client is None:
                    break
                self._client.send_voice_assistant_audio(audio_bytes[i:i + CHUNK])
                await asyncio.sleep(SLEEP_S)
        except Exception as e:
            logger.warning(f"[core2.audio] audio send failed: {e!r}")

    @staticmethod
    def _resample_int16(audio: bytes, src_rate: int, dst_rate: int) -> bytes:
        import numpy as np
        from scipy import signal
        samples = np.frombuffer(audio, dtype=np.int16)
        new_n = int(len(samples) * dst_rate / src_rate)
        resampled = signal.resample(samples.astype(np.float32), new_n)
        # Clip to int16 range to avoid wrap; resample can overshoot a touch.
        np.clip(resampled, -32768, 32767, out=resampled)
        return resampled.astype(np.int16).tobytes()

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
        # STT_VAD_END is the only event that causes voice_assistant.cpp to
        # transition through STOP_MICROPHONE → call mic_source_->stop() →
        # release I2S0. Without it, the mic keeps owning I2S0 and the later
        # speaker_->start() (triggered by TTS_START) fails immediately, so
        # TTS audio is silently dropped. Send VAD_END FIRST, then STT_END /
        # INTENT_START which only fire user-visible triggers.
        self._send_event(VA.VOICE_ASSISTANT_STT_VAD_END)
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

    async def rearm_listen(self, delay_s: float = 12.0):
        """Trigger Core2 voice_assistant.start so the next user turn captures
        without waiting for a touch. Voice_assistant.cpp's request_start only
        works in State::IDLE — anything else is a silent no-op. After RUN_END
        the device walks RESPONSE_FINISHED → drain speaker_buffer_ → wait for
        speaker to flush DMA → speaker_->stop() → IDLE. That sequence takes
        roughly (audio_duration + 5s) — for a typical 6s TTS response, ~11s.
        Default 12s is a conservative wait; if you really need faster turn,
        poll Core2's voice_assistant state via the API (TODO).
        """
        if self._closed or self._client is None or self._start_listen_svc is None:
            return
        try:
            await asyncio.sleep(delay_s)
            if self._closed or self._client is None:
                return
            await self._client.execute_service(self._start_listen_svc, {})
            logger.info("[core2.audio] auto-rearm: start_listening sent")
        except Exception as e:
            logger.warning(f"[core2.audio] rearm failed: {e!r}")

    async def close(self):
        self._closed = True
        if self._reconnect is not None:
            await self._reconnect.stop()
