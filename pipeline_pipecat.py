"""Edge-AIGuard Day 1.5 — Full-duplex streaming pipeline using pipecat.

Streams ASR (sherpa-onnx) -> LLM (Ollama) -> TTS (Piper) end-to-end with
barge-in via Silero VAD. Radar state injected into LLM system prompt every
turn (mutating LLMContext._messages[0] to preserve KV-cache prefix).

Modes:
  default       mic -> ASR -> LLM -> TTS -> speaker
  --text        stdin -> LLM -> TTS (no mic, no STT, no VAD)
  --no-audio    TTS bytes -> /tmp/edge_aiguard_last.raw (no speaker)
  --no-radar    radar.fake_mode='normal' instead of real ESPHome
  --wav PATH    one-shot: feed PATH through STT, exit (smoke test)

Speculative LLM prefill enabled by default — sherpa-onnx partials drive
async /api/generate warmups (see speculation.py). Disable with
--no-speculation for A/B comparison.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from loguru import logger

import sherpa_onnx
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.utils.time import time_now_iso8601

sys.path.insert(0, str(Path(__file__).parent))
from core2_transport import Core2Transport
from esp_client import Core2Client, Core2StatusUpdater, periodic_radar_push
from prompts import build_system_prompt
from radar import RadarReader
from speculation import SpeculativePrefillProcessor
from tts_aggregator import PhraseAwareTextAggregator

MODEL_DIR = Path(__file__).parent / "models" / "streaming-zipformer-en"
ENCODER = MODEL_DIR / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
DECODER = MODEL_DIR / "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
JOINER = MODEL_DIR / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
TOKENS = MODEL_DIR / "tokens.txt"
BPE_VOCAB = MODEL_DIR / "bpe.vocab"            # token<space>score format derived from bpe.model
HOTWORDS_FILE = MODEL_DIR / "hotwords.txt"     # domain phrases to bias toward

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")
RADAR_IP = os.environ.get("RADAR_IP", "seeedstudio-mr60bha2-kit-12fd18.lan")
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_US-amy-medium")
PIPER_DIR = Path(os.environ.get("PIPER_DIR", "~/piper/piper")).expanduser()
CORE2_HOST = os.environ.get("CORE2_HOST", "edge-aiguard-core2.local")
CORE2_NOISE_PSK = os.environ.get("CORE2_NOISE_PSK", "")

NO_AUDIO_OUT = Path("/tmp/edge_aiguard_last.raw")
SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 22050


def load_radar(args) -> RadarReader:
    """Return a RadarReader, or a fake-mode shim if --no-radar."""
    if not args.no_radar:
        try:
            return RadarReader(host=RADAR_IP)
        except Exception as e:
            logger.warning(f"radar connect failed ({e!r}); using fake_mode")
    fake = RadarReader.__new__(RadarReader)
    fake.fake_mode = "normal"
    fake.hr_window = []
    fake.br_window = []
    fake.presence = True
    fake.running = False
    return fake


class SherpaOnnxSTTService(STTService):
    """Streaming ASR via sherpa-onnx Zipformer transducer (LibriSpeech int8).

    Emits InterimTranscriptionFrame on partial updates and TranscriptionFrame
    when the recognizer's endpoint detector fires. Endpoint-driven finals
    instead of VAD-driven, since the recognizer's endpointer is ASR-tuned.
    """

    def __init__(
        self,
        *,
        model_dir: Path = MODEL_DIR,
        num_threads: int = 2,
        sample_rate: int = SAMPLE_RATE,
        **kwargs,
    ):
        # pipecat's STTService requires every Settings field to be explicitly
        # set or it logs a noisy "NOT_GIVEN" validate_complete error. The
        # sherpa-onnx model is identified by on-disk files rather than a
        # vendor model name; we pass static placeholders so the validation
        # passes without affecting behavior.
        super().__init__(
            sample_rate=sample_rate,
            audio_passthrough=True,
            model="sherpa-onnx-streaming-zipformer-en",
            language=Language.EN,
            **kwargs,
        )
        self._model_dir = model_dir
        self._num_threads = num_threads
        self._recognizer = None
        self._stream = None
        # process_generator awaits run_stt serially, but the lock makes the
        # "sherpa stream is not thread-safe" contract explicit.
        self._lock = asyncio.Lock()
        self._last_partial = ""

    def _init_recognizer(self):
        if self._recognizer is not None:
            return
        if self._sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"sherpa-onnx model expects {SAMPLE_RATE} Hz, got {self._sample_rate}"
            )
        logger.info(f"[stt] loading sherpa-onnx from {self._model_dir}")
        # Accuracy tuning:
        #  - modified_beam_search (max_active_paths=4) over greedy_search:
        #    +20-30% WER, RTF still ~0.13 on Pi 5 (well under 0.8 gate).
        #  - hotwords_file: domain phrases ("heart rate", "breathing rate",
        #    "France"…) get a hotwords_score bias so the decoder prefers
        #    them over phonetic look-alikes (e.g. "FRENF" → "FRANCE").
        #    Requires bpe_vocab (sentencepiece model) to tokenize raw text.
        #    hotwords_score=2.0 is a moderate bias — too high and we'd hear
        #    hallucinated hotwords on noise; 1.5 is the lib default.
        kwargs = dict(
            encoder=str(ENCODER),
            decoder=str(DECODER),
            joiner=str(JOINER),
            tokens=str(TOKENS),
            num_threads=self._num_threads,
            sample_rate=self._sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20.0,
            decoding_method="modified_beam_search",
            max_active_paths=4,
            provider="cpu",
        )
        if BPE_VOCAB.exists() and HOTWORDS_FILE.exists():
            kwargs.update(
                modeling_unit="bpe",
                bpe_vocab=str(BPE_VOCAB),
                hotwords_file=str(HOTWORDS_FILE),
                hotwords_score=2.0,
            )
            logger.info(f"[stt] hotwords enabled from {HOTWORDS_FILE.name}")
        else:
            logger.info("[stt] hotwords disabled (bpe.model or hotwords.txt missing)")
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        self._stream = self._recognizer.create_stream()
        logger.info("[stt] sherpa-onnx ready")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._init_recognizer()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if self._recognizer is None or self._stream is None:
            return
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        # diagnostic: log every 50th call to confirm audio actually reaches sherpa
        # (we suspected silent failure where chunks streamed but no partials emitted).
        self._calls = getattr(self, "_calls", 0) + 1
        if self._calls % 50 == 1:
            logger.info(f"[stt] run_stt #{self._calls}: {len(samples)} samples, "
                        f"last_partial={self._last_partial!r}")
        async with self._lock:
            self._stream.accept_waveform(self._sample_rate, samples)
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            text = self._recognizer.get_result(self._stream).strip()
            is_endpoint = self._recognizer.is_endpoint(self._stream)

            if is_endpoint:
                if text:
                    yield TranscriptionFrame(
                        text=text,
                        user_id=self._user_id or "user",
                        timestamp=time_now_iso8601(),
                        language=Language.EN,
                        finalized=True,
                    )
                self._recognizer.reset(self._stream)
                self._last_partial = ""
            elif text and text != self._last_partial:
                self._last_partial = text
                yield InterimTranscriptionFrame(
                    text=text,
                    user_id=self._user_id or "user",
                    timestamp=time_now_iso8601(),
                    language=Language.EN,
                )


class RadarSystemPromptUpdater(FrameProcessor):
    """Refresh LLMContext system message with build_system_prompt(radar_state)
    on every final user TranscriptionFrame. Mutates messages[0] in place to
    preserve the static prefix that prompts.py optimized for KV-cache reuse.
    """

    def __init__(self, *, context: LLMContext, get_state):
        super().__init__()
        self._context = context
        self._get_state = get_state
        # Reserve slot 0 for the system message.
        if not context._messages or context._messages[0].get("role") != "system":
            context._messages.insert(0, {"role": "system", "content": ""})

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            try:
                state = self._get_state()
                self._context._messages[0]["content"] = build_system_prompt(state)
                logger.debug(
                    f"[radar] state={state.get('category')} "
                    f"hr={state.get('hr')} br={state.get('br')}"
                )
            except Exception as e:
                logger.warning(f"[radar] update failed: {e!r}")
        await self.push_frame(frame, direction)


class TextInputProcessor(FrameProcessor):
    """Source for --text mode. submit(text) pushes a final TranscriptionFrame
    downstream, mimicking what STT would emit at endpoint."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Forward control / system frames so StartFrame, EndFrame, etc. propagate.
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def submit(self, text: str):
        frame = TranscriptionFrame(
            text=text,
            user_id="user",
            timestamp=time_now_iso8601(),
            language=Language.EN,
            finalized=True,
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)


class RawWriterProcessor(FrameProcessor):
    """Tail for --no-audio mode. Appends raw 16-bit PCM from TTS to a flat
    .raw file. Play with:
        aplay -r 22050 -f S16_LE -t raw /tmp/edge_aiguard_last.raw
    Matches existing pipeline.py convention.
    """

    def __init__(self, output_path: Path):
        super().__init__()
        self._path = output_path
        self._path.write_bytes(b"")
        self._total = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            with open(self._path, "ab") as f:
                f.write(frame.audio)
            self._total += len(frame.audio)
            logger.debug(f"[wav] +{len(frame.audio)}B (total {self._total}B)")
        await self.push_frame(frame, direction)


def build_pipeline(args, radar, core2_client=None, core2_audio_transport=None):
    """Construct pipeline based on args. Returns (pipeline, task, runner, refs)."""
    context = LLMContext(messages=[])
    pair = LLMContextAggregatorPair(context)
    user_agg = pair.user()
    assistant_agg = pair.assistant()

    radar_updater = RadarSystemPromptUpdater(
        context=context, get_state=radar.get_state
    )
    llm = OLLamaLLMService(model=LLM_MODEL)
    tts = PiperTTSService(voice_id=PIPER_VOICE, download_dir=PIPER_DIR)
    # Replace default sentence-only aggregator with phrase-aware one so
    # commas/colons also trigger TTS — cuts TTFA on multi-clause replies.
    tts._text_aggregator = PhraseAwareTextAggregator(
        aggregation_type=tts._text_aggregator._aggregation_type
    )

    transport = None
    text_input = None
    raw_writer = None
    stt = None

    if core2_audio_transport is not None:
        # Core2 mode: audio I/O goes over voice_assistant API instead of Pi ALSA.
        # Core2 firmware does its own VAD/endpointing — we still rely on the
        # streaming STT's endpoint detector for finalisation.
        transport = core2_audio_transport
    else:
        need_transport = (not args.text) or (not args.no_audio)
        if need_transport:
            vad = None
            if not args.text:
                vad = SileroVADAnalyzer(
                    sample_rate=SAMPLE_RATE,
                    params=VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.5),
                )
            transport = LocalAudioTransport(
                LocalAudioTransportParams(
                    audio_in_enabled=not args.text,
                    audio_out_enabled=not args.no_audio,
                    audio_in_sample_rate=SAMPLE_RATE,
                    audio_out_sample_rate=TTS_SAMPLE_RATE,
                    audio_in_channels=1,
                    audio_out_channels=1,
                    vad_analyzer=vad,
                )
            )

    spec = None
    if args.text:
        text_input = TextInputProcessor()
        head = [text_input]
    else:
        stt = SherpaOnnxSTTService(model_dir=MODEL_DIR, sample_rate=SAMPLE_RATE)
        head = [transport.input(), stt]
        if not args.no_speculation:
            spec = SpeculativePrefillProcessor(
                get_state=radar.get_state,
                build_system_prompt=build_system_prompt,
                model=LLM_MODEL,
            )
            head.append(spec)

    # Radar updater BEFORE user_agg so the system message is fresh by the time
    # user_agg appends user message and emits LLMContextFrame to the LLM.
    middle = [radar_updater, user_agg, llm, tts]

    if args.no_audio:
        raw_writer = RawWriterProcessor(NO_AUDIO_OUT)
        tail = [raw_writer, assistant_agg]
    else:
        tail = [transport.output(), assistant_agg]

    # Core2 status display tap-in (after assistant_agg so it observes the full
    # frame stream in both directions without altering ordering of any other
    # processor). No-op when --core2 is not set.
    if core2_client is not None:
        tail.append(Core2StatusUpdater(client=core2_client, get_state=radar.get_state))

    pipeline = Pipeline(head + middle + tail)
    task = PipelineTask(pipeline, idle_timeout_secs=1800)
    runner = PipelineRunner(handle_sigint=False)

    refs = {
        "context": context,
        "text_input": text_input,
        "transport": transport,
        "stt": stt,
        "spec": spec,
        "llm": llm,
        "tts": tts,
        "raw_writer": raw_writer,
    }
    return pipeline, task, runner, refs


async def text_mode_loop(task, runner, refs):
    """REPL: read stdin lines, push each as a final TranscriptionFrame."""
    text_input: TextInputProcessor = refs["text_input"]

    async def feed_stdin():
        # Wait for StartFrame to flow through the pipeline before submitting
        await asyncio.sleep(0.6)
        loop = asyncio.get_running_loop()
        print("\n>>> Edge-AIGuard text mode (q to quit) <<<")
        while True:
            try:
                line = await loop.run_in_executor(
                    None, lambda: input("\n> ").strip()
                )
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break
            await text_input.submit(line)
            await asyncio.sleep(0.5)
        await task.queue_frames([EndFrame()])

    await asyncio.gather(runner.run(task), feed_stdin())


async def wav_smoke_test(args):
    """Standalone STT smoke test: bypass pipeline, drive SherpaOnnxSTTService directly."""
    from scipy.io import wavfile

    print(f"\n=== STT smoke test on {args.wav} ===\n")
    sr, samples = wavfile.read(args.wav)
    if samples.dtype != np.int16:
        samples = (samples * 32768).astype(np.int16)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.int16)
    audio_dur = len(samples) / sr
    print(f"audio: {audio_dur:.2f}s @ {sr}Hz, {len(samples)} samples")

    stt = SherpaOnnxSTTService(model_dir=MODEL_DIR, sample_rate=SAMPLE_RATE)
    # In framework usage, StartFrame populates _sample_rate via start();
    # standalone we set it directly so _init_recognizer can validate.
    stt._sample_rate = SAMPLE_RATE
    stt._init_recognizer()

    chunk_samples = 320  # 20 ms @ 16 kHz, matching LocalAudioInputTransport default
    last_partial = ""
    for i in range(0, len(samples) - chunk_samples + 1, chunk_samples):
        chunk = samples[i : i + chunk_samples].tobytes()
        async for frame in stt.run_stt(chunk):
            if isinstance(frame, InterimTranscriptionFrame):
                if frame.text != last_partial:
                    print(f"[partial] {frame.text}")
                    last_partial = frame.text
            elif isinstance(frame, TranscriptionFrame):
                print(f"[FINAL]   {frame.text}")
                last_partial = ""

    # Flush with 3s of silence in 20ms chunks so the endpoint detector's
    # trailing-silence rule (rule2: ≥1.2s) accumulates across iterations.
    silence_chunk = np.zeros(chunk_samples, dtype=np.int16).tobytes()
    for _ in range(int(3 * SAMPLE_RATE / chunk_samples)):
        async for frame in stt.run_stt(silence_chunk):
            if isinstance(frame, TranscriptionFrame):
                print(f"[FINAL]   {frame.text}")


def parse_args():
    p = argparse.ArgumentParser(description="Edge-AIGuard pipecat pipeline")
    p.add_argument("--text", action="store_true",
                   help="text input mode (no microphone)")
    p.add_argument("--no-audio", action="store_true",
                   help="write TTS to .raw file instead of speaker")
    p.add_argument("--no-radar", action="store_true",
                   help="use radar fake_mode='normal' instead of real ESPHome")
    p.add_argument("--wav", help="STT smoke test: feed WAV file and exit")
    p.add_argument("--core2", action="store_true",
                   help="push pipeline status + radar to Core2 display "
                        "(needs CORE2_HOST and CORE2_NOISE_PSK env vars)")
    p.add_argument("--core2-audio", action="store_true",
                   help="route mic + speaker through Core2 via voice_assistant "
                        "(replaces LocalAudioTransport; needs CORE2_HOST + PSK)")
    p.add_argument("--no-speculation", action="store_true",
                   help="disable speculative LLM prefill (for A/B comparison)")
    return p.parse_args()


# TODO(v2): integrate tools.py via OpenAILLMService function-calling
async def main():
    args = parse_args()

    if args.wav:
        return await wav_smoke_test(args)

    radar = load_radar(args)

    if (args.core2 or args.core2_audio) and not CORE2_NOISE_PSK:
        logger.error("--core2 / --core2-audio require CORE2_NOISE_PSK env var (32-byte base64)")
        return

    core2_client = None
    core2_radar_task = None
    if args.core2:
        core2_client = Core2Client(CORE2_HOST, CORE2_NOISE_PSK)
        await core2_client.start()
        core2_radar_task = asyncio.create_task(
            periodic_radar_push(core2_client, radar.get_state)
        )

    core2_audio_transport = None
    if args.core2_audio:
        core2_audio_transport = Core2Transport(
            CORE2_HOST, CORE2_NOISE_PSK,
            sample_rate_in=SAMPLE_RATE,
            sample_rate_out=TTS_SAMPLE_RATE,
        )
        await core2_audio_transport.start()

    pipeline, task, runner, refs = build_pipeline(
        args, radar,
        core2_client=core2_client,
        core2_audio_transport=core2_audio_transport,
    )

    if args.core2_audio:
        audio_label = f"core2@{CORE2_HOST}"
    elif args.no_audio:
        audio_label = "off"
    else:
        audio_label = "local"
    print(
        f"Edge-AIGuard pipecat pipeline starting "
        f"(LLM={LLM_MODEL}, "
        f"radar={'fake' if args.no_radar else RADAR_IP}, "
        f"mode={'text' if args.text else 'mic'}, "
        f"audio={audio_label}, "
        f"status={'core2@' + CORE2_HOST if args.core2 else 'off'}, "
        f"speculation={'off' if args.no_speculation or args.text else 'on'})"
    )

    try:
        if args.text:
            await text_mode_loop(task, runner, refs)
        else:
            await runner.run(task)
    finally:
        if core2_radar_task:
            core2_radar_task.cancel()
            try:
                await core2_radar_task
            except (asyncio.CancelledError, Exception):
                pass
        if core2_client:
            await core2_client.close()
        if core2_audio_transport:
            await core2_audio_transport.close()
        if hasattr(radar, "stop") and getattr(radar, "running", False):
            radar.stop()


if __name__ == "__main__":
    asyncio.run(main())
