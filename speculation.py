"""Speculative LLM prefill — fire prefill for ASR partials so the real call hits hot KV-cache.

Plan: while the user is still speaking, sherpa-onnx emits InterimTranscriptionFrame
containing a growing partial transcript. Each partial triggers an async POST to
Ollama /api/generate with num_predict=1 + keep_alive=30m. By the time the final
TranscriptionFrame arrives and the real /v1/chat/completions call runs (via
pipecat's OllamaLLMService), Ollama's KV cache is loaded with the matching
prefix → real call's prefill is near-zero.

Design (per 2026-04-29 alignment):
  - Trigger: every InterimTranscriptionFrame (no debounce, no growth gate).
  - In-flight: cancel-on-new. The asyncio cancel does NOT actually abort the
    HTTP request — the thread keeps running and Ollama still completes the
    prefill, loading cache for the prior partial. Subsequent specs are just
    prefix extensions that prefill only the delta, so no work is wasted.
  - Multi-turn: v1 sends only `system + current partial`. First turn hits
    perfectly; later turns hit only the system prefix (since real has chat
    history before the user message). Future v2 may use /api/chat to render
    full history.
  - Radar drift: get_state() is queried fresh on every spec; we accept the
    ~67% baseline hit when HR/BR change between spec and real (validated in
    test_kv_cache.py scenario 7' after prompts.py refactor).

Validated by test_chat_template_alignment.py:
  - /api/generate and /v1/chat render byte-identical for equivalent inputs
    (Test 1: prompt_eval_count = 58 on both)
  - Cross-endpoint cache hit: 5.25× speedup, 80.9% wallclock saved (Test 2)
"""
import asyncio
import time

import requests
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
# Why 15s: Pi 5 cold prefill of a 150-tok system+prompt takes ~5-6s. With
# multiple partials queueing serially behind the first cold call, late specs
# can wait ~10s before Ollama gets to them. 15s gives headroom without
# letting a stuck request linger forever.
SPEC_TIMEOUT_S = 15.0

# Debounce / growth gate. Skip a partial if BOTH:
#   - it grew by less than MIN_GROWTH_CHARS chars beyond the last fired spec
#   - less than DEBOUNCE_S has passed since the last fired spec
# Either condition alone fires (fast typing or large jumps still trigger).
# First partial of an utterance always fires (last_spec_time=0 → elapsed huge).
MIN_GROWTH_CHARS = 6
DEBOUNCE_S = 0.2


class SpeculativePrefillProcessor(FrameProcessor):
    """Watches InterimTranscriptionFrame, fires async warmup to Ollama.

    Pipeline insertion point: right after STT, before RadarSystemPromptUpdater
    and the user aggregator. The frame passes through untouched; the spec is
    a side effect.
    """

    def __init__(
        self,
        *,
        get_state,
        build_system_prompt,
        model: str,
        keep_alive: str = "30m",
        min_growth_chars: int = MIN_GROWTH_CHARS,
        debounce_s: float = DEBOUNCE_S,
    ):
        super().__init__()
        self._get_state = get_state
        self._build_system_prompt = build_system_prompt
        self._model = model
        self._keep_alive = keep_alive
        self._min_growth = min_growth_chars
        self._debounce_s = debounce_s

        self._in_flight: asyncio.Task | None = None
        self._spec_count = 0
        self._spec_complete = 0
        self._spec_cancelled = 0
        self._spec_skipped = 0
        self._last_spec_text = ""
        self._last_spec_time = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            self._fire_async(frame.text)
        elif isinstance(frame, TranscriptionFrame):
            # Final transcript — leave the most recent in-flight spec alone
            # (its cache load is what the real call will benefit from). Just
            # reset tracking so the next utterance starts fresh.
            self._last_spec_text = ""
            self._last_spec_time = 0.0

        await self.push_frame(frame, direction)

    def _fire_async(self, partial: str):
        # Debounce + growth gate. Skip if partial is just a small extension
        # arriving too soon after the last spec — the previous spec's prefix
        # is still a valid prefix of this one, so skipping doesn't hurt
        # cache hit (final partial is always a superset). First partial of
        # an utterance fires unconditionally because _last_spec_time=0 makes
        # elapsed huge.
        now = time.monotonic()
        growth = len(partial) - len(self._last_spec_text)
        elapsed = now - self._last_spec_time
        if growth < self._min_growth and elapsed < self._debounce_s:
            self._spec_skipped += 1
            return

        # Cancel any prior in-flight spec on the asyncio side. The underlying
        # HTTP request keeps running in the thread pool, but that's fine —
        # Ollama still loads the cache for the prior partial, and subsequent
        # specs only prefill the delta tokens.
        if self._in_flight is not None and not self._in_flight.done():
            self._in_flight.cancel()
            self._spec_cancelled += 1

        try:
            state = self._get_state()
            system = self._build_system_prompt(state)
        except Exception as e:
            logger.warning(f"[spec] state/prompt build failed: {e!r}")
            return

        self._spec_count += 1
        seq = self._spec_count
        self._last_spec_text = partial
        self._last_spec_time = now
        self._in_flight = asyncio.create_task(self._fire(system, partial, seq))

    async def _fire(self, system: str, prompt: str, seq: int):
        """Best-effort prefill. Errors and cancels are swallowed."""
        body = {
            "model": self._model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"num_predict": 1, "temperature": 0.0},
        }
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(
                lambda: requests.post(
                    OLLAMA_GENERATE_URL, json=body, timeout=SPEC_TIMEOUT_S
                ).raise_for_status()
            )
        except asyncio.CancelledError:
            logger.debug(f"[spec #{seq}] cancelled (newer partial arrived)")
            raise
        except Exception as e:
            logger.warning(
                f"[spec #{seq}] failed after "
                f"{(time.perf_counter()-t0)*1000:.0f}ms: {e!r}"
            )
            return
        dt = (time.perf_counter() - t0) * 1000
        self._spec_complete += 1
        logger.info(
            f"[spec #{seq}] complete in {dt:.0f}ms — "
            f"prefix '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'"
        )

    @property
    def stats(self) -> dict:
        return {
            "fired": self._spec_count,
            "completed": self._spec_complete,
            "cancelled": self._spec_cancelled,
            "skipped": self._spec_skipped,
        }
