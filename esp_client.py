"""Core2 ESPHome native API client + pipecat FrameProcessor.

Pushes pipeline state (Listening / Thinking / Speaking / Ready) and the
latest radar reading (HR/BR) to Core2's `update_status` service. Designed
to drop into `pipeline_pipecat.py` without disturbing the streaming path
— absent `--core2-host`, the rest of the pipeline is untouched.

Used in two places:
  1. `Core2Client`: maintains the aioesphomeapi connection, exposes
     `push(status, hr, br)` which is fire-and-forget from any task.
  2. `Core2StatusUpdater(FrameProcessor)`: turns pipecat frame transitions
     into status strings. A separate periodic task in pipeline_pipecat
     pushes radar-only updates while no conversation is active.
"""
import asyncio
from typing import Optional

from loguru import logger

from aioesphomeapi import APIClient, ReconnectLogic, UserService
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseStartFrame,
    StartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class Core2Client:
    """Maintains a connection to Core2 and serialises `update_status` calls.

    Reconnects automatically. `push()` is non-blocking — it enqueues the
    payload; a background drain task awaits the API call. If the queue
    grows (Core2 unreachable), older payloads get coalesced.
    """

    def __init__(self, host: str, noise_psk: str, port: int = 6053):
        self._host = host
        self._port = port
        self._noise_psk = noise_psk
        self._client: Optional[APIClient] = None
        self._reconnect: Optional[ReconnectLogic] = None
        self._update_svc: Optional[UserService] = None
        self._latest: Optional[tuple[str, float, float]] = None
        self._dirty = asyncio.Event()
        self._closed = False
        self._drain_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._client = APIClient(
            self._host, self._port, password="", noise_psk=self._noise_psk
        )

        async def on_connect() -> None:
            logger.info(f"[core2] connected to {self._host}")
            _, services = await self._client.list_entities_services()
            self._update_svc = next(
                (s for s in services if s.name == "update_status"), None
            )
            if not self._update_svc:
                logger.warning("[core2] update_status service missing — check yaml")
            elif self._latest is not None:
                # Re-push latest after reconnect so the screen catches up.
                self._dirty.set()

        async def on_disconnect(expected: bool) -> None:
            self._update_svc = None
            level = logger.debug if expected else logger.warning
            level(f"[core2] disconnected from {self._host} (expected={expected})")

        self._reconnect = ReconnectLogic(
            client=self._client,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
            zeroconf_instance=None,
            name=self._host,
        )
        await self._reconnect.start()
        self._drain_task = asyncio.create_task(self._drain_loop())

    def push(self, status: str, hr: float, br: float) -> None:
        """Schedule a status push. Coalesces — only the latest survives."""
        self._latest = (status, hr, br)
        self._dirty.set()

    async def _drain_loop(self) -> None:
        while not self._closed:
            try:
                await self._dirty.wait()
                self._dirty.clear()
                if self._closed:
                    break
                if self._update_svc is None or self._latest is None:
                    continue
                status, hr, br = self._latest
                try:
                    await self._client.execute_service(
                        self._update_svc,
                        {"new_status": status, "new_hr": hr, "new_br": br},
                    )
                except Exception as e:
                    logger.warning(f"[core2] push failed: {e!r}")
                    # Don't tight-loop on errors; the reconnect logic will heal
                    # the connection and on_connect re-sets _dirty.
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[core2] drain loop error: {e!r}")
                await asyncio.sleep(1)

    async def close(self) -> None:
        self._closed = True
        self._dirty.set()  # let drain loop exit
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._reconnect:
            await self._reconnect.stop()


class Core2StatusUpdater(FrameProcessor):
    """Translates pipecat frame transitions into Core2 status strings.

    Frame -> status mapping:
        UserStartedSpeakingFrame   -> "Listening..."
        UserStoppedSpeakingFrame   -> "Thinking..."
        LLMFullResponseStartFrame  -> "Thinking..." (idempotent)
        TTSStartedFrame            -> "Speaking..."
        TTSStoppedFrame            -> "Ready"
        StartFrame                 -> "Ready"

    Radar values are read every push from the supplied callable so the
    display always reflects the latest state without blocking on the
    radar's own update cadence.
    """

    def __init__(self, *, client: Core2Client, get_state):
        super().__init__()
        self._client = client
        self._get_state = get_state

    def _radar_pair(self) -> tuple[float, float]:
        try:
            s = self._get_state()
            hr = s.get("hr") or 0.0
            br = s.get("br") or 0.0
            return float(hr), float(br)
        except Exception:
            return 0.0, 0.0

    def _emit(self, status: str) -> None:
        hr, br = self._radar_pair()
        self._client.push(status, hr, br)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._emit("Ready")
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._emit("Listening...")
        elif isinstance(frame, (UserStoppedSpeakingFrame, LLMFullResponseStartFrame)):
            self._emit("Thinking...")
        elif isinstance(frame, TTSStartedFrame):
            self._emit("Speaking...")
        elif isinstance(frame, TTSStoppedFrame):
            self._emit("Ready")

        await self.push_frame(frame, direction)


async def periodic_radar_push(
    client: Core2Client, get_state, interval: float = 2.0
) -> None:
    """Background task: refresh Core2 HR/BR while no conversation is active.

    The status string is left as whatever the last frame transition set —
    we only nudge the numbers. If radar has no data we push 0.
    """
    while True:
        try:
            await asyncio.sleep(interval)
            s = get_state()
            hr = float(s.get("hr") or 0.0)
            br = float(s.get("br") or 0.0)
            # Reuse last known status by re-pushing the cached tuple's status
            # (Core2Client._latest), or default to "Ready" before first frame.
            last = getattr(client, "_latest", None)
            status = last[0] if last else "Ready"
            client.push(status, hr, br)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"[core2] periodic push error: {e!r}")
