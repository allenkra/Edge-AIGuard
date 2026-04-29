"""
MR60BHA2 Sensor Kit - 通过 ESPHome native API 读取 HR/BR/presence
后台线程持续订阅, 主线程通过 get_state() O(1) 取最新窗口均值
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
        self.fake_mode = None

        self._entity_map = {}

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
                print(f"[radar] connection error: {e}, retry in 5s")
                await asyncio.sleep(5)

    async def _connect(self):
        client = APIClient(self.host, self.port, self.password)
        await client.connect(login=True)
        print(f"[radar] connected to {self.host}")

        entities, _ = await client.list_entities_services()
        for e in entities:
            name = (getattr(e, "name", "") or "").lower()
            obj_id = (getattr(e, "object_id", "") or "").lower()
            self._entity_map[e.key] = (name, obj_id)

        def on_state(state: EntityState):
            if state.key not in self._entity_map:
                return
            name, obj_id = self._entity_map[state.key]
            value = getattr(state, "state", None)
            if value is None:
                return

            search = f"{name} {obj_id}"
            now = time.time()

            if any(k in search for k in ["heart", "hr", "pulse", "bpm"]):
                if isinstance(value, (int, float)) and 30 < value < 200:
                    self.hr_window.append((now, float(value)))
            elif any(k in search for k in ["breath", "br", "respiration"]):
                if isinstance(value, (int, float)) and 5 < value < 40:
                    self.br_window.append((now, float(value)))
            elif any(k in search for k in ["presence", "occupancy", "detected"]):
                self.presence = bool(value)

        await client.subscribe_states(on_state)

        while self.running:
            await asyncio.sleep(1)

    def get_state(self):
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
                "source": "fake",
            }

        if not self.hr_window:
            return {
                "hr": None, "br": None,
                "presence": self.presence,
                "category": "no_user" if not self.presence else "unknown",
                "source": "real",
            }

        hr = sum(v for _, v in self.hr_window) / len(self.hr_window)
        br = (sum(v for _, v in self.br_window) / len(self.br_window)
              if self.br_window else 0)
        return {
            "hr": round(hr, 1),
            "br": round(br, 1),
            "presence": self.presence,
            "category": self._classify(hr, br),
            "source": "real",
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
