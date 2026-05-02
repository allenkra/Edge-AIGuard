"""One-shot trigger for Core2's start_listening ESPHome action.

Workaround for the touchscreen on_press path that has never fired.
Usage: python trigger_listen.py  (env: CORE2_HOST, CORE2_NOISE_PSK)
"""

import asyncio
import os
import sys

from aioesphomeapi import APIClient


async def main() -> int:
    host = os.environ.get("CORE2_HOST", "edge-aiguard-core2.local")
    psk = os.environ.get("CORE2_NOISE_PSK", "")
    if not psk:
        print("CORE2_NOISE_PSK env var required", file=sys.stderr)
        return 2

    client = APIClient(host, 6053, password="", noise_psk=psk)
    await client.connect(login=True)
    try:
        _, services = await client.list_entities_services()
        svc = next((s for s in services if s.name == "start_listening"), None)
        if svc is None:
            print("start_listening service missing on Core2 — check yaml", file=sys.stderr)
            return 1
        await client.execute_service(svc, {})
        print(f"[trigger] start_listening -> {host}")
        return 0
    finally:
        await client.disconnect()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
