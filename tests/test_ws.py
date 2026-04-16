"""Integration test: WebSocket sender + receiver on localhost."""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))

import websockets


async def run_receiver(port, ready_event, stop_event):
    """Minimal receiver that collects messages."""
    messages = []

    async def handler(ws):
        async for raw in ws:
            messages.append(json.loads(raw))

    async with websockets.serve(handler, "127.0.0.1", port):
        ready_event.set()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=10)
        except asyncio.TimeoutError:
            pass

    return messages


async def run_sender(port, results_dir):
    """Import and run sender for a few seconds."""
    from ws_sender import send_loop
    try:
        await asyncio.wait_for(
            send_loop(f"ws://127.0.0.1:{port}"),
            timeout=3
        )
    except asyncio.TimeoutError:
        pass


def test_sender_receiver_roundtrip():
    """Sender tails CSV, receiver gets hello + header + telemetry lines."""

    async def _test():
        port = 19876
        ready = asyncio.Event()
        stop = asyncio.Event()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "telemetry.csv")
            with open(csv_path, "w") as f:
                f.write("timestamp,gpu_idx,temp_c,power_w\n")
                f.write("2026/04/16 12:00:00.000, 0, 72, 350\n")
                f.write("2026/04/16 12:00:01.000, 0, 73, 352\n")

            os.environ["RESULTS_DIR"] = tmpdir

            recv_task = asyncio.create_task(
                run_receiver(port, ready, stop)
            )
            await ready.wait()
            await run_sender(port, tmpdir)
            stop.set()
            messages = await recv_task

        types = [m["type"] for m in messages]
        assert "hello" in types, f"No hello message, got: {types}"
        assert "csv_header" in types, f"No csv_header, got: {types}"
        assert "telemetry" in types, f"No telemetry data, got: {types}"

        telemetry_msgs = [m for m in messages if m["type"] == "telemetry"]
        assert len(telemetry_msgs) >= 2, f"Expected >=2 telemetry lines, got {len(telemetry_msgs)}"
        assert "72" in telemetry_msgs[0]["data"]

    asyncio.run(_test())
