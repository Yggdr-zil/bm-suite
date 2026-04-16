#!/usr/bin/env python3
"""CU Benchmark Suite — WebSocket Telemetry Sender

Runs on the benchmarked instance. Tails telemetry.csv and pushes each new
line to the server's ws_receiver over WebSocket. Also sends benchmark
lifecycle events (step start/complete, scores).

Started by run_all.sh at the beginning; killed at the end.

Usage:
    python3 ws_sender.py ws://server:9876

Environment:
    RESULTS_DIR  — directory containing telemetry.csv (set by run_all.sh)
    CU_WS_URL    — WebSocket server URL (alternative to CLI arg)
"""
import asyncio
import json
import os
import sys
import time

RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")


async def tail_csv(path: str, queue: asyncio.Queue):
    """Tail a file and put new lines into the queue."""
    for _ in range(30):
        if os.path.exists(path):
            break
        await asyncio.sleep(0.5)

    if not os.path.exists(path):
        await queue.put(json.dumps({"type": "error", "msg": f"telemetry.csv not found after 15s"}))
        return

    with open(path) as f:
        header = f.readline().strip()
        if header:
            await queue.put(json.dumps({"type": "csv_header", "header": header}))

        while True:
            line = f.readline()
            if line:
                line = line.strip()
                if line:
                    await queue.put(json.dumps({"type": "telemetry", "data": line}))
            else:
                await asyncio.sleep(0.5)


async def watch_events(queue: asyncio.Queue):
    """Watch for benchmark event files (written by run_all.sh steps)."""
    events_dir = os.path.join(RESULTS_DIR, ".events")
    seen = set()

    while True:
        if os.path.isdir(events_dir):
            for fname in sorted(os.listdir(events_dir)):
                if fname not in seen:
                    seen.add(fname)
                    fpath = os.path.join(events_dir, fname)
                    try:
                        with open(fpath) as f:
                            data = f.read().strip()
                        await queue.put(json.dumps({"type": "event", "name": fname, "data": data}))
                    except Exception:
                        pass
        await asyncio.sleep(1)


async def send_loop(ws_url: str):
    """Main loop: tail CSV + watch events, send over WebSocket."""
    try:
        import websockets
    except ImportError:
        print("  [ws_sender] websockets not installed — streaming disabled.")
        return

    queue = asyncio.Queue()
    csv_path = os.path.join(RESULTS_DIR, "telemetry.csv")

    print(f"  [ws_sender] Connecting to {ws_url}...")

    retry_delay = 1
    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                print(f"  [ws_sender] Connected.")
                retry_delay = 1

                hello = {
                    "type": "hello",
                    "results_dir": RESULTS_DIR,
                    "hostname": os.uname().nodename,
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                }
                await ws.send(json.dumps(hello))

                tasks = [
                    asyncio.create_task(tail_csv(csv_path, queue)),
                    asyncio.create_task(watch_events(queue)),
                ]

                try:
                    while True:
                        msg = await queue.get()
                        await ws.send(msg)
                except Exception:
                    for t in tasks:
                        t.cancel()
                    raise

        except (ConnectionRefusedError, OSError) as e:
            print(f"  [ws_sender] Connection failed ({e}), retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)
        except Exception as e:
            print(f"  [ws_sender] Error: {e}, reconnecting...")
            await asyncio.sleep(2)


def main():
    ws_url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CU_WS_URL", "")
    if not ws_url:
        print("  [ws_sender] No WS URL (CU_WS_URL or arg) — streaming disabled.")
        return

    try:
        asyncio.run(send_loop(ws_url))
    except KeyboardInterrupt:
        print("  [ws_sender] Stopped.")


if __name__ == "__main__":
    main()
