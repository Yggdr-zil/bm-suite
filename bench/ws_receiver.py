#!/usr/bin/env python3
"""CU Benchmark Suite — WebSocket Telemetry Receiver

Runs on your server. Accepts connections from ws_sender on remote instances.
Logs all telemetry to disk per-run and prints a live console view.

Usage:
    python3 ws_receiver.py                    # listen on 0.0.0.0:9876
    python3 ws_receiver.py --port 9876        # explicit port
    python3 ws_receiver.py --dir ~/cu-live    # custom output directory

Each connected instance gets its own log directory:
    {output_dir}/{hostname}_{timestamp}/
        telemetry.csv      — full telemetry stream
        events.log         — benchmark lifecycle events
        session.json       — connection metadata
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone


class SessionManager:
    """Tracks active benchmark sessions and their telemetry."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.sessions = {}  # ws_id -> session_info
        os.makedirs(output_dir, exist_ok=True)

    def register(self, ws_id: str, hello: dict) -> str:
        hostname = hello.get("hostname", "unknown")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_name = f"{hostname}_{ts}"
        session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)

        session = {
            "name": session_name,
            "dir": session_dir,
            "hostname": hostname,
            "connected_at": time.time(),
            "csv_file": open(os.path.join(session_dir, "telemetry.csv"), "w"),
            "event_file": open(os.path.join(session_dir, "events.log"), "w"),
            "csv_header_written": False,
            "line_count": 0,
            "last_event": "",
            "latest_temps": "",
        }

        with open(os.path.join(session_dir, "session.json"), "w") as f:
            json.dump({**hello, "session_name": session_name}, f, indent=2)

        self.sessions[ws_id] = session
        return session_name

    def handle_message(self, ws_id: str, raw: str):
        session = self.sessions.get(ws_id)
        if not session:
            return

        msg = json.loads(raw)
        msg_type = msg.get("type")

        if msg_type == "csv_header":
            header = msg["header"]
            session["csv_file"].write(header + "\n")
            session["csv_file"].flush()
            session["csv_header_written"] = True

        elif msg_type == "telemetry":
            data = msg["data"]
            session["csv_file"].write(data + "\n")
            session["csv_file"].flush()
            session["line_count"] += 1
            session["latest_temps"] = data

        elif msg_type == "event":
            name = msg.get("name", "?")
            data = msg.get("data", "")
            ts = datetime.now(timezone.utc).isoformat()
            session["event_file"].write(f"{ts} {name}: {data}\n")
            session["event_file"].flush()
            session["last_event"] = f"{name}: {data}"

    def disconnect(self, ws_id: str):
        session = self.sessions.pop(ws_id, None)
        if session:
            session["csv_file"].close()
            session["event_file"].close()

    def get_status(self) -> list:
        result = []
        for ws_id, s in self.sessions.items():
            result.append({
                "name": s["name"],
                "hostname": s["hostname"],
                "lines": s["line_count"],
                "last_event": s["last_event"],
                "latest": s["latest_temps"][:80] if s["latest_temps"] else "",
            })
        return result


async def handle_connection(websocket, manager: SessionManager):
    ws_id = str(id(websocket))
    session_name = None

    try:
        async for raw in websocket:
            msg = json.loads(raw)

            if msg.get("type") == "hello" and session_name is None:
                session_name = manager.register(ws_id, msg)
                print(f"  [receiver] New session: {session_name} from {msg.get('hostname')}")
                continue

            if session_name:
                manager.handle_message(ws_id, raw)

    except Exception as e:
        print(f"  [receiver] Connection error: {e}")
    finally:
        if session_name:
            session = manager.sessions.get(ws_id, {})
            lines = session.get("line_count", 0) if isinstance(session, dict) else 0
            print(f"  [receiver] Disconnected: {session_name} ({lines} telemetry lines logged)")
        manager.disconnect(ws_id)


async def status_printer(manager: SessionManager, interval: float = 2.0):
    """Print live status of active sessions."""
    while True:
        sessions = manager.get_status()
        if sessions:
            for s in sessions:
                if s["lines"] % 10 == 0 and s["lines"] > 0:
                    temp_snippet = s["latest"].split(",")[2] if "," in s["latest"] else "?"
                    print(f"  [{s['hostname']}] {s['lines']} samples | temp={temp_snippet}C | {s['last_event']}")
        await asyncio.sleep(interval)


async def run_server(host: str, port: int, output_dir: str):
    import websockets

    manager = SessionManager(output_dir)

    print(f"================================================================")
    print(f"  CU Bench Telemetry Receiver")
    print(f"  Listening:  ws://{host}:{port}")
    print(f"  Output:     {output_dir}")
    print(f"  {datetime.now(timezone.utc)}")
    print(f"================================================================")

    status_task = asyncio.create_task(status_printer(manager))

    async with websockets.serve(
        lambda ws: handle_connection(ws, manager),
        host, port,
        ping_interval=20,
        ping_timeout=10,
    ):
        await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description="CU Bench Telemetry Receiver")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9876, help="Listen port (default: 9876)")
    parser.add_argument("--dir", default=os.path.expanduser("~/cu-live"),
                        help="Output directory (default: ~/cu-live)")
    args = parser.parse_args()

    try:
        asyncio.run(run_server(args.host, args.port, args.dir))
    except KeyboardInterrupt:
        print("\n  [receiver] Stopped.")


if __name__ == "__main__":
    main()
