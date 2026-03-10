import os
import shutil
import time
import threading
import tempfile
import json
from fastapi.testclient import TestClient
from app import app

GIFS_DIR = "GIFS"

def test_websocket_gif_update(tmp_path):
    # Setup: ensure GIFS dir is clean
    if os.path.exists(GIFS_DIR):
        shutil.rmtree(GIFS_DIR)
    os.makedirs(GIFS_DIR, exist_ok=True)

    client = TestClient(app)

    # Start WebSocket in a thread
    received_payloads = []
    def ws_thread():
        with client.websocket_connect("/ws/telemetry") as ws:
            # Wait for the backend to broadcast
            try:
                payload = ws.receive_json(timeout=5)
                received_payloads.append(payload)
            except Exception as e:
                import logging
                logging.error(f"WebSocket Test Error: {e}")

    t = threading.Thread(target=ws_thread)
    t.start()
    time.sleep(1)  # Give the server time to start

    # Simulate adding a GIF
    gif_path = os.path.join(GIFS_DIR, "new_best.gif")
    with open(gif_path, "wb") as f:
        f.write(os.urandom(1024))
    time.sleep(2)  # Allow watchdog to trigger

    t.join(timeout=5)
    assert received_payloads, "No payload received over WebSocket."
    payload = received_payloads[0]
    assert payload["type"] == "gif_update"
    assert payload["new_path"].endswith("/static/gifs/new_best.gif")
