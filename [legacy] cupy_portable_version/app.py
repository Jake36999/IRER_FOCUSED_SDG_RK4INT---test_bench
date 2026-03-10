# --- WebSocket Telemetry Endpoint ---
import os
import asyncio
import logging
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = FastAPI()
active_connections: set[WebSocket] = set()

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("event") == "START_HUNT":
                await websocket.send_text(json.dumps({
                    "type": "log",
                    "message": "Hunt sequence initialized by Sovereign Auditor."
                }))
                # Optionally trigger hunt task here (no nested app or imports)
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
    finally:
        active_connections.discard(websocket)

# --- Ensure GIFS directory exists ---
GIFS_DIR = Path(__file__).parent / "GIFS"
GIFS_DIR.mkdir(exist_ok=True)

# Mount /static/gifs to GIFS folder
app.mount("/static/gifs", StaticFiles(directory=GIFS_DIR), name="gifs")

# Mount /static to project root (for heatmap PNG and other static assets)
PROJECT_ROOT = Path(__file__).parent
app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(__file__).parent / "UI" / "index.html"
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Directory Monitoring for GIF Updates ---
class GifWatcherHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop

    def on_modified(self, event):
        if event.src_path.endswith(".gif"):
            asyncio.run_coroutine_threadsafe(self.broadcast_gif_update(), self.loop)

    async def broadcast_gif_update(self):
        payload = {
            "type": "gif_update",
            "control_path": "/static/gifs/control_run.gif" if os.path.exists("GIFS/control_run.gif") else None,
            "prev_path": "/static/gifs/previous_best.gif" if os.path.exists("GIFS/previous_best.gif") else None,
            "new_path": "/static/gifs/new_best.gif" if os.path.exists("GIFS/new_best.gif") else None,
            # Optionally, read SSE from a metadata.json file here
        }
        for connection in list(active_connections):
            try:
                await connection.send_text(json.dumps(payload))
            except Exception:
                active_connections.discard(connection)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    event_handler = GifWatcherHandler(loop)
    observer = Observer()
    gifs_dir = Path(__file__).parent / "GIFS"
    observer.schedule(event_handler, str(gifs_dir), recursive=False)
    observer.start()

    # ...existing code...
