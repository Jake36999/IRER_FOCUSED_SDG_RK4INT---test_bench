# FastAPI WebSocket API Documentation

## WebSocket Endpoint

**URL:** `/ws/telemetry`

**Description:**
- Real-time telemetry and event updates for the ASTE UI.
- Broadcasts JSON payloads to all connected clients when new GIFs are added or when log/metrics/status events occur.

---

## Event Types & JSON Schemas

### 1. `gif_update`
```
{
  "type": "gif_update",
  "control_path": "/static/gifs/control_run.gif" | null,
  "prev_path": "/static/gifs/previous_best.gif" | null,
  "new_path": "/static/gifs/new_best.gif" | null
}
```
- **Description:** Sent when a new .gif is added to the GIFS/ directory. Paths are null if the file does not exist.

### 2. `log`
```
{
  "type": "log",
  "level": "INFO" | "WARNING" | "ERROR",
  "message": "..."
}
```
- **Description:** Log messages from the backend.

### 3. `metrics`
```
{
  "type": "metrics",
  "sse": float,
  "pcs": float,
  "ic": float,
  "timestamp": "ISO8601 string"
}
```
- **Description:** Broadcasts updated simulation metrics.

### 4. `status`
```
{
  "type": "status",
  "state": "running" | "idle" | "error",
  "details": "..."
}
```
- **Description:** Backend status updates.

---

## Notes
- All payloads are sent as JSON text frames.
- The client should handle reconnects and re-subscribe to `/ws/telemetry` if the connection drops.
- Additional event types may be added in future releases.
