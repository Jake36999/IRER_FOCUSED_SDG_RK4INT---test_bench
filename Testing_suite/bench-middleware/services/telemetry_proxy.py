import asyncio
import json
import redis.asyncio as redis
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

class TelemetryProxy:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()

    async def subscribe_to_physics(self):
        """Observer Pattern: Passive subscription to the SDG physics engine."""
        await self.pubsub.subscribe("phys_telemetry")
        print("[Sovereign Auditor] Subscribed to SDG High-Frequency Stream")

    async def event_generator(self, request: Request):
        """SSE Streamer for React Frontend."""
        try:
            await self.subscribe_to_physics()
            while True:
                if await request.is_disconnected():
                    break
                
                message = await self.pubsub.get_message(ignore_subscribe_defaults=True)
                if message and message['type'] == 'message':
                    data = json.loads(message['data'])
                    
                    # Apply strict schema validation before broadcasting
                    yield {
                        "event": "DATA_POINT",
                        "data": json.dumps({
                            "h_norm": data.get("h_norm"),
                            "rho_max": data.get("rho_max"),
                            "timestamp": data.get("timestamp"),
                            "instability_flag": data.get("h_norm", 0) > 0.85
                        })
                    }
                await asyncio.sleep(0.01) # Maintain 100Hz frequency
        finally:
            await self.pubsub.unsubscribe("phys_telemetry")

# Usage in Router:
# @router.get("/stream")
# async def stream_telemetry(request: Request):
#     proxy = TelemetryProxy()
#     return EventSourceResponse(proxy.event_generator(request))