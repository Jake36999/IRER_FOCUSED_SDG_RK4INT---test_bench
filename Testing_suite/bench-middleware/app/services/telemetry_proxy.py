import os
import asyncio
import redis.asyncio as redis

class TelemetryProxy:
    def __init__(self):
        self.host = os.getenv("TARGET_REDIS_HOST", "localhost")
        self.port = int(os.getenv("TARGET_REDIS_PORT", "6379"))
        self.channel = "phys_telemetry"

    async def listen(self):
        from app.models.schemas import TelemetryPacket
        from pydantic import ValidationError
        import json
        try:
            r = redis.Redis(host=self.host, port=self.port, decode_responses=True)
            pubsub = r.pubsub()
            await pubsub.subscribe(self.channel)
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Strict validation: parse as TelemetryPacket (discriminated union)
                        packet = None
                        try:
                            packet = TelemetryPacket.parse_obj(json.loads(message["data"]))
                            yield f"data: {json.dumps(packet.dict())}\n\n"
                        except ValidationError as ve:
                            # Fail safely, don't kill the stream
                            safe_payload = {"type": "UNKNOWN_FORMAT", "raw": message["data"]}
                            yield f"data: {json.dumps(safe_payload)}\n\n"
                    except Exception as e:
                        yield f'data: {{"status": "error", "detail": "{str(e)}"}}\n\n'
        except Exception as e:
            yield f'data: {{"status": "waiting for signal", "detail": "{str(e)}"}}\n\n'
            await asyncio.sleep(2)
