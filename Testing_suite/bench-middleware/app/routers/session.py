from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

# In-memory store for session events (replace with SQLite for production)
session_events = []

class SessionEvent(BaseModel):
    timestamp: str
    event_type: str
    user: Optional[str] = None
    detail: Optional[str] = None

@router.get("/api/v1/session/events", response_model=List[SessionEvent])
async def get_session_events():
    return session_events

@router.post("/api/v1/session/events", response_model=SessionEvent)
async def post_session_event(event: SessionEvent):
    event.timestamp = event.timestamp or datetime.utcnow().isoformat() + "Z"
    session_events.append(event)
    return event
