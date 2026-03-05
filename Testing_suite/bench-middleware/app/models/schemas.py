from typing import List, Union

# Telemetry SSE packet models
class TelemetryDataPoint(BaseModel):
    type: str = "DATA_POINT"
    value: float
    rho: float
    timestamp: str

class TelemetryAnomaly(BaseModel):
    type: str = "ANOMALY"
    drift_score: float
    h_norm: float
    timestamp: str

# Discriminated union for SSE
TelemetryPacket = Union[TelemetryDataPoint, TelemetryAnomaly]
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ScanRequest(BaseModel):
    version: str
    path: str
    mode: str = "full"
    lmstudio_url: Optional[str] = None
    ai_persona: Optional[str] = None
    rules: Optional[str] = "rules/governance.yaml"
    force_fresh_scan: bool = False
    parameters: Optional[Dict[str, Any]] = None

class ScanResponse(BaseModel):
    version: str
    status: str
    detail: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
