
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sse_starlette.sse import EventSourceResponse
from services.telemetry_proxy import TelemetryProxy

# KEL Reporter import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../irer-test-bench-scanner')))
try:
    from kel_reporter import KelReporter
except ImportError:
    KelReporter = None
from fastapi.responses import JSONResponse
@app.get("/api/kel/summary", response_class=JSONResponse)
async def kel_summary():
    """Return the latest KEL executive summary as JSON."""
    if KelReporter is None:
        raise HTTPException(status_code=500, detail="KelReporter not available")
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../irer-test-bench-scanner/reports/scan_results.json'))
    summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../irer-test-bench-scanner/reports/scan_summary.json'))
    try:
        reporter = KelReporter(results_path, summary_path)
        results = reporter.load_results()
        summary = reporter.summarize(results)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from routers import scanner
from routers import ai
from routers import session
from routers import pipeline



from fastapi import Request
from pydantic import BaseModel

class KelQueryRequest(BaseModel):
    file_path: str
    error_context: str
    selection: str = None

class KelQueryResponse(BaseModel):
    status: str
    remedies: list = []
    message: str = None


app.include_router(scanner.router)
app.include_router(ai.router)
app.include_router(session.router)
app.include_router(pipeline.router)

# CORS for UI at localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Body
@app.post("/api/v1/kel/query", response_model=KelQueryResponse)
async def kel_query(request: KelQueryRequest = Body(...)):
    if KelReporter is None:
        return KelQueryResponse(status="ERROR", remedies=[], message="KelReporter not available")
    try:
        file_path = request.file_path
        error_context = request.error_context
        selection = request.selection
        # Load results and find matching file/issue
        results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../irer-test-bench-scanner/reports/scan_results.json'))
        summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../irer-test-bench-scanner/reports/scan_summary.json'))
        reporter = KelReporter(results_path, summary_path)
        results = reporter.load_results()
        # Find relevant remedies for the file and context
        for entry in results:
            if entry.get("file") == file_path and (not error_context or error_context in str(entry)):
                remedies = entry.get("remedies", [])
                status = entry.get("status", "OK")
                if status == "UNINDEXED":
                    return KelQueryResponse(status="UNINDEXED", remedies=[], message="Manual triage required.")
                return KelQueryResponse(status=status, remedies=remedies, message="Remedial suggestions found.")
        return KelQueryResponse(status="NOT_FOUND", remedies=[], message="No remedial suggestions found.")
    except Exception as e:
        return KelQueryResponse(status="ERROR", remedies=[], message=str(e))

app.include_router(scanner.router, prefix="/api")
app.include_router(ai.router, prefix="/api")

telemetry_proxy = TelemetryProxy()

@app.get("/events/stream")
async def events_stream():
    async def event_generator():
        async for message in telemetry_proxy.listen():
            yield {"data": message}
    return EventSourceResponse(event_generator())

# Alias for frontend SSE consumption
@app.get("/stream")
async def stream_alias():
    async def event_generator():
        async for message in telemetry_proxy.listen():
            yield {"data": message}
    return EventSourceResponse(event_generator())
