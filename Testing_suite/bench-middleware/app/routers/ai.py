from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.ai_client import analyze_code

class AnalyzeRequest(BaseModel):
    filepath: str
    code_snippet: str

router = APIRouter()

@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        result = await analyze_code(request.filepath, request.code_snippet)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
