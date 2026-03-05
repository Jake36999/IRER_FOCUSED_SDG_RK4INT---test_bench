
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import difflib
import os

router = APIRouter()


class FixRequest(BaseModel):
    version: str
    filepath: str
    remedialCode: str


class FixResponse(BaseModel):
    version: str
    status: str
    detail: Optional[str] = None
    patch: Optional[str] = None


@router.post("/api/v1/pipeline/fix", response_model=FixResponse)
async def apply_fix(request: FixRequest):
    # Version check
    if not hasattr(request, 'version') or request.version != '1.0.0':
        return FixResponse(version=request.version if hasattr(request, 'version') else '', status="error", detail="API contract version mismatch or missing.", patch=None)
    # Observer contract: never mutate the target, only generate a patch
    try:
        with open(request.filepath, "r", encoding="utf-8") as f:
            original_lines = f.readlines()
        remedial_lines = request.remedialCode.splitlines(keepends=True)
        patch = ''.join(difflib.unified_diff(
            original_lines,
            remedial_lines,
            fromfile=request.filepath,
            tofile=request.filepath + ".remedial",
        ))
        if not patch:
            return FixResponse(status="noop", detail="No changes detected.", patch=None)
        return FixResponse(status="success", detail="Patch generated.", patch=patch)
    except Exception as e:
        return FixResponse(status="error", detail=str(e), patch=None)
