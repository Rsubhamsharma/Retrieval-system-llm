from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from app.services.pipeline import run_pipeline

router = APIRouter()

TEAM_TOKEN = "9af68b6b875cb17656117b037d8499a9918b5fc23ad61973efccd67450032f3e"


class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]


class RunResponse(BaseModel):
    answers: List[str]


@router.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(payload: RunRequest, Authorization: Optional[str] = Header(None)):
    if Authorization is None or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = Authorization.split(" ")[-1]
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    detailed_answers = await run_pipeline(document_url=str(payload.documents), questions=payload.questions)

    return RunResponse(
        answers=[item["answer"] for item in detailed_answers],
    )