# Handles alert endpoints.

from fastapi import APIRouter

router = APIRouter()

@router.post("/send")
async def send_alert(risk: int, label: str, rationale: str):
    return {
        "status": "Alert sent",
        "details": {"risk": risk, "label": label, "rationale": rationale}
    }

@router.get("/summary")
async def get_summary():
    return {"summary": "No summary yet"}