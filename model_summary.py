from fastapi import APIRouter

router = APIRouter(prefix="/summary", tags=["Summary"])

@router.get('/')
def summary():
    return {"model":"v1","status":"ready"}