from fastapi import APIRouter

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])

@router.get("/")
def get_diagnostics():
    return {"model": "price-prediction", "status": "healthy", "metrics": {"mae": 0.0, "rmse": 0.0}}