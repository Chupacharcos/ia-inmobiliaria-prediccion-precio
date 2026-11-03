from fastapi import APIRouter

router = APIRouter(prefix="/metrics", tags=["Model Metrics"])

@router.get("/")
def get_model_metrics():
    return {"accuracy": 0.95, "precision": 0.92, "recall": 0.90}