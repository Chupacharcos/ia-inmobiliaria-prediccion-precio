from fastapi import APIRouter
router = APIRouter(prefix="/explain", tags=["Explainability"])
@router.get('/shap/{property_id}')
def shap(property_id: int):
    return {"property_id": property_id, "shap_values": [0.1, -0.2, 0.05]}
