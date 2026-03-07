"""
Router ML para FastAPI — Predicción de precio inmobiliario.
Se carga dinámicamente desde /var/www/chatbot/src/api.py con sys.path + importlib.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter(prefix="/ml", tags=["ML"])

_executor = ThreadPoolExecutor(max_workers=2)

ARTIFACTS = Path(__file__).parent / "artifacts"


# ── Schemas ───────────────────────────────────────────────────────────────────

class InmobiliarioRequest(BaseModel):
    sqft_living:   float = Field(default=1800, ge=100,  le=20000, description="Superficie habitable (sqft)")
    bedrooms:      float = Field(default=3,    ge=0,    le=20,    description="Numero de habitaciones")
    bathrooms:     float = Field(default=2,    ge=0,    le=15,    description="Numero de banos")
    floors:        float = Field(default=1,    ge=1,    le=4,     description="Numero de plantas")
    condition:     float = Field(default=3,    ge=1,    le=5,     description="Estado de conservacion (1-5)")
    grade:         float = Field(default=7,    ge=1,    le=13,    description="Calidad de construccion (1-13)")
    yr_built:      float = Field(default=1990, ge=1800, le=2015,  description="Ano de construccion")
    yr_renovated:  float = Field(default=0,    ge=0,    le=2015,  description="Ano de renovacion (0 = nunca)")
    lat:           float = Field(default=47.5, ge=-90.0, le=90.0,   description="Latitud")
    long:          float = Field(default=-122.2, ge=-180.0, le=180.0, description="Longitud")
    sqft_lot:      float = Field(default=7500, ge=100,  le=1000000, description="Superficie de la parcela (sqft)")
    view:          float = Field(default=0,    ge=0,    le=4,     description="Calidad de las vistas (0-4)")
    waterfront:    float = Field(default=0,    ge=0,    le=1,     description="Vistas al agua (0/1)")
    sqft_living15: Optional[float] = Field(default=None, description="Superficie media de los 15 vecinos mas cercanos")
    sqft_above:    Optional[float] = Field(default=None, description="Superficie sobre rasante")
    zipcode:       Optional[float] = Field(default=98103, description="Codigo postal")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def ml_health():
    """Estado del sistema ML."""
    from model import is_loaded
    return {
        "status": "ok",
        "models": {
            "inmobiliario": is_loaded(),
        }
    }


@router.post("/inmobiliario/predict")
async def predict_inmobiliario(data: InmobiliarioRequest):
    """
    Predice el precio de una vivienda usando GBM + MLP ensemble.
    La primera llamada carga el modelo en memoria (~1-2s).
    Las siguientes son instantáneas (<10ms).
    """
    if not (ARTIFACTS / "gbm_model.joblib").exists():
        raise HTTPException(
            status_code=503,
            detail="El modelo aún no ha sido entrenado. Ejecuta: cd /var/www/proyecto-inmobiliario && python3 train.py"
        )

    from model import get_predictor

    loop = asyncio.get_event_loop()
    try:
        predictor = await loop.run_in_executor(_executor, get_predictor)
        result = await loop.run_in_executor(_executor, predictor.predict, data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@router.get("/inmobiliario/stats")
async def inmobiliario_stats():
    """Métricas del modelo (R², MAE, n_samples) para mostrar en la demo."""
    metadata_path = ARTIFACTS / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=503, detail="Modelo no entrenado aún.")

    metadata = json.loads(metadata_path.read_text())

    from model import is_loaded
    return {
        "cargado": is_loaded(),
        "gbm_r2":   metadata.get("gbm_r2"),
        "gbm_mae":  metadata.get("gbm_mae"),
        "mlp_r2":   metadata.get("mlp_r2"),
        "mlp_mae":  metadata.get("mlp_mae"),
        "n_samples": metadata.get("n_samples_total"),
        "dataset":   metadata.get("dataset"),
        "fecha_entrenamiento": metadata.get("fecha_entrenamiento"),
        "features":  metadata.get("features"),
        "feature_importances": metadata.get("feature_importances"),
    }
