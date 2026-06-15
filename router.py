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
from visualizacion_datos import router as visualizacion_datos_router
router.include_router(visualizacion_datos_router)
from visualizacion import router as visualizacion_router
router.include_router(visualizacion_router)
from analytics_endpoint import router as analytics_endpoint_router
router.include_router(analytics_endpoint_router)
from analytics import router as analytics_router
router.include_router(analytics_router)
from utils_inmobiliario import router as utils_inmobiliario_router
router.include_router(utils_inmobiliario_router)
from utils import router as utils_router
router.include_router(utils_router)

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


@router.post("/inmobiliario/explain")
async def explain_inmobiliario(data: InmobiliarioRequest):
    """
    Devuelve SHAP values para explicar una predicción individual (waterfall chart).
    """
    if not (ARTIFACTS / "gbm_model.joblib").exists():
        raise HTTPException(status_code=503, detail="Modelo no entrenado.")

    from model import get_predictor

    loop = asyncio.get_event_loop()
    try:
        predictor = await loop.run_in_executor(_executor, get_predictor)
        payload = data.model_dump()
        result = await loop.run_in_executor(_executor, predictor.explain, payload)
        # Devolver el input para que el frontend pueda construir narrativas humanas con SHAP
        if isinstance(result, dict):
            result.setdefault("input", payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error SHAP: {str(e)}")


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


# ── ESPAÑA (modelo idealista18, datos reales 2018) ──────────────────────────────

class EspanaRequest(BaseModel):
    city: str        = Field(default="Madrid", description="Madrid | Barcelona | Valencia")
    area_m2: float   = Field(default=80,  ge=20,  le=800, description="Superficie construida (m²)")
    rooms: float     = Field(default=3,   ge=0,   le=12,  description="Habitaciones")
    bathrooms: float = Field(default=1,   ge=0,   le=8,   description="Baños")
    year_built: float= Field(default=1980, ge=1850, le=2018, description="Año de construcción")
    floor: float     = Field(default=2,   ge=-1,  le=40,  description="Planta")
    # Geografía: la aporta la ZONA elegida (coords reales del dataset).
    lat: float       = Field(default=40.42, description="Latitud (de la zona)")
    long: float      = Field(default=-3.70, description="Longitud (de la zona)")
    distance_to_center: float = Field(default=3.0, ge=0, le=40, description="Distancia al centro (km)")
    has_lift: int    = Field(default=1, ge=0, le=1)
    has_terrace: int = Field(default=0, ge=0, le=1)
    has_parking: int = Field(default=0, ge=0, le=1)
    has_ac: int      = Field(default=0, ge=0, le=1)
    has_pool: int    = Field(default=0, ge=0, le=1)
    has_garden: int  = Field(default=0, ge=0, le=1)
    is_duplex: int   = Field(default=0, ge=0, le=1)
    is_studio: int   = Field(default=0, ge=0, le=1)
    is_top_floor: int= Field(default=0, ge=0, le=1)


@router.post("/inmobiliario/predict_es")
async def predict_inmobiliario_es(data: EspanaRequest):
    """Predice precio en España (Madrid/Barcelona/Valencia) con el modelo
    entrenado sobre datos REALES de idealista18 (2018)."""
    if not (ARTIFACTS / "es_gbm_model.joblib").exists():
        raise HTTPException(503, "Modelo España no entrenado. Ejecuta python3 train_es.py")
    import model_es
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_executor, model_es.predict, data.model_dump())
    except Exception as e:
        raise HTTPException(500, f"Error en predicción ES: {e}")


@router.get("/inmobiliario/stats_es")
async def inmobiliario_stats_es():
    """Métricas + zonas por ciudad del modelo España (para la demo)."""
    if not (ARTIFACTS / "es_metadata.json").exists():
        raise HTTPException(503, "Modelo España no entrenado.")
    import model_es
    m = model_es.get_metadata()
    return {
        "cargado": model_es.is_loaded(),
        "dataset": m.get("dataset"),
        "dataset_doi": m.get("dataset_doi"),
        "cities": m.get("cities"),
        "n_total": m.get("n_total"),
        "metrics": m.get("metrics"),
        "city_stats": m.get("city_stats"),
        "city_zones": m.get("city_zones"),
        "top_features": m.get("top_features"),
        "synthetic": m.get("synthetic", False),
    }
