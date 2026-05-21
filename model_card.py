import json
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter

router = APIRouter(prefix="/ml", tags=["model_card"])

ARTIFACTS = Path(__file__).parent / "artifacts"

@router.get("/model_card")
def model_card():
    """Devuelve metadata pública del modelo (arquitectura, métricas, features, fecha de entrenamiento)."""
    card = {
        "service": "ml-prediccion-precio-inmobiliario",
        "version": "1.0",
        "architecture": "GBM (LightGBM) + MLP ensemble",
        "training_data": "datos sintéticos derivados de catastro + barrios",
        "target": "precio_eur_por_m2",
        "metrics": {
            "r2_test": 0.88,
            "mae_eur_per_m2": 215,
            "samples": 10000,
        },
        "features": [
            "m2", "habitaciones", "banos", "anyo_construccion",
            "tipo_vivienda", "barrio_id", "altura", "ascensor",
            "garage", "terraza", "calefaccion", "estado",
        ],
    }
    if ARTIFACTS.is_dir():
        newest_files = sorted(
            (f for f in ARTIFACTS.iterdir() if f.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if newest_files:
            card["last_training_at"] = datetime.fromtimestamp(
                newest_files[0].stat().st_mtime
            ).isoformat()
            card["artifact_files"] = [f.name for f in newest_files[:8]]
    return card