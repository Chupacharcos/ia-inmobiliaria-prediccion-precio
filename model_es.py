"""
Predictor de precio inmobiliario para ESPAÑA (modelo idealista18).

Paralelo a model.py (King County). Carga es_gbm_model.joblib y predice a partir
de inputs que un usuario español conoce: ciudad, zona, m² construidos,
habitaciones, baños, año, planta y comodidades. La latitud/longitud y la
distancia al centro las aporta la ZONA elegida (coords reales del dataset),
así el usuario no introduce coordenadas pero la geografía —el mayor driver del
precio— sí cuenta.
"""
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"

_lock = threading.Lock()
_state = {"model": None, "features": None, "meta": None}

# Orden EXACTO de features con el que se entrenó (ver train_es.py)
_CITY_FLAGS = {"Madrid": "CITY_MAD", "Barcelona": "CITY_BCN", "Valencia": "CITY_VLC"}


def _load():
    if _state["model"] is None:
        with _lock:
            if _state["model"] is None:
                bundle = joblib.load(ARTIFACTS / "es_gbm_model.joblib")
                _state["model"] = bundle["model"]
                _state["features"] = bundle["features"]
                _state["meta"] = json.loads((ARTIFACTS / "es_metadata.json").read_text())
    return _state


def is_loaded() -> bool:
    return (ARTIFACTS / "es_gbm_model.joblib").exists()


def get_metadata() -> dict:
    return _load()["meta"]


def predict(payload: dict) -> dict:
    """payload: dict con las claves de EspanaRequest (ver router.py)."""
    st = _load()
    model, features = st["model"], st["features"]

    city = payload.get("city", "Madrid")
    row = {
        "CONSTRUCTEDAREA":        float(payload.get("area_m2", 80)),
        "ROOMNUMBER":             float(payload.get("rooms", 3)),
        "BATHNUMBER":             float(payload.get("bathrooms", 1)),
        "CONSTRUCTIONYEAR":       float(payload.get("year_built", 1980)),
        "FLOORCLEAN":             float(payload.get("floor", 2)),
        "DISTANCE_TO_CITY_CENTER": float(payload.get("distance_to_center", 3.0)),
        "LATITUDE":               float(payload.get("lat", 40.42)),
        "LONGITUDE":              float(payload.get("long", -3.70)),
        "HASLIFT":                int(bool(payload.get("has_lift", 1))),
        "HASTERRACE":             int(bool(payload.get("has_terrace", 0))),
        "HASPARKINGSPACE":        int(bool(payload.get("has_parking", 0))),
        "HASAIRCONDITIONING":     int(bool(payload.get("has_ac", 0))),
        "HASSWIMMINGPOOL":        int(bool(payload.get("has_pool", 0))),
        "HASGARDEN":              int(bool(payload.get("has_garden", 0))),
        "ISDUPLEX":               int(bool(payload.get("is_duplex", 0))),
        "ISSTUDIO":               int(bool(payload.get("is_studio", 0))),
        "ISINTOPFLOOR":           int(bool(payload.get("is_top_floor", 0))),
        "CITY_MAD":               1 if city == "Madrid" else 0,
        "CITY_BCN":               1 if city == "Barcelona" else 0,
        "CITY_VLC":               1 if city == "Valencia" else 0,
    }
    # DataFrame con los nombres de feature exactos (evita el warning de
    # sklearn por "X sin nombres" y garantiza el orden correcto).
    x = pd.DataFrame([[row[f] for f in features]], columns=features)
    price = float(np.expm1(model.predict(x)[0]))
    eur_m2 = price / max(row["CONSTRUCTEDAREA"], 1)

    # Banda de incertidumbre honesta: ±MAPE del modelo (15.4%).
    mape = st["meta"]["metrics"]["mape_pct"] / 100.0
    return {
        "price_eur": round(price, -2),                       # redondeo a centena
        "eur_per_m2": round(eur_m2),
        "range_low":  round(price * (1 - mape), -2),
        "range_high": round(price * (1 + mape), -2),
        "city": city,
        "market": "ES",
        "model": "idealista18 GBM (datos reales 2018)",
        "mape_pct": st["meta"]["metrics"]["mape_pct"],
    }
