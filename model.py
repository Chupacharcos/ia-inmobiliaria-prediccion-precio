"""
Clase predictor singleton para el modelo de precio inmobiliario.
Se carga de forma lazy en la primera request (patrón thread-safe).
"""

import json
import threading
import numpy as np
import joblib
import torch
from pathlib import Path

from train import PricePredictor, FEATURES

ARTIFACTS = Path(__file__).parent / "artifacts"

_lock = threading.Lock()
_instance = None


class InmobiliarioPredictor:
    """Singleton thread-safe. Se instancia una vez y reutiliza en todas las requests."""

    def __init__(self):
        self.metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
        self.gbm = joblib.load(ARTIFACTS / "gbm_model.joblib")
        self.scaler = joblib.load(ARTIFACTS / "scaler.joblib")

        ckpt = torch.load(ARTIFACTS / "mlp_model.pt", map_location="cpu", weights_only=False)
        self.mlp = PricePredictor(input_dim=ckpt["input_dim"])
        self.mlp.load_state_dict(ckpt["state_dict"])
        self.mlp.eval()
        self._y_mean = ckpt["y_mean"]
        self._y_std = ckpt["y_std"]

        self.features = FEATURES

    # ── Feature engineering (replica train.py) ───────────────────────────────

    def _build_vector(self, data: dict) -> np.ndarray:
        """Construye el vector de features a partir del dict del formulario web."""
        def v(key, default=0.0):
            val = data.get(key, default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        sqft_living = v("sqft_living", 1800)
        bedrooms    = v("bedrooms", 3)
        bathrooms   = v("bathrooms", 2)
        floors      = v("floors", 1)
        condition   = v("condition", 3)
        grade       = v("grade", 7)
        yr_built    = v("yr_built", 1990)
        yr_renovated = v("yr_renovated", 0)
        lat         = v("lat", 47.5)
        long        = v("long", -122.2)
        sqft_lot    = v("sqft_lot", 7500)
        view        = v("view", 0)
        waterfront  = v("waterfront", 0)
        sqft_living15 = v("sqft_living15", sqft_living)
        sqft_above  = v("sqft_above", sqft_living)
        zipcode     = v("zipcode", 98103)

        # Features derivadas (igual que train.py)
        yr_renovated_flag = 1.0 if yr_renovated > 0 else 0.0
        renovated         = yr_renovated_flag
        age               = max(0.0, 2015.0 - yr_built)
        rooms_total       = bedrooms + bathrooms
        sqft_per_room     = sqft_living / max(rooms_total, 1)

        # Zipcode encoding: si no está en el mapa, usar precio medio
        zipcode_encoded = self.metadata.get("zipcode_map", {}).get(
            str(int(zipcode)),
            self.metadata.get("price_mean", 540000)
        )

        feature_map = {
            'bedrooms':          bedrooms,
            'bathrooms':         bathrooms,
            'sqft_living':       sqft_living,
            'sqft_lot':          sqft_lot,
            'floors':            floors,
            'waterfront':        waterfront,
            'view':              view,
            'condition':         condition,
            'grade':             grade,
            'sqft_above':        sqft_above,
            'yr_built':          yr_built,
            'yr_renovated_flag': yr_renovated_flag,
            'lat':               lat,
            'long':              long,
            'sqft_living15':     sqft_living15,
            'zipcode_encoded':   zipcode_encoded,
            'age':               age,
            'renovated':         renovated,
            'rooms_total':       rooms_total,
            'sqft_per_room':     sqft_per_room,
        }

        return np.array([feature_map[f] for f in self.features], dtype=np.float32)

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        X = self._build_vector(data).reshape(1, -1)

        # GBM
        price_gbm = float(self.gbm.predict(X)[0])

        # MLP
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            pred_n = self.mlp(torch.FloatTensor(X_scaled)).item()
        price_mlp = pred_n * self._y_std + self._y_mean

        # Ensemble 60/40
        price_ensemble = price_gbm * 0.60 + price_mlp * 0.40

        # Intervalo de confianza: ±10% del precio + diferencia entre modelos
        diff = abs(price_gbm - price_mlp)
        margin = max(price_ensemble * 0.10, diff * 0.60)

        # Importancias top-8 para la demo
        top_features = sorted(
            self.metadata["feature_importances"].items(),
            key=lambda x: x[1], reverse=True
        )[:8]

        return {
            "precio_estimado": round(price_ensemble, -2),       # redondear a centenas
            "precio_gbm":      round(price_gbm, -2),
            "precio_mlp":      round(price_mlp, -2),
            "rango_min":       round(price_ensemble - margin, -2),
            "rango_max":       round(price_ensemble + margin, -2),
            "feature_importances": [
                {"feature": k, "pct": v} for k, v in top_features
            ],
            "modelo_stats": {
                "gbm_r2":   self.metadata["gbm_r2"],
                "gbm_mae":  self.metadata["gbm_mae"],
                "mlp_r2":   self.metadata["mlp_r2"],
                "mlp_mae":  self.metadata["mlp_mae"],
                "n_samples": self.metadata["n_samples_total"],
                "dataset":   self.metadata["dataset"],
            },
        }

    def get_stats(self) -> dict:
        return {
            "cargado":   True,
            "metadata":  self.metadata,
        }


def get_predictor() -> InmobiliarioPredictor:
    """Devuelve el singleton, creándolo si es la primera vez (double-checked locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = InmobiliarioPredictor()
    return _instance


def is_loaded() -> bool:
    return _instance is not None
