# Predicción de Precio Inmobiliario

Modelo de Machine Learning que predice el precio de una vivienda a partir de sus características. Entrenado con **21.563 transacciones reales** del dataset King County House Sales (Washington State, EE.UU., 2014–2015).

Demo interactiva en producción: [adrianmoreno-dev.com/demo/prediccion-precio-inmobiliario](https://adrianmoreno-dev.com/demo/prediccion-precio-inmobiliario)

---

## Resultados

| Modelo | R² | MAE |
|---|---|---|
| HistGradientBoostingRegressor (GBM) | **0.894** | $63.017 |
| Red Neuronal MLP (PyTorch) | **0.890** | $66.388 |
| **Ensemble 60/40** | **~0.893** | ~$64K |

El ensemble combina ambos modelos ponderando 60% GBM + 40% MLP, y genera un intervalo de confianza dinámico basado en la divergencia entre predicciones.

---

## Arquitectura

```
Formulario web
      │
      ▼
FastAPI  POST /ml/inmobiliario/predict
      │
      ├─▶ HistGradientBoostingRegressor  ──┐
      │   (400 iter, depth 7, CPU <5ms)    │
      │                                    ├─▶ Ensemble 60/40 ─▶ precio + rango
      └─▶ MLP PyTorch                     │
          (3 capas, ~25K params, CPU <5ms) ┘
```

La inferencia es asíncrona (`asyncio.run_in_executor`) para no bloquear el event loop de FastAPI. Los modelos se cargan en memoria en la primera petición (patrón singleton thread-safe) y se reutilizan en las siguientes.

---

## Features del modelo (20 variables)

| Feature | Descripción |
|---|---|
| `sqft_living` | Superficie habitable (sqft) |
| `sqft_lot` | Superficie de la parcela (sqft) |
| `sqft_above` | Superficie sobre rasante |
| `sqft_living15` | Superficie media de los 15 vecinos más cercanos |
| `bedrooms` | Número de habitaciones |
| `bathrooms` | Número de baños |
| `floors` | Número de plantas |
| `condition` | Estado de conservación (1–5) |
| `grade` | Calidad de construcción (1–13) |
| `yr_built` | Año de construcción |
| `yr_renovated_flag` | Ha sido reformada (0/1) |
| `waterfront` | Vistas al agua (0/1) |
| `view` | Calidad de vistas (0–4) |
| `lat` / `long` | Coordenadas geográficas |
| `zipcode_encoded` | Código postal (target-encoded: precio medio por zona) |
| `age` | Antigüedad de la vivienda (2015 - yr_built) |
| `renovated` | Alias de yr_renovated_flag |
| `rooms_total` | bedrooms + bathrooms |
| `sqft_per_room` | sqft_living / rooms_total |

Las tres variables más influyentes según permutation importance son **zona (zipcode)**, **superficie habitable** y **calidad de construcción**.

---

## Estructura del proyecto

```
/
├── train.py          # Script de entrenamiento (ejecutar offline una vez)
├── model.py          # Clase InmobiliarioPredictor (singleton, carga lazy)
├── router.py         # FastAPI APIRouter con los endpoints REST
└── artifacts/
    ├── gbm_model.joblib   # Modelo GBM serializado
    ├── mlp_model.pt       # Pesos MLP PyTorch + normalización
    ├── scaler.joblib      # StandardScaler para entrada del MLP
    ├── metadata.json      # Métricas, feature importances, fecha de entrenamiento
    └── test_data.pkl      # Muestra del conjunto de test (para visualizaciones)
```

---

## Endpoints REST

El router se monta en la API FastAPI principal bajo el prefijo `/ml`.

### `POST /ml/inmobiliario/predict`

Devuelve la predicción de precio para una vivienda.

**Body (JSON):**
```json
{
  "sqft_living": 1800,
  "bedrooms": 3,
  "bathrooms": 2,
  "floors": 1,
  "condition": 3,
  "grade": 7,
  "yr_built": 1990,
  "yr_renovated": 0,
  "lat": 47.5,
  "long": -122.2,
  "sqft_lot": 7500,
  "view": 0,
  "waterfront": 0,
  "zipcode": 98103
}
```

**Respuesta:**
```json
{
  "precio_estimado": 431600,
  "precio_gbm": 445200,
  "precio_mlp": 411800,
  "rango_min": 366900,
  "rango_max": 496200,
  "feature_importances": [
    {"feature": "zipcode_encoded", "pct": 28.4},
    {"feature": "sqft_living", "pct": 17.1},
    ...
  ],
  "modelo_stats": {
    "gbm_r2": 0.894,
    "gbm_mae": 63017,
    "mlp_r2": 0.890,
    "mlp_mae": 66388,
    "n_samples": 21563,
    "dataset": "King County House Sales (Washington State, USA)"
  }
}
```

### `GET /ml/inmobiliario/stats`

Devuelve las métricas del modelo y los metadatos del entrenamiento.

### `GET /ml/health`

Estado de carga de los modelos.

---

## Cómo re-entrenar

```bash
cd /var/www/proyecto-inmobiliario
source /var/www/chatbot/venv/bin/activate
python3 train.py
```

El script descarga automáticamente el dataset desde GitHub mirrors, entrena ambos modelos (~3–5 min en CPU) y sobreescribe los artifacts. No requiere GPU.

---

## Stack técnico

- **Python 3.12**
- **scikit-learn 1.8** — HistGradientBoostingRegressor, StandardScaler, permutation_importance
- **PyTorch 2.10** — MLP con BatchNorm, Dropout, AdamW + CosineAnnealingLR
- **FastAPI** — endpoints REST, validación con Pydantic
- **joblib** — serialización de modelos sklearn
- **pandas / numpy** — carga y feature engineering del dataset
