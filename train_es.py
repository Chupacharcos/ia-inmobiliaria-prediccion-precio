"""
Entrenamiento del modelo de precio inmobiliario para ESPAÑA.

Modelo PARALELO al de King County (train.py) — no lo sustituye. El selector de
mercado en la demo elige entre King County (modelo original) y España (este).

Dataset: idealista18 (Páez et al.) — listados REALES de Idealista de 2018 para
las 3 mayores ciudades de España (Madrid, Barcelona, Valencia), publicado como
producto de datos abierto con DOI 10.1177/23998083241242844. NO es sintético.

Ejecutar offline (los .rda van en data_es/, ver README_ES):
  cd /var/www/proyecto-inmobiliario
  source /var/www/chatbot/venv/bin/activate
  python3 train_es.py

Genera en artifacts/:
  es_gbm_model.joblib  — HistGradientBoostingRegressor (entrena sobre log(precio))
  es_metadata.json     — métricas reales, importancias, ciudades, procedencia
"""
import json
import joblib
import numpy as np
import pandas as pd
import rdata
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data_es"
ARTIFACTS.mkdir(exist_ok=True)

CITIES = ["Madrid", "Barcelona", "Valencia"]

# Features REALES disponibles en idealista18. Elegidas por ser las que un
# usuario puede conocer/introducir + las geográficas que el modelo necesita.
NUM_FEATURES = [
    "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "CONSTRUCTIONYEAR",
    "FLOORCLEAN", "DISTANCE_TO_CITY_CENTER", "LATITUDE", "LONGITUDE",
]
BOOL_FEATURES = [
    "HASLIFT", "HASTERRACE", "HASPARKINGSPACE", "HASAIRCONDITIONING",
    "HASSWIMMINGPOOL", "HASGARDEN", "ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR",
]
TARGET = "PRICE"


def _load_city(name: str) -> pd.DataFrame:
    path = DATA / f"{name}_Sale.rda"
    parsed = rdata.read_rda(str(path))
    df = parsed[f"{name}_Sale"]
    df = pd.DataFrame(df)  # asegurar DataFrame plano
    df["CITY"] = name
    return df


def main():
    frames = [_load_city(c) for c in CITIES]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.astype(str)  # rdata da np.str_; sklearn exige str
    print(f"Total listados crudos (3 ciudades): {len(df):,}")

    # ── Limpieza honesta ────────────────────────────────────────────────────
    keep = NUM_FEATURES + BOOL_FEATURES + ["CITY", TARGET]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Tipos numéricos
    for c in NUM_FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BOOL_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # City one-hot (el modelo aprende el nivel de precio de cada ciudad)
    df["CITY_MAD"] = (df["CITY"] == "Madrid").astype(int)
    df["CITY_BCN"] = (df["CITY"] == "Barcelona").astype(int)
    df["CITY_VLC"] = (df["CITY"] == "Valencia").astype(int)

    # Filtros de saneamiento: precios y áreas plausibles (quita outliers/errores)
    df = df.dropna(subset=[TARGET, "CONSTRUCTEDAREA"])
    df = df[(df[TARGET].between(20_000, 5_000_000)) &
            (df["CONSTRUCTEDAREA"].between(20, 800)) &
            (df["ROOMNUMBER"].between(0, 12))]
    print(f"Tras saneamiento: {len(df):,} listados")

    feat_cols = [str(c) for c in (NUM_FEATURES + BOOL_FEATURES + ["CITY_MAD", "CITY_BCN", "CITY_VLC"])]
    # Reconstruir X con nombres de columna Python str puros (rdata deja np.str_
    # residual que sklearn rechaza). Usamos .values + columns explícitas.
    X = pd.DataFrame(df[feat_cols].to_numpy(dtype=float), columns=feat_cols)
    # Entrenar sobre log(precio): el precio inmobiliario es multiplicativo,
    # log estabiliza la varianza y da mejor MAE relativo.
    y = np.log1p(df[TARGET].values)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=8,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(X_tr, y_tr)

    # ── Métricas REALES en € (deshaciendo el log) ───────────────────────────
    pred_log = model.predict(X_te)
    pred = np.expm1(pred_log)
    true = np.expm1(y_te)
    mae = mean_absolute_error(true, pred)
    medae = median_absolute_error(true, pred)
    r2 = r2_score(y_te, pred_log)          # R² en espacio log (el que optimiza)
    r2_eur = r2_score(true, pred)          # R² en € (más exigente con outliers)
    mape = float(np.mean(np.abs((true - pred) / true)) * 100)

    print(f"\n── Métricas (test, {len(true):,} pisos reales) ──")
    print(f"  R² (log)     : {r2:.3f}")
    print(f"  R² (€)       : {r2_eur:.3f}")
    print(f"  MAE          : {mae:,.0f} €")
    print(f"  Mediana error: {medae:,.0f} €")
    print(f"  MAPE         : {mape:.1f} %")

    # Importancias por permutación (ligero: muestra posicional alineada)
    from sklearn.inspection import permutation_importance
    n = min(3000, len(X_te))
    perm = permutation_importance(
        model, X_te.iloc[:n], y_te[:n], n_repeats=3, random_state=1, n_jobs=1)
    importances = sorted(
        [(c, float(round(v, 4))) for c, v in zip(feat_cols, perm.importances_mean)],
        key=lambda x: -x[1])[:8]

    # ── Guardar artefactos ──────────────────────────────────────────────────
    joblib.dump({"model": model, "features": feat_cols}, ARTIFACTS / "es_gbm_model.joblib")

    # Precio mediano por ciudad (para defaults útiles en el frontend)
    city_stats = {}
    for c in CITIES:
        sub = df[df["CITY"] == c]
        city_stats[c] = {
            "n": int(len(sub)),
            "median_price": int(sub[TARGET].median()),
            "median_eur_m2": int((sub[TARGET] / sub["CONSTRUCTEDAREA"]).median()),
        }

    # Zonas representativas por ciudad (centro/intermedia/periferia) según
    # terciles de DISTANCE_TO_CITY_CENTER, con coords medianas REALES. El
    # frontend ofrece estas zonas y envía sus coords/distancia al backend —
    # así el usuario no introduce lat/long pero la geografía sí cuenta.
    city_zones = {}
    for c in CITIES:
        sub = df[df["CITY"] == c].dropna(subset=["DISTANCE_TO_CITY_CENTER", "LATITUDE", "LONGITUDE"])
        q1, q2 = sub["DISTANCE_TO_CITY_CENTER"].quantile([0.33, 0.66])
        zones = []
        for label, mask in [
            ("Centro",     sub["DISTANCE_TO_CITY_CENTER"] <= q1),
            ("Intermedia", (sub["DISTANCE_TO_CITY_CENTER"] > q1) & (sub["DISTANCE_TO_CITY_CENTER"] <= q2)),
            ("Periferia",  sub["DISTANCE_TO_CITY_CENTER"] > q2),
        ]:
            z = sub[mask]
            zones.append({
                "label": label,
                "lat": round(float(z["LATITUDE"].median()), 5),
                "long": round(float(z["LONGITUDE"].median()), 5),
                "distance_to_center": round(float(z["DISTANCE_TO_CITY_CENTER"].median()), 1),
                "median_eur_m2": int((z[TARGET] / z["CONSTRUCTEDAREA"]).median()),
            })
        city_zones[c] = zones

    metadata = {
        "market": "ES",
        "dataset": "idealista18 (Páez et al., 2024) — listados reales Idealista 2018",
        "dataset_doi": "10.1177/23998083241242844",
        "cities": CITIES,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_total": int(len(df)),
        "metrics": {
            "r2_log": round(r2, 3),
            "r2_eur": round(r2_eur, 3),
            "mae_eur": int(mae),
            "median_abs_error_eur": int(medae),
            "mape_pct": round(mape, 1),
        },
        "city_stats": city_stats,
        "city_zones": city_zones,
        "top_features": importances,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "synthetic": False,
    }
    (ARTIFACTS / "es_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\n✅ Guardado es_gbm_model.joblib + es_metadata.json")
    print(json.dumps(city_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
