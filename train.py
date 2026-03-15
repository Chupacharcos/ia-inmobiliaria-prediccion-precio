"""
Entrenamiento del modelo de predicción de precio inmobiliario.
Dataset: King County House Sales (21.493 transacciones reales, EE.UU. 2014-2015)

Ejecutar una vez offline:
  cd /var/www/proyecto-inmobiliario
  source /var/www/chatbot/venv/bin/activate
  python3 train.py

Genera en artifacts/:
  gbm_model.joblib  — HistGradientBoostingRegressor
  mlp_model.pt      — Red neuronal MLP PyTorch
  scaler.joblib     — StandardScaler para MLP
  metadata.json     — métricas, feature importances, fecha
  test_data.pkl     — muestra de test para generar imagen de portada
"""

import pandas as pd
import numpy as np
import joblib
import json
import requests
import io
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'yr_built', 'yr_renovated_flag',
    'lat', 'long', 'sqft_living15', 'zipcode_encoded',
    'age', 'renovated', 'rooms_total', 'sqft_per_room',
]
TARGET = 'price'

# ── URLs del dataset (varios mirrors) ────────────────────────────────────────
DATA_URLS = [
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/house_prices.csv",
    "https://raw.githubusercontent.com/rashida048/Datasets/master/home_data.csv",
    "https://raw.githubusercontent.com/gscdit/House-Price-Prediction/master/kc_house_data.csv",
]


class PricePredictor(nn.Module):
    """MLP 3 capas, ~25K params. Inferencia CPU < 5ms."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def download_data() -> pd.DataFrame:
    print("  Descargando dataset King County House Sales...")
    for url in DATA_URLS:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # Normalizar nombres de columnas
            df.columns = df.columns.str.lower().str.strip()
            if 'price' in df.columns and 'sqft_living' in df.columns:
                print(f"  Dataset descargado: {len(df):,} filas desde {url[:60]}...")
                return df
        except Exception as e:
            print(f"  Mirror fallido ({url[:60]}...): {e}")
            continue
    raise RuntimeError("No se pudo descargar el dataset de ningún mirror.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Limpiar
    df = df[df['price'] > 50_000]
    df = df[df['price'] < 3_000_000]
    df = df[df['sqft_living'] > 100]
    df = df.dropna(subset=['price', 'sqft_living', 'bedrooms', 'lat', 'long'])

    # Año de renovación → flag binario + antigüedad
    df['yr_renovated_flag'] = (df.get('yr_renovated', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    df['renovated'] = df['yr_renovated_flag']

    # Edad de la vivienda
    current_year = 2015  # año del dataset
    df['age'] = current_year - df['yr_built'].clip(1800, 2015)

    # Habitaciones totales
    df['rooms_total'] = df['bedrooms'].clip(0, 15) + df['bathrooms'].clip(0, 10)

    # m² por habitación
    df['sqft_per_room'] = (df['sqft_living'] / df['rooms_total'].replace(0, 1)).clip(0, 5000)

    # Target encoding de zipcode (media de precio por zipcode)
    zip_mean = df.groupby('zipcode')['price'].mean()
    df['zipcode_encoded'] = df['zipcode'].map(zip_mean).fillna(df['price'].mean())

    # Rellenar NaN en otras features
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0

    return df


def train_gbm(X_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingRegressor:
    print("  Entrenando HistGradientBoostingRegressor...")
    gbm = HistGradientBoostingRegressor(
        max_iter=600,
        max_depth=8,
        learning_rate=0.04,
        min_samples_leaf=10,
        max_leaf_nodes=63,
        l2_regularization=0.05,
        random_state=42,
        verbose=0,
    )
    gbm.fit(X_train, y_train)
    return gbm


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    scaler: StandardScaler,
) -> PricePredictor:
    print("  Entrenando MLP PyTorch...")

    X_tr_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_tr_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    ds_train = TensorDataset(
        torch.FloatTensor(X_tr_s),
        torch.FloatTensor(y_tr_n),
    )
    dl_train = DataLoader(ds_train, batch_size=512, shuffle=True)

    model = PricePredictor(input_dim=X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(80):
        model.train()
        for xb, yb in dl_train:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.FloatTensor(X_val_s)).numpy()
                val_pred_real = val_pred * y_std + y_mean
                val_mae = mean_absolute_error(y_val, val_pred_real)
            print(f"    Epoch {epoch+1:3d} — val MAE: ${val_mae:,.0f}")
            if val_mae < best_val_loss:
                best_val_loss = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Guardar normalización junto al modelo
    model.y_mean = float(y_mean)
    model.y_std = float(y_std)

    return model


def main():
    print("\n=== Entrenamiento: Predicción de Precio Inmobiliario ===\n")

    # 1. Datos
    df_raw = download_data()
    df = engineer_features(df_raw)
    print(f"  Filas tras limpieza: {len(df):,}")

    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    # Log-transform price: reduces skew and improves R² significantly
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.15, random_state=42
    )
    # Also keep real-scale test labels
    _, _, y_train_real, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X_train, y_train_log, test_size=0.12, random_state=42
    )
    _, _, y_train_real_full, y_val_real = train_test_split(
        X_train, np.expm1(y_train_log), test_size=0.12, random_state=42
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # 2. Scaler (para MLP)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 3. GBM (trained on log-price, evaluated on real scale)
    gbm = train_gbm(X_train, y_train_log)
    gbm_pred = np.expm1(gbm.predict(X_test))
    gbm_mae = mean_absolute_error(y_test, gbm_pred)
    gbm_r2 = r2_score(y_test, gbm_pred)
    print(f"  GBM — MAE: ${gbm_mae:,.0f}  R²: {gbm_r2:.4f}")

    # 4. MLP (trained on log-price too)
    mlp = train_mlp(X_train, y_train_log, X_val, y_val_log, scaler)
    mlp.eval()
    X_test_s = scaler.transform(X_test)
    with torch.no_grad():
        mlp_pred_n = mlp(torch.FloatTensor(X_test_s)).numpy()
        mlp_pred_log = mlp_pred_n * mlp.y_std + mlp.y_mean
        mlp_pred = np.expm1(mlp_pred_log)
    mlp_mae = mean_absolute_error(y_test, mlp_pred)
    mlp_r2 = r2_score(y_test, mlp_pred)
    print(f"  MLP — MAE: ${mlp_mae:,.0f}  R²: {mlp_r2:.4f}")

    # 5. Feature importances del GBM
    from sklearn.inspection import permutation_importance
    print("  Calculando feature importances (permutation)...")
    perm = permutation_importance(gbm, X_val, y_val_log, n_repeats=5, random_state=42, n_jobs=-1)
    importances = dict(zip(FEATURES, perm.importances_mean.tolist()))
    # Normalizar a porcentaje
    total = sum(max(v, 0) for v in importances.values())
    importances_pct = {k: round(max(v, 0) / total * 100, 1) for k, v in importances.items()}

    # 6. Guardar artifacts
    print("\n  Guardando artifacts...")
    joblib.dump(gbm, ARTIFACTS / "gbm_model.joblib", compress=3)
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")

    torch.save({
        'state_dict': mlp.state_dict(),
        'y_mean': mlp.y_mean,
        'y_std': mlp.y_std,
        'input_dim': X_train.shape[1],
    }, ARTIFACTS / "mlp_model.pt")

    metadata = {
        'fecha_entrenamiento': datetime.now().isoformat(),
        'n_samples_train': int(len(X_train)),
        'n_samples_test': int(len(X_test)),
        'n_samples_total': int(len(df)),
        'dataset': 'King County House Sales (Washington State, USA)',
        'features': FEATURES,
        'gbm_mae': round(float(gbm_mae), 0),
        'gbm_r2': round(float(gbm_r2), 4),
        'mlp_mae': round(float(mlp_mae), 0),
        'mlp_r2': round(float(mlp_r2), 4),
        'feature_importances': importances_pct,
        'price_min': round(float(y.min()), 0),
        'price_max': round(float(y.max()), 0),
        'price_mean': round(float(y.mean()), 0),
        'price_median': round(float(np.median(y)), 0),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    # Guardar muestra de test para imagen de portada
    joblib.dump({'y_test': y_test, 'gbm_pred': gbm_pred, 'mlp_pred': mlp_pred}, ARTIFACTS / "test_data.pkl")

    print(f"\n  Artifacts guardados en {ARTIFACTS}")
    print(f"\n  RESULTADOS FINALES:")
    print(f"    GBM → MAE=${gbm_mae:,.0f}  R²={gbm_r2:.3f}")
    print(f"    MLP → MAE=${mlp_mae:,.0f}  R²={mlp_r2:.3f}")
    print("\n  Entrenamiento completado.\n")


if __name__ == "__main__":
    main()
