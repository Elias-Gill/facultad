import os
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field
import joblib
import time

# ============ CONFIG ============
DATA_PATH = Path(".")  # cambia si tus csv están en otro lado
TRAIN_CSV = DATA_PATH / "train.csv"
MAX_TRAIN_ROWS = 800
MISSING_THRESHOLD = 0.5

@dataclass
class ModelParameters:
    enet_alpha: float = 0.01
    enet_l1_ratio: float = 0.5
    xgb_n_estimators: int = 350
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 8
    lgb_learning_rate: float = 0.05
    ensemble_weights: dict = field(default_factory=lambda: {'enet': 0.45, 'xgb': 0.55, 'lgb': 0.45})
    vol_window: int = 20
    signal_multiplier_low_vol: float = 600.0
    signal_multiplier_high_vol: float = 400.0
    vol_scaling: float = 1.2

params = ModelParameters()

# ============ CARGA Y FEATURES ============
def load_train():
    df = (
        pl.read_csv(TRAIN_CSV)
        # Tu CSV tiene forward_returns como retorno total y market_forward_excess_returns como exceso
        .rename({"market_forward_excess_returns": "target"})   # ← esta sí existe
        .drop(["forward_returns", "risk_free_rate"])           # ← quitamos las que no usamos
        .with_columns(pl.exclude("date_id").cast(pl.Float64, strict=False))
        .filter(pl.col("date_id") >= 37)
        .tail(MAX_TRAIN_ROWS)
    )

    # Calculamos missing solo en columnas que realmente existen
    missing = {c: df[c].is_null().mean() for c in df.columns if c not in ["date_id", "target"]}
    features = [
        c for c, m in missing.items()
        if m <= MISSING_THRESHOLD and c not in ["date_id", "target"]
    ]

    return df.select(["date_id", "target"] + features), features

def add_features(df: pl.DataFrame, is_train: bool = True, medians: dict = None):
    base = [c for c in df.columns if c[0] in "DEIMPSV"]

    # features derivados simples
    if all(c in df.columns for c in ["I1","I2","I7","I9","M11"]):
        df = df.with_columns([
            (pl.col("I2") - pl.col("I1")).alias("U1"),
            (pl.col("M11") / ((pl.col("I2")+pl.col("I9")+pl.col("I7"))/3)).alias("U2")
        ])
    else:
        df = df.with_columns([pl.lit(0.0).alias("U1"), pl.lit(0.0).alias("U2")])

    # interacciones
    for a,b,name in [("V1","S1","V1_S1"), ("M11","V1","M11_V1"), ("I9","S1","I9_S1")]:
        if a in df.columns and b in df.columns:
            df = df.with_columns((pl.col(a)*pl.col(b)).alias(name))

    # imputación
    for c in base:
        if c.startswith("I"):
            df = df.with_columns(pl.col(c).forward_fill().backward_fill())
        med = medians.get(c, df[c].median()) if medians else df[c].median()
        df = df.with_columns(pl.col(c).fill_null(med if med is not None else 0))

    derived = [c for c in df.columns if c.startswith(("U","V1_S1","M11_V1","I9_S1"))]
    final_features = base + derived
    cols = ["date_id"] + final_features + (["target"] if is_train else [])
    return df.select(cols), final_features

# ============ ENTRENAMIENTO ============
start = time.time()
train_df, base_features = load_train()
train_df, features = add_features(train_df, is_train=True)
medians = {c: train_df[c].median() for c in base_features}

X = train_df.select(features).to_pandas().astype(float)
y = train_df["target"].to_pandas()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# modelos
enet = ElasticNet(alpha=params.enet_alpha, l1_ratio=params.enet_l1_ratio, max_iter=1_000_000, random_state=42)
enet.fit(X_scaled, y)

xgb_model = xgb.XGBRegressor(
    n_estimators=params.xgb_n_estimators,
    max_depth=params.xgb_max_depth,
    learning_rate=params.xgb_learning_rate,
    random_state=42,
    objective="reg:squarederror"
)
xgb_model.fit(X_scaled, y)

lgb_model = lgb.LGBMRegressor(
    n_estimators=params.lgb_n_estimators,
    max_depth=params.lgb_max_depth,
    learning_rate=params.lgb_learning_rate,
    random_state=42,
    objective="regression",
    verbosity=-1
)
lgb_model.fit(X_scaled, y)

print(f"Entrenamiento terminado en {time.time()-start:.1f}s")

# guardado
joblib.dump({
    "scaler": scaler,
    "enet": enet,
    "xgb": xgb_model,
    "lgb": lgb_model,
    "features": features,
    "medians": medians,
    "params": params,
    "v1_median": train_df["V1"].median() if "V1" in train_df.columns else 0.0
}, "hull_model.pkl")

print("Modelo guardado como hull_model.pkl")
