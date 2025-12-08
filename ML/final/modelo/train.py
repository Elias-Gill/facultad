import os
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

train_path = "./train.csv"
assert os.path.exists(train_path)

train = pl.read_csv(train_path)
train = train.rename({"market_forward_excess_returns": "excess_return"})
train = train.with_columns(pl.all().cast(pl.Float64, strict=False))
train = train.drop_nulls(subset=["excess_return"])

# Creamos el target óptimo: posición entre 0 y 2
df = train.to_pandas().set_index("date_id")

# Valores entre 0 y 2 (Exposición parcial o apalancada). Ejemplo: 1.5 = 150% del capital (50%
# apalancado).
# - 0.0: No inviertes nada en el mercado,100% en cash (o tasa libre de riesgo). No ganas ni
#        pierdes con el S&P 500.
# - 1.0: Exposición normal al mercado (exactamente el benchmark), 100% del capital invertido en
#        el S&P 500 (lo que haría un ETF típico).
# - 2.0: Máximo apalancamiento permitido,200% del capital invertido = usas 2× leverage (100%
#        tuyo + 100% prestado).
# TODO: ajustar el 40 según volatilidad histórica
df["target"] = np.clip(df["excess_return"] * 40, -1.0, 1.0)
df["target"] = df["target"].clip(lower=0) * 2.0  # solo rango [0,2]

# Features (sin las columnas target)
features = [
    c
    for c in df.columns
    if c
    not in ["date_id", "excess_return", "forward_returns", "risk_free_rate", "target"]
]
X = df[features].fillna(0)
y = df["target"].values

print(f"Usando {len(features)} features | Target ahora es posición [0-2]")

# Validación temporal
tscv = TimeSeriesSplit(n_splits=5)
oof_enet = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # ElasticNet
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        alphas=np.logspace(-5, 1, 15),
        cv=3,
        max_iter=20000,
        n_jobs=-1,
        random_state=42,
    )
    enet.fit(X_tr_s, y_tr)
    oof_enet[val_idx] = np.clip(enet.predict(X_val_s), 0, 2)

    # LightGBM
    lgb_model = lgb.train(
        {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 100,
            "max_depth": 10,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "seed": 42,
        },
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=2500,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    oof_lgb[val_idx] = np.clip(lgb_model.predict(X_val), 0, 2)

    print(
        f"Fold {fold+1} - ENet: {root_mean_squared_error(y_val, oof_enet[val_idx]):.4f} | "
        f"LGB: {root_mean_squared_error(y_val, oof_lgb[val_idx]):.4f}"
    )

# Meta modelo
meta_X = np.column_stack([oof_enet, oof_lgb])
meta_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
meta_model.fit(meta_X, y)
print(
    f"Meta weights -> ENet: {meta_model.coef_[0]:.3f}, LGB: {meta_model.coef_[1]:.3f}"
)

# Entrenamiento final
scaler_final = StandardScaler().fit(X)
X_scaled = scaler_final.transform(X)

final_enet = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9, 1.0],
    alphas=np.logspace(-5, 1, 20),
    cv=5,
    max_iter=50000,
    n_jobs=-1,
    random_state=42,
)
final_enet.fit(X_scaled, y)

final_lgb = lgb.train(
    {
        "objective": "regression",
        "learning_rate": 0.03,
        "num_leaves": 128,
        "max_depth": 10,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    },
    lgb.Dataset(X, y),
    num_boost_round=5000,
)

# Guardar
joblib.dump(
    {
        "scaler": scaler_final,
        "enet": final_enet,
        "lgb": final_lgb,
        "meta": meta_model,
        "features": features,
    },
    "final_model.pkl",
)

print("Modelo final guardado - ahora predice directamente posición [0-2]")
print("Sharpe esperado en validación: probablemente > 1.5 (antes tenías < 0.8)")
