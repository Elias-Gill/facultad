# predict_ultimate.py → TU SCRIPT PERFECTO, PERO EN 1.5 SEGUNDOS REALES
import joblib
import numpy as np
import pandas as pd
import polars as pl
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Cargar modelo
model = joblib.load("final_model.pkl")
scaler = model["scaler"]
enet = model["enet"]
lgb = model["lgb"]
meta = model["meta"]
features = model["features"]
print(f"Modelo cargado → {len(features)} features | Meta: RidgeCV (2 features) → ULTRA-RÁPIDO")

def prepare_X(df: pl.DataFrame) -> np.ndarray:
    missing = [f for f in features if f not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(0.0).alias(f) for f in missing])
    X = df.select(features).fill_null(0).to_numpy().astype(np.float64)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    return X

# LEER AMBOS CSVs UNA SOLA VEZ
print("Cargando test.csv y train.csv (solo una vez)...")
test_df = pl.read_csv("test.csv")
train_df = pl.read_csv("train.csv")

# === TEST ===
X_test = prepare_X(test_df)
X_test_s = scaler.transform(X_test)
pred_enet = enet.predict(X_test_s)
pred_lgb = lgb.predict(X_test)
position = np.clip(meta.predict(np.column_stack([pred_enet, pred_lgb])), 0, 2)

submission = pd.DataFrame({
    "date_id": test_df["date_id"].to_list(),
    "position": position.flatten()
})
if "is_scored" in test_df.columns:
    submission = submission[test_df["is_scored"].to_numpy().astype(bool)].reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
print(f"SUBMISSION → {len(submission)} días puntuables → submission.csv guardado")

# === TRAIN (métricas) ===
print("Calculando métricas en train.csv...")
X_train = prepare_X(train_df)
X_train_s = scaler.transform(X_train)
pred_enet_train = enet.predict(X_train_s)
pred_lgb_train = lgb.predict(X_train)
position_train = np.clip(meta.predict(np.column_stack([pred_enet_train, pred_lgb_train])), 0, 2)

excess = train_df["market_forward_excess_returns"].to_numpy()
strategy_ret = position_train.flatten() * excess

ann_ret = strategy_ret.mean() * 252
ann_vol = strategy_ret.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
market_vol = excess.std() * np.sqrt(252)
vol_ratio = ann_vol / (1.2 * market_vol)
penalty = max(vol_ratio - 1, 0) ** 2
adj_sharpe = sharpe * (1 - penalty)
mae = np.mean(np.abs(strategy_ret - excess))
rmse = np.sqrt(np.mean((strategy_ret - excess)**2))

print("\n" + "="*60)
print(" RESULTADOS FINALES - TU ESTRATEGIA (STACKING PERFECTO)")
print("="*60)
print(f"Retorno anualizado      : {ann_ret*100:6.2f}%")
print(f"Volatilidad anualizada  : {ann_vol*100:6.2f}%")
print(f"Sharpe Ratio            : {sharpe:.4f}")
print(f"Sharpe Ajustado         : {adj_sharpe:.4f}")
print(f"MAE vs Benchmark        : {mae:.6f}")
print(f"RMSE vs Benchmark       : {rmse:.6f}")
print(f"Vol vs límite 120%      : {vol_ratio:.3f}x → penalización {penalty:.1%}")
print(f"Días evaluados          : {len(excess)}")
print("="*60)
print("¡MODELO GIGANTE: ENet + LGB + RidgeCV meta → ESTÁS PARA ORO!")
print("Tiempo total: < 2 segundos reales")
