# predict_and_evaluate_hull.py → COMO TU SCRIPT, PERO CON EL MODELO HULL TACTICAL
import joblib
import numpy as np
import pandas as pd
import polars as pl
import warnings
from dataclasses import dataclass, field

# Silenciar warnings molestos
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Fix para joblib
@dataclass
class ModelParameters:
    ensemble_weights: dict = field(default_factory=lambda: {'enet': 0.45, 'xgb': 0.55, 'lgb': 0.45})
    signal_multiplier_low_vol: float = 600.0
    signal_multiplier_high_vol: float = 400.0
    vol_scaling: float = 1.2

# Cargar modelo
model = joblib.load("hull_model.pkl")
scaler = model["scaler"]
enet = model["enet"]
xgb = model["xgb"]
lgb = model["lgb"]
features = model["features"]
medians = model["medians"]
params = model["params"]
v1_median = model.get("v1_median", 0.0)

print(f"Modelo HULL TACTICAL cargado → {len(features)} features")

def add_features(df: pl.DataFrame) -> pl.DataFrame:
    num_cols = [c for c in df.columns if c[0] in "DEIMPSV"]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in num_cols])

    if all(c in df.columns for c in ["I1","I2","I7","I9","M11"]):
        avg = (pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3
        df = df.with_columns([
            (pl.col("I2") - pl.col("I1")).alias("U1"),
            (pl.col("M11") / (avg + 1e-8)).alias("U2")
        ])
    else:
        df = df.with_columns([pl.lit(0.0).alias("U1"), pl.lit(0.0).alias("U2")])

    for a, b, name in [("V1","S1","V1_S1"), ("M11","V1","M11_V1"), ("I9","S1","I9_S1")]:
        if a in df.columns and b in df.columns:
            df = df.with_columns((pl.col(a) * pl.col(b)).alias(name))

    for c in num_cols:
        df = df.with_columns(pl.col(c).fill_null(medians.get(c, 0.0)))

    return df

def make_predictions(csv_path: str):
    try:
        df = pl.read_csv(csv_path)
    except:
        df = pl.read_csv("../data/" + csv_path.split("/")[-1])

    # Limpiar columnas de retorno
    if "market_forward_excess_returns" in df.columns:
        df = df.rename({"market_forward_excess_returns": "target"})
    if "forward_returns" in df.columns: df = df.drop("forward_returns")
    if "risk_free_rate" in df.columns: df = df.drop("risk_free_rate")

    df = add_features(df)

    X = df.select(features).to_pandas()[features].astype(np.float64).fillna(0).values
    X_scaled = scaler.transform(X)

    pred_enet = enet.predict(X_scaled)
    pred_xgb = xgb.predict(X_scaled)
    pred_lgb = lgb.predict(X_scaled)

    raw_pred = (params.ensemble_weights["enet"] * pred_enet +
                params.ensemble_weights["xgb"] * pred_xgb +
                params.ensemble_weights["lgb"] * pred_lgb)

    vol = max(float(df["V1"].cast(pl.Float64).mean() or 0.01), 0.01)
    use_low = "V1" in df.columns and float(df["V1"][0]) < v1_median
    multiplier = params.signal_multiplier_low_vol if use_low else params.signal_multiplier_high_vol

    position = np.clip(raw_pred * multiplier / (vol * params.vol_scaling), 0.0, 2.0)

    result = pd.DataFrame({
        "date_id": df["date_id"].to_list(),
        "position": position
    })

    if "is_scored" in df.columns:
        mask = df["is_scored"].cast(pl.Boolean).to_numpy()
        result = result[mask].reset_index(drop=True)
        if "target" in df.columns:
            excess = df["target"].cast(pl.Float64).to_numpy()[mask]
            result["excess_return"] = excess
            result["strategy_return"] = result["position"] * excess
    else:
        if "target" in df.columns:
            excess = df["target"].cast(pl.Float64).to_numpy()
            result["excess_return"] = excess
            result["strategy_return"] = result["position"] * excess

    return result

# ======================= EJECUCIÓN =======================
print("\nPredicciones en test.csv")
test_preds = make_predictions("test.csv")
print(f"{len(test_preds)} días puntuables")
print(test_preds[["date_id", "position"]].head(10))
test_preds[["date_id", "position"]].to_csv("submission.csv", index=False)
print("submission.csv guardado")

print("\nEvaluando en train.csv (todo el histórico)")
train_eval = make_predictions("train.csv")

if "strategy_return" in train_eval.columns:
    r = train_eval["strategy_return"]
    b = train_eval["excess_return"]

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    market_vol = b.std() * np.sqrt(252)
    vol_ratio = ann_vol / (1.2 * market_vol)
    penalty = max(vol_ratio - 1, 0) ** 2
    adj_sharpe = sharpe * (1 - penalty)
    mae = np.mean(np.abs(r - b))
    rmse = np.sqrt(np.mean((r - b)**2))

    print("\n" + "="*60)
    print(" RESULTADOS FINALES - HULL TACTICAL ENSAMBLE")
    print("="*60)
    print(f"Retorno anualizado      : {ann_ret*100:6.2f}%")
    print(f"Volatilidad anualizada  : {ann_vol*100:6.2f}%")
    print(f"Sharpe Ratio            : {sharpe:.4f}")
    print(f"Sharpe Ajustado (120%)  : {adj_sharpe:.4f}")
    print(f"MAE vs Benchmark        : {mae:.6f}")
    print(f"RMSE vs Benchmark       : {rmse:.6f}")
    print(f"Vol vs límite 120%      : {vol_ratio:.3f}x → penalización {penalty:.1%}")
    print(f"Días evaluados          : {len(r)}")
    print("="*60)
    print("¡AHORA SÍ PODÉS COMPARAR CON TU MODELO!")
else:
    print("No hay retornos (es test.csv)")

print("\nScript terminado sin errores")
