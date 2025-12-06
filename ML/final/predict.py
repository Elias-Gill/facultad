# predict_and_evaluate.py  ← VERSIÓN FINAL QUE FUNCIONA 100%
import joblib
import numpy as np
import pandas as pd
import polars as pl

# Cargar modelo
model = joblib.load("final_model.pkl")
scaler = model["scaler"]
enet = model["enet"]
lgb = model["lgb"]
meta = model["meta"]
features = model["features"]

print(f"Modelo cargado → {len(features)} features esperadas")

def prepare_features(df: pl.DataFrame) -> np.ndarray:
    df = df.sort("date_id")
    
    # Seleccionar solo las features que el modelo conoce
    X_pl = df.select(features)
    
    # Rellenar NaNs con 0 (exactamente como hiciste en entrenamiento)
    X_pl = X_pl.fill_null(0)
    
    # Convertir a pandas y luego a numpy (float64 puro)
    X = X_pl.to_pandas()
    X = X[features]  # fuerza orden exacto
    X = X.astype(np.float64)  # ← clave para evitar errores de tipo
    X = X.fillna(0)           # doble seguridad
    
    # Verificación final
    assert X.shape[1] == len(features)
    assert X.isna().sum().sum() == 0
    assert np.all(np.isfinite(X.values))
    
    return X.values  # devolvemos directamente el array numpy

def make_predictions(csv_path: str):
    df = pl.read_csv(csv_path)
    X = prepare_features(df)
    
    # Predicciones
    X_scaled = scaler.transform(X)
    pred_enet = enet.predict(X_scaled)
    pred_lgb = lgb.predict(X)
    stacked = np.column_stack((pred_enet, pred_lgb))
    position = np.clip(meta.predict(stacked).flatten(), 0.0, 2.0)
    
    result = pd.DataFrame({
        "date_id": df["date_id"].to_numpy(),
        "position": position
    })
    
    # Filtrar por is_scored si existe
    if "is_scored" in df.columns:
        scored_mask = df["is_scored"].to_numpy().astype(bool)
        result = result[scored_mask].reset_index(drop=True)
        # También filtramos retornos si existen
        if "market_forward_excess_returns" in df.columns:
            excess = df["market_forward_excess_returns"].to_numpy()[scored_mask]
            result["excess_return"] = excess
            result["strategy_return"] = result["position"] * excess
    else:
        # Si no hay is_scored (es train.csv), usamos todo
        if "market_forward_excess_returns" in df.columns:
            excess = df["market_forward_excess_returns"].to_numpy()
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

    print("\n" + "="*55)
    print("       RESULTADOS FINALES - TU ESTRATEGIA")
    print("="*55)
    print(f"Retorno anualizado     : {ann_ret*100:6.2f}%")
    print(f"Volatilidad anualizada : {ann_vol*100:6.2f}%")
    print(f"Sharpe Ratio           : {sharpe:.3f}")
    print(f"Sharpe Ajustado        : {adj_sharpe:.3f}")
    print(f"MAE vs Benchmark       : {mae:.6f}")
    print(f"RMSE vs Benchmark      : {rmse:.6f}")
    print(f"Vol vs límite 120%     : {vol_ratio:.2f}x → penalización: {penalty:.1%}")
    print("="*55)
    print("¡TODO LISTO PARA TU INFORME!")
else:
    print("No hay retornos (es test.csv)")

print("\nScript terminado sin errores")
