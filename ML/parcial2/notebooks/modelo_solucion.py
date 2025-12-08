import os
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field
import kaggle_evaluation.default_inference_server
import time

# ============ PATHS ============
DATA_PATH = Path('/kaggle/input/hull-tactical-market-prediction/')

# ============ MODEL CONFIGS ============
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
    signal_multiplier_low_vol: float = 600.0  # For low V1
    signal_multiplier_high_vol: float = 400.0  # For high V1
    min_signal: float = 0.0
    max_signal: float = 2.0
    vol_scaling: float = 1.2
    retrain_freq: int = 1  # Retrain every row
    missing_threshold: float = 0.5
    max_train_rows: int = 800  # Reduced

# Initialize parameters
params = ModelParameters()

# ============ DATA LOADING AND PREPROCESSING ============
def load_trainset() -> pl.DataFrame:
    df = (
            pl.read_csv(DATA_PATH / "train.csv")
            .rename({'market_forward_excess_returns': 'target'})
            .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
            .filter(pl.col('date_id') >= 37)
            .tail(params.max_train_rows)
            )
    missing_counts = {col: df[col].is_null().mean() for col in df.columns}
    feature_cols = [
            col for col, miss_rate in missing_counts.items() 
            if miss_rate <= params.missing_threshold and col not in ['date_id', 'target']
            ]
    keep_cols = ['date_id', 'target'] + feature_cols
    if len(keep_cols) != len(set(keep_cols)):
        raise ValueError(f"Duplicate columns in keep_cols: {keep_cols}")
    return df.select(keep_cols)

def load_testset() -> pl.DataFrame:
    df = (
            pl.read_csv(DATA_PATH / "test.csv")
            .with_columns(pl.exclude('date_id', 'is_scored').cast(pl.Float64, strict=False))
            )
    train_cols = load_trainset().columns
    feature_cols = [col for col in train_cols if col not in ['date_id', 'target']]
    return df.select(['date_id', 'is_scored', 'lagged_market_forward_excess_returns'] + feature_cols)

def create_features(df: pl.DataFrame, is_train: bool = False, median_values: dict = None) -> pl.DataFrame:
    feature_prefixes = ['D', 'E', 'I', 'M', 'P', 'S', 'V']
    base_features = [col for col in df.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]

    # Derived features
    required_cols = ['I1', 'I2', 'I7', 'I9', 'M11']
    if all(col in base_features for col in required_cols):
        df = df.with_columns(
                (pl.col("I2") - pl.col("I1")).alias("U1"),
                (pl.col("M11") / ((pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3)).alias("U2")
                )
    else:
        df = df.with_columns(
                pl.lit(0.0).alias("U1"),
                pl.lit(0.0).alias("U2")
                )

    # Interaction features
    if 'V1' in base_features and 'S1' in base_features:
        df = df.with_columns((pl.col("V1") * pl.col("S1")).alias("V1_S1"))
    if 'M11' in base_features and 'V1' in base_features:
        df = df.with_columns((pl.col("M11") * pl.col("V1")).alias("M11_V1"))
    if 'I9' in base_features and 'S1' in base_features:
        df = df.with_columns((pl.col("I9") * pl.col("S1")).alias("I9_S1"))

    # Training-only features
    if is_train:
        if 'S1' in base_features:
            df = df.with_columns(pl.col("S1").shift(1).alias("S1_lag1"))
        if 'P1' in base_features:
            df = df.with_columns(pl.col("P1").shift(1).alias("P1_lag1"))
        if 'I9' in base_features:
            df = df.with_columns(pl.col("I9").shift(1).alias("I9_lag1"))
        if 'V1' in base_features:
            df = df.with_columns(pl.col("V1").rolling_mean(window_size=5).alias("V1_roll_mean_5"))
        if 'target' in df.columns:
            df = df.with_columns(pl.col("target").rolling_std(window_size=5).alias("target_roll_std_5"))

    # Impute missing values
    for col in base_features:
        if col.startswith('I'):
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).forward_fill()).fill_null(pl.col(col).backward_fill()))
        median = median_values.get(col, df[col].median()) if median_values else df[col].median()
        df = df.with_columns(pl.col(col).fill_null(median if median is not None else 0.0))

    # Impute derived and additional features
    derived_features = ["U1", "U2", "V1_S1", "M11_V1", "I9_S1"]
    additional_features = ["S1_lag1", "P1_lag1", "I9_lag1", "V1_roll_mean_5", "target_roll_std_5"] if is_train else []
    for col in derived_features + additional_features:
        if col in df.columns:
            median = median_values.get(col, df[col].median()) if median_values else df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median if median is not None else 0.0))

    # Feature list (exclude training-only features)
    features = base_features + derived_features
    select_cols = ["date_id"] + features + (["target"] if is_train else [])
    return df.select(select_cols)

# ============ MODEL TRAINING ============
start_time = time.time()
train = load_trainset()
train = create_features(train, is_train=True)
features = [col for col in train.columns if col not in ['date_id', 'target', 'S1_lag1', 'P1_lag1', 'I9_lag1', 'V1_roll_mean_5', 'target_roll_std_5']]

# Cache median values for imputation (only for features used in model)
median_values = {col: train[col].median() if col in train.columns and train[col].is_null().mean() < 1.0 else 0.0 for col in features}

# Check for NaNs
X_train = train.select(features).to_pandas()
if X_train.isna().any().any():
    raise ValueError(f"NaNs found in X_train for columns: {X_train.columns[X_train.isna().any()].tolist()}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = train['target'].to_pandas()

# Train ElasticNet
enet_model = ElasticNet(alpha=params.enet_alpha, l1_ratio=params.enet_l1_ratio, max_iter=1000000)
enet_model.fit(X_train, y_train)

# Train XGBoost
xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params.xgb_n_estimators,
        max_depth=params.xgb_max_depth,
        learning_rate=params.xgb_learning_rate,
        random_state=42
        )
xgb_model.fit(X_train, y_train)

# Train LightGBM
lgb_model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=params.lgb_n_estimators,
        max_depth=params.lgb_max_depth,
        learning_rate=params.lgb_learning_rate,
        random_state=42
        )
lgb_model.fit(X_train, y_train)

# Check startup time
if time.time() - start_time > 900:
    raise RuntimeError("Startup time exceeded 900 seconds")

# State for online learning
previous_lagged = None
test_row_count = 0
last_allocation = 0.0
v1_median = train['V1'].median() if 'V1' in train.columns else 0.0

# ============ VOLATILITY ESTIMATION ============
def estimate_volatility(test: pl.DataFrame, train: pl.DataFrame) -> float:
    vol = test['V1'][0] if 'V1' in test.columns else (train['target'].tail(params.vol_window).std() or 0.01)
    recent_returns = train['target'].tail(params.vol_window).to_numpy()
    if len(recent_returns) > 1:
        garch_vol = np.sqrt(0.3 * np.var(recent_returns) + 0.7 * vol**2)
        return max(garch_vol, 0.01)
    return max(vol, 0.01)

# ============ PREDICTION FUNCTION ============
def predict(test: pl.DataFrame) -> float:
    global previous_lagged, train, enet_model, xgb_model, lgb_model, scaler, test_row_count, last_allocation, v1_median

    # Online learning: Update training data
    if previous_lagged is not None and 'lagged_market_forward_excess_returns' in previous_lagged.columns:
        append_row = previous_lagged.with_columns(
                pl.col('lagged_market_forward_excess_returns').alias('target')
                )
        append_row = create_features(append_row, is_train=False, median_values=median_values)
        if append_row.height > 0:
            append_row = append_row.with_columns(pl.lit(None).cast(pl.Float64).alias('target'))
            train = train.vstack(append_row.select(['date_id', 'target'] + features))
            if train.height > params.max_train_rows:
                train = train.tail(params.max_train_rows)

        # Retrain every `retrain_freq` rows
        if test_row_count % params.retrain_freq == 0:
            X_train = scaler.fit_transform(train.select(features).to_pandas())
            y_train = train['target'].to_pandas()
            if y_train.isna().any():
                raise ValueError("NaNs found in y_train during retraining")
            enet_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            lgb_model.fit(X_train, y_train)

    # Preprocess test data
    test = create_features(test, is_train=False, median_values=median_values)

    # Ensure no NaNs in test data
    X_test = test.select(features).to_pandas()
    if X_test.isna().any().any():
        raise ValueError(f"NaNs found in X_test for columns: {X_test.columns[X_test.isna().any()].tolist()}")

    X_test = scaler.transform(X_test)

    # Ensemble prediction
    enet_pred = enet_model.predict(X_test)[0]
    xgb_pred = xgb_model.predict(X_test)[0]
    lgb_pred = lgb_model.predict(X_test)[0]
    raw_pred = (params.ensemble_weights['enet'] * enet_pred +
                params.ensemble_weights['xgb'] * xgb_pred +
                params.ensemble_weights['lgb'] * lgb_pred)

    # Estimate volatility and dynamic signal multiplier
    vol = estimate_volatility(test, train)
    signal_multiplier = params.signal_multiplier_low_vol if ('V1' in test.columns and test['V1'][0] < v1_median) else params.signal_multiplier_high_vol

    # Convert to signal
    signal = np.clip(
            raw_pred * signal_multiplier,
            params.min_signal,
            params.max_signal
            )

    # Volatility-adjusted allocation
    allocation = min(params.max_signal, max(params.min_signal, signal / (vol * params.vol_scaling)))

    # Smooth allocation
    transaction_cost = 0.00003  # Reduced to 0.003%
    allocation = (0.75 * allocation + 0.25 * last_allocation) * (1 - transaction_cost)  # Adjusted smoothing
    last_allocation = allocation

    # Update state
    previous_lagged = test
    test_row_count += 1

    return float(allocation)

# ============ LAUNCH SERVER ============
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway((str(DATA_PATH),))
