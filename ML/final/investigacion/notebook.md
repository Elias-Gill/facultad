---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  nbformat: 4
  nbformat_minor: 0
---

``` python
from google.colab import files
uploaded = files.upload()
```

``` python
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# --- 1. Carga y Preparación Inicial de Datos ---
df = pd.read_csv('commodity_futures.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print("Datos cargados y convertidos a series de tiempo.")
print(df.head())
print("\nInformación de nulos inicial:")
print(df.isnull().sum())

# --- 2. Manejo de Valores Faltantes (ffill) ---
df.fillna(method='ffill', inplace=True)
print("\nNulos después de ffill:")
print(df.isnull().sum().sum())

# --- 3. Feature Engineering ---
# Variable objetivo: diferencial Cobre - Aluminio al día siguiente
df['Spread_Cu_Al'] = df['COPPER'] - df['ALUMINIUM']
df['Target_Spread'] = df['Spread_Cu_Al'].shift(-1)
df.dropna(subset=['Target_Spread'], inplace=True)
print("\nVariable objetivo (Target_Spread) creada.")

# Parámetros nuevos: LAGS hasta 15 y ventana MA = 15
n_lags = 15
window = 15

# Crear lags para las mismas variables que usaste antes (Spread, COPPER, GOLD)
for lag in range(1, n_lags + 1):
    df[f'Lag_{lag}_Spread'] = df['Spread_Cu_Al'].shift(lag)
    df[f'Lag_{lag}_Copper'] = df['COPPER'].shift(lag)
    df[f'Lag_{lag}_USD'] = df['GOLD'].shift(lag)

# Medias móviles y volatilidad usando ventana de 15 días
df['MA_Spread'] = df['Spread_Cu_Al'].rolling(window=window).mean()
df['Volatilidad_Spread'] = df['Spread_Cu_Al'].rolling(window=window).std()
df['Ratio_Spread_MA'] = df['Spread_Cu_Al'] / df['MA_Spread']
df['MA_Copper'] = df['COPPER'].rolling(window=window).mean()

# Limpieza final (elimina NaN generados por lags y rollings)
df.dropna(inplace=True)
print(f"Dataset final con {len(df)} filas después de Feature Engineering y limpieza.")
print(df.tail())

# --- 4. División de Datos Cronológica (80/10/10) ---
data_size = len(df)
train_size = int(0.80 * data_size)
validation_size = int(0.10 * data_size)
test_size = data_size - train_size - validation_size

train_df = df.iloc[:train_size]
validation_df = df.iloc[train_size : train_size + validation_size]
test_df = df.iloc[train_size + validation_size :]

# Definición de X (Features) y Y (Target)
features = [col for col in df.columns if col not in ['Target_Spread', 'Spread_Cu_Al']]
X_train, y_train = train_df[features], train_df['Target_Spread']
X_val, y_val = validation_df[features], validation_df['Target_Spread']
X_test, y_test = test_df[features], test_df['Target_Spread']

print(f"\nConjunto de Entrenamiento (80%): {len(X_train)} días (datos más antiguos)")
print(f"Conjunto de Validación (10%): {len(X_val)} días")
print(f"Conjunto de Prueba (10%): {len(X_test)} días (datos más recientes)")

# --- 5. Estandarización de Características ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# --- 6. Ajuste de Modelo: Random Forest Regressor ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)
rf_model = RandomForestRegressor(random_state=42)

print("\nIniciando Grid Search con Validación Cruzada Expansiva...")
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
final_rf_model = grid_search.best_estimator_

print("--- Resultados del Grid Search ---")
print(f"Mejores Hiperparámetros: {best_params}")
print(f"Mejor RMSE (en CV): {sqrt(-best_score):.4f}")
print("-----------------------------------")

# --- 7. Evaluación Final del Modelo ---
y_pred_test = final_rf_model.predict(X_test_scaled)

rmse = sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("\n--- Métricas de Desempeño en el Conjunto de Prueba ---")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")
print("-------------------------------------------------------")

# Importancia de variables
importances = final_rf_model.feature_importances_
feature_names = X_train.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 20 Variables más Importantes ---")
print(feature_importance_df.head(20))

# Visualización de la Predicción
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Real (Target_Spread)')
plt.plot(y_test.index, y_pred_test, label='Predicción RF', alpha=0.7)
plt.title('Predicción del Diferencial de Precios (Test Set) - Lags 1..15')
plt.xlabel('Fecha')
plt.ylabel('Spread de Precios')
plt.legend()
plt.grid(True)
plt.show()

# Función de Evaluación
def evaluate_model(y_true, y_pred, model_name):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results = {
        'Modelo': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    return results

# Baseline (Last Value)
y_pred_baseline = test_df['Spread_Cu_Al'].values
baseline_results = evaluate_model(y_test, y_pred_baseline, "Baseline (Last Value)")

print("\n--- Evaluación del Baseline (Last Value) ---")
print(f"RMSE: {baseline_results['RMSE']:.4f}")
print(f"MAE: {baseline_results['MAE']:.4f}")
print(f"R2: {baseline_results['R2']:.4f}")

# ARIMA (walk-forward) usando 'Spread_Cu_Al' historia
y_train_arima = train_df['Spread_Cu_Al']
y_test_arima = test_df['Spread_Cu_Al']
order = (2, 1, 1)

y_pred_arima = []
history = [x for x in y_train_arima.values]

print("\nIniciando predicción Walk-Forward con ARIMA (2,1,1)...")
for t in range(len(y_test_arima)):
    try:
        model = sm.tsa.arima.ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)[0]
        y_pred_arima.append(yhat)
        obs = y_test_arima.values[t]
        history.append(obs)
    except Exception as e:
        y_pred_arima.append(history[-1])

arima_results = evaluate_model(y_test.values, y_pred_arima, "ARIMA (2,1,1) Regressor")

print("\n--- Evaluación del Modelo ARIMA ---")
print(f"RMSE: {arima_results['RMSE']:.4f}")
print(f"MAE: {arima_results['MAE']:.4f}")
print(f"R2: {arima_results['R2']:.4f}")

# Resumen comparativo
rf_results = {
    'Modelo': 'Random Forest Regressor (Óptimo)',
    'RMSE': rmse,
    'MAE': mae,
    'R2': r2
}

final_summary = pd.DataFrame([rf_results, baseline_results, arima_results])
final_summary = final_summary.sort_values(by='RMSE')

print("\n=======================================================")
print("           RESUMEN COMPARATIVO DE MODELOS")
print("=======================================================")
print(final_summary.to_string(index=False, float_format="%.4f"))
print("=======================================================")
```

    Datos cargados y convertidos a series de tiempo.
                NATURAL GAS   GOLD  WTI CRUDE  BRENT CRUDE  SOYBEANS    CORN  \
    Date                                                                       
    2000-01-03          NaN    NaN        NaN          NaN    456.50  200.75   
    2000-01-04        2.176  283.7      25.55        24.39    464.25  203.00   
    2000-01-05        2.168  282.1      24.91        23.73    469.25  203.00   
    2000-01-06        2.196  282.4      24.78        23.62    468.00  203.75   
    2000-01-07        2.173  282.9      24.22        23.09    471.50  207.00   

                COPPER  SILVER  LOW SULPHUR GAS OIL  LIVE CATTLE  ...     ZINC  \
    Date                                                          ...            
    2000-01-03     NaN     NaN                  NaN       69.700  ...  1237.50   
    2000-01-04  0.8480   5.375               213.50       69.075  ...  1215.00   
    2000-01-05  0.8565   5.210               213.00       68.975  ...  1209.25   
    2000-01-06  0.8530   5.167               211.25       70.075  ...  1212.00   
    2000-01-07  0.8540   5.195               205.25       70.875  ...  1209.25   

                ULS DIESEL  NICKEL   WHEAT  SUGAR  GASOLINE  COFFEE  LEAN HOGS  \
    Date                                                                         
    2000-01-03         NaN  8446.0  247.50   6.10       NaN  116.50     55.975   
    2000-01-04       67.78  8314.0  247.25   5.77       NaN  116.25     55.625   
    2000-01-05       66.55  8307.0  249.75   5.81       NaN  118.60     55.075   
    2000-01-06       66.28  8252.0  248.50   5.77       NaN  116.85     55.175   
    2000-01-07       64.75  8174.0  251.75   5.84       NaN  114.15     55.625   

                HRW WHEAT  COTTON  
    Date                           
    2000-01-03     274.25   51.07  
    2000-01-04     274.00   50.73  
    2000-01-05     276.25   51.56  
    2000-01-06     275.00   52.08  
    2000-01-07     277.75   53.96  

    [5 rows x 23 columns]

    Información de nulos inicial:
    NATURAL GAS               9
    GOLD                     10
    WTI CRUDE                 9
    BRENT CRUDE               2
    SOYBEANS                  9
    CORN                      8
    COPPER                    9
    SILVER                   13
    LOW SULPHUR GAS OIL       2
    LIVE CATTLE               8
    SOYBEAN OIL              13
    ALUMINIUM                47
    SOYBEAN MEAL             10
    ZINC                     46
    ULS DIESEL                9
    NICKEL                   46
    WHEAT                    14
    SUGAR                     9
    GASOLINE               1491
    COFFEE                   12
    LEAN HOGS                 8
    HRW WHEAT                14
    COTTON                   13
    dtype: int64

    Nulos después de ffill:
    1492

    Variable objetivo (Target_Spread) creada.
    Dataset final con 4608 filas después de Feature Engineering y limpieza.
                NATURAL GAS    GOLD  WTI CRUDE  BRENT CRUDE  SOYBEANS    CORN  \
    Date                                                                        
    2023-07-28        2.613  1960.0      79.96        84.14   1506.75  528.75   
    2023-07-31        2.634  1970.5      81.80        85.56   1445.75  504.00   
    2023-08-01        2.560  1940.7      81.37        84.91   1446.75  497.00   
    2023-08-02        2.477  1937.4      79.49        83.20   1429.75  488.25   
    2023-08-03        2.565  1932.0      81.55        85.14   1428.75  480.75   

                COPPER  SILVER  LOW SULPHUR GAS OIL  LIVE CATTLE  ...  \
    Date                                                          ...   
    2023-07-28  3.9130  24.475               878.75      178.650  ...   
    2023-07-31  4.0080  24.972               877.25      178.050  ...   
    2023-08-01  3.9085  24.326               911.50      179.500  ...   
    2023-08-02  3.8435  23.872               884.50      178.025  ...   
    2023-08-03  3.8995  23.697               899.50      178.500  ...   

                Lag_14_Spread  Lag_14_Copper  Lag_14_USD  Lag_15_Spread  \
    Date                                                                  
    2023-07-28     -2099.9685         3.7715      1931.0     -2087.7360   
    2023-07-31     -2124.4965         3.7535      1937.1     -2099.9685   
    2023-08-01     -2194.9080         3.8420      1961.7     -2124.4965   
    2023-08-02     -2235.0705         3.9295      1963.8     -2194.9080   
    2023-08-03     -2235.0785         3.9215      1962.5     -2235.0705   

                Lag_15_Copper  Lag_15_USD    MA_Spread  Volatilidad_Spread  \
    Date                                                                     
    2023-07-28         3.7640      1935.4 -2177.294633           36.070943   
    2023-07-31         3.7715      1931.0 -2187.463533           34.164832   
    2023-08-01         3.7535      1937.1 -2193.640533           30.101612   
    2023-08-02         3.8420      1961.7 -2191.724433           30.918837   
    2023-08-03         3.9295      1963.8 -2188.326433           28.522785   

                Ratio_Spread_MA  MA_Copper  
    Date                                    
    2023-07-28         0.996579   3.846033  
    2023-07-31         1.029732   3.861800  
    2023-08-01         1.010718   3.872133  
    2023-08-02         0.988339   3.872233  
    2023-08-03         0.998069   3.870233  

    [5 rows x 74 columns]

    Conjunto de Entrenamiento (80%): 3686 días (datos más antiguos)
    Conjunto de Validación (10%): 460 días
    Conjunto de Prueba (10%): 462 días (datos más recientes)

    Iniciando Grid Search con Validación Cruzada Expansiva...
    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    --- Resultados del Grid Search ---
    Mejores Hiperparámetros: {'max_depth': None, 'max_features': 0.8, 'min_samples_split': 2, 'n_estimators': 100}
    Mejor RMSE (en CV): 177.8487
    -----------------------------------

    --- Métricas de Desempeño en el Conjunto de Prueba ---
    Raíz del Error Cuadrático Medio (RMSE): 102.8022
    Error Absoluto Medio (MAE): 56.3444
    Coeficiente de Determinación (R²): 0.9272
    -------------------------------------------------------

    --- Top 20 Variables más Importantes ---
                   Feature  Importance
    11           ALUMINIUM    0.779897
    23        Lag_1_Spread    0.148861
    26        Lag_2_Spread    0.052151
    29        Lag_3_Spread    0.011576
    0          NATURAL GAS    0.000923
    68           MA_Spread    0.000657
    20           LEAN HOGS    0.000208
    15              NICKEL    0.000200
    17               SUGAR    0.000190
    13                ZINC    0.000189
    9          LIVE CATTLE    0.000171
    22              COTTON    0.000166
    19              COFFEE    0.000164
    69  Volatilidad_Spread    0.000163
    12        SOYBEAN MEAL    0.000162
    10         SOYBEAN OIL    0.000144
    70     Ratio_Spread_MA    0.000140
    7               SILVER    0.000136
    32        Lag_4_Spread    0.000135
    35        Lag_5_Spread    0.000134

![](vertopal_05005f5dddd74181ba1cb6e050b6f292/fd664fe9d79f332e19cc4823a01f2bb342c45ccc.png)


    --- Evaluación del Baseline (Last Value) ---
    RMSE: 49.7445
    MAE: 34.8561
    R2: 0.9830

    Iniciando predicción Walk-Forward con ARIMA (2,1,1)...

    --- Evaluación del Modelo ARIMA ---
    RMSE: 86.4382
    MAE: 54.0159
    R2: 0.9486

    =======================================================
               RESUMEN COMPARATIVO DE MODELOS
    =======================================================
                              Modelo     RMSE     MAE     R2
               Baseline (Last Value)  49.7445 34.8561 0.9830
             ARIMA (2,1,1) Regressor  86.4382 54.0159 0.9486
    Random Forest Regressor (Óptimo) 102.8022 56.3444 0.9272
    =======================================================
