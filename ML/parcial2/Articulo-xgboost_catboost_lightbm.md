# Apilamiento de Ensambles: Combinando XGBoost, LightGBM y CatBoost para Mejorar el Rendimiento del Modelo
**Por Stephen Echessa** *Publicado el 29 de julio, 2024 – 8 min de lectura*

En el mundo en constante evolución del aprendizaje automático, donde numerosos algoritmos
compiten por la supremacía, los **ensambles apilados (stacked ensembles)** destacan como una
técnica robusta que combina las fortalezas de múltiples modelos para ofrecer un rendimiento
predictivo superior.

En este artículo, exploraremos el concepto de stacking, por qué es beneficioso, y veremos una
guía paso a paso para implementarlo en Python usando tres algoritmos de última generación:
**XGBoost, CatBoost y LightGBM**.

## ¿Qué es el Stacking y por qué usarlo?

Imagina una carrera de relevos donde cada corredor sobresale en un segmento distinto de la
pista.
Esa es la esencia del **stacking**:
una técnica de aprendizaje en ensamble que busca mejorar el poder predictivo de los modelos
combinando múltiples modelos individuales, también llamados *aprendices base* (base learners).

La idea central es que, al aprovechar las fortalezas y compensar las debilidades de distintos
modelos, se puede crear un modelo final que supere a cualquiera de ellos por separado.

Cada modelo individual tiene méritos y limitaciones, y en distintos problemas uno puede superar
al otro.
El stacking resuelve esto alimentando las predicciones de los modelos base a un modelo de nivel
superior llamado **meta-aprendiz** (*meta-learner* o *blender*), que combina esas entradas para
generar la predicción final.

### Ventajas del stacking:

* **Mayor precisión:** al agregar las predicciones de múltiples modelos se reduce el sesgo
  individual y se mejora el rendimiento.
* **Más robustez:** los ensambles apilados tienden a generalizar mejor y reducen el riesgo de
  *overfitting*.
* **Flexibilidad:** se pueden combinar modelos muy distintos (árboles, redes, regresiones),
  capturando patrones más diversos en los datos.

## Los Titanes: XGBoost, CatBoost y LightGBM

Estos tres algoritmos son ampliamente reconocidos en tareas de regresión por su gran desempeño.

* **XGBoost (Extreme Gradient Boosting):** rápido, escalable, robusto y popular en competencias
  de Kaggle.
* **CatBoost (Categorical Boosting):** maneja automáticamente variables categóricas, evitando
  mucha preprocesamiento.
  Es muy resistente al sobreajuste.
* **LightGBM (Light Gradient Boosting Machine):** diseñado para velocidad y escalabilidad en
  datasets masivos, eficiente tanto en entrenamiento como en predicción.

## ¿Cómo funciona el Stacking?

El proceso se puede resumir en estos pasos:

1. **Preprocesar y dividir los datos:** limpiar, seleccionar variables y separar en
   entrenamiento y validación.
2. **Entrenar los modelos base:** aquí usaremos XGBoost, CatBoost y LightGBM.
3. **Generar predicciones:** los modelos base generan predicciones en el set de validación;
   esas predicciones serán las entradas del meta-modelo.
4. **Entrenar el meta-modelo:** algoritmos simples como regresión lineal, lasso, ridge o
   elastic net suelen usarse aquí.
5. **Predicción final:** el ensamble entrenado se usa sobre datos nuevos.

En scikit-learn, si activamos `passthrough=True`, también podemos pasar las variables
originales junto con las predicciones de los modelos base al meta-modelo.

## Implementación Paso a Paso en Python

Usaremos el dataset de **Boston Housing**, con 14 atributos.

### 1. Cargar los datos

```python
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

boston = fetch_openml('boston')

boston.frame.info()
```

### 2. Preprocesamiento

Convertimos las variables categóricas a enteros.

```python
boston.data['RAD'] = boston.data['RAD'].astype(int).apply(lambda x: 0 if x==24 else x)
boston.data['CHAS'] = boston.data['CHAS'].astype(int)

boston.data.info()
```

### 3. Entrenar los modelos base

Primero instalamos dependencias:

```bash
pip install xgboost lightgbm catboost
```

Luego importamos y preparamos:

```python
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.2, random_state=101
)

# Función de evaluación
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    print('----------------------------------------------\n')
    print('Train R2 Score: ', round(r2_score(y_true=y_train, y_pred=train_pred), 6))
    print('Train RMSE: ', sqrt(mean_squared_error(y_true=y_train, y_pred=train_pred)))
    print('----------------------------------------------\n')
    test_pred = model.predict(X_test)
    print('Test R2 Score: ', round(r2_score(y_true=y_test, y_pred=test_pred), 6))
    print('Test RMSE: ', sqrt(mean_squared_error(y_true=y_test, y_pred=test_pred)))
```

### 4. Resultados de los modelos base

**XGBoost**

```python
xgb_model = XGBRegressor(random_state=101)
xgb_model.fit(X_train, y_train)
evaluate_model(xgb_model, X_train, y_train, X_test, y_test)
```

* Train R²:
  0.9999 – RMSE:
  0.017
* Test R²:
  0.8601 – RMSE:
  3.96

**LightGBM**

```python
light_gbm = LGBMRegressor(random_state=101, verbosity=-1)
light_gbm.fit(X_train, y_train)
evaluate_model(light_gbm, X_train, y_train, X_test, y_test)
```

* Train R²:
  0.9767 – RMSE:
  1.34
* Test R²:
  0.8403 – RMSE:
  4.23

**CatBoost**

```python
cat_model = CatBoostRegressor(random_state=101, verbose=0, cat_features=['CHAS', 'RAD'])
cat_model.fit(X_train, y_train)
evaluate_model(cat_model, X_train, y_train, X_test, y_test)
```

* Train R²:
  0.9936 – RMSE:
  0.70
* Test R²:
  0.8861 – RMSE:
  3.57

### 5. Modelo Apilado (Stacked Regression)

```python
estimators = [
    ('xgb', xgb_model),
    ('lgb', light_gbm),
    ('cat', cat_model),
]

stack = StackingRegressor(estimators=estimators, final_estimator=ElasticNetCV())
stack.fit(X_train, y_train)
evaluate_model(stack, X_train, y_train, X_test, y_test)
```

* Train R²:
  0.9927 – RMSE:
  0.74
* Test R²:
  0.8894 – RMSE:
  3.52

Mejora leve sobre CatBoost.

### 6. Reducción de complejidad

**Stacking sin XGBoost**

```python
estimators = [
    ('lgb', light_gbm),
    ('cat', cat_model),
]

stack = StackingRegressor(estimators=estimators, final_estimator=ElasticNetCV())
stack.fit(X_train, y_train)
evaluate_model(stack, X_train, y_train, X_test, y_test)
```

* Test R²:
  0.8905 – RMSE:
  3.50

**Stacking sin LightGBM**

```python
estimators = [
    ('xgb', xgb_model),
    ('cat', cat_model),
]

stack = StackingRegressor(estimators=estimators, final_estimator=ElasticNetCV())
stack.fit(X_train, y_train)
evaluate_model(stack, X_train, y_train, X_test, y_test)
```

* Test R²:
  0.8914 – RMSE:
  3.49

El rendimiento mejora aún más al reducir complejidad.

## Conclusiones

Los ensambles apilados son una técnica poderosa que puede mejorar significativamente el
rendimiento combinando algoritmos distintos.

* Al apilar XGBoost, CatBoost y LightGBM logramos un modelo más **preciso, robusto y
  eficiente**.
* Sin embargo, el stacking implica **más complejidad, mayor costo computacional** y menor
  interpretabilidad.

Aun así, si dudas entre qué modelo usar, recuerda:
**no tienes que elegir, apílalos.**
