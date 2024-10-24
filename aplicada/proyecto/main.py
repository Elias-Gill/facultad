# --------------------------------------------------------------------
# Importación de bibliotecas necesarias

import re
import time

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# NOTE: descomentar estas lineas si es la primera vez que se corre el programa
# import nltk
# nltk.download('vader_lexicon')

# --------------------------------------------------------------------
# Carga de datos y configuración inicial

inicio_tiempo = time.time()

# Cargar conjunto de datos de entrenamiento
conjunto_entrenamiento = pd.read_csv("./dataset/test_data.csv", encoding="ISO-8859-1")

# Extraer texto de los tweets y etiquetas de sentimiento
textos_tweets = conjunto_entrenamiento.Sentence
etiquetas_sentimiento = conjunto_entrenamiento.Sentiment

print(f"Número total de tweets: {len(textos_tweets)}")

# --------------------------------------------------------------------
# Generar variables del universo y funciones de pertenencia difusa

# Rangos para positivo y negativo: [0, 1]
# Rango para salida: [0, 10] en puntos porcentuales
x_positivo = np.arange(0, 1, 0.1)
x_negativo = np.arange(0, 1, 0.1)
x_salida = np.arange(0, 10, 1)

# Funciones de pertenencia difusa para positivo
positivo_bajo = fuzz.trimf(x_positivo, [0, 0, 0.5])
positivo_medio = fuzz.trimf(x_positivo, [0, 0.5, 1])
positivo_alto = fuzz.trimf(x_positivo, [0.5, 1, 1])

# Funciones de pertenencia difusa para negativo
negativo_bajo = fuzz.trimf(x_negativo, [0, 0, 0.5])
negativo_medio = fuzz.trimf(x_negativo, [0, 0.5, 1])
negativo_alto = fuzz.trimf(x_negativo, [0.5, 1, 1])

# Funciones de pertenencia difusa para salida
salida_negativa = fuzz.trimf(x_salida, [0, 0, 5])  # Escala: Neg Neu Pos
salida_neutral = fuzz.trimf(x_salida, [0, 5, 10])
salida_positiva = fuzz.trimf(x_salida, [5, 10, 10])

# --------------------------------------------------------------------
# Preprocesamiento de texto y análisis de sentimiento

tweets_procesados = []
sentimientos = []
resultados_sentimiento = []
documentos_sentimiento = []

for i in range(len(textos_tweets)):
    tweet_original = conjunto_entrenamiento.Sentence[i]
    tweet_minusc = tweet_original.lower()
    tweets_procesados.append(tweet_minusc)  # Convertido a minúsculas
    sentimientos.append(conjunto_entrenamiento.Sentiment[i])


def descontraccion(frase):  # Preprocesamiento de texto
    # Específico
    frase = re.sub(r"won't", "will not", frase)
    frase = re.sub(r"can\'t", "can not", frase)
    frase = re.sub(r"@", "", frase)  # Eliminación de @
    frase = re.sub(r"http\S+", "", frase)  # Eliminación de URLs
    frase = re.sub(r"#", "", frase)  # Procesamiento de hashtags

    # General
    frase = re.sub(r"n\'t", " not", frase)
    frase = re.sub(r"\'re", " are", frase)
    frase = re.sub(r"\'s", " is", frase)
    frase = re.sub(r"\'d", " would", frase)
    frase = re.sub(r"\'ll", " will", frase)
    frase = re.sub(r"\'t", " not", frase)
    frase = re.sub(r"\'ve", " have", frase)
    frase = re.sub(r"\'m", " am", frase)
    return frase


for k in range(len(textos_tweets)):
    tweets_procesados[k] = descontraccion(tweets_procesados[k])

analizador_sentimiento = SentimentIntensityAnalyzer()

for j in range(len(textos_tweets)):
    documentos_sentimiento.append(sentimientos[j])
    puntuaciones = analizador_sentimiento.polarity_scores(tweets_procesados[j])
    puntuacion_positiva = puntuaciones["pos"]
    puntuacion_negativa = puntuaciones["neg"]
    puntuacion_neutral = puntuaciones["neu"]
    puntuacion_compuesta = puntuaciones["compound"]

    print(f"{j + 1} {tweets_procesados[j]:-<65} {str(puntuaciones)}")

    print("\nPuntuación positiva para cada tweet:")
    if puntuacion_positiva == 1:
        puntuacion_positiva = 0.9
    else:
        puntuacion_positiva = round(puntuacion_positiva, 1)
    print(puntuacion_positiva)

    print("\nPuntuación negativa para cada tweet:")
    if puntuacion_negativa == 1:
        puntuacion_negativa = 0.9
    else:
        puntuacion_negativa = round(puntuacion_negativa, 1)
    print(puntuacion_negativa)

    # Necesitamos la activación de nuestras funciones de pertenencia difusa en estos valores.
    nivel_positivo_bajo = fuzz.interp_membership(
        x_positivo, positivo_bajo, puntuacion_positiva
    )
    nivel_positivo_medio = fuzz.interp_membership(
        x_positivo, positivo_medio, puntuacion_positiva
    )
    nivel_positivo_alto = fuzz.interp_membership(
        x_positivo, positivo_alto, puntuacion_positiva
    )

    nivel_negativo_bajo = fuzz.interp_membership(
        x_negativo, negativo_bajo, puntuacion_negativa
    )
    nivel_negativo_medio = fuzz.interp_membership(
        x_negativo, negativo_medio, puntuacion_negativa
    )
    nivel_negativo_alto = fuzz.interp_membership(
        x_negativo, negativo_alto, puntuacion_negativa
    )

    # Ahora aplicamos nuestras reglas y las aplicamos. La regla 1 concierne a comida mala o buena.
    # El operador OR significa que tomamos el máximo de estas dos.
    regla_activa_1 = np.fmin(nivel_positivo_bajo, nivel_negativo_bajo)
    regla_activa_2 = np.fmin(nivel_positivo_medio, nivel_negativo_bajo)
    regla_activa_3 = np.fmin(nivel_positivo_alto, nivel_negativo_bajo)
    regla_activa_4 = np.fmin(nivel_positivo_bajo, nivel_negativo_medio)
    regla_activa_5 = np.fmin(nivel_positivo_medio, nivel_negativo_medio)
    regla_activa_6 = np.fmin(nivel_positivo_alto, nivel_negativo_medio)
    regla_activa_7 = np.fmin(nivel_positivo_bajo, nivel_negativo_alto)
    regla_activa_8 = np.fmin(nivel_positivo_medio, nivel_negativo_alto)
    regla_activa_9 = np.fmin(nivel_positivo_alto, nivel_negativo_alto)

    # Ahora aplicamos esto cortando la parte superior de la función de pertenencia correspondiente con `np.fmin`

    n1 = np.fmax(regla_activa_4, regla_activa_7)
    n2 = np.fmax(n1, regla_activa_8)
    activacion_salida_bajo = np.fmin(n2, salida_negativa)

    neu1 = np.fmax(regla_activa_1, regla_activa_5)
    neu2 = np.fmax(neu1, regla_activa_9)
    activacion_salida_medio = np.fmin(neu2, salida_neutral)

    p1 = np.fmax(regla_activa_2, regla_activa_3)
    p2 = np.fmax(p1, regla_activa_6)
    activacion_salida_alto = np.fmin(p2, salida_positiva)

    salida_cero = np.zeros_like(x_salida)

    # Agregar todas tres funciones de pertenencia de salida juntas
    agregada = np.fmax(
        activacion_salida_bajo, np.fmax(activacion_salida_medio, activacion_salida_alto)
    )

    # Calcular resultado desdifuso
    salida = fuzz.defuzz(x_salida, agregada, "centroid")
    resultado = round(salida, 2)

    # --------------------------------------------------------------------
    # Visualización de la actividad de pertenencia de salida

    print("\nFuerza de activación de Negativa (wneg): " + str(round(n2, 4)))
    print("Fuerza de activación de Neutra (wneu): " + str(round(neu2, 4)))
    print("Fuerza de activación de Positiva (wpos): " + str(round(p2, 4)))

    print("\nConsecuentes MF resultantes:")
    print("activacion_salida_bajo: " + str(activacion_salida_bajo))
    print("activacion_salida_medio: " + str(activacion_salida_medio))
    print("activacion_salida_alto: " + str(activacion_salida_alto))

    print("\nSalida agregada: " + str(agregada))

    print("\nSalida desdifusa: " + str(resultado))

    # Escala : Neg Neu Pos
    if 0 < (resultado) < 3.33:  # R
        print("\nSalida después de la defuzzificación: Negativa")
        resultados_sentimiento.append("Negativa")

    elif 3.34 < (resultado) < 6.66:
        print("\nSalida después de la defuzzificación: Neutra")
        resultados_sentimiento.append("Neutra")

    elif 6.67 < (resultado) < 10:
        print("\nSalida después de la defuzzificación: Positiva")
        resultados_sentimiento.append("Positiva")

    print("Sentimiento del documento: " + str(sentimientos[j]) + "\n")

# --------------------------------------------------------------------
# Evaluación del rendimiento del modelo

contador = 0
for k in range(len(textos_tweets)):
    if documentos_sentimiento[k] == resultados_sentimiento[k]:
        contador += 1

# Precisión global
precision_global = accuracy_score(documentos_sentimiento, resultados_sentimiento)
print(f"Precisión global: {round(precision_global * 100, 2)}%")

# Informe de clasificación detallado
print("\nInforme de clasificación:")
print(classification_report(documentos_sentimiento, resultados_sentimiento))

# Matriz de confusión
print("\nMatriz de confusión:")
print(confusion_matrix(documentos_sentimiento, resultados_sentimiento))

# Métricas macro
precicion_macro = precision_score(
    documentos_sentimiento, resultados_sentimiento, average="macro"
)
recall_macro = recall_score(
    documentos_sentimiento, resultados_sentimiento, average="macro"
)
f1_macro = f1_score(documentos_sentimiento, resultados_sentimiento, average="macro")

print(f"\nPuntuación de precisión (MACRO): {round(precicion_macro * 100, 2)}%")
print(f"Puntuación de recuerdo (MACRO): {round(recall_macro * 100, 2)}%")
print(f"Puntuación F1 (MACRO): {round(f1_macro * 100, 2)}%")

# Métricas micro
f1_micro = f1_score(documentos_sentimiento, resultados_sentimiento, average="micro")
print(f"Puntuación F1 (MICRO): {round(f1_micro * 100, 2)}%")

# --------------------------------------------------------------------
# Tiempo de ejecución

fin_tiempo = time.time()
tiempo_ejecucion = fin_tiempo - inicio_tiempo
print(f"Tiempo de ejecución: {round(tiempo_ejecucion, 3)} segundos")
