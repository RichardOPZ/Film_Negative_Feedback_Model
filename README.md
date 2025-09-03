# Film_Negative_Feedback_Model

🎬 Film Junky Union — Clasificación Automática de Reseñas Negativas

Film Junky Union es una comunidad vanguardista para los aficionados del cine clásico. Como parte de su misión por mejorar la experiencia de los usuarios, se ha desarrollado un sistema de clasificación automática de reseñas negativas utilizando un conjunto de datos de IMDB.

🧠 Objetivo del Proyecto

Entrenar un modelo de clasificación binaria capaz de detectar automáticamente las reseñas negativas con un rendimiento de al menos F1-score ≥ 0.85, usando técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático supervisado.

📊 Datos Utilizados

Fuente: Reseñas de películas de IMDB

Etiquetas: positive (1) y negative (0)

Preprocesamiento:

Limpieza de HTML y caracteres especiales

Normalización de texto

Vectorización con TF-IDF usando n-gramas y stopwords en inglés

🔍 Modelos Evaluados

Se entrenaron y evaluaron múltiples modelos de clasificación:

DummyClassifier (baseline)

Logistic Regression

LightGBMClassifier

BERT (modelo 9) — solo a nivel experimental

📌 Observaciones:

BERT (modelo 9) mostró predicciones excesivamente polarizadas, otorgando calificaciones extremas incluso en reseñas con tonos intermedios. Esto, junto con el alto costo computacional, lo hace poco adecuado para esta tarea específica.

Modelos 2 y 4 mostraron un buen desempeño, aunque con una tendencia a subestimar la polaridad de ciertas reseñas.

Modelo 3 (LightGBMClassifier) resultó ser el más balanceado y efectivo, logrando los mejores puntajes en accuracy, F1, average precision score y ROC AUC, cumpliendo con el objetivo del proyecto.

🏆 Resultados
Métrica	Valor (Modelo 3)
F1 Score	≥ 0.85 ✅
Accuracy	🔼 Alta
ROC AUC	🔼 Alta
Average Precision Score	🔼 Alta
⚙️ Librerías Utilizadas
import math
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import sklearn.metrics as metrics
from lightgbm import LGBMClassifier
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

🚀 Conclusión

El sistema desarrollado permite clasificar automáticamente reseñas de películas, destacando aquellas con contenido negativo, con alta precisión y eficiencia. La solución implementada es escalable y lista para integrarse en entornos reales, como parte de las funcionalidades de Film Junky Union para mejorar la experiencia del usuario.