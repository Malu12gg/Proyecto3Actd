#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 19:15:44 2023

@author: valeriarrondon
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer


df=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 3/Proyecto 3/datosRedIcfes.csv", header=0)
df = df.drop('fami_estratovivienda', axis=1)
df = df.drop('cole_genero', axis=1)
df = df.drop('cole_mcpio_ubicacion', axis=1)


# Definir los intervalos y categorías para la variable objetivo
intervalos = [100, 200, 300, 400]
categorias = ['100-200', '200-300', '300-400']


X = df.drop('punt_global', axis=1)
y = df['punt_global']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir las columnas categóricas
columnas_categoricas = ['cole_area_ubicacion', 'cole_caracter', 'cole_depto_ubicacion', 'cole_jornada', 'cole_naturaleza']

# Crear una instancia del codificador one-hot
encoder = OneHotEncoder(sparse=False)

# Crear una transformación para aplicar el codificador solo a las columnas categóricas
transformer = ColumnTransformer([('encoder', encoder, columnas_categoricas)], remainder='passthrough')

# Ajustar el codificador y transformar las columnas categóricas en dummies
X_train_encoded = transformer.fit_transform(X_train)
X_test_encoded = transformer.transform(X_test)

# Crear una instancia del modelo de regresión logística
model = LogisticRegression()

# Ajustar el modelo utilizando los datos de entrenamiento
model.fit(X_train_encoded, y_train)

# Realizar la predicción en los datos de prueba
y_pred = model.predict(X_test_encoded)

# Agrupar la variable objetivo en intervalos
# Agrupar la variable objetivo en intervalos
y_test_intervalos = pd.cut(y_test, bins=intervalos, labels=categorias)
y_pred_intervalos = pd.cut(y_pred, bins=intervalos, labels=categorias)

# Convertir las etiquetas a valores numéricos utilizando LabelEncoder
le = LabelEncoder()
y_test_numerico = le.fit_transform(y_test_intervalos)
y_pred_numerico = le.transform(y_pred_intervalos)

# Calcular la precisión de la predicción
accuracy = accuracy_score(y_test_numerico, y_pred_numerico)
print('Precisión:', accuracy)

# Calcular la precisión de la predicción
accuracy = accuracy_score(y_test_intervalos, y_pred_intervalos)
print('Precisión:', accuracy)


