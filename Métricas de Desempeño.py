#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:27:59 2023

@author: valeriarrondon
"""

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from pgmpy.models import BayesianNetwork

df=pd.read_csv("/Users/valeriarrondon/Documents/Octavo Semestre/Analítica Computacional/Módulo 3/Proyecto 3/datosRedIcfes.csv", header=0)

print(df.isnull().any())

df = df.drop('fami_estratovivienda', axis=1)
df = df.drop('cole_genero', axis=1)
df = df.drop('cole_mcpio_ubicacion', axis=1)


# Definir la estructura de la red bayesiana
model = BayesianNetwork([
                          ( "cole_depto_ubicacion" , "punt_global" ),
                          ( "cole_area_ubicacion" , "punt_global" ),
                          ( "cole_caracter" , "punt_global" ),
                          (  "cole_naturaleza", "punt_global" ),
                          ( "cole_jornada", "punt_global"  ),

])

emv = MaximumLikelihoodEstimator( model = model , data = df )
# Estimar para nodos sin padres
# Estimar para nodo depto
cpdem_depto = emv.estimate_cpd( node ="cole_depto_ubicacion")
print( cpdem_depto )


cpdem_are = emv.estimate_cpd( node ="cole_area_ubicacion")
print( cpdem_are )

# Estimar para nodo cole caracter
cpdem_caract = emv.estimate_cpd( node ="cole_caracter")
print( cpdem_caract )

# Estimar para nodo cole naturaleza
cpdem_natu = emv.estimate_cpd( node ="cole_naturaleza")
print( cpdem_natu )

cpdem_jor = emv.estimate_cpd( node ="cole_jornada")
print( cpdem_jor )



model.fit(data=df , estimator = MaximumLikelihoodEstimator)

for i in model.nodes():
    print(model.get_cpds(i) )
    
import itertools

# Dividir los datos en conjuntos de entrenamiento y prueba
entrenamiento, prueba = train_test_split(df, test_size=0.2, random_state=42)


# Se realiza la inferencia sobre el modelo 
# Se colocan las predicciones en una lista

inference = VariableElimination(model)
#predict= []
predicciones = []

for i, row in prueba.iterrows():
    query = inference.query(variables=['punt_global'], evidence={
        'cole_depto_ubicacion': row['cole_depto_ubicacion'],
        'cole_area_ubicacion': row['cole_area_ubicacion'],
        'cole_caracter': row['cole_caracter'],
        'cole_naturaleza': row['cole_naturaleza'],
        'cole_jornada': row['cole_jornada']
    })
    
    
    # Num
    max_query = None
    max_prob = 0
    
    for i, prob in enumerate(query.values):
        if prob > max_prob:
            max_query = i
            max_prob = prob
    predicciones.append(max_query)



reales = prueba["punt_global"].tolist()



import pandas as pd

# Función para convertir los valores en rangos
def convertir_a_rango(valor):
    if valor >= 100 and valor <= 200:
        return '100-200'
    elif valor > 200 and valor <= 300:
        return '200-300'
    elif valor > 300 and valor <= 400:
        return '300-400'
    else:
        return None

# Convertir las etiquetas de prueba a objetos Series de Pandas
y_test_series = pd.Series(reales)

# Convertir las etiquetas predichas a objetos Series de Pandas
y_pred_series = pd.Series(predicciones)

# Convertir las etiquetas de prueba a rangos
y_test_rangos = y_test_series.apply(convertir_a_rango)

# Convertir las etiquetas predichas a rangos
y_pred_rangos = y_pred_series.apply(convertir_a_rango)

# Calcular los verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos
verdaderos_positivos = sum((y_test_rangos == '200-300') & (y_pred_rangos == '200-300'))
verdaderos_negativos = sum((y_test_rangos != '200-300') & (y_pred_rangos != '200-300'))
falsos_positivos = sum((y_test_rangos != '200-300') & (y_pred_rangos == '200-300'))
falsos_negativos = sum((y_test_rangos == '200-300') & (y_pred_rangos != '200-300'))

print("Verdaderos Positivos:", verdaderos_positivos)
print("Verdaderos Negativos:", verdaderos_negativos)
print("Falsos Positivos:", falsos_positivos)
print("Falsos Negativos:", falsos_negativos)


from sklearn import metrics
Accuracy = metrics.accuracy_score(y_test_rangos, predicciones)
Precision = metrics.precision_score(y_test_rangos, predicciones, average ='macro')
Sensitivity_recall = metrics.recall_score(reales, predicciones, average ='macro')
F1_score = metrics.f1_score(reales, predicciones, average ='macro')

print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"F1_score":F1_score})


from sklearn import metrics

report = metrics.classification_report(y_test_rangos, y_pred_rangos)
print(report)


