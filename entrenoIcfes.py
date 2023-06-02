import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import LabelEncoder as le
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

from pgmpy.estimators import PC
from pgmpy.sampling import BayesianModelSampling

from pgmpy.estimators import StructureEstimator

import pandas as pd  

from collections import deque
from itertools import permutations

import networkx as nx
from tqdm.auto import trange

from pgmpy.base import DAG
from pgmpy.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)

df=pd.read_csv("C:\\Users\\ricky\\Downloads\\Proyecto3\\datosRedIcfesss.csv", header=0)


#PuntajeGlobal
"""intervalos3 = [0, 100, 200, 300, 400, 500]
categorias3 = ['0-100', '101-200', '201-300', '301- 400', '401-500']
df['punt_global'] = pd.cut(df['punt_global'], bins=intervalos3, labels=categorias3)"""

#print(df)
df.to_csv('datosRedIcfes.csv', index=False)

np.random.seed(42)
#  Dataframe con datos de prueba
df_20 = df.sample(frac=0.2)
#print (df_20)
# Dataframe con datos de entrenamiento
df_80= df.drop(df_20.index)

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
dff= df_80
#Entreno por puntajes
scoring_method = K2Score(data=dff) 
esth = HillClimbSearch(data=dff) 

#prueba 1 
estimated_model = esth.estimate( scoring_method=scoring_method, max_indegree=4,max_iter=int(1e4))
#primer blacklist
blacklistT =[('cole_area_ubicacion', 'cole_mcpio_ubicacion'), ('cole_area_ubicacion', 'cole_jornada'), ('cole_area_ubicacion', 'fami_estratovivienda'), ('cole_area_ubicacion', 'cole_naturaleza'), ('cole_area_ubicacion', 'cole_caracter'), ('cole_caracter', 'cole_genero'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'fami_estratovivienda'),  ('cole_jornada', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_depto_ubicacion'), ('cole_mcpio_ubicacion', 'cole_jornada'), ('cole_mcpio_ubicacion', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_caracter'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_estratovivienda'), ('cole_naturaleza', 'punt_global'), ('cole_naturaleza', 'cole_genero'), ('cole_naturaleza', 'fami_estratovivienda')]
blacklistTT=[('cole_area_ubicacion', 'cole_mcpio_ubicacion'), ('cole_area_ubicacion', 'cole_jornada'), ('cole_area_ubicacion', 'fami_estratovivienda'), ('cole_area_ubicacion', 'cole_naturaleza'), ('cole_area_ubicacion', 'cole_caracter'), ('cole_caracter', 'cole_genero'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'fami_estratovivienda'),  ('cole_jornada', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_depto_ubicacion'), ('cole_mcpio_ubicacion', 'cole_jornada'), ('cole_mcpio_ubicacion', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_caracter'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_estratovivienda'), ('cole_naturaleza', 'punt_global'), ('cole_naturaleza', 'cole_genero'), ('cole_naturaleza', 'fami_estratovivienda'),
 ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_depto_ubicacion'),  ('cole_caracter', 'cole_area_ubicacion'),  ('cole_depto_ubicacion', 'cole_jornada'), ('cole_genero', 'cole_jornada'), ('cole_genero', 'fami_estratovivienda'), ('cole_genero', 'cole_depto_ubicacion'), ('cole_genero', 'cole_naturaleza'), ('cole_genero', 'cole_caracter'), ('cole_jornada', 'cole_mcpio_ubicacion'),  ('cole_jornada', 'cole_area_ubicacion'),  ('cole_naturaleza', 'cole_mcpio_ubicacion'), ('cole_naturaleza', 'cole_caracter'), ('cole_naturaleza', 'cole_depto_ubicacion'), ('cole_naturaleza', 'cole_area_ubicacion'), ('fami_estratovivienda', 'cole_naturaleza'), ('fami_estratovivienda', 'cole_depto_ubicacion'), ('fami_estratovivienda', 'cole_caracter')]
blacklistTTT=[('cole_area_ubicacion', 'cole_mcpio_ubicacion'), ('cole_area_ubicacion', 'cole_jornada'), ('cole_area_ubicacion', 'fami_estratovivienda'), ('cole_area_ubicacion', 'cole_naturaleza'), ('cole_area_ubicacion', 'cole_caracter'), ('cole_caracter', 'cole_genero'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'fami_estratovivienda'),  ('cole_jornada', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_depto_ubicacion'), ('cole_mcpio_ubicacion', 'cole_jornada'), ('cole_mcpio_ubicacion', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_caracter'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_estratovivienda'), ('cole_naturaleza', 'punt_global'), ('cole_naturaleza', 'cole_genero'), ('cole_naturaleza', 'fami_estratovivienda'),
 ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_depto_ubicacion'),  ('cole_caracter', 'cole_area_ubicacion'),  ('cole_depto_ubicacion', 'cole_jornada'), ('cole_genero', 'cole_jornada'), ('cole_genero', 'fami_estratovivienda'), ('cole_genero', 'cole_depto_ubicacion'), ('cole_genero', 'cole_naturaleza'), ('cole_genero', 'cole_caracter'), ('cole_jornada', 'cole_mcpio_ubicacion'),  ('cole_jornada', 'cole_area_ubicacion'),  ('cole_naturaleza', 'cole_mcpio_ubicacion'), ('cole_naturaleza', 'cole_caracter'), ('cole_naturaleza', 'cole_depto_ubicacion'), ('cole_naturaleza', 'cole_area_ubicacion'), ('fami_estratovivienda', 'cole_naturaleza'), ('fami_estratovivienda', 'cole_depto_ubicacion'), ('fami_estratovivienda', 'cole_caracter'), 
 ('cole_area_ubicacion', 'cole_genero'),  ('cole_depto_ubicacion', 'cole_caracter'), ('cole_depto_ubicacion', 'fami_estratovivienda'), ('cole_depto_ubicacion', 'cole_genero'), ('cole_jornada', 'fami_estratovivienda'), ('cole_jornada', 'cole_depto_ubicacion'), ('cole_jornada', 'cole_caracter'), ('fami_estratovivienda', 'cole_mcpio_ubicacion'), ('fami_estratovivienda', 'cole_genero'), ('fami_estratovivienda', 'cole_area_ubicacion'), ('punt_global', 'cole_depto_ubicacion')]
blacklistTTTT=[('cole_area_ubicacion', 'cole_mcpio_ubicacion'), ('cole_area_ubicacion', 'cole_jornada'), ('cole_area_ubicacion', 'fami_estratovivienda'), ('cole_area_ubicacion', 'cole_naturaleza'), ('cole_area_ubicacion', 'cole_caracter'), ('cole_caracter', 'cole_genero'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'fami_estratovivienda'),  ('cole_jornada', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_depto_ubicacion'), ('cole_mcpio_ubicacion', 'cole_jornada'), ('cole_mcpio_ubicacion', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_caracter'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_estratovivienda'), ('cole_naturaleza', 'punt_global'), ('cole_naturaleza', 'cole_genero'), ('cole_naturaleza', 'fami_estratovivienda'),
 ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_depto_ubicacion'),  ('cole_caracter', 'cole_area_ubicacion'),  ('cole_depto_ubicacion', 'cole_jornada'), ('cole_genero', 'cole_jornada'), ('cole_genero', 'fami_estratovivienda'), ('cole_genero', 'cole_depto_ubicacion'), ('cole_genero', 'cole_naturaleza'), ('cole_genero', 'cole_caracter'), ('cole_jornada', 'cole_mcpio_ubicacion'),  ('cole_jornada', 'cole_area_ubicacion'),  ('cole_naturaleza', 'cole_mcpio_ubicacion'), ('cole_naturaleza', 'cole_caracter'), ('cole_naturaleza', 'cole_depto_ubicacion'), ('cole_naturaleza', 'cole_area_ubicacion'), ('fami_estratovivienda', 'cole_naturaleza'), ('fami_estratovivienda', 'cole_depto_ubicacion'), ('fami_estratovivienda', 'cole_caracter'), 
 ('cole_area_ubicacion', 'cole_genero'),  ('cole_depto_ubicacion', 'cole_caracter'), ('cole_depto_ubicacion', 'fami_estratovivienda'), ('cole_depto_ubicacion', 'cole_genero'), ('cole_jornada', 'fami_estratovivienda'), ('cole_jornada', 'cole_depto_ubicacion'), ('cole_jornada', 'cole_caracter'), ('fami_estratovivienda', 'cole_mcpio_ubicacion'), ('fami_estratovivienda', 'cole_genero'), ('fami_estratovivienda', 'cole_area_ubicacion'), ('punt_global', 'cole_depto_ubicacion'),
  ('cole_depto_ubicacion', 'cole_naturaleza'), ('cole_genero', 'cole_mcpio_ubicacion'), ('cole_genero', 'cole_area_ubicacion'), ('cole_jornada', 'cole_naturaleza'), ('punt_global', 'fami_estratovivienda')]
blacklistTTTTF=[('cole_area_ubicacion', 'cole_mcpio_ubicacion'), ('cole_area_ubicacion', 'cole_jornada'), ('cole_area_ubicacion', 'fami_estratovivienda'), ('cole_area_ubicacion', 'cole_naturaleza'), ('cole_area_ubicacion', 'cole_caracter'), ('cole_caracter', 'cole_genero'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'fami_estratovivienda'),  ('cole_jornada', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_depto_ubicacion'), ('cole_mcpio_ubicacion', 'cole_jornada'), ('cole_mcpio_ubicacion', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_caracter'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_estratovivienda'), ('cole_naturaleza', 'punt_global'), ('cole_naturaleza', 'cole_genero'), ('cole_naturaleza', 'fami_estratovivienda'),
 ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_depto_ubicacion'),  ('cole_caracter', 'cole_area_ubicacion'),  ('cole_depto_ubicacion', 'cole_jornada'), ('cole_genero', 'cole_jornada'), ('cole_genero', 'fami_estratovivienda'), ('cole_genero', 'cole_depto_ubicacion'), ('cole_genero', 'cole_naturaleza'), ('cole_genero', 'cole_caracter'), ('cole_jornada', 'cole_mcpio_ubicacion'),  ('cole_jornada', 'cole_area_ubicacion'),  ('cole_naturaleza', 'cole_mcpio_ubicacion'), ('cole_naturaleza', 'cole_caracter'), ('cole_naturaleza', 'cole_depto_ubicacion'), ('cole_naturaleza', 'cole_area_ubicacion'), ('fami_estratovivienda', 'cole_naturaleza'), ('fami_estratovivienda', 'cole_depto_ubicacion'), ('fami_estratovivienda', 'cole_caracter'), 
 ('cole_area_ubicacion', 'cole_genero'),  ('cole_depto_ubicacion', 'cole_caracter'), ('cole_depto_ubicacion', 'fami_estratovivienda'), ('cole_depto_ubicacion', 'cole_genero'), ('cole_jornada', 'fami_estratovivienda'), ('cole_jornada', 'cole_depto_ubicacion'), ('cole_jornada', 'cole_caracter'), ('fami_estratovivienda', 'cole_mcpio_ubicacion'), ('fami_estratovivienda', 'cole_genero'), ('fami_estratovivienda', 'cole_area_ubicacion'), ('punt_global', 'cole_depto_ubicacion'),
  ('cole_depto_ubicacion', 'cole_naturaleza'), ('cole_genero', 'cole_mcpio_ubicacion'), ('cole_genero', 'cole_area_ubicacion'), ('cole_jornada', 'cole_naturaleza'), ('punt_global', 'fami_estratovivienda'),('fami_estratovivienda', 'cole_jornada'), ('punt_global', 'cole_genero')]


"""for black in blacklistTT:
  # check if the count of black is > 1 (repeating item)
  if blacklistTT.count(black) > 1:
  # if True, remove the first occurrence of black
    blacklistTT.remove(black)

print(blacklistTT)""" 

estimated_modelbla = esth.estimate( scoring_method=scoring_method, max_indegree=4, black_list= blacklistTTTTF,max_iter=int(1e4))
print(estimated_modelbla) 
print("nodos")
print(estimated_modelbla.nodes()) 
print("edges")
print(estimated_modelbla.edges()) 

print(scoring_method.score(estimated_modelbla)) 
#BICSCORE
scoring_methodd = BicScore(data=dff)
esthB = HillClimbSearch(data=dff) 

# Iteración1 BIC estimated_modelh = esthB.estimate( scoring_method=scoring_methodd, max_indegree=4, max_iter=int(1e4))
#blacklist =
#blacklistT = [('cole_caracter', 'cole_genero'), ('cole_depto_ubicacion', 'cole_genero'), ('cole_jornada', 'cole_naturaleza'), ('cole_jornada', 'cole_caracter'), ('cole_jornada', 'cole_area_ubicacion'), ('cole_jornada', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_jornada'), ('cole_mcpio_ubicacion', 'cole_caracter'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_estratovivienda'), ('cole_naturaleza', 'fami_estratovivienda'), ('cole_naturaleza', 'cole_genero')]
#blacklistT = [('cole_caracter', 'cole_genero'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'cole_jornada'), ('cole_caracter', 'cole_area_ubicacion'),('cole_depto_ubicacion', 'cole_caracter'), ('cole_depto_ubicacion', 'cole_jornada'), ('cole_depto_ubicacion', 'cole_naturaleza'), ('cole_depto_ubicacion', 'fami_estratovivienda'), ('cole_genero', 'cole_jornada'), ('cole_genero', 'cole_caracter'), ('cole_genero', 'cole_naturaleza'), ('cole_jornada', 'cole_mcpio_ubicacion'),('cole_depto_ubicacion', 'cole_genero'), ('cole_jornada', 'cole_naturaleza'), ('cole_jornada', 'cole_caracter'), ('cole_jornada', 'cole_area_ubicacion'), ('cole_jornada', 'cole_genero'), ('cole_mcpio_ubicacion', 'cole_jornada'), ('cole_mcpio_ubicacion', 'cole_caracter'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_estratovivienda'), ('cole_naturaleza', 'fami_estratovivienda'), ('cole_naturaleza', 'cole_genero')]

"""estimated_modelh = esthB.estimate( scoring_method=scoring_methodd, max_indegree=4, max_iter=int(1e4))
print(estimated_modelh) 
print("nodos bic")
print(estimated_modelh.nodes()) 
print("edges bic")
print(estimated_modelh.edges()) 

print(scoring_method.score(estimated_modelh))"""
"""
#por restricciones
est =PC(data=dff) 
estimated_model = est.estimate(variant="stable", max_cond_vars=4 )
print(estimated_model)
print("nodos restriccion")
print(estimated_model.nodes())
print(estimated_model.edges())"""

