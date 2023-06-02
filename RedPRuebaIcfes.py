import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import LabelEncoder as le
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
df=pd.read_csv("C:\\Users\\ricky\\Downloads\\Proyecto3\\datosREd.csv", header=0)


#PuntajeGlobal
intervalos3 = [0, 100, 200, 300, 400, 500]
categorias3 = ['0-100', '101-200', '201-300', '301- 400', '401-500']
df['punt_global'] = pd.cut(df['punt_global'], bins=intervalos3, labels=categorias3)

print(df)
df.to_csv('datosRedIcfes.csv', index=False)

np.random.seed(42)
#  Dataframe con datos de prueba
df_20 = df.sample(frac=0.2)

# Dataframe con datos de entrenamiento
df_80= df.drop(df_20.index)

model = BayesianNetwork([
                          ( "cole_depto_ubicacion" , "cole_cod_mcpio_ubicacion" ),
                          ( "cole_cod_mcpio_ubicacion" , "cole_area_ubicacion" ),
                          ( "cole_area_ubicacion" , "punt_global" ),
                          ( "cole_caracter" , "cole_jornada" ),
                          (  "cole_naturaleza", "cole_jornada" ),
                          ( "cole_jornada", "punt_global"  ),
                           ( "fami_estratovivienda", "punt_global"  ),
                           ( "cole_genero", "punt_global"  ),
])

emv = MaximumLikelihoodEstimator( model = model , data = df )
# Estimar para nodos sin padres
# Estimar para nodo depto
cpdem_depto = emv.estimate_cpd( node ="cole_depto_ubicacion")
print( cpdem_depto )
# Estimar para nodo cole genero
cpdem_genero = emv.estimate_cpd( node ="cole_genero")
print( cpdem_genero )
# Estimar para nodo cole naturaleza
cpdem_natu = emv.estimate_cpd( node ="cole_naturaleza")
print( cpdem_natu )
# Estimar para nodo cole caracter
cpdem_caract = emv.estimate_cpd( node ="cole_caracter")
print( cpdem_caract )
# Estimar para estrato 
cpdem_estr = emv.estimate_cpd( node ="fami_estratovivienda", )
print( cpdem_estr)
model.fit(data=df , estimator = MaximumLikelihoodEstimator
)
for i in model.nodes():
    print(model.get_cpds(i) )
   

