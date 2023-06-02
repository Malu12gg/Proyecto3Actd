#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:46:54 2023

@author: paulaescobar
"""

#TMPDIR=/data/tmp/ sudo pip install --cache-dir=/data/tmp/ torch
#pip install torch --no-cache-dir
#TMPDIR=/data/tmp/ sudo pip3 install --cache-dir=/data/tmp/ pgmpy
#pip3 install sqlalchemy
#pip3 install psycopg2-binary


import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output

import plotly.express as px
import pandas as pd

from pgmpy.sampling import BayesianModelSampling

from pgmpy.readwrite import XMLBIFReader

from sqlalchemy import create_engine
from sqlalchemy import text
import sqlalchemy as db
from sqlalchemy import func
import psycopg2

import plotly.graph_objects as go

# Read model from XML BIF file 
reader = XMLBIFReader("model.xml")
model = reader.get_model()

infer = VariableElimination(model)

#Create SQL

engine = create_engine('postgresql://postgres:Proyecto32023@proyecto3db.cf0fdevvpqbv.us-east-1.rds.amazonaws.com/postgres', echo=False)

#municipio = pd.read_csv("/Users/paulaescobar/Documents/ACTD/Proyecto3/Municipio.csv", sep = ",")
#municipio.columns
#municipio.columns = (["cole_depto_ubicacion", "cole_mcpio_ubicacion", "Media", "Minimo", "Maximo"])

#municipio.to_sql('Municipio', con=engine, if_exists='replace', index=False)

#area = pd.read_csv("/Users/paulaescobar/Documents/ACTD/Proyecto3/Area.csv", sep = ",")
#area.columns
#area.columns = (["cole_mcpio_ubicacion", "cole_area_ubicacion", "Media", "Minimo", "Maximo"])

#area.to_sql('Area', con=engine, if_exists='replace', index=False)

#naturaleza = pd.read_csv("/Users/paulaescobar/Documents/ACTD/Proyecto3/Naturaleza.csv", sep = ",")
#naturaleza.columns
#naturaleza.columns = (["cole_mcpio_ubicacion", "cole_naturaleza", "Media", "Minimo", "Maximo"])

#naturaleza.to_sql('Naturaleza', con=engine, if_exists='replace', index=False)

#Create Metadata object

meta_data = db.MetaData()
meta_data.reflect(bind=engine)
        
#Get table from Metadata object
        
Municipio = meta_data.tables['Municipio']
Area = meta_data.tables['Area']
Naturaleza = meta_data.tables['Naturaleza']

#Municipios

query = db.select([Municipio.c.cole_mcpio_ubicacion]).order_by(Municipio.c.cole_mcpio_ubicacion)
result = engine.execute(query).fetchall()     
df = pd.DataFrame(result)
df.columns = (['Municipio'])
munic = df['Municipio']

#TABLERO DASH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    
    html.H1("Puntaje Prueba Saber 11: Análisis municipal en Antioquia",
            style={'text-align': 'center', 'color': 'white', 'backgroundColor': '#0b0347'}),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        
        dcc.Tab(label='Instrucciones', value='tab-1', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.Br(),
            
            html.Img(src=app.get_asset_url('imagen.jpeg'), style={'widht': '20%'}),
            
            html.H4("En esta herramienta encontrará información para el apoyo en la comparación del estado de la educación en los municipios de Antioquia y Bogotá, y la toma de decisiones asociada, a partir del puntaje global obtenido por estudiantes locales en la Prueba Saber 11. La información parte de la página de Datos Abiertos del gobierno colombiano.",
                    style={'text-align': 'justify', 'backgroundColor': '#6a94c4', 'font-family': 'Courier New, monospace'}),
            
            html.Br(),
            
            html.H4("La herramienta se divide en 4 secciones posibles para consulta:",
                    style={'text-align': 'justify', 'backgroundColor': '#cee5ed', 'font-family': 'Courier New, monospace'}),
            
            html.H5("1) Puntaje por Municipio: Consulte gráficas que comparan el puntaje global promedio por municipio en comparación con Bogotá.", style={'font-family': 'Courier New, monospace'}),
            
            html.H5("2) Puntaje por Área de Ubicación: Consulte gráficas que relacionan el puntaje global promedio con el área de ubicación del colegio.", style={'font-family': 'Courier New, monospace'}),
            
            html.H5("3) Puntaje por Naturaleza del Colegio: Consulte gráficas que relacionan el puntaje global promedio con la naturaleza del colegio.", style={'font-family': 'Courier New, monospace'}),
            
            html.H5("4) Predicción de Puntaje: Consulte la probabilidad de obtener cierto rango para el puntaje global promedio, a partir de valores de municipio, área, carácter, jornada y naturaleza del colegio.", style={'font-family': 'Courier New, monospace'}),
            
            ]),
        
        dcc.Tab(label='Puntaje por Municipio', value='tab-2', style=tab_style, selected_style=tab_selected_style, children=[
            
            dcc.Graph(id='graph1'),
            
            dcc.Graph(id='graph2'),
            
            ]),
        
        dcc.Tab(label='Puntaje por Área de Ubicación', value='tab-3', style=tab_style, selected_style=tab_selected_style, children=[
        
            dcc.Graph(id='graph3'),
            
            dcc.Graph(id='graph4'),
        
            ]),
        
        dcc.Tab(label='Puntaje por Naturaleza del Colegio', value='tab-4', style=tab_style, selected_style=tab_selected_style, children=[
            
            dcc.Graph(id='graph5'),
            
            dcc.Graph(id='graph6'),
            
            ]),
        
        dcc.Tab(label='Predicción: Puntaje', value='tab-5', style=tab_style, selected_style=tab_selected_style, children=[
            
            html.Br(),
            
            html.H5("Seleccione los valores de interés para predecir la probabilidad de obtener cierto rango de puntaje global:", style={'font-family': 'Courier New, monospace'}),
            
            html.Br(),
            
            html.Div(children=[
            
                html.Div(children=[
                    html.Div(children=[
                            html.Label("Municipio:", htmlFor = "dropdownMunic"),
                            dcc.Dropdown(
                                id="dropdownMunic",
                                options=[m for m in munic],
                                value='ABEJORRAL',
                                clearable=False,
                                ),
                            ],style=dict(width='33%', font = 'Courier New, monospace')),
                    
                    html.Div(children=[
                            html.Label("Área de ubicación del colegio:", htmlFor = "dropdownArea"),
                            dcc.Dropdown(
                                id="dropdownArea",
                                options=["RURAL","URBANO"],
                                value="RURAL",
                                clearable=False,
                                ),
                        ],style=dict(width='33%', font = 'Courier New, monospace')),
                    
                     html.Div(children=[
                            html.Label("Carácter del colegio:", htmlFor = "dropdownCar"),
                            dcc.Dropdown(
                                id="dropdownCar",
                                options=['ACADÉMICO','NO APLICA', 'TÉCNICO', 'TÉCNICO/ACADÉMICO'],
                                value='ACADÉMICO',
                                clearable=False,
                                ),
                        ],style=dict(width='33%', font = 'Courier New, monospace')),
                
                html.Br(),
                    
                    html.Div(children=[
                            html.Label("Jornada del colegio:", htmlFor = "dropdownJor"),
                            dcc.Dropdown(
                                id="dropdownJor",
                                options=['COMPLETA', 'MAÑANA', 'NOCHE', 'SABATINA', 'TARDE', 'UNICA'],
                                value="COMPLETA",
                                clearable=False,
                                ),
                        ],style=dict(width='33%', font = 'Courier New, monospace')),
                    
                     html.Div(children=[
                            html.Label("Naturaleza del colegio:", htmlFor = "dropdownNat"),
                            dcc.Dropdown(
                                id="dropdownNat",
                                options=['OFICIAL','NO OFICIAL'],
                                value='OFICIAL',
                                clearable=False,
                                ),
                        ],style=dict(width='33%', font = 'Courier New, monospace')),
                     ],style=dict(display='flex')),
                
                html.Br(),
                
                dcc.Graph(id='graph7'),
            
                html.H6("*Si la combinación de variables del paciente no cuenta con evidencias previas, la gráfica aparece en blanco*", style={'font-family': 'Courier New, monospace'}),
            
                ]),
            
            ]),

    ], style=tabs_styles),

])

@app.callback(Output('graph1', 'figure'),
              Output('graph2', 'figure'),
              Output('graph3', 'figure'),
              Output('graph4', 'figure'),
              Output('graph5', 'figure'),
              Output('graph6', 'figure'),
              Output('graph7', 'figure'),
              Input('tabs', 'value'),
              Input('dropdownMunic', 'value'),
              Input('dropdownArea', 'value'),
              Input('dropdownCar', 'value'),
              Input('dropdownJor', 'value'),
              Input('dropdownNat', 'value'))

def update_output_div(tab, selected_municipio, selected_area, selected_caracter, selected_jornada, selected_naturaleza):
    if tab == 'tab-2':
      
        #Graph1  
      
        query = db.select([Municipio.c.cole_mcpio_ubicacion, Municipio.c.Media]).order_by(Municipio.c.Media.desc()).limit(11)
        result = engine.execute(query).fetchall()
        df = pd.DataFrame(result)
        df.columns = (['Municipio', 'Media'])
        df = df.loc[::-1]
        df = df.round(2)
        col = ['#c0c8cf']*11
        col[7] = '#09093b'
        
        fig = go.Figure(data=[go.Bar(x = df['Media'], y = df['Municipio'], orientation = 'h', text = df['Media'], marker_color = col,
                         )
                ])
        fig.update_layout(height = 500, title_text = 'Top 10 Municipios VS Bogotá según Puntaje Global Promedio', font = dict(family = 'Courier New, monospace', size = 18, color = 'black'), plot_bgcolor = 'rgba(0,0,0,0)')
        
        #Graph2
        
        query = db.select([Municipio.c.cole_mcpio_ubicacion, Municipio.c.Media, Municipio.c.Minimo, Municipio.c.Maximo]).order_by(Municipio.c.Media.desc()).limit(11)
        result = engine.execute(query).fetchall()
        df = pd.DataFrame(result)
        df = df.loc[::-1]
        df = df.round(2)
        df.columns = (['Municipio', 'Media', 'Mínimo', 'Máximo'])
        
        fig2 = go.Figure(data=[go.Scatter(x = df['Media'], y = df['Municipio'], mode = 'markers+text', text = df['Media'], showlegend = False, marker = dict(size = 20), orientation = 'h', marker_color = col,
                                 )
                        ])
        
        fig2.add_trace(go.Scatter(y=df["Municipio"], 
                                 x=df["Mínimo"], 
                                 mode="markers+text",
                                 text = df['Mínimo'],
                                 orientation = 'h',
                                 showlegend=False, 
                                 marker=dict(size=20),
                                 marker_color = col))
        
        fig2.add_trace(go.Scatter(y=df["Municipio"], 
                                 x=df["Máximo"], 
                                 mode="markers+text",
                                 text = df['Máximo'],
                                 orientation = 'h',
                                 showlegend=False,
                                 marker=dict(size=20),
                                 marker_color = col))
        
        for i, row in df.iterrows():
        
            if row["Mínimo"]!=row["Máximo"] and row["Municipio"]!="BOGOTÁ D.C.": 
                
                fig2.add_shape( 
                    dict(type="line", 
                         y0=row["Municipio"], 
                         y1=row["Municipio"], 
                         x0=row["Mínimo"], 
                         x1=row["Máximo"], 
                         line=dict(color='#c0c8cf',width=2)
                        ))
                    
            elif row["Mínimo"]!=row["Máximo"] and row["Municipio"]=="BOGOTÁ D.C.": 
                
                fig2.add_shape( 
                    dict(type="line", 
                         y0=row["Municipio"], 
                         y1=row["Municipio"], 
                         x0=row["Mínimo"], 
                         x1=row["Máximo"], 
                         line=dict(color='#09093b',width=2),
                        ))
        
        fig2.update_layout(height = 800, title="Top 10 Municipios VS Bogotá Puntaje Global Promedio, Mínimo y Máximo", font = dict(family = 'Courier New, monospace', size = 18, color = 'black'), plot_bgcolor = 'rgba(0,0,0,0)', xaxis_range=[0, 500])
        fig2.update_traces(textposition = 'top center')

        return fig, fig2, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    elif tab == 'tab-3':
        
        #Graph3 graph4

        query = db.select([Area.c.cole_mcpio_ubicacion, Area.c.cole_area_ubicacion, Area.c.Media, Area.c.Minimo, Area.c.Maximo]).order_by(Area.c.Media.desc())
        result = engine.execute(query).fetchall()
        df = pd.DataFrame(result)
        df = df.round(2)
        df.columns = (['Municipio', 'Area', 'Media', 'Mínimo', 'Máximo'])
        
        rural = df.loc[df["Area"]=='RURAL']
        urbano = df.loc[df["Area"]=='URBANO']
        
        rural = rural.head(11)
        urbano = urbano.head(11)
        
        col = ['#89e8a1']*11
        col[8] = '#09093b'
        
        fig3 = go.Figure(data=[go.Bar(y = rural['Media'], x = rural['Municipio'], text = rural['Media'], marker_color = col,
                                 )
                        ])
        fig3.update_layout(height = 500, title_text = 'Top 10 Municipios VS Bogotá según Puntaje Global Promedio en Colegios Rurales', font = dict(family = 'Courier New, monospace', size = 18, color = 'black'), plot_bgcolor = 'rgba(0,0,0,0)')
        
        col = ['#fca2a6']*11
        col[5] = '#09093b'
        
        fig4 = go.Figure(data=[go.Bar(y = urbano['Media'], x = urbano['Municipio'], text = urbano['Media'], marker_color = col,
                                 )
                        ])
        fig4.update_layout(height = 500, title_text = 'Top 10 Municipios VS Bogotá según Puntaje Global Promedio en Colegios Urbanos', font = dict(family = 'Courier New, monospace', size = 18, color = 'black'), plot_bgcolor = 'rgba(0,0,0,0)')

        return dash.no_update, dash.no_update, fig3, fig4, dash.no_update, dash.no_update, dash.no_update
    
    elif tab == 'tab-4':
        
        query = db.select([Naturaleza.c.cole_mcpio_ubicacion, Naturaleza.c.cole_naturaleza, Naturaleza.c.Media, Naturaleza.c.Minimo, Naturaleza.c.Maximo]).order_by(Naturaleza.c.Media.desc())
        result = engine.execute(query).fetchall()
        df = pd.DataFrame(result)
        df = df.round(1)
        df.columns = (['Municipio', 'Naturaleza', 'Media', 'Mínimo', 'Máximo'])
        
        oficial = df.loc[df["Naturaleza"]=='OFICIAL']
        nooficial = df.loc[df["Naturaleza"]=='NO OFICIAL']
        
        oficial = oficial.head(16)
        nooficial = nooficial.head(16)
        
        
        
        col = ['#c685de']*16
        col[11] = '#09093b'
        
        fig5 = go.Figure(data=[go.Bar(y = oficial['Media'], x = oficial['Municipio'], text = oficial['Media'], marker_color = col,
                                 )
                        ])
        fig5.update_layout(height = 500, title_text = 'Top 15 Municipios VS Bogotá según Puntaje Global Promedio en Colegios Oficiales', font = dict(family = 'Courier New, monospace', size = 16, color = 'black'), plot_bgcolor = 'rgba(0,0,0,0)')
        
        col = ['#f29d6f']*16
        col[14] = '#09093b'
        
        fig6 = go.Figure(data=[go.Bar(y = nooficial['Media'], x = nooficial['Municipio'], text = nooficial['Media'], marker_color = col,
                                 )
                        ])
        fig6.update_layout(height = 500, title_text = 'Top 15 Municipios VS Bogotá según Puntaje Global Promedio en Colegios No Oficiales', font = dict(family = 'Courier New, monospace', size = 16, color = 'black'), plot_bgcolor = 'rgba(0,0,0,0)')
        
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, fig5, fig6, dash.no_update
  
    elif tab == 'tab-5':
        
        #Graph5
        
        car = 'ACAD_MICO'
        muni = ''
        nat = ''
        jor = ''
        
        #Caracter
        if selected_caracter == 'ACADÉMICO':
            car = 'ACAD_MICO'
        elif selected_caracter == 'NO APLICA':
            car = 'NO_APLICA'
        elif selected_caracter == 'TÉCNICO':
            car = 'T_CNICO'
        elif selected_caracter == 'TÉCNICO/ACADÉMICO':
            car = 'T_CNICO_ACAD_MICO'
            
        #Municipio
        if selected_municipio == 'BOGOTÁ D.C.':
            muni = 'BOGOT__D.C.'
        else:
            muni = selected_municipio
            
        #Municipio
        if selected_naturaleza == 'NO OFICIAL':
            nat = 'NO_OFICIAL'
        else:
            nat = selected_naturaleza
            
        #Jornada
        if selected_jornada == 'MAÑANA':
            jor = 'MA_ANA'
        else:
            jor = selected_jornada
        
        posterior_num = infer.query(["punt_global"], evidence={"cole_mcpio_ubicacion": muni, "cole_area_ubicacion": selected_area, 'cole_caracter': car, 'cole_jornada': jor, 'cole_naturaleza': nat})
        dff = pd.DataFrame(posterior_num.values)
        dff.columns = (["Probabilidad"])
        dff = dff.round(3)
        dff['punt_global'] = ['101-200', '201-300', '301-400', '401-500']
        
        fig7 = go.Figure(data=[go.Bar(y = dff['Probabilidad'], x = dff['punt_global'], text = dff['Probabilidad'], marker_color = 'lightblue',
                                 )
                        ])
        fig7.update_layout(height = 500, title_text = 'Distribución de Probabilidad para Rango Obtenido en Puntaje Global', font = dict(family = 'Courier New, monospace', size = 18, color = 'black'), plot_bgcolor = 'rgba(0,0,0,0)')
        
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, fig7
    
    elif tab == 'tab-1':
      return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)

