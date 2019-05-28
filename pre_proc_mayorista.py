# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:39:47 2019

@author: COJ

IMPORTA/FORMATEA/LIMPIA/  BASE DE DATOS ODEPA MAYORISTA
Y CREA UN ARCHIVO CSV LIMPIO

"""

###############################################################################
""" Directorio raiz, a modificar por cada usuario """
root= 'D:\\Dropbox\\Documentos\\IPC_ML\\'
###############################################################################

import os
os.chdir(root+'Git')
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from aux_funcs import P_equiv,limpia


""" 1.2   IMPORTA/FORMATEA/LIMPIA/ NUEVA BASE DE DATOS ODEPA  """

df= pd.read_csv(root+'Data\\precios_frutas_hortalizas_odepa.csv',sep="\t",encoding="latin-1",
                decimal=',',index_col=0)
df= df[df.Producto.isin(['Limón','Palta'])]

# pasa a mayuscula los valores de la col Producto
df['Producto']=df['Producto'].map(lambda x: str(x).upper())

# elimina acento y caracteres extraños
df['Producto']=df['Producto'].apply(lambda x: limpia(x))

# elimina puntos para que pueda evaluar correctamente: puntos como "thousands separator"
for x in ['Volumen','Preciomínimo','Preciomáximo','Preciopromedio']:
    df[x]= df[x].apply(lambda x: x.replace('.',''))

# transforma str --> numeric
for x in ['Volumen','Preciomínimo','Preciomáximo','Preciopromedio']:
    df[x]= pd.to_numeric(df[x], errors='coerce')

df.Desde= pd.to_datetime(df.Desde,dayfirst=True)
df.Hasta= pd.to_datetime(df.Hasta,dayfirst=True)

# calcula # kilos en Unidad de comercialización
df['ud_com']= df['Unidad decomercialización'].apply(lambda x: P_equiv(x))
df['p']= df.Preciopromedio / df.ud_com

df.to_csv(root+'Data\\bd_mayorista_procesada.csv')

#%%

df2= pd.read_csv(root+'Data\\bd_mayorista_procesada.csv',index_col=0,parse_dates=['Desde','Hasta'])
