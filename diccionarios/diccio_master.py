# -*- coding: utf-8 -*-
"""
Created on Sun May 12 08:14:06 2019

@author: COJ

Crea diccionario hjson de productos odepa con las caracteristicas de cada producto.

https://stackoverflow.com/questions/7100125/storing-python-dictionaries
"""

###############################################################################
""" Directorio raiz, a modificar por cada usuario """
root= 'D:\\Dropbox\\Documentos\\IPC_ML\\'
###############################################################################


import os
os.chdir(root+'Git\\')
import pandas as pd
import hjson
import time
from aux_funcs import limpia


""" 1  IMPORTA/FORMATEA/LIMPIA/ BASE DE DATOS ODEPA   """

df= pd.read_csv(root+'Data\\precios_frutas_hortalizas_odepa.csv',sep="\t",encoding="latin-1",
                decimal=',')
df=df.iloc[:,1:]  

# elimina puntos para que pueda evaluar correctamente: puntos como "thousands separator"
for x in ['Volumen','Preciominimo','Preciomaximo','Preciopromedio']:
    df[x]= df[x].apply(lambda x: x.replace('.',''))

# transforma str --> numeric
for x in ['Volumen','Preciominimo','Preciomaximo','Preciopromedio']:
    df[x]= pd.to_numeric(df[x], errors='coerce')

df.Desde= pd.to_datetime(df.Desde,dayfirst=True)
df.Hasta= pd.to_datetime(df.Hasta,dayfirst=True)

# elimina acento y caracteres extraños
for x0 in ['Producto','Variedad','Calidad','Origen','Unidad decomercializacion']:
    df[x0]=df[x0].apply(lambda x: limpia(x))

# elimina acento y caracteres extraños de los nombres de columnas
df.columns= pd.Series(df.columns).apply(lambda x: limpia(x))

#%%


""" CREA DICCIONARIO"""

dic_grande= {}
keys1= list(df.Producto.unique()) # tomate, papa, etc...
keys1.sort()

keys2= ['Mercado','Variedad','Calidad','Origen','Unidad decomercializacion']
keys1.sort()

# puebla diccionario grande en base a todos los "chicos por producto"
for i in keys1:    
    values2= [list(df[df.Producto==i][x].unique()) for x in keys2]
    [x.sort() for x in values2]
    dic_chico= dict(zip(keys2, values2))  
    dic_grande[i]= dic_chico
   

#%%


""" GUARDA DICCIONARIO """
x= time.localtime()
x= 'diccioh_'+str(x[0])+'_'+str(x[1])+'_'+str(x[2])+'.json'

with open(x, 'w') as fp:
    hjson.dump(dic_grande, fp, sort_keys=True, indent=4)

    
    
#""" LLAMA DICCIONARIO """
#
#with open('diccioh_2019_5_16.json') as fp:
#    data= hjson.loads(fp.read())
