# -*- coding: utf-8 -*-
"""
Created on Sun May 12 08:14:06 2019

@author: COJ

Crea diccionario hjson de productos odepa con las caracteristicas de cada producto.

https://stackoverflow.com/questions/7100125/storing-python-dictionaries
"""

import os
os.chdir('D:\\Dropbox\\BA\\Clientes\\HSBC\\2 IPC\\modelos_por_producto\\')
import pandas as pd
import hjson
import time


""" 1  IMPORTA/FORMATEA/LIMPIA/ BASE DE DATOS ODEPA   """

df= pd.read_csv('precios_frutas_hortalizas_odepa.csv',sep="\t",encoding="latin-1",
                decimal=',')
df=df.iloc[:,1:]  

# elimina puntos para que pueda evaluar correctamente: puntos como "thousands separator"
for x in ['Volumen','Preciomínimo','Preciomáximo','Preciopromedio']:
    df[x]= df[x].apply(lambda x: x.replace('.',''))

# transforma str --> numeric
for x in ['Volumen','Preciomínimo','Preciomáximo','Preciopromedio']:
    df[x]= pd.to_numeric(df[x], errors='coerce')

df.Desde= pd.to_datetime(df.Desde,dayfirst=True)
df.Hasta= pd.to_datetime(df.Hasta,dayfirst=True)

for x in ['Mercado','Variedad','Calidad','Origen','Unidad decomercialización']:
    print(df[x].value_counts())
    print()

#%%


""" CREA DICCIONARIO"""

dic_grande= {}
keys1= list(df.Producto.unique()) # tomate, papa, etc...
keys2= ['Mercado','Variedad','Calidad','Origen','Unidad decomercialización']

# puebla diccionario grande en base a todos los "chicos por producto"
for i in keys1:    
    values2= [list(df[df.Producto==i][x].unique()) for x in keys2]
    dic_chico = dict(zip(keys2, values2))  
    dic_grande[i]= dic_chico

#%%


""" GUARDA DICCIONARIO """
x= time.localtime()
x= 'diccioh_'+str(x[0])+'_'+str(x[1])+'_'+str(x[2])+'.json'

with open(x, 'w') as fp:
    hjson.dump(dic_grande, fp, sort_keys=True, indent=4)

    
    
""" LLAMA DICCIONARIO """

with open('diccioh_2019_5_16.json') as fp:
    data= hjson.loads(fp.read())
