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

df= pd.read_csv(root+'Data\\precios_odepa_consum_total.csv',sep="\t",encoding="latin-1",
                decimal=',', index_col=0)



# elimina acento y caracteres extraños
for x0 in ['Calidad','Procedencia','Producto','Region','Sector','Tipo punto monitoreo',
           'Unidad','Variedad']:
    df[x0]=df[x0].apply(lambda x: str(x))
    df[x0]=df[x0].apply(lambda x: limpia(x))



#%%


""" CREA DICCIONARIO"""

dic_grande= {}
keys1= list(df.Producto.unique()) # tomate, papa, etc...
keys1.sort()

keys2= ['Calidad','Procedencia','Region','Sector','Tipo punto monitoreo','Unidad','Variedad']
keys2.sort()

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
