# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:21:12 2019

@author: COJ
"""

###############################################################################
""" Directorio raiz, a modificar por cada usuario """
root= 'D:\\Dropbox\\Documentos\\IPC_ML\\'
###############################################################################


import os
os.chdir(root+'Data\\Bases_precio_consumidor')
import pandas as pd
import glob

L= list(glob.glob("*.csv"))

#inicializo empty dataframe con los nombres de cols que necesito
df= pd.DataFrame(columns=pd.read_csv(L[0],sep='\t',encoding="ISO-8859-1",index_col=0).columns)

for l in L:
    dfl= pd.read_csv(l,sep='\t',encoding="ISO-8859-1",index_col=0)
    print('\n',dfl.Producto.value_counts(),'\n')
    df= pd.concat([df,dfl])
    

df.to_csv(root+'Data\\precios_odepa_consum_total.csv',sep='\t',encoding="ISO-8859-1",)
