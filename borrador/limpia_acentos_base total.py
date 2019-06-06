# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:05:06 2019

@author: COJ
"""

import pandas as pd
root='D:\\Dropbox\\Documentos\\IPC_ML\\'
import os
os.chdir(root+'Git')
from aux_funcs import limpia



df= pd.read_csv(root+'Data\\precios_frutas_hortalizas_odepa.csv',sep="\t",
                    encoding="latin-1",decimal=',', index_col=0)

for x0 in ['Producto','Variedad','Calidad','Origen','Unidad decomercialización']:
    df[x0]=df[x0].apply(lambda x: limpia(x))
    
df.columns= pd.Series(df.columns).apply(lambda x: limpia(x))

df.to_csv(root+'Data\\precios_frutas_hortalizas_odepa.csv',sep="\t", encoding="latin-1")