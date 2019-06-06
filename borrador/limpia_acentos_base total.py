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
os.chdir(root+'Data\\Bases_precio_consumidor')


for file in os.listdir(root+'Data\\Bases_precio_consumidor'):
    df= pd.read_csv(file,sep="\t", encoding="latin-1",decimal=',', index_col=0)
    
    for x0 in ['Producto', 'Region', 'Sector', 'Unidad']:
        df[x0]= df[x0].apply(lambda x: str(x))
        df[x0]= df[x0].apply(lambda x: limpia(x))
        
    df.columns= pd.Series(df.columns).apply(lambda x: limpia(x))
    
    df.to_csv(file,sep="\t", encoding="latin-1")
    

