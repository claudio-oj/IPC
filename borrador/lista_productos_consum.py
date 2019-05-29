# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:43:32 2019

@author: COJ
"""

import pandas as pd
import glob
import os
os.chdir('D:\\Dropbox\\Documentos\\IPC_ML\\Data\\Bases_precio_consumidor\\')

L= list(glob.glob("*.csv"))

for l in L:
    df= pd.read_csv(l,sep='\t',encoding = "ISO-8859-1")
    print(l,'\n',df.Producto.unique(),'\n')
    

