# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:38:19 2019

@author: COJ
"""

###############################################################################
""" Directorio raiz, a modificar por cada usuario """
root= 'D:\\Dropbox\\Documentos\\IPC_ML\\'
###############################################################################

import os
os.chdir(root+'Git')
import pandas as pd
from dateutil.parser import parse
from aux_funcs import limpia


"""IMPORTA INDICE PUBLICADO INE DEL LIMON """


""" 1 empalma archivo mas antiguo """
df1= pd.read_excel(root+'Data\\ipc_producto_referencial_diciembre2013.xlsx',header=4,usecols=range(6,67))
df1= df1[df1['GLOSA']=='LIMON']
df1= df1.T
df1= df1.iloc[1:]
df1.columns=["ine"] 
df1.index= pd.to_datetime(df1.index)


""" 2 empalma archivo del medio """
df2= pd.read_excel(root+'Data\\Analisis_ranking_vol_explicada_IPC.xlsx',header=4,usecols=range(6,67))
df2= df2[df2['GLOSA']=='LIMON']
df2= df2.T
df2= df2.iloc[1:]
df2.columns=["ine"]
df2.index= pd.to_datetime(df2.index)


""" 3 empalma archivo más nuevo """
df3= pd.read_excel(root+'Data\\ipc-2019-xls.xlsx',header=3,usecols=range(0,10+1))
df3['Glosa']=df3['Glosa'].apply(lambda x: limpia(x))
df3= df3[df3['Glosa']=='LIMON']

# crea indice datetime
df3.index= [parse(str(df3['Año'][x])+' '+str(df3['Mes'][x]) +' 1') for x in df3.index]
df3 = df3.rename(columns={'Índice': 'ine'})
df3= df3[['ine']]


""" 4= 1+2+3 concat todo """
df= pd.concat([df1,df2,df3])

# transforma fecha a fin de mes
df.index= df.resample('M', convention='end').asfreq().index

df= df.pct_change()


