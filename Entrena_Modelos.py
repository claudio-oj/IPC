# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:35:34 2019

@author: COJ
"""

###############################################################################
""" Directorio raiz, a modificar por cada usuario """
root= 'D:\\Dropbox\\Documentos\\IPC_ML\\'
###############################################################################

import os
os.chdir(root+'Git')
import pandas as pd
from aux_funcs import ipc_import,odepa_cons_impo, odepa_may_impo, run_model


#""" IMPORTA DATA """
#
#prods= ['Limon','Papa','Tomate']
#
## procesa Y's
#Y= ipc_import(root=root, producto=prods[0].upper())
#for p in prods[1:]:
#    Y= Y.join(ipc_import(root=root, producto=p.upper()))
#Y.dropna(axis=0,inplace=True) #elimina rows con nan's
#
## procesa X's
#X= pd.DataFrame(index= Y.index)
#for p in prods:
#    X= X.join(odepa_may_impo( producto=p),rsuffix='_may')
#    X= X.join(odepa_cons_impo(producto=p),rsuffix='_con')
#X.dropna(axis=1,inplace=True) #elimina cols con nan's
#
#X= X[X.index.isin(Y.index)] # se queda solo con el indice que esta en Y



#%%

# run model, elige modelo
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso,ElasticNet     

run_model(LinearRegression(),         X,Y, lags=[1,2,3], n_splits=20,degree=1, grafs=False)
run_model(SGDRegressor(penalty='l1') ,X,Y, lags=[1,2,3], n_splits=20,degree=1, grafs=False)
run_model(Ridge(alpha=0.1),           X,Y, lags=[1,2,3], n_splits=20,degree=1, grafs=False)
run_model(Lasso(alpha=0.3)           ,X,Y, lags=[1,2,3], n_splits=20,degree=1, grafs=False)
run_model(ElasticNet(alpha=0.2,l1_ratio=0.2) ,X,Y, lags=[1,2,3], n_splits=20,degree=1, grafs=False)

#%%
from sklearn.tree import DecisionTreeRegressor
run_model(DecisionTreeRegressor(criterion='mae') ,X,Y, lags=[1,2,3,4], n_splits=25,degree=1, grafs=False)

#%%
from sklearn.svm import SVR
run_model(SVR(kernel='rbf',degree=3,epsilon=0.075) ,X,Y, lags=[1,2,3], n_splits=50, grafs=False)
run_model(SVR(kernel='rbf',degree=2,epsilon=0.075) ,X,Y, lags=[1,2,3], n_splits=50, grafs=False)

#%%
from sklearn.neural_network import MLPRegressor
run_model(MLPRegressor(hidden_layer_sizes=(4,4,4),solver='lbfgs') ,X,Y, lags=[1,2,3], n_splits=50, grafs=False)

# guarda en carpeta modelo ganador

