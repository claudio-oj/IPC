# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:35:18 2019       python 3.6

@author: COJ https://www.odepa.gob.cl/precios/series-historicas-de-frutas-y-hortalizas
"""

import os
os.chdir('D:\\BA\\Clientes\\HSBC\\2_IPC\\modelos_por_producto\\')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from aux_funcs import P_equiv,isnan,crealag,run_model


""" 1.1    IMPORTA INDICE PUBLICADO INE DEL LIMON """
dfine= pd.read_excel('Analisis_ranking_vol_explicada_IPC.xlsx',header=4,usecols=range(6,67))
dfine= dfine[dfine['GLOSA']=='LIMÓN']
dfine= dfine.T
dfine= dfine.iloc[1:]
dfine.columns=["ine"]
dfine.index= pd.to_datetime(dfine.index)

# empalma archivo mas antiguo
dfine2= pd.read_excel('ipc_producto_referencial_diciembre2013.xlsx',header=4,usecols=range(6,67))
dfine2= dfine2[dfine2['GLOSA']=='LIMON']
dfine2= dfine2.T
dfine2= dfine2.iloc[1:]
dfine2.columns=["ine"]
dfine2.index= pd.to_datetime(dfine2.index)

dfine= pd.concat([dfine2,dfine])


# transforma fecha a fin de mes
dfine.index= dfine.resample('M', convention='end').asfreq().index

dfine= dfine.pct_change()

##Plot limon INE
#dfine.plot(figsize=(10,6),fontsize=22)
#plt.title("cambio % Limon INE", fontsize=22)
#plt.legend(prop={'size': 18})
#plt.xlabel('Time',fontsize=18)
#plt.show()




""" 1.2   IMPORTA/FORMATEA/LIMPIA/ NUEVA BASE DE DATOS ODEPA --> LIMON  """

df= pd.read_csv('precios_frutas_hortalizas_odepa.csv',sep="\t",encoding="latin-1",
                decimal=',')
df= df[df.Producto=='Limón']
df=df.iloc[:,1:]
      
#df.Producto= df.Producto.apply(lambda x: unidecode.unidecode(x)) 

# elimina puntos para que pueda evaluar correctamente: puntos como "thousands separator"
for x in ['Volumen','Preciomínimo','Preciomáximo','Preciopromedio']:
    df[x]= df[x].apply(lambda x: x.replace('.',''))

# transforma str --> numeric
for x in ['Volumen','Preciomínimo','Preciomáximo','Preciopromedio']:
    df[x]= pd.to_numeric(df[x], errors='coerce')

df.Desde= pd.to_datetime(df.Desde,dayfirst=True)
df.Hasta= pd.to_datetime(df.Hasta,dayfirst=True)


###############################################################################


""" 1.3    PROCESA BASE DE DATOS """

for x in ['Mercado','Variedad','Calidad','Origen','Unidad decomercialización']:
    print(df[x].value_counts())
    print()

# calcula # kilos en Unidad de comercialización
df['kilos']= df['Unidad decomercialización'].apply(lambda x: P_equiv(x))

df['pk']= df.Preciopromedio / df.kilos



""" 1.4    SLICE DF POR LA SELECCION DEL PPT """ 

# Filtro VARIEDAD
df= df[df['Variedad']=='Sin especificar']

# Filtro MERCADO
filter_list = [
        'Central Lo Valledor de Santiago',
        'Vega Central Mapocho de Santiago',
        'Vega poniente de Santiago'
               ]
df= df[df.Mercado.isin(filter_list)]

# Filtro ORIGEN
filter_list = [
        'Provincia de Santiago',
#        'Zona central',
#        "Región de O'Higgins",
#        'Centro-norte',
#        'Provincia de Limarí',
        'Región Metropolitana',
#        'Provincia del Elquí',
#        'Provincia de Quillota',
#        'Provincia de Melipilla',
#        'Provincia de Petorca'
               ]
df= df[df.Origen.isin(filter_list)]

# Filtro UNIDAD DECOMERCIALIZACIÓN
filter_list = [
               '$/kilo (en malla de 20 kilos)',
               '$/malla 18 kilos',
               '$/malla 20 kilos',
               '$/malla 16 kilos',
#               '$/caja 20 kilos',
#               '$/caja 24 kilos',
#               '$/caja 18 kilos',
               '$/malla 15 kilos',
               '$/kilo',
               '$/kilo (en caja de 15 kilos)'
               ]
df= df[df['Unidad decomercialización'].isin(filter_list)]

# Filtro CALIDAD
calidades = [
        '1a amarillo',
        '2a amarillo',
        '1a plateado',
        '3a amarillo',
        '2a plateado',
        '3a plateado'
               ]
df= df[df.Calidad.isin(calidades)]

#df.pk.plot()


""" 1.5   CREA DATAFRAME DE CALIDADES DEL LIMON """

# creo df con index fechas unicas, para guardar series de calidades del limon.
dfc= pd.DataFrame(index=df.Hasta.unique())

# weighted average
for c in calidades:
    dfc[c]= df[df.Calidad==c].groupby(df.Hasta).apply(lambda x: np.average(x.pk, weights=x.Volumen))

## PLOT
#dfc['2014':].plot(figsize=(10,6),fontsize=22)
#plt.title("Tipos de Limon en ODEPA: Precio/kilo", fontsize=22)
#plt.legend(prop={'size': 18})
#plt.xlabel('Time Steps',fontsize=18)
#plt.show()


""" 1.6      INTERPOLA DATOS FALTANTES DE CALIDADES DE LIMON"""

# cuenta nan's por columna
for x in dfc.columns: 
    print(x, dfc[x].isna().sum(), ' nans de ',len(dfc))
    
# elimina calidad 3a plateado por excesivos nan's....
dfc.drop(labels='3a plateado',axis=1, inplace=True)

# crea df con cambios porcentuales de cada CALIDAD del limon
dfcgrow= dfc.pct_change(limit=1)

# pisa con nan's los ceros de la formula pct --> evita calc erroneos
dfcgrow= dfcgrow.where(dfc.isna()==False,dfc)

# promedia los pct de cada calidad en una sola columna
dfcgrow= dfcgrow.mean(axis=1)

# pisa los nan de crecimiento con el pct fila anterior
dfcgrow= dfcgrow.interpolate(method='pad')



""" 1.7       PUEBLA LOS NAN CON EL CRECIMIENTO PROMEDIO DE DFCGROW """

dfcshift= dfc.shift(1)

for col in dfc.columns:
    for ind,row in dfc[col].iteritems():
        
        if isnan(row)==True:
            # indice ordinal
            ilug= dfc[col].index.get_loc(ind)                   
            dfc[col][ind]= dfc[col].iloc[ilug-1] * (1+dfcgrow[ind])


# alarga el indice a nivel "cambio diario"
dfc= dfc.reindex( pd.date_range(start=dfc.index[0], end=dfc.index[-1]))
dfc= dfc.interpolate(method='index')


           
            
""" 1.8        CREA DF PARA INPUT MACHINE LEARNING """

dfc= dfc.asfreq('M')
dfc= dfc.pct_change()

dfine= dfine.join(dfc, how='inner')
dfine.dropna(inplace=True)

#dfine.plot()

#limpia variables, para empezar el modulo de ML
df=dfine
del c, calidades,col,dfine,dfine2,dfc,dfcgrow,dfcshift,filter_list,ilug,ind,row,x


#%%
###############################################################################
###############################################################################
###############################################################################



""" SECCION 2:     CREA MODELOS MACHINE LEARNING
regresion lineal
modelos regularizados
SVR
Redes """


""" 2.1.         REGRESION LINEAL, SIN LAGS         buen punto de partida..."""


from sklearn.linear_model import LinearRegression   

#crea X,Y
Y= pd.DataFrame(df.iloc[:,0])
X= df.iloc[:,1:]

# corre modelo, genera metricas y graficos
run_model(LinearRegression(),X,Y, lags=[1,2,12],grafs=False)




#%%


""" 2.2         REGRESION LINEAL, CON 2 LAGS. """

from sklearn.linear_model import LinearRegression  

# crea X con lags
df2= pd.concat([ df, crealag(df.iloc[:,1:],1), crealag(df.iloc[:,1:],3) ] , axis=1)
df2.dropna(inplace=True)

#crea X,Y
Y= pd.DataFrame(df.iloc[3:,0])
X= df2.iloc[:,1:]


# crea modelo Regresion Lineal
lin_reg=LinearRegression()
lin_reg.fit(X,Y)
lin_reg.intercept_, lin_reg.coef_


# corre modelo, genera metricas y graficos
run_model(lin_reg,X,Y)


#%%


""" 2.3         REGRESION LINEAL de 2do Orden CON 3 LAGS,         es malo"""

from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures

    
# crea X con lags
df2= pd.concat([ df, crealag(df.iloc[:,1:],1), crealag(df.iloc[:,1:],3) ] , axis=1)
df2.dropna(inplace=True)

#crea X,Y
Y= pd.DataFrame(df.iloc[3:,0])
X= df2.iloc[:,1:]

# WARNING: DEGREE >=2 PRODUCE MUCHO OVERFITING
poly_features= PolynomialFeatures(degree=1,include_bias=False)
data_poly= poly_features.fit_transform(X)

# crea modelo Regresion Lineal
lin_reg=LinearRegression()
lin_reg.fit(data_poly,Y)
lin_reg.intercept_, lin_reg.coef_

# corre modelo, genera metricas y graficos
run_model(lin_reg,data_poly, Y)



#%%


""" 2.4.1      RIDGE  1er orden, 2 lags, es bueno"""

from sklearn.linear_model import Ridge


# crea X con lags
df2= pd.concat([ df, crealag(df.iloc[:,1:],1), crealag(df.iloc[:,1:],2) ] , axis=1)
df2.dropna(inplace=True)

#crea X,Y
Y= pd.DataFrame(df2.iloc[:,0])
X= df2.iloc[:,1:]

poly_features= PolynomialFeatures(degree=1,include_bias=False)
data_poly= poly_features.fit_transform(X)

ridge_reg= Ridge(alpha=1)
ridge_reg.fit(data_poly,Y)
#cte,thetas= ridge_reg.intercept_,ridge_reg.coef_

# corre modelo, genera metricas y graficos
run_model(ridge_reg, data_poly, Y)

#%%


""" 2.4.2     RIDGE  2do orden, 2 lags, es bueno"""

from sklearn.linear_model import Ridge


# crea X con lags
df2= pd.concat([ df, crealag(df.iloc[:,1:],1), crealag(df.iloc[:,1:],2) ] , axis=1)
df2.dropna(inplace=True)

#crea X,Y
Y= pd.DataFrame(df2.iloc[:,0])
X= df2.iloc[:,1:]

poly_features= PolynomialFeatures(degree=2,include_bias=False)
data_poly= poly_features.fit_transform(X)

ridge_reg= Ridge(alpha=1)
ridge_reg.fit(data_poly,Y)
#cte,thetas= ridge_reg.intercept_,ridge_reg.coef_

# corre modelo, genera metricas y graficos
run_model(ridge_reg, data_poly, Y)

#%%


""" 2.5      LASSO   ,   ESCALA PRIMERO..."""

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# crea X con lags
df2= pd.concat([ df, crealag(df.iloc[:,1:],1), crealag(df.iloc[:,1:],2) ] , axis=1)
df2.dropna(inplace=True)

#crea X,Y
Y= pd.DataFrame(df2.iloc[:,0])
X= df2.iloc[:,1:]

poly_features= PolynomialFeatures(degree=2,include_bias=False)
data_poly= poly_features.fit_transform(X)

scaler=StandardScaler()
data_poly=scaler.fit_transform(data_poly)

lasso_reg= Lasso(alpha=0.1,random_state=1) # alpha=0 <--> Reg Lin
#lasso_reg.fit(data_poly,Y)
#cte,thetas=lasso_reg.intercept_,lasso_reg.coef_

# corre modelo, genera metricas y graficos
run_model(lasso_reg, data_poly, Y)

#%%


""" 2.6      ElasticNet , ESCALA PRIMERO...   """

from sklearn.linear_model import ElasticNet  #MAE-->L1    , #RMSE -->L2

# crea X con lags
df2= pd.concat([ df, crealag(df.iloc[:,1:],1), crealag(df.iloc[:,1:],2) ] , axis=1)
df2.dropna(inplace=True)

#crea X,Y
Y= pd.DataFrame(df2.iloc[:,0])
X= df2.iloc[:,1:]

poly_features= PolynomialFeatures(degree=2,include_bias=False)
data_poly= poly_features.fit_transform(X)

scaler=StandardScaler()
data_poly=scaler.fit_transform(data_poly)

elastic_net= ElasticNet(alpha=0.2,l1_ratio=0.2,max_iter=1000000,fit_intercept=False,tol=0.0000001)
#elastic_net.fit(data_poly,Y)
#cte,thetas=elastic_net.intercept_,elastic_net.coef_

# corre modelo, genera metricas y graficos
run_model(elastic_net, data_poly, Y)

#%%

""" 2.7      SVR    """

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# crea X con lags
df2= pd.concat([ df, crealag(df.iloc[:,1:],1), crealag(df.iloc[:,1:],2) ] , axis=1)
df2.dropna(inplace=True)

#crea X,Y
Y= pd.DataFrame(df2.iloc[:,0])
X= df2.iloc[:,1:]

poly_features= PolynomialFeatures(degree=2,include_bias=False)
data_poly= poly_features.fit_transform(X)

scaler=StandardScaler()
data_poly=scaler.fit_transform(data_poly)

svr_model= SVR(kernel='rbf',degree=2,epsilon=0.075)

# corre modelo, genera metricas y graficos
run_model(svr_model, data_poly, Y)

#%%


""" 2.8 REDES NEURONALES .... MLP REGRESSOR!!... """

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
#import pickle
# crea X con lags
df2= pd.concat([ df, crealag(df.iloc[:,1:],1), crealag(df.iloc[:,1:],2) ] , axis=1)
df2.dropna(inplace=True)

#crea X,Y
Y= pd.DataFrame(df2.iloc[:,0])
X= df2.iloc[:,1:]

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.1)

mlp_clf = Pipeline([
("poly_feat", PolynomialFeatures(degree=2)),
 ("scaler", StandardScaler()),
("mlp_clf", MLPRegressor(hidden_layer_sizes=(10,10,10),batch_size=50))
])

mlp_clf.fit(X_train, Y_train)
y_pred_tr=  mlp_clf.predict(X_train)
y_pred_tst= mlp_clf.predict(X_test)

train_error=mean_absolute_error(Y_train,y_pred_tr)
test_error=mean_absolute_error(Y_test,y_pred_tst)

print('train error ', round(100*train_error,2))
print('test error ',  round(100*test_error,2))
print('95th percentile of Test Error ',round( 100*np.percentile(test_error,3)))

#Modelo Final    
Y2= pd.DataFrame(index=Y.index)
Y2['Model_pred']= mlp_clf.predict(X)
Y2['Actual']=Y
Y2.plot(figsize=(11,6),fontsize=22)
plt.title("Model vs Actual", fontsize=22)
plt.legend(prop={'size': 18})
plt.show()

#''' te guardé la mejor red a pickle '''
#modelos_pkl_filename = 'mejor_red_limon.pkl'
## Open the file to save as pkl file
#models_trained_pkl = open(modelos_pkl_filename, 'wb')
#pickle.dump(mlp_clf, models_trained_pkl) #OBJETO DE PYTHON A GUARDAR
#models_trained_pkl.close()

#%%

#' asi se importa un pickle (un modelo ya entrenado) '
#import pickle 
#
#modelos_pkl_filename = 'mejor_red_limon.pkl'
#mlp_clf_archivo= open(modelos_pkl_filename, 'rb')
#mlp_clf= pickle.load(mlp_clf_archivo)
#
#'''luego lo ocupas igual q siempre...'''
#
#Y2= pd.DataFrame(index=Y.index)
#Y2['Model_pred']= mlp_clf.predict(X)
#Y2['Actual']=Y
#Y2.plot(figsize=(11,6),fontsize=22)
#plt.title("Model vs Actual", fontsize=22)
#plt.legend(prop={'size': 18})
#plt.show()
