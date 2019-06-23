# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:25:51 2019

@author: COJ
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline 
from sklearn.multioutput import MultiOutputRegressor


def crealag(df, n):
    new_columns = ["{}_Lag{:02d}".format(variable, n) for variable in df.columns]
    new_df = df.shift(n)
    new_df.columns = new_columns
    return new_df


def da(Y_test,Y_pred):
    """ directional accuracy metric
    Y_test, Y_pred: pd.Series"""
    
    prod= Y_test * Y_pred   
    prod= pd.Series(prod.flatten())
    
    return prod.gt(0).sum() / len(prod)


def run_model(model, X, Y,lags=False,n_splits=50,degree=1,grafs=True):
    """ Corre modelos de Machine Learning, escala data, entrega metricas de performance
    y crea graficos
    
    Parameters
    ----------
    
    model: objeto model sklearn
    
    X: DataFrame. features
    
    Y: DataFrame. etiquetas    
    
    grafs : boolean, imprime graficos.
    
    lags : list de integers (de lags de X a incluir). i.e. lags=[1,2,12]
    
    output : DataFrame con las metricas de performance del modelo
    """
    
    """ crea lags """
    if lags != False:
        dlags={}
        
        #crea df's de lags
        for i in lags:
            dlags[i]= crealag(X, i)
                 
        # concat df de X con los df de lags   
        for i in lags:            
            X= pd.concat([X,dlags[i]],axis=1)
        
        # elimina las filas que quedan en desuso x la creación de nan's
        X.dropna(inplace=True)
        Y= Y.iloc[lags[-1]:,:]

    
    train_errors, test_errors, test_da, perc75_errors, perc95_errors=[],[],[],[],[]
    
    pipe= Pipeline([
        ("poly_feat", PolynomialFeatures(degree=degree,include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", MultiOutputRegressor(model))])
    
    
    # time series cross validator
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.values[train_index], X.values[test_index]
        Y_train, Y_test = Y.values[train_index], Y.values[test_index]
        
        # Fit        
        pipe.fit(X_train,Y_train)
        Y_train_pred= pipe.predict(X_train)
        Y_test_pred = pipe.predict(X_test)
        
        train_errors.append(mean_absolute_error(Y_train,Y_train_pred))
        test_errors.append( mean_absolute_error(Y_test,Y_test_pred))
        test_da.append(     da(Y_test,Y_test_pred) )
        
        perc95_errors.append(np.percentile(abs(Y_test-Y_test_pred) ,95))
        perc75_errors.append(np.percentile(abs(Y_test-Y_test_pred) ,75))
    
    train_errors= np.asarray(train_errors)
    test_errors= np.asarray(test_errors)
    
    learning_curves= np.concatenate((train_errors.reshape(-1,1),test_errors.reshape(-1,1)),axis=1)
    learning_curves= pd.DataFrame(data=learning_curves,columns=["train_error","test_error"])
    
    
    if grafs==True:        
    
        # PLot learning Curves
        learning_curves.plot(figsize=(11,6),fontsize=22)
        plt.xlabel("Training set size",fontsize=18)
        plt.ylabel("Mean Absolute Error",fontsize=18)
        plt.title("Model Learning Error Curves", fontsize=22)
        plt.legend(prop={'size': 18})
        plt.ylim(ymin=0)
        plt.show()
        
        #Modelo Final    
        Y2= pd.DataFrame(index=Y.index)
        Y2['Model_pred']= pipe.predict(X)
        Y2['Actual']=Y
        Y2.plot(figsize=(11,6),fontsize=22)
        plt.title("Model vs Actual", fontsize=22)
        plt.legend(prop={'size': 18})
        plt.show()
    
    return pd.DataFrame([round(100*np.mean(train_errors[-50:]) ,2),
              round(100*np.mean(test_errors[-50:])  ,2),
              round(100*np.mean(test_da[-50:])  ,2),              
              round(100*np.mean(perc75_errors[-50:]) ,2),
              round(100*np.mean(perc95_errors[-50:]) ,2)],         
             index=['train MAE','test MAE','test_da','75th percentile of test MAE','95th percentile of test MAE'],
             columns=['performance'])
    
    

def get_num(string):  
    '''This function retrieves numbers from a string and converts them to integers'''  
   
    # Create empty string to store numbers as a string  
    num = ''  
    # Loop through characters in the string  
    for i in string:  
        # If one of the characters is a number, add it to the empty string  
        if i in '1234567890':  
            num+=i  
    # Convert the string of numbers to an integer  
    integer = int(num)  
    return integer  



def P_equiv(x0):
    """ transforma la unidad de comercializacion a -->  # de kilos o # unidades
    x: str """
    
    #exactamente iguales
    if '$/kilo'==x0 or '$/unidad' == x0:
        return 1
    elif '$/cien' == x0:
        return 100
    
    # numero dentro de string
    else: 
        try:            
            if '$/kilo' in x0  or  '$/unidad' in x0:
                return 1
            
            elif ('$/bins' or '$/caja') and 'kilos' in x0:
                x1= get_num(x0)
                return int(x1)
            
            elif '$/caja' and 'unidades' in x0:
                x1= get_num(x0)
                return int(x1)
            
            else:   
                x1= get_num(x0)
                return int(x1)
    
        except:
            return float('nan')


def isnan(num):
    " evalua si num es nan"
    return num != num


def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out



def limpia(s):
    ''' proceso 0 funcion que reemplaza acentos para uniformar nombres de productos '''

    import re
    from unicodedata import normalize
    
    # -> NFD y eliminar diacríticos
    s = re.sub(
            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
            normalize( "NFD", s), 0, re.I
        )
    
    # -> NFC
    s = normalize( 'NFC', s)
    return s



def ipc_import(root,producto):
    
    """ Función que importa df del ipc (en %), del producto...
    
    root: Directorio raiz, a modificar por cada usuario i.e 'D:\\Dropbox\\Documentos\\IPC_ML\\' 
    
    producto: str i.e 'LIMON'
    """
    
    from dateutil.parser import parse
    os.chdir(root+'Git')
        
    """ 1 empalma archivo mas antiguo """
    df1= pd.read_excel(root+'Data\\ipc_producto_referencial_diciembre2013.xlsx',header=4,usecols=range(6,67))
    df1= df1[df1['GLOSA']==producto]
    df1= df1.T
    df1= df1.iloc[1:]
    df1.columns=[str(producto)]
    df1.index= pd.to_datetime(df1.index)
    
    
    """ 2 empalma archivo del medio """
    df2= pd.read_excel(root+'Data\\Analisis_ranking_vol_explicada_IPC.xlsx',header=4,usecols=range(6,67))
    df2= df2[df2['GLOSA']==producto]
    df2= df2.T
    df2= df2.iloc[1:]
    df2.columns=[str(producto)]
    df2.index= pd.to_datetime(df2.index)
    
    
    """ 3 empalma archivo más nuevo """
    df3= pd.read_excel(root+'Data\\ipc-2019-xls.xlsx',header=3,usecols=range(0,10+1))
    df3['Glosa']=df3['Glosa'].apply(lambda x: limpia(x))
    df3= df3[df3['Glosa']==producto]
    
    # crea indice datetime
    df3.index= [parse(str(df3['Año'][x])+' '+str(df3['Mes'][x]) +' 1') for x in df3.index]
    df3 = df3.rename(columns={'Índice': str(producto)})
    df3= df3[[str(producto)]]
    
    """ 4= 1+2+3 concat todo """
    df= pd.concat([df1,df2,df3])
    
    # transforma fecha a fin de mes
    df.index= df.resample('M', convention='end').asfreq().index
    
    return df.pct_change()
    
#ipc_import(root='D:\\Dropbox\\Documentos\\IPC_ML\\',producto='PALTA')
    


def odepa_may_impo(root='D:\\Dropbox\\Documentos\\IPC_ML\\', producto='Limon'):
    
    """ importa base odepa mayorista
    
    Parameters
    ----------
    root:
    producto: str o list of strings
    return: df
    """

    import os
    os.chdir(root+'Git')
    import hjson
    from aux_funcs import P_equiv
    
    
    """ 1.2   IMPORTA/FORMATEA/LIMPIA/ NUEVA BASE DE DATOS ODEPA  """
    
    df= pd.read_csv(root+'Data\\precios_frutas_hortalizas_odepa.csv',sep="\t",
                    encoding="latin-1",decimal=',', index_col=0)
    
    df= df[df.Producto.isin([producto])]
    
    df.Desde= pd.to_datetime(df.Desde,dayfirst=True)
    df.Hasta= pd.to_datetime(df.Hasta,dayfirst=True)
    
    
    # elimina puntos para que pueda evaluar correctamente: puntos como "thousands separator"
    for x in ['Volumen','Preciominimo','Preciomaximo','Preciopromedio']:
        df[x]= df[x].apply(lambda x: x.replace('.',''))
    
    # transforma str --> numeric
    for x in ['Volumen','Preciominimo','Preciomaximo','Preciopromedio']:
        df[x]= pd.to_numeric(df[x], errors='coerce')
    
    # calcula # kilos en Unidad de comercialización
    df['ud_com']= df['Unidad decomercializacion'].apply(lambda x: P_equiv(x))
    df['p']= df.Preciopromedio / df.ud_com
    
    
    
    """ filtra según criterios de diccionario """
    
    with open(root+'Git\\diccionarios\\diccio_odepa_may.json') as fp:
        d= hjson.loads(fp.read())
    
    df= df[ df.Variedad.isin( d[producto]['Variedad']  )]
    df= df[ df.Mercado.isin( d[producto]['Mercado']  )]
    df= df[ df.Origen.isin( d[producto]['Origen']  )]
    df= df[ df.Calidad.isin( d[producto]['Calidad']  )]
    df= df[ df['Unidad decomercializacion'].isin( d[producto]['Unidad decomercializacion']  )]
    
    # creo df con index fechas unicas, para guardar series de calidades del limon.
    dfc= pd.DataFrame(index=df.Hasta.unique())
    
    for c in df.Calidad.unique():
        dfc[c]= df[df.Calidad==c].groupby('Hasta').apply(lambda x: np.average(x.p, weights=x.Volumen))
    
    # crea df con cambios porcentuales de cada CALIDAD del limon
    dfcgrow= dfc.pct_change(limit=1)
    
    # pisa con nan's los ceros de la formula pct --> evita calc erroneos
    dfcgrow= dfcgrow.where(dfc.isna()==False,dfc)
    
    # promedia los pct de cada calidad en una sola columna
    dfcgrow= dfcgrow.mean(axis=1)

    # pisa los nan de crecimiento con el pct fila anterior
    dfcgrow= dfcgrow.interpolate(method='pad')
    
    
    
    """     PUEBLA LOS NAN CON EL CRECIMIENTO PROMEDIO DE DFCGROW """
    
    for col in dfc.columns:
        for ind,row in dfc[col].iteritems():
            
            if isnan(row)==True:
                # indice ordinal
                ilug= dfc[col].index.get_loc(ind)                   
                dfc[col][ind]= dfc[col].iloc[ilug-1] * (1+dfcgrow[ind])
    
    
    # alarga el indice a nivel "cambio diario"
    dfc= dfc.reindex( pd.date_range(start=dfc.index[0], end=dfc.index[-1]))
    dfc= dfc.interpolate(method='index')
    
    #mensualiza
    dfc= dfc.asfreq('M')
    dfc= dfc.pct_change()
    
    dfc.dropna(inplace=True)
    
    return dfc



###############################################################################
###############################################################################


def odepa_cons_impo(root='D:\\Dropbox\\Documentos\\IPC_ML\\', producto='Limon'):
    
    """ importa base odepa mayorista
    
    Parameters
    ----------
    root:
    producto: str o list of strings
    return: df
    """

    import os
    os.chdir(root+'Git')
    import hjson
    
    
    """ 1.2   IMPORTA/FORMATEA/LIMPIA/ BASE DE DATOS ODEPA CONSUMIDOR """
    
    if producto==producto:
#    if producto in ['Limon','Papa']:
    
        df= pd.read_csv(root+'Data\\precios_odepa_consum_total.csv',
                        sep="\t", encoding="latin-1",decimal=',', index_col=0)
        
        df= df[df.Producto.isin([producto])]
        
        df['Fecha inicio'] = pd.to_datetime(df['Fecha inicio'],dayfirst=True)
        df['Fecha termino']= pd.to_datetime(df['Fecha termino'],dayfirst=True)  
        
        # transforma str --> numeric
        df['Precio promedio']= pd.to_numeric(df['Precio promedio'], errors='coerce')
        
            
        """ filtra según criterios de diccionario """
        
        with open(root+'Git\\diccionarios\\diccio_odepa_consum.json') as fp:
            d= hjson.loads(fp.read())
        
        df= df[ df.Calidad.isin( d[producto]['Calidad']  )] if (d[producto]['Calidad'] != [str('nan')]) else df
        df= df[ df.Procedencia.isin( d[producto]['Procedencia'] )] if (d[producto]['Procedencia'] != [str('nan')]) else df
        df= df[ df.Region.isin( d[producto]['Region']  )] if (d[producto]['Region'] != [str('nan')]) else df
        df= df[ df.Sector.isin( d[producto]['Sector']  )] if (d[producto]['Sector'] != [str('nan')]) else df
        df= df[ df['Tipo punto monitoreo'].isin( d[producto]['Tipo punto monitoreo']  )] if (d[producto]['Tipo punto monitoreo'] != [str('nan')]) else df
        df= df[ df.Unidad.isin( d[producto]['Unidad']  )] if (d[producto]['Unidad'] != [str('nan')]) else df
        df= df[ df.Variedad.isin( d[producto]['Variedad']  )] if (d[producto]['Variedad'] != [str('nan')]) else df
        
        
        """ creo df con index fechas unicas, para guardar series de calidades del limon."""
        dfc= pd.DataFrame(index=df['Fecha termino'].unique())
        
        for c in df.Calidad.unique():
            dfc[c]= df[['Calidad','Fecha termino','Precio promedio']][df.Calidad==c].groupby('Fecha termino').mean()
        
        # crea df con cambios porcentuales de cada CALIDAD del limon
        dfcgrow= dfc.pct_change(limit=1)
        
        # pisa con nan's los ceros de la formula pct --> evita calc erroneos
        dfcgrow= dfcgrow.where(dfc.isna()==False,dfc)
        
        # promedia los pct de cada calidad en una sola columna
        dfcgrow= dfcgrow.mean(axis=1)
    
        # pisa los nan de crecimiento con el pct fila anterior
        dfcgrow= dfcgrow.interpolate(method='pad')
    
    
    
        """     PUEBLA LOS NAN CON EL CRECIMIENTO PROMEDIO DE DFCGROW """
        
        for col in dfc.columns:
            for ind,row in dfc[col].iteritems():
                
                if isnan(row)==True:
                    # indice ordinal
                    ilug= dfc[col].index.get_loc(ind)                   
                    dfc[col][ind]= dfc[col].iloc[ilug-1] * (1+dfcgrow[ind])
        
        
        # alarga el indice a nivel "cambio diario"
        dfc= dfc.reindex( pd.date_range(start=dfc.index[0], end=dfc.index[-1]))
        dfc= dfc.interpolate(method='index')
        
        #mensualiza
        dfc= dfc.asfreq('M')
        dfc= dfc.pct_change()
        
        dfc.dropna(inplace=True)
        
        return dfc

    
    
    
    else:
        return print('nada')
    

