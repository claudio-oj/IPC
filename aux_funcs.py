# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:25:51 2019

@author: COJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split



def run_model(model, X, Y):
    """ corre modelos de Machine Learning y crea graficos """
    X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2,random_state=1)
    m0=len(X_train.copy())
    train_errors, test_errors, perc75_errors, perc95_errors=[],[],[],[]
    
    for m in range(5,m0+1):
        model.fit(X_train[:m],Y_train[:m])
        Y_train_pred= model.predict(X_train)
        Y_test_pred= model.predict(X_test)
        train_errors.append(mean_absolute_error(Y_train,Y_train_pred))
        test_errors.append(mean_absolute_error(Y_test,Y_test_pred))
        perc95_errors.append(np.percentile(abs(Y_test.values-Y_test_pred) ,95))
        perc75_errors.append(np.percentile(abs(Y_test.values-Y_test_pred) ,75))
    
    train_errors= np.asarray(train_errors)
    test_errors= np.asarray(test_errors)
    
    learning_curves=np.concatenate((train_errors.reshape(-1,1),test_errors.reshape(-1,1)),axis=1)
    learning_curves=pd.DataFrame(data=learning_curves,columns=["train_error","test_error"])
    
    print('train MAE ', round(100*np.mean(train_errors[-50:]) ,2) )
    print('test MAE ',  round(100*np.mean(test_errors[-50:])  ,2) )
    print('75th percentile of test MAE ',round(100*np.mean(perc75_errors[-50:]) ,2))
    print('95th percentile of test MAE ',round(100*np.mean(perc95_errors[-50:]) ,2))
    
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
    Y2['Model_pred']= model.predict(X)
    Y2['Actual']=Y
    Y2.plot(figsize=(11,6),fontsize=22)
    plt.title("Model vs Actual", fontsize=22)
    plt.legend(prop={'size': 18})
    plt.show()
    
    

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
    """ transforma la unidad de comercializacion a -->  # de kilos
    x: str """
    try:
        x1= get_num(x0)

        if '$/kilo' in x0:
            return 1    
        else:        
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

def crealag(df, n):
    new_columns = ["{}_Lag{:02d}".format(variable, n) for variable in df.columns]
    new_df = df.shift(n)
    new_df.columns = new_columns
    return new_df
    