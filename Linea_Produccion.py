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
from aux_funcs import odepa_cons_impo, odepa_may_impo


df= odepa_may_impo()
df.plot()
