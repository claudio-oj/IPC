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
from aux_funcs import ipc_import,odepa_cons_impo, odepa_may_impo


""" LIMON """
df= ipc_import(root=root, producto='LIMON')
df= df.join(odepa_may_impo())
df= df.join(odepa_cons_impo(),rsuffix='_cons')

# run model, elige modelo

# guarda en carpeta modelo ganador

