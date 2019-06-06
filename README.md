<center> <img src="https://github.com/claudio-oj/IPC/blob/master/imagenes/logo_BA.JPG" width="150" />
</center>

# IPC_model
data, proceso, modelos



Diccionarios:
  - codigo que crea diccionario en formato HJSON de caracteristicas x producto odepa mayorista.
  - codigo que crea diccionario en formato HJSON de caracteristicas x producto odepa consumidor

Modelos:
  - guarda los modelos ML entrenados, de acuerdo a la metodologia de grupos (G1,G2,G3...)

  
<img src="https://github.com/claudio-oj/IPC/blob/master/imagenes/ipc_ml_diagram.png"
     width="1000" 
     style="float: center; margin-right: 10px;" />


Definiciones:
- todos los archivos de data se guardaron sin acentos. Por ende, luego de cada web scraping hay que remover
los acentos para concatenar la data.

- En base consumidor: para los productos AZUCAR y MIEL, la base no traía nombre de producto.
Agregué a mano esos nombres de producto --> replicar en el scraping