#https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/

import numpy as np
import shutil, os
import sys

from os import scandir, getcwd
from sklearn.model_selection import train_test_split
def ls(ruta = getcwd()):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]



# número de imágenes en cada carpeta res(taza A,angulo 000,distancia 60)
X = np.arange(0,288)

letras = "ABCDE"

os.mkdir("data/train")     
os.mkdir("data/validation")     
os.mkdir("data/test")     
for letra in letras:
    os.mkdir('data/train/'+letra)     
    os.mkdir('data/test/'+letra)     
    os.mkdir('data/validation/'+letra)     



for letra in letras:
    X_train , X_test = train_test_split(X,test_size = 0.20)
    print (X_train)
    print (X_test)
    lista_arq = ls('res'+letra)
    X_val_train , X_val_test = train_test_split(X_train,test_size = 0.20)
    print (X_val_train)
    print (X_val_test)
    for train in X_val_train:
        origen = 'res'+letra+'/'+lista_arq[train]
        destino = 'data/train/'+letra+'/' + lista_arq[train]
        try:
            shutil.copyfile(origen, destino)
            print("Archivo copiado")
        except:
            print("Se ha producido un error")
    for test in X_val_test:
        origen = 'res'+letra+'/' + lista_arq[test]
        destino = 'data/validation/'+letra+'/' + lista_arq[test]
        try:
            shutil.copyfile(origen, destino)
            print("Archivo copiado")
        except:
            print("Se ha producido un error")        
    for test in X_test:
        origen = 'res'+letra+'/' + lista_arq[test]
        destino = 'data/test/'+letra+'/' + lista_arq[test]
        try:
            shutil.copyfile(origen, destino)
            print("Archivo copiado")
        except:
            print("Se ha producido un error")        


sys.exit()
