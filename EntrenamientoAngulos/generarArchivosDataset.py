import keras


import numpy as np
import shutil, os
import sys

from os import scandir, getcwd
from sklearn.model_selection import train_test_split
def ls(ruta = getcwd()):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]



# número de imágenes en cada carpeta res(taza A,angulo 000,distancia 60)
X = np.arange(0,75)

#letras = [ "000","010","020","030","040","050","060","070","080","090",
#           "100","110","120","130","140","150","160","170","180","190",
#           "200","210","220","230","240","250","260","270","280","290",
#           "300","310","320","330","340","350"]

#letras = [  "000","020","040","060","080",
#            "100","120","140","160","180","190",
#            "200","220","240","260","280","290",
#            "300","320","340"]

letras = ["000","045","090","135","180","225","270","315"]

os.mkdir("data/train")     
os.mkdir("data/validation")     
os.mkdir("data/test")     
for letra in letras:
    os.mkdir('data/train/'+letra)     
    os.mkdir('data/test/'+letra)     
    os.mkdir('data/validation/'+letra)     



for letra in letras:
#    X = np.arange(0,100)
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
