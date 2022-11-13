#codigo para convertir etiquetas de boundingBox a etiquetas de yolov7 
import os
import math

folder = r'/home/leonel/Documents/Escuela/Servicio/mediciones/todas//'
folder2 = r'/home/leonel/Documents/Escuela/Servicio/mediciones/todas/'

with os.scandir(folder) as entries:
    for entry in entries:
        etiqueta = []
        indice = entry.name.index("_")
        etiqueta.append(int(entry.name[indice+2:indice+3])-1)
        
        #folder2 es para darle la ubicación del archivo
        archivo = open(folder2+entry.name,'r')
        linea = archivo.readline()
        #print(linea)
        coords = []
        for i in range(3):
            linea =  linea.strip()
            indice = linea.index(" ")
            coords.append(int(linea[:indice]))
            linea = linea[indice:]
        coords.append(int(linea[:]))
            
        #print (coords)
        
        etiqueta.append(math.floor((coords[0]+coords[2])/2)/856)
        etiqueta.append(math.floor((coords[1]+coords[3])/2)/480)
        etiqueta.append(math.floor(coords[2]-coords[0])/856)
        etiqueta.append(math.floor(coords[3]-coords[1])/480)
        print(etiqueta)
        archivo.close()
        linea = ' '.join(map(str,etiqueta))
        print(linea)
        archivo = open(folder2+entry.name,'w')
        archivo.write(linea)
        archivo.close
