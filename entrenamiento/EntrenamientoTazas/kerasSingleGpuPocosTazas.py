import GPUtil
import time
import multiprocessing
import tensorflow as tf
import numpy as np
import warnings
import os
import pickle

import pandas as pd

from keras.backend.tensorflow_backend import set_session, clear_session, get_session
from tensorflow.python.framework.errors_impl import ResourceExhaustedError, UnknownError
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.client import device_lib
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import plot_model
import matplotlib.pyplot as plt 
from os import scandir, getcwd
from operator import itemgetter
import numpy as np

import seaborn as sn
import seaborn as sns

import sys


#tf.debugging.set_log_device_placement(True)

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 32} )

mem_amount = 0.6
gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=mem_amount)
config = ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2,gpu_options=gpu_options)
#config = ConfigProto(gpu_options=gpu_options)
sess =  tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)



def ls(ruta = getcwd()):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

base_dir = 'data'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')

letras=os.listdir(base_dir+'/train')
n_classes=len(letras)

letras.sort()
index=0
arreglo = np.zeros(len(letras))
for letra in letras:
    arreglo[index]=index
    index=index+1

dic = dict(zip(letras,arreglo))
items = dic.items()




model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(192,(5,5),activation='relu',input_shape=(48,48,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(192,(5,5),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(192,(5,5),activation='relu'),  #extra
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(768, activation='relu'),
    tf.keras.layers.Dense(384, activation='relu'),
    tf.keras.layers.Dense(n_classes,activation='softmax')
])
#model.summary()

#plot_model(model, to_file='modeloAngulo.png')
tf.keras.utils.plot_model(model, to_file='modeloTaza.png')


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
#model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['acc'])


train_datagen = ImageDataGenerator(
    rescale=1./255,
    #rotation_range=180,
    #width_shift_range=0.2,
    #shear_range=0.2,
   #zoom_range=0.2,
    #horizontal_flip=True,
    #fill_mode='nearest'
    )
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=16,
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(48,48),
    batch_size=16,
    class_mode='categorical')


history = model.fit(
    train_generator,
    steps_per_epoch = 10,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=5,
    verbose=2)

f = open('history.pckl', 'wb')
pickle.dump(history.history, f)
f.close()



fig = plt.figure(figsize=(10,5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Modelo (Rendimiento)')
plt.ylabel('Rendimiento')
plt.xlabel('Epoca')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
fig.savefig('EntrenamientoAcc.jpg', bbox_inches='tight', dpi=150)


#plt.show()

fig = plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Modelo (Pérdida)')
plt.ylabel('Pérdida')
plt.xlabel('Epoca')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
#plt.show()
fig.savefig('EntrenamientoLoss.jpg', bbox_inches='tight', dpi=150)


MC=np.zeros((n_classes,n_classes))
for letra in letras:
    listaTest = ls(base_dir+'/test/'+letra)
    for imagenAct in listaTest:
        nombre=base_dir+'/test/'+letra+'/'+imagenAct
        img = load_img(nombre,target_size=(48,48))
        x = img_to_array(img)
        x = np.expand_dims(x,axis=0)
        images  = np.vstack([x])
        classes = model.predict(images)
        numbers_sort = sorted(enumerate(classes[0]), key=itemgetter(1),  reverse=True)
        index, value = numbers_sort[0]
        MC[int(dic.get(letra))][index]+=1     

for y in range(n_classes):
    suma=0
    for x in range(n_classes):
        suma=suma+MC[y][x]
        print ("%.0f" % MC[y][x],end="   ")
    mull=100.00*MC[y][y]/suma
    print ("%.2f" % mull)
suma=0
total=0
for y in range(n_classes):
    for x in range(n_classes):
        total=total+MC[y][x]
    suma=suma+MC[y][y]

print(suma)
print(total)
print(suma/total*100)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model500Epocas.h5")



sum = np.sum(MC, axis=1)

print(sum)

MC = MC * 100.0 / ( 1.0 * sum )
df_cm = pd.DataFrame(MC, letras,letras)
#fig = plt.figure()
fig = plt.figure(figsize=(15, 15), dpi=80)

plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
res = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=100.0, fmt='.2f', cmap=cmap)
#res.invert_yaxis()

#plt.xticks(np.linspace(10, 350, 18, endpoint=True))
#plt.yticks(np.linspace(10, 350, 18, endpoint=True))

#plt.yticks(arregloNombre, letras,va='center')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight' )
plt.close()



