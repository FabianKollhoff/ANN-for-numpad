# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:13:15 2020

@author: Fabian Kollhoff
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils.generic_utils import get_custom_objects
import pickle
import keras
from keras.utils import to_categorical

#Laden der Bilder
pickle_in = open("X_64_1.pickle","rb")
X = pickle.load(pickle_in)
#Lade korespondierende Angaben der Kategorie der Bilder (Label)
pickle_in = open("y_64_1.pickle","rb")
y = pickle.load(pickle_in)

#Teile Alle werte durch den Maximalwert von 255, damit die Werte der Bilder zwischen 0-1 sind.
X = X/255.0

#Definiere die Bildgroesse
IMG_SIZE = 64

#Konvertiere die Graustufenbilder in ein Numpy-Array
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Konvertiere die Label der Bilder in ein Numpy-Array
y = np.array(y)
#Konvertiere die Label in Zahlenwerten von 0-9 in 10-dimensionale Vektoren 
y_binary = to_categorical(y)

def custom_gelu(x):
	""" custom_gelu(x)
		Die Funktion berechnet aus der Aktivität x mit Hilfe der GeLU-Funktion, die in
		der Arbeit vorgestellt wurde, die neue Aktivität.
	"""
    return x *  0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

#Fuege neue GeLU-Funktion hinzu
get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})

#Erstelle ein lineares Modell
model = Sequential()

#Füge die Layer und mit Aktivitätsfunktion hinzu
model.add(Conv2D(32, (7, 7), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(MaxPooling2D((3, 3)))

model.add(Conv2D(32, (3,3)))
model.add(Activation(custom_gelu))
model.add(Dropout(0.4))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(40, (3, 3), activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))

#Kompiliere das Modells mit Festlegen der kategorischen Kreuzentropiefunktion und dem Optimierungsalgorithmus Adam.
#Zum Schluss wird die Genauigkeit als zusätzlicher Parameter, der mit angeben wird beim Training, definiert.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Erstelle ein Tensorbaord, auf welchem der Graph später aufgezeichnet werden kann.
tbCallBack = keras.callbacks.TensorBoard(log_dir='CNN_64_6_6', histogram_freq=0, write_graph=True, write_images=True)

#Trainiere das Modell mit den angeben Parametern
model.fit(X, y_binary, batch_size=256, epochs=100,verbose=1, validation_split=0.2, callbacks=[tbCallBack])

#Auswertung des Modells mit allen Daten
scores = model.evaluate(X, y_binary, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Speichere Modell als Datei.
print("Saved model to disk")
model.save('CNN_64_6_6.model')