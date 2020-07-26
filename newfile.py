import tensorflow
import keras
import numpy
import cv2
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
dataset=[]
classname=["stone","paper","scissors"]
for index,value in enumerate(classname):
    for files in os.listdir("maskdataset"+"/"+str(value)):
        img=cv2.imread('maskdataset/'+str(value)+'/'+files,0)
        dataset.append([img,index])
import random
import numpy
random.shuffle(dataset)
x_data=[]
y_data=[]
for i in dataset:
    x_data.append(i[0])
    y_data.append(i[1])
x_train=numpy.array(x_data[:100])
x_test=numpy.array(x_data[100:])
y_train=to_categorical(y_data[:100])
y_test=to_categorical(y_data[:100])
x_train=numpy.expand_dims(x_train,-1)
x_test=numpy.expand_dims(x_test,-1)
model=Sequential()
model.add(Convolution2D(32,3,3,input_shape=(128,128,1),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(output_dim=64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(output_dim=3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),validation_data=(x_train,y_train),
                    steps_per_epoch=600, epochs=10)


