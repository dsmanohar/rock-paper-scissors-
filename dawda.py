import tensorflow
import keras
import numpy
import cv2
import PIL
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model=Sequential()
model.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(output_dim=64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(output_dim=3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'maskdataset/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        'maskdataset/test',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')
model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=200)
model.save("newmodel.h5")