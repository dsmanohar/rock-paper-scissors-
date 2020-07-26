import keras
import cv2
import numpy
from keras.models import load_model
img=cv2.imread('maskimage2.PNG')
img=img/255
img = numpy.expand_dims(img, axis=0)
model = load_model('newmodel.h5')
pred = model.predict_classes(img)
print(pred)