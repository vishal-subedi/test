#importing modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import cv2
from sklearn.metrics import confusion_matrix
ss = StandardScaler()


#CNN
clf = Sequential()

#1st Convolution layer
clf.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
#1st Max Pooling layer
clf.add(MaxPooling2D(pool_size = (2, 2)))
#2nd Convolution Layer
clf.add(Convolution2D(32, 3, 3, activation = 'relu'))
#2nd Convolution layer
clf.add(MaxPooling2D(pool_size = (2, 2)))

#Flatten layer
clf.add(Flatten())

#Fully Connected layers 2 hidden and 1 outout layer
clf.add(Dense(128, activation = 'relu'))
clf.add(Dense(32, activation = 'relu'))
clf.add(Dense(1, activation = 'sigmoid'))

#setting parameters for backpropogation
clf.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy']) 



#image augmentation
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('CNN/dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory('CNN/dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')



#training
clf.fit_generator(train_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)




#reading image
img = cv2.imread('Downloads/cat3.jpg')
#resizing image according to training dataset
res_img = cv2.resize(img, (64, 64))
#resizing again according to convolution layer input
y = np.reshape(res_img, [1, 64, 64, 3])

#testng
pred = clf.predict(y)

#predicted class
if pred == 0:
    print('Cat')
else:
    print('Dog')
