import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
import tensorflow as tf


def load_data(path):
    img_size = (256,256)
    imgs_source = []
    imgs_target = []
    image=cv2.imread(path,0)
    image=cv2.resize(image,img_size)
    image_array=np.array(image)/255
    return image_array
    
    
filepath="Normal-3.png"
image=load_data(filepath)
plt.imshow(image,cmap="gray")
plt.title(filepath)
plt.show()
img_channels=1
img_shape=(256,256,1)
imagelist=np.array(image).reshape(-1,256,256,1)
print(imagelist.shape)
print(imagelist[0].shape)

autoencoder = tf.keras.models.load_model("5000bs256.h5")
pred = autoencoder.predict(imagelist)
print(len(pred))
print(pred[0].shape)
plt.imsave("supressed.png",pred[0].reshape(256,256),cmap="gray")
