import pandas as pd
import os
import shutil #copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf
autoencoder = tf.keras.models.load_model("5000bs256.h5")
#make the dataset
#give path
source_dir="Preprocessed_Clahe15/Lung_Opacity/"
target_dir="claheBR/Lung_Opacity/"
'''if not os.path.exists(target_dir):
    os.mkdir(target_dir)
    print("Pneumonia_bone_removal folder created")
'''

def load_data(path):
    img_size = (256,256)
    imgs_source = []
    imgs_target = []
    image=cv2.imread(path,0)
    image=cv2.resize(image,img_size)
    image_array=np.array(image)/255
    return image_array

cnt=0
for i in range(len(filenames)):
    filepath=os.path.join(source_dir,filenames[i])
    image=load_data(filepath)
    imagelist=np.array(image).reshape(-1,256,256,1)
    pred = autoencoder.predict(imagelist)
    updated_filename=target_dir+"BR"+filenames[i]+".jpg"
    plt.imsave(updated_filename,pred[0].reshape(256,256),cmap="gray")
    cnt=cnt+1
    if(cnt==1):
        print(cnt," Image done")
        break

