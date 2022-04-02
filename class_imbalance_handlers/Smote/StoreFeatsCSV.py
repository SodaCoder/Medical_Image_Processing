import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
import os
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Global Data
allImagesPath = "../5classPreprocessed/all_images"
trainValPath = "../5classPreprocessed/train_val.csv"
testPath = "../5classPreprocessed/test.csv"
base_model = tf.keras.applications.ResNet50(pooling = "avg", include_top = False, weights= "imagenet")

# Extracting Features, Preparing the data, and Storing the data 
def createStoreImageDataset(csvPath, allImagesPath, storeFeatureCSVPath):
    global base_model

    imageDf = pd.read_csv(csvPath, skiprows = [0])
    records = imageDf.to_records(index = False)
    result = list(records)
    imageFeatList = []
    labelList = []
    
    input = Input(shape=(224, 224, 3),name = 'image_input')
    x = base_model(input)
    x = Flatten()(x)
    model = Model(inputs=input, outputs=x)
    
    for index in tqdm(range(len(result))):
        imgName = result[index][0]
        label = int(result[index][1])
        
        image = cv2.imread(allImagesPath + "/" + imgName)
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))
        if image.any() == None:
            break
        image = np.expand_dims(image, axis = 0)
        x = preprocess_input(image)
        features = model.predict(x)

        features_reduce = features.squeeze()
        imageFeatList.append(features_reduce)
        labelList.append(label)

    imageFeatList = np.array(imageFeatList)
    scaler = MinMaxScaler()
    imageFeatList = scaler.fit_transform(imageFeatList)
    labelList = np.array(labelList)
    labelList = labelList.reshape(labelList.shape[0], 1)
    print("Shape of ImageFeatureList: ", imageFeatList.shape)
    print("Shape of labelList: ", labelList.shape)
    datasetMatrix = np.concatenate((imageFeatList, labelList), axis=1)
    pd.DataFrame(datasetMatrix).to_csv(storeFeatureCSVPath, sep=',', header=None, index=None)
    print("1st Image Feature is : ", imageFeatList[0], " Length is: ", len(imageFeatList[0]), " Label is: ", labelList[0])
    #return imageFeatList, labelList

print("Creating the combined Training and Validation Data...")
createStoreImageDataset(trainValPath, allImagesPath, "../5classPreprocessed/train_val_feat_label.csv")
print("Creating the Test Data...")
createStoreImageDataset(testPath, allImagesPath, "../5classPreprocessed/test_feat_label.csv")
