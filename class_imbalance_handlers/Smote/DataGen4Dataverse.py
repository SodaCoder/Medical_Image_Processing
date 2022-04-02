import numpy as np
from numpy import *
import cv2
import pandas as pd
import os
from tqdm import tqdm

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

allImagesPath = "../Output_5cls/all_images"
trainPath = "../Output_5cls/train.csv"
valPath = "../Output_5cls/val.csv"
testPath = "../Output_5cls/test.csv"
trainStoreNPZPath = "../Output_5cls/Dataset5_raw_train.npz"
valStoreNPZPath = "../Output_5cls/Dataset5_raw_val.npz"
testStoreNPZPath = "../Output_5cls/Dataset5_raw_test.npz"

def createStoreImageDataset(csvPath, allImagesPath, storeNPZPath):
    imageDf = pd.read_csv(csvPath, skiprows = [0])
    records = imageDf.to_records(index = False)
    result = list(records)
    
    numSamples = len(result)
    
    imagesArr = np.zeros((numSamples, 224, 224, 3))
    imageLabelArr = np.zeros((numSamples, 1))
    imageNameList = []
    
    for index in tqdm(range(numSamples)):
        imgName = result[index][0]
        label = int(result[index][1])
        
        imageNameList.append(imgName)
        imageLabelArr[index] = label
        
        image = cv2.imread(allImagesPath + "/" + imgName)
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))
        if image.any() == None:
            break
        imagesArr[index] = image.astype(int)
      
    imageNameArr = np.asarray(imageNameList)
    imageNameArr = imageNameArr.reshape(imageNameArr.shape[0], 1)
    
    print("Started saving to a compressed .npz file...")
    np.savez_compressed(storeNPZPath, image = imagesArr, image_name = imageNameArr, image_label = imageLabelArr)
    print("Completed saving to a compressed .npz file...")


print("Creating the Training Data...")
createStoreImageDataset(trainPath, allImagesPath, trainStoreNPZPath)
print("Creating the Validation Data...")
createStoreImageDataset(valPath, allImagesPath, valStoreNPZPath)
print("Creating the Test Data...")
createStoreImageDataset(testPath, allImagesPath, testStoreNPZPath)
