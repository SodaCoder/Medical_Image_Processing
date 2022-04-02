import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.np_utils import to_categorical
import dense_suppli as spp
import os

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Global Data
EPOCHS = 200
EARLY_SOPPING_PATIENCE = 10
filePath = "./logs/5classPreprocessed_MLP_resnet50_smote.h5"
allImagesPath = "../5classPreprocessed/all_images"
trainValPath = "../5classPreprocessed/train_val.csv"
testPath = "../5classPreprocessed/test.csv"
AdamOpt=Adam(0.0002, 0.5)
latDim, modelSamplePd, resSamplePd=100, 5000, 500
batchSize, max_step=32, 50000
fileName = ["../5classPreprocessed/train_val_feat_label.csv", "../5classPreprocessed/test_feat_label.csv"]
fileStart = "../Covidnet_Dataset_Preprocessed/"
fileStart='./logs/5classPreprocessed/'
fileEnd, savePath='_Model.h5', fileStart+'/'
base_model = tf.keras.applications.ResNet50(pooling = "avg", include_top = False, weights= "imagenet")

# Function to create a dense MLP
def denseMlpCreate():
    
    imIn=Input(shape = (2048,))
    x = Dense(784)(imIn)
    x = LeakyReLU(alpha = 0.1)(x)
    x = Dense(256)(imIn)
    x = LeakyReLU(alpha = 0.1)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    mlpFinal = Dense(5, activation = "softmax")(x)
    mlp = Model(imIn, mlpFinal)
    mlp.summary()
    return mlp

trainS, labelTr=spp.fileRead(fileName[0])
testS, labelTs=spp.fileRead(fileName[1])

n, m = trainS.shape[0], testS.shape[0]
# sm = SMOTE(random_state = 42)
# X_train_smote, Y_train_smote = sm.fit_resample(X_train, Y_train)

labelTr, labelTs, c, pInClass, classMap=spp.relabel(labelTr, labelTs)
print("Relabeling is done...")
imbalancedCls, toBalance, imbClsNum, ir=spp.irFind(pInClass, c)
print("Imbalance Ratio is found...")
  
print("Now SMOTE started...")
trainS_mlp, labelTr_mlp, checkImages=spp.smote(trainS, labelTr, pInClass, toBalance, classMap)
print("SMOTE finished...")
labelsCat_mlp=to_categorical(labelTr_mlp)

mlp = denseMlpCreate()
mlp.compile(optimizer = AdamOpt, loss = "mean_squared_error", metrics=["accuracy"])

n_mlp=trainS_mlp.shape[0]
shuffleIndex=np.random.choice(np.arange(n_mlp), size=(n_mlp,), replace=False)
trainS_mlp=trainS_mlp[shuffleIndex]
labelTr_mlp=labelTr_mlp[shuffleIndex]
labelsCat_mlp=labelsCat_mlp[shuffleIndex]
batchDiv, numBatches, _=spp.batchDivision(n_mlp, batchSize)
      
iter=np.int(np.ceil(max_step/resSamplePd)+1)
acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))
ppvSaveTr, ppvSaveTs=np.zeros((iter, c)), np.zeros((iter, c))            

print("Shape of Training Set: ", trainS.shape)

step=0
while step<max_step:
    for j in range(numBatches):
        x1, x2=batchDiv[j, 0], batchDiv[j+1, 0]
        mlp.train_on_batch(trainS_mlp[x1:x2], labelsCat_mlp[x1:x2])
       
        if step%resSamplePd==0:
            saveStep=int(step//resSamplePd)
            pLabel=np.argmax(mlp.predict(trainS), axis=1)
            acsa, gm, tpr, confMat, acc, ppv=spp.indices(pLabel, labelTr)
            print('Train: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            print('PPV: ', np.round(ppv, 2))
            acsaSaveTr[saveStep], gmSaveTr[saveStep], accSaveTr[saveStep]=acsa, gm, acc
            confMatSaveTr[saveStep]=confMat
            tprSaveTr[saveStep]=tpr
            ppvSaveTr[saveStep] = ppv
            pLabel=np.argmax(mlp.predict(testS), axis=1)
            acsa, gm, tpr, confMat, acc, ppv=spp.indices(pLabel, labelTs)
            print('Test: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            print('PPV: ', np.round(ppv, 2))
            acsaSaveTs[saveStep], gmSaveTs[saveStep], accSaveTs[saveStep]=acsa, gm, acc
            confMatSaveTs[saveStep]=confMat
            tprSaveTs[saveStep]=tpr
            ppvSaveTs[saveStep] = ppv

        if step%modelSamplePd==0 and step!=0:
            mlp.save(savePath+'MLP_'+str(step)+fileEnd)
                        
        step=step+1
        if step>=max_step: break

print("Shape of Training Set: ", trainS.shape)

pLabel=np.argmax(mlp.predict(trainS), axis=1)
acsa, gm, tpr, confMat, acc, ppv = spp.indices(pLabel, labelTr)
print('Performance on Train Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
print('PPV: ', np.round(ppv, 2))
acsaSaveTr[-1], gmSaveTr[-1], accSaveTr[-1]=acsa, gm, acc
confMatSaveTr[-1]=confMat
tprSaveTr[-1]=tpr
ppvSaveTr[-1] = ppv

pLabel=np.argmax(mlp.predict(testS), axis=1)
acsa, gm, tpr, confMat, acc, ppv = spp.indices(pLabel, labelTs)
print('Performance on Test Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
print('PPV: ', np.round(ppv, 2))
acsaSaveTs[-1], gmSaveTs[-1], accSaveTs[-1]=acsa, gm, acc
confMatSaveTs[-1]=confMat
tprSaveTs[-1]=tpr
ppvSaveTs[-1] = ppv

mlp.save(savePath+'MLP_'+str(step)+fileEnd)
resSave=savePath+'Results'
np.savez(resSave, acsa=acsa, gm=gm, tpr=tpr, confMat=confMat, acc=acc)
recordSave=savePath+'Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr,acsaSaveTs=acsaSaveTs, gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs, ppvSaveTr = ppvSaveTr, ppvSaveTs = ppvSaveTs)
