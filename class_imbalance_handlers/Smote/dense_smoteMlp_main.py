# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import os
import numpy as np
import dense_suppli as spp
import dense_net as nt
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

fileName=['Mnist_100_trainData.csv', 'Mnist_100_testData.csv']
fileStart='Mnist_100_Smote_mlp'
fileEnd, savePath='_Model.h5', fileStart+'/'
adamOpt=Adam(0.0002, 0.5)
latDim, modelSamplePd, resSamplePd=100, 5000, 500
plt.ion()

batchSize, max_step=32, 50000

trainS, labelTr=spp.fileRead(fileName[0])
testS, labelTs=spp.fileRead(fileName[1])

n, m=trainS.shape[0], testS.shape[0]
trainS, testS=(trainS-127.5)/127.5, (testS-127.5)/127.5

labelTr, labelTs, c, pInClass, classMap=spp.relabel(labelTr, labelTs)
imbalancedCls, toBalance, imbClsNum, ir=spp.irFind(pInClass, c)

trainS_mlp, labelTr_mlp, checkImages=spp.smote(trainS, labelTr, pInClass, toBalance, classMap)
labelsCat_mlp=to_categorical(labelTr_mlp)

if not os.path.exists(fileStart):
    os.makedirs(fileStart)

fig, axs=plt.subplots(imbClsNum, 3)
for i1 in range(imbClsNum):
    for i2 in range(3):
        img=np.reshape(checkImages[(i1*3)+i2, :], (28, 28))
        img=image.array_to_img(np.expand_dims(img, axis=-1), scale=True)
        axs[i1,i2].imshow(img, cmap='gray')
        axs[i1,i2].axis('off')
plt.show()
plt.pause(5)
figFileName=savePath+fileStart+'.png'
plt.savefig(figFileName, bbox_inches='tight')
print('SMOTE data generation finished, training ...')

mlp=nt.denseMlpCreate()
mlp.compile(loss='mean_squared_error', optimizer=adamOpt)

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

step=0
while step<max_step:
    for j in range(numBatches):
        x1, x2=batchDiv[j, 0], batchDiv[j+1, 0]
        mlp.train_on_batch(trainS_mlp[x1:x2], labelsCat_mlp[x1:x2])

        if step%resSamplePd==0:
            saveStep=int(step//resSamplePd)

            pLabel=np.argmax(mlp.predict(trainS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
            print('Train: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTr[saveStep], gmSaveTr[saveStep], accSaveTr[saveStep]=acsa, gm, acc
            confMatSaveTr[saveStep]=confMat
            tprSaveTr[saveStep]=tpr

            pLabel=np.argmax(mlp.predict(testS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
            print('Test: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTs[saveStep], gmSaveTs[saveStep], accSaveTs[saveStep]=acsa, gm, acc
            confMatSaveTs[saveStep]=confMat
            tprSaveTs[saveStep]=tpr

        if step%modelSamplePd==0 and step!=0:
            mlp.save(savePath+'MLP_'+str(step)+fileEnd)

        step=step+1
        if step>=max_step: break

pLabel=np.argmax(mlp.predict(trainS), axis=1)
acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
print('Performance on Train Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
acsaSaveTr[-1], gmSaveTr[-1], accSaveTr[-1]=acsa, gm, acc
confMatSaveTr[-1]=confMat
tprSaveTr[-1]=tpr

pLabel=np.argmax(mlp.predict(testS), axis=1)
acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
print('Performance on Test Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
acsaSaveTs[-1], gmSaveTs[-1], accSaveTs[-1]=acsa, gm, acc
confMatSaveTs[-1]=confMat
tprSaveTs[-1]=tpr

mlp.save(savePath+'MLP_'+str(step)+fileEnd)

resSave=savePath+'Results'
np.savez(resSave, acsa=acsa, gm=gm, tpr=tpr, confMat=confMat, acc=acc)
recordSave=savePath+'Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr, acsaSaveTs=acsaSaveTs, gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs)

