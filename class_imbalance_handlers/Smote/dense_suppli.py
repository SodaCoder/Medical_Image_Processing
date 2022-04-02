# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from keras.utils.np_utils import to_categorical

def relabel(labelTr, labelTs):
    unqLab, pInClass=np.unique(labelTr, return_counts=True)
    sortedUnqLab=np.argsort(pInClass, kind='mergesort')
    c=sortedUnqLab.shape[0]
    labelsNewTr=np.zeros((labelTr.shape[0],))-1
    labelsNewTs=np.zeros((labelTs.shape[0],))-1
    pInClass=np.sort(pInClass)
    classMap=list()
    for i in range(c):
        labelsNewTr[labelTr==unqLab[sortedUnqLab[i]]]=i
        labelsNewTs[labelTs==unqLab[sortedUnqLab[i]]]=i
        classMap.append(np.where(labelsNewTr==i)[0])
    return labelsNewTr, labelsNewTs, c, pInClass, classMap

def irFind(pInClass, c, irIgnore=1):
    ir=pInClass[-1]/pInClass
    imbalancedCls=np.arange(c)[ir>irIgnore]
    toBalance=np.subtract(pInClass[-1], pInClass[imbalancedCls])
    imbClsNum=toBalance.shape[0]
    if imbClsNum==0: sys.exit('No imbalanced classes found, exiting ...')
    return imbalancedCls, toBalance, imbClsNum, ir

def fileRead(fileName):
    dataTotal=np.loadtxt(fileName, delimiter=',')
    data=dataTotal[:, :-1]
    labels=dataTotal[:, -1]
    return data, labels

def indices(pLabel, tLabel):
    confMat=confusion_matrix(tLabel, pLabel)
    nc=np.sum(confMat, axis=1)
    tp=np.diagonal(confMat)
    tpr=tp/nc
    ppv = tp/(np.sum(confMat, axis = 0))
    
    acsa=np.mean(tpr)
    gm=np.prod(tpr)**(1/confMat.shape[0])
    acc=np.sum(tp)/np.sum(nc)
    return acsa, gm, tpr, confMat, acc, ppv

def smote(trainS, trainL, pInClass, toBalance, classMap, numNeighbor=5):
    n, d=trainS.shape[0], trainS.shape[1]
    trainSCopy, trainLCopy=np.copy(trainS), np.copy(trainL)
    checkImages=np.zeros((3*toBalance.shape[0], trainS.shape[1]))
    for i in range(toBalance.shape[0]):
        newPoints=np.zeros((toBalance[i], d))
        indices=np.random.randint(0, pInClass[i], (toBalance[i],))
        labelsNew=np.ones((toBalance[i],))*i
        for j in range(toBalance[i]):
            point=np.expand_dims(trainS[classMap[i][indices[j]], :], axis=0)
            distance=cdist(point, trainS[classMap[i], :])
            distInd=np.argsort(distance, axis=1)
            distInd=distInd[0, 1:1+numNeighbor]
            neighbor=np.random.randint(0, numNeighbor)
            neighborPoint=trainS[classMap[i][distInd[neighbor]], :]
            alpha=np.random.rand()
            newPoints[j, :]=(alpha*point)+((1-alpha)*neighborPoint)
        trainSCopy=np.vstack((trainSCopy, newPoints))
        trainLCopy=np.hstack((trainLCopy, labelsNew))
        checkImages[(i*3):((i+1)*3), :]=newPoints[0:3, :]
    return trainSCopy, trainLCopy, checkImages

def randomLabelGen(toBalance, batchSize, c):
    cumProb=np.cumsum(toBalance/np.sum(toBalance))
    bins=np.insert(cumProb, 0, 0)
    randomValue=np.random.rand(batchSize,)
    randLabel=np.digitize(randomValue, bins)-1
    randLabel_cat=to_categorical(randLabel)
    labelPadding=np.zeros((batchSize, c-randLabel_cat.shape[1]))
    randLabel_cat=np.hstack((randLabel_cat, labelPadding))
    return randLabel_cat

def batchDivision(n, batchSize):
    numBatches, residual=int(np.ceil(n/batchSize)), int(n%batchSize)
    if residual==0:
        residual=batchSize
    batchDiv=np.zeros((numBatches+1,1), dtype='int64')
    batchSizeStore=np.ones((numBatches, 1), dtype='int64')
    batchSizeStore[0:-1, 0]=batchSize
    batchSizeStore[-1, 0]=residual
    for i in range(numBatches):
        batchDiv[i]=i*batchSize
    batchDiv[numBatches]=batchDiv[numBatches-1]+residual
    return batchDiv, numBatches, batchSizeStore

def rearrange(labelsCat, numImbCls):
    labels=np.argmax(labelsCat, axis=1)
    arrangeMap=list()
    for i in range(numImbCls):
        arrangeMap.append(np.where(labels==i)[0])
    return arrangeMap
