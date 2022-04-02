#For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import sys, os
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
#from keras.utils.np_utils import to_categorical
from scipy import linalg
#from keras.preprocessing import image

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

# For selecting a GPU
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def calc_mu_sigma(real_feats, generated_feats):

    mu1=np.mean(real_feats, axis=0)
    mu2=np.mean(generated_feats, axis=0)

    sigma1=np.cov(real_feats, rowvar=False)
    sigma2=np.cov(generated_feats, rowvar=False)

    return mu1, sigma1, mu2, sigma2
    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)+np.trace(sigma2)-(2*tr_covmean))


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
    acsa=np.mean(tpr)
    gm=np.prod(tpr)**(1/confMat.shape[0])
    acc=np.sum(tp)/np.sum(nc)
    return acsa, gm, tpr, confMat, acc

def smote(trainS, trainL, pInClass, toBalance, classMap, numNeighbor=5):
    n, d=trainS.shape[0], trainS.shape[1]
    trainSCopy, trainLCopy=np.copy(trainS), np.copy(trainL)
    checkImages=np.zeros((3*toBalance.shape[0], trainS.shape[1]))
    
    generatedSamples = np.empty([0, d])
    generatedSamplesLabel = np.empty([0, 1])
    
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
        
        labelsNew = np.array(labelsNew)
        labelsNew = labelsNew.reshape(labelsNew.shape[0], 1)
        generatedSamples = np.concatenate((generatedSamples, newPoints), axis = 0)
        generatedSamplesLabel = np.concatenate((generatedSamplesLabel, labelsNew), axis = 0)
        
        checkImages[(i*3):((i+1)*3), :]=newPoints[0:3, :]
    
    #generatedSamples = np.array(generatedSamples)
    #print('Generated Samples Shape: ', generatedSamples.shape)
    #print(generatedSamples)
    generatedSamples = generatedSamples.reshape(generatedSamples.shape[0], d)    
    #generatedSamplesLabel = np.array(generatedSamplesLabel)
    #print('Generated Samples Labels Shape: ', generatedSamplesLabel.shape)
    generatedSamplesLabel = generatedSamplesLabel.reshape(generatedSamplesLabel.shape[0], 1)
      
    return trainSCopy, trainLCopy, checkImages, generatedSamples, generatedSamplesLabel
    

fileName=['../DatasetBfinalPreprocessing/trainWLabel.csv', '../DatasetBfinalPreprocessing/testWLabel.csv']
trainS, labelTr = fileRead(fileName[0])
testS, labelTs = fileRead(fileName[1])

n, m=trainS.shape[0], testS.shape[0]

labelTr, labelTs, c, pInClass, classMap = relabel(labelTr, labelTs)
imbalancedCls, toBalance, imbClsNum, ir= irFind(pInClass, c)

fdList = []
ind_range = 10
for index in range(ind_range):
    print('SMOTE started...')
    trainS_mlp, labelTr_mlp, checkImages, generatedSamples, generateSamplesLabel = smote(trainS, labelTr, pInClass, toBalance, classMap)
    print('SMOTE finished...')

    mu1, sigma1, mu2, sigma2 = calc_mu_sigma(trainS, generatedSamples)
    fdList.append(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))
    print('Frechet Distance is: ', fdList[index])
    
fdList = np.array(fdList)
fdList = fdList.reshape(fdList.shape[0], 1)

finalFDMean = np.mean(fdList)
finalFDStd = np.cov(fdList, rowvar=False)

print('final FD: Mean = %f, Std = %f' %(finalFDMean, finalFDStd))
#final FD: Mean = 8.364541, Std = 0.000746
