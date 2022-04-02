# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
#from keras.utils.np_utils import to_categorical
from scipy import linalg

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

# For selecting a GPU
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

genModel = keras.models.load_model('../DatasetBfinalPreprocessing/gamo_models_50000/GEN_50000_Model.h5')
gen_processed = []
for i1 in range(3):
    if i1 == 0:
        gen_processed.append(keras.models.load_model('../DatasetBfinalPreprocessing/gamo_models_50000/GenP_0_50000_Model.h5'))
    elif i1 == 1:
        gen_processed.append(keras.models.load_model('../DatasetBfinalPreprocessing/gamo_models_50000/GenP_1_50000_Model.h5'))
    elif i1 == 2:
        gen_processed.append(keras.models.load_model('../DatasetBfinalPreprocessing/gamo_models_50000/GenP_2_50000_Model.h5'))

storedGenFeatureCSVPath = '../DatasetBfinalPreprocessing/GenratedFeaturesWoLabel.csv'
storedOriginalCVSPath = '../DatasetBfinalPreprocessing/trainWLabel.csv'

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


def fileRead(fileName):
    dataTotal=np.loadtxt(fileName, delimiter=',')
    data=dataTotal[:, :-1]
    labels=dataTotal[:, -1]
    return data, labels

latDim = 128
c = 4
genFinalArr = []

sampPerClss = 1000
#sampPerClss = 3
labelArr = np.zeros((3 * sampPerClss, 1))
index = 0
print('Collecting 3000 generated data...')
for numSamp in tqdm(range(sampPerClss)):
    for i1 in range(3):
        testNoise=np.random.normal(0, 1, (1, latDim))
        testLabel=np.zeros((1, c))
        testLabel[:, i1] = 1
        labelArr[index] = i1 
        genFinal = genModel.predict([testNoise, testLabel])
        genFinal = gen_processed[i1].predict(genFinal)
        genFinalArr.append(genFinal)
        index = index + 1
        
genFinalArr = np.array(genFinalArr)

genFinalArr = genFinalArr.reshape(3 * sampPerClss, 2048)
pd.DataFrame(np.concatenate((genFinalArr, labelArr), axis=1)).to_csv(storedGenFeatureCSVPath, sep=',', header=None, index=None)
print(genFinalArr.shape)

originalData, origLabel = fileRead(storedOriginalCVSPath)
generatedData, _ = fileRead(storedGenFeatureCSVPath)

originalDataShort = np.empty((0, 2048))
print('Collecting 3000 Original data...')
for index in tqdm(range(len(origLabel))):
    if origLabel[index] == 2:
        continue
    else:
        originalDataShort = np.concatenate((originalDataShort, originalData[index].reshape(1, 2048)), axis = 0)

randArr = np.random.randint(originalDataShort.shape[0], size = 3 * sampPerClss)
originalData = originalDataShort[randArr, :]

mu1, sigma1, mu2, sigma2 = calc_mu_sigma(originalData, generatedData)
fd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
print('Frechet Distance is: ', fd)
