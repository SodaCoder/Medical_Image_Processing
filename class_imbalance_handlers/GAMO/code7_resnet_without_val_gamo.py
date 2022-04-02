# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import os
import numpy as np
import code7_dense_suppli as spp
import code7_dense_net as nt
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from plot_confusion_mat import plot_confusion_mat
import os
import tensorflow as tf
import random
#%% defintion
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(0)
   os.environ['TF_DETERMINISTIC_OPS'] = '1'
   tf.random.set_seed(0)
   np.random.seed(0)
   random.seed(0)
   
def rotate(l, n):
    return l[n:] + l[:n]

def get_data_label(data_dir,text_file): 
    x = []
    y = []
    mapping={
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
    }
    line_content= text_file
    for i in range(0,len(line_content)):
        file_name = line_content[i].split(" ")[1]
        x.append(np.load(data_dir + file_name+'.npy'))
        y.append(line_content[i].split(" ")[2].rstrip())
    y =list(map(mapping.get, y))

    return np.asarray(x,dtype = np.float32), to_categorical(np.asarray(y,dtype = np.int32))

#%% parameters

adamOpt=Adam(0.0002, 0.9)
latDim, modelSamplePd, resSamplePd=128, 50, 2
batchSize, max_step=32, 50000

#%% paths

fileStart='C:/Users/susmi/covidx4_final/output/resnet_output_without_val/gamo_mlp/'
fileEnd, savePath='_Model.h5', fileStart+'/'
intermediate_datafolder = 'C:/Users/susmi/covidx4_final/output/resnet_output_without_val/intermediate/'
train_txt = 'C:/Users/susmi/covidx4_final/pyfiles/train_txt.txt'
val_txt = 'C:/Users/susmi/covidx4_final/pyfiles/val_txt.txt'
test_txt = 'C:/Users/susmi/covidx4_final/pyfiles/test_COVIDx4.txt'
#%% load data
reset_random_seeds()

with open(test_txt, 'r') as fr:
     test_files = fr.readlines()
with open(val_txt, 'r') as fr:
     val_files = fr.readlines()
with open(train_txt, 'r') as fr:
     train_files = fr.readlines()
         
x_train, y_train = get_data_label(intermediate_datafolder,train_files+val_files)
x_test, y_test = get_data_label(intermediate_datafolder,test_files)

trainS = np.squeeze(x_train)
labelTr = np.argmax(y_train,axis = 1)
testS = np.squeeze(x_test)
labelTs = np.argmax(np.squeeze(y_test),axis = 1)

n, m=trainS.shape[0], testS.shape[0]

labelTr, labelTs, c, pInClass, _=spp.relabel(labelTr, labelTs)
imbalancedCls, toBalance, imbClsNum, ir=spp.irFind(pInClass, c)

labelsCat=to_categorical(labelTr)

shuffleIndex=np.random.choice(np.arange(n), size=(n,), replace=False)
trainS=trainS[shuffleIndex]
labelTr=labelTr[shuffleIndex]
labelsCat=labelsCat[shuffleIndex]
classMap=list()
for i in range(c):
    classMap.append(np.where(labelTr==i)[0])
#%%

# model initialization
mlp=nt.denseMlpCreate()
mlp.compile(loss='mean_squared_error', optimizer=adamOpt)
mlp.trainable=False

dis=nt.denseDisCreate()
dis.compile(loss='mean_squared_error', optimizer=adamOpt)
dis.trainable=False

gen=nt.denseGamoGenCreate(latDim)

gen_processed, genP_mlp, genP_dis=list(), list(), list()
for i in range(imbClsNum):
    dataMinor=trainS[classMap[i], :]
    numMinor=dataMinor.shape[0]
    gen_processed.append(nt.denseGenProcessCreate(numMinor, dataMinor))

    ip1=Input(shape=(latDim,))
    ip2=Input(shape=(c,))
    op1=gen([ip1, ip2])
    op2=gen_processed[i](op1)
    op3=mlp(op2)
    genP_mlp.append(Model(inputs=[ip1, ip2], outputs=op3))
    genP_mlp[i].compile(loss='mean_squared_error', optimizer=adamOpt)

    ip1=Input(shape=(latDim,))
    ip2=Input(shape=(c,))
    ip3=Input(shape=(c,))
    op1=gen([ip1, ip2])
    op2=gen_processed[i](op1)
    op3=dis([op2, ip3])
    genP_dis.append(Model(inputs=[ip1, ip2, ip3], outputs=op3))
    genP_dis[i].compile(loss='mean_squared_error', optimizer=adamOpt)

# for record saving
batchDiv, numBatches, bSStore=spp.batchDivision(n, batchSize)
genClassPoints=int(np.ceil(batchSize/c))

if not os.path.exists(fileStart):
    os.makedirs(fileStart)
picPath=savePath+'Pictures'
if not os.path.exists(picPath):
    os.makedirs(picPath)

iter=np.int(np.ceil(max_step/resSamplePd)+1)
acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))

# training
max_acsa =  -float('Inf')
step=0
while step<max_step:
    for j in range(numBatches):
        print(j)
        x1, x2=batchDiv[j, 0], batchDiv[j+1, 0]
        validR=np.ones((bSStore[j, 0],1))-np.random.uniform(0,0.1, size=(bSStore[j, 0], 1))
        mlp.train_on_batch(trainS[x1:x2], labelsCat[x1:x2])
        dis.train_on_batch([trainS[x1:x2], labelsCat[x1:x2]], validR)

        invalid=np.zeros((bSStore[j, 0], 1))+np.random.uniform(0, 0.1, size=(bSStore[j, 0], 1))
        randNoise=np.random.normal(0, 1, (bSStore[j, 0], latDim))
        fakeLabel=spp.randomLabelGen(toBalance, bSStore[j, 0], c)
        rLPerClass=spp.rearrange(fakeLabel, imbClsNum)
        fakePoints=np.zeros((bSStore[j, 0], 1024))
        genFinal=gen.predict([randNoise, fakeLabel])
        for i1 in range(imbClsNum):
            if rLPerClass[i1].shape[0]!=0:
                temp=genFinal[rLPerClass[i1]]
                fakePoints[rLPerClass[i1]]=gen_processed[i1].predict(temp)

        mlp.train_on_batch(fakePoints, fakeLabel)
        dis.train_on_batch([fakePoints, fakeLabel], invalid)
        #print(fakePoints)
        for i1 in range(imbClsNum):
            validA=np.ones((genClassPoints, 1))
            randomLabel=np.zeros((genClassPoints, c))
            randomLabel[:, i1]=1
            randNoise=np.random.normal(0, 1, (genClassPoints, latDim))
            oppositeLabel=np.ones((genClassPoints, c))-randomLabel
            genP_mlp[i1].train_on_batch([randNoise, randomLabel], oppositeLabel)
            genP_dis[i1].train_on_batch([randNoise, randomLabel, randomLabel], validA)

        if step%resSamplePd==0:
            saveStep=int(step//resSamplePd)

            pLabel=np.argmax(mlp.predict(trainS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
            print('Train: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTr[saveStep], gmSaveTr[saveStep], accSaveTr[saveStep]=acsa, gm, acc
            confMatSaveTr[saveStep]=confMat
            tprSaveTr[saveStep]=tpr
            acsa_train = acsa
            
            pLabel=np.argmax(mlp.predict(testS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
            print('Test: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTs[saveStep], gmSaveTs[saveStep], accSaveTs[saveStep]=acsa, gm, acc
            confMatSaveTs[saveStep]=confMat
            tprSaveTs[saveStep]=tpr

            # for i1 in range(imbClsNum):
            #     testNoise=np.random.normal(0, 1, (3, latDim))
            #     testLabel=np.zeros((3, c))
            #     testLabel[:, i1]=1
               # genFinal=gen.predict([testNoise, testLabel])
                #genImages=gen_processed[i1].predict(genFinal)
                #genImages=np.reshape(genImages, (3, 32, 32))
               # for i2 in range(3):
                #    img=image.array_to_img(np.expand_dims(genImages[i2], axis=-1), scale=True)
                #    axs[i1,i2].imshow(img, cmap='gray')
                 #   axs[i1,i2].axis('off')
            #plt.show()
            #plt.pause(5)

            #figFileName=picPath+'/'+fileStart+'_'+str(step)+'.png'
            #plt.savefig(figFileName, bbox_inches='tight')

        if step%modelSamplePd==0 and step!=0:
            if np.average(acsa_train) >= max_acsa:
                max_acsa = acsa_train
                saved_step = step
                print('saved at step',step)
                direcPath=savePath+'gamo_models'
                if not os.path.exists(direcPath):
                    os.makedirs(direcPath)
                gen.save(direcPath+'/GEN_'+fileEnd)
                mlp.save(direcPath+'/MLP_'+fileEnd)
                dis.save(direcPath+'/DIS_'+fileEnd)
                for i in range(imbClsNum):
                    gen_processed[i].save(direcPath+'/GenP_'+str(i)+'_'+fileEnd)

        step=step+2
        if step>=max_step: break

#figFileName=picPath+'/'+fileStart+'_'+str(step)+'.png'
#plt.savefig(figFileName, bbox_inches='tight')

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

direcPath=savePath+'gamo_models_'+str(step)
if not os.path.exists(direcPath):
    os.makedirs(direcPath)
gen.save(direcPath+'/GEN_'+str(step)+fileEnd)
mlp.save(direcPath+'/MLP_'+str(step)+fileEnd)
dis.save(direcPath+'/DIS_'+str(step)+fileEnd)
for i in range(imbClsNum):
    gen_processed[i].save(direcPath+'/GenP_'+str(i)+'_'+str(step)+fileEnd)

resSave=savePath+'Results'
np.savez(resSave, acsa=acsa, gm=gm, tpr=tpr, confMat=confMat, acc=acc)
recordSave=savePath+'Record' 
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr, acsaSaveTs=acsaSaveTs, gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs)
#%%
import matplotlib.pyplot as plt
a = np.load(fileStart+'/Record.npz')
b = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
c = b['data1tprSaveTr']
axes = plt.gca()
plt.plot(c)
axes.set_xlim([-100,5000])
axes.set_ylim([.7,1.1])
plt.legend(['COVVID-19', 'Peumonia','Normal'], loc='lower right')
plt.title('train tpr')
plt.ylabel('tpr')
plt.xlabel('steps')
#%%
import matplotlib.pyplot as plt
a = np.load(fileStart+'Record.npz')
b = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
c = b['data1acsaSaveTr']
d = b['data1acsaSaveTs']

axes = plt.gca()
plt.plot(c)
plt.plot(d)
#axes.set_xlim([-100,10000])
axes.set_ylim([.5,1.1])
plt.legend(['Train', 'Test'], loc='lower right')
plt.title('model acsa')
plt.ylabel('acsa')
plt.xlabel('steps')
 #%%
import numpy as np
a = np.load(fileStart+'Record.npz')
b = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
c = b['data1acsaSaveTr']
#e = np.average(c[:,0:2],axis = 1,weights = [.5,.5])
e = c#[:,0]

#indices = [i for i, x in enumerate(e) if x == max(e)]
index = list(e).index(max(e))
#index = 17400
matrix = b['data1confMatSaveTs'][index]
matrix = np.flip(matrix)
matrix = matrix.astype('float')
class_names = ['Normal', 'Pneumonia', 'COVID-19']
title = "Confusion Matrix for after GAMO"
plot_confusion_mat(cm=matrix, normalize = False, target_names = class_names, title = title)

