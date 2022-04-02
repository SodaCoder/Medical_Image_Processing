import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras import metrics
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.np_utils import to_categorical
import dense_suppli as spp
import os
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Global Data
BATCH_SIZE = 128
EPOCHS = 200
EARLY_SOPPING_PATIENCE = 10
filePath = "./logs/MLP_resnet50_smote.h5"
allImagesPath = "./claheplussharpening/all_images"
trainPath = "./claheplussharpening/train.csv"
valPath = "./claheplussharpening/val.csv"
testPath = "./claheplussharpening/test.csv"
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
    mlpFinal = Dense(4, activation = "softmax")(x)
    mlp = Model(imIn, mlpFinal)
    mlp.summary()
    return mlp

# Extracting Features and Preparing the data
def createImageDataset(csvPath, allImagesPath):
    global base_model

    imageDf = pd.read_csv(csvPath, skiprows = [0])
    records = imageDf.to_records(index = False)
    result = list(records)
    imageFeatList = []
    labelList = []
    
    input = Input(shape=(299,299,3),name = 'image_input')
    x = base_model(input)
    x = Flatten()(x)
    model = Model(inputs=input, outputs=x)
    
    for index in tqdm(range(len(result))):
        imgName = result[index][0]
        label = int(result[index][1])
        
        image = cv2.imread(allImagesPath + "/" + imgName)
        if image.any() == None:
            break
        image = np.expand_dims(image, axis = 0)
        x = preprocess_input(image)
        features = model.predict(x)

        features_reduce = features.squeeze()
        imageFeatList.append(features_reduce)
        labelList.append(label)

    imageFeatList = np.array(imageFeatList)
    labelList = np.array(labelList)
    #print("1st Image Feature is : ", imageFeatList[0], " Length is: ", len(imageFeatList[0]), " Label is: ", labelList[0])
    return imageFeatList, labelList

print("Creating the Training Data...")
X_train, Y_train = createImageDataset(trainPath, allImagesPath)
print("Creating the Validation Data...")
X_val, Y_val = createImageDataset(valPath, allImagesPath)
print("Creating the Test Data...")
X_test, Y_test = createImageDataset(testPath, allImagesPath)

# sm = SMOTE(random_state = 42)
# X_train_smote, Y_train_smote = sm.fit_resample(X_train, Y_train)

labelTr, labelTs, c, pInClass, classMap = spp.relabel(Y_train, Y_val)
print("Relabeling is done...")
imbalancedCls, toBalance, imbClsNum, ir = spp.irFind(pInClass, c)
print("Imbalance Ratio is found...")
print("Now SMOTE started...")
X_train_smote, Y_train_smote, _ = spp.smote(X_train, labelTr, pInClass, toBalance, classMap)
print("SMOTE finished...")

# Early stopping if the validation loss doesn't decrease anymore
early_stopping = EarlyStopping(monitor = 'val_loss', patience = EARLY_SOPPING_PATIENCE, verbose = 2, mode = "min")
# We always keep the best model in case of early stopping
model_checkpoint = ModelCheckpoint(filepath = filePath, monitor = "val_loss", save_best_only = True, verbose = 2, mode = "min", save_weights_only = True)

steps_per_epoch = len(X_train_smote) // BATCH_SIZE if len(X_train_smote) > BATCH_SIZE else len(X_train_smote)
validation_steps = len(X_val) // BATCH_SIZE if len(X_val) > BATCH_SIZE else len(X_val)

datagen = ImageDataGenerator()
'''
# We will fine tune a model, pretained on imbalanced preprocessed covid-19 dataset
weightsPath = "./balacedmainclaheTOTALdataResnset.h5"
model = Sequential()
model.add(tf.keras.applications.ResNet50(pooling = "avg", include_top = False, weights= weightsPath))
model.add(Dense(4, activation = "softmax"))
#model.layers[0].trainable = False
'''
model = denseMlpCreate()
model.compile(optimizer = Adam(lr = 0.0001), loss = "mean_squared_error", metrics=["accuracy"])


# Reshaping SMOTEd training and validation set to have rank 4
# As, 2048 = 64 x 32
X_train_smote = X_train_smote.reshape(X_train_smote.shape[0], 64, 32, 1)
X_train_smote = (tf.image.grayscale_to_rgb(tf.constant(X_train_smote))).numpy()
X_val = X_val.reshape(X_val.shape[0], 64, 32, 1)
X_val = (tf.image.grayscale_to_rgb(tf.constant(X_val))).numpy()

Y_train_smote = to_categorical(Y_train_smote)
print("Training the model...")
model.fit_generator(
    generator = datagen.flow(X_train_smote, Y_train_smote, batch_size = BATCH_SIZE, shuffle = True),
    steps_per_epoch = steps_per_epoch,
    epochs = EPOCHS,
    validation_data = datagen.flow(X_val, Y_val, batch_size = BATCH_SIZE, shuffle = True),
    validation_steps = validation_steps,
    callbacks = [early_stopping, model_checkpoint],
    verbose = 2
)

# We reload the best epoch weight before keep going
model.load_weights(filePath)


# Reshaping the test data for evaluation
# As, 2048 = 64 x 32
X_test = X_test.reshape(X_test.shape[0], 64, 32, 1)
X_test = (tf.image.grayscale_to_rgb(tf.constant(X_test))).numpy()

print("Testing the model...")
Y_test = to_categorical(Y_test)
score = model.evaluate(X_test, Y_test, verbose = 2)
