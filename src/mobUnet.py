import cv2
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    print(x.shape)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def modelMobnet():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    #encoder = ResNet50(input_tensor=inputs, weights="imagenet", include_top=False)
    encoder.summary()
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    #encoder_output = encoder.get_layer("block_13_expand_relu").output
    encoder_output=encoder.get_layer('block_13_expand_relu').output
    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        print(x_skip.shape)
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    modelNew = Model(inputs, x)
    return modelNew

import tensorflow as tf

trainedModel=modelMobnet()
# Restore the weights
trainedModel.load_weights('model_checkpoint')

path="test/CHNCXR_0560_1.png"
x = read_image(path)
y_pred = trainedModel.predict(np.expand_dims(x, axis=0))[0] > 0.5
h, w, _ = x.shape
white_line = np.ones((h, 10, 3))
maskimg=(x*mask_parse(y_pred))
plt.imsave("maskimg.png", maskimg, cmap="gray")
all_images = [x, white_line, mask_parse(y_pred), maskimg]
image = np.concatenate(all_images, axis=1)
fig = plt.figure(figsize=(12, 12))
a = fig.add_subplot(1, 1, 1)
plt.imsave("Mob_Unet.png", image, cmap="gray")