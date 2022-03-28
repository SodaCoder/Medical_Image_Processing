# A small code for the decision making of a single image applied to a pretrained model.

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import keras
from tensorflow.keras import utils
# We need not to create and train the model again
# No need to train the model, its a pre defined trained model
model = load_model("/content/trained_model.h5")
#model.compile(optimizer='adam', loss='b_crossentropy', metrics=['accuracy'])

# Testing the Normal Image
image = cv2.imread("/content/wong-0005.jpg")
# image = cv2.imread("covid19dataset/test/covid/nejmoa2001191_f3-PA.jpeg")    # 0
image = cv2.resize(image, (128, 128))
image = np.reshape(image, [1, 128, 128, 3])
classes = model.predict(image)
result=np.argmax(classes)
if result==0:
  print("COVID")
elif result==1:
  print("NORMAL")
else:
  print("PNEUMONIA")
