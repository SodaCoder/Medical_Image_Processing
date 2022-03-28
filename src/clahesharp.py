from PIL import Image,ImageEnhance
import cv2
import numpy as np

filename=input("enter the filename:")
image = cv2.imread(filename,0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
cl1 = clahe.apply(image)
cv2.imwrite("cl.png",cl1)
image = Image.open("cl.png")
enhancer=ImageEnhance.Sharpness(image)
image=enhancer.enhance(4.0)
image = np.array(image)
cv2.imwrite("clahesharp.png",image)
