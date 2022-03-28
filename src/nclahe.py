import cv2 
import numpy as np
import os

filename=input("enter the filename:")
image = cv2.imread(filename,0)

image = cv2.normalize(image,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
cl1 = clahe.apply(image)
cv2.imshow("N-CLAHE",cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()
