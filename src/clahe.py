import cv2 
import numpy as np
import os

filename=input("enter the filename:")
image = cv2.imread(filename,0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
cl1 = clahe.apply(image)
filename=filename+"10.png"
cv2.imwrite(filename,cl1)
cv2.imshow("CLAHE",cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()