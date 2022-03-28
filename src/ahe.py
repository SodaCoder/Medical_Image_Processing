import cv2 
import numpy as np
import os

filename=input("enter the filename:")
image = cv2.imread(filename,0)
equ = cv2.equalizeHist(image)
cv2.imshow("AHE",equ)
cv2.waitKey(0)
cv2.destroyAllWindows()
