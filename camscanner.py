import cv2
import numpy as np
import pytesseract
from PIL import Image

img = cv2.imread('red.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh=cv2.threshold(gray,60,255,1)
blur = cv2.medianBlur(thresh,1)



print(pytesseract.image_to_string(blur))

cv2.imshow('ImageWindow1', gray)
cv2.imshow('ImageWindow2', thresh)
cv2.imshow('ImageWindow3', blur)

cv2.waitKey(0)


