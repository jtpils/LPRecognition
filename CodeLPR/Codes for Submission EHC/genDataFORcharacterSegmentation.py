# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:44:36 2015

@author: ireti
"""

# Load data

import cv2
import numpy as np
#from statistics import *

from os import listdir
from os.path import isfile, join, isdir



'''
#### Generate dataset for training character recognition data for the SVM
####    classifier using images of characters of different MS Word fonts
'''
imgBGR = cv2.imread('LETTERDATASET.PNG',1)

gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
threshImage = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
#cv2.imshow('Thresh Data',threshImage)
threshImage = 255 - threshImage
#cv2.imshow('Plate Binarized Picture', threshImage)
#cv2.waitKey(0)

contours, hierarchy = cv2.findContours(threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
imgOrig = imgBGR.copy()
for cnt in range(len(contours)):    
    x,y,w,h = cv2.boundingRect(contours[cnt])
    box = imgOrig[y:y+h, x:x+w]
    
    fileName = './letterSegments/'+str(cnt)+'.png'                
    cv2.imwrite(fileName, box)
    
print 'done!'
cv2.waitKey(0)
