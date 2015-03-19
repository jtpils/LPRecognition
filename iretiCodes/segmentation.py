# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:21:14 2015

@author: ireti
"""


# Load data

import cv2
import numpy as np

from os import listdir
from os.path import isfile, join


def segmentPlateConnectedComponent(imgBGR):
    gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    plate_area = gray_image.shape[0] * gray_image.shape[1]
    cv2.imshow('Black and White',gray_image)
    
    threshImage = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    
    threshImage = 255 - threshImage
    cv2.imshow('Plate Binarized Picture', threshImage)
    
    contours, hierarchy = cv2.findContours(threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ## rank connected regions
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    for cnt in range(len(contours)):
        ## approximate the contour
        perimeter = cv2.arcLength(contours[cnt], True)
        approx = cv2.approxPolyDP(contours[cnt], 0.1 * perimeter, True)    
        
        x,y,w,h = cv2.boundingRect(contours[cnt])
        area = w*h
        
        #set threshold here
        aspect_ratio = float(w)/h
        area = w*h
        if float(w)/h < 6 and float(h)/w < 6:
            if area > 0.01*plate_area and area < .4*plate_area:
                cv2.rectangle(imgBGR,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow('Connected Regions Iterate', imgBGR)
                cv2.waitKey(0)
                
from pylab import *                
def segmentPlateHistogram(imgBGR):
    gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    plate_area = gray_image.shape[0] * gray_image.shape[1]
    cv2.imshow('Black and White',gray_image)
    
    threshImage = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    
    threshImage = 255 - threshImage
    cv2.imshow('Plate Binarized Picture', threshImage)
    
    print threshImage.shape
    histData = sum(threshImage, axis=0)
       
    
    print histData
    plot(histData)
    show()
    
    cv2.waitKey(0)
    

    

plateDataLocation = './plateData/'
path = plateDataLocation

files = [ f for f in listdir(path) if isfile(join(path,f)) ]
histogramDatabase = []
for i in range(len(files)):
    location = path     #'./data/'
    print files[i]
    print location+files[i]
    imgBGR = cv2.imread(location+files[i],1)
    segmentPlateHistogram(imgBGR)
    
    segmentPlateConnectedComponent(imgBGR)
        
#        cv2.drawContours(imgBGR, contours[cnt], 0, (0,255,0), 3)
        
        
    

    cv2.waitKey(0)
#    histogram = hog(gray_image)    
#    histogramDatabase.append(histogram)
#return np.asarray(histogramDatabase)
    
    
    
