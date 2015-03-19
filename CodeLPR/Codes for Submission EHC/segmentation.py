# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:21:14 2015

@author: ireti
"""

''' This code takes as input a given licese plate and segments the characters in it 
The output images of the segmented character is saved into an output folder.
'''

import cv2
import numpy as np

from os import listdir
from os.path import isfile, join


''' This function takes as input the image of the plate and the output filename
The image is binarize and connected components are extracted.
The constraints of aspect ratio, area and common midpoints are then used to 
pick the actual characters from the candidate regions.
'''
def segmentPlateConnectedComponent(imgBGR, filename):
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
    
    imgOrig = imgBGR.copy()
    plateWidth = imgOrig.shape[1]
    plateHeight = imgOrig.shape[0]
    
    print 'plateWidth', plateWidth
    print 'plateHeight', plateHeight
    
    midpointHeight = []
    heights = []
    candidateLocation = []
    
    
    for cnt in range(len(contours)):
        ## approximate the contour
        perimeter = cv2.arcLength(contours[cnt], True)
        approx = cv2.approxPolyDP(contours[cnt], 0.1 * perimeter, True)    
        
        x,y,w,h = cv2.boundingRect(contours[cnt])
        area = w*h
        
        ## use constraints on area and aspect ratio to pick false positives
        #set threshold here
        aspect_ratio = float(w)/h
        area = w*h
        if float(w)/h < 6 and float(h)/w < 6:
            if area > 0.01*plate_area and area < .4*plate_area:
                midpointHeight.append(y+(h/2))
                heights.append(h)
                candidateLocation.append([x,y,w,h])         #store location of posible candidates
    
    # use priors about vertical midpoints to eliminate false positives
    meanMid = mean(midpointHeight)
    stdMid = std(midpointHeight)    
    meanHeight = mean(heights)
    stdHeight = std(heights)
    
    print midpointHeight
    print meanMid
    print stdMid
    
    # the midpoint of true candidates would be within one standard deviation of the midpoints of majority of the candidates
    for points in range(len(midpointHeight)):
        if(midpointHeight[points] > meanMid-stdMid and midpointHeight[points] < meanMid+stdMid and
            heights[points] > meanHeight-stdHeight and heights[points] < meanHeight+stdHeight):
            x,y,w,h = candidateLocation[points]
            box = imgOrig[y:y+h, x:x+w]             
            fileName = './letterSegments/'+filename[:-4]+str(points)+'.png'                
            cv2.imwrite(fileName, box)
            
            cv2.rectangle(imgBGR,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.imshow('Connected Regions Iterate', imgBGR)
            cv2.waitKey(0)
                
    dirLocation = './SegmentationOutput/'
    outfile = dirLocation+ filename[:-4] + '_output.png'
    print outfile
    cv2.imwrite(outfile, imgBGR)

    
########################################################
#### Main program
#### program starts running from here
    
    
plateDataLocation = './plateData/'
path = plateDataLocation
    
files = [ f for f in listdir(path) if isfile(join(path,f)) ]
histogramDatabase = []
for i in range(len(files)):
    location = path     #'./data/'
    print files[i]
    print location+files[i]
    imgBGR = cv2.imread(location+files[i],1)    
    segmentPlateConnectedComponent(imgBGR, files[i])
    print 'Character Segmementation done..'
    cv2.waitKey(0)
    
