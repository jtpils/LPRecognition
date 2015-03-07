# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:24:28 2015

@author: ireti
"""

import cv2
import numpy as np 


def localizePlate(location):
    #read in image
#    imgBGR = cv2.imread('Caltech1999/image_0068.jpg',1)
    

    print 
    print    
    
    imgBGR = cv2.imread('Caltech1999/'+location,1)
    print 'Caltech1999/'+location
#    cv2.imshow('Car Picture', imgBGR)
#    cv2.waitKey(0)
    
    #convert to grey
    gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image,(5,5),0)
    
    # GET EDGES
    cannyedges = cv2.Canny(gray_image,100,200)
    #cv2.imwrite('edgedImageCanny.png',cannyedges)
    image = cannyedges 
    
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ## rank connected regions
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    print 'ready to draw contours'
    count = 0
    img = imgBGR.copy()
    for cnt in range(len(contours)):
        # approximate the contour
        
        x,y,w,h = cv2.boundingRect(contours[cnt])
        aspect_ratio = float(w)/h
        area = w*h
        if aspect_ratio > 1.2 and aspect_ratio < 4:
            if area > 500 and area < 10000:
                count = count + 1
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    print 'number of contours found: ', count
#    cv2.imshow('Connected Regions ', img)
#    cv2.waitKey(0)
    
    dirLocation = './Output/'
    outfile = dirLocation+ location[:-4] + '_output.png'
    print outfile
    cv2.imwrite(outfile, img )
#    cv2.imwrite('./Output/connectedRegionsArea.png',img )
    



#localizePlate('image_0068.jpg')

from os import listdir
from os.path import isfile, join
#mypath = '/home/ireti/Desktop/Winter 2015/CS 231a/Course Project/Codes/CS231A-Project/Caltech1999'
mypath = './Caltech1999'

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

print len(onlyfiles)
print onlyfiles[1]

for i in range(len(onlyfiles)):
    localizePlate(onlyfiles[i])

print 'Done'