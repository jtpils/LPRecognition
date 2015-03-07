# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 18:49:59 2015

@author: ireti

"""

import cv2
import numpy as np 

# invert binary image, maxValue can be 1 or 255
def invertBinaryImage(image, maxValue):
    print np.max(image)
    print np.mean(image)
#    image = np.ones(image.shape, np.int8) * maxValue - image;
    image = ((image*0 ) + 1)* maxValue - image;
    return image
        
    
# use sobel edge detector to find horizontal and vertical edges
def findEdges(image, kernelSize = 5):
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=kernelSize)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=kernelSize)
    return (sobelx + sobely)





#read in image
imgBGR = cv2.imread('../Caltech1999/image_0002.jpg',1)
#print type(imgBGR)
cv2.imshow('Car Picture', imgBGR)
cv2.waitKey(0)

#convert to grey
gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png',gray_image)

cv2.imwrite('gray_imageBlurred.png',cv2.GaussianBlur(gray_image,(5,5),0))
gray_image = cv2.GaussianBlur(gray_image,(5,5),0)

cv2.imshow('Black and White',imgBGR)
cv2.waitKey(0)

## threshold before edge detection


#convert to binary image
threshImage = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imshow('Car Picture', threshImage)
threshImage = gray_image
print 'adaptive thresholding only'

''' useless
cv2.waitKey(0)
ret,threshImage = cv2.threshold(threshImage,127,255,cv2.THRESH_BINARY)
cv2.imshow('Car Picture', threshImage)
print 'ordinary hard thresholding of adaptive thresholding output'
'''


cv2.waitKey(0)
cv2.imshow('Car Picture', invertBinaryImage(threshImage, 255))
print 'inverted image of adaptive thresholding only'
cv2.imwrite('thresholded_image.png',invertBinaryImage(threshImage, 255))


cv2.waitKey(0)
# find edges
cv2.imshow('Car Picture', findEdges(invertBinaryImage(threshImage, 255), kernelSize = 3))
print 'found edges of inverted image of adaptive thresholding only'
cv2.imwrite('edgedImage2.png',findEdges(threshImage, kernelSize = 3))
cv2.imwrite('edgedImage.png',findEdges(invertBinaryImage(threshImage, 255), kernelSize = 3))

cannyedges = cv2.Canny(threshImage,100,200)
cv2.imwrite('edgedImageCanny.png',cannyedges)

#sobelx = cv2.Sobel(cannyedges,cv2.CV_64F,1,0,ksize=5) # x 
sobelycannyedges = cv2.Sobel(cannyedges,cv2.CV_64F,0,1,ksize=9) # y 
cv2.imwrite('sobelycannyedges.png',sobelycannyedges)

cannyedgesINV = cv2.Canny(invertBinaryImage(threshImage, 255),100,200)
cv2.imwrite('edgedImageCanny2.png',cannyedgesINV)

#blur = cv2.GaussianBlur(img,(5,5),0)
            
cv2.waitKey(0) 
th3 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imshow('Car Picture', th3)



cv2.waitKey(0)
cv2.imshow('Car Picture', invertBinaryImage(th3, 255))

cv2.waitKey(0)
cv2.imshow('Car Picture', findEdges(invertBinaryImage(th3, 255), kernelSize = 3))




##  do edge detection before thresholding






## Connected COmponents
print 'Starting connected components part'
from cv2 import *

image = findEdges(invertBinaryImage(th3, 255), kernelSize = 3)
print type(image)

cv2.waitKey(0)
contours, hierarchy = cv2.findContours( image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('Car Picture', findEdges(invertBinaryImage(image, 255), kernelSize = 3))

cv2.waitKey(0)

#cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) → contours, hierarchy