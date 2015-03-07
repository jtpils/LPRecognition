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
     
#    sobelx = cv2.Sobel(image,cv2.CV_8u,0,ksize=kernelSize)
#    sobely = cv2.Sobel(image,cv2.CV_8u,0,1,ksize=kernelSize)
    return (sobelx + sobely)





#read in image
imgBGR = cv2.imread('Caltech1999/image_0113.jpg',1)
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
cv2.imwrite('thresholded_image_Inverted.png',invertBinaryImage(threshImage, 255))


cv2.waitKey(0)
# find edges
cv2.imshow('Car Picture', findEdges(invertBinaryImage(threshImage, 255), kernelSize = 3))
print 'found edges of inverted image of adaptive thresholding only'
cv2.imwrite('edgedImage2.png',findEdges(threshImage, kernelSize = 3))
cv2.imwrite('edgedImage.png',findEdges(invertBinaryImage(threshImage, 255), kernelSize = 3))

cannyedges = cv2.Canny(threshImage,50,250)
cv2.imwrite('edgedImageCanny.png',cannyedges)

cannyedgesINV = cv2.Canny(invertBinaryImage(threshImage, 255),100,200)
cv2.imwrite('edgedImageCanny2.png',cannyedgesINV)

laplacianEdges =  cv2.Laplacian(threshImage,cv2.CV_8UC1)
cv2.imwrite('laplacianEdges.png',laplacianEdges)

laplacian_threshImage = cv2.adaptiveThreshold(laplacianEdges,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite('laplacian_threshImage.png',laplacian_threshImage)

laplacianEdgesINV = cv2.Laplacian(invertBinaryImage(threshImage, 255),cv2.CV_8UC1)
cv2.imwrite('laplacianEdgesINV.png',laplacianEdgesINV)

laplacian_threshImageINV = cv2.adaptiveThreshold(laplacianEdgesINV,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite('laplacian_threshImageINV.png',laplacian_threshImageINV )


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
cv2.imshow('Connected Regions', image)
#print 'changing integer format now'
#cv2.waitKey(0)
#abs_sobel64f = np.absolute(image)
#image = np.uint8(abs_sobel64f)
#cv2.imshow('Connected Regions', image)
#cv2.waitKey(0)

print type(image)

image = cannyedges #laplacian_threshImage #laplacianEdges #cannyedges

cv2.imshow('Connected Regions', image)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours( image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print 'ready to draw contours'
print type(contours)
print 'number of contours found: ', len(contours)

'''
image = imgBGR.copy()
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('Connected Regions', image)
cv2.imwrite('connectedRegions.png',image )

imageIterate = imgBGR
for i in range(len(contours)): 
    print i    
    cnt = contours[i]
    cv2.drawContours(imageIterate, [cnt], 0, (0,255,0), 3)
    cv2.imshow('Connected Regions Iterate', imageIterate)
    cv2.waitKey(0)
'''

## rank connected regions
contours = sorted(contours, key = cv2.contourArea, reverse = True)
print type(contours)

print len(contours)
contoursQUAD = []
img = imgBGR.copy()
for cnt in range(len(contours)):
    # approximate the contour
    perimeter = cv2.arcLength(contours[cnt], True)
    approx = cv2.approxPolyDP(contours[cnt], 0.1 * perimeter, True)
    
    
    
#    img = imgBGR.copy()
    x,y,w,h = cv2.boundingRect(contours[cnt])
#    box = cv2.boxPoints(x,y,w,h)
    aspect_ratio = float(w)/h
    area = w*h
    if aspect_ratio > 1.2 and aspect_ratio < 6:
        if area > 500 and area < 10000:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#            cv2.imshow('Connected Regions Iterate', img)
#            cv2.waitKey(0)


#    box=cv2.boxPoints(rect)
    
#    cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
#    cv2.imshow('Connected Regions Iterate', img)
#    cv2.waitKey(0)
    
#    rect = cv2.minAreaRect(cnt)
#    box = cv2.boxPoints(rect)
#    box = np.int0(box)
    
    print len(approx)
    if len(approx) == 4:
        x,y,w,h = cv2.boundingRect(contours[cnt])
        aspect_ratio = float(w)/h
        print 'aspect ratio: ', aspect_ratio
        
        if aspect_ratio > 1.2:
            area = cv2.contourArea(contours[cnt])
            print area
            if area > 500:
                contoursQUAD.append( contours[cnt])
#            print 'new contour'
#        print approx
#        cv2.waitKey(0)
        
#    if len(approx) != 4:
#        del contours[cnt]
#    if cnt == len(contours)-1:
#        break
#        contours.remove(cnt)



cv2.imshow('Connected Regions ', img)
cv2.waitKey(0)

cv2.imwrite('connectedRegionsArea.png',img )



contours = contoursQUAD
print 'The remaining contours are: ', len(contours)
#contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

image = imgBGR.copy()
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('Connected Regions', image)
cv2.imwrite('connectedRegionsQUAD.png',image )

cv2.destroyAllWindows()
imageSortedCOntour = imgBGR.copy()
# loop over our contours
count = 0
for cnt in contours:
    
    # approximate the contour
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)
    print len(approx)
    print approx
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    print 'aspect ratio: ', aspect_ratio
    area = cv2.contourArea(cnt)
    print 'area:  ', area
    
    
    
    count = count + 1
    print count
    
    imageSortedCOntour = imgBGR.copy()
    cv2.drawContours(imageSortedCOntour, [cnt], 0, (0,255,0), 3)
    cv2.imshow('Connected Regions Iterate', imageSortedCOntour)
    cv2.waitKey(0)
    
    
    
     
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
#    if len(approx) == 4:
#        screenCnt = approx
#    break
    

#cv2.imshow('Car Picture', findEdges(invertBinaryImage(image, 255), kernelSize = 3))
#cv2.waitKey(0)

#cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) → contours, hierarchy