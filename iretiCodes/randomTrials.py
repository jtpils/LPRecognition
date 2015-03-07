# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 16:28:54 2015

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
imgBGR = cv2.imread('Caltech1999/image_0008.jpg',1)
cv2.imshow('Car Picture', imgBGR)
cv2.waitKey(0)

#convert to grey
gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png',gray_image)

#cv2.imwrite('gray_imageBlurred.png',cv2.GaussianBlur(gray_image,(5,5),0))
gray_image = cv2.GaussianBlur(gray_image,(5,5),0)

#cv2.imshow('Black and White',gray_image)
#cv2.waitKey(0)

## threshold before edge detection


cannyedges = cv2.Canny(gray_image,100,200)
cv2.imwrite('edgedImageCanny.png',cannyedges)

#cannyedgesINV = cv2.Canny(invertBinaryImage(gray_image, 255),100,200)
#cv2.imwrite('edgedImageCanny2.png',cannyedgesINV)

laplacianEdges =  cv2.Laplacian(gray_image,cv2.CV_8UC1)
cv2.imwrite('laplacianEdges.png',laplacianEdges)

cannyLaplaceEdges = cv2.Canny(laplacianEdges,100,200)
cv2.imwrite('cannyLaplaceEdges.png',cannyLaplaceEdges)

laplacianEdgesBlurred = cv2.GaussianBlur(laplacianEdges,(3,3),0)
laplacianEdgesBlurred = laplacianEdges
cannyLaplaceBlurrededges = cv2.Canny(laplacianEdgesBlurred,100,200)
cv2.imwrite('cannyLaplaceBlurrededges.png', cannyLaplaceBlurrededges)


laplacianEdgesINV = cv2.Laplacian(invertBinaryImage(gray_image, 255),cv2.CV_8UC1)
cv2.imwrite('laplacianEdgesINV.png',laplacianEdgesINV)

laplacianEdgesAGG = laplacianEdges + laplacianEdgesINV
cv2.imwrite('laplacianEdgesAGG.png',laplacianEdgesAGG)

cannyLaplaceEdgesAGG = cv2.Canny(laplacianEdgesAGG,100,200)
cv2.imwrite('cannyLaplaceEdgesAGG.png',cannyLaplaceEdgesAGG)

laplacianEdgesAGGBlurred = cv2.GaussianBlur(laplacianEdgesAGG,(3,3),0)
cannyLaplaceBlurrededgesAGG = cv2.Canny(laplacianEdgesAGGBlurred,100,200)
cv2.imwrite('cannyLaplaceBlurrededgesAGG.png', cannyLaplaceBlurrededgesAGG)



#
#laplacian_threshImage = cv2.adaptiveThreshold(laplacianEdges,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY,11,2)
#cv2.imwrite('laplacian_threshImage.png',laplacian_threshImage)

#laplacianEdgesINV = cv2.Laplacian(invertBinaryImage(gray_image, 255),cv2.CV_8UC1)
#cv2.imwrite('laplacianEdgesINV.png',laplacianEdgesINV)
#
#laplacian_threshImageINV = cv2.adaptiveThreshold(laplacianEdgesINV,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY,11,2)
#cv2.imwrite('laplacian_threshImageINV.png',laplacian_threshImageINV )