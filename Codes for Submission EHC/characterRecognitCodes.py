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

##########################################
## This fuction extraction the feature sets by applying the histogram of gradient 
## method to the edges extracted using sobel edge detector
## Also integrates the histogram of gradient to pixel values of image.
################################################
def hog(img):
    mainFeature = genHist(img)
    bin_n = 16 # Number of bins
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))            #converts angle to range from 0 to bin_n

    # Divide to 4 sub-squares
    xdivider = img.shape[0]/2
    ydivider = img.shape[1]/2
        
    bin_cells = bins[:xdivider,:ydivider], bins[xdivider:,:ydivider], bins[:xdivider,ydivider:], bins[xdivider:,ydivider:]
    mag_cells = mag[:xdivider,:ydivider], mag[xdivider:,:ydivider], mag[:xdivider,ydivider:], mag[xdivider:,ydivider:]
    
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]

    histsNormalized = [h/sum(h) for h in hists]
    hists = histsNormalized    
    
    
    level1HOG = np.hstack(hists)
    level2HOG = hogMoreFeatures(img)
       

    print level1HOG.shape
    hist = np.concatenate((mainFeature,level1HOG), axis=0)
    
    mainFeatureGX = genHist(gx,3)    
    mainFeatureGY = genHist(gy,3)    
    hist = np.concatenate((hist,mainFeatureGX), axis=0)
    hist = np.concatenate((hist,mainFeatureGY), axis=0)
    
    mainFeatureGX = genHist(gx,4)    
    mainFeatureGY = genHist(gy,4)    
    hist = np.concatenate((hist,mainFeatureGX), axis=0)
    hist = np.concatenate((hist,mainFeatureGY), axis=0)
    
    return hist
    
    
##########################################
## This fuction extraction the feature sets by applying the histogram of gradient 
## method to the edges to pixel values of image.
################################################
def genHist(img, divisions = 2):
    print 'starting method genHist(img)'
    width = img.shape[0]
    height = img.shape[1]
    
    
    width_step = int(np.round(width/divisions))
    height_step = int(np.round(height/divisions))
    
    histogram = []
    
    for i in range(divisions):
        for j in range(divisions):
            startW = i*width_step
            startH = i*height_step
            window = img[startW: np.min([startW+width_step, width-1]), startH: np.min([startH+height_step, height])]
            
            histogram.append(sum(sum(window==0))/float(width_step*width_step))
            
    return  np.asarray(histogram)


##########################################
## This fuction an extra level of histogram of gradient(4 divisions)
## applied to the edges extracted using sobel edge detector
################################################
def hogMoreFeatures(img):
    bin_n = 16 # Number of bins
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))
    
    # Divide to 16 sub-squares
    xdivider = img.shape[0]/4
    ydivider = img.shape[1]/4
        
    bin_cells = bins[:xdivider,:ydivider], bins[xdivider:(2*xdivider),:ydivider], \
        bins[(2*xdivider):(3*xdivider),:ydivider], bins[(3*xdivider):,:ydivider], \
        bins[:xdivider,ydivider:(2*ydivider)], bins[xdivider:(2*xdivider),ydivider:(2*ydivider)], \
        bins[(2*xdivider):(3*xdivider),ydivider:(2*ydivider)], bins[(3*xdivider):,ydivider:(2*ydivider)], \
        bins[:xdivider,(2*ydivider):(3*ydivider)], bins[xdivider:(2*xdivider),(2*ydivider):(3*ydivider)],\
        bins[(2*xdivider):(3*xdivider),(2*ydivider):(3*ydivider)], bins[(3*xdivider):,(2*ydivider):(3*ydivider)], \
        bins[:xdivider,(3*ydivider):], bins[xdivider:(2*xdivider),(3*ydivider):],  \
        bins[(2*xdivider):(3*xdivider),(3*ydivider):], bins[(3*xdivider):,(3*ydivider):]

    mag_cells = mag[:xdivider,:ydivider], mag[xdivider:(2*xdivider),:ydivider], \
        mag[(2*xdivider):(3*xdivider),:ydivider], mag[(3*xdivider):,:ydivider], \
        mag[:xdivider,ydivider:(2*ydivider)], mag[xdivider:(2*xdivider),ydivider:(2*ydivider)],  \
        mag[(2*xdivider):(3*xdivider),ydivider:(2*ydivider)], mag[(3*xdivider):,ydivider:(2*ydivider)], \
        mag[:xdivider,(2*ydivider):(3*ydivider)], mag[xdivider:(2*xdivider),(2*ydivider):(3*ydivider)],  \
        mag[(2*xdivider):(3*xdivider),(2*ydivider):(3*ydivider)], mag[(3*xdivider):,(2*ydivider):(3*ydivider)], \
        mag[:xdivider,(3*ydivider):], mag[xdivider:(2*xdivider),(3*ydivider):],  \
        mag[(2*xdivider):(3*xdivider),(3*ydivider):], mag[(3*xdivider):,(3*ydivider):]
    
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]

    histsNormalized = [h/sum(h) for h in hists]
    hists = histsNormalized    
    
    hist = np.hstack(hists)
    return hist


############################################
### Main Code starts here..
############################################

path = './letterSegments_simone/'
folders = [ f for f in listdir(path) if isdir(join(path,f)) ]

histogramDatabase = []
letterLabels = []

for i in range(len(folders)):   
#    print join(path,folders[i])+'/'
    currentFolder = path+folders[i]+'/'
    
    files = [ f for f in listdir(currentFolder) if isfile(join(currentFolder,f)) ]

    for j in range(len(files)):
        location = currentFolder
        print files[j]
        print location+files[j]
        letterValue = folders[i]
        print letterValue
        imgBGR = cv2.imread(location+files[j],1)
        gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        threshImage = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
        threshImage = 255 - threshImage
#        cv2.imshow('Current letter', threshImage)
#        cv2.waitKey(0)
        
        histogram = hog(gray_image)    
        histogramDatabase.append(histogram)
        letterLabels.append(ord(letterValue))
        
#        print histogramDatabase
#        print letterLabels
    
print 'data load succesfully'

###################################################
#################################################
print 'Data loaded successfully /n /n \n \n'
features = np.asarray(histogramDatabase)
labels = np.float32(np.asarray(letterLabels))

print features.shape
print labels.shape

datalength = features.shape[0]
# sort trainData and testData
trainFraction = 1
featuresTrain = features[:trainFraction*datalength, :]
featuresTest = features[trainFraction*datalength:, :]
featuresTest = featuresTrain

labelsTrain = labels[:trainFraction*datalength]
labelsTest = labels[trainFraction*datalength:]
labelsTest = labelsTrain

print featuresTrain.shape
print featuresTest.shape

print labelsTrain.shape
print labelsTest.shape


######     Now training      ########################
print 'now training..'
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
print labelsTrain
svm = cv2.SVM()

index = np.arange(featuresTrain.shape[0])
index = np.random.permutation(index)
print index
featuresTrain = featuresTrain[index, :]
labelsTrain = labelsTrain[index]
svm.train(np.float32(featuresTrain),np.float32(labelsTrain), params=svm_params)
svm.save('svm_data_OCR.dat')



######     Now testing      ########################
print 'now testing... '
result = svm.predict_all(np.float32(featuresTest))
print np.squeeze(result)
print labelsTest


#######   Check Accuracy   ########################
print 'computing results...'
mask = np.squeeze(result)==np.squeeze(labelsTest)
print mask
correct = np.count_nonzero(mask)
print correct*100.0/result.size

OCRmodel = cv2.SVM()
#OCRmodel.load("./svm_data_OCR_90_5.dat")
OCRmodel.load("./svm_data_OCR.dat")
print 'saved model'
newResult = np.squeeze(OCRmodel.predict_all(np.float32(featuresTest)))
mask = np.squeeze(newResult)==np.squeeze(labelsTest)
correct = np.count_nonzero(mask)
print correct*100.0/result.size



##############
#test on fresh image

# image_00915  image_00830      image_00141
#tes10
#tes16
#tes17
#tes25
imgBGR = cv2.imread('tes10.png',1)
#imgBGR = cv2.imread('207.png',1)
gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
threshImage = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,2)
threshImage = 255 - threshImage
#cv2.imshow('Current letter', threshImage)
#cv2.waitKey(0)

feature = np.float32(hog(gray_image))
test_result = OCRmodel.predict(feature)
print (test_result)
print chr(int(test_result))
print chr(int(svm.predict(feature)))

