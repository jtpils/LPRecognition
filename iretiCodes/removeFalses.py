# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:58:48 2015

@author: ireti
"""


import cv2
import numpy as np



def hog(img):
    bin_n = 16 # Number of bins
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    xdivider = img.shape[0]/2
    ydivider = img.shape[1]/2
        
    bin_cells = bins[:xdivider,:ydivider], bins[xdivider:,:ydivider], bins[:xdivider,ydivider:], bins[xdivider:,ydivider:]
    mag_cells = mag[:xdivider,:ydivider], mag[xdivider:,:ydivider], mag[:xdivider,ydivider:], mag[xdivider:,ydivider:]
    
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]

    histsNormalized = [h/sum(h) for h in hists]
    hists = histsNormalized    
    
    hist = np.hstack(hists)
    return hist
    


# Load data
from os import listdir
from os.path import isfile, join

def getData(path):
    files = [ f for f in listdir(path) if isfile(join(path,f)) ]
    histogramDatabase = []
    for i in range(len(files)):
        location = path     #'./data/'
#        print files[i]
#        print location+files[i]
        imgBGR = cv2.imread(location+files[i],1)
        gray_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('Black and White',gray_image)

#        cv2.waitKey(0)
        histogram = hog(gray_image)    
        histogramDatabase.append(histogram)
    return np.asarray(histogramDatabase)
    
#mypath = '/home/ireti/Desktop/Winter 2015/CS 231a/Course Project/Codes/CS231A-Project/Caltech1999'
mypath = './Caltech1999'        #training data

plateDataLocation = './plateData/'
nonplateDataLocation = './nonplateData/'

plateHist = getData(plateDataLocation)
nonplateHist = getData(nonplateDataLocation)

print 'Data loaded successfully /n /n \n \n'
numPlates = plateHist.shape[0]
numNonPlates = nonplateHist.shape[0]

responsePlates = np.ones(numPlates)
responseNonPlates = np.ones(numNonPlates)

# sort trainData and testData
plateHistTrain = plateHist[:0.7*numPlates]
plateHistTest = plateHist[0.7*numPlates:]

nonplateHistTrain = nonplateHist[:0.7*numNonPlates]
nonplateHistTest = nonplateHist[0.7*numNonPlates:]

print plateHistTrain.shape
print nonplateHistTrain.shape

trainData = np.concatenate((plateHistTrain,nonplateHistTrain), axis=0)
trainResponse = np.float32(np.zeros(trainData.shape[0]))
trainResponse[:plateHistTrain.shape[0]] = 1         #set number plate label to 1

print plateHistTest.shape
print nonplateHistTest.shape

print type(trainData[0][0])
print type(trainData)
print trainData.shape


testData = np.concatenate((plateHistTest,nonplateHistTest), axis=0)
testResponse = np.float32(np.zeros(testData.shape[0]))
testResponse[:plateHistTest.shape[0]] = 1           #set number plate label to 1


######     Now training      ########################
print 'now training..'
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

svm = cv2.SVM()

index = np.arange(trainData.shape[0])
index = np.random.permutation(index)
print index
trainData = trainData[index, :]
trainResponse = trainResponse[index]
svm.train(np.float32(trainData),trainResponse, params=svm_params)
svm.save('svm_data.dat')


######     Now testing      ########################
print 'now testing... '
result = svm.predict_all(np.float32(testData))
print np.squeeze(result)
print testResponse


#######   Check Accuracy   ########################
print 'computing results...'
mask = np.squeeze(result)==np.squeeze(testResponse)
#print mask
correct = np.count_nonzero(mask)
print correct*100.0/result.size

model2 = cv2.SVM()
model2.load("./svm_data.dat")
print 'saved model'
print model2.predict_all(np.float32(testData))