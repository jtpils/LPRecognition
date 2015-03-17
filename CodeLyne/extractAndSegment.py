import cv2
import numpy as np 
import matplotlib as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import math
import separateCharaters
import copy


#Read images in
for n in range(1):
##        m = n+1;
##        name = 'image_0' + str(m).zfill(3)
##        imgname = '../Caltech1999/image_0' + str(m).zfill(3) + '.jpg'
##        print imgname
##        imgBGR = cv2.imread(imgname,1)
        imgname = '../Caltech1999/image_0111.jpg'
        imgBGR = cv2.imread(imgname,1)
        name = 'test2'
        #Display image
        #cv2.imshow('Car Picture', imgBGR)
        #cv2.waitKey(0)


        #Classify image pixels into numColors color bins
        #Gray white, white, light white 
        grayWhite = 100
        white = 170
        lightWhite = 240
        threshold = [grayWhite,white,lightWhite]

        #CA License plate width to height ratio
        WHR = 2.0
        WHR_tol1 = 0.2
        WHR_tol2 = 0.4
        minA = 300
        maxA = 6000

        licensePlates = []
        imgBGRCopy = copy.deepcopy(imgBGR)
        #Turn image to grayscale before thresholding on the intensity of white
        gray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
        plateDetected = 0
        for th in threshold:
                ret,gray2 = cv2.threshold(gray,th, 255,cv2.THRESH_BINARY_INV)
                cv2.imshow('GrayScale after thresholding',gray2)
                contours, hier = cv2.findContours(gray2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                        (x,y,w,h) = cv2.boundingRect(cnt)
                        if abs(float(w)/h - WHR) < WHR_tol1 and minA<cv2.contourArea(cnt)<maxA: #and cv2.contourArea(cnt) > 0.7*w*h:
                                cv2.rectangle(imgBGRCopy,(x,y),(x+w,y+h),[0,255,0],3)
                                licensePlates.append((x,y,w,h))
                        #elif abs(float(w)/h - WHR) < WHR_tol2 and minA<cv2.contourArea(cnt)<maxA: #and cv2.contourArea(cnt) > 0.7*w*h:
                        #	cv2.rectangle(imgBGR,(x,y),(x+w,y+h),[0,0,255],3)
                                
        for i in range(len(licensePlates)):
                (x,y,w,h) = licensePlates[i]
                #print (x,y,w,h)
                (horStarts, horEnds, base,height) = separateCharaters.separateCharacters(imgBGR[y:y+h,x:x+w,:],'grayWhite')
                
                
        dirLocation = 'SegmentationOutput/'
        outfile = dirLocation+ name + '_seg.png'
        print outfile
        cv2.imwrite(outfile, imgBGRCopy)
        #cv2.imshow('IMG2',imgBGR)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

