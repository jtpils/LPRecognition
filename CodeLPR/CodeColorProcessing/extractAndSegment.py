import cv2
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import math
import separateCharaters
import copy

        
def plotBoundingRegionsCV2(imgBGRCopy,starts,ends,base ,height):
        for i in range(len(starts)):
                xy = (starts[i] ,base)
                width = (ends[i]-1) - starts[i]
                cv2.rectangle(imgBGRCopy,(xy[0],xy[1]),(xy[0]+width,xy[1]+height),[0,0,0],1)

def extractAndSegment():
        #Read images in
        for n in range(126):
                m = n+1;
                name = 'image_0' + str(m).zfill(3)
                imgname = '../Caltech1999/image_0' + str(m).zfill(3) + '.jpg'
                print imgname
                imgBGR = cv2.imread(imgname,1)

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
                thFinal = 0
                plates =['']
                for th in threshold:
                        ret,gray2 = cv2.threshold(gray,th, 255,cv2.THRESH_BINARY_INV)
                        #cv2.imshow('GrayScale after thresholding',gray2)
                        contours, hier = cv2.findContours(gray2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                                (x,y,w,h) = cv2.boundingRect(cnt)
                                if abs(float(w)/h - WHR) < WHR_tol1 and minA<cv2.contourArea(cnt)<maxA: #and cv2.contourArea(cnt) > 0.7*w*h:
                                        cv2.rectangle(imgBGRCopy,(x,y),(x+w,y+h),[0,255,0],3)
                                        thFinal = th
                                        licensePlates.append((x,y,w,h))
                                        (horStarts, horEnds, base,height) = separateCharaters.separateCharacters(imgBGR[y:y+h,x:x+w,:],'grayWhite',thFinal)
                                        if len(horStarts) != 0:
                                                ret,gray2 = cv2.threshold(gray,thFinal, 255,cv2.THRESH_BINARY_INV)
                                                plotBoundingRegionsCV2(imgBGRCopy, horStarts+x, horEnds+x, base+y, height)
                                                plates.append(readPlate(gray2, horStarts+x, horEnds+x, base+y, height))
                                                #Might omit other detected plates (picks the first plate)
                dirLocation = 'PlateNumber/'
                if len(plates) ==1:
                        plate = "None"
                else:
                        plate = plates[1]
                outfile = dirLocation+ name + '_#' + plate+ '.png'
                cv2.imwrite(outfile, imgBGRCopy)
                


def readPlate(imgGray, starts, ends, base, height):
        while len(starts) >= 8:
                starts = starts[1:]
                ends = ends[1:]
        imgGray = 255-imgGray
        letterNum = [str(i) for i in range(10)] + [chr(i) for i in range(65,65+26,1)]
        plate = ''
        for i in xrange(len(starts)):
                width = (ends[i]-1) - starts[i]
                #cv2.imshow('GrayImg', imgGray)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                character = imgGray[base:base+height,starts[i]:starts[i]+width]
                #cv2.imshow('Character', character)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                scores = [0,]
                for j in range(len(letterNum)):
                        char = letterNum[j]
                        imgname = '../Templates/cartemp' + char +'.PNG'
                        temp = cv2.imread(imgname,1)
                        ret,temp = cv2.threshold(temp,175, 255,cv2.THRESH_BINARY)
                        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                        BWtemp = np.array(temp) >= 130
                        for ik in range(temp.shape[0]):
                                for jk in range(temp.shape[1]):
                                        temp[ik][jk] = BWtemp[ik][jk] * 255
                        #cv2.imshow(imgname,temp)
                        maxRunning = -float("inf")
                        for scale in np.linspace(0.5, 1, 10):
                                tempResized = cv2.resize(temp,(int(width/scale), int(height/scale)), interpolation = cv2.INTER_AREA)
                                res = cv2.matchTemplate(tempResized,character,cv2.TM_CCOEFF)
                                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                                alpha = 0
                                maxval = max_val -alpha*(sum(sum(x) for x in tempResized)- sum(sum(x) for x in character))
                                if i in [1, 2 ,3]:
                                        if j <= 9:
                                                max_val = -float("inf")
                                                #print "ij1", i, " ", j
                                        else: pass
                                elif i not in [1, 2, 3]:
                                        if j > 9:
                                                max_val = -float("inf")
                                                #print "ij2", i, " ", j
                                maxRunning =  max(maxRunning,max_val)
                        scores.append(maxRunning)
                mx = np.argmax(np.array(scores[1:]))
                plate = plate + letterNum[mx]
        return plate
                


if __name__ == '__main__':
	extractAndSegment()
