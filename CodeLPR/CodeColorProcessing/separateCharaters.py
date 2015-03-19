import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import math

def separateCharacters(BGRlicensePlate, illumination,thFinal): 
        #Parameters
        numChar = 7
        vertZones = 1
        Bstd = BGRlicensePlate[:,:,0].reshape(-1).std()
        Gstd = BGRlicensePlate[:,:,1].reshape(-1).std()
        Rstd = BGRlicensePlate[:,:,2].reshape(-1).std()
        maxStd = max([Bstd,Gstd,Rstd])
        indMax = np.argmax([Bstd,Gstd,Rstd])

        #Threshold over channel with highest std
        thresholdArray = {'grayWhite':thFinal+7}
        threshold = thresholdArray[illumination]
        threshChannel = BGRlicensePlate[:,:,indMax]
        ret, BW = cv2.threshold(threshChannel,threshold,255,cv2.THRESH_BINARY)

        #Build vertical Histogram
        BWvert = np.rot90(BW)
        vertHist = (255-BWvert).sum(0)/255
        

        #Find Histogram Vertical regions
        (vertStarts,vertEnds,vertT) = findRegions(vertHist,vertZones)

        #Crop middle vertical zone
        if len(vertStarts) == 0:
                return [], [] , [] , []
        BWhorz = BW[vertStarts[0]:vertEnds[0],:]
        
        #Build horizontal Histogram
        horHist = (255-BWhorz).sum(0)/255
        
        #Find Histogram Horizontal regions
        (horStarts,horEnds,t) =findRegions(horHist,numChar)
        
        #Return coordinates of character segments, top character line coordinate, and character height
        return np.array(horStarts)-1, np.array(horEnds)+2, vertStarts[0]-1,vertEnds[0]+1-1-(vertStarts[0]-1)

        
        
def findRegions(horHist, numRegions):
        numRegions = numRegions + 2 
        numPixelCols = horHist.shape[0]
        segThreshold = numPixelCols/8 
        finalThreshold = segThreshold
        top = numPixelCols
        bottom = 0
        oldRegionStarts = []
        oldRegionEnds = []
        regionStarts = []
        regionEnds = []
        while top >= bottom + 1: 
                thresholdedHist = (horHist >= segThreshold)
                current =0; 
                previous = 0;
                previousIdx = 0;
                regionStarts = []
                regionEnds = []
                for i in range(numPixelCols):
                        current = thresholdedHist[i]
                        if current and (not previous):
                                regionStarts.append(i)
                                previous = current
                                previousIdx = i
                        elif (not current) and  previous:
                                regionEnds.append(i)
                                previous = current
                                previousIdx  = i
                if len(regionStarts) <  numRegions: 
                        bottom = segThreshold
                        segThreshold = (segThreshold + top)/2.0
                        
                else:
                        oldRegionStarts = regionStarts 
                        oldRegionEnds = regionEnds
                        finalThreshold = segThreshold
                        top = segThreshold
                        segThreshold= (segThreshold + bottom)/2.0 
                        
        if len(oldRegionStarts)  != len(oldRegionEnds): 
                oldRegionEnds.append(numPixelCols)

        #Trim Plate edges
        oldRegionStarts = oldRegionStarts[1:-1]
        oldRegionEnds = oldRegionEnds[1:-1]
                
        #Remove regions with one pixel
        finalRegionStarts = []
        finalRegionEnds = []
        for i in range(len(oldRegionStarts)):
                if oldRegionEnds[i] > oldRegionStarts[i] + 1:
                        finalRegionStarts.append(oldRegionStarts[i])
                        finalRegionEnds.append(oldRegionEnds[i])
        return (finalRegionStarts,finalRegionEnds,finalThreshold)        
        

if __name__ == '__main__':
        BGRlicensePlate = cv2.imread('LPs/LP3.PNG')
        separateCharacters(BGRlicensePlate,'grayWhite')


