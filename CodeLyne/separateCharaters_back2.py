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

        #cv2.imshow('Plate',BGRlicensePlate)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #Find Channel with max standard deviation
        Bstd = BGRlicensePlate[:,:,0].reshape(-1).std()
        Gstd = BGRlicensePlate[:,:,1].reshape(-1).std()
        Rstd = BGRlicensePlate[:,:,2].reshape(-1).std()
        maxStd = max([Bstd,Gstd,Rstd])
        indMax = np.argmax([Bstd,Gstd,Rstd])
        #print [Bstd,Gstd,Rstd]

        #Threshold over channel with highest std
        thresholdArray = {'grayWhite':thFinal}
        threshold = thresholdArray[illumination]
        threshChannel = BGRlicensePlate[:,:,indMax]
        #cv2.imshow('Channel with maxstd',threshChannel)
        ret, BW = cv2.threshold(threshChannel,threshold,255,cv2.THRESH_BINARY)
        #cv2.imshow('After Thresholding', BW)
        #BW = BW.transpose()

        #Find Percentage of white pixels -USE THIS TO THRESHOLD
        #percentWhite = 1.0*BW[BW==255].shape[0]/BW.size
        #print percentWhite


        #Build vertical Histogram
        #horHist = (255-BW).sum(0)/255
        BWvert = np.rot90(BW)
        vertHist = (255-BWvert).sum(0)/255
        #plt.imshow(BWvert, cmap=cm.Greys_r)
        #plt.plot(vertHist)
        #fig1 , axes1 = plt.subplots(2,1,sharex=True)
        #axes1[0].imshow(BWvert, cmap=cm.Greys_r)
        #axes1[1].plot(verHist)
        

        #Find Histogram Vertical regions
        (vertStarts,vertEnds,vertT) = findRegions(vertHist,vertZones)
        #print (vertStarts,vertEnds,vertT)
        #Show plate vertical region
        #plotBoundingRegions(vertStarts,vertEnds,0,BWvert.shape[0])
        #currentAxis1 = plt.gca()
        #currentAxis1.add_patch(Rectangle((vertStarts[0],0),vertEnds[0]-1-vertStarts[0],BWvert.shape[0],facecolor="none",edgecolor="red"))

        #Crop middle vertical zone
        if len(vertStarts) == 0:
                return [], [] , [] , []
        #print "Points", vertStarts[0],vertEnds[0]
        BWhorz = BW[vertStarts[0]:vertEnds[0],:]
        #plt.imshow(BW,cmap=cm.Greys_r)
        
        #Build horizontal Histogram
        horHist = (255-BWhorz).sum(0)/255
        #plt.plot(horHist)
        
        #Find Histogram Horizontal regions
        (horStarts,horEnds,t) =findRegions(horHist,numChar)
        #plt.imshow(BW, cmap = cm.Greys_r)
        #plt.imshow(BGRlicensePlate)
        #plotBoundingRegions(horStarts, horEnds, vertStarts[0],vertEnds[0]-1-vertStarts[0])
        #print (horStarts,horEnds,horT)
        #print len(horStarts), len(horEnds)
        #histThreshold = [t for i in range(len(horHist))]
        #plt.plot(histThreshold, color="red")

        #Plot Histograms
        #fig , axes = plt.subplots(2,1,sharex=True)
        #axes[0].plot(horHist)
        #axes[0].plot(histThreshold, color="red")
        #axes[1].imshow(BW, cmap = cm.Greys_r)
        #plt.plot(horHist)
        #plt.plot(histThreshold, color="red")
        


        #Plot character boxes boundaries
        #plt.imshow(BW, cmap = cm.Greys_r)
        #plt.show()
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #Return coordinates of character segments, top character line coordinate, and character height
        #return np.array(horStarts)-1, np.array(horEnds)+2, vertStarts[0]-1,vertEnds[0]+1-1-(vertStarts[0]-1)
        topCropPercent = 1.0/3
        bottomCropPercent = 1.0/8
        top = int(math.floor(topCropPercent*BGRlicensePlate.shape[0]))
        bottom = int(math.ceil(BGRlicensePlate.shape[0]*(1-bottomCropPercent)))
        return np.array(horStarts)-1, np.array(horEnds)+2, top,bottom-top

        
        
def findRegions(horHist, numRegions):
        numRegions = numRegions + 2 #Account for plate boarders
        numPixelCols = horHist.shape[0]
        segThreshold = numPixelCols/8 #Assumes at least 1/8 of the pixels are in the central region. Starting point is crucial. It has to be a good threshold or it fails
        finalThreshold = segThreshold
        top = numPixelCols
        bottom = 0
        oldRegionStarts = []
        oldRegionEnds = []
        regionStarts = []
        regionEnds = []
        while top >= bottom + 1: #To avoid infinite loop from continuous real numbers
                thresholdedHist = (horHist >= segThreshold)
                current =0; #Current histogram col (x) in horizontal direction
                previous = 0;
                previousIdx = 0;
                regionStarts = []
                regionEnds = []
                #print "NewThreshold = " + str(segThreshold)
                #print "Top = " + str(top)
                #print "Bottom = " + str(bottom)
                #print horHist[0]
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
                #print "Starts = " ,regionStarts[1:-1]
                #print "Ends = " ,regionEnds[1:]
                if len(regionStarts) <  numRegions:  #Increase threshold
                        bottom = segThreshold
                        segThreshold = (segThreshold + top)/2.0
                        
                else:
                        oldRegionStarts = regionStarts #Update last set of valid regions
                        oldRegionEnds = regionEnds
                        finalThreshold = segThreshold
                        top = segThreshold
                        segThreshold= (segThreshold + bottom)/2.0 #Decrease threshold
                #print "Length = " + str(len(regionStarts))
                #print "\n"
                        
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
        
def plotBoundingRegions(starts,ends,base ,height):
        for i in range(len(starts)):
                xy = (starts[i] ,base)
                width = (ends[i]-1) - starts[i]
                rect = Rectangle(xy,width,height,facecolor ="none", edgecolor = "green",linewidth='2.0')
                currentAxis = plt.gca()
                currentAxis.add_patch(rect)
        plt.show()
        
                
                


def cropTopBottom(BGRlicensePlate):
        #Crop top and bottom parts of the license plates
        topCropPercent = 1.0/3
        bottomCropPercent = 1.0/6
        top = int(math.floor(topCropPercent*BGRlicensePlate.shape[0]))
        bottom = int(math.ceil(BGRlicensePlate.shape[0]*(1-bottomCropPercent)))
        #BGRlicensePlate = BGRlicensePlate[range(top,bottom),:,:]
        return BGRlicensePlate[range(top,bottom),:,:]










if __name__ == '__main__':
        BGRlicensePlate = cv2.imread('LPs/LP3.PNG')
        separateCharacters(BGRlicensePlate,'grayWhite')
