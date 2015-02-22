import cv2
import numpy as np 


#Read image in
imgBGR = cv2.imread('Caltech1999/image_0126.jpg',1)
#imgRGB = imgBGR[:,:,::-1]

#Show image
cv2.imshow('Car Picture', imgBGR)
cv2.waitKey(0)
#Display image


#Classify image pixels into numColors color bins
#Dark blue, blue, light blue, 
#Dark red, red, light red
#Gray white, white, light white
#Dark Yellow, Yellow, Light yellow
#Other
#Do we need black?
# Build a vertical and horizontal histogram for each color
# input: image file
# output: vertColorHist(numColors,numVertPixels) 
#		  horColorHist(numColors,numHorPixels)

cv2.destroyAllWindows()