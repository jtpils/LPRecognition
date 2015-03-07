import cv2
import numpy as np 
import matplotlib as plt


#Read image in
imgBGR = cv2.imread('Caltech1999/image_0113.jpg',1)

#imgRGB = imgBGR[:,:,::-1]

#Display image
cv2.imshow('Car Picture', imgBGR)
#cv2.waitKey(0)


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

darkRed = 80;
margin = 10;
second = 42;

imgBlue = imgBGR[:,:,0]
imgGreen = imgBGR[:,:,1]
imgRed = imgBGR[:,:,2]

WHR = 2.0;

gray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
ret,gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
gray2 = gray.copy()
cv2.imshow('IMG1',gray)
contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	(x,y,w,h) = cv2.boundingRect(cnt)
	if abs(float(w)/h - WHR) < 0.4 and 400<cv2.contourArea(cnt)<10000: #and cv2.contourArea(cnt) > 0.7*w*h:
	#if 200<cv2.contourArea(cnt)<10000:
	#if abs(float(w)/h - WHR) < 0.5: 
		print abs(float(w)/h - WHR)
		cv2.rectangle(imgBGR,(x,y),(x+w,y+h),[0,255,0],1)
		print cv2.contourArea(cnt)
		print w*h

cv2.imshow('IMG2',imgBGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot truly red pixels in red
#isRed = (imgRed > darkRed -margin)*(imgRed < darkRed+margin)*(imgBlue < second + margin)*(imgBlue > second - margin)*(imgGreen < second + margin)*(imgGreen > second - margin)
# isRed = (imgRed < darkRed)*(imgRed > 1.1*imgBlue)*(imgRed > 1.1*imgGreen)
# minval = imgBGR.min(axis=2)
# maxval = imgBGR.max(axis=2)
# #isDarkRed =  (abs(minval-maxval) < 5)*(imgRed >60) #(imgRed<90)
# redPixels = imgBGR
# redPixels[:,:,2] = 255*isRed
# redPixels[:,:,0] = imgBlue*(1-isRed)
# redPixels[:,:,1] = imgGreen*(1- isRed)
# cv2.imshow('redPixels', redPixels)

# # Plot truly blue pixels in blue
# # isBlue = (imgBlue > darkRed)*(imgBlue>1.5*imgRed)*(imgBlue>1.5*imgGreen)
# # bluePixels = imgBGR
# # bluePixels[:,:,0] = 255*isBlue
# # bluePixels[:,:,2] = imgRed*(1-isBlue)
# # bluePixels[:,:,1] = imgGreen*(1- isBlue)
# # cv2.imshow('bluePixels', bluePixels)


cv2.waitKey(0)

cv2.destroyAllWindows()