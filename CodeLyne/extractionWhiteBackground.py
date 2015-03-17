import cv2
import numpy as np 
import matplotlib as plt


#Read images in
for n in range(126):
	m = n+1;
	name = 'image_0' + str(m).zfill(3)
	imgname = '../Caltech1999/image_0' + str(m).zfill(3) + '.jpg'
	print imgname
	imgBGR = cv2.imread(imgname,1)

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
				cv2.rectangle(imgBGR,(x,y),(x+w,y+h),[0,255,0],3)
			#elif abs(float(w)/h - WHR) < WHR_tol2 and minA<cv2.contourArea(cnt)<maxA: #and cv2.contourArea(cnt) > 0.7*w*h:
			#	cv2.rectangle(imgBGR,(x,y),(x+w,y+h),[0,0,255],3)
				
		
			
	dirLocation = 'Output/'
	outfile = dirLocation+ name + '_output.png'
	print outfile
	cv2.imwrite(outfile, imgBGR)
	#cv2.imshow('IMG2',imgBGR)
	#cv2.waitKey(0)
	cv2.destroyAllWindows()
