import numpy as np
import cv2

def laneDetection(img):

	#Converting RGB to Grayscale 
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Converting RGB to HSV
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	#Setting the Threshold values for Lane colours i.e., Yellow and White
	yellowLow = np.array([170, 175, 175], dtype="uint8") 
	yellowHigh = np.array([200,255,255], dtype="uint8")
	whiteLow = np.array([50, 50, 50], dtype="uint8") 
	whiteHigh = np.array([200,255,255], dtype="uint8")

	#Creating the mask with Yellow, White thresholds mentioned above
	yellowMask = cv2.inRange(hsvImg, yellowLow, yellowHigh)
	whiteMask = cv2.inRange(hsvImg,whiteLow,whiteHigh)
	combMask = cv2.bitwise_or(grayImg, yellowMask, whiteMask)
	
	#Masking the GrayImage with Combined mask
	yandwMaskedImg = cv2.bitwise_and(grayImg, combMask)

	#Blurring operation Gaussian Blur
	gaussianBlurImg = cv2.GaussianBlur(yandwMaskedImg, (5,5), 0)
	#Implementing Canny Edge Detection Algo
	edgeImg = cv2.Canny(gaussianBlurImg,50,150)

	#Creating the mask for ROI(Region of Interest)
	blackImg = np.zeros(edgeImg.shape, dtype="uint8")
	roiPoints = np.array([[(0,54), (800,54), (0,628), (800,628)]], dtype=np.int32)
	roiMask = cv2.fillPoly(blackImg, roiPoints,(255))
	#ROI Image 
	roiImg = cv2.bitwise_and(roiMask,edgeImg)

	#Getting the Probablistic Hough Lines 
	houghLines = cv2.HoughLinesP(roiImg, 2, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=100)
	
	#For drawing the Hough lines, creating the new image
	lineImg = np.zeros(img.shape,dtype="uint8")

	# Drawing the lines on the created new image(lineImg)
	for line in houghLines:
	    for x1,y1,x2,y2 in line:
	        cv2.line(lineImg, (x1, y1), (x2, y2), (0,0,255), 10)

	#Now weighted average of original image and line image to get detected lanes on the image
	alpha=0.8
	beta=1
	lambdaa=0
	finalLineDetectImg = cv2.addWeighted(lineImg, alpha, img, beta, lambdaa)

	return finalLineDetectImg

if __name__ == "__main__":
	cap = cv2.VideoCapture('new.mp4') 	

	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			processed = laneDetection(frame)
			cv2.imshow('Frame',processed)
			cv2.waitKey(50)

		else:
			break
	 
	# When everything done, release the video capture object
	cap.release()
	 
	# Closes all the frames
	cv2.destroyAllWindows()