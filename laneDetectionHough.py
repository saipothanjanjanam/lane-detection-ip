import numpy as np
import cv2

def laneDetection(img):

	#Converting RGB to Grayscale 
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Converting RGB to HSV
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	#Setting the Threshold values for Lane colours i.e., Yellow and White
	yellowLow = np.array([170, 170, 30], dtype="uint8") 
	yellowHigh = np.array([200,200,140], dtype="uint8")
	whiteLow = np.array([170, 170, 170], dtype="uint8") 
	whiteHigh = np.array([200,200,200], dtype="uint8")

	#Creating the mask with Yellow, White thresholds mentioned above
	yellowMask = cv2.inRange(hsvImg, yellowLow, yellowHigh)
	whiteMask = cv2.inRange(hsvImg,whiteLow,whiteHigh)
	combMask = cv2.bitwise_or(grayImg, yellowMask, whiteMask)
	
	#Masking the GrayImage with Combined mask
	yandwMaskedImg = cv2.bitwise_and(grayImg, combMask)

	#Blurring operation Gaussian Blur
	gaussianBlurImg = cv2.GaussianBlur(yandwMaskedImg, (5,5), 0)
	
	#Implementing Sobel Vertical Edge Detection Algo
	edgeImg = cv2.Sobel(gaussianBlurImg,cv2.CV_8U,1,0,ksize=5)
	(thresh, edgeImg) = cv2.threshold(edgeImg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	
	# #Creating the mask for ROI(Region of Interest)
	# blackImg = np.zeros(edgeImg.shape, dtype="uint8")
	# roiPoints = np.array([[(0,628), (800,628), (0,55), (800,55)]], dtype=np.int32)
	# roiMask = cv2.fillPoly(blackImg, roiPoints,(255))
	# #ROI Image 
	# roiImg = cv2.bitwise_and(roiMask,edgeImg)

	#For drawing the Hough lines, creating the new image
	lineImg = np.zeros(img.shape,dtype="uint8")

	#Hough Lines and drawing them
	houghLines = cv2.HoughLines(edgeImg,1,np.pi/180,200)

	rhoL = 0
	thetaL = 0
	rhoR = 0
	thetaR = 0
	thetaAvg = 0
	rhoAvg = 0
	countL = 0
	countR = 0

	if houghLines is not None:
		for _ in houghLines:
			rho = _[0][0]
			theta = _[0][1]

			if rho<0:
				rhoL += rho
				thetaL += theta
				countL +=1
			else:
				rhoR += rho
				thetaR += theta
				countR += 1

		if countL > 0:
			rhoAvg = rhoL/countL
			thetaAvg = thetaL/countL
			a = np.cos(thetaAvg)
			b = np.sin(thetaAvg)
			x0 = a*rhoAvg
			y0 = b*rhoAvg
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(lineImg,(x1,y1),(x2,y2),(0,0,255),2)
			 
		if countR > 0:
			rhoAvg = rhoR/countR
			thetaAvg = thetaR/countR
			a = np.cos(thetaAvg)
			b = np.sin(thetaAvg)
			x0 = a*rhoAvg
			y0 = b*rhoAvg
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(lineImg,(x1,y1),(x2,y2),(0,0,255),2)

	# Now weighted average of original image and line image to get detected lanes on the image
	alpha=0.8
	beta=1
	lambdaa=0
	finalLineDetectImg = cv2.addWeighted(lineImg, alpha, img, beta, lambdaa)

	return finalLineDetectImg

if __name__ == "__main__":
	cap = cv2.VideoCapture('rec.mkv') 	

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