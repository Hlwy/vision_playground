# Created by: Hunter Young
# Date: 10/5/17
#
# Script Description:
# 	Script is designed to take various commandline arguments making a very simple
# 	and user-friendly method to take any video file of interest as input and extract
# 	all the possible images available into a seperate folder, in addition to outputting
# 	a .csv file logging any additional useful information that is associated with each image.
#
# Current Recommended Usage: (in terminal)
# 	python parse_video.py -p /path/to/videos/home/directory -v name_of_video.mp4 -o name_of_output_data_file
#
# ASSUMPTIONS:
# 	Assumes that the video file name is formatted like so, "X_X_MMDDYY_TimeOfDay_SysTimeMsecRecordingBegins.mp4"

# from sklearn.cluster import KMeans
import cv2
import os
import csv
import argparse
from matplotlib import pyplot as plt
import numpy as np

# Setup commandline argument(s) structures
ap = argparse.ArgumentParser(description='Image Segmentation')
ap.add_argument("--pic", "-p", type=str, default='40.png', metavar='FILE', help="Name of video file to parse")
# Store parsed arguments into array of variables
args = vars(ap.parse_args())

# Extract stored arguments array into individual variables for later usage in script
_img = args["pic"]

img = cv2.imread(_img)	# Input video file as OpenCV VideoCapture device
tmp = cv2.resize(img, (640,480))
hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)

lower_red = np.array([35, 52, 118])
upper_red = np.array([255, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(tmp, tmp, mask = mask)

test = cv2.pyrMeanShiftFiltering(res,10, 45, 3)
# cv2.imshow('mean shift',test)
gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# ret, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('threshold',thresh)
edges = cv2.Canny(thresh,50,200,apertureSize = 3)
# cv2.imshow('edges',edges)
res2 = cv2.bitwise_and(tmp, tmp, mask = thresh)
cv2.imshow('and',res2)

kernel = np.ones((20,20),np.uint8)
opening = cv2.morphologyEx(res2,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imshow('opening',opening)
closing = cv2.morphologyEx(res2,cv2.MORPH_CLOSE,kernel, iterations = 2)
cv2.imshow('closing',closing)


while(1):
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
