# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO
import os, sys, csv, fnmatch
import math, random
import numpy as np
import cv2
from matplotlib import pyplot as plt


def find_lowest_windows(img_left,img_right,line_left,line_right,midpointL=None,midpointR=None, window_size=[30,30], flag_beginning=True,verbose=False):
	lineL = np.array(line_left); lineR = np.array(line_right)

	if flag_beginning == True:
		# Find the value found by sliding a window down starting at the beginning coordinate for the lines
		locL = lineL[:,0]; locR = lineR[:,-1]
		winMinL = slide_window_down(img_left,locL,size=window_size)
		winMinR = slide_window_down(img_right,locR,size=window_size)
		if verbose == True:
			print("Sliding Window Mins @ Beginning: ",winMinL,winMinR)

		# Find the value found by sliding a window down starting at the middle coordinate for the lines
		rL,cL = lineL.shape[:2]
		rR,cR = lineR.shape[:2]
		if verbose == True:
			print("Shape Left", rL, cL)
			print("Shape Right", rR, cR)

		midIdxL = cL // 2
		midIdxR = cR // 2
		if verbose == True:
			print("Left Middle Index", midIdxL)
			print("Right Middle Index", midIdxR)

		locL = lineL[:,midIdxL]; locR = lineR[:,midIdxR]
		winMinMidL = slide_window_down(img_left,locL,size=window_size)
		winMinMidR = slide_window_down(img_right,locR,size=window_size)

		if verbose == True:
			print("Sliding Window Mins @ Middle: ",winMinMidL,winMinMidR)

		if winMinMidL > winMinL:
			winMinL = winMinMidL
		if winMinMidR > winMinR:
			winMinR = winMinMidR
	else:
		locL = lineL[:,-1]; locR = lineR[:, 0]
		winMinL = slide_window_down(img_left,locL,size=window_size)
		winMinR = slide_window_down(img_right,locR,size=window_size)
		if verbose == True:
			print("Locations Used @ End: ",locL,locR)
			print("Sliding Window Mins @ End: ",winMinL,winMinR)

	return [winMinL,winMinR]

def slide_window_down(img,location,size,threshold=400,display=None, verbose=False,flag_show=False):
	good_inds = []
	window = 0
	flag_done = False
	tmp = np.copy(img)
	try:
		d = display.dtype
		display_windows = np.copy(display)
	except:
		display_windows = np.copy(img)

	window_height,window_width = size
	maxheight,maxwidth = tmp.shape[:2]

	x_current = abs(int(location[0])); 	y_current = abs(int(location[1]))

	# Move x-coordinate if we are at the edge so we can capture a bit more information
	if x_current <= window_width:
		x_current = window_width
	elif x_current + window_width > maxwidth:
		x_current = maxwidth - window_width
	elif x_current - window_width < 0:
		x_current = window_width
	if verbose == True:
		print("Starting Location: ", x_current, y_current)

	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	while not flag_done:
		prev_good_inds = good_inds

		if y_current >= maxheight:
			flag_done = True
			if verbose == True:
				print("Exiting: Reached max image height.")
			continue

		# Check for [Y] edge conditions
		if y_current - window_height >= 0:
			win_y_low = y_current - window_height
		else:
			win_y_low = 0

		if y_current + window_height <= maxheight:
			win_y_high = y_current + window_height
		else:
			win_y_high = maxheight

		# Check for [X] edge conditions
		if x_current - window_width >= 0:
			win_x_low = x_current - window_width
		else:
			win_x_low = 0

		if x_current + window_width <= maxwidth:
			win_x_high = x_current + window_width
		else:
			win_x_high = maxwidth

		cv2.circle(display_windows,(x_current,y_current),2,(255,0,0),-1)
		cv2.rectangle(display_windows,(win_x_low,win_y_high),(win_x_high,win_y_low),(255,0,0), 2)

		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

		if verbose == True:
			print("	Current Window [" + str(window) + "] Center: " + str(x_current) + ", " + str(y_current))
			print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
			print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))
			print("		Current Window # Good Pixels: " + str(len(good_inds)))

		if len(good_inds) > threshold:
			y_current = y_current + window_height
		else:
			if verbose == True:
				print("	Not enough good pixels @ " + str(x_current) + ", " + str(y_current))
			try:
				y_mean = np.int(np.mean(nonzeroy[good_inds]))
				y_current = y_mean
			except:
				try:
					y_mean = np.int(np.mean(nonzeroy[prev_good_inds]))
					y_current = y_mean # + window_height
				except:
					y_mean = y_current
					y_current = y_current
			if verbose == True:
				print("Moving Y current to Mean Y Pixels: ", y_mean)
			cv2.circle(display_windows,(x_current,y_current),2,(255,255,0),-1)
			flag_done = True

	if verbose == True:
		print("Last Y found: ", y_current)
	if flag_show == True:
		cv2.imshow("Sliding Window Down",display_windows)
	return y_current


def slide_window_right(img,location,size=[30,30],threshold=400,display=None, verbose=False,flag_show=False):
	window = 0
	good_inds = []
	flag_done = False
	tmp = np.copy(img)
	try:
		d = display.dtype
		display_windows = np.copy(display)
	except:
		display_windows = np.copy(img)

	h,w = tmp.shape[:2]
	window_height,window_width = size

	x_current = abs(int(location[0]))
	y_current = abs(int(location[1]))

	# Move x-coordinate if we are at the edge so we can capture a bit more information
	if x_current <= window_width: x_current = window_width
	if y_current >= h: y_current = h - window_height
	if verbose == True: print("Starting Location: ", x_current, y_current)

	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	while not flag_done:
		prev_good_inds = good_inds

		if x_current >= w:
			flag_done = True
			if verbose == True:
				print("Exiting: Reached max image width.")
			continue

		# Check for [Y] edge conditions
		if y_current - window_height >= 0: win_y_low = y_current - window_height
		else: win_y_low = 0

		if y_current + window_height <= h: win_y_high = y_current + window_height
		else: win_y_high = h

		# Check for [X] edge conditions
		if x_current - window_width >= 0: win_x_low = x_current - window_width
		else: win_x_low = 0

		if x_current + window_width <= w: win_x_high = x_current + window_width
		else: win_x_high = w

		cv2.circle(display_windows,(x_current,y_current),2,(255,255,255),-1)
		cv2.rectangle(display_windows,(win_x_low,win_y_high),(win_x_high,win_y_low),(255,0,0), 2)

		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

		if verbose == True:
			print("	Current Window [" + str(window) + "] Center: " + str(x_current) + ", " + str(y_current))
			print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
			print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))
			print("		Current Window # Good Pixels: " + str(len(good_inds)))

		if len(good_inds) > threshold:
			x_current = x_current + window_width
		else:
			if verbose == True: print("	Not enough good pixels @ " + str(x_current) + ", " + str(y_current))
			try:
				x_mean = np.int(np.mean(nonzerox[good_inds]))
				x_current = x_mean
			except:
				try:
					x_mean = np.int(np.mean(nonzerox[prev_good_inds]))
					x_current = x_mean
				except:
					x_mean = x_current
					x_current = x_current
			if verbose == True: print("Moving X current to Mean X Pixels: ", x_mean)
			cv2.circle(display_windows,(x_current,y_current),2,(255,255,0),-1)
			flag_done = True

	if verbose == True: print("Last X found: ", x_current)
	if flag_show == True: cv2.imshow("Sliding Window Right",display_windows)
	return x_current


def slide_window_left(img,location,size=[30,30],threshold=400,display=None, verbose=False,flag_show=False):
	window = 0
	good_inds = []
	flag_done = False
	tmp = np.copy(img)
	try:
		d = display.dtype
		display_windows = np.copy(display)
	except:
		display_windows = np.copy(img)

	window_height,window_width = size
	maxheight,maxwidth = tmp.shape[:2]

	x_current = abs(int(location[0])); 	y_current = abs(int(location[1]))

	# Move x-coordinate if we are at the edge so we can capture a bit more information
	if x_current >= maxwidth:
		x_current = maxwidth - window_width
	if y_current >= maxheight:
		y_current = maxheight - window_height
	if verbose == True:
		print("Starting Location: ", x_current, y_current)

	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	while not flag_done:
		prev_good_inds = good_inds

		if x_current <= 0:
			flag_done = True
			if verbose == True:
				print("Exiting: Reached max image width.")
			continue

		# Check for [Y] edge conditions
		if y_current - window_height >= 0:
			win_y_low = y_current - window_height
		else:
			win_y_low = 0

		if y_current + window_height <= maxheight:
			win_y_high = y_current + window_height
		else:
			win_y_high = maxheight

		# Check for [X] edge conditions
		if x_current - window_width >= 0:
			win_x_low = x_current - window_width
		else:
			win_x_low = 0

		if x_current + window_width <= maxwidth:
			win_x_high = x_current + window_width
		else:
			win_x_high = maxwidth

		cv2.circle(display_windows,(x_current,y_current),2,(255,255,255),-1)
		cv2.rectangle(display_windows,(win_x_low,win_y_high),(win_x_high,win_y_low),(0,0,255), 2)

		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

		if verbose == True:
			print("	Current Window [" + str(window) + "] Center: " + str(x_current) + ", " + str(y_current))
			print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
			print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))
			print("		Current Window # Good Pixels: " + str(len(good_inds)))

		if len(good_inds) > threshold:
			x_current = x_current - window_width
		else:
			if verbose == True:
				print("	Not enough good pixels @ " + str(x_current) + ", " + str(y_current))
			try:
				x_mean = np.int(np.mean(nonzerox[good_inds]))
				x_current = x_mean
			except:
				try:
					x_mean = np.int(np.mean(nonzerox[prev_good_inds]))
					x_current = x_mean - window_width
				except:
					x_mean = x_current
					x_current = x_current
			if verbose == True:
				print("Moving X current to Mean X Pixels: ", x_mean)
			cv2.circle(display_windows,(x_current,y_current),2,(0,255,255),-1)
			flag_done = True

	if verbose == True:
		print("Last X found: ", x_current)
	if flag_show == True:
		cv2.imshow("Sliding Window Left",display_windows)
	return x_current


def slide_window(center, ):
	# Slide window from previousy found center (Clip at image edges)
	# Update vertical [Y] window edges
	if(yk - dWy >= 0): wy_low = yk - dWy
	else: wy_low = 0

	if(yk + dWy <= h): wy_high = yk + dWy
	else: wy_high = h

	# Update horizontal [X] window edges
	if(xk - dWx >= 0): wx_low = xk - dWx
	else: wx_low = 0

	if(xk + dWx <= w): wx_high = xk + dWx
	else: wx_high = w
