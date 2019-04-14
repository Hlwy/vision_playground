# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO
import os, sys, fnmatch
import cv2, time
import numpy as np

from matplotlib import pyplot as plt
from img_utils import *


def make_uv_overlay(_img, umap, vmap,border_width=2):
    img = np.copy(_img)
    try: img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    except: pass
    blank = np.ones((umap.shape[0],vmap.shape[1],3),np.uint8)*255
    # borderb = np.ones((2,umap.shape[1],3),dtype=np.uint8); borderb[:] = border_color
    # borders = np.ones((vmap.shape[0],2,3),dtype=np.uint8); borders[:] = border_color
    borderb = np.ones((border_width,umap.shape[1]+border_width+vmap.shape[1],3),dtype=np.uint8); borderb[:] = [255,255,255]
    borders = np.ones((vmap.shape[0],border_width,3),dtype=np.uint8); borders[:] = [255,255,255]
    borders2 = np.ones((umap.shape[0],border_width,3),dtype=np.uint8); borders2[:] = [255,255,255]

    # pt1 = np.concatenate((np.copy(_img), vmap), axis=1)
    # pt2 = np.concatenate((umap,blank),axis=1)

    # numap = cv2.applyColorMap(umap,cv2.COLORMAP_JET)
    # nvmap = cv2.applyColorMap(vmap,cv2.COLORMAP_JET)
    numap = cv2.applyColorMap(umap,cv2.COLORMAP_PARULA)
    nvmap = cv2.applyColorMap(vmap,cv2.COLORMAP_PARULA)
    # print(img.shape,borders.shape, nvmap.shape)
    # print(numap.shape,borders2.shape, blank.shape)
    pt1 = np.concatenate((img,borders, nvmap), axis=1)
    pt2 = np.concatenate((numap,borders2,blank),axis=1)
    overlay = np.concatenate((pt1,borderb,pt2),axis=0)
    # overlay = cv2.cvtColor(overlay,cv2.COLOR_GRAY2BGR)
    return overlay

def umapping(_img, nBins=255, use_normalized=False, use_alternate_mapping=True, mask_deadzones=True, deadzone=5, verbose=True):
    img = np.copy(_img);                       imMean = np.mean(img);      imStd = np.std(img)
    norm = (img - np.mean(img))/np.std(img);   normMean = np.mean(norm);   normStd = np.std(norm)
    h, w = norm.shape[:2]

    dmin = np.min(img);        dmax = np.max(img)
    dminN = np.min(norm);      dmaxN = np.max(norm)

    if(use_normalized):
        dmap = norm
        ds = np.linspace(dminN, dmaxN,nBins+1)
        dLims = (dminN,dmaxN)
    elif(not use_normalized and use_alternate_mapping):
        dmap = img
        ds = 0; dLims = (0,1)
    else:
        dmap = norm
        ds = np.linspace(dmin, dmax,nBins+1)
        dLims = (dmin,dmax)

    umap = np.zeros((nBins,w), dtype=np.uint8)
    for col in range(w):
        data = dmap[:,col]
        if(mask_deadzones):
            data[:deadzone] = 0
            data[-deadzone:] = 0
        if(not use_normalized and use_alternate_mapping):
            mu = np.mean(data)
            std = np.std(data)
            if(std == 0.0): std = 1
            data = (data - mu)/float(std)
            mx = np.max(data)
            mn = np.min(data)
            # print("U-Map Scan [%d] --- Min, Max, Mean, Std: %.3f, %.3f, %.3f, %.3f" % (col, mn, mx, mu, std) )
            # urow, edges = np.histogram(data,nBins,(dminN,dmaxN))
            urow, edges = np.histogram(data,nBins)
        else:
            urow, edges = np.histogram(data,nBins, dLims)
        umap[:,col] = urow

    if(verbose):
        print("""------------------------------------\r\n[U-Mapping] Input Image (%d x %d):\r\n------------------------------------
   * Un-Normalized Disparity Limits      = (%d, %d)
   * Normalized Disparity Limits         = (%.3f, %.3f)
   * U-Map Disparity Limits              = (%.3f, %.3f)
   * Un-Normalized Statistics   (%s, std) = (%.3f, %.3f)
   * Normalized Statistics      (%s, std) = (%.3f, %.3f)
   * U-Map Statistics           (%s, std) = (%.3f, %.3f)"""
             % (h,w,dmin,dmax,dminN,dmaxN, np.min(umap),np.max(umap), u"\u03BC",imMean,imStd,u"\u03BC",normMean,normStd,u"\u03BC",np.mean(umap),np.std(umap)) )

    return umap

def vmapping(_img, nBins=255, use_normalized=False, use_alternate_mapping=True, mask_deadzones=True, deadzone=5, verbose=True):
    img = np.copy(_img);                       imMean = np.mean(img);      imStd = np.std(img)
    norm = (img - np.mean(img))/np.std(img);   normMean = np.mean(norm);   normStd = np.std(norm)
    h, w = norm.shape[:2]

    dmin = np.min(img);        dmax = np.max(img)
    dminN = np.min(norm);      dmaxN = np.max(norm)

    if(use_normalized):
        dmap = norm
        ds = np.linspace(dminN, dmaxN,nBins+1)
        dLims = (dminN,dmaxN)
    elif(not use_normalized and use_alternate_mapping):
        dmap = img
        ds = 0; dLims = (0,1)
    else:
        dmap = norm
        ds = np.linspace(dmin, dmax,nBins+1)
        dLims = (dmin,dmax)

    vmap = np.zeros((h,nBins), dtype=np.uint8)
    for row in range(h):
        data = dmap[row,:]
        if(mask_deadzones):
            data[:deadzone] = 0
            data[-deadzone:] = 0
        if(not use_normalized and use_alternate_mapping):
            mu = np.mean(data)
            std = np.std(data)
            # if(std == 0.0): std = 1
            data = (data - mu)/float(std)
            mx = np.max(data)
            mn = np.min(data)
            # print("V-Map Scan [%d] --- Min, Max, Mean, Std: %.3f, %.3f, %.3f, %.3f" % (row, mn, mx, mu, std) )
            ds = np.linspace(dminN, dmaxN,nBins+1)
            # vcol, edges = np.histogram(data,nBins,(dminN,dmaxN))#,ds,(dmin,dmax))
            vcol, edges = np.histogram(data,nBins)
        else:
            vcol, edges = np.histogram(data,nBins, dLims)
        vmap[row,:] = vcol

    if(verbose):
        print("""------------------------------------\r\n[V-Mapping] Input Image (%d x %d):\r\n------------------------------------
   * Un-Normalized Disparity Limits      = (%d, %d)
   * Normalized Disparity Limits         = (%.3f, %.3f)
   * V-Map Disparity Limits              = (%.3f, %.3f)
   * Un-Normalized Statistics   (%s, std) = (%.3f, %.3f)
   * Normalized Statistics      (%s, std) = (%.3f, %.3f)
   * V-Map Statistics           (%s, std) = (%.3f, %.3f)"""
             % (h,w,dmin,dmax,dminN,dmaxN, np.min(vmap),np.max(vmap), u"\u03BC",imMean,imStd,u"\u03BC",normMean,normStd,u"\u03BC",np.mean(vmap),np.std(vmap)) )

    return vmap

"""
============================================================================
	Create the UV Disparity Mappings from a given depth (disparity) image
============================================================================
"""
def get_uv_map(img, verbose=False, timing=False):
	dt = 0
	if(timing): t0 = time.time()
	dmax = np.max(img) + 1

	# Determine stats for U and V map images
	h, w = img.shape[:2]
	hu, wu = dmax, w
	hv, wv = h, dmax
	histRange = (0,dmax)

	if(verbose):
		print("[UV Mapping] Input Image Size: (%d, %d) --- w/ max disparity = %.3f" % (h,w, dmax))
		print("[UV Mapping] Disparity Map Sizes --- U-Map (%.2f, %.2f) ||  V-Map (%.2f, %.2f)" % (hu, wu, hv, wv))

	umap = np.zeros((dmax,w,1), dtype=np.uint8)
	vmap = np.zeros((h,dmax,1), dtype=np.uint8)

	for i in range(0,w):
		uscan = img[:,i]
		urow = cv2.calcHist([uscan],[0],None,[dmax],histRange)
		if(verbose): print("\t[U Mapping] Scan[%d] (%s) ---- Scan Histogram (%s)" % (i,', '.join(map(str, uscan.shape)), ', '.join(map(str, urow.shape))))
		umap[:,i] = urow

	for i in range(0,h):
		vscan = img[i,:]
		vrow = cv2.calcHist([vscan],[0],None,[dmax],histRange)
		if(verbose): print("\t[V Mapping] Scan [%d] (%s) ---- Scan Histogram (%s)" % (i,', '.join(map(str, vscan.shape)), ', '.join(map(str, vrow.shape))))
		vmap[i,:] = vrow

	umap = np.reshape(umap,(dmax,w))
	vmap = np.reshape(vmap,(h,dmax))

	if(verbose): print("\t[UV Mapping] U Map = (%s) ----- V Map = (%s)" % (', '.join(map(str, umap.shape)),', '.join(map(str, vmap.shape)) ))

	if(timing):
		t1 = time.time()
		dt = t1 - t0
		print("\t[UV Mapping] --- Took %f seconds to complete" % (dt))
	return umap,vmap, dt

"""
============================================================================
	Attempt to find the horizontal bounds for detected contours
============================================================================
"""
def extract_contour_bounds(cnts, verbose=False, timing=False):
	dt = 0
	xBounds = []
	disparityBounds = []

	if(timing): t0 = time.time()
	for cnt in cnts:
		try:
			x,y,rectw,recth = cv2.boundingRect(cnt)
			xBounds.append([x, x + rectw])
			disparityBounds.append([y, y + recth])
		except: pass

	if(timing):
		t1 = time.time()
		dt = t1 - t0
		print("\t[extract_contour_bounds] --- Took %f seconds to complete" % (dt))

	return xBounds, disparityBounds, dt

"""
============================================================================
	Attempt to find the y boundaries for potential obstacles found from
	the U-Map
============================================================================
"""
def obstacle_search(_vmap, x_limits, pixel_thresholds=(1,30), window_size=None, verbose=False, timing=False):
	flag_done = False
	flag_found_start = False
	count = 0; nWindows = 0; dt = 0
	yLims = []; windows = []

	try: img = cv2.cvtColor(_vmap,cv2.COLOR_GRAY2BGR)
	except:
		img = np.copy(_vmap)
		print("[WARNING] obstacle_search ------------  Unnecessary Image Color Converting")
	# Limits
	h,w = img.shape[:2]
	xmin, xmax = x_limits
	pxlMin, pxlMax = pixel_thresholds
	# Starting Pixel Coordinate
	yk = prev_yk = 0
	xk = (xmax + xmin) / 2
	# Window Size
	if(window_size is None):
		dWy = 20
		dWx = abs(xk - xmin)
		if(dWx <= 1): dWx = 2
	else: dWx, dWy = np.int32(window_size)/2

	if(yk <= 0): yk = 0 + dWy
	if(verbose): print("Starting Location: (%d, %d)" % (xk, yk) )

	if(timing): t0 = time.time()

	# Grab all nonzero pixels
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Begin Sliding Window Search Technique
	while(not flag_done):
		if(xk >= w): # If we are at image edges we must stop
			flag_done = True
			if(verbose): print("Exiting: Reached max image width.")
		elif(yk + dWy >= h):
			flag_done = True
			if(verbose): print("Exiting: Reached max image height.")

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

		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
					(nonzerox >= wx_low) &  (nonzerox < wx_high)).nonzero()[0]
		nPxls = len(good_inds)
		if(verbose):
			print("Current Window [" + str(count) + "] ----- Center: " + str(xk) + ", " + str(yk) + " ----- # of good pixels = "  + str(nPxls))

		# Record mean coordinates of pixels in window and update new window center
		if(nPxls >= pxlMax):
			if(nWindows == 0):
				yLims.append(yk - dWy)
			windows.append([(wx_low,wy_high),(wx_high,wy_low)])
			nWindows += 1
			prev_yk = yk
			flag_found_start = True
		elif(nPxls <= pxlMin and flag_found_start):
			flag_done = True
			yLims.append(prev_yk + dWy)
		# Update New window center coordinates
		xk = xk
		yk = yk + 2*dWy
		count += 1

	if(timing):
		t1 = time.time()
		dt = t1 - t0
		print("\t[obstacle_search] --- Took %f seconds to complete" % (dt))
	return yLims, windows, dt

"""
===============================================================================
===============================================================================
===============================================================================
===============================================================================
"""

def find_obstacles(vmap, dLims, xLims, search_thresholds = (3,30), verbose=False):
	obs = []; windows = []
	nObs = len(dLims)
	for i in range(nObs):
		xs = xLims[i]
		ds = dLims[i]
		ys,ws,_ = obstacle_search(vmap, ds, search_thresholds)
		if(len(ys) <= 0):
			if(verbose): print("[INFO] Found obstacle with zero height. Skipping...")
		elif(ys[0] == ys[1]):
			if(verbose): print("[INFO] Found obstacle with zero height. Skipping...")
		else:
			obs.append([
				(xs[0],ys[0]),
				(xs[1],ys[1])
			])
			windows.append(ws)
	return obs, len(obs), windows

"""
===============================================================================
===============================================================================
===============================================================================
===============================================================================
"""

def find_contours(_umap, threshold = 30.0, threshold_method = "perimeter", offset=(0,0), debug=False):
	try: umap = cv2.cvtColor(_umap,cv2.COLOR_BGR2GRAY)
	except:
		umap = _umap
		print("[WARNING] find_contours --- Unnecessary Image Color Converting")

	_, contours, hierarchy = cv2.findContours(umap,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,offset=offset)

	if(threshold_method == "perimeter"):
		filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt,True) >= threshold]
		if(debug):
			raw_perimeters = [cv2.arcLength(cnt,True) for cnt in contours]
			filtered_perimeters = [cv2.arcLength(cnt,True) for cnt in filtered_contours]
			print("Raw Contour Perimeters:",raw_perimeters)
			print("Filtered Contour Perimeters:",filtered_perimeters)
	elif(threshold_method == "area"):
		filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= threshold]
		if(debug):
			raw_areas = [cv2.contourArea(cnt) for cnt in contours]
			filtered_areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
			print("Raw Contour Areas:",raw_areas)
			print("Filtered Contour Areas:",filtered_areas)
	else:
		print("[ERROR] find_contours --- Unsupported filtering method!")

	return filtered_contours

"""
===============================================================================
===============================================================================
===============================================================================
===============================================================================
"""

"""
============================================================================
	Abstract a mask image for filtering out the ground from a V-Map
============================================================================
"""
def get_vmap_mask(vmap, threshold=20, maxStep=14, deltas=(0,20), mask_size = [10,30], window_size = [10,30], draw_method=1, verbose=False, timing=False):
	flag_done = False
	count = 0 ; dt = 0
	good_inds = []; mean_pxls = []; windows = []; masks = []
	# Sizes
	h,w = vmap.shape[:2]
	dWy,dWx = np.int32(window_size)/2
	dMy,dMx = np.int32(mask_size)/2
	dx,dy = np.int32(deltas)
	# ==========================================================================

	# Create a black template to create mask with
	black = np.zeros((h,w,3),dtype=np.uint8)

	if(timing): t0 = time.time()
	# ==========================================================================

	# Take the bottom strip of the input image to find potential points to start sliding from
	y0 = abs(int(h-dy))
	hist = np.sum(vmap[y0:h,0:w], axis=0)
	x0 = abs(int(np.argmax(hist[:,0])))
	# ==========================================================================

	# Prevent initial search coordinates from clipping search window at edges
	if(x0 <= dWx): xk = dWx
	else: xk = x0
	if(y0 >= h): yk = h - dWy
	else: yk = y0
	if(verbose): print("[get_vmap_mask] --- Starting Location: ", xk, yk)

	# ==========================================================================

	# Grab all nonzero pixels
	nonzero = vmap.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Begin Sliding Window Search Technique
	while(count <= maxStep and not flag_done):
		# TODO: Modify search window width depending on the current disparity
		if(xk >= w/2): dWx = dWx
		elif(xk >= w):  # If we are at image width we must stop
			flag_done = True
			if(verbose): print("Exiting: Reached max image width.")

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

		windows.append([ (wx_low,wy_high),(wx_high,wy_low) ])

		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
					(nonzerox >= wx_low) &  (nonzerox < wx_high)).nonzero()[0]
		nPxls = len(good_inds)
		if verbose == True:
			print("\tCurrent Window [" + str(count) + "] Center: " + str(xk) + ", " + str(yk) + " ----- # Good Pixels = " + str(nPxls))

		# Record mean coordinates of pixels in window and update new window center
		if(nPxls >= threshold):
			xmean = np.int(np.mean(nonzerox[good_inds]))
			ymean = np.int(np.mean(nonzeroy[good_inds]))
			mean_pxls.append(np.array([xmean,ymean]))

			# Draw current window onto mask template
			my_low = ymean - dMy;          	my_high = ymean + dMy
			mx_low = xmean - dMx; 			mx_high = xmean + dMx
			masks.append([ (mx_low,my_high), (mx_high,my_low) ])
			# Update New window center coordinates
			xk = xmean + dWx
			yk = ymean - 2*dWy
		else: flag_done = True
		count += 1
	mean_pxls = np.array(mean_pxls)

	# ==========================================================================
	if(draw_method == 1):
		# ploty = np.linspace(0, mean_pxls[-1,0])
		# try: # Try fitting polynomial nonzero data
		# 	fit = np.poly1d(np.polyfit(mean_pxls[:,0],mean_pxls[:,1], 3))
		# 	plotx = fit(ploty) # Generate x and y values for plotting
		# except:
		# 	print("ERROR: Function 'polyfit' failed!")
		# 	fit = [0, 0]; plotx = [0, 0]
		# xs = np.asarray(ploty,dtype=np.int32)
		# ys = np.asarray(plotx,dtype=np.int32)
		# pts = np.vstack(([xs],[ys])).transpose()

		pts = cv2.approxPolyDP(mean_pxls,3,0)
		# cv2.polylines(tmpVmap, [vs], 0, (255,0,255),20)
		cv2.polylines(black, [pts], 0, (255,255,255),20)
	else:
		for mask in masks:
			cv2.rectangle(black,mask[0],mask[1],(255,255,255), cv2.FILLED)
	# ==========================================================================
	cv2.rectangle(black,(0,0),(dx,h),(255,255,255), cv2.FILLED)

	# ==========================================================================
	mask = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)
	mask_inv = cv2.bitwise_not(mask)

	if(timing):
		t1 = time.time()
		dt = t1 - t0
		print("\t[get_vmap_mask] --- Took %f seconds to complete" % (dt))

	return mask, mask_inv, mean_pxls, windows, dt

"""
===============================================================================
===============================================================================
===============================================================================
===============================================================================
"""

def uv_pipeline(_img, timing=True, threshU1=7, threshU2=20, flag_displays=False):
    dt = 0
    if(timing): t0 = time.time()
    img = cv2.imread(_img, cv2.IMREAD_GRAYSCALE)
    kernelI = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelI)

    h, w = img.shape[:2]
    dead_x = 2; dead_y = 10

    raw_umap, raw_vmap, dt = get_uv_map(img)
    # =========================================================================
    deadzoneU = raw_umap[1:dead_y+1, :]
    _, deadzoneU = cv2.threshold(deadzoneU, 95, 255,cv2.THRESH_BINARY)

    cv2.rectangle(raw_umap,(0,0),(raw_umap.shape[1],dead_y),(0,0,0), cv2.FILLED)
    cv2.rectangle(raw_umap,(0,raw_umap.shape[0]-dead_y),(raw_umap.shape[1],raw_umap.shape[0]),(0,0,0), cv2.FILLED)

    cv2.rectangle(raw_vmap,(0,0),(dead_x, raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
    cv2.rectangle(raw_vmap,(raw_vmap.shape[1]-dead_x,0),(raw_vmap.shape[1],raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
    # =========================================================================
    umap0 = np.copy(raw_umap)
    vmap0 = np.copy(raw_vmap)
    try:
        raw_umap = cv2.cvtColor(raw_umap,cv2.COLOR_GRAY2BGR)
        raw_vmap = cv2.cvtColor(raw_vmap,cv2.COLOR_GRAY2BGR)
    except:
        print("[WARNING] ------------  Unnecessary Raw Mappings Color Converting")
    # ==========================================================================
    #							U-MAP Specific Functions
    # ==========================================================================

    stripsU = strip_image(raw_umap, nstrips=6)
    _, stripU1 = cv2.threshold(stripsU[0], threshU1, 255,cv2.THRESH_BINARY)
    _, stripU2 = cv2.threshold(stripsU[1], threshU2, 255,cv2.THRESH_BINARY)
    _, stripU3 = cv2.threshold(stripsU[2], 30, 255,cv2.THRESH_BINARY)
    _, stripU4 = cv2.threshold(stripsU[3], 40, 255,cv2.THRESH_BINARY)
    _, stripU5 = cv2.threshold(stripsU[4], 40, 255,cv2.THRESH_BINARY)
    _, stripU6 = cv2.threshold(stripsU[5], 40, 255,cv2.THRESH_BINARY)

    hUs = stripU1.shape[0]
    blankStrip = np.zeros((hUs-dead_y,w),dtype=np.uint8)
    deadzone_mask = np.concatenate((deadzoneU,blankStrip), axis=0)
    try: deadzone_mask = cv2.cvtColor(deadzone_mask, cv2.COLOR_GRAY2BGR)
    except: print("[WARNING] ------------  Unnecessary Deadzone Image Color Converting")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
    stripU1 = cv2.morphologyEx(stripU1, cv2.MORPH_CLOSE, kernel)
    stripU2 = cv2.morphologyEx(stripU2, cv2.MORPH_CLOSE, kernel)
    stripU3 = cv2.morphologyEx(stripU3, cv2.MORPH_OPEN, kernel)

    kernelD = cv2.getStructuringElement(cv2.MORPH_RECT,(50,5))
    deadzone_mask = cv2.morphologyEx(deadzone_mask, cv2.MORPH_CLOSE, kernelD)
    stripU1 = cv2.addWeighted(stripU1, 1.0, deadzone_mask, 1.0, 0)
    # ==========================================================================
    fCnts1 = find_contours(stripU1, 55.0, offset=(0,0))
    fCnts2 = find_contours(stripU2, 100.0, offset=(0,hUs))
    fCnts3 = find_contours(stripU3, 80.0, offset=(0,hUs*2))
    fCnts4 = find_contours(stripU4, 40.0, offset=(0,hUs*3))
    fCnts5 = find_contours(stripU5, 40.0, offset=(0,hUs*4))
    fCnts6 = find_contours(stripU6, 40.0, offset=(0,hUs*5))
    filtered_contours = fCnts1 + fCnts2 + fCnts3 + fCnts4 + fCnts5 + fCnts6
    xLims, dLims, _ = extract_contour_bounds(filtered_contours)

    # ==========================================================================
    #							V-MAP Specific Functions
    # ==========================================================================
    stripsV = strip_image(raw_vmap, nstrips=5, horizontal_strips=False)
    print(np.max(stripsV[1]))

    _, stripV1 = cv2.threshold(stripsV[0], 9, 255,cv2.THRESH_BINARY)
    _, stripV2 = cv2.threshold(stripsV[1], 50, 255,cv2.THRESH_BINARY)
    _, stripV3 = cv2.threshold(stripsV[2], 40, 255,cv2.THRESH_BINARY)
    _, stripV4 = cv2.threshold(stripsV[3], 40, 255,cv2.THRESH_BINARY)
    _, stripV5 = cv2.threshold(stripsV[4], 40, 255,cv2.THRESH_BINARY)
    # newV = np.concatenate((stripV1,stripV2,stripV3,stripV4), axis=1)
    newV = np.concatenate((stripV1,stripV2,stripV3,stripV4,stripV5), axis=1)

    mask, maskInv,mPxls, ground_wins,_ = get_vmap_mask(newV, maxStep = 15)
    vmap = cv2.bitwise_and(newV,newV,mask=maskInv)

    # =========================================================================
    # umap = np.concatenate((stripU1,stripU2,stripU3,stripU4), axis=0)
    umap = np.concatenate((stripU1,stripU2,stripU3,stripU4,stripU5,stripU6), axis=0)
    try: umap = cv2.cvtColor(umap, cv2.COLOR_BGR2GRAY)
    except: print("[WARNING] ------------  Unnecessary Umap Color Converting")
    try: vmap = cv2.cvtColor(vmap, cv2.COLOR_BGR2GRAY)
    except: print("[WARNING] ------------  Unnecessary Vmap Color Converting")

    # ==========================================================================
    print("[INFO] Beginning Obstacle Search....")
    obs, nObs, windows = find_obstacles(vmap, dLims, xLims)
    # =========================================================================
    if(timing):
        t1 = time.time()
        dt = t1 - t0
        print("\t[uv_pipeline] --- Took %f seconds to complete" % (dt))

    if(flag_displays):
        color = (255,0,255)
        borderb = np.ones((1,w,3),dtype=np.uint8); borderb[:] = color
        borders = np.ones((h,1,3),dtype=np.uint8); borders[:] = color
        # newUdisp = np.concatenate((stripU1,borderb,stripU2,borderb,stripU3,borderb,stripU4), axis=0)
        # newVdisp = np.concatenate((stripV1,borders,stripV2,borders,stripV3,borders,stripV4), axis=1)
        newUdisp = np.concatenate((stripU1,borderb,stripU2,borderb,stripU3,borderb,stripU4,borderb,stripU5,borderb,stripU6), axis=0)
        newVdisp = np.concatenate((stripV1,borders,stripV2,borders,stripV3,borders,stripV4,borders,stripV5), axis=1)

        disp_cnts = cv2.cvtColor(np.copy(umap), cv2.COLOR_GRAY2BGR)
        for cnt in filtered_contours:
            cv2.drawContours(disp_cnts, [cnt], 0, (255,0,0), 2)

        disp_wins = cv2.cvtColor(np.copy(vmap), cv2.COLOR_GRAY2BGR)
        for wins in windows:
            for win in wins:
                cv2.rectangle(disp_wins,win[0],win[1],(255,255,0), 1)

        for pxl in mPxls:
            cv2.circle(disp_wins,(pxl[0],pxl[1]),2,(255,0,255),-1)

        disp_ground = cv2.cvtColor(np.copy(vmap), cv2.COLOR_GRAY2BGR)
        for win in ground_wins:
            cv2.rectangle(disp_ground,win[0],win[1],(255,255,0), 1)

        plt.figure(1)
        plt.imshow(umap0)
        plt.show()

        plt.figure(2)
        plt.imshow(newVdisp)
        plt.show()

        plt.figure(3)
        plt.imshow(newUdisp)
        plt.show()

        plt.figure(4)
        plt.imshow(disp_cnts)
        plt.show()

        plt.figure(5)
        plt.imshow(disp_wins)
        plt.show()

        plt.figure(6)
        plt.imshow(disp_ground)
        plt.show()

    return umap, vmap, img, obs, nObs, mPxls,newV, mask, maskInv, dt
