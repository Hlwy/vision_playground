#!/usr/bin/env python

import cv2
import rospy
import cython
import numpy as np
import os, csv, time, math
# from sklearn import linear_model
from matplotlib import pyplot as plt

from sensor_msgs.msg import Image,CompressedImage
from cv_bridge import CvBridge, CvBridgeError

from VBOATS import VBOATS
from devices.d415_camera import *
from hyutils.img_utils import *
from hyutils.uv_mapping_utils import *

# @cython.boundscheck(False)
# cpdef unsigned char[:, :] hough_line_fast(unsigned char [:, :]accumulator,long [:] x_idxs,long [:]y_idxs, double [:] cos_t,double [:] sin_t, long diag_len,long num_thetas):
# #     global diag_len,num_thetas,cos_t,sin_t
#     cdef int i, x, y, rho
#     for i in range(len(x_idxs)):
#         x = x_idxs[i]
#         y = y_idxs[i]
#
#         for t_idx in range(num_thetas):
#             # Calculate rho. diag_len is added for a positive index
#             rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
#             accumulator[rho, t_idx] += 1
#
#     return accumulator

def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos
def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(accumulator, cmap='jet',extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None: plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# def estimateGndLineCoeffs(filtered_vmap, stopP = 0.85, maxTrials=30,verbose = False,show_plots=False):
#     gndImg = np.copy(filtered_vmap)
#     nonzero = gndImg.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     nonzerox = np.reshape(nonzerox,(nonzerox.shape[0],1))
#
#     ransac = linear_model.RANSACRegressor(stop_probability=stopP,max_trials=maxTrials)
#     ransac.fit(nonzerox, nonzeroy)
#     inlier_mask = ransac.inlier_mask_
#     outlier_mask = np.logical_not(inlier_mask)
#
#     line_X = np.arange(nonzerox.min(), nonzerox.max())[:, np.newaxis]
#     line_y_ransac = ransac.predict(line_X)
#
#     if(show_plots):
#         plt.figure("RANSAC")
#         plt.clf()
#         plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.0,right=1.0,top=1.0, bottom=0.0)
#         plt.axis([0.0, 255.0, 480, 0])
#         plt.imshow(gndImg,interpolation='bilinear')
#         plt.scatter(nonzerox,nonzeroy, color='yellowgreen', marker='.',label='Inliers')
#         plt.plot(line_X, line_y_ransac, color='red', linewidth=2, label='RANSAC regressor')
#         plt.show()
#
#     m = ransac.estimator_.coef_
#     b = ransac.estimator_.intercept_
#     if(verbose): print("Estimated coefficients (RANSAC): m = %.3f | b = %.3f" % (m,b))
#     return m,b, line_X,line_y_ransac

def isGndPresent(found_lines,minDeg=-89.0, maxDeg=-26.0):
    cnt = 0
    summer = 0
    flag = False
    avgTheta = 0.0
    minAng = np.deg2rad(minDeg)
    maxAng = np.deg2rad(maxDeg)
    angs = found_lines[:,0,1]
    for theta in angs:
        angle = theta-np.pi
        if(minAng <= angle <= maxAng):
            summer += angle
            cnt += 1
    if(cnt > 0):
        avgTheta = summer/float(cnt)
        flag = True
    return flag, avgTheta

def estimateGndLineCoeffsHough(filtered_vmap,hough_thresh=100, deg_offset=2.0):
    sets = []
    # print(filtered_vmap.shape)
    lines = cv2.HoughLines(filtered_vmap,1,np.pi/180, hough_thresh)

    # if(lines is None): lines = cv2.HoughLines(filtered_vmap,1,np.pi/180, 50)
    if(lines is None):
        print("couldnt find hough lines")
        return sets

    validGnd, avgTheta = isGndPresent(lines)
    if(validGnd):
        avgAngle = np.rad2deg(avgTheta)
        minAng = np.deg2rad(avgAngle - deg_offset)
        maxAng = np.deg2rad(avgAngle + deg_offset)
        h,w = filtered_vmap.shape[:2]
        for i in range(0,len(lines)):
            for r,theta in lines[i]:
                angle = theta-np.pi
                if(minAng <= angle <= maxAng):
                    a = np.cos(theta);            b = np.sin(theta)
                    x0 = a*r;                     y0 = b*r;
                    x1 = int(x0 + 1000*(-b));     y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b));     y2 = int(y0 - 1000*(a))
                    pt1 = (x1,y1)
                    pt2 = (x2,y2)
                    pts = [(0,h),pt1,pt2,(w, h),(0, h)]
                    pts = np.array(pts)
                    sets.append(pts)
    else: print("No Ground Found")
    return sets

def estimateGndLineCoeffsHough2(filtered_vmap,hough_thresh=100, deg_offset=2.0):
    sets = []
    validGnd = False
    bestM = 0.0
    bestIntercept = 0
    # print(filtered_vmap.shape)
    lines = cv2.HoughLines(filtered_vmap,1,np.pi/180, hough_thresh)

    # if(lines is None): lines = cv2.HoughLines(filtered_vmap,1,np.pi/180, 50)
    if(lines is None):
        print("couldnt find hough lines")
        return validGnd,bestM,bestIntercept, sets

    validGnd, avgTheta = isGndPresent(lines)
    if(validGnd):
        avgAngle = np.rad2deg(avgTheta)
        minAng = np.deg2rad(avgAngle - deg_offset)
        maxAng = np.deg2rad(avgAngle + deg_offset)
        h,w = filtered_vmap.shape[:2]
        for i in range(0,len(lines)):
            for r,theta in lines[i]:
                angle = theta-np.pi
                if(minAng <= angle <= maxAng):
                    a = np.cos(theta);            b = np.sin(theta)
                    x0 = a*r;                     y0 = b*r;
                    x1 = int(x0 + 1000*(-b));     y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b));     y2 = int(y0 - 1000*(a))
                    m = float(y2 - y1)/float(x2 - x1)
                    b = int(y1 - m * x1)
                    if(b > bestIntercept):
                        bestIntercept = b
                        bestM = m
    else: print("No Ground Found")
    return validGnd,bestM,bestIntercept, sets

class vboat_testing_node:
    def __init__(self):
        rospy.init_node('vboat_testing_node')
        rootDir = os.path.dirname(os.path.abspath(__file__))
        imgsPath = os.path.abspath(os.path.join(rootDir, "..")) + "/images"

        self.fps = rospy.get_param('~fps', 30)
        self.save_path = rospy.get_param('~save_path',imgsPath)
        self.flag_save_imgs = rospy.get_param('~flag_save_imgs', True)
        self.flag_show_imgs = rospy.get_param('~flag_show_imgs', True)

        self.cam_tf = rospy.get_param('~cam_tf', "d415_camera_depth_optical_frame")
        self.image_topic = rospy.get_param('~image_topic', "d415/depth/image_raw")

        # self.dist_pub = rospy.Publisher(rospy.get_namespace()+"/obstacles/distances",FloatArrayStamped,queue_size=1000)
        # self.ang_pub = rospy.Publisher(rospy.get_namespace()+"/obstacles/angles",FloatArrayStamped,queue_size=1000)

        self.image_pub = rospy.Publisher("vboat/image",Image,queue_size=1000)
        self.bridge = CvBridge()

        self.cam = CameraD415(flag_save=False,use_statistics=False,fps=30, verbose=True)

        self.intr = self.cam.get_intrinsics()
        self.extr = self.cam.get_extrinsics()
        self.focal = [self.intr["depth"].fx, self.intr["depth"].fy]
        self.ppoint = [self.intr["depth"].ppx, self.intr["depth"].ppy]
        self.baseline = self.extr.translation[0]

        self.vboat = VBOATS()
        self.vboat.dead_x = 3
        self.vboat.dead_y = 3

        self.r = rospy.Rate(60)
        self.rgb = []
        self.depth = []
        self.disparity = []
        self.umap = []
        self.vmap = []

        self.camcount = 0
        self.count = 0
        self.disp_obs = None
        self.prevM = None
        self.prevB = None

        if self.flag_save_imgs:
            self.rgbDir = os.path.join(self.save_path, "rgb")
            self.depthDir = os.path.join(self.save_path, "depth")
            self.obsDir = os.path.join(self.save_path, "processed")
            dirs = [self.save_path,self.rgbDir, self.depthDir, self.obsDir]
            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)
                    print("Created directory \'%s\' " % dir)
                else: print("Directory \'%s\' already exists" % dir)
        print("[INFO] vboat_testing_node --- Started...")

    def camCallback(self, _rgb, _depth):
        tmp = _depth/65535.0
        ratio = np.max(_depth)/65535.0
        depth = np.uint8(tmp*255)
        depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)

        tmp2 = _depth*0.001
        loc = np.where(tmp2 == 0.0)
        tmp2[loc] = 1.0
        disparity = (self.focal[0]*self.baseline)/tmp2
        disparity[loc] = 0.0
        self.disparity2uintGain = (255)/np.max(disparity)
        disparity = np.uint8(disparity*self.disparity2uintGain)

        self.rgb = np.copy(_rgb)
        self.depth = np.copy(depth)
        self.disparity = np.copy(disparity)
        self.camcount+=1

    def draw_obstacle_image(self):
        display_obstacles = cv2.cvtColor(self.vboat.img, cv2.COLOR_GRAY2BGR)
        [cv2.rectangle(display_obstacles,ob[0],ob[1],(150,0,0),1) for ob in self.vboat.obstacles]
        disp_obs = np.copy(display_obstacles)
        return disp_obs

    def save_image_to_file(self, img, path):
        try: cv2.imwrite(path, img)
        except:
            print("[ERROR] vboat_node.py ---- Could not save image to file \'%s\'" % path)
            pass

    def filterVmap(self, _vmap, thresholds = [0.85,0.85,0.75,0.5], verbose=False, debug=False):
        raw_vmap = np.copy(_vmap)
        vMax = np.max(_vmap)
        nThreshs = int(math.ceil((vMax/256.0)) * len(thresholds))
        if(debug): print("vMax, nThreshs" % (vMax, nThreshs))

        # Rough Theshold top of vmap heavy prior
        h, w = raw_vmap.shape[:2]
        dh = h / 3
        topV = raw_vmap[0:dh, :]
        botV = raw_vmap[dh:h, :]
        tmpThresh = int(np.max(topV)*0.05)
        _,topV = cv2.threshold(topV, tmpThresh,255,cv2.THRESH_TOZERO)
        preVmap = np.concatenate((topV,botV), axis=0)

        stripsPV = []
        stripsV = strip_image(preVmap, nstrips=nThreshs, horizontal_strips=False)
        for i, strip in enumerate(stripsV):
            tmpMax = np.max(strip)
            tmpMean = np.mean(strip)
            tmpStd = np.std(strip)
            if(tmpMean == 0):
                stripsPV.append(strip)
                continue
            if(verbose): print("---------- [Strip %d] ---------- \r\n\tMax = %.1f, Mean = %.1f, Std = %.1f" % (i,tmpMax,tmpMean,tmpStd))
            dratio = vMax/255.0
            relRatio = (tmpMax-tmpStd)/float(vMax)
            rrelRatio = (tmpMean)/float(tmpMax)
            if(verbose): print("\tRatios: %.3f, %.3f, %.3f" % (dratio, relRatio, rrelRatio))
            if(relRatio >= 0.4): gain = relRatio + rrelRatio
            else: gain = 1.0 - (relRatio + rrelRatio)
            thresh = int(thresholds[i]* gain * tmpMax)
            if(verbose): print("\tGain = %.2f, Thresh = %d" % (gain,thresh))
            _, tmpStrip = cv2.threshold(strip, thresh,255,cv2.THRESH_TOZERO)
            # _, tmpStrip = cv2.threshold(strip, thresh,255,cv2.THRESH_BINARY)
            stripsPV.append(tmpStrip)

        filtV = np.concatenate(stripsPV, axis=1)
        return filtV, stripsV, stripsPV

    def filterUmap(self, _umap, thresholds = [0.85,0.85,0.75,0.5], verbose=False, debug=False):
        raw_umap = np.copy(_umap)
        uMax = np.max(_umap)
        nThreshs = int(math.ceil((uMax/256.0)) * len(thresholds))
        if(debug): print("uMax, nThreshs" % (uMax, nThreshs))

        # Rough Theshold top of vmap heavy prior
        _,preUmap = cv2.threshold(raw_umap, 2,255,cv2.THRESH_TOZERO)

        stripsPU = []
        stripsU = strip_image(preUmap, nstrips=nThreshs)
        for i, strip in enumerate(stripsU):
            tmpMax = np.max(strip)
            tmpMean = np.mean(strip)
            tmpStd = np.std(strip)
            if(tmpMean == 0):
                stripsPU.append(strip)
                continue
            if(verbose): print("---------- [Strip %d] ---------- \r\n\tMax = %.1f, Mean = %.1f, Std = %.1f" % (i,tmpMax,tmpMean,tmpStd))
            dratio = uMax/255.0
            relRatio = (tmpMax-tmpStd)/float(uMax)
            rrelRatio = (tmpMean)/float(tmpMax)
            if(verbose): print("\tRatios: %.3f, %.3f, %.3f" % (dratio, relRatio, rrelRatio))
            if(relRatio >= 0.4): gain = relRatio + rrelRatio
            else: gain = 1.0 - (relRatio + rrelRatio)
            thresh = int(thresholds[i]* gain * tmpMax)
            if(verbose): print("\tGain = %.2f, Thresh = %d" % (gain,thresh))
            _, tmpStrip = cv2.threshold(strip, thresh,255,cv2.THRESH_TOZERO)
            # _, tmpStrip = cv2.threshold(strip, thresh,255,cv2.THRESH_BINARY)
            stripsPU.append(tmpStrip)

        filtU = np.concatenate(stripsPU, axis=0)
        return filtU, stripsU, stripsPU

    def extractGndLine(self, slope, intercept, validSlopeThresh = 0.3, dslopeThresh = 2.0, dinterceptThresh = 300.0, interceptOffset = -15.0, verbose = False, debug = False):
        if(self.prevM is None): self.prevM = slope
        if(self.prevB is None): self.prevB = intercept
        if(debug): print("Previous line coefficients = %.3f, %.3f" % (self.prevM,self.prevB))
        h,w = self.vmap.shape[:2]
        validGnd = False

        # Only return a ground line if estimated slope is steep enough (i.e. try to prevent flat lines)
        if(slope >= validSlopeThresh):
            # Check for significant schanges in segmented ground line
            dSlope = math.fabs(self.prevM - slope)
            dB = math.fabs(self.prevB - intercept)
            if(debug): print("dSlope, dIntercept = %.3f, %.3f" % (dSlope,dB))
            # Filter out sudden jumps in segmented ground line, if warrented
            if((dSlope >= dslopeThresh) or (dB >= dinterceptThresh)):
                newSlope, newIntercept = self.smoothUnstableGndLine((slope,intercept),(self.prevM,self.prevB))
            else: newSlope = slope; newIntercept = intercept
            # Store coefficients for next iteration
            self.prevM = slope
            self.prevB = intercept
            # Extract ground line parameters
            yf = int(newSlope*w + newIntercept);  pt1 = (0,int(newIntercept));  pt2 = (w,int(yf+interceptOffset))
            # Create point array for removing anything below ground line (for removing noise)
            pts = np.array([(0, h),pt1,pt2,(w, h),(0, h)])
            validGnd = True
        else:
            self.prevM = None; self.prevB = None
            pts = None
            if(verbose): print("No Ground Detected.")
        return validGnd, pts

    def smoothUnstableGndLine(self, curCoeffs, prevCoeffs, kds=(0.05,0.5), verbose=False, debug=True):
        """ ----------------------
                EXPERIMENTAL
        -------------------------- """

        if(debug): print("curM, prevM: %.3f, %.3f" % (curCoeffs[0], prevCoeffs[0]))
        if(debug): print("curB, prevB: %.3f, %.3f" % (curCoeffs[1], prevCoeffs[1]))
        errorM = prevCoeffs[0] - curCoeffs[0]
        errorB = prevCoeffs[1] - curCoeffs[1]
        if(debug): print("errorM = %.3f | errorB = %.3f" % (errorM, errorB))
        dSlopes = math.fabs(curCoeffs[0] - prevCoeffs[0])
        dIntercepts = math.fabs(curCoeffs[1] - prevCoeffs[1])
        if(debug): print("dSlope = %.3f | dIntercept = %.3f" % (dSlopes, dIntercepts))

        mDGain = dSlopes * kds[0]
        bDGain = dIntercepts * kds[1]
        if(debug): print("mDGain = %.3f | bDGain = %.3f" % (mDGain,bDGain))
        newSlope = (prevCoeffs[0]) - mDGain
        newIntercept = prevCoeffs[1] - bDGain
        # newSlope = (errorM) - mDGain
        # newIntercept = errorB - bDGain
        if(debug):
            print("newSlope = %.3f | newIntercept = %.3f" % (newSlope,newIntercept))
            print("- ---------------------------------- -")
        return newSlope, newIntercept

    def removeGround(self, _vmap, _pts, debug_line_vis=True):
        img = np.copy(_vmap)
        if(len(img.shape)<3): color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        else: color = img

        cv2.fillPoly(color, [_pts], (0,0,0))
        if(debug_line_vis):
            ground_display = np.copy(img)
            if(len(ground_display.shape)>2): ground_display = cv2.cvtColor(ground_display,cv2.COLOR_BGR2GRAY)
            ground_display = cv2.applyColorMap(ground_display,cv2.COLORMAP_PARULA)
            cv2.line(ground_display,tuple(_pts[1]), tuple(_pts[2]), (0,255,255),1)
        else: ground_display = img

        if(len(color.shape)>2): vmapNoGnd = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
        else: vmapNoGnd = color

        return vmapNoGnd, ground_display

    def removeGnd2(self, _vmap, sets):
        # print(_vmap.shape)
        display = np.copy(_vmap)
        if(len(display.shape) < 3):
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        for pts in sets:
            # print(pts)
            cv2.fillPoly(display, [pts], (0,0,0))
            cv2.line(display,tuple(pts[1]), tuple(pts[2]), (0,0,0),4)

        return cv2.cvtColor(display,cv2.COLOR_BGR2GRAY)

    def obstacle_search_disparity(self, _vmap, x_limits, pixel_thresholds=(1,30), window_size=None, lineCoeffs=None, verbose=False, timing=False):
        flag_done = False
        try_last_search = False
        tried_last_resort = False
        flag_found_start = False
        flag_hit_limits = False
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

        if(lineCoeffs is not None): yf = int(xk * lineCoeffs[0] + lineCoeffs[1])
        else: yf = h

        # Window Size
        if(window_size is None):
            dWy = 10
            dWx = abs(xk - xmin)
            if(dWx <= 2):
                if xk <= 5: dWx = 1
                else:  dWx = 2
        else: dWx, dWy = np.int32(window_size)/2

        if(yk <= 0): yk = 0 + dWy
        if(verbose): print("Object Search Window [%d x %d] -- Starting Location (%d, %d)" % (dWx,dWy,xk, yk) )

        if(timing): t0 = time.time()

        # Grab all nonzero pixels
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Begin Sliding Window Search Technique
        while(not flag_done):
            if(xk >= w): # If we are at image edges we must stop
                flag_hit_limits = True
                if(verbose): print("[INFO] Reached max image width.")
            if((yk >= yf) and (lineCoeffs is not None)):
                flag_hit_limits = True
                if(verbose): print("[INFO] Reached Estimated Ground line.")
            elif(yk + dWy >= h):
                flag_hit_limits = True
                if(verbose): print("[INFO] Reached max image height.")

            if(flag_hit_limits): flag_done = True

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
                    if flag_hit_limits:
                        flag_done = False
                        if(verbose): print("\tTrying Last Ditch Search...")
                else:
                    yLims.append(yk)
                    try_last_search = False
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

        if(verbose): print("[%d] Good Windows" % nWindows)
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[obstacle_search] --- Took %f seconds to complete" % (dt))

        return yLims, windows, dt

    def find_obstacles_disparity(self, vmap, dLims, xLims, search_thresholds = (3,30), ground_detected=True, lineCoeffs=None, verbose=False,timing=False):
        obs = []; obsUmap = []; windows = []; ybounds = []; dBounds = []
        nObs = len(dLims)

        if(timing): t0 = time.time()

        for i in range(nObs):
            xs = xLims[i]
            ds = dLims[i]
            ys,ws,_ = self.obstacle_search_disparity(vmap, ds, search_thresholds, lineCoeffs=lineCoeffs, verbose=verbose)
            # if self.debug_obstacle_search: print("Y Limits[",len(ys),"]: ", ys)
            if(len(ys) <= 2 and ground_detected):
                if(verbose): print("[INFO] Found obstacle with zero height. Skipping...")
            elif(len(ys) <= 1):
                if(verbose): print("[INFO] Found obstacle with zero height. Skipping...")
            elif(ys[0] == ys[1]):
                if(verbose): print("[INFO] Found obstacle with zero height. Skipping...")
            else:
                ybounds.append(ys)
                obs.append([
                    (xs[0],ys[0]),
                    (xs[1],ys[-1])
                ])
                obsUmap.append([
                    (xs[0],ds[0]),
                    (xs[1],ds[1])
                ])
                windows.append(ws)
                dBounds.append(ds)

        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("\t[INFO] find_obstacles() --- Took %f seconds (%.2f Hz) to complete" % (dt, 1/dt))
        return obs, obsUmap, ybounds, dBounds, windows, len(obs)

    def calculate_distance(self, umap, xs, ds, ys, dsbuffer=1,
        use_principle_point=True, use_disparity_buffer=False, verbose=False):

        # focal=[462.138,462.138],
        # baseline=0.055
        # dscale=0.001
        # pp=[320.551,232.202]

        nonzero = umap.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if(use_disparity_buffer): ds[0] = ds[0] - dsbuffer

        good_inds = ((nonzeroy >= ds[0]) & (nonzeroy < ds[1]) &
                     (nonzerox >= xs[0]) &  (nonzerox < xs[1])).nonzero()[0]

        xmean = np.int(np.mean(nonzerox[good_inds]))
        dmean = np.mean(nonzeroy[good_inds])
        ymean = np.mean(ys)
        # zgain = 1.0/(self.cam.dscale)#*(65535.0/255.0)
        zgain = 1.0/(self.disparity2uintGain*self.cam.dscale*1.35)
        z = 1.0/dmean
        # z = dmean*(self.focal[0]*self.baseline)
        # z = (self.focal[0]*self.baseline)/dmean
        # z = (462.138*self.baseline)/dmean
        z = z * zgain

        if use_principle_point:
            px = self.ppoint[0]
            py = self.ppoint[1]
        else: px = py = 0

        x = ((xmean - px)/self.focal[0])*z
        y = ((ymean - py)/self.focal[1])*z
        dist = np.sqrt(x*x+z*z)

        if(verbose): print("X, Y, Z: %.3f, %.3f, %.3f" % (x,y, dist))
        return dist,x,y,z

    def extract_obstacle_information(self, umap, xBounds, dBounds, obstacles, verbose=True):
        distances = [];  angles = []
        xs = xBounds
        ds = np.array(dBounds)
        obs = obstacles
        nObs = len(ds)
        nObsNew = 0
        if verbose: print("[INFO] VBOATS.py --- %d Obstacles Found" % (nObs))
        if(nObs is not 0):
            for i in range(nObs):
                disparities = ds[i]
                us = [obs[i][0][0], obs[i][1][0]]
                vs = [obs[i][0][1], obs[i][1][1]]
                z,ux,uy,uz = self.calculate_distance(umap,us,disparities,vs)

                theta = math.atan2(ux,uz)
                theta = np.nan_to_num(theta)
                # if verbose: print("\tObstacle [%d] Distance, Angle: %.3f, %.3f" % (i,z,np.rad2deg(theta)))

                if(z > 0.05):
                    distances.append(z)
                    angles.append(theta)
                    nObsNew+=1
                    if verbose: print("\tObstacle [%d] Distance, Angle: %.3f, %.3f" % (i,z,np.rad2deg(theta)))
        else:
            distances.append(-1)
            angles.append(0)
        if(nObsNew == 0):
            distances.append(-1)
            angles.append(0)

        return distances,angles

    def update(self):
        nObs = 0
        lineParams = None
        if(self.flag_show_imgs):
            cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)
            cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('ground_line', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('filtered_vmap', cv2.WINDOW_NORMAL)

        try:
            img = np.copy(self.disparity)
            raw_umap, raw_vmap, _ = self.vboat.get_uv_map(img)
            cv2.rectangle(raw_vmap,(0,0),(3, raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
            cv2.rectangle(raw_umap,(0,0),(raw_umap.shape[1], 3),(0,0,0), cv2.FILLED)
            self.umap = np.copy(raw_umap)
            self.vmap = np.copy(raw_vmap)
        except:
            print("[WARNING] Failed to get UV Maps")
            pass

        # try: vmapFiltered,_,_ = self.filterVmap(raw_vmap,[0.65,0.85,0.35,0.5])
        try:
            vmapFiltered,_,_ = self.filterVmap(raw_vmap,[0.25,0.15,0.35,0.35])
            vmapLessFiltered,_,_ = self.filterVmap(raw_vmap,[0.15,0.15,0.01,0.01])
            # vmapFiltered,_,_ = self.filterVmap(raw_vmap,[0.25,0.15,0.45,0.35])
        except:
            print("[WARNING] Failed to get Filter V-Map")
            pass

        # try: sets = estimateGndLineCoeffsHough(raw_vmap)
        try:
            # sets = estimateGndLineCoeffsHough(vmapFiltered)
            validGnd,bestM,bestIntercept, sets = estimateGndLineCoeffsHough2(vmapFiltered)
            if validGnd: lineParams = [bestM,bestIntercept]
        except:
            print("[WARNING] Failed estimateGndLineCoeffsHough")
            pass

        try:
            filtU, strips, stripsT = self.filterUmap(raw_umap,[0.15,0.15,0.25,0.35])
            kernelI = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            filtU = cv2.morphologyEx(filtU, cv2.MORPH_CLOSE, kernelI)
            filtered_contours,contours = self.vboat.find_contours(filtU,threshold = 100.0, max_thresh=-1)
        except:
            print("[WARNING] Failed Umap Filtering")
            pass

        try:
            # copy = np.copy(curImg)
            # dispO = cv2.applyColorMap(copy,cv2.COLORMAP_PARULA)
            # copy = cv2.cvtColor(copy,cv2.COLOR_GRAY2BGR)
            # dispO = cv2.cvtColor(dispO,cv2.COLOR_RGB2BGR)

            # dispU = cv2.applyColorMap(filtU,cv2.COLORMAP_PARULA)
            # dispU = cv2.cvtColor(dispU,cv2.COLOR_RGB2BGR)
            # [cv2.drawContours(dispU, [cnt], 0, (255,255,255), 1) for cnt in filtered_contours]
            # pplot(dispU,"Contours")

            xLims, dLims, _ = self.vboat.extract_contour_bounds(filtered_contours)
            obs, obsU, ybounds, dbounds, windows, nObs = self.find_obstacles_disparity(vmapLessFiltered, dLims, xLims, ground_detected=validGnd, lineCoeffs = lineParams, verbose=False)
            # print(nObs)

            # for i in range(0,len(xLims)):
            #     avgX = int(np.mean(xLims[i]))
            #     avgD = int(np.mean(dLims[i]))
            #     cv2.circle(dispU,(avgX,avgD),2,(255,0,255),-1)
            # pplot(dispU,"Contours")

            cpyV = np.copy(vmapLessFiltered)
            dispWindows = cv2.applyColorMap(cpyV,cv2.COLORMAP_PARULA)
            # dispWindows = cv2.cvtColor(dispWindows,cv2.COLOR_RGB2BGR)

            for wins in windows:
                for win in wins:
                    cv2.rectangle(dispWindows,win[0],win[1],(0,255,255), 1)
            if validGnd:
                w = cpyV.shape[1]
                tmpy = int(w * bestM + bestIntercept)
                cv2.line(dispWindows,(0,bestIntercept), (w,tmpy), (0,0,255),1)
            # pplot(dispWindows,"Windows")

            # alpha=0.35
            # [cv2.rectangle(copy, ob[0], ob[1],(255,0,0),3) for ob in obs]
            # cv2.addWeighted(copy, alpha, dispO, 1 - alpha, 0, dispO)
            # pplot(dispO,"Obstacles")
        except:
            print("[WARNING] Failed obstacle detection")
            pass

        try:
            distances, angles = self.extract_obstacle_information(raw_umap, xLims, dbounds, obs)
        except:
            print("[WARNING] Failed obstacle extraction")
            pass

        # try:
        #     vmapNoGnd = self.removeGnd2(raw_vmap, sets)
        #     # if(len(sets > 0)):
        #     # else: vmapNoGnd = np.copy(vmapFiltered)
        # except:
        #     print("[WARNING] Failed removeGnd2")
        #     vmapNoGnd = np.copy(raw_vmap)
        #     pass

        # try: m,b, xs,ys = estimateGndLineCoeffs(vmapFiltered)
        # except:
        #     print("[WARNING] Failed estimateGndLineCoeffs")
        #     pass
        #
        # try: validGnd, pts = self.extractGndLine(m,b)
        # except:
        #     print("[WARNING] Failed to extract Ground line")
        #     pass
        #
        # try:
        #     if(validGnd): vmapNoGnd, vizGndLine = self.removeGround(raw_vmap,pts)
        #     else:
        #         vizGndLine = np.copy(vmapFiltered)
        #         if(len(vizGndLine.shape)>2): vizGndLine = cv2.cvtColor(vizGndLine,cv2.COLOR_BGR2GRAY)
        #         vizGndLine = cv2.applyColorMap(vizGndLine,cv2.COLORMAP_PARULA)
        # except:
        #     print("[WARNING] Failed to remove Ground line")
        #     pass

        if(self.flag_show_imgs):
            # h,w = vmapFiltered.shape[:2]
            # display2 = np.copy(vmapFiltered)
            # display1 = cv2.cvtColor(display2, cv2.COLOR_GRAY2BGR)
            # display2 = cv2.applyColorMap(display2,cv2.COLORMAP_PARULA)
            # tmpy = int(w * bestM + bestIntercept)
            # cv2.line(display2,(0,bestIntercept), (w,tmpy), (255,0,0),1)
            try:
                overlay = make_uv_overlay(cv2.cvtColor(self.disparity,cv2.COLOR_GRAY2BGR),self.umap,self.vmap)
                cv2.imshow('overlay', overlay)
            except:
                print("[WARNING] Failed to create Overlay")
                pass
            cv2.imshow('disparity', dispWindows)
            # cv2.imshow('ground_line', vmapNoGnd)
            # cv2.imshow('filtered_vmap', vmapFiltered)
        return nObs

    def start(self):
        dt = 0
        count = savecnt = 0
        while not rospy.is_shutdown():
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'): break
            try:
                rgb, depth = self.cam.read()
                if((rgb is None) or (depth is None)): continue

                t0 = time.time()
                self.camCallback(rgb, depth)
                nObs = self.update()

                # self.vboat.pipelineV1(self.img, timing=True)

                t1 = time.time()
                dt = t1 - t0
                print("[INFO] vboat_testing_node::loop() --- Found %d Obstacles in %f seconds (%.2f Hz)" % (nObs,dt, 1/dt))

                # time = rospy.Time.now()
                # try: self.image_pub.publish(self.bridge.cv2_to_imgmsg(display_obstacles, "bgr8"))
                # except CvBridgeError as e: print(e)
                if self.flag_save_imgs:
                    if(key == ord('s')):
                        img_suffix = "frame_" + str(savecnt) + ".png"
                        if self.rgb is not None:
                            rgb_file = "rgb_" + img_suffix
                            rgb_path = os.path.join(self.rgbDir,rgb_file)
                            self.save_image_to_file(self.rgb,rgb_path)
                        if self.disparity is not None:
                            depth_file = "depth_" + img_suffix
                            depth_path = os.path.join(self.depthDir,depth_file)
                            self.save_image_to_file(self.disparity,depth_path)

                            # obs_file = "obstacle_" + img_suffix
                            # obs_path = os.path.join(self.obsDir,obs_file)
                            # self.save_image_to_file(display_obstacles,obs_path)
                        savecnt += 1
                count+=1
            except: pass
            self.r.sleep()

if __name__ == "__main__" :
    vnode = vboat_testing_node()
    vnode.start()
