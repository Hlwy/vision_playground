#!/usr/bin/env python

import cv2
import rospy
import numpy as np
import os, csv, time, math
from sklearn import linear_model
from matplotlib import pyplot as plt

from sensor_msgs.msg import Image,CompressedImage
from cv_bridge import CvBridge, CvBridgeError

from VBOATS import VBOATS
from devices.d415_camera import *
from hyutils.img_utils import *
from hyutils.uv_mapping_utils import *

def estimateGndLineCoeffs(filtered_vmap, stopP = 0.85, maxTrials=30,verbose = False,show_plots=False):
    gndImg = np.copy(filtered_vmap)
    nonzero = gndImg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    nonzerox = np.reshape(nonzerox,(nonzerox.shape[0],1))

    ransac = linear_model.RANSACRegressor(stop_probability=stopP,max_trials=maxTrials)
    ransac.fit(nonzerox, nonzeroy)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(nonzerox.min(), nonzerox.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    if(show_plots):
        plt.figure("RANSAC")
        plt.clf()
        plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.0,right=1.0,top=1.0, bottom=0.0)
        plt.axis([0.0, 255.0, 480, 0])
        plt.imshow(gndImg,interpolation='bilinear')
        plt.scatter(nonzerox,nonzeroy, color='yellowgreen', marker='.',label='Inliers')
        plt.plot(line_X, line_y_ransac, color='red', linewidth=2, label='RANSAC regressor')
        plt.show()

    m = ransac.estimator_.coef_
    b = ransac.estimator_.intercept_
    if(verbose): print("Estimated coefficients (RANSAC): m = %.3f | b = %.3f" % (m,b))
    return m,b, line_X,line_y_ransac

class vboat_testing_node:
    def __init__(self):
        rospy.init_node('vboat_testing_node')
        rootDir = os.path.dirname(os.path.abspath(__file__))
        imgsPath = os.path.abspath(os.path.join(rootDir, "..")) + "/images"

        self.fps = rospy.get_param('~fps', 30)
        self.save_path = rospy.get_param('~save_path',imgsPath)
        self.flag_save_imgs = rospy.get_param('~flag_save_imgs', False)
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
        self.focal = self.intr["depth"].fx
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
        depth = np.uint8(tmp*255)
        depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)

        tmp2 = _depth*0.001
        loc = np.where(tmp2 == 0.0)
        tmp2[loc] = 1.0
        disparity = (self.focal*self.baseline)/tmp2
        disparity[loc] = 0.0
        self.disparity2uintGain = 255/np.max(disparity)
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

    def filterVmap(self,_vmap, thresholds = [0.85,0.85,0.75,0.5], verbose=False, debug=False):
        raw_vmap = np.copy(_vmap)
        cv2.rectangle(raw_vmap,(0,0),(1, raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
        vMax = np.max(_vmap)
        nThreshs = int(math.ceil((vMax/256.0)) * 4)
        if(debug): print("vMax, nThreshs" % (vMax, nThreshs))

        # Rough Theshold top of vmap heavy prior
        h, w = raw_vmap.shape[:2]
        dh = h / 3
        topV = raw_vmap[0:dh, :]
        botV = raw_vmap[dh:h, :]
        tmpThresh = int(np.max(topV)*0.9)
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

    def update(self):
        if(self.flag_show_imgs):
            cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
            cv2.namedWindow('ground_line', cv2.WINDOW_NORMAL)

        try:
            img = np.copy(self.disparity)
            raw_umap, raw_vmap, _ = self.vboat.get_uv_map(img)
            self.umap = np.copy(raw_umap)
            self.vmap = np.copy(raw_vmap)
        except:
            print("[WARNING] Failed to get UV Maps")
            pass

        try: vmapFiltered,_,_ = self.filterVmap(raw_vmap,[0.65,0.85,0.35,0.5])
        except:
            print("[WARNING] Failed to get Filter V-Map")
            pass

        try: m,b, xs,ys = estimateGndLineCoeffs(vmapFiltered)
        except:
            print("[WARNING] Failed estimateGndLineCoeffs")
            pass

        try: validGnd, pts = self.extractGndLine(m,b)
        except:
            print("[WARNING] Failed to extract Ground line")
            pass

        try:
            if(validGnd): vmapNoGnd, vizGndLine = self.removeGround(raw_vmap,pts)
            else:
                vizGndLine = np.copy(vmapFiltered)
                if(len(vizGndLine.shape)>2): vizGndLine = cv2.cvtColor(vizGndLine,cv2.COLOR_BGR2GRAY)
                vizGndLine = cv2.applyColorMap(vizGndLine,cv2.COLORMAP_PARULA)
        except:
            print("[WARNING] Failed to remove Ground line")
            pass

        if(self.flag_show_imgs):
            try:
                overlay = make_uv_overlay(cv2.cvtColor(self.disparity,cv2.COLOR_GRAY2BGR),self.umap,vmapFiltered)
                cv2.imshow('overlay', overlay)
            except:
                print("[WARNING] Failed to create Overlay")
                pass
            # cv2.imshow('disparity', self.disparity)
            cv2.imshow('ground_line', vizGndLine)


    def start(self):
        dt = 0
        count = 0
        while not rospy.is_shutdown():
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'): break
            try:
                rgb, depth = self.cam.read()
                if((rgb is None) or (depth is None)): continue

                t0 = time.time()
                self.camCallback(rgb, depth)
                self.update()

                # self.vboat.pipelineTest(self.img, threshU1=0.25, threshU2=0.1, threshV1=5, threshV2=70, timing=True)

                t1 = time.time()
                dt = t1 - t0
                # print("[INFO] vboat_testing_node::loop() --- Took %f seconds (%.2f Hz) to complete" % (dt, 1/dt))

                # time = rospy.Time.now()
                # try: self.image_pub.publish(self.bridge.cv2_to_imgmsg(display_obstacles, "bgr8"))
                # except CvBridgeError as e: print(e)

                if self.flag_save_imgs:
                    img_suffix = "frame_" + str(count) + ".png"
                    if self.rgb is not None:
                        rgb_file = "rgb_" + img_suffix
                        rgb_path = os.path.join(self.rgbDir,rgb_file)
                        self.save_image_to_file(self.rgb,rgb_path)
                    if self.img is not None:
                        depth_file = "depth_" + img_suffix
                        depth_path = os.path.join(self.depthDir,depth_file)
                        self.save_image_to_file(self.img,depth_path)

                        obs_file = "obstacle_" + img_suffix
                        obs_path = os.path.join(self.obsDir,obs_file)
                        self.save_image_to_file(display_obstacles,obs_path)

                count+=1
            except: pass
            self.r.sleep()

if __name__ == "__main__" :
    vnode = vboat_testing_node()
    vnode.start()
