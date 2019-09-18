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

def groundSeg(filtered_vmap, stopP = 0.85, maxTrials=30,verbose = False,show_plots=False):
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

    def prefilter_vmap(self,_vmap, thresholds = [0.85,0.85,0.75,0.5], verbose=False, debug=False):
        vMax = np.max(_vmap)
        nThreshs = int(math.ceil((vMax/256.0)) * 4)
        if(debug): print("vMax, nThreshs" % (vMax, nThreshs))

        stripsPV = []
        stripsV = strip_image(_vmap, nstrips=nThreshs, horizontal_strips=False)
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

    def update(self):
        img = np.copy(self.disparity)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # disparity = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        raw_umap, raw_vmap, dt = self.vboat.get_uv_map(img)
        self.umap = np.copy(raw_umap)
        self.vmap = np.copy(raw_vmap)

        cv2.rectangle(raw_vmap,(0,0),(1, raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
        ###########################################
        filtV,_,_ = self.prefilter_vmap(raw_vmap)
        m,b, xs,ys = groundSeg(filtV)
        # m,b, xs,ys = get_linear_ground_mask(filtV)
        ###########################################
        color = cv2.cvtColor(raw_vmap,cv2.COLOR_GRAY2BGR)
        ground_display = np.copy(color)
        ground_display= cv2.applyColorMap(ground_display,cv2.COLORMAP_PARULA)
        if(m >= 0.3):
            # Filter out sudden jumps in segmented ground line
            if(self.prevM is None): self.prevM = m
            if(self.prevB is None): self.prevB = b
            dSlope = math.fabs(self.prevM - m)
            dB = math.fabs(self.prevB - b)
            print("dSlope, dIntercept = %.3f, %.3f" % (dSlope,dB))
            if((dSlope >= 2.0) or (dB >= 300.0)):
                m = self.prevM
                b = self.prevB
            self.prevM = m
            self.prevB = b
            h,w = filtV.shape[:2]
            yf = int(m*w + b)
            pt1 = (0,int(b))
            pt2 = (w,yf-15)
            pts = np.array([(0, h),pt1,pt2,(w, h),(0, h)])
            cv2.fillPoly(color, [pts], (0,0,0))
            cv2.line(ground_display,pt1, pt2, (0,255,255),1)
        else:
            self.prevM = None; self.prevB = None
            print("No Ground Detected.")

        vmap_mod = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
        return vmap_mod,ground_display

    def start(self):
        dt = 0
        count = 0
        if(self.flag_show_imgs):
            cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)
            cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
            cv2.namedWindow('ground_line', cv2.WINDOW_NORMAL)
        while not rospy.is_shutdown():
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'): break
            try:
                rgb, depth = self.cam.read()
                if((rgb is None) or (depth is None)): continue

                t0 = time.time()

                self.camCallback(rgb, depth)
                vmapMod,gndLine = self.update()
                # self.vboat.pipelineTest(self.img, threshU1=0.25, threshU2=0.1, threshV1=5, threshV2=70, timing=True)

                t1 = time.time()
                dt = t1 - t0
                # print("[INFO] vboat_testing_node::loop() --- Took %f seconds (%.2f Hz) to complete" % (dt, 1/dt))

                # time = rospy.Time.now()
                # try: self.image_pub.publish(self.bridge.cv2_to_imgmsg(display_obstacles, "bgr8"))
                # except CvBridgeError as e: print(e)

                if(self.flag_show_imgs):
                    overlay = make_uv_overlay(self.rgb,self.umap,vmapMod)
                    cv2.imshow('overlay', overlay)
                    cv2.imshow('disparity', self.disparity)
                    cv2.imshow('ground_line', gndLine)

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
