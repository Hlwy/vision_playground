#!/usr/bin/env python

import cv2
import numpy as np
import os, csv, time, argparse, math
from matplotlib import pyplot as plt

import rospy
from sensor_msgs.msg import Image,CompressedImage
from cv_bridge import CvBridge, CvBridgeError


from VBOATS import VBOATS
from devices.d415_camera import *
from hyutils.img_utils import *
from hyutils.uv_mapping_utils import *
from sklearn import linear_model

def find_rough_ground_breakpoint(_img, verbose = False, debug = False,show_plots=False):
    h, _ = _img.shape[:2]
    hist = vertical_hist(_img)
    smHist = histogram_sliding_filter(hist,window_size=8)
    if(show_plots):
        plt.figure(1)
        plt.clf()
        plt.title('Histogram of the image')
        plt.plot(range(smHist.shape[0]), smHist[:])
    cnt = 0
    idxs = []
    lastIdx = None
    flag_started = False
    argIdx = smHist.shape[0]/4
    if(verbose): print("Starting Search Index: %d" % (argIdx))
    for i, val in enumerate(smHist[argIdx:]):
        idx = i+argIdx
        dval = val - smHist[idx-1]
        if(dval <= -15.0):
            if(lastIdx is None):
                lastIdx = idx
                idxs.append(idx)
            cnt += 1
            flag_started = True
            if(verbose): print("[#%d] Found Negative Slope = %.1f" % (idx, dval))
        if(dval >= 0.0):
            if(flag_started):
                cnt = 0
                lastIdx = None
                if(debug): print("reseting")
                flag_started = False
        if(debug): print(idx,val, dval)
    if((cnt >= 1) and (lastIdx is not None)):
        if(debug): print("Found %d Indices with sufficient negative slopes: %s" % (len(idxs),str(idxs)))
        if(h-20 <= lastIdx <= h):
            try:
                lastIdx = idxs[-2]
                if(h-20 <= lastIdx <= h): lastIdx = -1
            except:
                lastIdx = -1
        if(verbose): print("cnt, lowest idx", cnt, lastIdx)
        return lastIdx
    else: return -1


def get_linear_ground_mask(img, debug_timings=False,verbose=False):
        # img = np.copy(_img)
        h, _ = img.shape[:2]
        if(debug_timings):
            dt = 0
            t0 = time.time()

        lowIdx = find_rough_ground_breakpoint(img)
        if(lowIdx == -1): hm = h/2
        else: hm = lowIdx

        if(hm >= h-30): hm = h/2

        if(verbose): print("Thresholding Below %d" % (hm))
        topHalf = img[0:hm, :]
        botHalf = img[hm:h, :]
        _,ttHalf = cv2.threshold(topHalf, 225,255,cv2.THRESH_TOZERO)
        _,tbHalf = cv2.threshold(botHalf, 64,255,cv2.THRESH_TOZERO)
        tmp = np.concatenate((ttHalf,tbHalf), axis=0)

        nonzero = tmp.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        nonzerox = np.reshape(nonzerox,(nonzerox.shape[0],1))

        line_X = np.arange(nonzerox.min(), nonzerox.max())[:, np.newaxis]

        ransac = linear_model.RANSACRegressor(stop_probability=0.85,max_trials=30)
        ransac.fit(nonzerox, nonzeroy)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        line_y_ransac = ransac.predict(line_X)

        if(debug_timings):
            t1 = time.time()
            dt = t1 - t0
            print("[INFO] get_linear_ground_mask() --- Took %f seconds (%.2f Hz) to complete" % (dt, 1/dt))

        m = ransac.estimator_.coef_
        b = ransac.estimator_.intercept_
        # print("Estimated coefficients (RANSAC): m = %.3f | b = %.3f" % (m,b))
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
        self.image_pub = rospy.Publisher("vboat_tester/image",Image,queue_size=1000)

        self.bridge = CvBridge()
        self.cam = CameraD415(flag_save=False,use_statistics=False,fps=self.fps)

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

        self.rgbDir = os.path.join(self.save_path, "rgb")
        self.depthDir = os.path.join(self.save_path, "depth")
        self.obsDir = os.path.join(self.save_path, "processed")

        if self.flag_save_imgs:
            dirs = [self.save_path,self.rgbDir, self.depthDir, self.obsDir]
            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)
                    print("Created directory \'%s\' " % dir)
                else: print("Directory \'%s\' already exists" % dir)

    def camCallback(self, _rgb, _depth):
        tmp = _depth/65535.0
        depth = np.uint8(tmp*255)
        # depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)

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

    def update(self):
        img = np.copy(self.disparity)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        disparity = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        raw_umap, raw_vmap, dt = self.vboat.get_uv_map(disparity)

        self.umap = np.copy(raw_umap)
        self.vmap = np.copy(raw_vmap)

        ###########################################
        cv2.rectangle(raw_vmap,(0,0),(1, raw_vmap.shape[0]),(0,0,0), cv2.FILLED)
        thresh0 = int(np.max(raw_vmap)*0.025)
        _,tmp = cv2.threshold(raw_vmap, thresh0,255,cv2.THRESH_TOZERO)

        color = cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)
        ground_display = np.copy(color)
        ###########################################
        m,b, xs,ys = get_linear_ground_mask(tmp)
        if(m >= 0.2):
            h,w = tmp.shape[:2]
            yf = int(m*w + b)
            pt1 = (0,int(b))
            pt2 = (w,yf-15)
            pts = np.array([(0, h),pt1,pt2,(w, h),(0, h)])
            cv2.fillPoly(color, [pts], (0,0,0))
            cv2.line(ground_display,pt1, pt2, (0,255,255),1)
        else: print("No Ground Detected.")

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
            try:
                rgb, depth = self.cam.read()
                if((rgb is None) or (depth is None)): continue

                t0 = time.time()

                self.camCallback(rgb, depth)
                vmapMod,gndLine = self.update()
                # self.vboat.pipelineTest(self.img, threshU1=0.25, threshU2=0.1, threshV1=5, threshV2=70, timing=True)

                t1 = time.time()
                dt = t1 - t0
                print("[INFO] vboat_testing_node::loop() --- Took %f seconds (%.2f Hz) to complete" % (dt, 1/dt))

                # time = rospy.Time.now()
                # try: self.image_pub.publish(self.bridge.cv2_to_imgmsg(display_obstacles, "bgr8"))
                # except CvBridgeError as e: print(e)

                if(self.flag_show_imgs):
                    overlay = make_uv_overlay(self.rgb,self.umap,vmapMod)
                    cv2.imshow('overlay', overlay)
                    cv2.imshow('disparity', self.disparity)
                    cv2.imshow('ground_line', gndLine)
                    cv2.waitKey(2)

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
