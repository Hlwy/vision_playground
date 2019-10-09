#!/usr/bin/env python

import cv2
import numpy as np
import rospy, tf
import os, csv, time, math
from matplotlib import pyplot as plt

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image,CameraInfo,CompressedImage
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from VBOATS import VBOATS
from devices.d415_camera import *
from hyutils.img_utils import *
from hyutils.uv_mapping_utils import *

class vboat_testing_node:
    def __init__(self):
        rospy.init_node('vboat_testing_node')
        rootDir = os.path.dirname(os.path.abspath(__file__))
        imgsPath = os.path.abspath(os.path.join(rootDir, "..")) + "/images"

        self.ns = rospy.get_namespace()
        self.fps = rospy.get_param('~fps', 30)
        self.save_path = rospy.get_param('~save_path',imgsPath)
        self.flag_save_imgs = rospy.get_param('~flag_save_imgs', False)
        self.flag_show_imgs = rospy.get_param('~flag_show_imgs', True)
        self.cam_name = rospy.get_param('~camera_name',"d415")

        self.base_tf_frame = rospy.get_param('~base_tf_frame', "base_link")
        self.cam_base_tf_frame = rospy.get_param('~cam_base_tf_frame', self.cam_name+"/rgb_base_frame")
        self.cam_tf_frame = rospy.get_param('~cam_tf_frame', self.cam_name+"/rgb_optical_frame")
        self.image_topic = rospy.get_param('~image_topic', self.ns+self.cam_name+"/depth/image_raw")
        self.image_topic_rgb = rospy.get_param('~rgb_image_topic', self.ns+self.cam_name+"/rgb/image_raw")
        self.camera_info_topic = rospy.get_param('~camera_info_topic', self.ns+self.cam_name+"/rgb/camera_info")

        # self.cam_tf = rospy.get_param('~cam_tf', "d415_camera_depth_optical_frame")
        # self.image_topic = rospy.get_param('~image_topic', "d415/depth/image_raw")

        # self.dist_pub = rospy.Publisher(rospy.get_namespace()+"/obstacles/distances",FloatArrayStamped,queue_size=1000)
        # self.ang_pub = rospy.Publisher(rospy.get_namespace()+"/obstacles/angles",FloatArrayStamped,queue_size=1000)

        self.image_pub = rospy.Publisher("vboat/image",Image,queue_size=1000)
        self.rgb_pub = rospy.Publisher(self.image_topic_rgb,Image,queue_size=10)
        self.pub_info = rospy.Publisher(self.camera_info_topic,CameraInfo,queue_size=10)
        self.depth_pub = rospy.Publisher(self.image_topic,Image,queue_size=10)

        self.bridge = CvBridge()
        self.br = tf.TransformBroadcaster()

        self.cam = CameraD415(flag_save=False,use_statistics=False,fps=self.fps, verbose=True)

        self.intr = self.cam.get_intrinsics()
        self.extr = self.cam.get_extrinsics()

        fx = self.intr["depth"].fx
        fy = self.intr["depth"].fy
        ppx = self.intr['depth'].ppx
        ppy = self.intr['depth'].ppy

        self.focal = [fx, fy]
        self.ppoint = [ppx, ppy]
        self.baseline = self.extr.translation[0]

        self.info_msg = CameraInfo()
        self.info_msg.width = self.intr['depth'].width
        self.info_msg.height = self.intr['depth'].height
        self.info_msg.K = [fx, 0, ppx,
                           0, fy, ppy,
                           0, 0, 1]

        self.info_msg.D = [0, 0, 0, 0]

        self.info_msg.P = [fx, 0, ppx, 0,
                           0, fy, ppy, 0,
                           0, 0, 1, 0]

        self.vboat = VBOATS()
        self.vboat.dead_x = 3
        self.vboat.dead_y = 3

        self.r = rospy.Rate(60)
        self.rgb = []
        self.depth = []
        self.disparity = []
        self.umap = []
        self.vmap = []

        self.count = 0
        self.camcount = 0
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
        # print("[INFO] vboat_testing_node::camCallback() --- Converting frames...")
        tmp = _depth/65535.0
        ratio = np.max(_depth)/65535.0
        depth = np.uint8(tmp*255)

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

        # print("[INFO] vboat_testing_node::camCallback() --- Publishing ROS frames...")
        self.depth_pub.publish(self.bridge.cv2_to_imgmsg(self.depth, "8UC1"))
        try:
            msg = self.create_camera_info_msg()
            self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(self.rgb, "bgr8"))
            msg.header.stamp = self.bridge.cv2_to_imgmsg(self.rgb, "bgr8").header.stamp
            self.pub_info.publish(msg)
        except CvBridgeError as e:
            print(e)

        self.camcount+=1

    def create_camera_info_msg(self):
        self.info_msg.header.stamp = rospy.Time.now()
        self.info_msg.header.seq = self.count
        self.info_msg.header.frame_id = self.cam_tf_frame

        self.br.sendTransform((0, 0, 0), tf.transformations.quaternion_from_euler(-(np.pi/2.0), 0, (np.pi/2.0)), rospy.Time.now(), self.cam_tf_frame,self.cam_base_tf_frame)

        # self.br.sendTransform((0.21,0.0,0.02), tf.transformations.quaternion_from_euler(0, 0, np.pi), rospy.Time.now(), self.cam_base_tf_frame,self.base_tf_frame)
        self.br.sendTransform((0.0,0.0,0.0), tf.transformations.quaternion_from_euler(0, 0, np.pi), rospy.Time.now(), self.cam_base_tf_frame,self.base_tf_frame)
        return self.info_msg

    def save_image_to_file(self, img, path):
        try: cv2.imwrite(path, img)
        except:
            print("[ERROR] vboat_node.py ---- Could not save image to file \'%s\'" % path)
            pass

    def update(self):
        nObs = 0
        lineParams = None
        if(self.flag_show_imgs):
            # cv2.namedWindow('ground_line', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('filtered_vmap', cv2.WINDOW_NORMAL)
            cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)
            cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)

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

        try:
            vmapFiltered,_,_ = self.vboat.filter_disparity_umap(raw_vmap,[0.25,0.15,0.35,0.35])
            vmapLessFiltered,_,_ = self.vboat.filter_disparity_vmap(raw_vmap,[0.15,0.15,0.01,0.01])
        except:
            print("[WARNING] Failed to get Filter V-Map")
            pass

        try:
            validGnd,bestM,bestIntercept, sets = self.vboat.estimate_houghline_coeffs(vmapFiltered)
            if validGnd: lineParams = [bestM,bestIntercept]
        except:
            print("[WARNING] Failed estimateGndLineCoeffsHough")
            pass

        try:
            filtU, strips, stripsT = self.vboat.filter_disparity_umap(raw_umap,[0.15,0.15,0.25,0.35])
            kernelI = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            filtU = cv2.morphologyEx(filtU, cv2.MORPH_CLOSE, kernelI)
            filtered_contours,contours = self.vboat.find_contours(filtU,threshold = 100.0, max_thresh=-1)
        except:
            print("[WARNING] Failed Umap Filtering")
            pass

        try:
            xLims, dLims, _ = self.vboat.extract_contour_bounds(filtered_contours)
            obs, obsU, ybounds, dbounds, windows, nObs = self.vboat.find_obstacles_disparity(vmapLessFiltered, dLims, xLims, ground_detected=validGnd, lineCoeffs = lineParams, verbose=False)
            # print(nObs)
        except:
            print("[WARNING] Failed obstacle detection")
            pass

        try:
            distances, angles = self.vboat.extract_obstacle_information_disparity(raw_umap, xLims, dbounds, obs)
        except:
            print("[WARNING] Failed obstacle extraction")
            pass

        if(self.flag_show_imgs):
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
            # ----------

            # h,w = vmapFiltered.shape[:2]
            # display2 = np.copy(vmapFiltered)
            # display1 = cv2.cvtColor(display2, cv2.COLOR_GRAY2BGR)
            # display2 = cv2.applyColorMap(display2,cv2.COLORMAP_PARULA)
            # tmpy = int(w * bestM + bestIntercept)
            # cv2.line(display2,(0,bestIntercept), (w,tmpy), (255,0,0),1)
            # cv2.imshow('ground_line', vmapNoGnd)
            # cv2.imshow('filtered_vmap', vmapFiltered)
            try:
                overlay = make_uv_overlay(cv2.cvtColor(self.disparity,cv2.COLOR_GRAY2BGR),self.umap,self.vmap)
                cv2.imshow('overlay', overlay)
            except:
                print("[WARNING] Failed to create Overlay")
                pass
            cv2.imshow('disparity', dispWindows)
        return nObs

    def start(self, debug=False):
        dt = 0
        count = savecnt = 0
        while True:
            if(rospy.is_shutdown()): break
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'): break
            try:
                rgb, depth = self.cam.read()
                if((rgb is None) and (depth is None)):
                    print("[INFO] vboat_testing_node::loop() --- Grabbed frames both None, Skipping...")
                    continue
                elif((rgb is not None) and (depth is None)):
                    if(debug): print("[INFO] vboat_testing_node::loop() --- Depth frame grabbed is None")
                    depth = np.zeros_like(rgb)
                elif((rgb is None) and (depth is not None)):
                    if(debug): print("[INFO] vboat_testing_node::loop() --- RGB frame grabbed is None")
                    rgb = np.zeros_like(depth)

                if(debug): print("[INFO] vboat_testing_node::loop() --- Successful frame grab")
                t0 = time.time()
                self.camCallback(rgb, depth)
                nObs = self.update()

                t1 = time.time()
                dt = t1 - t0
                print("[INFO] vboat_testing_node::loop() --- Found %d Obstacles in %f seconds (%.2f Hz)" % (nObs,dt, 1/dt))

                # time = rospy.Time.now()
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
                self.count+=1
            except: pass
            self.r.sleep()

if __name__ == "__main__" :
    vnode = vboat_testing_node()
    vnode.start()
