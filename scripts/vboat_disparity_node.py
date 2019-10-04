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

        self.cam = CameraD415(flag_save=False,use_statistics=False,fps=30, verbose=True)

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

# # ----------------------------------------------------
# # ----------------------------------------------------

# class obstacle_detector_node:
#     def __init__(self):
#         rospy.init_node('obstacle_detector')
#         rootDir = os.path.dirname(os.path.abspath(__file__))
#         imgsPath = os.path.abspath(os.path.join(rootDir, "..")) + "/images"
#
#         self.fps = rospy.get_param('~fps', 15)
#         self.save_path = rospy.get_param('~save_path',imgsPath)
#         self.flag_use_ros = rospy.get_param('~flag_use_ros', False)
#         self.flag_save_imgs = rospy.get_param('~flag_save_imgs', True)
#         self.flag_publish_imgs = rospy.get_param('~flag_publish_imgs', False)
#
#         self.base_tf = rospy.get_param('~base_tf', "/ugv1/odom")
#         self.cam_tf = rospy.get_param('~cam_tf', "/ugv1/d415_camera_depth_optical_frame")
#         self.image_topic = rospy.get_param('~image_topic', "/ugv1/d415/depth/image_raw")
#
#         self.dist_pub = rospy.Publisher(rospy.get_namespace()+"/obstacles/distances",FloatArrayStamped,queue_size=1000)
#         self.ang_pub = rospy.Publisher(rospy.get_namespace()+"/obstacles/angles",FloatArrayStamped,queue_size=1000)
#
#         if self.flag_use_ros:
#             if self.flag_publish_imgs:
#                 self.image_pub = rospy.Publisher("vboat/obstacles/image",Image,queue_size=1000)
#             else: self.image_pub = None
#
#             self.bridge = CvBridge()
#             self.image_sub = rospy.Subscriber(self.image_topic,Image,self.rosCallback)
#             self.cam = None
#         else:
#             self.image_pub = None
#             self.image_sub = None
#             self.bridge = None
#             self.cam = CameraD415(flag_save=False,use_statistics=False,fps=self.fps)
#
#         self.vboat = VBOATS()
#         self.vboat.dead_x = 3
#         self.vboat.dead_y = 3
#
#         self.r = rospy.Rate(40)
#         self.img = []
#         self.rgb = []
#         self.obs_disp = []
#
#         self.roscount = 0
#         self.camcount = 0
#         self.count = 0
#         self.disp_obs = None
#
#         self.rgbDir = os.path.join(self.save_path, "rgb")
#         self.depthDir = os.path.join(self.save_path, "depth")
#         self.obsDir = os.path.join(self.save_path, "processed")
#
#         if self.flag_save_imgs:
#             dirs = [self.save_path,self.rgbDir, self.depthDir, self.obsDir]
#             for dir in dirs:
#                 if not os.path.exists(dir):
#                     os.makedirs(dir)
#                     print("Created directory \'%s\' " % dir)
#                 else: print("Directory \'%s\' already exists" % dir)
#
#     def rosCallback(self,data):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
#             cv_image = np.float32(cv_image)
#             tmp = cv_image/65535
#             depth = np.uint8(tmp*255)
#             self.img = np.copy(depth)
#             self.roscount+=1
#         except CvBridgeError as e: print(e)
#
#     def camCallback(self):
#         rgb, depth = self.cam.read()
#         if((rgb is None) or (depth is None)):
#             continue
#         tmp = depth/self.cam.dmax_avg
#         depth = np.uint8(tmp*255)
#         depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)
#         self.img = np.copy(depth)
#         self.rgb = rgb
#         self.camcount+=1
#
#     def draw_obstacle_image(self):
#         display_obstacles = cv2.cvtColor(self.vboat.img, cv2.COLOR_GRAY2BGR)
#         [cv2.rectangle(display_obstacles,ob[0],ob[1],(150,0,0),1) for ob in self.vboat.obstacles]
#         disp_obs = np.copy(display_obstacles)
#         return disp_obs
#
#     def save_image_to_file(self, img, path):
#         try: cv2.imwrite(path, img)
#         except:
#             print("[ERROR] vboat_node.py ---- Could not save image to file \'%s\'" % path)
#             pass
#
#     def get_camera_pose(self, cam_frame = '/ugv1/d415_camera_depth_optical_frame', base_frame = '/ugv1/odom'):
#         listener = tf.TransformListener()
#         listener.waitForTransform(base_frame,cam_frame, rospy.Time(0), rospy.Duration(8.0))
#         (trans,rot) = listener.lookupTransform(base_frame,cam_frame, rospy.Time(0))
#         roll,pitch,yaw = tf.transformations.euler_from_quaternion(rot)
#
#         pose = np.array(trans+[np.rad2deg(yaw)])
#         Tmat = np.array(trans).T
#         Rmat = tf.transformations.euler_matrix(roll,pitch,yaw,axes='sxyz')
#         return pose, Rmat, Tmat
#
#     def sim_update(self):
#         umap = self.vboat.umap_raw
#         xs = self.vboat.xBounds
#         ds = np.array(self.vboat.dbounds)
#         obs = self.vboat.obstacles
#         # ppx = 320.551; ppy = 232.202; fx = 626.464; fy = 626.464; b = 0.055
#         # pp = [ppx, ppy];   focal = [fx,fy]
#         dGain = (65535/255)
#
#         obid = 0; i = 0
#         X0 = np.array([ [5.0, -3.0, 0.21336] ])
#         pose,RotM,Tmat = self.get_camera_pose(self.cam_tf,self.base_tf)
#
#         Xk = pose[:2]; yaw = pose[-1]; ang = np.deg2rad(yaw)
#         x1 = X0[obid,0];     y1 = X0[obid,1]
#         dx = x1 - Xk[0];     dy = y1 - Xk[1]
#         true_dist = np.sqrt(dx*dx + dy*dy)-X0[obid,2]
#         nObs = len(ds)
#         # print("[%d] Obstacles Found" % nObs)
#         distances = []
#         angles = []
#         if(nObs is not 0):
#             for i in range(nObs):
#                 disparities = ds[i]
#                 us = [obs[i][0][0], obs[i][1][0]]
#                 vs = [obs[i][0][1], obs[i][1][1]]
#                 z,ux,uy,uz = self.vboat.calculate_distance(umap,us,disparities,vs)
#
#                 theta = math.acos((uz/z))
#                 distances.append(z)
#                 angles.append(theta)
#
#                 pxl = np.array([ [ux],[uy],[z] ])
#                 RotM = RotM[:3,:3]
#                 T = Tmat.reshape((3, 1))*-1
#                 pos = self.vboat.transform_pixel_to_world(RotM,pxl,T)
#
#                 strs = []
#                 strs.append(', '.join(map(str, np.around(pos.T[0][:2],3))))
#                 strs.append(', '.join(map(str, np.around(X0[obid,:2],3))))
#
# #                 print(
# # """
# # Detected Obstacle Stats:
# # ========================
# #     * Distances (True, Est.)    : %.3f, %.3f
# #     * Estimated Position (X,Y,Z): %s
# #     * True Position (X,Y,Z)     : %s
# # """
# #         % (true_dist,z,strs[0],strs[1]) )
#         else:
#             distances.append(-1)
#             angles.append(0)
#         return distances, angles
#
#     def update(self):
#         umap = self.vboat.umap_raw
#         xs = self.vboat.xBounds
#         ds = np.array(self.vboat.dbounds)
#         obs = self.vboat.obstacles
#
#         nObs = len(ds)
#         print "%d Obstacles Found" % (nObs)
#         distances = []
#         angles = []
#         if(nObs is not 0):
#             for i in range(nObs):
#                 disparities = ds[i]
#                 us = [obs[i][0][0], obs[i][1][0]]
#                 vs = [obs[i][0][1], obs[i][1][1]]
#                 z,ux,uy,uz = self.vboat.calculate_distance(umap,us,disparities,vs)
#                 theta = math.atan2(ux,uz)
#
#                 if (z <= 5.0): print " - Obstacle #%d: %.3f m, %.3f rad" % (i+1, z, theta)
#                 distances.append(z)
#                 angles.append(theta)
#         else:
#             distances.append(-1)
#             angles.append(0)
#
#         return distances, angles
#
#     def start(self):
#         count = 0
#         while not rospy.is_shutdown():
#             try:
#                 if not self.flag_use_ros: self.camCallback()
#
#                 # self.vboat.pipeline(self.img, threshU1=5,threshU2=20, threshV2=70)
#                 self.vboat.pipelineTest(self.img, threshU1=0.25, threshU2=0.1, threshV1=5, threshV2=70, timing=True)
#
#                 display_obstacles = self.draw_obstacle_image()
#                 dists,angs = self.update()
#
#                 time = rospy.Time.now()
#                 dist_data = FloatArrayStamped()
#                 dist_data.header.stamp = time
#                 dist_data.header.seq = count
#                 dist_data.data = dists
#
#                 # ang_data = Float32MultiArray()
#                 ang_data = FloatArrayStamped()
#                 ang_data.header.stamp = time
#                 ang_data.header.seq = count
#                 ang_data.data = angs
#
#                 self.dist_pub.publish(dist_data)
#                 self.ang_pub.publish(ang_data)
#                 # print("Publishing")
#
#                 if(self.flag_publish_imgs):
#                     try: self.image_pub.publish(self.bridge.cv2_to_imgmsg(display_obstacles, "bgr8"))
#                     except CvBridgeError as e: print(e)
#
#                 if self.flag_save_imgs:
#                     img_suffix = "frame_" + str(count) + ".png"
#                     if self.rgb is not None:
#                         rgb_file = "rgb_" + img_suffix
#                         rgb_path = os.path.join(self.rgbDir,rgb_file)
#                         self.save_image_to_file(self.rgb,rgb_path)
#                     if self.img is not None:
#                         depth_file = "depth_" + img_suffix
#                         depth_path = os.path.join(self.depthDir,depth_file)
#                         self.save_image_to_file(self.img,depth_path)
#
#                         obs_file = "obstacle_" + img_suffix
#                         obs_path = os.path.join(self.obsDir,obs_file)
#                         self.save_image_to_file(display_obstacles,obs_path)
#
#                 count+=1
#             except: pass
#             self.r.sleep()
#
# if __name__ == "__main__" :
#     vnode = obstacle_detector_node()
#     vnode.start()
#
#
# # ----------------------------------------------------
# # ----------------------------------------------------

# def estimateGndLineCoeffsHough(filtered_vmap,hough_thresh=100, deg_offset=2.0):
#     sets = []
#     # print(filtered_vmap.shape)
#     lines = cv2.HoughLines(filtered_vmap,1,np.pi/180, hough_thresh)
#
#     # if(lines is None): lines = cv2.HoughLines(filtered_vmap,1,np.pi/180, 50)
#     if(lines is None):
#         print("couldnt find hough lines")
#         return sets
#
#     validGnd, avgTheta = isGndPresent(lines)
#     if(validGnd):
#         avgAngle = np.rad2deg(avgTheta)
#         minAng = np.deg2rad(avgAngle - deg_offset)
#         maxAng = np.deg2rad(avgAngle + deg_offset)
#         h,w = filtered_vmap.shape[:2]
#         for i in range(0,len(lines)):
#             for r,theta in lines[i]:
#                 angle = theta-np.pi
#                 if(minAng <= angle <= maxAng):
#                     a = np.cos(theta);            b = np.sin(theta)
#                     x0 = a*r;                     y0 = b*r;
#                     x1 = int(x0 + 1000*(-b));     y1 = int(y0 + 1000*(a))
#                     x2 = int(x0 - 1000*(-b));     y2 = int(y0 - 1000*(a))
#                     pt1 = (x1,y1)
#                     pt2 = (x2,y2)
#                     pts = [(0,h),pt1,pt2,(w, h),(0, h)]
#                     pts = np.array(pts)
#                     sets.append(pts)
#     else: print("No Ground Found")
#     return sets
