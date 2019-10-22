#!/usr/bin/env python

import cv2
import numpy as np
import rospy, tf
import os, csv, time, math
from threading import Thread

from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image,CameraInfo,CompressedImage, PointCloud2, PointField
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from devices.d415_camera import *

class d415_camera_node:
    def __init__(self):
        rospy.init_node('d415_camera_node')
        rootDir = os.path.dirname(os.path.abspath(__file__))
        imgsPath = os.path.abspath(os.path.join(rootDir, "..")) + "/images"

        self.ns = rospy.get_namespace()
        self.fps = rospy.get_param('~fps', 60)
        self.update_rate = rospy.get_param('~update_rate', 90)
        self.save_path = rospy.get_param('~save_path',imgsPath)
        self.flag_save_imgs = rospy.get_param('~flag_save_imgs', False)
        self.use_float_depth = rospy.get_param('~use_float_depth', True)
        self.publishTf = rospy.get_param('~publish_tf', True)
        self.enable_pc = rospy.get_param('~enable_pointcloud', False)
        self.get_aligned = rospy.get_param('~use_aligned', True)
        self.cam_name = rospy.get_param('~camera_name',"d415")

        self.base_tf_frame = rospy.get_param('~base_tf_frame', "base_link")
        cam_base_tf_frame = rospy.get_param('~cam_base_tf_frame', "_base_link")
        rgb_optical_tf_frame = rospy.get_param('~rgb_optical_tf_frame', "/rgb_optical_frame")
        depth_optical_tf_frame = rospy.get_param('~depth_optical_tf_frame', "/depth_optical_frame")
        self.cam_base_tf_frame = self.cam_name + cam_base_tf_frame
        self.rgb_optical_tf_frame = self.cam_name + rgb_optical_tf_frame
        self.depth_optical_tf_frame = self.cam_name + depth_optical_tf_frame

        depth_image_topic = rospy.get_param('~depth_image_topic', "/depth/image_raw")
        image_topic_rgb = rospy.get_param('~rgb_image_topic', "/rgb/image_raw")
        depth_info_topic = rospy.get_param('~depth_info_topic', "/depth/camera_info")
        rgb_info_topic = rospy.get_param('~rgb_info_topic', "/rgb/camera_info")
        cloud_topic = rospy.get_param('~cloud_topic', "/cloud")
        self.depth_image_topic = self.ns + self.cam_name + depth_image_topic
        self.image_topic_rgb = self.ns + self.cam_name + image_topic_rgb
        self.depth_info_topic = self.ns + self.cam_name + depth_info_topic
        self.rgb_info_topic = self.ns + self.cam_name + rgb_info_topic
        self.cloud_topic = self.ns + self.cam_name + cloud_topic

        self.depth_info_pub = rospy.Publisher(self.depth_info_topic,CameraInfo,queue_size=10)
        self.rgb_info_pub = rospy.Publisher(self.rgb_info_topic,CameraInfo,queue_size=10)
        self.rgb_pub = rospy.Publisher(self.image_topic_rgb,Image,queue_size=10)
        self.depth_pub = rospy.Publisher(self.depth_image_topic,Image,queue_size=10)
        if(self.enable_pc): self.cloud_pub = rospy.Publisher(self.cloud_topic,PointCloud2,queue_size=10)
        else: self.cloud_pub = None

        self.bridge = CvBridge()
        self.br = tf.TransformBroadcaster()

        res = (848, 480)
        self.cam = CameraD415(flag_save=False,use_statistics=False,fps=self.fps, depth_resolution=res, verbose=True)
        self.intr = self.cam.get_intrinsics()
        self.extr = self.cam.get_extrinsics()

        fxd = self.intr["depth"].fx
        fyd = self.intr["depth"].fy
        ppxd = self.intr['depth'].ppx
        ppyd = self.intr['depth'].ppy

        fxc = self.intr["color"].fx
        fyc = self.intr["color"].fy
        ppxc = self.intr['color'].ppx
        ppyc = self.intr['color'].ppy

        self.focal = [fxd, fyd]
        self.ppoint = [ppxd, ppyd]
        self.baseline = self.extr.translation[0]

        print("""Camera Info:
------------
Color:
    fx, fy: %.2f, %.2f
    cx, cy: %.2f, %.2f

Depth:
    fx, fy: %.2f, %.2f
    cx, cy: %.2f, %.2f
    baseline: %.2f
------------""" % (fxc,fyc,ppxc,ppyc, fxd,fyd,ppxd,ppyd,self.baseline))

        self.rgb_info_msg = CameraInfo()
        self.rgb_info_msg.header.frame_id = self.rgb_optical_tf_frame
        self.rgb_info_msg.width = self.intr['color'].width
        self.rgb_info_msg.height = self.intr['color'].height
        self.rgb_info_msg.distortion_model = "plumb_bob"
        self.rgb_info_msg.K = [fxc, 0, ppxc, 0, fyc, ppyc, 0, 0, 1]
        self.rgb_info_msg.D = [0, 0, 0, 0]
        self.rgb_info_msg.P = [fxc, 0, ppxc, 0, 0, fyc, ppyc, 0, 0, 0, 1, 0]

        self.depth_info_msg = CameraInfo()
        self.depth_info_msg.header.frame_id = self.depth_optical_tf_frame
        self.depth_info_msg.width = self.intr['depth'].width
        self.depth_info_msg.height = self.intr['depth'].height
        self.depth_info_msg.distortion_model = "plumb_bob"
        self.depth_info_msg.K = [fxd, 0, ppxd, 0, fyd, ppyd, 0, 0, 1]
        self.depth_info_msg.D = [0, 0, 0, 0]
        self.depth_info_msg.P = [fxd, 0, ppxd, 0, 0, fyd, ppyd, 0, 0, 0, 1, 0]

        self.pc2_msg_fields = [PointField('x', 0, PointField.FLOAT32, 1),PointField('y', 4, PointField.FLOAT32, 1),PointField('z', 8, PointField.FLOAT32, 1)]

        self.r = rospy.Rate(self.update_rate)
        self.rgb = []
        self.depth = []
        self.disparity = []

        self.count = 0
        self.camcount = 0
        self.prev_func_count = None
        self.flag_kill_thread = False

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
        print("[INFO] d415_camera_node --- Started...")

        if(self.enable_pc):
            self.cloudPubThread = Thread(target=self.cloudPubFunc,args=())
            self.cloudPubThread.start()
        else:
            self.cloudPubThread = None

    def __del__(self):
        self.flag_kill_thread = True
        if(self.cloudPubThread is not None):
            print("[INFO] d415_camera_node::__del__() ---- Killing ready_flag publisher Thread...")
            self.cloudPubThread.join()

    def cloudPubFunc(self):
        while not self.flag_kill_thread:
            if(rospy.is_shutdown()): break
            if(self.prev_func_count is None): self.prev_func_count = self.camcount

            if(self.prev_func_count != self.camcount):
                self.prev_func_count = self.camcount
                # print("[INFO] d415_camera_node::loop() --- Getting pointcloud...")
                cloud = self.cam.get_pointcloud()
                # points = cloud.tolist()
                header = Header()
                header.seq = self.camcount
                header.stamp = rospy.Time.now()
                header.frame_id = self.depth_optical_tf_frame
                pc2_msg = point_cloud2.create_cloud(header, self.pc2_msg_fields, np.asarray(cloud.get_vertices()) )
                # pc2_msg = point_cloud2.create_cloud(header, self.pc2_msg_fields, points )
                self.cloud_pub.publish(pc2_msg)

    def camCallback(self, _rgb, _depth, debug=False):
        if(debug): print("[INFO] d415_camera_node::camCallback() --- Converting frames...")
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

        if(debug): print("[INFO] d415_camera_node::camCallback() --- Publishing ROS frames...")
        try:
            curT = rospy.Time.now()
            rgbImgMsg = self.bridge.cv2_to_imgmsg(self.rgb, "bgr8")
            if(self.use_float_depth): depthImgMsg = self.bridge.cv2_to_imgmsg(_depth*self.cam.dscale, "32FC1")
            else: depthImgMsg = self.bridge.cv2_to_imgmsg(self.depth, "8UC1")

            rgbImgMsg.header.stamp = curT
            rgbImgMsg.header.seq = self.camcount + 1
            # rgbImgMsg.header.frame_id = self.rgb_optical_tf_frame
            rgbImgMsg.header.frame_id = self.cam_base_tf_frame

            depthImgMsg.header.stamp = curT
            depthImgMsg.header.seq = self.camcount + 1
            # depthImgMsg.header.frame_id = self.depth_optical_tf_frame
            depthImgMsg.header.frame_id = self.cam_base_tf_frame

            self.rgb_pub.publish(rgbImgMsg)
            self.depth_pub.publish(depthImgMsg)

            infoMsgRgb, infoMsgDepth = self.create_camera_info_msg()
            infoMsgRgb.header.stamp = rgbImgMsg.header.stamp
            infoMsgDepth.header.stamp = depthImgMsg.header.stamp

            self.rgb_info_pub.publish(infoMsgRgb)
            self.depth_info_pub.publish(infoMsgDepth)

        except CvBridgeError as e: print(e)
        self.camcount+=1

    def create_camera_info_msg(self):
        curT = rospy.Time.now()
        self.rgb_info_msg.header.stamp = curT
        self.rgb_info_msg.header.seq = self.count

        self.depth_info_msg.header.stamp = curT
        self.depth_info_msg.header.seq = self.count
        if(self.publishTf):
            self.br.sendTransform((0,0,0), tf.transformations.quaternion_from_euler(0,0,0), curT, self.depth_optical_tf_frame,self.cam_base_tf_frame)
            self.br.sendTransform((0,0,0), tf.transformations.quaternion_from_euler(0,0,0), curT, self.rgb_optical_tf_frame,self.cam_base_tf_frame)
            # self.br.sendTransform((0.21,0.0,0.02), tf.transformations.quaternion_from_euler(-(np.pi/2.0), 0, -(np.pi/2.0)), rospy.Time.now(), self.cam_base_tf_frame,self.base_tf_frame)
            self.br.sendTransform((0.0,0.0,0.0), tf.transformations.quaternion_from_euler(-(np.pi/2.0), 0, -(np.pi/2.0)), rospy.Time.now(), self.cam_base_tf_frame,self.base_tf_frame)
        return self.rgb_info_msg, self.depth_info_msg

    def save_image_to_file(self, img, path):
        try: cv2.imwrite(path, img)
        except:
            print("[ERROR] d415_camera_node.py ---- Could not save image to file \'%s\'" % path)
            pass

    def start(self, debug=False):
        dt = 0
        count = savecnt = 0
        while not rospy.is_shutdown():
            if(rospy.is_shutdown()): break
            try:
                # t0 = time.time()
                rgb, depth = self.cam.read(flag_aligned=self.get_aligned)
                if((rgb is None) and (depth is None)):
                    print("[INFO] d415_camera_node::loop() --- Grabbed frames both None, Skipping...")
                    continue
                elif((rgb is not None) and (depth is None)):
                    if(debug): print("[INFO] d415_camera_node::loop() --- Depth frame grabbed is None")
                    depth = np.zeros_like(rgb)
                elif((rgb is None) and (depth is not None)):
                    if(debug): print("[INFO] d415_camera_node::loop() --- RGB frame grabbed is None")
                    rgb = np.zeros_like(depth)

                if(debug): print("[INFO] d415_camera_node::loop() --- Successful frame grab")
                self.camCallback(rgb, depth)
                # t1 = time.time()
                # dt = t1 - t0
                # print("[INFO] d415_camera_node::loop() --- FrameRate = %.2f Hz (%f seconds)" % (1/dt,dt))

                if self.flag_save_imgs:
                    img_suffix = "frame_" + str(savecnt) + ".png"
                    if self.rgb is not None:
                        rgb_file = "rgb_" + img_suffix
                        rgb_path = os.path.join(self.rgbDir,rgb_file)
                        self.save_image_to_file(self.rgb,rgb_path)
                    if self.disparity is not None:
                        depth_file = "depth_" + img_suffix
                        depth_path = os.path.join(self.depthDir,depth_file)
                        self.save_image_to_file(self.disparity,depth_path)
                    savecnt += 1
                count+=1
                self.count+=1
            except: pass
            self.r.sleep()

if __name__ == "__main__" :
    vnode = d415_camera_node()
    vnode.start(debug=False)
