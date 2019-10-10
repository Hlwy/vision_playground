#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import math, os, time
import tf

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

from offset_vboats.msg import FloatArrayStamped, DetectedObstacleData
from VBOATS import VBOATS
from devices.d415_camera import *
from hyutils.sys_utils import *

class obstacle_detector_node:
    def __init__(self):
        # rospy.init_node('obstacle_detector')
        rootDir = os.path.dirname(os.path.abspath(__file__))
        tmpPath = os.path.abspath(os.path.join(rootDir, "..")) + "/images"

        self.fps = rospy.get_param('~fps', 30)
        self.save_path = rospy.get_param('~save_path',tmpPath)
        self.flag_use_ros = rospy.get_param('~flag_use_ros', False)
        self.flag_save_imgs = rospy.get_param('~flag_save_imgs', False)
        self.flag_publish_imgs = rospy.get_param('~flag_publish_imgs', True)
        self.flag_publish_rgb = rospy.get_param('~flag_publish_rgb', True)
        self.flag_do_obstacle_detection = rospy.get_param('~flag_do_obstacle_detection', True)
        self.base_frame = rospy.get_param('~base_frame', "base_link")
        self.cam_base_frame = rospy.get_param('~cam_base_frame', "d415/rgb_base_frame")
        self.cam_frame = rospy.get_param('~cam_frame', "d415/rgb_optical_frame")
        self.image_topic = rospy.get_param('~image_topic', rospy.get_namespace()+"d415/depth/image_raw")
        self.image_topic_rgb = rospy.get_param('~rgb_image_topic', rospy.get_namespace()+"d415/rgb/image_raw")
        self.camera_info_topic = rospy.get_param('~camera_info_topic', rospy.get_namespace()+"d415/rgb/camera_info")

        self.dist_pub = rospy.Publisher(rospy.get_namespace()+"obstacles/distances",FloatArrayStamped,queue_size=10)
        self.ang_pub = rospy.Publisher(rospy.get_namespace()+"obstacles/angles",FloatArrayStamped,queue_size=10)
        self.dist_and_ang_pub = rospy.Publisher(rospy.get_namespace()+"obstacles/distances_and_angles",DetectedObstacleData,queue_size=10)
        self.r = rospy.Rate(30)

        if self.flag_publish_rgb:
            self.rgb_pub = rospy.Publisher(self.image_topic_rgb,Image,queue_size=10)
            self.pub_info = rospy.Publisher(self.camera_info_topic,CameraInfo,queue_size=10)
        else:
            self.rgb_pub = None
            self.pub_info = None

        self.depth_pub = rospy.Publisher(self.image_topic,Image,queue_size=10)

        self.bridge = CvBridge()

        if self.flag_use_ros:
            if self.flag_publish_imgs:
                self.image_pub = rospy.Publisher(rospy.get_namespace()+"vboat/obstacles/image",Image,queue_size=10)
            else: self.image_pub = None

            self.image_sub = rospy.Subscriber(self.image_topic, Image, self.rosCallback)
            self.cam = None
        else:
            if self.flag_publish_imgs:
                self.image_pub = rospy.Publisher(rospy.get_namespace()+"vboat/obstacles/image",Image,queue_size=10)
            else: self.image_pub = None
            self.image_sub = None
            self.cam = CameraD415(flag_save=False, use_statistics=False, fps=self.fps)
            self.intr = self.cam.get_intrinsics()

        self.vboat = VBOATS()
        self.vboat.debug = False
        self.vboat.dead_x = 0
        self.vboat.dead_y = 3       # Configurable: 0-5

        self.img = []
        self.rgb = []
        self.obs_disp = []

        self.roscount = 0
        self.camcount = 0
        self.count = 0
        self.disp_obs = None

        self.info_msg = CameraInfo()
        fx = self.intr['color'].fx
        fy = self.intr['color'].fy
        cx = self.intr['color'].ppx
        cy = self.intr['color'].ppy

        self.info_msg.header.frame_id = self.cam_frame
        self.info_msg.width = self.intr['color'].width
        self.info_msg.height = self.intr['color'].height
        self.info_msg.K = [fx, 0, cx,
                          0, fy, cy,
                          0, 0, 1]

        self.info_msg.D = [0, 0, 0, 0]

        self.info_msg.P = [fx, 0, cx, 0,
                          0, fy, cy, 0,
                          0, 0, 1, 0]

        if self.flag_save_imgs:
            self.rgb_parent = create_new_directory("rgb",self.save_path)
            self.depth_parent = create_new_directory("depth",self.save_path)
            self.obs_parent = create_new_directory("processed",self.save_path)
        else:
            self.rgb_parent = None
            self.depth_parent = None
            self.obs_parent = None
        self.br = tf.TransformBroadcaster()

    def rosCallback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            cv_image = np.float32(cv_image)
            tmp = cv_image/65535
            depth = np.uint8(tmp*255)
            self.img = np.copy(depth)
            self.roscount+=1
        except CvBridgeError as e:
            print(e)

    def camCallback(self, timing=False, debug=False):
        if(debug): print("[INFO] camCallback Starting...")
        if(timing): t0 = time.time()
        if(debug): print("[INFO] camCallback reading frames...")
        rgb, depth = self.cam.read()
        if((rgb is None) or (depth is None)):
            print("[INFO] obstacle_detector_node::camCallback --- Both frames are None")
            return None
        tmp = depth/self.cam.dmax_avg
        depth = np.uint8(tmp*255)
        # depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)
        # self.img = np.copy(depth)
        if(debug): print("[INFO] camCallback copying frames...")
        self.depth = np.copy(depth)
        self.rgb = np.copy(rgb)

        if(debug): print("[INFO] camCallback converting frames...")
        depth_img_msg = self.bridge.cv2_to_imgmsg(self.depth, "8UC1")
        rgb_img_msg = self.bridge.cv2_to_imgmsg(self.rgb, "bgr8")
        if(debug): print("[INFO] camCallback publishing frames...")
        self.depth_pub.publish(depth_img_msg)
        self.rgb_pub.publish(rgb_img_msg)

        if self.flag_publish_rgb:
            try:
                if(debug): print("[INFO] camCallback creating camera info msg...")
                msg = self.create_camera_info_msg()
                msg.header.stamp = rgb_img_msg.header.stamp
                if(debug): print("[INFO] camCallback publishing camera info msg...")
                self.pub_info.publish(msg)
            except CvBridgeError as e:
                print(e)

        self.camcount+=1
        if(debug): print("[INFO] camCallback checking timing...")
        if(timing):
            t1 = time.time()
            dt = t1 - t0
            print("[INFO] obstacle_detector_node::camCallback() --- Took %f seconds (%.2f Hz) to complete" % (dt, 1/dt))
        return depth

    def create_camera_info_msg(self):
        curT = rospy.Time.now()
        # self.info_msg.header.stamp = rospy.Time.now()
        self.info_msg.header.seq = self.count

        self.br.sendTransform((0, 0, 0), tf.transformations.quaternion_from_euler(-(np.pi/2.0), 0, (np.pi/2.0)), curT, self.cam_frame,self.cam_base_frame)
        self.br.sendTransform((0.21,0.0,0.02), tf.transformations.quaternion_from_euler(0, 0, np.pi), curT, self.cam_base_frame,self.base_frame)
        return self.info_msg

    def draw_obstacle_image(self):
        display_obstacles = cv2.cvtColor(self.vboat.img, cv2.COLOR_GRAY2BGR)

        for ob in self.vboat.obstacles:
            cv2.rectangle(display_obstacles,ob[0],ob[1],(150,0,0),1)
        disp_obs = np.copy(display_obstacles)
        return disp_obs

    def save_image_to_file(self, img, path):
        try:
            cv2.imwrite(path, img)
        except:
            print("[ERROR] vboat_node.py ---- Could not save image to file \'%s\'" % path)
            pass

    def start(self):
        self.count = 0
        while not rospy.is_shutdown():
            try:
                if not self.flag_use_ros:
                    # print("[INFO] Grabbing Frames...")
                    self.camCallback()

                if(self.flag_do_obstacle_detection):
                    # self.vboat.pipelineV0(self.img, threshU1=5, threshU2=10, threshV2=70, timing=True)
                    # self.vboat.pipelineTest(self.img, threshU1=0.25, threshU2=0.1, threshV1=5, threshV2=70, timing=True)
                    self.vboat.pipelineV1(self.depth, timing=True)

                    dists,angs = self.vboat.extract_obstacle_information()

                    time = rospy.Time.now()
                    dist_data = FloatArrayStamped()
                    dist_data.header.stamp = time
                    dist_data.header.seq = self.count
                    dist_data.data = dists
                    self.dist_pub.publish(dist_data)

                    ang_data = FloatArrayStamped()
                    ang_data.header.stamp = time
                    ang_data.header.seq = self.count
                    ang_data.data = angs
                    self.ang_pub.publish(ang_data)

                    dist_and_ang_data = DetectedObstacleData()
                    dist_and_ang_data.distance_data = dist_data
                    dist_and_ang_data.angle_data = ang_data
                    self.dist_and_ang_pub.publish(dist_and_ang_data)

                if self.flag_save_imgs:
                    img_suffix = "frame_" + str(self.count) + ".png"
                    if self.rgb is not None:
                        rgb_file = "rgb_" + img_suffix
                        rgb_path = os.path.join(self.rgb_parent,rgb_file)
                        self.save_image_to_file(self.rgb,rgb_path)
                    if self.img is not None:
                        depth_file = "depth_" + img_suffix
                        depth_path = os.path.join(self.depth_parent,depth_file)
                        self.save_image_to_file(self.img,depth_path)
                        if(self.flag_do_obstacle_detection):
                            obs_file = "obstacle_" + img_suffix
                            obs_path = os.path.join(self.obs_parent,obs_file)
                            display = self.draw_obstacle_image()
                            self.save_image_to_file(display,obs_path)
                    print("Saved [%d] Images" % self.count)

                if self.flag_publish_imgs and self.flag_do_obstacle_detection:
                    display = self.draw_obstacle_image()
                    try:
                        self.image_pub.publish(self.bridge.cv2_to_imgmsg(display, "bgr8"))
                    except CvBridgeError as e:
                        print(e)

                self.count+=1
            except:
                # print("No Obstacles..")
                pass

            self.r.sleep()

if __name__ == "__main__" :
    # import argparse
    #
    # ap = argparse.ArgumentParser(description='Extract images from a video file.')
    # ap.add_argument("--save_imgs", "-s", action="store_true", help="Flag to save images to file")
    # args = vars(ap.parse_args())
    # flag_save = args["save_imgs"]

    rospy.init_node('obstacle_detector')
    vnode = obstacle_detector_node()
    vnode.start()
