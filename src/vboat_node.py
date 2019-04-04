#!/usr/bin/env python

import cv2
import numpy as np
import rospy, rosbag, tf
import os, csv, time, argparse, math
from matplotlib import pyplot as plt

from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from test_navigation.msg import FloatArrayStamped
from VBOATS import VBOATS

class obstacle_detector_node:
    def __init__(self):
        rospy.init_node('obstacle_detector')
        self.bridge = CvBridge()


        self.base_tf = rospy.get_param('~base_tf', "/ugv1/odom")
        self.cam_tf = rospy.get_param('~cam_tf', "/ugv1/d415_camera_depth_optical_frame")
        self.image_topic = rospy.get_param('~image_topic', "/ugv1/d415/depth/image_raw")

        self.image_pub = rospy.Publisher("vboat/obstacles/image",Image,queue_size=1000)
        # self.dist_pub = rospy.Publisher("/obstacles/distances",Float32MultiArray,queue_size=1000)
        # self.ang_pub = rospy.Publisher("/obstacles/angles",Float32MultiArray,queue_size=1000)
        self.dist_pub = rospy.Publisher("/obstacles/distances",FloatArrayStamped,queue_size=1000)
        self.ang_pub = rospy.Publisher("/obstacles/angles",FloatArrayStamped,queue_size=1000)
        self.image_sub = rospy.Subscriber(self.image_topic,Image,self.callback)

        self.vboat = VBOATS()
        self.vboat.dead_x = 0
        self.vboat.dead_y = 3
        self.vboat.flag_simulation = True
        self.r = rospy.Rate(40)
        self.img = []
        self.obs_disp = []
        self.flag_initialization = False
        self.nSamples = 100
        self.dSum = 0
        self.dmaxAvg = 0
        self.count = 0
        self.disp_obs = None

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            cv_image = np.float32(cv_image)
            tmp = cv_image/65535
            depth = np.uint8(tmp*255)
            self.img = np.copy(depth)
        except CvBridgeError as e: print(e)

    def get_camera_pose(self, cam_frame = '/ugv1/d415_camera_depth_optical_frame', base_frame = '/ugv1/odom'):
        listener = tf.TransformListener()
        listener.waitForTransform(base_frame,cam_frame, rospy.Time(0), rospy.Duration(8.0))
        (trans,rot) = listener.lookupTransform(base_frame,cam_frame, rospy.Time(0))
        roll,pitch,yaw = tf.transformations.euler_from_quaternion(rot)

        pose = np.array(trans+[np.rad2deg(yaw)])
        Tmat = np.array(trans).T
        Rmat = tf.transformations.euler_matrix(roll,pitch,yaw,axes='sxyz')
        return pose, Rmat, Tmat

    def update(self):
        umap = self.vboat.umap_raw
        xs = self.vboat.xBounds
        ds = np.array(self.vboat.dbounds)
        obs = self.vboat.obstacles
        # ppx = 320.551; ppy = 232.202; fx = 626.464; fy = 626.464; b = 0.055
        # pp = [ppx, ppy];   focal = [fx,fy]
        dGain = (65535/255)

        obid = 0; i = 0
        X0 = np.array([ [5.0, -3.0, 0.21336] ])

        pose,RotM,Tmat = self.get_camera_pose(self.cam_tf,self.base_tf)

        Xk = pose[:2]; yaw = pose[-1]; ang = np.deg2rad(yaw)
        x1 = X0[obid,0];     y1 = X0[obid,1]
        dx = x1 - Xk[0];     dy = y1 - Xk[1]
        true_dist = np.sqrt(dx*dx + dy*dy)-X0[obid,2]
        nObs = len(ds)
        # print("[%d] Obstacles Found" % nObs)
        distances = []
        angles = []
        if(nObs is not 0):
            for i in range(nObs):
                disparities = ds[i]
                us = [obs[i][0][0], obs[i][1][0]]
                vs = [obs[i][0][1], obs[i][1][1]]
                z,ux,uy,uz = self.vboat.calculate_distance(umap,us,disparities,vs)

                theta = math.acos((uz/z))

                distances.append(z)
                angles.append(theta)

                pxl = np.array([ [ux],[uy],[z] ])
                RotM = RotM[:3,:3]
                T = Tmat.reshape((3, 1))*-1
                pos = self.vboat.transform_pixel_to_world(RotM,pxl,T)

                strs = []
                strs.append(', '.join(map(str, np.around(pos.T[0][:2],3))))
                strs.append(', '.join(map(str, np.around(X0[obid,:2],3))))

#                 print(
# """
# Detected Obstacle Stats:
# ========================
#     * Distances (True, Est.)    : %.3f, %.3f
#     * Estimated Position (X,Y,Z): %s
#     * True Position (X,Y,Z)     : %s
# """
#         % (true_dist,z,strs[0],strs[1]) )
        else:
            distances.append(-1)
            angles.append(0)
        return distances, angles


    def start(self):
        count = 0
        while not rospy.is_shutdown():
            try:
                self.vboat.pipeline(self.img, threshU1=5,threshU2=20, threshV2=70)
                display_obstacles = cv2.cvtColor(self.vboat.img, cv2.COLOR_GRAY2BGR)

                for ob in self.vboat.obstacles:
                    cv2.rectangle(display_obstacles,ob[0],ob[1],(150,0,0),1)

                self.disp_obs = np.copy(display_obstacles)
                dists,angs = self.update()

                time = rospy.Time.now()
                # dist_data = Float32MultiArray()
                dist_data = FloatArrayStamped()
                dist_data.header.stamp = time
                dist_data.header.seq = count
                dist_data.data = dists
                dist_data.data = dists

                # ang_data = Float32MultiArray()
                ang_data = FloatArrayStamped()
                ang_data.header.stamp = time
                ang_data.header.seq = count
                ang_data.data = angs
                ang_data.data = angs
                self.dist_pub.publish(dist_data)
                self.ang_pub.publish(ang_data)
                count+=1
                # print("Publishing")

                try:
                    self.obs_disp = np.copy(display_obstacles)
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(display_obstacles, "bgr8"))
                except CvBridgeError as e: print(e)
            except:pass
            self.r.sleep()
            # rospy.spin()

if __name__ == "__main__" :
    vnode = obstacle_detector_node()
    vnode.start()
