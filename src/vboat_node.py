import cv2
import rospy
import rosbag
import numpy as np
import os, csv, time, argparse
from matplotlib import pyplot as plt

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from VBOATS import VBOATS

class vboat_node:
    def __init__(self):
        rospy.init_node('vboat_pipeline')
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/obstacles",Image,queue_size=1000)
        self.image_sub = rospy.Subscriber("/ugv/d415/depth/image_raw",Image,self.callback)

        self.vboat = VBOATS()
        self.vboat.dead_x = 0
        self.vboat.dead_y = 5
        self.r = rospy.Rate(40)
        self.img = []
        self.obs_disp = []

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            cv_image = np.uint16(cv_image)
            self.img = np.copy(cv_image)
        except CvBridgeError as e:
            print(e)
            pass

        cv2.imshow("image", cv_image)
        cv2.waitKey(3)

        try:
            self.vboat.pipeline(np.uint8(cv_image), threshU1=8,threshU2=20, threshV2=70)
            display_obstacles = cv2.cvtColor(self.vboat.img, cv2.COLOR_GRAY2BGR)
            for ob in self.vboat.obstacles:
                cv2.rectangle(display_obstacles,ob[0],ob[1],(150,0,0),1)

            self.obs_disp = np.copy(display_obstacles)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(display_obstacles, "bgr8"))
        except:
            pass



    def start(self):
        while not rospy.is_shutdown():
            self.r.sleep()

if __name__ == "__main__" :
    vnode = vboat_node()
    vnode.start()
