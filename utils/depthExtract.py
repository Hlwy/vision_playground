import cv2, sys
import rosbag
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

args = sys.argv[1]
bag = rosbag.Bag(str(args))
_topics = ['camera/depth/image_rect_raw', 'rosout']

topics = bag.get_type_and_topic_info()[1].keys()
types = []
# for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
#     types.append(bag.get_type_and_topic_info()[1].values()[i][0])
#
# print(types)
count = 0
nMsgs = bag.get_message_count(topic_filters="/camera/depth/image_rect_raw")
for topic, msg, t in bag.read_messages(topics='/camera/depth/image_rect_raw'):
    cv_image = bridge.imgmsg_to_cv2(msg, "8UC1")
    print(msg.header)
    name = './frames/frame_' + str(count) + '.png'
    cv2.imwrite(name, cv_image)
    count+=1
    cv2.imshow('topic', cv_image)
    key=cv2.waitKey(1)
    if key==1048603:
        exit(1);


bag.close()
print(nMsgs)
