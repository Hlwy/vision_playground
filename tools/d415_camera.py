import cv2, time
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

flag_save = 0

# Configure depth and color streams
if flag_save:
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    writerRGB = cv2.VideoWriter("realsense_rgb.avi", fourcc, 60,(640, 480), True)
    writerD = cv2.VideoWriter("realsense_depth.avi", fourcc, 60,(640, 480), True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
# time.sleep(2.0)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        # norm_depth_image = cv2.normalize(depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # norm_depth_image = cv2.cvtColor(norm_depth_image, cv2.COLOR_GRAY2RGB)
        norm_depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # print("Depth Scale: %f" % depth_scale)
        # Show images
        dImg = depth_image*depth_scale
        dRgb = color_image

        if flag_save:
            writerRGB.write(dRgb)
            writerD.write(depth_colormap)

        cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        cv2.imshow('Disparity', norm_depth_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    if flag_save:
        writerRGB.release()
        writerD.release()
