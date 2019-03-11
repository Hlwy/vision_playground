import numpy as np
import cv2, time, pprint
import pyrealsense2 as rs
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

class CameraD415(object):
    def __init__(self, flag_save=0,use_statistics=0):
        self.fps = 60
        self.flag_save = flag_save
        self.frames = None

        # Attempt to establish camera configuration
        try:
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)
        except:
            self.config = None
            print("[ERROR] Could not establish CameraD415 Config!")

        # Attempt to establish camera pipeline
        try: self.pipeline = rs.pipeline()
        except:
            self.pipeline = None
            print("[ERROR] Could not establish CameraD415 Pipeline!")

        # Attempt to start streaming device
        try: self.profile = self.pipeline.start(self.config)
        except:
            self.profile = None
            print("[ERROR] Could not establish CameraD415 Profile!")

        self.dscale = self.get_depth_scale()
        self.intrinsics = self.get_intrinsics()
        self.extrinsics = self.get_extrinsics()

        if(self.flag_save): # Configure depth and color streams
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.writerD = cv2.VideoWriter("realsense_d415_depth.avi", fourcc, self.fps,(640, 480), True)
            self.writerRGB = cv2.VideoWriter("realsense_d415_rgb.avi", fourcc, self.fps,(640, 480), True)
        else:
            self.writerD = None
            self.writerRGB = None
            print("[ERROR] Could not establish CameraD415 VideoWriter's!")


        if(use_statistics): self.dmax_avg = self.calculate_statistics(duration=5.0)
        else: self.dmax_avg = 65535

    def __del__(self):
        print("[INFO] Closing CameraD415 object")
        if(self.pipeline is not None): self.pipeline.stop()
        if(self.writerD is not None): self.writerD.release()
        if(self.writerRGB is not None): self.writerRGB.release()

    def get_intrinsics(self):
        frames = self.pipeline.wait_for_frames()
        device_intrinsics = {}
        for stream in self.profile.get_streams():
            skey = stream.stream_name().lower()
            device_intrinsics[skey] = stream.as_video_stream_profile().get_intrinsics()
        return device_intrinsics

    def get_extrinsics(self):
        frames = self.pipeline.wait_for_frames()
        streams = self.profile.get_streams()
        device_extrinsics = streams[0].as_video_stream_profile().get_extrinsics_to(streams[1])
        return device_extrinsics

    def get_depth_scale(self):
        if(self.profile is not None):
            depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        else: depth_scale = -1
        return depth_scale

    def read(self):
        self.frames = self.pipeline.wait_for_frames()
        rgb = self.get_rgb_image()
        depth = self.get_depth_image()
        return rgb, depth

    def get_rgb_image(self):
        if(self.frames is not None):
            frame = self.frames.get_color_frame()
            img = np.asanyarray(frame.get_data())
        else:
            img = None
        return img

    def get_depth_image(self):
        if(self.frames is not None):
            frame = self.frames.get_depth_frame()
            img = np.asanyarray(frame.get_data(),dtype=np.float32)
        else:
            img = None
        return img

    def loop(self):
        while True:
            rgb, depth = self.read()
            if((rgb is None) or (depth is None)):
                continue
            tmp = depth/self.dmax_avg
            depth = np.uint8(tmp*255)
            depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)
            # depth = np.uint8(depth)
            if(self.writerD is not None):
                self.writerD.write(depth)
            if(self.writerRGB is not None):
                self.writerRGB.write(rgb)

    def calculate_statistics(self, duration=10.0):
        print("Calculating Average Max Disparity.....")
        t0 = time.time()
        sum, count = 0, 0
        while True:
            t1 = time.time()
            dt = t1 - t0
            if(dt<= duration):
                _, depth = self.read()
                if(depth is None): continue
                sum += np.max(depth)
                count+=1
            else: break
        avg = sum / float(count)
        print("Average Max Disparity, Sum, Count: %.3f, %.3f, %d" % (avg,sum,count))
        return avg

if __name__ == '__main__':
    cam = CameraD415(flag_save=1,use_statistics=0)
    print("Looping...")
    cam.loop()
    print("Exiting...")
