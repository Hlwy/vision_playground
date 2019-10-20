import numpy as np
import cv2, time, pprint
import threading
import pyrealsense2 as rs
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

class CameraD415(object):
    def __init__(self, flag_save=False,use_statistics=False, fps=60,depth_resolution=(640,480),rgb_resolution=(640,480),verbose=False):
        self.lock = threading.Lock()
        self.fps = fps
        self.depth_resolution = depth_resolution
        self.rgb_resolution = rgb_resolution
        self.flag_save = flag_save
        self.frames = None
        self.aligned_frames = None
        self.rgb_frame = None
        self.depth_frame = None

        self.writerD = None
        self.writerRGB = None

        ctx = rs.context()
        # print(len(ctx.devices))
        for d in ctx.devices:
            if verbose: print ('Found device: ', \
            d.get_info(rs.camera_info.name), ' ', \
            d.get_info(rs.camera_info.serial_number))
        self.dev = ctx.devices[0]
        print("[INFO] CameraD415() -- Resetting Hardware...")
        self.dev.hardware_reset()
        time.sleep(5)
        is_success = self.hardware_startup(fps=self.fps,depth_resolution=self.depth_resolution,rgb_resolution=self.rgb_resolution)
        if not is_success:
            is_success = self.reset()
            if not is_success:
                is_retry_success = self.reset()
                if not is_retry_success:
                    raise ValueError('A very specific bad thing happened.')
        else: print("[INFO] CameraD415() -- Initialization successful!!!")

        self.dscale = self.get_depth_scale()
        self.intrinsics = self.get_intrinsics()
        self.extrinsics = self.get_extrinsics()
        self.align = rs.align(rs.stream.color)

        if(self.flag_save): # Configure depth and color streams
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.writerD = cv2.VideoWriter("realsense_d415_depth.avi", fourcc, self.fps,self.depth_resolution, True)
            self.writerRGB = cv2.VideoWriter("realsense_d415_rgb.avi", fourcc, self.fps,self.rgb_resolution, True)

        if(use_statistics): self.dmax_avg = self.calculate_statistics(duration=5.0)
        else: self.dmax_avg = 65535

    def reset(self):
        print("[INFO] CameraD415() -- Resetting Camera...")
        self.dev.hardware_reset()
        is_success = self.hardware_startup(fps=self.fps,depth_resolution=self.depth_resolution,rgb_resolution=self.rgb_resolution)
        return is_success

    def hardware_startup(self,fps=60,depth_resolution=(640,480),rgb_resolution=(640,480)):
        # Attempt to establish camera configuration
        try:
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, rgb_resolution[0], rgb_resolution[1], rs.format.bgr8, self.fps)
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

        if self.profile is None: return False
        else: return True

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

    def read(self, flag_aligned = False):
        self.frames = self.pipeline.wait_for_frames()
        if(flag_aligned): self.aligned_frames = self.align.process(self.frames)

        rgb = self.get_rgb_image(flag_aligned=flag_aligned)
        depth = self.get_depth_image(flag_aligned=flag_aligned)

        return rgb, depth

    def get_rgb_image(self, flag_aligned = False):
        self.lock.acquire()
        if(flag_aligned and (self.aligned_frames is not None)):
            frame = self.aligned_frames.get_color_frame()
            self.rgb_frame = frame
            img = np.asanyarray(frame.get_data())
        elif(self.frames is not None):
            frame = self.frames.get_color_frame()
            self.rgb_frame = frame
            img = np.asanyarray(frame.get_data())
        else: img = None
        self.lock.release()
        return img

    def get_depth_image(self, flag_aligned = False):
        self.lock.acquire()
        if(flag_aligned and (self.aligned_frames is not None)):
            frame = self.aligned_frames.get_depth_frame()
            self.depth_frame = frame
            img = np.asanyarray(frame.get_data(),dtype=np.float32)
        elif(self.frames is not None):
            frame = self.frames.get_depth_frame()
            self.depth_frame = frame
            img = np.asanyarray(frame.get_data(),dtype=np.float32)
        else: img = None
        self.lock.release()
        return img

    def get_pointcloud(self):
        pc = rs.pointcloud()
        # points = rs.points()
        self.lock.acquire()
        if(self.depth_frame is not None):
            # img = np.asanyarray(self.depth_frame.get_data(),dtype=np.float32)
            # [height, width] = img.shape
            # nx = np.linspace(0, width-1, width)
            # ny = np.linspace(0, height-1, height)
            # u, v = np.meshgrid(nx, ny)
            # x = (u.flatten() - self.intrinsics["depth"].ppx)/self.intrinsics["depth"].fx
            # y = (v.flatten() - self.intrinsics["depth"].ppy)/self.intrinsics["depth"].fy
            #
            # z = img.flatten() / 1000;
            # x = np.multiply(x,z)
            # y = np.multiply(y,z)
            #
            # x = x[np.nonzero(z)]
            # y = y[np.nonzero(z)]
            # z = z[np.nonzero(z)]
            # points = np.concatenate([[x,y,z]]).T

            pc.map_to(self.rgb_frame)
            points = pc.calculate(self.depth_frame)
            # points = np.asarray(points.get_vertices(), np.float32)
            # print(points.shape, len(points))
            # points = np.asarray(points.get_data())
            # vtx = np.asarray(points.get_vertices())
            # # points.("1.ply", cam.frames.get_color_frame())
        else: points = None
        self.lock.release()
        return points

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
    cam = CameraD415(flag_save=True,use_statistics=False)
    print("Looping...")
    cam.loop()
    print("Exiting...")
